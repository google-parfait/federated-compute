/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "fcp/aggregation/core/composite_key_combiner.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/input_tensor_list.h"
#include "fcp/aggregation/core/mutable_vector_data.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/vector_string_data.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

namespace {

template <typename T>
bool CheckDataTypeSupported() {
  return sizeof(T) <= sizeof(uint64_t);
}

template <>
bool CheckDataTypeSupported<string_view>() {
  // We will store the representation of a pointer to the string as an integer,
  // so ensure the size of a pointer is less than or equal to the size of a
  // 64-bit integer.
  return sizeof(intptr_t) == sizeof(uint64_t);
}

// Copies the bytes pointed to by source_ptr to the destination pointed to by
// dest_ptr and advances source_ptr to the next T.
//
// The number of bytes copied will be the size of the type T.
//
// It is the responsibility of the caller to ensure that source_ptr is only used
// in subsequent code if it still points to a valid T after being incremented.
template <typename T>
void CopyToDest(const void*& source_ptr, uint64_t* dest_ptr,
                std::unordered_set<std::string>& intern_pool) {
  auto typed_source_ptr = static_cast<const T*>(source_ptr);
  // Cast the pointer to the destination storage to a pointer to type T and set
  // the value it points to to the source value. This allows us to copy the
  // number of bytes in T to the destination storage even if T is smaller than
  // uint64_t, without copying extra bytes.
  // It would be problematic if T is larger than uint64_t, but the Create method
  // validated that this was not the case.
  T* typed_dest_ptr = reinterpret_cast<T*>(dest_ptr);
  *typed_dest_ptr = *typed_source_ptr;
  // Set source_ptr to point to the next T assuming that it points to
  // an array of T.
  source_ptr = static_cast<const void*>(++typed_source_ptr);
}

// Specialization of CopyToDest for DT_STRING data type that interns the
// string_view pointed to by value_ptr. The address of the string in the
// intern pool is then converted to a 64 bit integer and copied to the
// destination pointed to by dest_ptr. Finally source_ptr is incremented to the
// next string_view.
//
// It is the responsibility of the caller to ensure that source_ptr is only used
// in subsequent code if it still points to a valid string_view after being
// incremented.
template <>
void CopyToDest<string_view>(const void*& source_ptr, uint64_t* dest_ptr,
                             std::unordered_set<std::string>& intern_pool) {
  auto string_view_ptr = static_cast<const string_view*>(source_ptr);
  // Insert the string into the intern pool if it does not already exist. This
  // makes a copy of the string so that the intern pool owns the storage.
  auto it = intern_pool.emplace(*string_view_ptr).first;
  // The iterator of an unordered set may be invalidated by inserting more
  // elements, but the pointer to the underlying element is guaranteed to be
  // stable. https://en.cppreference.com/w/cpp/container/unordered_set
  // Thus, get the address of the string after dereferencing the iterator.
  const std::string* interned_string_ptr = &*it;
  // The stable address of the string can be interpreted as a 64-bit integer.
  intptr_t ptr_int = reinterpret_cast<intptr_t>(interned_string_ptr);
  // Set the destination storage to the integer representation of the string
  // address.
  *dest_ptr = static_cast<uint64_t>(ptr_int);
  // Set the source_ptr to point to the next string_view assuming that it points
  // to an array of string_view.
  source_ptr = static_cast<const void*>(++string_view_ptr);
}

// Given a vector of uint64_t pointers, where the data pointed to can be safely
// interpreted as type T, returns a Tensor of underlying data type
// corresponding to T and the same length as the input vector. Each element of
// the tensor is created by interpreting the data pointed to by the uint64_t
// pointer at that index as type T.
template <typename T>
StatusOr<Tensor> GetTensorForType(
    const std::vector<const uint64_t*>& key_iters) {
  auto output_tensor_data = std::make_unique<MutableVectorData<T>>();
  output_tensor_data->reserve(key_iters.size());
  for (const uint64_t* key_it : key_iters) {
    const T* ptr = reinterpret_cast<const T*>(key_it);
    output_tensor_data->push_back(*ptr);
  }
  return Tensor::Create(internal::TypeTraits<T>::kDataType,
                        TensorShape{key_iters.size()},
                        std::move(output_tensor_data));
}

// Specialization of GetTensorForType for DT_STRING data type.
// Given a vector of char pointers, where the data pointed to can be safely
// interpreted as a pointer to a string, returns a tensor of type DT_STRING
// and the same length as the input vector containing these strings.
// The returned tensor will own all strings it refers to and is thus safe to
// use after this class is destroyed.
template <>
StatusOr<Tensor> GetTensorForType<string_view>(
    const std::vector<const uint64_t*>& key_iters) {
  std::vector<std::string> strings_for_output;
  for (auto key_it = key_iters.begin(); key_it != key_iters.end(); ++key_it) {
    const intptr_t* ptr_to_string_address =
        reinterpret_cast<const intptr_t*>(*key_it);
    // The integer stored to represent a string is the address of the string
    // stored in the intern_pool_. Thus this integer can be safely cast to a
    // pointer and dereferenced to obtain the string.
    const std::string* ptr =
        reinterpret_cast<const std::string*>(*ptr_to_string_address);
    strings_for_output.push_back(*ptr);
  }
  return Tensor::Create(
      DT_STRING, TensorShape{key_iters.size()},
      std::make_unique<VectorStringData>(std::move(strings_for_output)));
}

}  // namespace

CompositeKeyCombiner::CompositeKeyCombiner(std::vector<DataType> dtypes)
    : dtypes_(dtypes) {
  for (DataType dtype : dtypes) {
    // Initialize to false to satisfy compiler that all cases in the DTYPE_CASES
    // switch statement are covered, even though the cases that don't result in
    // a value for data_type_supported will actually crash the program.
    bool data_type_supported = false;
    DTYPE_CASES(dtype, T, data_type_supported = CheckDataTypeSupported<T>());
    FCP_CHECK(data_type_supported)
        << "Unsupported data type for CompositeKeyCombiner: " << dtype;
  }
}

// Returns a single tensor containing the ordinals of the composite keys
// formed from the InputTensorList.
StatusOr<Tensor> CompositeKeyCombiner::Accumulate(
    const InputTensorList& tensors) {
  FCP_ASSIGN_OR_RETURN(TensorShape shape, CheckValidAndGetShape(tensors));

  // Determine the serialized size of the composite keys.
  size_t composite_key_size = sizeof(uint64_t) * tensors.size();

  std::vector<const void*> iterators;
  iterators.reserve(tensors.size());
  for (const Tensor* t : tensors) {
    iterators.push_back(t->data().data());
  }

  // Iterate over all the TensorDataIterators at once to get the value for the
  // composite key.
  auto ordinals = std::make_unique<MutableVectorData<int64_t>>();
  for (int i = 0; i < shape.NumElements(); ++i) {
    // Create a string with the correct amount of memory to store an int64
    // representation of the element in each input tensor at the current
    // index.
    std::string composite_key_data(composite_key_size, '\0');
    uint64_t* key_ptr = reinterpret_cast<uint64_t*>(composite_key_data.data());

    for (int j = 0; j < tensors.size(); ++j) {
      // Copy the 64-bit representation of the element into the position in the
      // composite key data corresponding to this tensor.
      DTYPE_CASES(dtypes_[j], T,
                  CopyToDest<T>(iterators[j], key_ptr++, intern_pool_));
    }
    auto [it, inserted] = composite_keys_.insert(
        {std::move(composite_key_data), composite_key_next_});
    if (inserted) {
      // This is the first time this CompositeKeyCombiner has encountered this
      // particular composite key.
      composite_key_next_++;
      // Save the string representation of the key in order to recover the
      // elements of the key when GetOutputKeys is called.
      key_vec_.push_back(it->first);
    }
    // Insert the ordinal representing the composite key into the
    // correct position in the output tensor.
    ordinals->push_back(it->second);
  }
  return Tensor::Create(internal::TypeTraits<int64_t>::kDataType, shape,
                        std::move(ordinals));
}

StatusOr<std::vector<Tensor>> CompositeKeyCombiner::GetOutputKeys() const {
  std::vector<Tensor> output_keys;
  // Creating empty tensors is not allowed, so if there are no keys yet,
  // which could happen if GetOutputKeys is called before Accumulate, return
  // an empty vector.
  if (key_vec_.empty()) return output_keys;
  // Otherwise reserve space for a tensor for each data type.
  output_keys.reserve(dtypes_.size());
  std::vector<const uint64_t*> key_iters;
  key_iters.reserve(key_vec_.size());
  for (string_view s : key_vec_) {
    key_iters.push_back(reinterpret_cast<const uint64_t*>(s.data()));
  }
  for (DataType dtype : dtypes_) {
    StatusOr<Tensor> t;
    DTYPE_CASES(dtype, T, t = GetTensorForType<T>(key_iters));
    FCP_RETURN_IF_ERROR(t.status());
    output_keys.push_back(std::move(t.value()));
    for (auto key_it = key_iters.begin(); key_it != key_iters.end(); ++key_it) {
      ++*key_it;
    }
  }
  return output_keys;
}

StatusOr<TensorShape> CompositeKeyCombiner::CheckValidAndGetShape(
    const InputTensorList& tensors) {
  if (tensors.size() == 0) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "InputTensorList must contain at least one tensor.";
  } else if (tensors.size() != dtypes_.size()) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "InputTensorList size " << tensors.size()
           << "is not the same as the length of expected dtypes "
           << dtypes_.size();
  }
  // All the tensors in the input list should have the same shape and have
  // a dense encoding.
  const TensorShape* shape = nullptr;
  for (int i = 0; i < tensors.size(); ++i) {
    const Tensor* t = tensors[i];
    if (shape == nullptr) {
      shape = &t->shape();
    } else {
      if (*shape != t->shape()) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "All tensors in the InputTensorList must have the expected "
                  "shape.";
      }
    }
    if (!t->is_dense())
      return FCP_STATUS(INVALID_ARGUMENT)
             << "All tensors in the InputTensorList must be dense.";
    // Ensure the data types of the input tensors match those provided to the
    // constructor of this CompositeKeyCombiner.
    DataType expected_dtype = dtypes_[i];
    if (expected_dtype != t->dtype()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "Tensor did not have expected dtype " << expected_dtype
             << " and instead had dtype " << t->dtype();
    }
  }
  return *shape;
}

}  // namespace aggregation
}  // namespace fcp
