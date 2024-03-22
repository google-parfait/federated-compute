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

#ifndef FCP_AGGREGATION_CORE_COMPOSITE_KEY_COMBINER_H_
#define FCP_AGGREGATION_CORE_COMPOSITE_KEY_COMBINER_H_

#include <cstddef>
#include <cstdint>
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
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// Class operating on sets of tensors of the same shape to combine indices for
// which the same combination of elements occurs, or in other words, indices
// containing the same composite key.
//
// This class contains two methods: Accumulate and GetOutputKeys, which can each
// be called multiple times.
//
// Accumulate takes in an InputTensorList of tensors of the same shape, and
// returns a Tensor of the same shape containing ordinals to represent the
// composite key that exists at each index. Composite keys are stored
// across calls to Accumulate, so if the same composite key is ever encountered
// in two different indices, whether in the same or a different call to
// Accumulate, the same ordinal will be returned in both these indices.
//
// GetOutputKeys returns the composite keys that have been seen in all previous
// calls to Accumulate, represented by a vector of Tensors. If the ordinal
// returned by Accumulate for that composite key was i, the composite key will
// be found at position i in the output vector.
//
// This class is not thread safe.
class CompositeKeyCombiner {
 public:
  ~CompositeKeyCombiner() = default;

  // CompositeKeyCombiner is not copyable or moveable.
  CompositeKeyCombiner(const CompositeKeyCombiner&) = delete;
  CompositeKeyCombiner& operator=(const CompositeKeyCombiner&) = delete;
  CompositeKeyCombiner(CompositeKeyCombiner&&) = delete;
  CompositeKeyCombiner& operator=(CompositeKeyCombiner&&) = delete;

  // Creates a CompositeKeyCombiner if inputs are valid or crashes otherwise.
  explicit CompositeKeyCombiner(std::vector<DataType> dtypes);

  // Returns a single tensor containing the ordinals of the composite keys
  // formed from the tensors in the InputTensorList.
  //
  // The shape of each of the input tensors must match the shape provided to the
  // constructor, and the dtypes of the input tensors must match the dtypes
  // provided to the constructor.
  //
  // For each index in the input tensors, the combination of elements from each
  // tensor at that index forms a "composite key." Across calls to Accumulate,
  // each unique composite key will be represented by a unique ordinal.
  //
  // The returned tensor is of data type DT_INT64 and the same shape that was
  // provided to the constructor.
  StatusOr<Tensor> Accumulate(const InputTensorList& tensors);

  // Obtains the vector of output keys ordered by their representative ordinal.
  //
  // The datatypes of the tensors in the output vector will match the data types
  // provided to the constructor.
  //
  // For each unique combination of elements that was seen across all calls to
  // Accumulate on this class so far, the vector of output tensors will include
  // that combination of elements. The ordering of the elements within the
  // output tensors will correspond to the ordinals returned by Accumulate. For
  // example, if Accumulate returned the integer 5 in the output tensor at
  // position 8 when it encountered this combination of elements in the input
  // tensor list at position 8, then the elements in the composite key will
  // appear at position 5 in the output tensors returned by this method.
  OutputTensorList GetOutputKeys() const;

  // Gets a reference to the expected types for this CompositeKeyCombiner.
  const std::vector<DataType>& dtypes() const { return dtypes_; }

 protected:
  // Creates ordinals for composite keys spread across input tensors.
  // Specifically, the i-th entry of the output is the ordinal for the composite
  // key made by combining the i-th entry of the first tensor, the i-th entry of
  // the second tensor, ...
  //
  // In a nested for loop, transfer the bytes into a string composite_key. For
  // each composite_key, call SaveCompositeKeyAndGetOrdinal to update
  // data structures with the composite_key and obtain a possibly new ordinal.
  // (refer to the end of this file for the definition of that function)
  //
  // The data structures for SaveCompositeKeyAndGetOrdinal are explicitly given
  // to this function. Allows for the use of temporary data structures in
  // DPCompositeKeyCombiner::AccumulateWithBound.
  std::unique_ptr<MutableVectorData<int64_t>> CreateOrdinals(
      const InputTensorList& tensors, size_t num_elements,
      std::unordered_map<std::string, int64_t>& composite_key_map,
      int64_t& current_ordinal, std::vector<string_view>& vector_of_keys);

  // Checks that the provided InputTensorList can be accumulated into this
  // CompositeKeyCombiner.
  StatusOr<TensorShape> CheckValidAndGetShape(const InputTensorList& tensors);

  // Functions to grant access to members
  inline std::unordered_map<std::string, int64_t>& GetCompositeKeys() {
    return composite_keys_;
  }
  inline int64_t& GetCompositeKeyNext() { return composite_key_next_; }
  inline std::vector<string_view>& GetKeyVec() { return key_vec_; }

 private:
  // The data types of the tensors in valid inputs to Accumulate, in this exact
  // order.
  // TODO(team): Use inlined vector to store the DataTypes instead.
  std::vector<DataType> dtypes_;
  // String views of the composite keys in the order they will appear in the
  // output tensors returned by GetOutputKeys.
  std::vector<string_view> key_vec_;
  // Set of unique strings encountered in tensors of type DT_STRING on calls to
  // Accumulate.
  // Used as an optimization to avoid storing the same string multiple
  // times even if it appears in many composite keys.
  // TODO(team): Intern directly into the output tensor instead to avoid
  // copies when creating the output tensors.
  std::unordered_set<std::string> intern_pool_;
  // Mapping of string representations of the composite keys seen so far to
  // their ordinal position in the output tensors returned by GetOutputKeys.
  std::unordered_map<std::string, int64_t> composite_keys_;
  // Number of unique composite keys encountered so far across all calls to
  // Accumulate.
  int64_t composite_key_next_ = 0;
};

// If composite_key is not in a mapping from composite keys to ordinals, the
// following function maps it to a new ordinal and adds a view of composite_key
// to a dedicated vector. Otherwise, the function retrieves the ordinal assigned
// to it. In either case, returns the ordinal that the composite_key has been
// mapped to.
inline int64_t SaveCompositeKeyAndGetOrdinal(
    std::string&& composite_key,
    std::unordered_map<std::string, int64_t>& composite_key_map,
    int64_t& current_ordinal, std::vector<string_view>& vector_of_keys) {
  auto [it, inserted] =
      composite_key_map.insert({composite_key, current_ordinal});
  if (inserted) {
    // This is the first time this CompositeKeyCombiner has encountered this
    // particular composite key.
    current_ordinal++;
    // Save the composite key in the dedicated vector so we can recover it
    // when GetOutputKeys is called.
    vector_of_keys.push_back(it->first);
  }
  // return the ordinal associated with the composite key
  return it->second;
}

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_COMPOSITE_KEY_COMBINER_H_
