/*
 * Copyright 2022 Google LLC
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

#ifndef FCP_AGGREGATION_CORE_AGG_VECTOR_H_
#define FCP_AGGREGATION_CORE_AGG_VECTOR_H_

#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>

#include "fcp/aggregation/core/agg_vector_iterator.h"
#include "fcp/aggregation/core/tensor_data.h"

namespace fcp::aggregation {

// AggVector is flattened one-dimensional strongly typed span of tensor values
// that provides immutable access to the values.
//
// AggVectors are more like views into the tensor data. The actual data
// continues to be owned by the tensors.
//
// Each AggVector is organized as a list of slices of contiguous blocks of
// strongly typed values. Each slice has a span of values and a start index -
// the position where the slice's span of values starts within the overall
// vector of values.
//
// Slices are ordered by the start index and are non-overlapping, and the gaps
// between the slices are assumed to have zero values.
//
// Each AggVector::Slice provides vector-like access to its values, including
// random access to the values and the standard c++ iterator implementation.
//
// Slices are copyable objects, however the actual slice data is owned by
// the containing Tensor and slices have just pointers to the data.
// A care should be taken to not access the slice data after the slice owning
// Tensor instance has been destroyed.
//
// Here is an example of code that accumulates an AggVector values into
// a dense vector:
// void Accumulate(
//     const AggVector<float>& agg_vector, std::vector<float>* result) {
//   for (int i = 0; i < agg_vector.num_slices(); i++) {
//     const auto& slice = agg_vector.get_slice(i);
//     // Start aggregation at the start_index for each slice.
//     float* dst = &result->at(slice.start_index());
//     for (auto v : slice) {
//       *(dst++) += v;
//     }
//   }
// }
//
template <typename T>
class AggVector final {
 public:
  using value_type = typename AggVectorIterator<T>::value_type;
  using const_iterator = AggVectorIterator<T>;

  // Iterator begin() function.
  const_iterator begin() const { return AggVectorIterator<T>(data_); }

  // Iterator end() function.
  const_iterator end() const { return AggVectorIterator<T>::end(); }

  // Entire AggVector length including gaps between the slices.
  size_t size() const { return size_; }

 private:
  // AggVector can be created only by Tensor::AsAggVector() method.
  friend class Tensor;
  explicit AggVector(const TensorData* data)
      : size_(data->byte_size() / sizeof(T)), data_(data) {}

  // The total length of the vector (in elements).
  size_t size_;
  // Tensor data, owned by the tensor object.
  const TensorData* data_;
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_CORE_AGG_VECTOR_H_
