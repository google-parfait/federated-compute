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

#ifndef FCP_AGGREGATION_CORE_TENSOR_SHAPE_H_
#define FCP_AGGREGATION_CORE_TENSOR_SHAPE_H_

#include <initializer_list>
#include <vector>

namespace fcp::aggregation {

// Represents a tensor shape as a collection of
// dimension sizes.
class TensorShape {
 public:
  TensorShape(std::initializer_list<size_t> dim_sizes)
      : dim_sizes_(dim_sizes) {}

  template <typename ForwardIterator>
  TensorShape(ForwardIterator first, ForwardIterator last)
      : dim_sizes_(first, last) {}

  // Gets the dimensions and their sizes.
  const auto& dim_sizes() const { return dim_sizes_; }

  // Gets the total number of elements (which is a multiplication of sizes of
  // all dimensions).
  // For a scalar tensor with zero dimensions this returns 1.
  size_t NumElements() const;

 private:
  // TODO(team): Consider optimizing the storage for better inlining
  // of small number of dimensions.
  using DimSizesVector = std::vector<size_t>;
  DimSizesVector dim_sizes_;
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_CORE_TENSOR_SHAPE_H_
