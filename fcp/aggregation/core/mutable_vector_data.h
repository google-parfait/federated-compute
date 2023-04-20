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

#ifndef FCP_AGGREGATION_CORE_MUTABLE_VECTOR_DATA_H_
#define FCP_AGGREGATION_CORE_MUTABLE_VECTOR_DATA_H_

#include <cstddef>
#include <vector>

#include "fcp/aggregation/core/tensor_data.h"

namespace fcp {
namespace aggregation {

// MutableVectorData implements TensorData by wrapping std::vector and using it
// as a backing storage. MutableVectorData can be mutated using std::vector
// methods.
template <typename T>
class MutableVectorData : public std::vector<T>, public TensorData {
 public:
  // Derive constructors from the base vector class.
  using std::vector<T>::vector;

  ~MutableVectorData() override = default;

  // Implementation of the base class methods.
  size_t byte_size() const override { return this->size() * sizeof(T); }
  const void* data() const override { return this->std::vector<T>::data(); }
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_MUTABLE_VECTOR_DATA_H_
