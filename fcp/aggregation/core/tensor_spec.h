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

#ifndef FCP_AGGREGATION_CORE_TENSOR_SPEC_H_
#define FCP_AGGREGATION_CORE_TENSOR_SPEC_H_

#include <string>
#include <utility>

#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor_shape.h"

namespace fcp {
namespace aggregation {

// A tuple representing tensor name, data type, and shape.
class TensorSpec final {
 public:
  TensorSpec(std::string name, DataType dtype, TensorShape shape)
      : name_(std::move(name)), dtype_(dtype), shape_(std::move(shape)) {}

  const std::string& name() const { return name_; }
  DataType dtype() const { return dtype_; }
  const TensorShape& shape() const { return shape_; }

 private:
  const std::string name_;
  const DataType dtype_;
  const TensorShape shape_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_TENSOR_SPEC_H_
