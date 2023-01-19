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

#ifndef FCP_AGGREGATION_CORE_TENSOR_AGGREGATOR_FACTORY_H_
#define FCP_AGGREGATION_CORE_TENSOR_AGGREGATOR_FACTORY_H_

#include <memory>

#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// This class is the interface for the abstract factory that creates an instance
// of a TensorAggregator derived class.
class TensorAggregatorFactory {
 public:
  virtual ~TensorAggregatorFactory() = default;

  // Creates an instance of a specific aggregator for the specified type of the
  // aggregation instrinsic and the tensor specifications.
  // TODO(team): Generalize this to allow multiple inputs and outputs,
  // and an arbitrary number of arguments.
  virtual StatusOr<std::unique_ptr<TensorAggregator>> Create(
      DataType dtype, TensorShape shape) const = 0;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_TENSOR_AGGREGATOR_FACTORY_H_
