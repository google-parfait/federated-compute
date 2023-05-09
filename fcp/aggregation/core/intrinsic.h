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

#ifndef FCP_AGGREGATION_CORE_INTRINSIC_H_
#define FCP_AGGREGATION_CORE_INTRINSIC_H_

#include <memory>
#include <utility>
#include <vector>

#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_spec.h"

namespace fcp {
namespace aggregation {

// A tuple representing an aggregation intrinsic.
class Intrinsic final {
 public:
  Intrinsic(std::vector<TensorSpec> inputs, std::vector<TensorSpec> outputs,
            std::unique_ptr<TensorAggregator> tensor_aggregator)
      : inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        tensor_aggregator_(std::move(tensor_aggregator)) {}

  Intrinsic(Intrinsic&& other) = default;
  Intrinsic& operator=(Intrinsic&& other) = default;

  const std::vector<TensorSpec>& inputs() const { return inputs_; }
  const std::vector<TensorSpec>& outputs() const { return outputs_; }
  const TensorAggregator& const_aggregator() const {
    return *tensor_aggregator_;
  }
  TensorAggregator& aggregator() { return *tensor_aggregator_; }

 private:
  const std::vector<TensorSpec> inputs_;
  const std::vector<TensorSpec> outputs_;
  std::unique_ptr<TensorAggregator> tensor_aggregator_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_INTRINSIC_H_
