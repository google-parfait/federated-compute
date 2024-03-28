/*
 * Copyright 2024 Google LLC
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

#ifndef FCP_AGGREGATION_CORE_DP_GROUP_BY_AGGREGATOR_H_
#define FCP_AGGREGATION_CORE_DP_GROUP_BY_AGGREGATOR_H_

#include <memory>

#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// Factory class for the DPGroupByAggregator.
class DPGroupByFactory final : public TensorAggregatorFactory {
 public:
  DPGroupByFactory() = default;

  // DPGroupByFactory isn't copyable or moveable.
  DPGroupByFactory(const DPGroupByFactory&) = delete;
  DPGroupByFactory& operator=(const DPGroupByFactory&) = delete;

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_DP_GROUP_BY_AGGREGATOR_H_
