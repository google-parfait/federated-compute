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

#include "fcp/aggregation/core/dp_group_by_aggregator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/dp_composite_key_combiner.h"
#include "fcp/aggregation/core/fedsql_constants.h"
#include "fcp/aggregation/core/group_by_aggregator.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/one_dim_grouping_aggregator.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// DPGroupByAggregator is a child class of GroupByAggregator.
// ::AggregateTensorsInternal enforces a bound on the number of composite keys
// (ordinals) that any one aggregation can contribute to.
// ::Report adds noise to aggregates and removes composite keys that have value
// below a threshold.
// This class is not thread safe.
class DPGroupByAggregator : public GroupByAggregator {
 protected:
  friend class DPGroupByFactory;

  // Constructs a DPGroupByAggregator.
  // This constructor is meant for use by the DPGroupByFactory; most callers
  // should instead create a DPGroupByAggregator from an intrinsic using the
  // factory, i.e.
  // `(*GetAggregatorFactory("fedsql_dp_group_by"))->Create(intrinsic)`
  //
  // Takes the same inputs as GroupByAggregator, in addition to:
  // * epsilon_per_agg: the privacy budget per nested intrinsic.
  // * delta_per_agg: the privacy failure parameter per nested intrinsic.
  // * l0_bound: the maximum number of composite keys one user can contribute to
  //   (assuming each DPGroupByAggregator::AggregateTensorsInternal call
  //    contains data from a unique user)
  DPGroupByAggregator(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs,
      const std::vector<Intrinsic>* intrinsics,
      std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
      int64_t l0_bound);

 private:
  // Returns either nullptr or a unique_ptr to a CompositeKeyCombiner, depending
  // on the input specification
  static std::unique_ptr<DPCompositeKeyCombiner> CreateDPKeyCombiner(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs, int64_t l0_bound);
};

DPGroupByAggregator::DPGroupByAggregator(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs,
    const std::vector<Intrinsic>* intrinsics,
    std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
    int64_t l0_bound)
    : GroupByAggregator(
          input_key_specs, output_key_specs, intrinsics,
          CreateDPKeyCombiner(input_key_specs, output_key_specs, l0_bound),
          std::move(aggregators)) {}

std::unique_ptr<DPCompositeKeyCombiner>
DPGroupByAggregator::CreateDPKeyCombiner(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs, int64_t l0_bound) {
  // If there are no input keys, support a columnar aggregation that aggregates
  // all the values in each column and produces a single output value per
  // column. This would be equivalent to having identical key values for all
  // rows.
  if (input_key_specs.empty()) {
    return nullptr;
  }

  // Otherwise create a DP-ready key combiner
  return std::make_unique<DPCompositeKeyCombiner>(
      GroupByAggregator::CreateKeyTypes(input_key_specs.size(), input_key_specs,
                                        *output_key_specs),
      l0_bound);
}

StatusOr<std::unique_ptr<TensorAggregator>> DPGroupByFactory::Create(
    const Intrinsic& intrinsic) const {
  // Check if the intrinsic is well-formed.
  FCP_RETURN_IF_ERROR(GroupByFactory::CheckIntrinsic(intrinsic, kDPGroupByUri));

  // DPGroupByAggregator expects parameters
  constexpr int64_t kEpsilonIndex = 0;
  constexpr int64_t kDeltaIndex = 1;
  constexpr int64_t kL0Index = 2;
  constexpr int kNumParameters = 3;

  // Ensure that the parameters list is valid and retrieve the values if so.
  if (intrinsic.parameters.size() != kNumParameters) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Expected " << kNumParameters
           << " parameters"
              " but got "
           << intrinsic.parameters.size() << " of them.";
  }

  // Epsilon must be a positive number
  if (internal::GetTypeKind(intrinsic.parameters[kEpsilonIndex].dtype()) !=
      internal::TypeKind::kNumeric) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Epsilon must be numerical.";
  }
  double epsilon = intrinsic.parameters[kEpsilonIndex].AsScalar<double>();
  if (epsilon <= 0) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Epsilon must be positive.";
  }

  // Delta must be a number between 0 and 1
  if (internal::GetTypeKind(intrinsic.parameters[kDeltaIndex].dtype()) !=
      internal::TypeKind::kNumeric) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Delta must be numerical.";
  }
  double delta = intrinsic.parameters[kDeltaIndex].AsScalar<double>();
  if (delta <= 0 || delta >= 1) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: Delta must lie between 0 and 1.";
  }

  // L0 bound must be a positive number
  if (internal::GetTypeKind(intrinsic.parameters[kL0Index].dtype()) !=
      internal::TypeKind::kNumeric) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: L0 bound must be numerical.";
  }
  int64_t l0_bound = intrinsic.parameters[kL0Index].AsScalar<int64_t>();
  if (l0_bound <= 0) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "DPGroupByFactory: L0 bound must be positive.";
  }

  // Create nested aggregators.
  std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> nested_aggregators;
  FCP_ASSIGN_OR_RETURN(nested_aggregators,
                       GroupByFactory::CreateAggregators(intrinsic));

  // Use new rather than make_unique here because the factory function that uses
  // a non-public constructor can't use std::make_unique, and we don't want to
  // add a dependency on absl::WrapUnique.
  return std::unique_ptr<DPGroupByAggregator>(new DPGroupByAggregator(
      intrinsic.inputs, &intrinsic.outputs, &intrinsic.nested_intrinsics,
      std::move(nested_aggregators), l0_bound));
}

REGISTER_AGGREGATOR_FACTORY(std::string(kDPGroupByUri), DPGroupByFactory);

}  // namespace aggregation
}  // namespace fcp
