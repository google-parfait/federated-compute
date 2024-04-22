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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fcp/aggregation/core/agg_core.pb.h"
#include "fcp/aggregation/core/dp_composite_key_combiner.h"
#include "fcp/aggregation/core/group_by_aggregator.h"
#include "fcp/aggregation/core/input_tensor_list.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/one_dim_grouping_aggregator.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
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
 public:
  // Performs the same checks as TensorAggregator::Report but also checks
  // magnitude of DP budget. If too large, simply releases noiseless aggregate.
  // Otherwise, applies NoiseAndThreshold to the noiseless aggregate.
  StatusOr<OutputTensorList> Report() && override;

  // Accessor to vector that indicates, for each aggregation, whether Laplace
  // noise was used to ensure DP. This information is independent of user data
  // and only depends on the constructor's parameters.
  // If called before Report(), the vector will be empty.
  std::vector<bool> laplace_was_used() const { return laplace_was_used_; }

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
      std::unique_ptr<CompositeKeyCombiner> key_combiner,
      std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
      double epsilon_per_agg, double delta_per_agg, int64_t l0_bound,
      int num_inputs);

 private:
  // Returns either nullptr or a unique_ptr to a CompositeKeyCombiner, depending
  // on the input specification
  static std::unique_ptr<DPCompositeKeyCombiner> CreateDPKeyCombiner(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs, int64_t l0_bound);

  // When merging two DPGroupByAggregators, norm bounding the aggregates will
  // destroy accuracy and is not needed for privacy. Hence, this function calls
  // CompositeKeyCombiner::Accumulate, which has no L0 norm bounding.
  StatusOr<Tensor> CreateOrdinalsByGroupingKeysForMerge(
      const InputTensorList& inputs) override;

  double epsilon_per_agg_;
  double delta_per_agg_;
  int64_t l0_bound_;

  // At index i, the boolean in the below vector indicates if laplace noise was
  // used to ensure DP for the i-th aggregation. The vector is empty before
  // Report() is called.
  std::vector<bool> laplace_was_used_;
};

// Factory class for the DPGroupByAggregator.
class DPGroupByFactory final : public TensorAggregatorFactory {
 public:
  DPGroupByFactory() = default;

  // DPGroupByFactory isn't copyable or moveable.
  DPGroupByFactory(const DPGroupByFactory&) = delete;
  DPGroupByFactory& operator=(const DPGroupByFactory&) = delete;

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override;

  StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override;

 private:
  StatusOr<std::unique_ptr<TensorAggregator>> CreateInternal(
      const Intrinsic& intrinsic,
      const GroupByAggregatorState* aggregator_state) const;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_DP_GROUP_BY_AGGREGATOR_H_
