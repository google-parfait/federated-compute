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

#include <memory>
#include <string>

#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/one_dim_grouping_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

constexpr char kGoogleSqlSumUri[] = "GoogleSQL:sum";

// Implementation of a generic sum grouping aggregator.
template <typename T>
class GroupingFederatedSum final : public OneDimGroupingAggregator<T> {
 public:
  using OneDimGroupingAggregator<T>::OneDimGroupingAggregator;
  using OneDimGroupingAggregator<T>::data;

 private:
  void AggregateVectorByOrdinals(const AggVector<int64_t>& ordinals_vector,
                                 const AggVector<T>& value_vector) override {
    auto value_it = value_vector.begin();
    for (auto o : ordinals_vector) {
      int64_t output_index = o.value;
      // If this function returned a failed Status at this point, the
      // data_vector_ may have already been partially modified, leaving the
      // GroupingAggregator in a bad state. Thus, check that the indices of the
      // ordinals tensor and the data tensor match with FCP_CHECK instead.
      //
      // TODO(team): Revisit the constraint that the indices of the
      // values must match the indices of the ordinals when sparse tensors are
      // implemented. It may be possible for the value to be omitted for a given
      // ordinal in which case the default value should be used.
      FCP_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      // Delegate the actual aggregation to the specific aggregation
      // intrinsic implementation.
      AggregateValue(output_index, value_it++.value());
    }
  }

  void AggregateVector(const AggVector<T>& value_vector) override {
    for (auto it : value_vector) {
      AggregateValue(it.index, it.value);
    }
  }

  inline void AggregateValue(int64_t i, T value) { data()[i] += value; }

  T GetDefaultValue() override { return static_cast<T>(0); }
};

template <typename T>
StatusOr<std::unique_ptr<TensorAggregator>> CreateGroupingFederatedSum(
    DataType dtype) {
  return std::unique_ptr<TensorAggregator>(new GroupingFederatedSum<T>(dtype));
}

// Not supported for DT_STRING
template <>
StatusOr<std::unique_ptr<TensorAggregator>>
CreateGroupingFederatedSum<string_view>(DataType dtype) {
  return FCP_STATUS(INVALID_ARGUMENT)
         << "GroupingFederatedSum isn't supported for DT_STRING datatype.";
}

// Factory class for the GroupingFederatedSum.
class GroupingFederatedSumFactory final : public TensorAggregatorFactory {
 public:
  GroupingFederatedSumFactory() = default;

  // GroupingFederatedSumFactory isn't copyable or moveable.
  GroupingFederatedSumFactory(const GroupingFederatedSumFactory&) = delete;
  GroupingFederatedSumFactory& operator=(const GroupingFederatedSumFactory&) =
      delete;

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    if (kGoogleSqlSumUri != intrinsic.uri) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: Expected intrinsic URI "
             << kGoogleSqlSumUri << " but got uri " << intrinsic.uri;
    }
    // Check that the configuration is valid for grouping_federated_sum.
    if (intrinsic.inputs.size() != 1) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: Exactly one input "
                "is expected but got "
             << intrinsic.inputs.size();
    }

    if (intrinsic.outputs.size() != 1) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: Exactly one output tensor is "
                "expected but got "
             << intrinsic.outputs.size();
    }

    if (!intrinsic.parameters.empty()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: No "
                "input parameters expected but got "
             << intrinsic.parameters.size();
    }

    if (!intrinsic.nested_intrinsics.empty()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: Not expected to have inner "
                "aggregations.";
    }

    const TensorSpec& input_spec = intrinsic.inputs[0];
    const TensorSpec& output_spec = intrinsic.outputs[0];
    if (input_spec.dtype() != output_spec.dtype() ||
        input_spec.shape() != output_spec.shape()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: Input and output tensors have "
                "mismatched specs.";
    }
    StatusOr<std::unique_ptr<TensorAggregator>> aggregator;
    DTYPE_CASES(input_spec.dtype(), T,
                aggregator = CreateGroupingFederatedSum<T>(input_spec.dtype()));
    return aggregator;
  }
};

REGISTER_AGGREGATOR_FACTORY(kGoogleSqlSumUri, GroupingFederatedSumFactory);

}  // namespace aggregation
}  // namespace fcp
