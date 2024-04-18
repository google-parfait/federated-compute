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
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "fcp/aggregation/core/agg_core.pb.h"
#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/dp_fedsql_constants.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/mutable_vector_data.h"
#include "fcp/aggregation/core/one_dim_grouping_aggregator.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// Below is an implementation of a sum grouping aggregator for numeric types,
// with clipping of Linfinity, L1, and L2 norms as determined by the
// parameters linfinity_bound_, l1_bound_, and l2_bound_.
// l1_bound_ and l2_bound_ can take on values <= 0 (which happens when they are
// not provided by the customer) in which case we do not make any adjustments
// to data meant for aggregation.
template <typename InputT, typename OutputT>
class DPGroupingFederatedSum final
    : public OneDimGroupingAggregator<InputT, OutputT> {
 public:
  using OneDimGroupingAggregator<InputT, OutputT>::OneDimGroupingAggregator;
  using OneDimGroupingAggregator<InputT, OutputT>::data;

  DPGroupingFederatedSum(InputT linfinity_bound, double l1_bound,
                         double l2_bound)
      : OneDimGroupingAggregator<InputT, OutputT>(),
        linfinity_bound_(linfinity_bound),
        l1_bound_(l1_bound),
        l2_bound_(l2_bound) {}

  DPGroupingFederatedSum(InputT linfinity_bound, double l1_bound,
                         double l2_bound,
                         std::unique_ptr<MutableVectorData<OutputT>> data,
                         int num_inputs)
      : OneDimGroupingAggregator<InputT, OutputT>(std::move(data), num_inputs),
        linfinity_bound_(linfinity_bound),
        l1_bound_(l1_bound),
        l2_bound_(l2_bound) {}

 private:
  // The following method clamps the input value to the linfinity bound.
  inline InputT Clamp(const InputT& input_value) {
    return std::min(std::max(input_value, -linfinity_bound_), linfinity_bound_);
  }

  // The following method returns a scalar such that, when it is applied to
  // the clamped version of local histogram, the l1 and l2 norms are at most
  // l1_bound_ and l2_bound_.
  inline double ComputeRescalingFactor(
      const absl::flat_hash_map<int64_t, InputT>& local_histogram) {
    // no re-scaling if norm bounds were not provided
    if (l1_bound_ <= 0 && l2_bound_ <= 0) {
      return 1.0;
    }

    // Compute norms after clamping magnitudes.
    double l1 = 0;
    double squared_l2 = 0;
    for (const auto& [unused, raw_value] : local_histogram) {
      // To do: optimize the number of Clamp calls. Currently called once in
      // this function and again in the final loop of AggregateVectorByOrdinals.
      InputT value = Clamp(raw_value);
      l1 += (value < 0) ? -value : value;
      squared_l2 += static_cast<double>(value) * static_cast<double>(value);
    }
    double l2 = sqrt(squared_l2);

    // Compute rescaling factor based on the norms.
    double rescaling_factor = 1.0;
    if (l1_bound_ > 0 && l1 > 0 && l1_bound_ / l1 < 1.0) {
      rescaling_factor = l1_bound_ / l1;
    }
    if (l2_bound_ > 0 && l2 > 0 && l2_bound_ / l2 < rescaling_factor) {
      rescaling_factor = l2_bound_ / l2;
    }
    return rescaling_factor;
  }

  // The following method is very much the same as GroupingFederatedSum's
  // except it clamps and rescales value_vector.
  void AggregateVectorByOrdinals(
      const AggVector<int64_t>& ordinals_vector,
      const AggVector<InputT>& value_vector) override {
    auto value_it = value_vector.begin();

    // Create a local histogram from ordinals & values, aggregating when there
    // are multiple values for the same ordinal.
    absl::flat_hash_map<int64_t, InputT> local_histogram;
    local_histogram.reserve(ordinals_vector.size());
    for (const auto& [index, ordinal] : ordinals_vector) {
      FCP_CHECK(value_it.index() == index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";

      // Only aggregate values of valid ordinals.
      if (ordinal >= 0) {
        local_histogram[ordinal] += value_it.value();
      }

      value_it++;
    }

    double rescaling_factor = ComputeRescalingFactor(local_histogram);

    // Propagate to the actual state
    for (const auto& [ordinal, value] : local_histogram) {
      // Compute the scaled value to satisfy the L1 and L2 constraints.
      double scaled_value = Clamp(value) * rescaling_factor;
      DCHECK(ordinal < data().size())
          << "Ordinal too big: " << ordinal << " vs. " << data().size();
      AggregateValue(ordinal, static_cast<OutputT>(scaled_value));
    }
  }

  // Norm bounds should not be applied when merging, since this input data
  // represents the pre-accumulated (and already per-client bounded) data from
  // multiple clients.
  void MergeVectorByOrdinals(const AggVector<int64_t>& ordinals_vector,
                             const AggVector<OutputT>& value_vector) override {
    auto value_it = value_vector.begin();
    for (auto o : ordinals_vector) {
      int64_t output_index = o.value;
      FCP_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      AggregateValue(output_index, value_it++.value());
    }
  }

  inline void AggregateValue(int64_t i, OutputT value) { data()[i] += value; }
  OutputT GetDefaultValue() override { return OutputT{0}; }

  const InputT linfinity_bound_;
  const double l1_bound_;
  const double l2_bound_;
};

// The following function creates a DPGFS object with a numerical input type.
// When the input type is integral, the output type is always int64_t.
// When the input type is floating point, the output type is always double.
template <typename InputT>
StatusOr<std::unique_ptr<TensorAggregator>> CreateDPGroupingFederatedSum(
    InputT linfinity_bound, double l1_bound, double l2_bound,
    const OneDimGroupingAggregatorState* aggregator_state) {
  if (internal::TypeTraits<InputT>::type_kind != internal::TypeKind::kNumeric) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "DPGroupingFederatedSum only supports numeric datatypes.";
  }

  if (linfinity_bound <= 0) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "DPGroupingFederatedSum: Linfinity bound must be positive.";
  }

  DataType input_type = internal::TypeTraits<InputT>::kDataType;
  switch (input_type) {
    case DT_INT32:
    case DT_INT64:
      return aggregator_state == nullptr
                 ? std::make_unique<DPGroupingFederatedSum<InputT, int64_t>>(
                       linfinity_bound, l1_bound, l2_bound)
                 : std::make_unique<DPGroupingFederatedSum<InputT, int64_t>>(
                       linfinity_bound, l1_bound, l2_bound,
                       MutableVectorData<int64_t>::CreateFromEncodedContent(
                           aggregator_state->vector_data()),
                       aggregator_state->num_inputs());
    case DT_FLOAT:
    case DT_DOUBLE:
      return aggregator_state == nullptr
                 ? std::make_unique<DPGroupingFederatedSum<InputT, double>>(
                       linfinity_bound, l1_bound, l2_bound)
                 : std::make_unique<DPGroupingFederatedSum<InputT, double>>(
                       linfinity_bound, l1_bound, l2_bound,
                       MutableVectorData<double>::CreateFromEncodedContent(
                           aggregator_state->vector_data()),
                       aggregator_state->num_inputs());
    default:
      return FCP_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory does not support "
             << DataType_Name(input_type);
  };
}

template <>
StatusOr<std::unique_ptr<TensorAggregator>> CreateDPGroupingFederatedSum(
    string_view linfinity_bound, double l1_bound, double l2_bound,
    const OneDimGroupingAggregatorState* aggregator_state) {
  return FCP_STATUS(INVALID_ARGUMENT)
         << "DPGroupingFederatedSum does not support DT_STRING.";
}

// A factory class for the GroupingFederatedSum.
// Permits parameters in the DPGroupingFederatedSum intrinsic,
// unlike GroupingFederatedSumFactory.
class DPGroupingFederatedSumFactory final
    : public OneDimBaseGroupingAggregatorFactory {
 public:
  DPGroupingFederatedSumFactory() = default;

  // DPGroupingFederatedSumFactory is not copyable or moveable.
  DPGroupingFederatedSumFactory(const DPGroupingFederatedSumFactory&) = delete;
  DPGroupingFederatedSumFactory& operator=(
      const DPGroupingFederatedSumFactory&) = delete;

 private:
  StatusOr<std::unique_ptr<TensorAggregator>> CreateInternal(
      const Intrinsic& intrinsic,
      const OneDimGroupingAggregatorState* aggregator_state) const override {
    FCP_CHECK(kDPSumUri == intrinsic.uri)
        << "DPGroupingFederatedSumFactory: Expected intrinsic URI " << kDPSumUri
        << " but got uri " << intrinsic.uri;
    // Check that the configuration is valid for grouping_federated_sum.
    if (intrinsic.inputs.size() != 1) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory: Exactly one input "
                "is expected but got "
             << intrinsic.inputs.size();
    }

    if (intrinsic.outputs.size() != 1) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory: Exactly one output tensor is "
                "expected but got "
             << intrinsic.outputs.size();
    }

    if (!intrinsic.nested_intrinsics.empty()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory: Not expected to have inner "
                "aggregations.";
    }

    const TensorSpec& input_spec = intrinsic.inputs[0];
    const TensorSpec& output_spec = intrinsic.outputs[0];
    if (input_spec.shape() != output_spec.shape()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory: Input and output tensors have "
                "mismatched shapes.";
    }

    const DataType input_type = input_spec.dtype();
    const DataType output_type = output_spec.dtype();

    if (input_type != output_type) {
      // In the GoogleSQL spec, summing floats produces doubles and summing
      // int32 produces int64. Disallow any other mixing of types.
      bool ok_integer = (input_type == DataType::DT_INT32 &&
                         output_type == DataType::DT_INT64);
      bool ok_float = (input_type == DataType::DT_FLOAT &&
                       output_type == DataType::DT_DOUBLE);
      if (!ok_integer && !ok_float) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "DPGroupingFederatedSumFactory: Input & output tensors have "
                  "mismatched dtypes: input tensor has dtype "
               << DataType_Name(input_type) << " and output tensor has dtype "
               << DataType_Name(output_type);
      }
    }
    if (internal::GetTypeKind(input_type) != internal::TypeKind::kNumeric) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory: DPGroupingFederatedSum only"
                " supports numeric datatypes.";
    }

    // Verify presence of all norm bounds
    constexpr int64_t kNumParameters = 3;
    if (intrinsic.parameters.size() != kNumParameters) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory: Linfinity, L1,"
                " and L2 bounds are expected.";
    }

    // Verify that the norm bounds are in numerical Tensors
    for (const auto& parameter_tensor : intrinsic.parameters) {
      if (internal::GetTypeKind(parameter_tensor.dtype()) !=
          internal::TypeKind::kNumeric) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "DPGroupingFederatedSumFactory: Norm bounds must be stored in"
                  " numerical Tensors.";
      }
    }

    const auto& linfinity_param = intrinsic.parameters[kLinfinityIndex];
    const double l1 = intrinsic.parameters[kL1Index].CastToScalar<double>();
    const double l2 = intrinsic.parameters[kL2Index].CastToScalar<double>();

    StatusOr<std::unique_ptr<TensorAggregator>> aggregator;
    DTYPE_CASES(
        input_type, T,
        aggregator = CreateDPGroupingFederatedSum<T>(
            linfinity_param.CastToScalar<T>(), l1, l2, aggregator_state));
    return aggregator;
  }
};

REGISTER_AGGREGATOR_FACTORY(kDPSumUri, DPGroupingFederatedSumFactory);

}  // namespace aggregation
}  // namespace fcp
