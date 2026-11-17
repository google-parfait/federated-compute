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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/agg_vector_aggregator.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/federated_state.pb.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/mutable_vector_data.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/tensor_spec.h"

namespace fcp {
namespace aggregation {

constexpr char kFederatedSumUri[] = "federated_sum";

// Implementation of a generic sum aggregator.
template <typename T>
class FederatedSum final : public AggVectorAggregator<T> {
 public:
  using AggVectorAggregator<T>::AggVectorAggregator;
  using AggVectorAggregator<T>::data;

  absl::StatusOr<std::string> Serialize() && override {
    FederatedSumAggregatorState aggregator_state;
    aggregator_state.set_num_inputs(this->GetNumInputs());
    *(aggregator_state.mutable_vector_data()) =
        this->vector_data().EncodeContent();
    return aggregator_state.SerializeAsString();
  }

 private:
  void AggregateVector(const AggVector<T>& agg_vector) override {
    for (auto v : agg_vector) {
      data()[v.index] += v.value;
    }
  }
};

// Factory class for the FederatedSum.
class FederatedSumFactory final : public TensorAggregatorFactory {
 public:
  FederatedSumFactory() = default;

  // FederatedSumFactory isn't copyable or moveable.
  FederatedSumFactory(const FederatedSumFactory&) = delete;
  FederatedSumFactory& operator=(const FederatedSumFactory&) = delete;

  absl::StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    return CreateInternal(intrinsic, nullptr);
  }

  absl::StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    FederatedSumAggregatorState aggregator_state;
    if (!aggregator_state.ParseFromString(serialized_state)) {
      return absl::InvalidArgumentError(
          "FederatedSumFactory: Failed to deserialize the "
          "FederatedSumAggregatorState.");
    }
    return CreateInternal(intrinsic, &aggregator_state);
  };

 private:
  absl::StatusOr<std::unique_ptr<TensorAggregator>> CreateInternal(
      const Intrinsic& intrinsic,
      const FederatedSumAggregatorState* aggregator_state) const {
    // Check that the configuration is valid for federated_sum.
    if (kFederatedSumUri != intrinsic.uri) {
      return absl::InvalidArgumentError(
          absl::StrCat("FederatedSumFactory: Expected intrinsic URI ",
                       kFederatedSumUri, " but got uri ", intrinsic.uri));
    }
    if (intrinsic.inputs.size() != 1) {
      return absl::InvalidArgumentError(
          "FederatedSumFactory: Exactly one "
          "input is expected.");
    }
    if (intrinsic.outputs.size() != 1) {
      return absl::InvalidArgumentError(
          "FederatedSumFactory: Exactly one output tensor is expected.");
    }
    if (!intrinsic.nested_intrinsics.empty()) {
      return absl::InvalidArgumentError(
          "FederatedSumFactory: Expected no nested intrinsics.");
    }
    if (!intrinsic.parameters.empty()) {
      return absl::InvalidArgumentError(
          "FederatedSumFactory: Expected no parameters.");
    }

    const TensorSpec& input_spec = intrinsic.inputs[0];
    const TensorSpec& output_spec = intrinsic.outputs[0];

    if (input_spec.dtype() != output_spec.dtype() ||
        input_spec.shape() != output_spec.shape()) {
      return absl::InvalidArgumentError(
          "FederatedSumFactory: Input and output tensors have mismatched "
          "specs.");
    }
    std::unique_ptr<TensorAggregator> aggregator;
    if (aggregator_state == nullptr) {
      NUMERICAL_ONLY_DTYPE_CASES(
          input_spec.dtype(), T,
          aggregator = std::make_unique<FederatedSum<T>>(
              input_spec.dtype(), std::move(input_spec.shape())));
      return aggregator;
    }

    NUMERICAL_ONLY_DTYPE_CASES(
        input_spec.dtype(), T,
        aggregator = std::make_unique<FederatedSum<T>>(
            input_spec.dtype(), std::move(input_spec.shape()),
            MutableVectorData<T>::CreateFromEncodedContent(
                aggregator_state->vector_data()),
            aggregator_state->num_inputs()));
    return aggregator;
  }
};

REGISTER_AGGREGATOR_FACTORY(kFederatedSumUri, FederatedSumFactory);

}  // namespace aggregation
}  // namespace fcp
