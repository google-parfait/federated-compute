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

#include "fcp/aggregation/core/agg_vector_aggregator.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

constexpr char kFederatedSumUri[] = "federated_sum";

// Implementation of a generic sum aggregator.
template <typename T>
class FederatedSum final : public AggVectorAggregator<T> {
 public:
  using AggVectorAggregator<T>::AggVectorAggregator;
  using AggVectorAggregator<T>::data;

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

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    // Check that the configuration is valid for federated_sum.
    if (kFederatedSumUri != intrinsic.uri) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedSumFactory: Expected intrinsic URI "
             << kFederatedSumUri << " but got uri " << intrinsic.uri;
    }
    if (intrinsic.inputs.size() != 1) {
      return FCP_STATUS(INVALID_ARGUMENT) << "FederatedSumFactory: Exactly one "
                                             "input is expected.";
    }
    if (intrinsic.outputs.size() != 1) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedSumFactory: Exactly one output tensor is expected.";
    }
    if (!intrinsic.nested_intrinsics.empty()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedSumFactory: Expected no nested intrinsics.";
    }
    if (!intrinsic.parameters.empty()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedSumFactory: Expected no parameters.";
    }

    const TensorSpec& input_spec = intrinsic.inputs[0];
    const TensorSpec& output_spec = intrinsic.outputs[0];

    if (input_spec.dtype() != output_spec.dtype() ||
        input_spec.shape() != output_spec.shape()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedSumFactory: Input and output tensors have mismatched "
                "specs.";
    }
    std::unique_ptr<TensorAggregator> aggregator;
    NUMERICAL_ONLY_DTYPE_CASES(
        input_spec.dtype(), T,
        aggregator = std::make_unique<FederatedSum<T>>(
            input_spec.dtype(), std::move(input_spec.shape())));
    return aggregator;
  }
};

// TODO(team): Revise the registration mechanism below.
#ifdef FCP_BAREMETAL
extern "C" void RegisterFederatedSum() {
  RegisterAggregatorFactory(kFederatedSumUri, new FederatedSumFactory());
}
#else
REGISTER_AGGREGATOR_FACTORY(kFederatedSumUri, FederatedSumFactory);
#endif

}  // namespace aggregation
}  // namespace fcp
