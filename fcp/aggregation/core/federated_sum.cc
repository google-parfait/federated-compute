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
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

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

template <typename T>
StatusOr<std::unique_ptr<TensorAggregator>> CreateFederatedSum(
    DataType dtype, TensorShape shape) {
  return std::unique_ptr<TensorAggregator>(new FederatedSum<T>(dtype, shape));
}

// Not supported for DT_STRING
template <>
StatusOr<std::unique_ptr<TensorAggregator>> CreateFederatedSum<string_view>(
    DataType dtype, TensorShape shape) {
  return FCP_STATUS(INVALID_ARGUMENT)
         << "FederatedSum isn't supported for DT_STRING datatype.";
}

// Factory class for the FederatedSum.
class FederatedSumFactory final : public TensorAggregatorFactory {
 public:
  FederatedSumFactory() = default;

  // FederatedSumFactory isn't copyable or moveable.
  FederatedSumFactory(const FederatedSumFactory&) = delete;
  FederatedSumFactory& operator=(const FederatedSumFactory&) = delete;

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      DataType dtype, TensorShape shape) const override {
    StatusOr<std::unique_ptr<TensorAggregator>> aggregator;
    DTYPE_CASES(dtype, T,
                aggregator = CreateFederatedSum<T>(dtype, std::move(shape)));
    return aggregator;
  }
};

// TODO(team): Revise the registration mechanism below.
#ifdef FCP_BAREMETAL
extern "C" void RegisterFederatedSum() {
  RegisterAggregatorFactory("federated_sum", new FederatedSumFactory());
}
#else
REGISTER_AGGREGATOR_FACTORY("federated_sum", FederatedSumFactory);
#endif

}  // namespace aggregation
}  // namespace fcp
