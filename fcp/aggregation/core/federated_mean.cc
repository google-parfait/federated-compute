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

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/input_tensor_list.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/mutable_vector_data.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_data.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

constexpr char kFederatedMeanUri[] = "federated_mean";
constexpr char kFederatedWeightedMeanUri[] = "federated_weighted_mean";

template <typename V, typename W>
class FederatedMean final : public TensorAggregator {
 public:
  explicit FederatedMean(DataType dtype, TensorShape shape,
                         MutableVectorData<V>* weighted_values_sum)
      : weighted_values_sum_(*weighted_values_sum),
        result_tensor_(
            Tensor::Create(dtype, shape,
                           std::unique_ptr<TensorData>(weighted_values_sum))
                .value()) {}

 private:
  Status MergeWith(TensorAggregator&& other) override {
    FCP_RETURN_IF_ERROR(CheckValid());
    FederatedMean* other_ptr = dynamic_cast<FederatedMean*>(&other);
    if (other_ptr == nullptr) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedMean::MergeWith: Can only merge with "
                "another FederatedMean.";
    }
    FCP_RETURN_IF_ERROR((*other_ptr).CheckValid());

    std::pair<std::vector<V>, W> other_internal_state =
        other_ptr->GetInternalState();
    if (other_internal_state.first.size() != weighted_values_sum_.size()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedMean::MergeWith: Can only merge weighted value sum "
                "tensors of equal length.";
    }

    for (int i = 0; i < weighted_values_sum_.size(); ++i) {
      weighted_values_sum_[i] += other_internal_state.first[i];
    }
    weights_sum_ += other_internal_state.second;
    num_inputs_ += other_ptr->GetNumInputs();
    return FCP_STATUS(OK);
  }

  Status AggregateTensors(InputTensorList tensors) override {
    for (const Tensor* tensor : tensors) {
      if (!tensor->is_dense()) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "FederatedMean::AggregateTensorsInternal: Only dense "
                  "tensors are supported.";
      }
    }

    AggVector<V> values = tensors[0]->AsAggVector<V>();
    // If the intrinsic is federated_weighted_mean, the second input tensor
    // will contain a scalar weight.
    if (tensors.size() > 1) {
      FCP_CHECK(tensors[1]->num_elements() == 1)
          << "FederatedMean::AggregateTensorsInternal: The weight must be a "
             "scalar.";
      AggVector<W> weights = tensors[1]->AsAggVector<W>();
      W weight = weights.begin().value();
      if (weight <= 0) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "FederatedMean::AggregateTensorsInternal: Only positive "
                  "weights are allowed.";
      }
      for (auto value : values) {
        weighted_values_sum_[value.index] += value.value * weight;
      }
      weights_sum_ += weight;
    } else {
      for (auto value : values) {
        weighted_values_sum_[value.index] += value.value;
      }
    }
    num_inputs_++;
    return FCP_STATUS(OK);
  }

  Status CheckValid() const override {
    if (output_consumed_) {
      return FCP_STATUS(FAILED_PRECONDITION)
             << "FederatedMean::CheckValid: Output has already been consumed.";
    }
    return FCP_STATUS(OK);
  }

  OutputTensorList TakeOutputs() && override {
    output_consumed_ = true;
    // Produce the final weighted mean values by dividing the weighted values
    // sum by the weights sum (tracked by weights_sum_ in the weighted case and
    // num_inputs_ in the non-weighted case).
    for (int i = 0; i < weighted_values_sum_.size(); ++i) {
      weighted_values_sum_[i] /=
          (weights_sum_ > 0 ? weights_sum_ : num_inputs_);
    }
    OutputTensorList outputs = std::vector<Tensor>();
    outputs.push_back(std::move(result_tensor_));
    return outputs;
  }

  int GetNumInputs() const override { return num_inputs_; }

  std::pair<std::vector<V>, W> GetInternalState() {
    output_consumed_ = true;
    return std::make_pair(std::move(weighted_values_sum_), weights_sum_);
  }

  bool output_consumed_ = false;
  std::vector<V>& weighted_values_sum_;
  // In the weighted case, use the weights_sum_ variable to track the total
  // weight. Otherwise, just rely on the num_inputs_ variable.
  W weights_sum_ = 0;
  Tensor result_tensor_;
  int num_inputs_ = 0;
};

// Factory class for the FederatedMean.
class FederatedMeanFactory final : public TensorAggregatorFactory {
 public:
  FederatedMeanFactory() = default;

  // FederatedMeanFactory isn't copyable or moveable.
  FederatedMeanFactory(const FederatedMeanFactory&) = delete;
  FederatedMeanFactory& operator=(const FederatedMeanFactory&) = delete;

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    // Check that the configuration is valid.
    if (kFederatedMeanUri == intrinsic.uri) {
      if (intrinsic.inputs.size() != 1) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "FederatedMeanFactory: Exactly one input is expected for "
                  "federated_mean intrinsic.";
      }
    } else if (kFederatedWeightedMeanUri == intrinsic.uri) {
      if (intrinsic.inputs.size() != 2) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "FederatedMeanFactory: Exactly two inputs are expected for "
                  "federated_weighted_mean intrinsic.";
      }
    } else {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Expected intrinsic URI "
             << kFederatedMeanUri << " or " << kFederatedWeightedMeanUri
             << " but got uri " << intrinsic.uri;
    }
    if (intrinsic.outputs.size() != 1) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Exactly one output tensor is expected.";
    }
    if (!intrinsic.nested_intrinsics.empty()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Expected no nested intrinsics.";
    }
    if (!intrinsic.parameters.empty()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Expected no parameters.";
    }

    const TensorSpec& input_value_spec = intrinsic.inputs[0];
    const TensorSpec& output_spec = intrinsic.outputs[0];

    if (input_value_spec.dtype() != output_spec.dtype() ||
        input_value_spec.shape() != output_spec.shape()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Input value tensor and output tensor "
                "have mismatched specs.";
    }
    if (input_value_spec.dtype() != DT_FLOAT &&
        input_value_spec.dtype() != DT_DOUBLE) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Input value tensor type must be "
                "DT_FLOAT or DT_DOUBLE.";
    }
    StatusOr<size_t> value_num_elements =
        input_value_spec.shape().NumElements();
    if (!value_num_elements.ok()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: All dimensions of value tensor shape "
                "must be known in advance.";
    }

    DataType input_value_type = input_value_spec.dtype();
    DataType input_weight_type;
    if (kFederatedWeightedMeanUri == intrinsic.uri) {
      input_weight_type = intrinsic.inputs[1].dtype();
      StatusOr<size_t> weight_num_elements =
          intrinsic.inputs[1].shape().NumElements();
      if (!weight_num_elements.ok() || weight_num_elements.value() != 1) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "FederatedMeanFactory: The weight must be a scalar.";
      }
    } else {
      input_weight_type = DT_INT32;
    }

    std::unique_ptr<TensorAggregator> aggregator;
    FLOATING_ONLY_DTYPE_CASES(
        input_value_type, V,
        NUMERICAL_ONLY_DTYPE_CASES(
            input_weight_type, W,
            aggregator = (std::make_unique<FederatedMean<V, W>>(
                input_value_type, input_value_spec.shape(),
                new MutableVectorData<V>(value_num_elements.value())))));
    return aggregator;
  }
};

static auto unused =
    ::fcp::aggregation::internal::Registrar<FederatedMeanFactory>(
        kFederatedMeanUri);
static auto unused_weighted =
    ::fcp::aggregation::internal::Registrar<FederatedMeanFactory>(
        kFederatedWeightedMeanUri);

}  // namespace aggregation
}  // namespace fcp
