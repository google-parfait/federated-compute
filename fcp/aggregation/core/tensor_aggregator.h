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

#ifndef FCP_AGGREGATION_CORE_TENSOR_AGGREGATOR_H_
#define FCP_AGGREGATION_CORE_TENSOR_AGGREGATOR_H_

#include <memory>
#include <utility>

#include "fcp/aggregation/core/aggregator.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// TensorAggregator is a base class for implementing Aggregation intrinsics
// with Tensor being an input and output type for the aggregation.
// TODO(team): Generalize this to the case with multiple input and
// output tensors.
class TensorAggregator
    : public Aggregator<const Tensor&, Tensor, TensorAggregator> {
 public:
  ~TensorAggregator() override = default;

  // Implementation of the base Aggregator class methods.
  Status Accumulate(const Tensor& tensor) override;
  Status MergeWith(TensorAggregator&& other) override;
  bool CanReport() const override;
  StatusOr<Tensor> Report() && override;

  // Returns the number of aggregated inputs.
  int num_inputs() const { return num_inputs_; }

 protected:
  // Construct TensorAggregator for the given tensor type and shape
  explicit TensorAggregator(Tensor result_tensor)
      : result_tensor_(std::move(result_tensor)), num_inputs_(0) {
    FCP_CHECK(CheckValid().ok());
  }

  // The actual implementation of the tensor aggregation to be provided by
  // a derived class.
  virtual void AggregateTensor(const Tensor& tensor) = 0;

  // Checks if the current TensorAggregator is valid e.g. the resulting tensor
  // hasn't been consumed.
  Status CheckValid() const;

 private:
  // Extracts the aggregated tensor and makes the current aggregator "consumed".
  Tensor TakeTensor() &&;

  Tensor result_tensor_;
  int num_inputs_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_TENSOR_AGGREGATOR_H_
