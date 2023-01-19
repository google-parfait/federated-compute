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

#include "fcp/aggregation/core/tensor_aggregator.h"

#include <utility>

namespace fcp {
namespace aggregation {

Status TensorAggregator::CheckValid() const {
  return result_tensor_.CheckValid();
}

Tensor TensorAggregator::TakeTensor() && { return std::move(result_tensor_); }

Status TensorAggregator::Accumulate(const Tensor& tensor) {
  FCP_RETURN_IF_ERROR(CheckValid());
  if (tensor.dtype() != result_tensor_.dtype()) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "TensorAggregator::Accumulate: dtype mismatch";
  }
  if (tensor.shape() != result_tensor_.shape()) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "TensorAggregator::Accumulate: tensor shape mismatch";
  }

  // Delegate aggregation to the derived class.
  num_inputs_++;
  AggregateTensor(tensor);
  return FCP_STATUS(OK);
}

bool TensorAggregator::CanReport() const { return CheckValid().ok(); }

Status TensorAggregator::MergeWith(TensorAggregator&& other) {
  FCP_RETURN_IF_ERROR(CheckValid());
  FCP_RETURN_IF_ERROR(other.CheckValid());

  if (other.result_tensor_.dtype() != result_tensor_.dtype()) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "TensorAggregator::MergeWith: dtype mismatch";
  }
  if (other.result_tensor_.shape() != result_tensor_.shape()) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "TensorAggregator::MergeWith: tensor shape mismatch";
  }

  num_inputs_ += other.num_inputs_;
  AggregateTensor(std::move(other).TakeTensor());
  return FCP_STATUS(OK);
}

StatusOr<Tensor> TensorAggregator::Report() && {
  FCP_RETURN_IF_ERROR(CheckValid());
  if (!CanReport()) {
    return FCP_STATUS(FAILED_PRECONDITION)
           << "TensorAggregator::Report: the report goal isn't met";
  }
  return std::move(*this).TakeTensor();
}

}  // namespace aggregation
}  // namespace fcp
