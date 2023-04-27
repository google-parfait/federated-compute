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

#include "fcp/aggregation/core/input_tensor_list.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

Status TensorAggregator::Accumulate(InputTensorList tensors) {
  FCP_RETURN_IF_ERROR(CheckValid());

  // Delegate aggregation to the derived class.
  return AggregateTensors(std::move(tensors));
}

bool TensorAggregator::CanReport() const { return CheckValid().ok(); }

StatusOr<OutputTensorList> TensorAggregator::Report() && {
  FCP_RETURN_IF_ERROR(CheckValid());
  if (!CanReport()) {
    return FCP_STATUS(FAILED_PRECONDITION)
           << "TensorAggregator::Report: the report goal isn't met";
  }
  return std::move(*this).TakeOutputs();
}

}  // namespace aggregation
}  // namespace fcp
