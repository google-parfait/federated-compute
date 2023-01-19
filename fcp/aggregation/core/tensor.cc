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

#include "fcp/aggregation/core/tensor.h"

#include <utility>

#include "fcp/aggregation/core/datatype.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

Status Tensor::CheckValid() const {
  if (dtype_ == DT_INVALID) {
    return FCP_STATUS(FAILED_PRECONDITION) << "Invalid Tensor dtype.";
  }

  size_t value_size = 0;
  CASES(dtype_, value_size = sizeof(T));

  // Verify that the storage is consistent with the value size in terms of
  // sparsity and alignment.
  // TODO(team): is there a more advanced sparsity validation that
  // needs to be done at this point?
  FCP_RETURN_IF_ERROR(data_->CheckValid(value_size));

  // Verify that the total size of the data is consistent with the value type
  // and the shape.
  if (data_->byte_size() != shape_.NumElements() * value_size) {
    return FCP_STATUS(FAILED_PRECONDITION)
           << "TensorData byte_size is inconsistent with the Tensor dtype and "
              "shape.";
  }

  return FCP_STATUS(OK);
}

StatusOr<Tensor> Tensor::Create(DataType dtype, TensorShape shape,
                                std::unique_ptr<TensorData> data) {
  Tensor tensor(dtype, std::move(shape), std::move(data));
  FCP_RETURN_IF_ERROR(tensor.CheckValid());
  return std::move(tensor);
}

}  // namespace aggregation
}  // namespace fcp
