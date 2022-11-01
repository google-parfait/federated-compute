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

#ifndef FCP_AGGREGATION_TENSORFLOW_CONVERTERS_H_
#define FCP_AGGREGATION_TENSORFLOW_CONVERTERS_H_

#include <memory>

#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/base/monitoring.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace fcp::aggregation::tensorflow {

// Converts Tensorflow DataType to Aggregation DataType.
// Returns an error status if the input data type isn't supported by
// the Aggregation Core.
StatusOr<DataType> ConvertDataType(::tensorflow::DataType dtype);

// Converts Tensorflow TensorShape to Aggregation TensorShape.
// Note that the Tensorflow shape is expected to be valid (it seems impossible
// to create an invalid shape).
TensorShape ConvertShape(const ::tensorflow::TensorShape& shape);

// Converts Tensorflow TensorSpecProto to Aggregation TensorSpec.
// Returns an error status if supplied TensorSpecProto data type or shape isn't
// supported by the Aggregation Core.
StatusOr<TensorSpec> ConvertTensorSpec(
    const ::tensorflow::TensorSpecProto& spec);

// Converts Tensorflow Tensor to Aggregation Tensor.
// Returns an error status if supplied Tensor data type or shape isn't
// supported by the Aggregation Core.
// Note that this function consumes the Tensorflow tensor.
StatusOr<Tensor> ConvertTensor(std::unique_ptr<::tensorflow::Tensor> tensor);

}  // namespace fcp::aggregation::tensorflow

#endif  // FCP_AGGREGATION_TENSORFLOW_CONVERTERS_H_
