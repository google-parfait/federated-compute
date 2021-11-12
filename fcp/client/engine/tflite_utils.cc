/*
 * Copyright 2021 Google LLC
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
#include "fcp/client/engine/tflite_utils.h"

#include "fcp/base/monitoring.h"
#include "tensorflow/lite/string_util.h"

namespace fcp {
namespace client {
namespace engine {
namespace {
// Returns whether the TfLiteTensor is a resource or variant tensor.
bool IsResourceOrVariant(const TfLiteTensor* tensor) {
  return tensor->type == kTfLiteResource || tensor->type == kTfLiteVariant;
}

// Returns the TF C API Data type that corresponds to the given TfLiteType.
tensorflow::DataType GetTensorFlowDataType(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return tensorflow::DataType::DT_FLOAT;
    case kTfLiteFloat32:
      return tensorflow::DataType::DT_FLOAT;
    case kTfLiteFloat16:
      return tensorflow::DataType::DT_HALF;
    case kTfLiteFloat64:
      return tensorflow::DataType::DT_DOUBLE;
    case kTfLiteInt16:
      return tensorflow::DataType::DT_INT16;
    case kTfLiteInt32:
      return tensorflow::DataType::DT_INT32;
    case kTfLiteUInt32:
      return tensorflow::DataType::DT_UINT32;
    case kTfLiteUInt8:
      return tensorflow::DataType::DT_UINT8;
    case kTfLiteInt8:
      return tensorflow::DataType::DT_INT8;
    case kTfLiteInt64:
      return tensorflow::DataType::DT_INT64;
    case kTfLiteUInt64:
      return tensorflow::DataType::DT_UINT64;
    case kTfLiteComplex64:
      return tensorflow::DataType::DT_COMPLEX64;
    case kTfLiteComplex128:
      return tensorflow::DataType::DT_COMPLEX128;
    case kTfLiteString:
      return tensorflow::DataType::DT_STRING;
    case kTfLiteBool:
      return tensorflow::DataType::DT_BOOL;
    case kTfLiteResource:
      return tensorflow::DataType::DT_RESOURCE;
    case kTfLiteVariant:
      return tensorflow::DataType::DT_VARIANT;
  }
}

}  // anonymous namespace
absl::StatusOr<tensorflow::Tensor> CreateTfTensorFromTfLiteTensor(
    const TfLiteTensor* tflite_tensor) {
  // If the tflite tensor is a resource or variant tensor, then we can assume
  // that it's underlying data is a pointer to a TF tensor and just return a
  // copy of that Tensor (which will share the underlying data with the original
  // tensor, and increase its refcount).
  if (IsResourceOrVariant(tflite_tensor)) {
    return absl::InvalidArgumentError(
        "Resource and Variant tensors are not supported as output tensor.");
  }

  // If the tflite tensor does not point to an already existing Tensorflow
  // tensor though, we have to make a copy of it because TfLite tensor and TF
  // tensor might have different memory alignment, and there is no guarantee
  // that the TfLite tensor will outlive the TF tensor.
  tensorflow::TensorShape shape;
  int num_dims = tflite_tensor->dims->size;
  for (int i = 0; i < num_dims; ++i) {
    shape.AddDim(tflite_tensor->dims->data[i]);
  }

  tensorflow::Tensor tf_tensor(GetTensorFlowDataType(tflite_tensor->type),
                               shape);
  if (tf_tensor.dtype() == tensorflow::DataType::DT_STRING) {
    if (tf_tensor.data()) {
      tensorflow::tstring* p =
          static_cast<tensorflow::tstring*>(tf_tensor.data());
      for (int i = 0; i < tflite::GetStringCount(tflite_tensor); ++p, ++i) {
        auto ref = tflite::GetString(tflite_tensor, i);
        p->assign(ref.str, ref.len);
      }
    }
  } else {
    FCP_CHECK(tf_tensor.tensor_data().size() == tflite_tensor->bytes);

    if (tflite_tensor->data.raw) {
      std::memcpy(tf_tensor.data(), tflite_tensor->data.raw,
                  tflite_tensor->bytes);
    }
  }

  return tf_tensor;
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
