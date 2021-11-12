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
#ifndef FCP_CLIENT_ENGINE_TFLITE_UTILS_H_
#define FCP_CLIENT_ENGINE_TFLITE_UTILS_H_

#include "absl/status/statusor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/c/common.h"

namespace fcp {
namespace client {
namespace engine {
// Converts a TfLiteTensor into TF Tensor. It is primarily used to convert the
// output tensors back to TF Tensor.
absl::StatusOr<tensorflow::Tensor> CreateTfTensorFromTfLiteTensor(
    const TfLiteTensor* tensor);

}  // namespace engine
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_ENGINE_TFLITE_UTILS_H_
