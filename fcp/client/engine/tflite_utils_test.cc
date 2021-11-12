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

#include "gtest/gtest.h"
#include "fcp/testing/testing.h"
#include "tensorflow/lite/string_util.h"

namespace fcp {
namespace client {
namespace engine {
namespace {

TEST(TfLiteUtilsTest, TestConvertFloat32Tensor) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = kTfLiteFloat32;
  tflite_tensor.allocation_type = kTfLiteDynamic;
  tflite_tensor.sparsity = nullptr;
  tflite_tensor.dims_signature = nullptr;
  TfLiteIntArray* dims = TfLiteIntArrayCreate(2);
  dims->data[0] = 1;
  dims->data[1] = 3;
  tflite_tensor.dims = dims;
  float data_arr[] = {1.1, 0.456, 0.322};
  std::vector<float_t> data(std::begin(data_arr), std::end(data_arr));
  size_t num_bytes = data.size() * sizeof(float_t);
  tflite_tensor.data.raw = static_cast<char*>(malloc(num_bytes));
  memcpy(tflite_tensor.data.raw, data.data(), num_bytes);
  tflite_tensor.bytes = num_bytes;

  auto tf_tensor_or = CreateTfTensorFromTfLiteTensor(&tflite_tensor);
  EXPECT_TRUE(tf_tensor_or.ok());
  tensorflow::Tensor tf_tensor = tf_tensor_or.value();
  EXPECT_EQ(tf_tensor.NumElements(), 3);
  auto* tf_data = static_cast<float_t*>(tf_tensor.data());
  for (float weight : data_arr) {
    EXPECT_EQ(*tf_data, weight);
    tf_data++;
  }

  TfLiteTensorFree(&tflite_tensor);
}

TEST(TfLiteUtilsTest, TestConvertStringTensor) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = kTfLiteString;
  tflite_tensor.is_variable = false;
  tflite_tensor.sparsity = nullptr;
  tflite_tensor.data.raw = nullptr;
  tflite_tensor.dims_signature = nullptr;
  TfLiteIntArray* dims = TfLiteIntArrayCreate(2);
  dims->data[0] = 1;
  dims->data[1] = 2;
  tflite_tensor.dims = dims;
  std::string data_arr[] = {std::string("a_str\0ing", 9), "b_string"};
  tflite::DynamicBuffer buf;
  for (const auto& value : data_arr) {
    buf.AddString(value.data(), value.length());
  }
  buf.WriteToTensor(&tflite_tensor, nullptr);

  auto tf_tensor_or = CreateTfTensorFromTfLiteTensor(&tflite_tensor);
  EXPECT_TRUE(tf_tensor_or.ok());
  tensorflow::Tensor tf_tensor = tf_tensor_or.value();
  EXPECT_EQ(tf_tensor.NumElements(), 2);
  auto* tf_data = static_cast<tensorflow::tstring*>(tf_tensor.data());
  for (const auto& str : data_arr) {
    EXPECT_EQ(*tf_data, str);
    tf_data++;
  }
  TfLiteTensorFree(&tflite_tensor);
}

TEST(TfLiteUtilsTest, TestConvertVariantTensor) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = kTfLiteVariant;
  EXPECT_THAT(CreateTfTensorFromTfLiteTensor(&tflite_tensor),
              IsCode(INVALID_ARGUMENT));
}

TEST(TfLiteUtilsTest, TestConvertResourceTensor) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = kTfLiteResource;
  EXPECT_THAT(CreateTfTensorFromTfLiteTensor(&tflite_tensor),
              IsCode(INVALID_ARGUMENT));
}

}  // namespace
}  // namespace engine
}  // namespace client
}  // namespace fcp
