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

#include "fcp/aggregation/tensorflow/converters.h"

#include <initializer_list>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/base/monitoring.h"
#include "fcp/testing/testing.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace fcp::aggregation::tensorflow {
namespace {

namespace tf = ::tensorflow;

tf::TensorShape CreateTfShape(std::initializer_list<int64_t> dim_sizes) {
  tf::TensorShape shape;
  EXPECT_TRUE(tf::TensorShape::BuildTensorShape(dim_sizes, &shape).ok());
  return shape;
}

tf::TensorSpecProto CreateTfTensorSpec(
    const std::string& name, tf::DataType dtype,
    std::initializer_list<int64_t> dim_sizes) {
  tf::TensorSpecProto spec;
  spec.set_name(name);
  spec.set_dtype(dtype);
  for (auto dim_size : dim_sizes) {
    spec.mutable_shape()->add_dim()->set_size(dim_size);
  }
  return spec;
}

TEST(ConvertersTest, ConvertDataType_Success) {
  EXPECT_EQ(*ConvertDataType(tf::DT_FLOAT), DT_FLOAT);
  EXPECT_EQ(*ConvertDataType(tf::DT_DOUBLE), DT_DOUBLE);
  EXPECT_EQ(*ConvertDataType(tf::DT_INT32), DT_INT32);
  EXPECT_EQ(*ConvertDataType(tf::DT_INT64), DT_INT64);
  EXPECT_EQ(*ConvertDataType(tf::DT_STRING), DT_STRING);
}

TEST(ConvertersTest, ConvertDataType_Unsupported) {
  EXPECT_THAT(ConvertDataType(tf::DT_VARIANT), IsCode(INVALID_ARGUMENT));
}

TEST(ConvertersTest, ConvertShape_Success) {
  EXPECT_EQ(ConvertShape(CreateTfShape({})), TensorShape({}));
  EXPECT_EQ(ConvertShape(CreateTfShape({1})), TensorShape({1}));
  EXPECT_EQ(ConvertShape(CreateTfShape({2, 3})), TensorShape({2, 3}));
}

TEST(ConvertersTest, ConvertTensorSpec_Success) {
  auto tensor_spec =
      ConvertTensorSpec(CreateTfTensorSpec("foo", tf::DT_FLOAT, {1, 2, 3}));
  ASSERT_THAT(tensor_spec, IsOk());
  EXPECT_EQ(tensor_spec->name(), "foo");
  EXPECT_EQ(tensor_spec->dtype(), DT_FLOAT);
  EXPECT_EQ(tensor_spec->shape(), TensorShape({1, 2, 3}));
}

TEST(ConvertersTest, ConvertTensorSpec_UnsupportedDataType) {
  EXPECT_THAT(
      ConvertTensorSpec(CreateTfTensorSpec("foo", tf::DT_VARIANT, {1, 2, 3})),
      IsCode(INVALID_ARGUMENT));
}

TEST(ConvertersTest, ConvertTensorSpec_UnsupportedShape) {
  EXPECT_THAT(
      ConvertTensorSpec(CreateTfTensorSpec("foo", tf::DT_FLOAT, {1, -1})),
      IsCode(INVALID_ARGUMENT));
}

TEST(ConvertersTest, ConvertTensor_Numeric) {
  tf::TensorProto tensor_proto = PARSE_TEXT_PROTO(R"pb(
    dtype: DT_FLOAT
    tensor_shape {
      dim { size: 2 }
      dim { size: 3 }
    }
    float_val: 1
    float_val: 2
    float_val: 3
    float_val: 4
    float_val: 5
    float_val: 6
  )pb");
  auto tensor = std::make_unique<tf::Tensor>();
  ASSERT_TRUE(tensor->FromProto(tensor_proto));
  EXPECT_THAT(*ConvertTensor(std::move(tensor)),
              IsTensor<float>({2, 3}, {1, 2, 3, 4, 5, 6}));
}

TEST(ConvertersTest, ConvertTensor_String) {
  tf::TensorProto tensor_proto = PARSE_TEXT_PROTO(R"pb(
    dtype: DT_STRING
    tensor_shape { dim { size: 3 } }
    string_val: "abcd"
    string_val: "foobar"
    string_val: "zzzzzzzzzzzzzz"
  )pb");
  auto tensor = std::make_unique<tf::Tensor>();
  ASSERT_TRUE(tensor->FromProto(tensor_proto));
  EXPECT_THAT(*ConvertTensor(std::move(tensor)),
              IsTensor<string_view>({3}, {"abcd", "foobar", "zzzzzzzzzzzzzz"}));
}

TEST(ConvertersTest, ConvertTensor_ScalarString) {
  tf::TensorProto tensor_proto = PARSE_TEXT_PROTO(R"pb(
    dtype: DT_STRING
    tensor_shape {}
    string_val: "0123456789"
  )pb");
  auto tensor = std::make_unique<tf::Tensor>();
  ASSERT_TRUE(tensor->FromProto(tensor_proto));
  EXPECT_THAT(*ConvertTensor(std::move(tensor)),
              IsTensor<string_view>({}, {"0123456789"}));
}

TEST(ConvertersTest, ConvertTensor_UnsupportedDataType) {
  auto tensor = std::make_unique<tf::Tensor>(tf::DT_VARIANT, CreateTfShape({}));
  EXPECT_THAT(ConvertTensor(std::move(tensor)), IsCode(INVALID_ARGUMENT));
}

}  // namespace
}  // namespace fcp::aggregation::tensorflow
