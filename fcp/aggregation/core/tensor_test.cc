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

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/base/monitoring.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

using testing::Eq;

TEST(TensorTest, Create_Dense) {
  auto t = Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1, 2, 3}));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(t->dtype(), Eq(DT_FLOAT));
  EXPECT_THAT(t->shape(), Eq(TensorShape{3}));
  EXPECT_TRUE(t->is_dense());
  EXPECT_THAT(t->AsAggVector<float>().size(), Eq(3));
}

TEST(TensorTest, Create_StringTensor) {
  auto t = Tensor::Create(DT_STRING, {2},
                          CreateTestData<string_view>({"foo", "bar"}));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(t->dtype(), Eq(DT_STRING));
  EXPECT_THAT(t->shape(), Eq(TensorShape{2}));
  EXPECT_TRUE(t->is_dense());
  EXPECT_THAT(t->AsAggVector<string_view>().size(), Eq(2));
}

TEST(TensorTest, Create_DataValidationError) {
  auto t = Tensor::Create(DT_FLOAT, {}, CreateTestData<char>({'a', 'b', 'c'}));
  EXPECT_THAT(t, IsCode(FAILED_PRECONDITION));
}

TEST(TensorTest, Create_DataSizeError) {
  auto t = Tensor::Create(DT_FLOAT, {1}, CreateTestData<float>({1, 2}));
  EXPECT_THAT(t, IsCode(FAILED_PRECONDITION));
}

struct FooBar {};

TEST(TensorTest, AsAggVector_TypeCheckFailure) {
  auto t = Tensor::Create(DT_FLOAT, {1}, CreateTestData<float>({1}));
  EXPECT_DEATH(t->AsAggVector<FooBar>(), "Incompatible tensor dtype()");
  EXPECT_DEATH(t->AsAggVector<int>(), "Incompatible tensor dtype()");
}

template <typename T>
std::string ToProtoContent(std::initializer_list<T> values) {
  return std::string(reinterpret_cast<char*>(std::vector(values).data()),
                     values.size() * sizeof(T));
}

template <>
std::string ToProtoContent(std::initializer_list<string_view> values) {
  // The following is the simplified version of serializing the string values
  // that works only for short strings that are shorter than 128 characters, in
  // which case string lengths can be encoded with one byte each.
  std::string content(values.size(), '\0');
  size_t index = 0;
  // Write sizes of strings first.
  for (string_view value : values) {
    FCP_CHECK(value.size() < 128);
    content[index++] = static_cast<char>(value.size());
  }
  // Append data of all strings.
  for (string_view value : values) {
    content.append(value.data(), value.size());
  }
  return content;
}

TEST(TensorTest, ToProto_Numeric_Success) {
  std::initializer_list<int32_t> values{1, 2, 3, 4};
  auto t = Tensor::Create(DT_INT32, {2, 2}, CreateTestData(values));
  TensorProto expected_proto;
  expected_proto.set_dtype(DT_INT32);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.set_content(ToProtoContent(values));
  EXPECT_THAT(t->ToProto(), EqualsProto(expected_proto));
}

TEST(TensorTest, ToProto_String_Success) {
  std::initializer_list<string_view> values{"abc",  "de",    "",
                                            "fghi", "jklmn", "o"};
  auto t = Tensor::Create(DT_STRING, {2, 3}, CreateTestData(values));
  TensorProto expected_proto;
  expected_proto.set_dtype(DT_STRING);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.mutable_shape()->add_dim_sizes(3);
  expected_proto.set_content(ToProtoContent(values));
  EXPECT_THAT(t->ToProto(), EqualsProto(expected_proto));
}

TEST(TensorTest, FromProto_Numeric_Success) {
  std::initializer_list<int32_t> values{5, 6, 7, 8, 9, 10};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.mutable_shape()->add_dim_sizes(3);
  tensor_proto.set_content(ToProtoContent(values));
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(*t, IsTensor({2, 3}, values));
}

TEST(TensorTest, FromProto_String_Success) {
  std::initializer_list<string_view> values{"aaaaaaaa", "b", "cccc", "ddddddd"};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.set_content(ToProtoContent(values));
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(*t, IsTensor({2, 2}, values));
}

TEST(TensorTest, LargeStringValuesSerialization) {
  std::string s1(123456, 'a');
  std::string s2(7890, 'b');
  std::string s3(1357924, 'c');
  auto t1 =
      Tensor::Create(DT_STRING, {3}, CreateTestData<string_view>({s1, s2, s3}));
  auto proto = t1->ToProto();
  auto t2 = Tensor::FromProto(proto);
  EXPECT_THAT(*t2, IsTensor<string_view>({3}, {s1, s2, s3}));
}

TEST(TensorTest, FromProto_Mutable_Success) {
  std::initializer_list<int32_t> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(10);
  tensor_proto.set_content(ToProtoContent(values));
  // Store the data pointer to make sure that the tensor retains the same data.
  void* data_ptr = tensor_proto.mutable_content()->data();
  auto t = Tensor::FromProto(std::move(tensor_proto));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(*t, IsTensor({10}, values));
  EXPECT_EQ(data_ptr, t->data().data());
}

TEST(TensorTest, FromProto_NegativeDimSize) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(-1);
  tensor_proto.set_content(ToProtoContent<int32_t>({1}));
  EXPECT_THAT(Tensor::FromProto(tensor_proto), IsCode(INVALID_ARGUMENT));
}

TEST(TensorTest, FromProto_InvalidStringContent) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_shape()->add_dim_sizes(1);
  tensor_proto.set_content("");
  EXPECT_THAT(Tensor::FromProto(tensor_proto), IsCode(INVALID_ARGUMENT));

  std::string content(1, '\5');
  tensor_proto.set_content(content);
  EXPECT_THAT(Tensor::FromProto(tensor_proto), IsCode(INVALID_ARGUMENT));

  content.append("abc");
  tensor_proto.set_content(content);
  EXPECT_THAT(Tensor::FromProto(tensor_proto), IsCode(INVALID_ARGUMENT));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
