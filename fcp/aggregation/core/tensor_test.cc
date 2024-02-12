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
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/base/monitoring.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

using testing::Eq;
using testing::HasSubstr;

TEST(TensorTest, Create_Dense) {
  auto t = Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1, 2, 3}));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(t->dtype(), Eq(DT_FLOAT));
  EXPECT_THAT(t->shape(), Eq(TensorShape{3}));
  EXPECT_THAT(t->num_elements(), Eq(3));
  EXPECT_TRUE(t->is_dense());
  EXPECT_THAT(t->AsAggVector<float>().size(), Eq(3));
}

TEST(TensorTest, Create_ZeroDataSize) {
  auto t = Tensor::Create(DT_INT32, {0}, CreateTestData<int>({}));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(t->dtype(), Eq(DT_INT32));
  EXPECT_THAT(t->shape(), Eq(TensorShape{0}));
  EXPECT_THAT(t->num_elements(), Eq(0));
  EXPECT_TRUE(t->is_dense());
  EXPECT_THAT(t->AsAggVector<int>().size(), Eq(0));
}

TEST(TensorTest, Create_ScalarTensor) {
  auto t = Tensor::Create(DT_INT32, {}, CreateTestData<int>({555}));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(t->dtype(), Eq(DT_INT32));
  EXPECT_THAT(t->shape(), Eq(TensorShape{}));
  EXPECT_THAT(t->num_elements(), Eq(1));
  EXPECT_TRUE(t->is_dense());
  EXPECT_THAT(t->AsAggVector<int>().size(), Eq(1));
  EXPECT_THAT(t->AsAggVector<int>().begin().value(), Eq(555));
}

TEST(TensorTest, Create_StringTensor) {
  auto t = Tensor::Create(DT_STRING, {2},
                          CreateTestData<string_view>({"foo", "bar"}));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(t->dtype(), Eq(DT_STRING));
  EXPECT_THAT(t->shape(), Eq(TensorShape{2}));
  EXPECT_THAT(t->num_elements(), Eq(2));
  EXPECT_TRUE(t->is_dense());
  EXPECT_THAT(t->AsAggVector<string_view>().size(), Eq(2));
}

TEST(TensorTest, Create_ShapeWithUnknownDimensions) {
  auto t = Tensor::Create(DT_FLOAT, {-1}, CreateTestData<float>({1, 2, 3}));
  EXPECT_THAT(t, IsCode(INVALID_ARGUMENT));
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

TEST(TensorTest, AsScalar_NumericScalarTensor) {
  auto t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({5.5f}));
  EXPECT_EQ(t->AsScalar<float>(), 5.5f);
  EXPECT_EQ(t->AsScalar<double>(), 5.5);
  EXPECT_EQ(t->AsScalar<int>(), 5);
}

TEST(TensorTest, AsScalar_StringScalarTensor) {
  auto t = Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"foo"}));
  EXPECT_EQ(t->AsScalar<string_view>(), "foo");
}

TEST(TensorTest, AsScalar_MismatchType) {
  auto t = Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"foo"}));
  EXPECT_DEATH(t->AsScalar<int>(), "Unsupported type");

  t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({5.5f}));
  EXPECT_DEATH(t->AsScalar<string_view>(), "Incompatible tensor dtype()");
}

TEST(TensorTest, AsScalar_NonScalar) {
  auto t = Tensor::Create(DT_STRING, {2},
                          CreateTestData<string_view>({"foo", "bar"}));
  EXPECT_DEATH(t->AsScalar<string_view>(),
               "AsScalar should only be used on scalar tensors");

  t = Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({5.5f, 5.7f, 5.9f}));
  EXPECT_DEATH(t->AsScalar<float>(),
               "AsScalar should only be used on scalar tensors");
}

TEST(TensorTest, AsSpan_NumericTensor) {
  auto t =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({5.5f, 5.7f, 5.9f}));
  auto span = t->AsSpan<float>();
  EXPECT_EQ(span.size(), 3);
  EXPECT_EQ(span.at(0), 5.5f);
  EXPECT_EQ(span.at(1), 5.7f);
  EXPECT_EQ(span.at(2), 5.9f);
}

TEST(TensorTest, AsSpan_StringTensor) {
  auto t = Tensor::Create(DT_STRING, {2},
                          CreateTestData<string_view>({"foo", "bar"}));
  auto span = t->AsSpan<string_view>();
  EXPECT_EQ(span.size(), 2);
  EXPECT_EQ(span.at(0), "foo");
  EXPECT_EQ(span.at(1), "bar");
}

TEST(TensorTest, AsSpan_MismatchType) {
  auto t = Tensor::Create(DT_STRING, {2},
                          CreateTestData<string_view>({"foo", "bar"}));
  EXPECT_DEATH(t->AsSpan<int>(), "Incompatible tensor dtype()");
}

TEST(TensorTest, ToProto_Int32_Success) {
  std::initializer_list<int32_t> values{1, 2, 3, 4};
  auto t = Tensor::Create(DT_INT32, {2, 2}, CreateTestData(values));
  TensorProto expected_proto;
  expected_proto.set_dtype(DT_INT32);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.set_content(ToProtoContent(values));
  EXPECT_THAT(t->ToProto(), EqualsProto(expected_proto));
}

TEST(TensorTest, ToProto_Uint64_Success) {
  std::initializer_list<uint64_t> values{4294967296, 4294967297, 4294967298,
                                         4294967299};
  auto t = Tensor::Create(DT_UINT64, {2, 2}, CreateTestData(values));
  TensorProto expected_proto;
  expected_proto.set_dtype(DT_UINT64);
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

TEST(TensorTest, FromProto_Int32_Success) {
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

TEST(TensorTest, FromProto_Uint64_Success) {
  std::initializer_list<uint64_t> values{4294967296, 4294967297, 4294967298,
                                         4294967299, 4294967300, 4294967301};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_UINT64);
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

TEST(TensorTest, FromProto_Int32_WithoutContent_Success) {
  std::initializer_list<int32_t> values{5, 6, 7, 8, 9, 10};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.mutable_shape()->add_dim_sizes(3);
  for (int32_t v : values) {
    tensor_proto.add_int_val(v);
  }
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(*t, IsTensor({2, 3}, values));
}

TEST(TensorTest, FromProto_Float_WithoutContent_Success) {
  std::initializer_list<float> values{1.2, 1.4, 1.6};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_FLOAT);
  tensor_proto.mutable_shape()->add_dim_sizes(3);
  for (auto v : values) {
    tensor_proto.add_float_val(v);
  }
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(*t, IsTensor({3}, values));
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

TEST(TensorTest, FromProto_MultipleFields) {
  std::initializer_list<int32_t> values{5, 6, 7, 8, 9, 10};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.mutable_shape()->add_dim_sizes(3);
  tensor_proto.set_content(ToProtoContent(values));
  for (int32_t v : values) {
    tensor_proto.add_int_val(v);
  }
  Status s = Tensor::FromProto(tensor_proto).status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      HasSubstr("Tensor proto contains multiple representations of data"));
}

TEST(TensorTest, FromProto_MismatchedType) {
  std::initializer_list<int32_t> values{5, 6, 7, 8, 9, 10};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_FLOAT);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.mutable_shape()->add_dim_sizes(3);
  for (int32_t v : values) {
    tensor_proto.add_int_val(v);
  }
  Status s = Tensor::FromProto(tensor_proto).status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("Tensor proto contains data of unexpected data type"));
}

TEST(TensorTest, FromProto_NoData) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_shape()->add_dim_sizes(1);
  Status s = Tensor::FromProto(tensor_proto).status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Tensor proto contains no data"));
}

TEST(TensorTest, FromProto_InvalidStringContent) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_shape()->add_dim_sizes(1);

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
