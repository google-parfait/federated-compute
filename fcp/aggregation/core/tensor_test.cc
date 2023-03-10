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

TEST(TensorTest, ToProto_Success) {
  std::initializer_list<int32_t> values{1, 2, 3, 4};
  auto t = Tensor::Create(DT_INT32, {2, 2}, CreateTestData(values));
  TensorProto expected_proto;
  expected_proto.set_dtype(DT_INT32);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.set_content(ToProtoContent(values));
  EXPECT_THAT(t->ToProto(), EqualsProto(expected_proto));
}

TEST(TensorTest, FromProto_Success) {
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

TEST(TensorTest, FromProto_NegativeSize) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(-1);
  tensor_proto.set_content(ToProtoContent<int32_t>({1}));
  EXPECT_THAT(Tensor::FromProto(tensor_proto), IsCode(INVALID_ARGUMENT));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
