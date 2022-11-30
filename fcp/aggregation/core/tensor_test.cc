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

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/testing/testing.h"

namespace fcp::aggregation {
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

TEST(TensorTest, Create_SparseZeroSlices) {
  auto t = Tensor::Create(DT_FLOAT, {3, 5}, CreateTestData<float>(15));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(t->dtype(), Eq(DT_FLOAT));
  EXPECT_THAT(t->shape(), Eq(TensorShape{3, 5}));
  EXPECT_FALSE(t->is_dense());
  EXPECT_THAT(t->AsAggVector<float>().size(), Eq(15));
}

TEST(TensorTest, Create_SparseMultipleSlices) {
  // Partial slice starting at offset zero.
  auto test_data1 = CreateTestData<float>(15);
  test_data1->AddSlice(0, {1, 2, 3});
  auto t1 = Tensor::Create(DT_FLOAT, {3, 5}, std::move(test_data1));
  EXPECT_THAT(t1, IsOk());
  EXPECT_FALSE(t1->is_dense());

  // Partial slice starting at offset other than zero.
  auto test_data2 = CreateTestData<float>(15);
  test_data2->AddSlice(3, {4, 5, 6});
  auto t2 = Tensor::Create(DT_FLOAT, {3, 5}, std::move(test_data2));
  EXPECT_THAT(t2, IsOk());
  EXPECT_FALSE(t2->is_dense());

  // Two partial slices.
  auto test_data3 = CreateTestData<float>(15);
  test_data3->AddSlice(0, {1, 2, 3});
  test_data3->AddSlice(12, {13, 14, 15});
  auto t3 = Tensor::Create(DT_FLOAT, {3, 5}, std::move(test_data3));
  EXPECT_THAT(t3, IsOk());
  EXPECT_FALSE(t3->is_dense());
}

TEST(TensorTest, Create_DataValidationError) {
  auto t = Tensor::Create(DT_FLOAT, {}, std::make_unique<TestData<char>>(3));
  EXPECT_THAT(t, IsCode(FAILED_PRECONDITION));
}

TEST(TensorTest, Create_DataSizeError) {
  auto t = Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>(5));
  EXPECT_THAT(t, IsCode(FAILED_PRECONDITION));
}

struct FooBar {};

TEST(TensorTest, AsAggVector_TypeCheckFailure) {
  auto t = Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>(4));
  EXPECT_DEATH(t->AsAggVector<FooBar>(), "Incompatible tensor dtype()");
  EXPECT_DEATH(t->AsAggVector<int>(), "Incompatible tensor dtype()");
}

}  // namespace
}  // namespace fcp::aggregation
