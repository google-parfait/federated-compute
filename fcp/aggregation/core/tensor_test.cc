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

using testing::ElementsAre;
using testing::Eq;

TEST(TensorTest, Create) {
  auto t = Tensor::Create(DT_FLOAT, {3, 5}, CreateTestData<float>(15));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(t->dtype(), Eq(DT_FLOAT));
  EXPECT_THAT(t->shape(), Eq(TensorShape{3, 5}));
  EXPECT_THAT(t->AsAggVector<float>().size(), Eq(15));
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

TEST(TensorTest, AsAggVector_NumSlices) {
  auto data = CreateTestData<float>(8);
  auto data_ptr = data.get();
  auto t = Tensor::Create(DT_FLOAT, {8}, std::move(data));
  auto vec = t->AsAggVector<float>();
  EXPECT_THAT(vec.num_slices(), Eq(0));

  data_ptr->AddSlice(1, {2, 3});
  data_ptr->AddSlice(5, {7});
  EXPECT_THAT(vec.num_slices(), Eq(2));
}

TEST(TensorTest, AsAggVector_GetSlice) {
  auto t = Tensor::Create(DT_FLOAT, {8},
                          CreateTestData<float>(8, {{4, {2, 3, 4, 5}}}));
  auto vec = t->AsAggVector<float>();
  auto slice = vec.get_slice(0);
  EXPECT_THAT(slice.start_index(), Eq(4));
  EXPECT_THAT(*slice.data(), Eq(2));
  EXPECT_THAT(slice[0], Eq(2));
  EXPECT_THAT(slice[1], Eq(3));
  EXPECT_THAT(slice[2], Eq(4));
  EXPECT_THAT(slice[3], Eq(5));
}

TEST(TensorTest, AsAggVector_SliceValueIterator) {
  auto t = Tensor::Create(DT_FLOAT, {8},
                          CreateTestData<float>(8, {{4, {2, 3, 4, 5}}}));
  auto vec = t->AsAggVector<float>();
  auto slice = vec.get_slice(0);

  EXPECT_THAT(slice.size(), Eq(4));

  // ElementsAre matcher is supposed to use the iterator.
  EXPECT_THAT(slice, ElementsAre(2, 3, 4, 5));

  // Also test the for-iterator loop, which uses the pre-increment ++ operator.
  float sum = 0;
  for (auto v : slice) {
    sum += v;
  }
  EXPECT_THAT(sum, Eq(14));

  // Also test the post-increment version of the ++ operator.
  sum = 0;
  for (auto it = slice.begin(); it != slice.end(); it++) {
    sum += *it;
  }
  EXPECT_THAT(sum, Eq(14));
}

}  // namespace
}  // namespace fcp::aggregation
