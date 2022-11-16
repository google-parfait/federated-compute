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

#include "fcp/aggregation/core/agg_vector.h"

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/testing/test_data.h"

namespace fcp::aggregation {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;

template <typename T>
using Pair = typename AggVectorIterator<T>::IndexValuePair;

TEST(AggVectorTest, PostIncrementIterator_ScalarTensor) {
  auto t = Tensor::Create(DT_INT32, {}, CreateTestData({5}));
  EXPECT_THAT(t->AsAggVector<int>(), ElementsAre(Pair<int>{0, 5}));
}

TEST(AggVectorTest, PostIncrementIterator_DenseTensor) {
  auto t = Tensor::Create(DT_INT32, {2}, CreateTestData({3, 14}));
  EXPECT_THAT(t->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 3}, Pair<int>{1, 14}));
}

TEST(AggVectorTest, PostIncrementIterator_TensorWithNoSlice) {
  auto t = Tensor::Create(DT_FLOAT, {8}, CreateTestData<float>(8));
  EXPECT_THAT(t->AsAggVector<float>(), ElementsAre());
}

TEST(AggVectorTest, PostIncrementIterator_SparseTensorWithOneSlice) {
  auto t = Tensor::Create(DT_INT32, {8},
                          CreateTestData<int>(8, {{4, {2, 3, 4, 5}}}));
  EXPECT_THAT(t->AsAggVector<int>(),
              ElementsAre(Pair<int>{4, 2}, Pair<int>{5, 3}, Pair<int>{6, 4},
                          Pair<int>{7, 5}));
}

TEST(AggVectorTest, PostIncrementIterator_SparseTensorWithTwoSlices) {
  auto data = CreateTestData<int>(8);
  auto data_ptr = data.get();
  auto t = Tensor::Create(DT_INT32, {8}, std::move(data));
  data_ptr->AddSlice(1, {2, 3});
  data_ptr->AddSlice(5, {7});
  EXPECT_THAT(t->AsAggVector<int>(),
              ElementsAre(Pair<int>{1, 2}, Pair<int>{2, 3}, Pair<int>{5, 7}));
}

TEST(AggVectorTest, PostIncrementIterator_ForLoopIterator) {
  auto t = Tensor::Create(DT_FLOAT, {8},
                          CreateTestData<float>(8, {{4, {2, 3, 4, 5}}}));
  float sum = 0;
  size_t expected_index = 4;
  for (auto [index, value] : t->AsAggVector<float>()) {
    EXPECT_THAT(index, Eq(expected_index++));
    sum += value;
  }
  EXPECT_THAT(sum, Eq(14));
}

TEST(AggVectorTest, PreIncrementIterator) {
  auto t = Tensor::Create(DT_FLOAT, {8},
                          CreateTestData<float>(8, {{4, {2, 3, 4, 5}}}));
  auto agg_vector = t->AsAggVector<float>();
  float sum = 0;
  size_t expected_index = 4;
  for (auto it = agg_vector.begin(); it != agg_vector.end(); it++) {
    EXPECT_THAT(it.index(), Eq(expected_index++));
    sum += it.value();
  }
  EXPECT_THAT(sum, Eq(14));
}

}  // namespace
}  // namespace fcp::aggregation
