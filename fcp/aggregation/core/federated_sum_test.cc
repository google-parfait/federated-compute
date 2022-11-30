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

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/testing/testing.h"

namespace fcp::aggregation {
namespace {

using ::testing::Eq;
using ::testing::IsTrue;

TEST(FederatedSumTest, ScalarAggregation_Succeeds) {
  auto aggregator =
      (*GetAggregatorFactory("federated_sum"))->Create(DT_INT32, {}).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t3), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());

  // Verify the resulting tensor.
  EXPECT_THAT(result.value(), IsTensor({}, {6}));
}

TEST(FederatedSumTest, DenseAggregation_Succeeds) {
  const TensorShape shape = {4};
  auto aggregator =
      (*GetAggregatorFactory("federated_sum"))->Create(DT_INT32, shape).value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t3), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->num_inputs(), Eq(3));

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());

  // Verify the resulting tensor.
  EXPECT_THAT(result.value(), IsTensor(shape, {14, 19, 23, 49}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result->is_dense());
}

TEST(FederatedSumTest, SparseAggregation_Succeeds) {
  const TensorShape shape = {4, 3};
  auto aggregator =
      (*GetAggregatorFactory("federated_sum"))->Create(DT_INT32, shape).value();
  Tensor t1 = Tensor::Create(DT_INT32, shape,
                             CreateTestData<int32_t>(12, {{0, {1, 2, 3}}}))
                  .value();
  Tensor t2 = Tensor::Create(
                  DT_INT32, shape,
                  CreateTestData<int32_t>(12, {{6, {10, 11, 12, 13, 14, 15}}}))
                  .value();
  Tensor t3 = Tensor::Create(
                  DT_INT32, shape,
                  CreateTestData<int32_t>(12, {{0, {4, 4, 4}}, {9, {4, 4, 4}}}))
                  .value();
  EXPECT_THAT(aggregator->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t3), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->num_inputs(), Eq(3));

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value(),
              IsTensor(shape, {5, 6, 7, 0, 0, 0, 10, 11, 12, 17, 18, 19}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result->is_dense());
}

TEST(AggVectorAggregationTest, Merge_Succeeds) {
  auto aggregator1 =
      (*GetAggregatorFactory("federated_sum"))->Create(DT_INT32, {}).value();
  auto aggregator2 =
      (*GetAggregatorFactory("federated_sum"))->Create(DT_INT32, {}).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator1->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator2->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator2->Accumulate(t3), IsOk());

  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->num_inputs(), Eq(3));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value(), IsTensor({}, {6}));
}

}  // namespace
}  // namespace fcp::aggregation
