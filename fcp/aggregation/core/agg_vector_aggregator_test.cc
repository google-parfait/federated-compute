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

#include "fcp/aggregation/core/agg_vector_aggregator.h"

#include <cstdint>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/input_tensor_list.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/base/monitoring.h"
#include "fcp/testing/testing.h"

#ifndef FCP_NANOLIBC
#include "fcp/aggregation/core/tensor.pb.h"
#endif

namespace fcp {
namespace aggregation {
namespace {

using testing::Eq;
using testing::IsFalse;
using testing::IsTrue;

// A simple Sum Aggregator
template <typename T>
class SumAggregator final : public AggVectorAggregator<T> {
 public:
  using AggVectorAggregator<T>::AggVectorAggregator;
  using AggVectorAggregator<T>::data;

 private:
  void AggregateVector(const AggVector<T>& agg_vector) override {
    for (auto [i, v] : agg_vector) {
      data()[i] += v;
    }
  }
};

TEST(AggVectorAggregatorTest, ScalarAggregation_Succeeds) {
  SumAggregator<int32_t> aggregator(DT_INT32, {});
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator.Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator.Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator.Accumulate(t3), IsOk());
  EXPECT_THAT(aggregator.CanReport(), IsTrue());

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({}, {6}));
}

TEST(AggVectorAggregatorTest, DenseAggregation_Succeeds) {
  const TensorShape shape = {4};
  SumAggregator<int32_t> aggregator(DT_INT32, shape);
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator.Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator.Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator.Accumulate(t3), IsOk());
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor(shape, {14, 19, 23, 49}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(AggVectorAggregationTest, Merge_Succeeds) {
  SumAggregator<int32_t> aggregator1(DT_INT32, {});
  SumAggregator<int32_t> aggregator2(DT_INT32, {});
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator1.Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator2.Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator2.Accumulate(t3), IsOk());

  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({}, {6}));
}

TEST(AggVectorAggregationTest, Aggregate_IncompatibleDataType) {
  SumAggregator<int32_t> aggregator(DT_INT32, {});
  Tensor t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({0})).value();
  EXPECT_THAT(aggregator.Accumulate(t), IsCode(INVALID_ARGUMENT));
}

TEST(AggVectorAggregationTest, Aggregate_IncompatibleShape) {
  SumAggregator<int32_t> aggregator(DT_INT32, {});
  Tensor t = Tensor::Create(DT_INT32, {2, 1}, CreateTestData({0, 1})).value();
  EXPECT_THAT(aggregator.Accumulate(t), IsCode(INVALID_ARGUMENT));
}

TEST(AggVectorAggregationTest, Merge_IncompatibleDataType) {
  SumAggregator<int32_t> aggregator1(DT_INT32, {});
  SumAggregator<float> aggregator2(DT_FLOAT, {});
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)),
              IsCode(INVALID_ARGUMENT));
}

TEST(AggVectorAggregationTest, Merge_IncompatibleShape) {
  SumAggregator<int32_t> aggregator1(DT_INT32, {3, 5});
  SumAggregator<int32_t> aggregator2(DT_INT32, {5, 3});
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)),
              IsCode(INVALID_ARGUMENT));
}

TEST(AggVectorAggregationTest, FailsAfterBeingConsumed) {
  SumAggregator<int32_t> aggregator(DT_INT32, {});
  EXPECT_THAT(std::move(aggregator).Report(), IsOk());

  // Now the aggregator instance has been consumed and should fail any
  // further operations.
  EXPECT_THAT(aggregator.CanReport(), IsFalse());  // NOLINT
  EXPECT_THAT(std::move(aggregator).Report(),
              IsCode(FAILED_PRECONDITION));  // NOLINT
  EXPECT_THAT(aggregator.Accumulate(         // NOLINT
                  Tensor::Create(DT_INT32, {}, CreateTestData({0})).value()),
              IsCode(FAILED_PRECONDITION));
  EXPECT_THAT(
      aggregator.MergeWith(SumAggregator<int32_t>(DT_INT32, {})),  // NOLINT
      IsCode(FAILED_PRECONDITION));

  // Passing this aggregator as an argument to another MergeWith must fail too.
  SumAggregator<int32_t> aggregator2(DT_INT32, {});
  EXPECT_THAT(aggregator2.MergeWith(std::move(aggregator)),  // NOLINT
              IsCode(FAILED_PRECONDITION));
}

TEST(AggVectorAggregatorTest, TypeCheckFailure) {
  EXPECT_DEATH(new SumAggregator<float>(DT_INT32, {}), "Incompatible dtype");
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
