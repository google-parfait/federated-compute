/*
 * Copyright 2023 Google LLC
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
#include "fcp/aggregation/core/one_dim_grouping_aggregator.h"

#include <climits>
#include <cstdint>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/base/monitoring.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

using testing::Eq;
using testing::IsFalse;
using testing::IsTrue;

// A simple Sum Aggregator
template <typename T>
class SumGroupingAggregator final : public OneDimGroupingAggregator<T> {
 public:
  using OneDimGroupingAggregator<T>::OneDimGroupingAggregator;
  using OneDimGroupingAggregator<T>::data;

 private:
  void AggregateVectorByOrdinals(const AggVector<int64_t>& ordinals_vector,
                                 const AggVector<T>& value_vector) override {
    auto value_it = value_vector.begin();
    for (auto o : ordinals_vector) {
      int64_t output_index = o.value;
      // If this function returned a failed Status at this point, the
      // data_vector_ may have already been partially modified, leaving the
      // GroupingAggregator in a bad state. Thus, check that the indices of the
      // ordinals tensor and the data tensor match with FCP_CHECK instead.
      //
      // TODO(team): Revisit the constraint that the indices of the
      // values must match the indices of the ordinals when sparse tensors are
      // implemented. It may be possible for the value to be omitted for a given
      // ordinal in which case the default value should be used.
      FCP_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      // Delegate the actual aggregation to the specific aggregation
      // intrinsic implementation.
      AggregateValue(output_index, value_it++.value());
    }
  }

  void AggregateVector(const AggVector<T>& value_vector) override {
    for (auto it : value_vector) {
      AggregateValue(it.index, it.value);
    }
  }

  inline void AggregateValue(int64_t i, T value) { data()[i] += value; }

  T GetDefaultValue() override { return static_cast<T>(0); }
};

// A simple Min Aggregator that works for int32_t
class MinGroupingAggregator final : public OneDimGroupingAggregator<int32_t> {
 public:
  using OneDimGroupingAggregator<int32_t>::OneDimGroupingAggregator;
  using OneDimGroupingAggregator<int32_t>::data;

 private:
  void AggregateVectorByOrdinals(
      const AggVector<int64_t>& ordinals_vector,
      const AggVector<int32_t>& value_vector) override {
    auto value_it = value_vector.begin();
    for (auto o : ordinals_vector) {
      int64_t output_index = o.value;
      // If this function returned a failed Status at this point, the
      // data_vector_ may have already been partially modified, leaving the
      // GroupingAggregator in a bad state. Thus, check that the indices of the
      // ordinals tensor and the data tensor match with FCP_CHECK instead.
      //
      // TODO(team): Revisit the constraint that the indices of the
      // values must match the indices of the ordinals when sparse tensors are
      // implemented. It may be possible for the value to be omitted for a given
      // ordinal in which case the default value should be used.
      FCP_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      // Delegate the actual aggregation to the specific aggregation
      // intrinsic implementation.
      AggregateValue(output_index, value_it++.value());
    }
  }

  void AggregateVector(const AggVector<int32_t>& value_vector) override {
    for (auto it : value_vector) {
      AggregateValue(it.index, it.value);
    }
  }

  inline void AggregateValue(int64_t i, int32_t value) {
    if (value < data()[i]) {
      data()[i] = value;
    }
  }
  int32_t GetDefaultValue() override { return INT_MAX; }
};

TEST(GroupingAggregatorTest, EmptyReport) {
  SumGroupingAggregator<int32_t> aggregator(DT_INT32);
  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(0));
}

TEST(GroupingAggregatorTest, ScalarAggregation_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator(DT_INT32);
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t2}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t3}), IsOk());
  EXPECT_THAT(aggregator.CanReport(), IsTrue());

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({1}, {6}));
}

TEST(GroupingAggregatorTest, DenseAggregation_Succeeds) {
  const TensorShape shape = {4};
  SumGroupingAggregator<int32_t> aggregator(DT_INT32);
  Tensor ordinals =
      Tensor::Create(DT_INT64, shape, CreateTestData<int64_t>({0, 1, 2, 3}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinals, &t1}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&ordinals, &t2}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&ordinals, &t3}), IsOk());
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor(shape, {14, 19, 23, 49}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(GroupingAggregatorTest, DifferentOrdinalsPerAccumulate_Succeeds) {
  const TensorShape shape = {4};
  SumGroupingAggregator<int32_t> aggregator(DT_INT32);
  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, shape, CreateTestData<int64_t>({3, 3, 2, 0}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator.Accumulate({&t1_ordinals, &t1}), IsOk());
  // Totals: [27, 0, 15, 4]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, shape, CreateTestData<int64_t>({1, 0, 1, 4}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator.Accumulate({&t2_ordinals, &t2}), IsOk());
  // Totals: [32, 11, 15, 4, 2]
  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, shape, CreateTestData<int64_t>({2, 2, 5, 1}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator.Accumulate({&t3_ordinals, &t3}), IsOk());
  // Totals: [32, 31, 29, 4, 2, 7]
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({6}, {32, 31, 29, 4, 2, 7}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(GroupingAggregatorTest, DifferentShapesPerAccumulate_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator(DT_INT32);
  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({2, 0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({17, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&t1_ordinals, &t1}), IsOk());
  // Totals: [3, 0, 17]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {6}, CreateTestData<int64_t>({1, 0, 1, 4, 3, 0}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 13, 2, 4, 5}))
          .value();
  EXPECT_THAT(aggregator.Accumulate({&t2_ordinals, &t2}), IsOk());
  // Totals: [13, 23, 17, 4, 2]
  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, {5}, CreateTestData<int64_t>({2, 2, 1, 0, 4}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({3, 11, 7, 6, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&t3_ordinals, &t3}), IsOk());
  // Totals: [13, 30, 31, 4, 2]
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {19, 30, 31, 4, 5}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(GroupingAggregatorTest,
     DifferentShapesPerAccumulate_NonzeroDefaultValue_Succeeds) {
  // Use a MinGroupingAggregator which has a non-zero default value so we can
  // test that when the output grows, elements are set to the default value.
  MinGroupingAggregator aggregator(DT_INT32);
  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({2, 0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({17, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&t1_ordinals, &t1}), IsOk());
  // Totals: [3, INT_MAX, 17]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {6}, CreateTestData<int64_t>({0, 0, 0, 4, 4, 0}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 13, 2, 4, -50}))
          .value();
  EXPECT_THAT(aggregator.Accumulate({&t2_ordinals, &t2}), IsOk());
  // Totals: [-50, INT_MAX, 17, INT_MAX, 2]
  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, {5}, CreateTestData<int64_t>({2, 2, 1, 0, 4}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({33, 11, 7, 6, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&t3_ordinals, &t3}), IsOk());
  // Totals: [-50, 7, 11, INT_MAX, 2]
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {-50, 7, 11, INT_MAX, 2}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(GroupingAggregatorTest, Merge_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1(DT_INT32);
  SumGroupingAggregator<int32_t> aggregator2(DT_INT32);
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator1.Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator2.Accumulate({&ordinal, &t2}), IsOk());
  EXPECT_THAT(aggregator2.Accumulate({&ordinal, &t3}), IsOk());

  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({1}, {6}));
}

TEST(GroupingAggregatorTest, Merge_BothEmpty_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1(DT_INT32);
  SumGroupingAggregator<int32_t> aggregator2(DT_INT32);

  // Merge the two empty aggregators together.
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(0));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(0));
}

TEST(GroupingAggregatorTest, Merge_ThisOutputEmpty_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1(DT_INT32);
  SumGroupingAggregator<int32_t> aggregator2(DT_INT32);

  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({3, 3, 2, 0}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator2.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator2 totals: [27, 0, 15, 4]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({1, 0, 1, 4}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator2.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator2 totals: [32, 11, 15, 4, 2]

  // Merge aggregator2 into aggregator1 which has not received any inputs.
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(2));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {32, 11, 15, 4, 2}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(GroupingAggregatorTest, Merge_OtherOutputEmpty_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1(DT_INT32);
  SumGroupingAggregator<int32_t> aggregator2(DT_INT32);

  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({3, 3, 2, 0}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [27, 0, 15, 4]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({1, 0, 1, 4}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator1 totals: [32, 11, 15, 4, 2]

  // Merge with aggregator2 which has not received any inputs.
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(2));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {32, 11, 15, 4, 2}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(GroupingAggregatorTest, Merge_OtherOutputHasFewerElements_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1(DT_INT32);
  SumGroupingAggregator<int32_t> aggregator2(DT_INT32);

  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({3, 3, 2, 0}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [27, 0, 15, 4]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({1, 0, 1, 4}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator1 totals: [32, 11, 15, 4, 2]

  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({2, 2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {2}, CreateTestData({3, 11})).value();
  EXPECT_THAT(aggregator2.Accumulate({&t3_ordinals, &t3}), IsOk());
  // aggregator2 totals: [0, 0, 14]

  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {32, 11, 29, 4, 2}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(GroupingAggregatorTest, Merge_OtherOutputHasMoreElements_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1(DT_INT32);
  SumGroupingAggregator<int32_t> aggregator2(DT_INT32);

  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({3, 3, 2, 0}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [27, 0, 15, 4]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({1, 0, 1, 4}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator1 totals: [32, 11, 15, 4, 2]

  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({2, 2, 5, 1}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator2.Accumulate({&t3_ordinals, &t3}), IsOk());
  // aggregator2 totals: [0, 20, 14, 0, 0, 7]

  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({6}, {32, 31, 29, 4, 2, 7}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(GroupingAggregatorTest,
     Merge_OtherOutputHasMoreElements_NonzeroDefaultValue_Succeeds) {
  // Use a MinGroupingAggregator which has a non-zero default value so we can
  // test that when the output grows, elements are set to the default value.
  MinGroupingAggregator aggregator1(DT_INT32);
  MinGroupingAggregator aggregator2(DT_INT32);
  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({2, 0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({-17, 3})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [3, INT_MAX, -17]

  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {6}, CreateTestData<int64_t>({0, 0, 0, 4, 4, 0}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 13, 2, 4, -50}))
          .value();
  EXPECT_THAT(aggregator2.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator2 totals: [-50, INT_MAX, INT_MAX, INT_MAX, 2]
  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, {5}, CreateTestData<int64_t>({2, 2, 1, 0, 4}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({33, 11, 7, 6, 3})).value();
  EXPECT_THAT(aggregator2.Accumulate({&t3_ordinals, &t3}), IsOk());
  // aggregator2 totals: [-50, 7, 11, INT_MAX, 2]

  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {-50, 7, -17, INT_MAX, 2}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(GroupingAggregatorTest, Aggregate_OrdinalTensorHasIncompatibleDataType) {
  SumGroupingAggregator<int32_t> aggregator(DT_INT32);
  Tensor ordinal =
      Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({0})).value();
  Tensor t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({0})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}), IsCode(INVALID_ARGUMENT));
}

TEST(GroupingAggregatorTest, Aggregate_IncompatibleDataType) {
  SumGroupingAggregator<int32_t> aggregator(DT_INT32);
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({0})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}), IsCode(INVALID_ARGUMENT));
}

TEST(GroupingAggregatorTest,
     Aggregate_OrdinalAndValueTensorsHaveIncompatibleShapes) {
  SumGroupingAggregator<int32_t> aggregator(DT_INT32);
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t = Tensor::Create(DT_INT32, {2}, CreateTestData({0, 1})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}), IsCode(INVALID_ARGUMENT));
}

TEST(GroupingAggregatorTest, Aggregate_MultidimensionalTensorsNotSupported) {
  SumGroupingAggregator<int32_t> aggregator(DT_INT32);
  Tensor ordinal =
      Tensor::Create(DT_INT64, {2, 2}, CreateTestData<int64_t>({0, 0, 0, 0}))
          .value();
  Tensor t =
      Tensor::Create(DT_INT32, {2, 2}, CreateTestData({0, 1, 2, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}), IsCode(INVALID_ARGUMENT));
}

TEST(GroupingAggregatorTest, Merge_IncompatibleDataType) {
  SumGroupingAggregator<int32_t> aggregator1(DT_INT32);
  SumGroupingAggregator<float> aggregator2(DT_FLOAT);
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)),
              IsCode(INVALID_ARGUMENT));
}

TEST(GroupingAggregatorTest, FailsAfterBeingConsumed) {
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({0})).value();
  SumGroupingAggregator<int32_t> aggregator(DT_INT32);
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}), IsOk());
  EXPECT_THAT(std::move(aggregator).Report(), IsOk());

  // Now the aggregator instance has been consumed and should fail any
  // further operations.
  EXPECT_THAT(aggregator.CanReport(), IsFalse());  // NOLINT
  EXPECT_THAT(std::move(aggregator).Report(),
              IsCode(FAILED_PRECONDITION));           // NOLINT
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}),  // NOLINT
              IsCode(FAILED_PRECONDITION));
  EXPECT_THAT(
      aggregator.MergeWith(SumGroupingAggregator<int32_t>(DT_INT32)),  // NOLINT
      IsCode(FAILED_PRECONDITION));

  // Passing this aggregator as an argument to another MergeWith must fail too.
  SumGroupingAggregator<int32_t> aggregator2(DT_INT32);
  EXPECT_THAT(aggregator2.MergeWith(std::move(aggregator)),  // NOLINT
              IsCode(FAILED_PRECONDITION));
}

TEST(GroupingAggregatorTest, TypeCheckFailure) {
  EXPECT_DEATH(new SumGroupingAggregator<float>(DT_INT32),
               "Incompatible dtype");
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
