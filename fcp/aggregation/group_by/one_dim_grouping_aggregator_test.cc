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

#include "fcp/aggregation/group_by/one_dim_grouping_aggregator.h"

#include <climits>
#include <cstdint>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/mutable_vector_data.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/group_by/group_by_state.pb.h"
#include "fcp/aggregation/group_by/partitioner.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsFalse;
using ::testing::IsTrue;
using ::testing::TestWithParam;

using OneDimGroupingAggregatorTest = TestWithParam<bool>;

// A simple Sum Aggregator
template <typename InputT, typename OutputT = InputT>
class SumGroupingAggregator final
    : public OneDimGroupingAggregator<InputT, OutputT> {
 public:
  using OneDimGroupingAggregator<InputT, OutputT>::OneDimGroupingAggregator;
  using OneDimGroupingAggregator<InputT, OutputT>::data;

  static SumGroupingAggregator<InputT, OutputT> FromProto(
      const OneDimGroupingAggregatorState& aggregator_state) {
    return SumGroupingAggregator<InputT, OutputT>(
        MutableVectorData<OutputT>::CreateFromEncodedContent(
            aggregator_state.vector_data()),
        aggregator_state.num_inputs());
  }

 private:
  void AggregateVectorByOrdinals(
      const AggVector<int64_t>& ordinals_vector,
      const AggVector<InputT>& value_vector) override {
    auto value_it = value_vector.begin();
    for (auto o : ordinals_vector) {
      int64_t output_index = o.value;
      // If this function returned a failed Status at this point, the
      // data_vector_ may have already been partially modified, leaving the
      // GroupingAggregator in a bad state. Thus, check that the indices of the
      // ordinals tensor and the data tensor match with ABSL_CHECK instead.
      //
      // TODO: b/266974165 - Revisit the constraint that the indices of the
      // values must match the indices of the ordinals when sparse tensors are
      // implemented. It may be possible for the value to be omitted for a given
      // ordinal in which case the default value should be used.
      ABSL_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      // Delegate the actual aggregation to the specific aggregation
      // intrinsic implementation.
      AggregateValue(output_index, value_it++.value());
    }
  }

  void MergeVectorByOrdinals(const AggVector<int64_t>& ordinals_vector,
                             const AggVector<OutputT>& value_vector) override {
    auto value_it = value_vector.begin();
    for (auto o : ordinals_vector) {
      int64_t output_index = o.value;
      ABSL_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      AggregateValue(output_index, value_it++.value());
    }
  }

  inline void AggregateValue(int64_t i, OutputT value) { data()[i] += value; }

  OutputT GetDefaultValue() override { return static_cast<OutputT>(0); }
};

// A simple Min Aggregator that works for int32_t
class MinGroupingAggregator final : public OneDimGroupingAggregator<int32_t> {
 public:
  using OneDimGroupingAggregator<int32_t>::OneDimGroupingAggregator;
  using OneDimGroupingAggregator<int32_t>::data;

  static MinGroupingAggregator FromProto(
      const OneDimGroupingAggregatorState& aggregator_state) {
    return MinGroupingAggregator(
        MutableVectorData<int32_t>::CreateFromEncodedContent(
            aggregator_state.vector_data()),
        aggregator_state.num_inputs());
  }

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
      // ordinals tensor and the data tensor match with ABSL_CHECK instead.
      //
      // TODO: b/266974165 - Revisit the constraint that the indices of the
      // values must match the indices of the ordinals when sparse tensors are
      // implemented. It may be possible for the value to be omitted for a given
      // ordinal in which case the default value should be used.
      ABSL_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      // Delegate the actual aggregation to the specific aggregation
      // intrinsic implementation.
      AggregateValue(output_index, value_it++.value());
    }
  }

  void MergeVectorByOrdinals(const AggVector<int64_t>& ordinals_vector,
                             const AggVector<int32_t>& value_vector) override {
    AggregateVectorByOrdinals(ordinals_vector, value_vector);
  }

  inline void AggregateValue(int64_t i, int32_t value) {
    if (value < data()[i]) {
      data()[i] = value;
    }
  }
  int32_t GetDefaultValue() override { return INT_MAX; }
};

absl::StatusOr<Partitioner> CreateDefaultPartitioner(int num_partitions) {
  std::vector<Tensor> key_tensors;
  key_tensors.push_back(Tensor({"a", "b", "c"}));
  key_tensors.push_back(Tensor(std::vector<int64_t>{1, 2, 3}));
  key_tensors.push_back(Tensor({2.0, 3.0, 4.0}));
  return Partitioner::Create(key_tensors, num_partitions);
}

TEST_P(OneDimGroupingAggregatorTest, EmptyReport) {
  SumGroupingAggregator<int32_t> aggregator;

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = SumGroupingAggregator<int32_t>::FromProto(state);
  }

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor<int32_t>({0}, {}));
}

TEST_P(OneDimGroupingAggregatorTest, ScalarAggregation_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t2}), IsOk());

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = SumGroupingAggregator<int32_t>::FromProto(state);
  }

  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t3}), IsOk());
  EXPECT_THAT(aggregator.CanReport(), IsTrue());

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({1}, {6}));
}

TEST_P(OneDimGroupingAggregatorTest, DenseAggregation_Succeeds) {
  const TensorShape shape = {4};
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinals(std::vector<int64_t>{0, 1, 2, 3});
  Tensor t1({1, 3, 15, 27});
  Tensor t2({10, 5, 1, 2});
  Tensor t3({3, 11, 7, 20});
  EXPECT_THAT(aggregator.Accumulate({&ordinals, &t1}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&ordinals, &t2}), IsOk());

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = SumGroupingAggregator<int32_t>::FromProto(state);
  }

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

TEST_P(OneDimGroupingAggregatorTest, DifferentOrdinalsPerAccumulate_Succeeds) {
  const TensorShape shape = {4};
  SumGroupingAggregator<int32_t> aggregator;
  Tensor t1_ordinals(std::vector<int64_t>{3, 3, 2, 0});
  Tensor t1({1, 3, 15, 27});
  EXPECT_THAT(aggregator.Accumulate({&t1_ordinals, &t1}), IsOk());
  // Totals: [27, 0, 15, 4]
  Tensor t2_ordinals(std::vector<int64_t>{1, 0, 1, 4});
  Tensor t2({10, 5, 1, 2});
  EXPECT_THAT(aggregator.Accumulate({&t2_ordinals, &t2}), IsOk());

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = SumGroupingAggregator<int32_t>::FromProto(state);
  }

  // Totals: [32, 11, 15, 4, 2]
  Tensor t3_ordinals(std::vector<int64_t>{2, 2, 5, 1});
  Tensor t3({3, 11, 7, 20});
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

TEST_P(OneDimGroupingAggregatorTest, DifferentShapesPerAccumulate_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor t1_ordinals(std::vector<int64_t>{2, 0});
  Tensor t1({17, 3});
  EXPECT_THAT(aggregator.Accumulate({&t1_ordinals, &t1}), IsOk());
  // Totals: [3, 0, 17]
  Tensor t2_ordinals(std::vector<int64_t>{1, 0, 1, 4, 3, 0});
  Tensor t2({10, 5, 13, 2, 4, 5});
  EXPECT_THAT(aggregator.Accumulate({&t2_ordinals, &t2}), IsOk());

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = SumGroupingAggregator<int32_t>::FromProto(state);
  }

  // Totals: [13, 23, 17, 4, 2]
  Tensor t3_ordinals(std::vector<int64_t>{2, 2, 1, 0, 4});
  Tensor t3({3, 11, 7, 6, 3});
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

TEST_P(OneDimGroupingAggregatorTest,
       DifferentShapesPerAccumulate_NonzeroDefaultValue_Succeeds) {
  // Use a MinGroupingAggregator which has a non-zero default value so we can
  // test that when the output grows, elements are set to the default value.
  MinGroupingAggregator aggregator;
  Tensor t1_ordinals(std::vector<int64_t>{2, 0});
  Tensor t1({17, 3});
  EXPECT_THAT(aggregator.Accumulate({&t1_ordinals, &t1}), IsOk());
  // Totals: [3, INT_MAX, 17]
  Tensor t2_ordinals(std::vector<int64_t>{0, 0, 0, 4, 4, 0});
  Tensor t2({10, 5, 13, 2, 4, -50});
  EXPECT_THAT(aggregator.Accumulate({&t2_ordinals, &t2}), IsOk());

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = MinGroupingAggregator::FromProto(state);
  }

  // Totals: [-50, INT_MAX, 17, INT_MAX, 2]
  Tensor t3_ordinals(std::vector<int64_t>{2, 2, 1, 0, 4});
  Tensor t3({33, 11, 7, 6, 3});
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

TEST_P(OneDimGroupingAggregatorTest, Merge_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;
  Tensor ordinal(std::vector<int64_t>{0});
  Tensor t1({1});
  Tensor t2({2});
  Tensor t3({3});
  EXPECT_THAT(aggregator1.Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator2.Accumulate({&ordinal, &t2}), IsOk());
  EXPECT_THAT(aggregator2.Accumulate({&ordinal, &t3}), IsOk());

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  EXPECT_THAT(aggregator2_num_inputs, Eq(2));
  auto aggregator2_result = std::move(aggregator2).Report();
  EXPECT_THAT(aggregator2_result, IsOk());
  EXPECT_THAT(aggregator2_result->size(), Eq(1));
  Tensor aggregator2_result_tensor = std::move(aggregator2_result.value()[0]);
  EXPECT_THAT(aggregator2_result_tensor, IsTensor<int32_t>({1}, {5}));
  EXPECT_THAT(aggregator1.MergeTensors({&ordinal, &aggregator2_result_tensor},
                                       aggregator2_num_inputs),
              IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({1}, {6}));
}

TEST_P(OneDimGroupingAggregatorTest, Merge_BothEmpty_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  // Merge the two empty aggregators together.
  Tensor empty_ordinals(std::vector<int64_t>{});
  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  EXPECT_THAT(aggregator1.MergeTensors({&empty_ordinals, &aggregator2_result},
                                       aggregator2_num_inputs),
              IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(0));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor<int32_t>({0}, {}));
}

TEST_P(OneDimGroupingAggregatorTest, Merge_ThisOutputEmpty_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  Tensor t1_ordinals(std::vector<int64_t>{3, 3, 2, 0});
  Tensor t1({1, 3, 15, 27});
  EXPECT_THAT(aggregator2.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator2 totals: [27, 0, 15, 4]
  Tensor t2_ordinals(std::vector<int64_t>{1, 0, 1, 4});
  Tensor t2({10, 5, 1, 2});
  EXPECT_THAT(aggregator2.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator2 totals: [32, 11, 15, 4, 2]

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  // Merge aggregator2 into aggregator1 which has not received any inputs.
  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  Tensor ordinals_for_merge(std::vector<int64_t>{2, 6, 1, 4, 3});
  EXPECT_THAT(
      aggregator1.MergeTensors({&ordinals_for_merge, &aggregator2_result},
                               aggregator2_num_inputs),
      IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(2));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({7}, {0, 15, 32, 2, 4, 0, 11}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(OneDimGroupingAggregatorTest, Merge_OtherOutputEmpty_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  Tensor t1_ordinals(std::vector<int64_t>{3, 3, 2, 0});
  Tensor t1({1, 3, 15, 27});
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [27, 0, 15, 4]
  Tensor t2_ordinals(std::vector<int64_t>{1, 0, 1, 4});
  Tensor t2({10, 5, 1, 2});
  EXPECT_THAT(aggregator1.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator1 totals: [32, 11, 15, 4, 2]

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  // Merge with aggregator2 which has not received any inputs.
  Tensor empty_ordinals(std::vector<int64_t>{});
  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  EXPECT_THAT(aggregator1.MergeTensors({&empty_ordinals, &aggregator2_result},
                                       aggregator2_num_inputs),
              IsOk());
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

TEST_P(OneDimGroupingAggregatorTest,
       Merge_OtherOutputHasFewerElements_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  Tensor t1_ordinals(std::vector<int64_t>{3, 3, 2, 0});
  Tensor t1({1, 3, 15, 27});
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [27, 0, 15, 4]
  Tensor t2_ordinals(std::vector<int64_t>{1, 0, 1, 4});
  Tensor t2({10, 5, 1, 2});
  EXPECT_THAT(aggregator1.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator1 totals: [32, 11, 15, 4, 2]

  Tensor t3_ordinals(std::vector<int64_t>{2, 2});
  Tensor t3({3, 11});
  EXPECT_THAT(aggregator2.Accumulate({&t3_ordinals, &t3}), IsOk());
  // aggregator2 totals: [0, 0, 14]

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  Tensor ordinals_for_merge(std::vector<int64_t>{2, 0, 1});
  EXPECT_THAT(
      aggregator1.MergeTensors({&ordinals_for_merge, &aggregator2_result},
                               aggregator2_num_inputs),
      IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {32, 25, 15, 4, 2}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(OneDimGroupingAggregatorTest,
       Merge_OtherOutputHasMoreElements_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  Tensor t1_ordinals(std::vector<int64_t>{3, 3, 2, 0});
  Tensor t1({1, 3, 15, 27});
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [27, 0, 15, 4]
  Tensor t2_ordinals(std::vector<int64_t>{1, 0, 1, 4});
  Tensor t2({10, 5, 1, 2});
  EXPECT_THAT(aggregator1.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator1 totals: [32, 11, 15, 4, 2]

  Tensor t3_ordinals(std::vector<int64_t>{2, 2, 5, 1});
  Tensor t3({3, 11, 7, 20});
  EXPECT_THAT(aggregator2.Accumulate({&t3_ordinals, &t3}), IsOk());
  // aggregator2 totals: [0, 20, 14, 0, 0, 7]

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  Tensor ordinals_for_merge(std::vector<int64_t>{1, 3, 5, 0, 2, 4});
  EXPECT_THAT(
      aggregator1.MergeTensors({&ordinals_for_merge, &aggregator2_result},
                               aggregator2_num_inputs),
      IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({6}, {32, 11, 15, 24, 9, 14}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(OneDimGroupingAggregatorTest,
       Merge_OtherOutputHasMoreElements_NonzeroDefaultValue_Succeeds) {
  // Use a MinGroupingAggregator which has a non-zero default value so we can
  // test that when the output grows, elements are set to the default value.
  MinGroupingAggregator aggregator1;
  MinGroupingAggregator aggregator2;
  Tensor t1_ordinals(std::vector<int64_t>{2, 0});
  Tensor t1({-17, 3});
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [3, INT_MAX, -17]

  Tensor t2_ordinals(std::vector<int64_t>{0, 0, 0, 4, 4, 0});
  Tensor t2({10, 5, 13, 2, 4, -50});
  EXPECT_THAT(aggregator2.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator2 totals: [-50, INT_MAX, INT_MAX, INT_MAX, 2]
  Tensor t3_ordinals(std::vector<int64_t>{2, 2, 1, 0, 4});
  Tensor t3({33, 11, 7, 6, 3});
  EXPECT_THAT(aggregator2.Accumulate({&t3_ordinals, &t3}), IsOk());
  // aggregator2 totals: [-50, 7, 11, INT_MAX, 2]

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = MinGroupingAggregator::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = MinGroupingAggregator::FromProto(state2);
  }

  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  Tensor ordinals_for_merge(std::vector<int64_t>{4, 3, 2, 1, 0});
  EXPECT_THAT(
      aggregator1.MergeTensors({&ordinals_for_merge, &aggregator2_result},
                               aggregator2_num_inputs),
      IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {2, INT_MAX, -17, 7, -50}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(OneDimGroupingAggregatorTest, ValidateInputs_WrongNumberOfInputs) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal =
      Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({0})).value();
  Tensor t1 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({0})).value();
  Tensor t2 = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({0})).value();
  EXPECT_THAT(aggregator.ValidateInputs({&ordinal, &t1, &t2}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OneDimGroupingAggregatorTest,
     ValidateInputs_OrdinalTensorHasIncompatibleDataType) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal =
      Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({0})).value();
  Tensor t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({0})).value();
  EXPECT_THAT(aggregator.ValidateInputs({&ordinal, &t}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OneDimGroupingAggregatorTest, ValidateInputs_IncompatibleDataType) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({0})).value();
  EXPECT_THAT(aggregator.ValidateInputs({&ordinal, &t}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OneDimGroupingAggregatorTest,
     ValidateInputs_OrdinalAndValueTensorsHaveIncompatibleShapes) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t({0, 1});
  EXPECT_THAT(aggregator.ValidateInputs({&ordinal, &t}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OneDimGroupingAggregatorTest,
     ValidateInputs_MultidimensionalTensorsNotSupported) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal =
      Tensor::Create(DT_INT64, {2, 2}, CreateTestData<int64_t>({0, 0, 0, 0}))
          .value();
  Tensor t =
      Tensor::Create(DT_INT32, {2, 2}, CreateTestData({0, 1, 2, 3})).value();
  EXPECT_THAT(aggregator.ValidateInputs({&ordinal, &t}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OneDimGroupingAggregatorTest, Merge_IncompatibleDataType) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<float> aggregator2;

  Tensor empty_ordinals(std::vector<int64_t>{});
  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  absl::Status s = aggregator1.MergeTensors(
      {&empty_ordinals, &aggregator2_result}, aggregator2_num_inputs);
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s.message(), HasSubstr("dtype mismatch"));
}

TEST(OneDimGroupingAggregatorTest, Merge_IncompatibleInputDataType) {
  SumGroupingAggregator<int32_t, int64_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  Tensor empty_ordinals(std::vector<int64_t>{});
  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  absl::Status s = aggregator1.MergeTensors(
      {&empty_ordinals, &aggregator2_result}, aggregator2_num_inputs);
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s.message(), HasSubstr("dtype mismatch"));
}

TEST(OneDimGroupingAggregatorTest, FailsAfterBeingConsumed) {
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({0})).value();
  SumGroupingAggregator<int32_t> aggregator;
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}), IsOk());
  EXPECT_THAT(std::move(aggregator).Report(), IsOk());

  // Now the aggregator instance has been consumed and should fail any
  // further operations.
  EXPECT_THAT(aggregator.CanReport(), IsFalse());  // NOLINT
  EXPECT_THAT(std::move(aggregator).Report(),
              StatusIs(absl::StatusCode::kFailedPrecondition));  // NOLINT
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}),             // NOLINT
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(OneDimGroupingAggregatorTest, Serialize_Unimplmeneted) {
  SumGroupingAggregator<int32_t> aggregator;
  absl::Status s = std::move(aggregator).Serialize().status();
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(OneDimGroupingAggregatorTest, ToProtoWithPartition_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal1(std::vector<int64_t>{0, 1, 2});
  Tensor ordinal2(std::vector<int64_t>{0, 1});
  Tensor t1({1, 2, 3});
  Tensor t2({2, 3, 4});
  Tensor t3({4, 5});
  EXPECT_THAT(aggregator.Accumulate({&ordinal1, &t1}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&ordinal1, &t2}), IsOk());

  FCP_ASSERT_OK_AND_ASSIGN(Partitioner partitioner,
                           CreateDefaultPartitioner(/*num_partitions=*/2));
  FCP_ASSERT_OK_AND_ASSIGN(
      auto states, std::move(aggregator).PartitionToProtos(partitioner));
  EXPECT_THAT(states.size(), Eq(2));
  aggregator = SumGroupingAggregator<int32_t>::FromProto(states[1]);

  EXPECT_THAT(aggregator.Accumulate({&ordinal2, &t3}), IsOk());
  EXPECT_THAT(aggregator.CanReport(), IsTrue());

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({2}, {7, 10}));
}

TEST(OneDimGroupingAggregatorTest,
     ToProtoWithPartition_SucceedsWithEmptySlice) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal(std::vector<int64_t>{0, 1});
  Tensor t1({1, 2});
  Tensor t2({2, 3});
  Tensor t3({5});
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t2}), IsOk());

  std::vector<Tensor> key_tensors;
  key_tensors.push_back(Tensor({"a", "c"}));
  key_tensors.push_back(Tensor(std::vector<int64_t>{3, 5}));
  FCP_ASSERT_OK_AND_ASSIGN(
      auto partitioner, Partitioner::Create(key_tensors, /*num_partitions=*/2));
  FCP_ASSERT_OK_AND_ASSIGN(
      auto states, std::move(aggregator).PartitionToProtos(partitioner));
  EXPECT_THAT(states.size(), Eq(2));
  aggregator = SumGroupingAggregator<int32_t>::FromProto(states[0]);
  EXPECT_THAT(aggregator.CanReport(), IsTrue());

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
}

INSTANTIATE_TEST_SUITE_P(
    OneDimGroupingAggregatorTestInstantiation, OneDimGroupingAggregatorTest,
    ::testing::ValuesIn<bool>({false, true}),
    [](const ::testing::TestParamInfo<OneDimGroupingAggregatorTest::ParamType>&
           info) { return info.param ? "SaveIntermediateState" : "None"; });

}  // namespace
}  // namespace aggregation
}  // namespace fcp
