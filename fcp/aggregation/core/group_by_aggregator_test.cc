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

#include "fcp/aggregation/core/group_by_aggregator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/agg_vector_aggregator.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/one_dim_grouping_aggregator.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/tensor_spec.h"
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

// A simple Sum Grouping Aggregator for use in tests.
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

  T GetDefaultValue() override { return T{}; }
};

// A simple Sum Aggregator
template <typename T>
class SumAggregator final : public AggVectorAggregator<T> {
 public:
  using AggVectorAggregator<T>::AggVectorAggregator;
  using AggVectorAggregator<T>::data;

 private:
  void AggregateVector(const AggVector<T>& agg_vector) override {
    // This aggregator is not expected to actually be used for aggregating in
    // the current tests.
    ASSERT_TRUE(false);
  }
};

TensorSpec CreateTensorSpec(std::string name, DataType dtype) {
  return TensorSpec(name, dtype, {-1});
}

template <typename T>
Intrinsic CreateInnerIntrinsic(DataType dtype) {
  return Intrinsic({CreateTensorSpec("value", dtype)},
                   {CreateTensorSpec("value", dtype)},
                   std::make_unique<SumGroupingAggregator<T>>(dtype));
}

TEST(GroupByAggregatorTest, EmptyReport) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(0));
}

TEST(GroupByAggregatorTest, ScalarAggregation_Succeeds) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor key =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"key_string"}))
          .value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator.Accumulate({&key, &t1}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&key, &t2}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&key, &t3}), IsOk());
  EXPECT_THAT(aggregator.CanReport(), IsTrue());

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor<string_view>({1}, {"key_string"}));
  EXPECT_THAT(result.value()[1], IsTensor({1}, {6}));
}

TEST(GroupByAggregatorTest, DenseAggregation_Succeeds) {
  const TensorShape shape = {4};
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor keys =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "one", "two", "three"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator.Accumulate({&keys, &t1}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&keys, &t2}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&keys, &t3}), IsOk());
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(2));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>(shape, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor(shape, {14, 19, 23, 49}));
  // Also ensure that the resulting tensors are dense.
  EXPECT_TRUE(result.value()[0].is_dense());
  EXPECT_TRUE(result.value()[1].is_dense());
}

TEST(GroupByAggregatorTest, DifferentKeysPerAccumulate_Succeeds) {
  const TensorShape shape = {4};
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor keys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "zero", "one", "two"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator.Accumulate({&keys1, &t1}), IsOk());
  // Totals: [4, 15, 27]
  Tensor keys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"one", "zero", "one", "three"}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator.Accumulate({&keys2, &t2}), IsOk());
  // Totals: [9, 26, 27, 2]
  Tensor keys3 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"two", "two", "four", "one"}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator.Accumulate({&keys3, &t3}), IsOk());
  // Totals: [9, 46, 41, 2, 7]
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensors.
  EXPECT_THAT(
      result.value()[0],
      IsTensor<string_view>({5}, {"zero", "one", "two", "three", "four"}));
  EXPECT_THAT(result.value()[1], IsTensor({5}, {9, 46, 41, 2, 7}));
}

TEST(GroupByAggregatorTest, DifferentShapesPerAccumulate_Succeeds) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));

  Tensor keys1 = Tensor::Create(DT_STRING, {2},
                                CreateTestData<string_view>({"zero", "one"}))
                     .value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&keys1, &t1}), IsOk());
  // Totals: [1, 3]
  Tensor keys2 =
      Tensor::Create(DT_STRING, {6},
                     CreateTestData<string_view>(
                         {"two", "one", "zero", "one", "three", "two"}))
          .value();
  Tensor t2 = Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 1, 2, 4, 9}))
                  .value();
  EXPECT_THAT(aggregator.Accumulate({&keys2, &t2}), IsOk());
  // Totals: [2, 10, 19, 4]
  Tensor keys3 =
      Tensor::Create(
          DT_STRING, {5},
          CreateTestData<string_view>({"two", "two", "one", "zero", "four"}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({3, 11, 7, 6, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&keys3, &t3}), IsOk());
  // Totals: [8, 17, 33, 4, 3]
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensors.
  EXPECT_THAT(
      result.value()[0],
      IsTensor<string_view>({5}, {"zero", "one", "two", "three", "four"}));
  EXPECT_THAT(result.value()[1], IsTensor({5}, {8, 17, 33, 4, 3}));
}

TEST(GroupByAggregatorTest, Accumulate_MultipleValueTensors_Succeeds) {
  const TensorShape shape = {4};
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor keys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "zero", "one", "two"}))
          .value();
  Tensor tA1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor tB1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({14, 11, 7, 14})).value();
  EXPECT_THAT(aggregator.Accumulate({&keys1, &tA1, &tB1}), IsOk());
  // Totals: [4, 15, 27], [25, 7, 14]
  Tensor keys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"one", "zero", "one", "three"}))
          .value();
  Tensor tA2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor tB2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 2, 8})).value();
  EXPECT_THAT(aggregator.Accumulate({&keys2, &tA2, &tB2}), IsOk());
  // Totals: [9, 26, 27, 2], [28, 10, 14, 8]
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(2));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor({4}, {9, 26, 27, 2}));
  EXPECT_THAT(result.value()[2], IsTensor({4}, {28, 10, 14, 8}));
}

TEST(GroupByAggregatorTest, Accumulate_NoValueTensors_Succeeds) {
  const TensorShape shape = {4};
  std::vector<Intrinsic> intrinsics;
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor keys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "zero", "one", "two"}))
          .value();
  EXPECT_THAT(aggregator.Accumulate({&keys1}), IsOk());
  Tensor keys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"one", "zero", "one", "three"}))
          .value();
  EXPECT_THAT(aggregator.Accumulate({&keys2}), IsOk());
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(2));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
}

TEST(GroupByAggregatorTest, Accumulate_MultipleKeyTensors_Succeeds) {
  const TensorShape shape = {4};
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key1", DT_STRING),
                                       CreateTensorSpec("key2", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor sizeKeys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "large"}))
          .value();
  Tensor animalKeys1 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator.Accumulate({&sizeKeys1, &animalKeys1, &t1}), IsOk());
  // Totals: [4, 15, 27]
  Tensor sizeKeys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"small", "large", "small", "small"}))
          .value();
  Tensor animalKeys2 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator.Accumulate({&sizeKeys2, &animalKeys2, &t2}), IsOk());
  // Totals: [9, 26, 27, 2]
  Tensor sizeKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "small"}))
          .value();
  Tensor animalKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"dog", "dog", "rabbit", "cat"}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator.Accumulate({&sizeKeys3, &animalKeys3, &t3}), IsOk());
  // Totals: [9, 46, 41, 2, 7]
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>(
                  {5}, {"large", "small", "large", "small", "small"}));
  EXPECT_THAT(
      result.value()[1],
      IsTensor<string_view>({5}, {"cat", "cat", "dog", "dog", "rabbit"}));
  EXPECT_THAT(result.value()[2], IsTensor({5}, {9, 46, 41, 2, 7}));
}

TEST(GroupByAggregatorTest,
     Accumulate_MultipleKeyTensors_SomeKeysNotInOutput_Succeeds) {
  const TensorShape shape = {4};
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> input_key_specs = {
      CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_STRING)};
  std::vector<TensorSpec> output_key_specs = {
      CreateTensorSpec("", DT_STRING), CreateTensorSpec("animals", DT_STRING)};
  GroupByAggregator aggregator(input_key_specs, &output_key_specs,
                               std::move(intrinsics));
  Tensor sizeKeys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "large"}))
          .value();
  Tensor animalKeys1 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator.Accumulate({&sizeKeys1, &animalKeys1, &t1}), IsOk());
  // Totals: [4, 15, 27]
  Tensor sizeKeys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"small", "large", "small", "small"}))
          .value();
  Tensor animalKeys2 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator.Accumulate({&sizeKeys2, &animalKeys2, &t2}), IsOk());
  // Totals: [9, 26, 27, 2]
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(2));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  // Verify the resulting tensors.
  // Only the second key tensor should be included in the output.
  EXPECT_THAT(result.value().size(), Eq(2));
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"cat", "cat", "dog", "dog"}));
  EXPECT_THAT(result.value()[1], IsTensor({4}, {9, 26, 27, 2}));
}

TEST(GroupByAggregatorTest, Accumulate_NoKeyTensors) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));

  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator.Accumulate({&t1}), IsOk());
  Tensor t2 = Tensor::Create(DT_INT32, {3}, CreateTestData({10, 5, 1})).value();
  EXPECT_THAT(aggregator.Accumulate({&t2}), IsOk());
  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({3, 11, 7, 20, 5})).value();
  EXPECT_THAT(aggregator.Accumulate({&t3}), IsOk());

  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor<int32_t>({1}, {108}));
}

TEST(GroupByAggregatorTest, Merge_Succeeds) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));
  Tensor key =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"foo"}))
          .value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  Tensor t4 = Tensor::Create(DT_INT32, {}, CreateTestData({4})).value();
  Tensor t5 = Tensor::Create(DT_INT32, {}, CreateTestData({5})).value();
  EXPECT_THAT(aggregator1.Accumulate({&key, &t1}), IsOk());
  EXPECT_THAT(aggregator1.Accumulate({&key, &t2}), IsOk());
  EXPECT_THAT(aggregator2.Accumulate({&key, &t3}), IsOk());
  EXPECT_THAT(aggregator2.Accumulate({&key, &t4}), IsOk());
  EXPECT_THAT(aggregator2.Accumulate({&key, &t5}), IsOk());

  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(5));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  EXPECT_THAT(result.value()[0], IsTensor<string_view>({1}, {"foo"}));
  EXPECT_THAT(result.value()[1], IsTensor({1}, {15}));
}

TEST(GroupByAggregatorTest, Merge_MultipleValueTensors_Succeeds) {
  const TensorShape shape = {4};
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));
  Tensor keys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "zero", "one", "two"}))
          .value();
  Tensor tA1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor tB1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({14, 11, 7, 14})).value();
  EXPECT_THAT(aggregator1.Accumulate({&keys1, &tA1, &tB1}), IsOk());
  // Totals: [4, 15, 27], [25, 7, 14]
  Tensor keys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"one", "zero", "one", "three"}))
          .value();
  Tensor tA2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor tB2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 2, 8})).value();
  EXPECT_THAT(aggregator1.Accumulate({&keys2, &tA2, &tB2}), IsOk());
  // aggregator1 totals: [9, 26, 27, 2], [28, 10, 14, 8]
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(2));

  // Create a second aggregator and accumulate an input with overlapping keys.
  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));
  Tensor keys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"three", "two", "three", "two"}))
          .value();
  Tensor tA3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({11, 3, 4, 2})).value();
  Tensor tB3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({6, 1, 4, 12})).value();
  EXPECT_THAT(aggregator2.Accumulate({&keys3, &tA3, &tB3}), IsOk());

  // Merge the two aggregators together.
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor({4}, {9, 26, 32, 17}));
  EXPECT_THAT(result.value()[2], IsTensor({4}, {28, 10, 27, 18}));
}

TEST(GroupByAggregatorTest, Merge_NoValueTensors_Succeeds) {
  const TensorShape shape = {4};
  std::vector<Intrinsic> intrinsics1;
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));
  Tensor keys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"zero", "zero", "one", "two"}))
          .value();
  EXPECT_THAT(aggregator1.Accumulate({&keys1}), IsOk());
  Tensor keys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"one", "zero", "one", "three"}))
          .value();
  EXPECT_THAT(aggregator1.Accumulate({&keys2}), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(2));

  // Create a second aggregator and accumulate an input with overlapping keys.
  std::vector<Intrinsic> intrinsics2;
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));
  Tensor keys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"three", "two", "three", "two"}))
          .value();
  EXPECT_THAT(aggregator2.Accumulate({&keys3}), IsOk());

  // Merge the two aggregators together.
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
}

TEST(GroupByAggregatorTest, Merge_MultipleKeyTensors_Succeeds) {
  const TensorShape shape = {4};
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key1", DT_STRING),
                                         CreateTensorSpec("key2", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));
  Tensor sizeKeys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "large"}))
          .value();
  Tensor animalKeys1 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1.Accumulate({&sizeKeys1, &animalKeys1, &t1}), IsOk());
  // aggregator1 totals: [4, 15, 27]
  Tensor sizeKeys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"small", "large", "small", "small"}))
          .value();
  Tensor animalKeys2 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator1.Accumulate({&sizeKeys2, &animalKeys2, &t2}), IsOk());
  // aggregator1 totals: [9, 26, 27, 2]

  // Create a second GroupByAggregator.
  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key1", DT_STRING),
                                         CreateTensorSpec("key2", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));
  Tensor sizeKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "small"}))
          .value();
  Tensor animalKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"dog", "dog", "rabbit", "cat"}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator2.Accumulate({&sizeKeys3, &animalKeys3, &t3}), IsOk());

  // Merge the two aggregators together.
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  // Merged totals: [9, 46, 41, 2, 7]
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));
  // Verify the resulting tensors.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>(
                  {5}, {"large", "small", "large", "small", "small"}));
  EXPECT_THAT(
      result.value()[1],
      IsTensor<string_view>({5}, {"cat", "cat", "dog", "dog", "rabbit"}));
  EXPECT_THAT(result.value()[2], IsTensor({5}, {9, 46, 41, 2, 7}));
}

TEST(GroupByAggregatorTest,
     Merge_MultipleKeyTensors_SomeKeysNotInOutput_Succeeds) {
  const TensorShape shape = {4};
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> input_key_specs_1 = {
      CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_STRING)};
  std::vector<TensorSpec> output_key_specs_1 = {
      CreateTensorSpec("", DT_STRING), CreateTensorSpec("animals", DT_STRING)};
  GroupByAggregator aggregator1(input_key_specs_1, &output_key_specs_1,
                                std::move(intrinsics1));
  Tensor sizeKeys1 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "large"}))
          .value();
  Tensor animalKeys1 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1.Accumulate({&sizeKeys1, &animalKeys1, &t1}), IsOk());
  // aggregator1 totals: [4, 15, 27]
  Tensor sizeKeys2 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"small", "large", "small", "small"}))
          .value();
  Tensor animalKeys2 =
      Tensor::Create(DT_STRING, shape,
                     CreateTestData<string_view>({"cat", "cat", "cat", "dog"}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator1.Accumulate({&sizeKeys2, &animalKeys2, &t2}), IsOk());
  // aggregator1 totals: [9, 26, 27, 2]

  // Create a second GroupByAggregator.
  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> input_key_specs_2 = {
      CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_STRING)};
  std::vector<TensorSpec> output_key_specs_2 = {
      CreateTensorSpec("", DT_STRING), CreateTensorSpec("animals", DT_STRING)};
  GroupByAggregator aggregator2(input_key_specs_2, &output_key_specs_2,
                                std::move(intrinsics2));
  Tensor sizeKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"large", "large", "small", "small"}))
          .value();
  Tensor animalKeys3 =
      Tensor::Create(
          DT_STRING, shape,
          CreateTestData<string_view>({"dog", "dog", "rabbit", "cat"}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator2.Accumulate({&sizeKeys3, &animalKeys3, &t3}), IsOk());

  // Merge the two aggregators together.
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  // Merged totals: [9, 46, 41, 2, 7]
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  // Verify the resulting tensors.
  // Only the second key tensor should be included in the output.
  EXPECT_THAT(result.value().size(), Eq(2));
  EXPECT_THAT(
      result.value()[0],
      IsTensor<string_view>({5}, {"cat", "cat", "dog", "dog", "rabbit"}));
  EXPECT_THAT(result.value()[1], IsTensor({5}, {9, 46, 41, 2, 7}));
}

TEST(GroupByAggregatorTest, Merge_NoKeyTensors) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t1}), IsOk());
  Tensor t2 = Tensor::Create(DT_INT32, {3}, CreateTestData({10, 5, 1})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t2}), IsOk());

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));
  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({3, 11, 7, 20, 5})).value();
  EXPECT_THAT(aggregator2.Accumulate({&t3}), IsOk());

  // Merge the two aggregators together.
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor<int32_t>({1}, {108}));
}

TEST(GroupByAggregatorTest, Merge_BothEmpty_Succeeds) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));

  // Merge the two empty aggregators together.
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(0));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(0));
}

TEST(GroupByAggregatorTest, Merge_ThisOutputEmpty_Succeeds) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));

  Tensor keys1 = Tensor::Create(DT_STRING, {2},
                                CreateTestData<string_view>({"zero", "one"}))
                     .value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 3})).value();
  EXPECT_THAT(aggregator2.Accumulate({&keys1, &t1}), IsOk());
  // aggregator2 totals: [1, 3]
  Tensor keys2 =
      Tensor::Create(DT_STRING, {6},
                     CreateTestData<string_view>(
                         {"two", "one", "zero", "one", "three", "two"}))
          .value();
  Tensor t2 = Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 1, 2, 4, 9}))
                  .value();
  EXPECT_THAT(aggregator2.Accumulate({&keys2, &t2}), IsOk());
  // aggregator2 totals: [2, 10, 19, 4]

  // Merge aggregator2 into aggregator1 which has not received any inputs.
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(2));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor({4}, {2, 10, 19, 4}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(GroupByAggregatorTest, Merge_OtherOutputEmpty_Succeeds) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));

  Tensor keys1 = Tensor::Create(DT_STRING, {2},
                                CreateTestData<string_view>({"zero", "one"}))
                     .value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 3})).value();
  EXPECT_THAT(aggregator1.Accumulate({&keys1, &t1}), IsOk());
  // aggregator1 totals: [1, 3]
  Tensor keys2 =
      Tensor::Create(DT_STRING, {6},
                     CreateTestData<string_view>(
                         {"two", "one", "zero", "one", "three", "two"}))
          .value();
  Tensor t2 = Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 1, 2, 4, 9}))
                  .value();
  EXPECT_THAT(aggregator1.Accumulate({&keys2, &t2}), IsOk());
  // aggregator1 totals: [2, 10, 19, 4]

  // Merge with aggregator2 which has not received any inputs.
  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)), IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(2));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(2));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0],
              IsTensor<string_view>({4}, {"zero", "one", "two", "three"}));
  EXPECT_THAT(result.value()[1], IsTensor({4}, {2, 10, 19, 4}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

// TODO(team): Expand on the tests below to check that even when
// Accumulate or MergeWith return INVALID_ARGUMENT, the internal state of the
// GroupByAggregator remains unaffected, exactly the same as if the failed
// operation had never been called.
TEST(GroupByAggregatorTest, Accumulate_KeyTensorHasIncompatibleDataType) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor key =
      Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1.2})).value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({0})).value();
  Status s = aggregator.Accumulate({&key, &t});
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      testing::HasSubstr("Tensor at position 0 did not have expected dtype"));
}

TEST(GroupByAggregatorTest, Accumulate_ValueTensorHasIncompatibleDataType) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor key =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"key_string"}))
          .value();
  Tensor t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1.2})).value();
  Status s = aggregator.Accumulate({&key, &t});
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      testing::HasSubstr("Tensor at position 1 did not have expected dtype"));
}

TEST(GroupByAggregatorTest, Accumulate_FewerTensorsThanExpected) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key1", DT_STRING),
                                       CreateTensorSpec("key2", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor key =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"key_string"}))
          .value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Status s = aggregator.Accumulate({&key, &t});
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr(
                  "GroupByAggregator should operate on 3 input tensors"));
}

TEST(GroupByAggregatorTest, Accumulate_MoreTensorsThanExpected) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor key1 = Tensor::Create(DT_STRING, {},
                               CreateTestData<string_view>({"key_string_1"}))
                    .value();
  Tensor key2 = Tensor::Create(DT_STRING, {},
                               CreateTestData<string_view>({"key_string_2"}))
                    .value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Status s = aggregator.Accumulate({&key1, &key2, &t});
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr(
                  "GroupByAggregator should operate on 2 input tensors"));
}

TEST(GroupByAggregatorTest, Accumulate_KeyTensorSmallerThanValueTensor) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor key = Tensor::Create(DT_STRING, {},
                              CreateTestData<string_view>({"key_string_1"}))
                   .value();
  Tensor t = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 2})).value();
  Status s = aggregator.Accumulate({&key, &t});
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr("Shape of value tensor at index 1 "
                                 "does not match expected shape."));
}

TEST(GroupByAggregatorTest, Accumulate_KeyTensorLargerThanValueTensor) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor key =
      Tensor::Create(DT_STRING, {3},
                     CreateTestData<string_view>(
                         {"key_string_1", "key_string_2", "key_string_3"}))
          .value();
  Tensor t = Tensor::Create(DT_INT32, {2}, CreateTestData({1, 2})).value();
  Status s = aggregator.Accumulate({&key, &t});
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr("Shape of value tensor at index 1 "
                                 "does not match expected shape."));
}

TEST(GroupByAggregatorTest, Accumulate_MultidimensionalTensorsNotSupported) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator(key_specs, &key_specs, std::move(intrinsics));
  Tensor key = Tensor::Create(DT_STRING, {2, 2},
                              CreateTestData<string_view>({"a", "b", "c", "d"}))
                   .value();
  Tensor t =
      Tensor::Create(DT_INT32, {2, 2}, CreateTestData({1, 2, 3, 4})).value();
  Status s = aggregator.Accumulate({&key, &t});
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr(
                  "Only scalar or one-dimensional tensors are supported."));
}

TEST(GroupByAggregatorTest, Merge_IncompatibleKeyType) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_FLOAT)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));
  Status s = aggregator1.MergeWith(std::move(aggregator2));
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr("Expected other GroupByAggregator to have "
                                 "the same key input and output specs"));
}

TEST(GroupByAggregatorTest, Merge_IncompatibleOutputKeySpec) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> input_key_specs_1 = {
      CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_STRING)};
  std::vector<TensorSpec> output_key_specs_1 = {
      CreateTensorSpec("", DT_STRING), CreateTensorSpec("key2", DT_STRING)};
  GroupByAggregator aggregator1(input_key_specs_1, &output_key_specs_1,
                                std::move(intrinsics1));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> input_key_specs_2 = {
      CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_STRING)};
  // Key1 is included in the output of aggregator2 but not included in the
  // output of aggregator1.
  std::vector<TensorSpec> output_key_specs_2 = {
      CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_STRING)};
  GroupByAggregator aggregator2(input_key_specs_2, &output_key_specs_2,
                                std::move(intrinsics2));
  Status s = aggregator1.MergeWith(std::move(aggregator2));
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr("Expected other GroupByAggregator to have "
                                 "the same key input and output specs"));
}

TEST(GroupByAggregatorTest,
     Merge_IncompatibleKeyType_InputTensorListTypesMatch) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING),
                                         CreateTensorSpec("key", DT_INT32)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));
  Status s = aggregator1.MergeWith(std::move(aggregator2));
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr("Expected other GroupByAggregator to have "
                                 "the same key input and output specs"));
}

TEST(GroupByAggregatorTest, Merge_IncompatibleValueType) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<float>(DT_FLOAT));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));
  Status s = aggregator1.MergeWith(std::move(aggregator2));
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr("Expected other GroupByAggregator to "
                                 "use inner intrinsics with the same inputs"));
}

TEST(GroupByAggregatorTest, Merge_DifferentNumKeys) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING),
                                         CreateTensorSpec("key2", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));
  Status s = aggregator1.MergeWith(std::move(aggregator2));
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr("Expected other GroupByAggregator to have "
                                 "the same key input and output specs"));
}

TEST(GroupByAggregatorTest, Merge_NonzeroVsZeroNumKeys) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));
  Status s = aggregator1.MergeWith(std::move(aggregator2));
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr("Expected other GroupByAggregator to have "
                                 "the same key input and output specs"));
}

TEST(GroupByAggregatorTest, Merge_DifferentNumValues) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));
  Status s = aggregator1.MergeWith(std::move(aggregator2));
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              testing::HasSubstr("Expected other GroupByAggregator to "
                                 "use the same number of inner intrinsics"));
}

TEST(GroupByAggregatorTest, Merge_DifferentTensorAggregatorImpl) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));

  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  SumGroupingAggregator<int32_t> sum_aggregator(DT_INT32);
  Status s = aggregator1.MergeWith(std::move(sum_aggregator));
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      testing::HasSubstr("Can only merge with another GroupByAggregator"));
}

TEST(GroupByAggregatorTest, FailsAfterBeingConsumed) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  Tensor key =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"foo"}))
          .value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();

  EXPECT_THAT(aggregator1.Accumulate({&key, &t}), IsOk());
  EXPECT_THAT(std::move(aggregator1).Report(), IsOk());

  // Now the aggregator instance has been consumed and should fail any
  // further operations.
  EXPECT_THAT(aggregator1.CanReport(), IsFalse());  // NOLINT
  EXPECT_THAT(std::move(aggregator1).Report(),
              IsCode(FAILED_PRECONDITION));        // NOLINT
  EXPECT_THAT(aggregator1.Accumulate({&key, &t}),  // NOLINT
              IsCode(FAILED_PRECONDITION));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));

  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)),  // NOLINT
              IsCode(FAILED_PRECONDITION));

  // Passing this aggregator as an argument to another MergeWith must fail
  // too.
  std::vector<Intrinsic> intrinsics3;
  intrinsics3.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_3 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator3(key_specs_3, &key_specs_3,
                                std::move(intrinsics3));
  EXPECT_THAT(aggregator3.MergeWith(std::move(aggregator1)),  // NOLINT
              IsCode(FAILED_PRECONDITION));
}

TEST(GroupByAggregatorTest, FailsAfterBeingConsumed_WhenEmpty) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));

  EXPECT_THAT(std::move(aggregator1).Report(), IsOk());

  // Now the aggregator instance has been consumed and should fail any
  // further operations.
  EXPECT_THAT(aggregator1.CanReport(), IsFalse());  // NOLINT
  EXPECT_THAT(std::move(aggregator1).Report(),
              IsCode(FAILED_PRECONDITION));  // NOLINT

  Tensor key =
      Tensor::Create(DT_STRING, {}, CreateTestData<string_view>({"foo"}))
          .value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  EXPECT_THAT(aggregator1.Accumulate({&key, &t}),  // NOLINT
              IsCode(FAILED_PRECONDITION));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));

  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)),  // NOLINT
              IsCode(FAILED_PRECONDITION));

  // Passing this aggregator as an argument to another MergeWith must fail
  // too.
  std::vector<Intrinsic> intrinsics3;
  intrinsics3.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_3 = {CreateTensorSpec("key", DT_STRING)};
  GroupByAggregator aggregator3(key_specs_3, &key_specs_3,
                                std::move(intrinsics3));
  EXPECT_THAT(aggregator3.MergeWith(std::move(aggregator1)),  // NOLINT
              IsCode(FAILED_PRECONDITION));
}

TEST(GroupByAggregatorTest, FailsAfterBeingConsumed_WhenNoKeys) {
  std::vector<Intrinsic> intrinsics1;
  intrinsics1.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_1 = {};
  GroupByAggregator aggregator1(key_specs_1, &key_specs_1,
                                std::move(intrinsics1));
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();

  EXPECT_THAT(aggregator1.Accumulate({&t}), IsOk());
  EXPECT_THAT(std::move(aggregator1).Report(), IsOk());

  // Now the aggregator instance has been consumed and should fail any
  // further operations.
  EXPECT_THAT(aggregator1.CanReport(), IsFalse());  // NOLINT
  EXPECT_THAT(std::move(aggregator1).Report(),
              IsCode(FAILED_PRECONDITION));  // NOLINT
  EXPECT_THAT(aggregator1.Accumulate({&t}),  // NOLINT
              IsCode(FAILED_PRECONDITION));

  std::vector<Intrinsic> intrinsics2;
  intrinsics2.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_2 = {};
  GroupByAggregator aggregator2(key_specs_2, &key_specs_2,
                                std::move(intrinsics2));

  EXPECT_THAT(aggregator1.MergeWith(std::move(aggregator2)),  // NOLINT
              IsCode(FAILED_PRECONDITION));

  // Passing this aggregator as an argument to another MergeWith must fail
  // too.
  std::vector<Intrinsic> intrinsics3;
  intrinsics3.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> key_specs_3 = {};
  GroupByAggregator aggregator3(key_specs_3, &key_specs_3,
                                std::move(intrinsics3));
  EXPECT_THAT(aggregator3.MergeWith(std::move(aggregator1)),  // NOLINT
              IsCode(FAILED_PRECONDITION));
}

TEST(GroupByAggregatorDeathTest, NoInputTensors) {
  std::vector<Intrinsic> intrinsics;
  std::vector<TensorSpec> key_specs = {};
  EXPECT_DEATH(
      new GroupByAggregator(key_specs, &key_specs, std::move(intrinsics)),
      "Must operate on a nonzero number of tensors.");
}

TEST(GroupByAggregatorDeathTest, InputAndOutputKeySizeMismatchCheckFailure) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> input_key_specs = {
      CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_FLOAT)};
  std::vector<TensorSpec> output_key_specs = {
      CreateTensorSpec("animals", DT_STRING)};
  EXPECT_DEATH(new GroupByAggregator(input_key_specs, &output_key_specs,
                                     std::move(intrinsics)),
               "Size of input_key_specs must match size of output_key_specs.");
}

TEST(GroupByAggregatorDeathTest, InputAndOutputDtypeMismatchCheckFailure) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> input_key_specs = {
      CreateTensorSpec("key1", DT_STRING), CreateTensorSpec("key2", DT_FLOAT)};
  std::vector<TensorSpec> output_key_specs = {
      CreateTensorSpec("", DT_STRING), CreateTensorSpec("animals", DT_STRING)};
  EXPECT_DEATH(
      new GroupByAggregator(input_key_specs, &output_key_specs,
                            std::move(intrinsics)),
      "Input and output tensor specifications must have matching data types");
}

TEST(GroupByAggregatorDeathTest, InputShapeCheckFailure) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> input_key_specs = {TensorSpec("key", DT_STRING, {8})};
  std::vector<TensorSpec> output_key_specs = {
      TensorSpec("key", DT_STRING, {-1})};
  EXPECT_DEATH(
      new GroupByAggregator(input_key_specs, &output_key_specs,
                            std::move(intrinsics)),
      "All input and output tensors must have one dimension of unknown size.");
}

TEST(GroupByAggregatorDeathTest, OutputShapeCheckFailure) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(CreateInnerIntrinsic<int32_t>(DT_INT32));
  std::vector<TensorSpec> input_key_specs = {
      TensorSpec("key", DT_STRING, {-1})};
  std::vector<TensorSpec> output_key_specs = {
      TensorSpec("key", DT_STRING, {8})};
  EXPECT_DEATH(
      new GroupByAggregator(input_key_specs, &output_key_specs,
                            std::move(intrinsics)),
      "All input and output tensors must have one dimension of unknown size.");
}

TEST(GroupByAggregatorDeathTest, TypesAndAggregatorsTypeMismatch) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(
      Intrinsic({CreateTensorSpec("value", DT_INT32)},
                {CreateTensorSpec("value", DT_INT32)},
                std::make_unique<SumGroupingAggregator<float>>(DT_FLOAT)));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  EXPECT_DEATH(
      new GroupByAggregator(key_specs, &key_specs, std::move(intrinsics)),
      "Expected OneDimGroupingAggregator of type");
}

TEST(GroupByAggregatorDeathTest, SubIntrinsicNotGroupingAggregator) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(Intrinsic(
      {CreateTensorSpec("value", DT_INT32)},
      {CreateTensorSpec("value", DT_INT32)},
      std::make_unique<SumAggregator<int32_t>>(DT_INT32, TensorShape{})));
  std::vector<TensorSpec> key_specs = {CreateTensorSpec("key", DT_STRING)};
  EXPECT_DEATH(
      new GroupByAggregator(key_specs, &key_specs, std::move(intrinsics)),
      "Expected OneDimGroupingAggregator of type");
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
