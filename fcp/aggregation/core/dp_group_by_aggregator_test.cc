/*
 * Copyright 2024 Google LLC
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
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/agg_vector_aggregator.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/mutable_vector_data.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
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
using testing::HasSubstr;
using testing::IsTrue;

TensorSpec CreateTensorSpec(std::string name, DataType dtype) {
  return TensorSpec(name, dtype, {-1});
}

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

template <typename EpsilonType, typename DeltaType, typename L0_BoundType>
std::vector<Tensor> CreateTopLevelParameters(EpsilonType epsilon,
                                             DeltaType delta,
                                             L0_BoundType l0_bound) {
  std::vector<Tensor> parameters;

  std::unique_ptr<MutableVectorData<EpsilonType>> epsilon_tensor =
      CreateTestData<EpsilonType>({epsilon});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<EpsilonType>::kDataType, {},
                     std::move(epsilon_tensor))
          .value());

  std::unique_ptr<MutableVectorData<DeltaType>> delta_tensor =
      CreateTestData<DeltaType>({delta});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<DeltaType>::kDataType, {},
                     std::move(delta_tensor))
          .value());

  std::unique_ptr<MutableVectorData<L0_BoundType>> l0_bound_tensor =
      CreateTestData<L0_BoundType>({l0_bound});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<L0_BoundType>::kDataType, {},
                     std::move(l0_bound_tensor))
          .value());
  return parameters;
}

std::vector<Tensor> CreateTopLevelParameters(double epsilon, double delta,
                                             int64_t l0_bound) {
  return CreateTopLevelParameters<double, double, int64_t>(epsilon, delta,
                                                           l0_bound);
}

std::vector<Tensor> CreateFewTopLevelParameters() {
  std::vector<Tensor> parameters;

  auto epsilon_tensor = CreateTestData({1.0});
  parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, std::move(epsilon_tensor)).value());

  return parameters;
}

std::vector<Tensor> CreateManyTopLevelParameters() {
  std::vector<Tensor> parameters;

  for (int i = 0; i < 4; i++) {
    auto t = CreateTestData({1.0});
    parameters.push_back(Tensor::Create(DT_DOUBLE, {}, std::move(t)).value());
  }

  return parameters;
}

template <typename InputType>
std::vector<Tensor> CreateNestedParameters(InputType linfinity_bound,
                                           double l1_bound, double l2_bound) {
  std::vector<Tensor> parameters;

  parameters.push_back(
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, {},
                     CreateTestData<InputType>({linfinity_bound}))
          .value());

  parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({l1_bound}))
          .value());

  parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({l2_bound}))
          .value());

  return parameters;
}

template <typename InputType>
Intrinsic CreateInnerIntrinsic(InputType linfinity_bound, double l1_bound,
                               double l2_bound) {
  return Intrinsic{
      "GoogleSQL:dp_sum",
      {CreateTensorSpec("value", internal::TypeTraits<InputType>::kDataType)},
      {CreateTensorSpec("value", internal::TypeTraits<InputType>::kDataType)},
      {CreateNestedParameters<InputType>(linfinity_bound, l1_bound, l2_bound)},
      {}};
}

template <typename InputType>
Intrinsic CreateIntrinsic(double epsilon = 100.0, double delta = 0.001,
                          int64_t l0_bound = 100,
                          InputType linfinity_bound = 100, double l1_bound = -1,
                          double l2_bound = -1) {
  Intrinsic intrinsic{"fedsql_dp_group_by",
                      {CreateTensorSpec("key", DT_STRING)},
                      {CreateTensorSpec("key_out", DT_STRING)},
                      {CreateTopLevelParameters(epsilon, delta, l0_bound)},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType>(linfinity_bound, l1_bound, l2_bound));
  return intrinsic;
}

// First batch of tests are about the parameters in the intrinsic
TEST(DPGroupByAggregatorTest, CatchWrongNumberOfParameters) {
  Intrinsic too_few{"fedsql_dp_group_by",
                    {CreateTensorSpec("key", DT_STRING)},
                    {CreateTensorSpec("key_out", DT_STRING)},
                    {CreateFewTopLevelParameters()},
                    {}};
  auto too_few_status = CreateTensorAggregator(too_few).status();
  EXPECT_THAT(too_few_status, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(too_few_status.message(),
              HasSubstr("Expected 3 parameters but got 1 of them"));

  // Too many parameters
  Intrinsic too_many{"fedsql_dp_group_by",
                     {CreateTensorSpec("key", DT_STRING)},
                     {CreateTensorSpec("key_out", DT_STRING)},
                     {CreateManyTopLevelParameters()},
                     {}};
  auto too_many_status = CreateTensorAggregator(too_many).status();
  EXPECT_THAT(too_many_status, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(too_many_status.message(),
              HasSubstr("Expected 3 parameters but got 4 of them"));
}

TEST(DPGroupByAggregatorTest, CatchInvalidParameterTypes) {
  Intrinsic intrinsic0{
      "fedsql_dp_group_by",
      {CreateTensorSpec("key", DT_STRING)},
      {CreateTensorSpec("key_out", DT_STRING)},
      {CreateTopLevelParameters<string_view, double, int64_t>("x", 0.1, 10)},
      {}};
  auto bad_epsilon = CreateTensorAggregator(intrinsic0).status();
  EXPECT_THAT(bad_epsilon, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(bad_epsilon.message(), HasSubstr("must be numerical"));

  Intrinsic intrinsic1{
      "fedsql_dp_group_by",
      {CreateTensorSpec("key", DT_STRING)},
      {CreateTensorSpec("key_out", DT_STRING)},
      {CreateTopLevelParameters<double, string_view, int64_t>(1.0, "x", 10)},
      {}};
  auto bad_delta = CreateTensorAggregator(intrinsic1).status();
  EXPECT_THAT(bad_delta, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(bad_delta.message(), HasSubstr("must be numerical"));

  Intrinsic intrinsic2{
      "fedsql_dp_group_by",
      {CreateTensorSpec("key", DT_STRING)},
      {CreateTensorSpec("key_out", DT_STRING)},
      {CreateTopLevelParameters<double, double, string_view>(1.0, 0.1, "x")},
      {}};
  auto bad_l0_bound = CreateTensorAggregator(intrinsic2).status();
  EXPECT_THAT(bad_l0_bound, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(bad_l0_bound.message(), HasSubstr("must be numerical"));
}

TEST(DPGroupByAggregatorTest, CatchInvalidParameterValues) {
  Intrinsic intrinsic0 = CreateIntrinsic<int64_t>(-1, 0.001, 10);
  auto bad_epsilon = CreateTensorAggregator(intrinsic0).status();
  EXPECT_THAT(bad_epsilon, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(bad_epsilon.message(), HasSubstr("Epsilon must be positive"));

  Intrinsic intrinsic1 = CreateIntrinsic<int64_t>(1.0, -1, 10);
  auto bad_delta = CreateTensorAggregator(intrinsic1).status();
  EXPECT_THAT(bad_delta, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(bad_delta.message(), HasSubstr("Delta must lie between 0 and 1"));

  Intrinsic intrinsic2 = CreateIntrinsic<int64_t>(1.0, 0.001, -1);
  auto bad_l0_bound = CreateTensorAggregator(intrinsic2).status();
  EXPECT_THAT(bad_l0_bound, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(bad_l0_bound.message(), HasSubstr("L0 bound must be positive"));
}

// Second batch of tests are dedicated to norm bounding, when there is only one
// inner aggregation (GROUP BY key, SUM(value))

// Function to execute the DPGroupByAggregator on one input where there is just
// one key per contribution and each contribution is to one aggregation
template <typename InputType>
StatusOr<OutputTensorList> SingleKeySingleAgg(
    const Intrinsic& intrinsic, const TensorShape shape,
    std::initializer_list<string_view> key_list,
    std::initializer_list<InputType> value_list) {
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor keys =
      Tensor::Create(DT_STRING, shape, CreateTestData<string_view>(key_list))
          .value();

  Tensor value_tensor =
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, shape,
                     CreateTestData<InputType>(value_list))
          .value();
  auto acc_status = group_by_aggregator->Accumulate({&keys, &value_tensor});

  EXPECT_THAT(acc_status, IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(1));

  return std::move(*group_by_aggregator).Report();
}

TEST(DPGroupByAggregatorTest, SingleKeySingleAggWithL0Bound) {
  // L0 bounding involves randomness so we should repeat things to catch errors.
  for (int i = 0; i < 9; i++) {
    auto intrinsic = CreateIntrinsic<int64_t>(100, 0.01, 1);
    auto result = SingleKeySingleAgg<int64_t>(
        intrinsic, {4}, {"zero", "one", "two", "zero"}, {1, 3, 15, 27});
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(2));

    // output is either (zero:28) OR (one:3) OR (two:15)
    // OR their reversals
    std::vector<string_view> possible_keys({"zero", "one", "two"});
    std::vector<int64_t> possible_values({28, 3, 15});
    bool found_match = false;
    for (int j = 0; j < 3; j++) {
      auto key_matcher = IsTensor<string_view>({1}, {possible_keys[j]});
      auto callable_key_matcher =
          ::testing::internal::MakePredicateFormatterFromMatcher(key_matcher);
      auto value_matcher = IsTensor<int64_t>({1}, {possible_values[j]});
      auto callable_value_matcher =
          ::testing::internal::MakePredicateFormatterFromMatcher(value_matcher);

      if (callable_key_matcher("result.value()[0]", result.value()[0]) &&
          callable_value_matcher("result.value()[1]", result.value()[1])) {
        found_match = true;
        break;
      }
    }
    EXPECT_TRUE(found_match);

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
  }
}

TEST(DPGroupByAggregatorTest, SingleKeySingleAggWithL0LinfinityBounds) {
  for (int i = 0; i < 9; i++) {
    // Use the same setup as before but now impose a maximum magnitude of 12
    auto intrinsic = CreateIntrinsic<int64_t>(100, 0.01, 1, 12);
    auto result = SingleKeySingleAgg<int64_t>(
        intrinsic, {4}, {"zero", "one", "two", "zero"}, {1, 3, 15, 27});
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(2));

    // output is either (zero:12) OR (one:3) OR (two:12)
    // OR their reversals
    std::vector<string_view> possible_keys({"zero", "one", "two"});
    std::vector<int64_t> possible_values({12, 3, 12});
    bool found_match = false;
    for (int j = 0; j < 3; j++) {
      auto key_matcher = IsTensor<string_view>({1}, {possible_keys[j]});
      auto callable_key_matcher =
          ::testing::internal::MakePredicateFormatterFromMatcher(key_matcher);
      auto value_matcher = IsTensor<int64_t>({1}, {possible_values[j]});
      auto callable_value_matcher =
          ::testing::internal::MakePredicateFormatterFromMatcher(value_matcher);

      if (callable_key_matcher("result.value()[0]", result.value()[0]) &&
          callable_value_matcher("result.value()[1]", result.value()[1])) {
        found_match = true;
        break;
      }
    }
    EXPECT_TRUE(found_match);

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
  }
}

TEST(DPGroupByAggregatorTest, SingleKeySingleAggWithL0LinfinityL1Bounds) {
  for (int i = 0; i < 9; i++) {
    // L0 bound is 4 (four keys), Linfinity bound is 50 (|value| <= 50),
    // and L1 bound is 100 (sum over |value| is <= 100)
    auto intrinsic = CreateIntrinsic<int64_t>(100, 0.01, 4, 50, 100);
    auto result = SingleKeySingleAgg<int64_t>(
        intrinsic, {5}, {"zero", "one", "two", "three", "four"},
        {60, 60, 60, 60, 60});
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(2));

    // output should look like (25, 25, 25, 25)
    EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {25, 25, 25, 25}));

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
  }
}

TEST(DPGroupByAggregatorTest, SingleKeySingleAggWithAllBounds) {
  for (int i = 0; i < 9; i++) {
    // L0 bound is 4 (four keys), Linfinity bound is 50 (|value| <= 50),
    // L1 bound is 100 (sum over |value| is <= 100), and L2 bound is 10
    auto intrinsic = CreateIntrinsic<int64_t>(100, 0.01, 4, 50, 100, 10);
    auto result = SingleKeySingleAgg<int64_t>(
        intrinsic, {5}, {"zero", "one", "two", "three", "four"},
        {60, 60, 60, 60, 60});
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(2));

    // output should look like (5, 5, 5, 5)
    EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {5, 5, 5, 5}));

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
  }
}

// Third: test norm bounding when there are > 1 inner aggregations
// (SUM(value1), SUM(value2) \\ GROUP BY key)
template <typename InputType>
Intrinsic CreateIntrinsic2(double epsilon = 100.0, double delta = 0.001,
                           int64_t l0_bound = 100,
                           InputType linfinity_bound1 = 100,
                           double l1_bound1 = -1, double l2_bound1 = -1,
                           InputType linfinity_bound2 = 100,
                           double l1_bound2 = -1, double l2_bound2 = -1) {
  Intrinsic intrinsic{"fedsql_dp_group_by",
                      {CreateTensorSpec("key", DT_STRING)},
                      {CreateTensorSpec("key_out", DT_STRING)},
                      {CreateTopLevelParameters(epsilon, delta, l0_bound)},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType>(linfinity_bound1, l1_bound1, l2_bound1));
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType>(linfinity_bound2, l1_bound2, l2_bound2));
  return intrinsic;
}

// Function to execute the DPGroupByAggregator on one input where there is just
// one key per contribution and each contribution is to two aggregations
template <typename InputType>
StatusOr<OutputTensorList> SingleKeyDoubleAgg(
    const Intrinsic& intrinsic, const TensorShape shape,
    std::initializer_list<string_view> key_list,
    std::initializer_list<InputType> value_list1,
    std::initializer_list<InputType> value_list2) {
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor keys =
      Tensor::Create(DT_STRING, shape, CreateTestData<string_view>(key_list))
          .value();

  Tensor value_tensor1 =
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, shape,
                     CreateTestData<InputType>(value_list1))
          .value();
  Tensor value_tensor2 =
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, shape,
                     CreateTestData<InputType>(value_list2))
          .value();
  auto acc_status =
      group_by_aggregator->Accumulate({&keys, &value_tensor1, &value_tensor2});

  EXPECT_THAT(acc_status, IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(1));

  return std::move(*group_by_aggregator).Report();
}

TEST(DPGroupByAggregatorTest, SingleKeyDoubleAggWithAllBounds) {
  for (int i = 0; i < 9; i++) {
    // L0 bound is 4.
    // For agg 1, Linfinity bound is 20 and no other bounds provided.
    // For agg 2, Linfinity bound is 50, L1 bound is 100, L2 bound is 10.
    auto intrinsic =
        CreateIntrinsic2<int64_t>(100, 0.01, 4, 20, -1, -1, 50, 100, 10);
    auto result = SingleKeyDoubleAgg<int64_t>(
        intrinsic, {5}, {"zero", "one", "two", "three", "four"},
        {60, 60, 60, 60, 60}, {60, 60, 60, 60, 60});
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(3));

    // first output should look like (20, 20, 20, 20)
    EXPECT_THAT(result.value()[1], IsTensor<int64_t>({4}, {20, 20, 20, 20}));

    // second output should look like (5, 5, 5, 5)
    EXPECT_THAT(result.value()[2], IsTensor<int64_t>({4}, {5, 5, 5, 5}));

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
    EXPECT_TRUE(result.value()[2].is_dense());
  }
}

// Fourth: test norm bounding, when there are > 1 keys and > 1 inner
// aggregations. (SUM(value1), SUM(value2) \\ GROUP BY key1, key 2)
template <typename InputType>
Intrinsic CreateIntrinsic3(double epsilon = 100.0, double delta = 0.001,
                           int64_t l0_bound = 100,
                           InputType linfinity_bound1 = 100,
                           double l1_bound1 = -1, double l2_bound1 = -1,
                           InputType linfinity_bound2 = 100,
                           double l1_bound2 = -1, double l2_bound2 = -1) {
  Intrinsic intrinsic{"fedsql_dp_group_by",
                      {CreateTensorSpec("key1", DT_STRING),
                       CreateTensorSpec("key2", DT_STRING)},
                      {CreateTensorSpec("key1_out", DT_STRING),
                       CreateTensorSpec("key2_out", DT_STRING)},
                      {CreateTopLevelParameters(epsilon, delta, l0_bound)},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType>(linfinity_bound1, l1_bound1, l2_bound1));
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType>(linfinity_bound2, l1_bound2, l2_bound2));
  return intrinsic;
}

template <typename InputType>
StatusOr<OutputTensorList> DoubleKeyDoubleAgg(
    const Intrinsic& intrinsic, const TensorShape shape,
    std::initializer_list<string_view> key_list1,
    std::initializer_list<string_view> key_list2,
    std::initializer_list<InputType> value_list1,
    std::initializer_list<InputType> value_list2) {
  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor keys1 =
      Tensor::Create(DT_STRING, shape, CreateTestData<string_view>(key_list1))
          .value();
  Tensor keys2 =
      Tensor::Create(DT_STRING, shape, CreateTestData<string_view>(key_list2))
          .value();

  Tensor value_tensor1 =
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, shape,
                     CreateTestData<InputType>(value_list1))
          .value();
  Tensor value_tensor2 =
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, shape,
                     CreateTestData<InputType>(value_list2))
          .value();
  auto acc_status = group_by_aggregator->Accumulate(
      {&keys1, &keys2, &value_tensor1, &value_tensor2});

  EXPECT_THAT(acc_status, IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());
  EXPECT_THAT(group_by_aggregator->GetNumInputs(), Eq(1));

  return std::move(*group_by_aggregator).Report();
}

TEST(DPGroupByAggregatorTest, DoubleKeyDoubleAggWithAllBounds) {
  for (int i = 0; i < 9; i++) {
    // L0 bound is 4.
    // For agg 1, Linfinity bound is 20 and no other bounds provided.
    // For agg 2, Linfinity bound is 50, L1 bound is 100, L2 bound is 10.
    auto intrinsic =
        CreateIntrinsic3<int64_t>(100, 0.01, 4, 20, -1, -1, 50, 100, 10);
    auto result = DoubleKeyDoubleAgg<int64_t>(
        intrinsic, {5}, {"red", "green", "green", "blue", "gray"},
        {"zero", "one", "two", "three", "four"}, {60, 60, 60, 60, 60},
        {60, 60, 60, 60, 60});
    EXPECT_THAT(result, IsOk());
    EXPECT_THAT(result->size(), Eq(4));

    // first output should look like (20, 20, 20, 20)
    EXPECT_THAT(result.value()[2], IsTensor<int64_t>({4}, {20, 20, 20, 20}));

    // second output should look like (5, 5, 5, 5)
    EXPECT_THAT(result.value()[3], IsTensor<int64_t>({4}, {5, 5, 5, 5}));

    // Check density
    EXPECT_TRUE(result.value()[0].is_dense());
    EXPECT_TRUE(result.value()[1].is_dense());
    EXPECT_TRUE(result.value()[2].is_dense());
    EXPECT_TRUE(result.value()[3].is_dense());
  }
}

// Fifth: test norm bounding on key-less data (norm bound = magnitude bound)
TEST(DPGroupByAggregatorTest, NoKeyTripleAggWithAllBounds) {
  Intrinsic intrinsic{"fedsql_dp_group_by",
                      {},
                      {},
                      {CreateTopLevelParameters(100, 0.01, 100)},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<int32_t>(10, 9, 8));
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<int32_t>(100, 9, -1));
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<int32_t>(100, -1, -1));

  auto group_by_aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({11})).value();
  EXPECT_THAT(group_by_aggregator->Accumulate({&t1, &t2, &t3}), IsOk());
  EXPECT_THAT(group_by_aggregator->CanReport(), IsTrue());

  auto result = std::move(*group_by_aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(3));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor<int64_t>({1}, {8}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({1}, {9}));
  EXPECT_THAT(result.value()[2], IsTensor<int64_t>({1}, {11}));
}

// Finally, test merge: intermediary aggregates should not be clipped or noised

TEST(DPGroupByAggregatorTest, MergeDoesNotDistortData_SingleKey) {
  // For any single user's data we will give to the aggregators, the norm bounds
  // below do nothing: each Accumulate call has 1 distinct key and a value of 1,
  // which satisfies the L0 bound and Linfinity bound constraints.
  Intrinsic intrinsic = CreateIntrinsic<int64_t>(100, 0.001, 1, 1, -1, -1);
  auto agg1 = CreateTensorAggregator(intrinsic).value();
  auto agg2 = CreateTensorAggregator(intrinsic).value();

  // agg1 gets one person's data, mapping "key" to 1
  Tensor key1 =
      Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"key"}))
          .value();
  Tensor data1 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
  EXPECT_THAT(agg1->Accumulate({&key1, &data1}), IsOk());

  // agg2 gets data from a lot more people. At the end, it will map "other key"
  // to 1000 and "yet another key" to 1000.
  for (int i = 0; i < 1000; i++) {
    Tensor key_a = Tensor::Create(DT_STRING, {1},
                                  CreateTestData<string_view>({"other key"}))
                       .value();
    Tensor data_a =
        Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
    EXPECT_THAT(agg2->Accumulate({&key_a, &data_a}), IsOk());

    Tensor key_b =
        Tensor::Create(DT_STRING, {1},
                       CreateTestData<string_view>({"yet another key"}))
            .value();
    Tensor data_b =
        Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
    EXPECT_THAT(agg2->Accumulate({&key_b, &data_b}), IsOk());
  }

  // Merge the aggregators. The result should contain the 3 different keys.
  // "key" should map to 1 while the other two keys should map to 1000.
  // If we wrote merge wrong, the code might do the following to agg2:
  // (1) pick one of "other key" or "yet another key" at random (l0_bound = 1),
  // or (2) force one of the sums from 1000 to 1 (linfinity_bound_ = 1)
  // or both
  auto merge_status = agg1->MergeWith(std::move(*agg2));
  EXPECT_THAT(merge_status, IsOk());
  auto result = std::move(*agg1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value()[0].num_elements(), Eq(3));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({3}, {1, 1000, 1000}));
}

TEST(DPGroupByAggregatorTest, MergeDoesNotDistortData_MultiKey) {
  Intrinsic intrinsic = CreateIntrinsic3<int64_t>(100, 0.001, 1, 1, -1, -1);
  auto agg1 = CreateTensorAggregator(intrinsic).value();
  auto agg2 = CreateTensorAggregator(intrinsic).value();

  // agg1 gets one person's data, mapping "red apple" to 1,1
  Tensor red =
      Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"red"}))
          .value();
  Tensor apple =
      Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"apple"}))
          .value();
  Tensor data1 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
  Tensor data2 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
  EXPECT_THAT(agg1->Accumulate({&red, &apple, &data1, &data2}), IsOk());

  // agg2 gets data from a lot more people. At the end, it will map "red apple"
  // to 1000,1000 and "white grape" to 1000,1000.
  for (int i = 0; i < 1000; i++) {
    EXPECT_THAT(agg2->Accumulate({&red, &apple, &data1, &data2}), IsOk());

    Tensor white =
        Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"white"}))
            .value();
    Tensor grape =
        Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"grape"}))
            .value();
    EXPECT_THAT(agg2->Accumulate({&white, &grape, &data1, &data2}), IsOk());
  }
  // Merge the aggregators. The result should contain two different keys.
  // "red apple" should map to 1001, 1001 while "white grape" should map to
  // 1000, 1000.
  auto merge_status = agg1->MergeWith(std::move(*agg2));
  EXPECT_THAT(merge_status, IsOk());
  auto result = std::move(*agg1).Report();
  ASSERT_THAT(result, IsOk());
  ASSERT_THAT(result.value()[0].num_elements(), Eq(2));
  EXPECT_THAT(result.value()[0], IsTensor<string_view>({2}, {"red", "white"}));
  EXPECT_THAT(result.value()[1],
              IsTensor<string_view>({2}, {"apple", "grape"}));
  EXPECT_THAT(result.value()[2], IsTensor<int64_t>({2}, {1001, 1000}));
}

TEST(DPGroupByAggregatorTest, MergeDoesNotDistortData_NoKeys) {
  Intrinsic intrinsic{"fedsql_dp_group_by",
                      {},
                      {},
                      {CreateTopLevelParameters(100, 0.01, 100)},
                      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<int64_t>(10, 9, 8));
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<int64_t>(100, 9, -1));
  auto agg1 = CreateTensorAggregator(intrinsic).value();
  auto agg2 = CreateTensorAggregator(intrinsic).value();
  Tensor data1 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
  Tensor data2 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
  EXPECT_OK(agg1->Accumulate({&data1, &data2}));
  // Aggregate should be 1, 1

  for (int i = 0; i < 1000; i++) {
    Tensor data3 =
        Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({10})).value();
    Tensor data4 =
        Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({10})).value();
    EXPECT_OK(agg2->Accumulate({&data3, &data4}));
  }
  // Aggregate should be 8000, 9000 due to contribution bounding.

  auto merge_status = agg1->MergeWith(std::move(*agg2));
  EXPECT_OK(merge_status);
  auto result = std::move(*agg1).Report();
  ASSERT_OK(result);
  ASSERT_EQ(result.value().size(), 2);
  EXPECT_THAT(result.value()[0], IsTensor<int64_t>({1}, {8001}));
  EXPECT_THAT(result.value()[1], IsTensor<int64_t>({1}, {9001}));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
