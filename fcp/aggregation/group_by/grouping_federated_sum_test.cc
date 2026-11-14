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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/aggregation/group_by/one_dim_grouping_aggregator.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsTrue;
using ::testing::TestWithParam;

using GroupingFederatedSumTest = TestWithParam<bool>;

Intrinsic GetDefaultIntrinsic() {
  // One "GoogleSQL:sum" intrinsic with a single int32 tensor of unknown size.
  return Intrinsic{"GoogleSQL:sum",
                   {TensorSpec{"foo", DT_INT32, {-1}}},
                   {TensorSpec{"foo_out", DT_INT64, {-1}}},
                   {},
                   {}};
}

TEST_P(GroupingFederatedSumTest, ScalarAggregationSucceeds) {
  auto aggregator = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator->Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator->Accumulate({&ordinal, &t2}), IsOk());

  if (GetParam()) {
    auto factory = dynamic_cast<const OneDimBaseGroupingAggregatorFactory*>(
        GetAggregatorFactory(GetDefaultIntrinsic().uri).value());
    auto one_dim_base_aggregator =
        std::unique_ptr<OneDimBaseGroupingAggregator>(
            dynamic_cast<OneDimBaseGroupingAggregator*>(aggregator.release()));
    auto state = std::move(*(one_dim_base_aggregator)).ToProto();
    aggregator = factory->FromProto(GetDefaultIntrinsic(), state).value();
  }

  EXPECT_THAT(aggregator->Accumulate({&ordinal, &t3}), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());

  EXPECT_THAT(std::move(*aggregator).Report(),
              IsOkAndHolds(ElementsAre(IsTensor<int64_t>({1}, {6}))));
}

TEST_P(GroupingFederatedSumTest, DenseAggregationSucceeds) {
  auto aggregator = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor ordinals(std::vector<int64_t>{0, 1, 2, 3});
  Tensor t1({1, 3, 15, 27});
  Tensor t2({10, 5, 1, 2});
  Tensor t3({3, 11, 7, 20});
  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t1}), IsOk());
  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t2}), IsOk());

  if (GetParam()) {
    auto factory = dynamic_cast<const OneDimBaseGroupingAggregatorFactory*>(
        GetAggregatorFactory(GetDefaultIntrinsic().uri).value());
    auto one_dim_base_aggregator =
        std::unique_ptr<OneDimBaseGroupingAggregator>(
            dynamic_cast<OneDimBaseGroupingAggregator*>(aggregator.release()));
    auto state = std::move(*(one_dim_base_aggregator)).ToProto();
    aggregator = factory->FromProto(GetDefaultIntrinsic(), state).value();
  }

  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t3}), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(3));

  EXPECT_THAT(
      std::move(*aggregator).Report(),
      IsOkAndHolds(ElementsAre(IsTensor<int64_t>({4}, {14, 19, 23, 49}))));
}

TEST_P(GroupingFederatedSumTest, DenseAggregationCastToLargerTypeSucceeds) {
  Intrinsic intrinsic{"GoogleSQL:sum",
                      {TensorSpec{"foo", DT_INT32, {-1}}},
                      {TensorSpec{"foo_out", DT_INT64, {-1}}},
                      {},
                      {}};
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor ordinals(std::vector<int64_t>{0, 1, 2, 3});
  Tensor t1({1, 3, 15, 27});
  Tensor t2({10, 5, 1, 2});
  Tensor t3({3, 11, 7, 20});
  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t1}), IsOk());
  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t2}), IsOk());

  if (GetParam()) {
    auto factory = dynamic_cast<const OneDimBaseGroupingAggregatorFactory*>(
        GetAggregatorFactory(GetDefaultIntrinsic().uri).value());
    auto one_dim_base_aggregator =
        std::unique_ptr<OneDimBaseGroupingAggregator>(
            dynamic_cast<OneDimBaseGroupingAggregator*>(aggregator.release()));
    auto state = std::move(*(one_dim_base_aggregator)).ToProto();
    aggregator = factory->FromProto(intrinsic, state).value();
  }

  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t3}), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(3));

  EXPECT_THAT(
      std::move(*aggregator).Report(),
      IsOkAndHolds(ElementsAre(IsTensor<int64_t>({4}, {14, 19, 23, 49}))));
}

TEST_P(GroupingFederatedSumTest,
       DenseAggregationCastToLargerFloatTypeSucceeds) {
  Intrinsic intrinsic{"GoogleSQL:sum",
                      {TensorSpec{"foo", DT_FLOAT, {-1}}},
                      {TensorSpec{"foo_out", DT_DOUBLE, {-1}}},
                      {},
                      {}};
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  Tensor ordinals(std::vector<int64_t>{0, 1, 2, 3});
  Tensor t1({1.f, 3.f, 15.f, 27.f});
  Tensor t2({10.f, 5.f, 1.f, 2.f});
  Tensor t3({3.f, 11.f, 7.f, 20.f});
  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t1}), IsOk());
  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t2}), IsOk());

  if (GetParam()) {
    auto factory = dynamic_cast<const OneDimBaseGroupingAggregatorFactory*>(
        GetAggregatorFactory(GetDefaultIntrinsic().uri).value());
    auto one_dim_base_aggregator =
        std::unique_ptr<OneDimBaseGroupingAggregator>(
            dynamic_cast<OneDimBaseGroupingAggregator*>(aggregator.release()));
    auto state = std::move(*(one_dim_base_aggregator)).ToProto();
    aggregator = factory->FromProto(intrinsic, state).value();
  }

  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t3}), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(3));

  EXPECT_THAT(
      std::move(*aggregator).Report(),
      IsOkAndHolds(ElementsAre(IsTensor<double>({4}, {14, 19, 23, 49}))));
}

TEST_P(GroupingFederatedSumTest, MergeSucceeds) {
  auto aggregator1 = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  auto aggregator2 = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor ordinal(std::vector<int64_t>{0});
  Tensor t1({1});
  Tensor t2({2});
  Tensor t3({3});
  EXPECT_THAT(aggregator1->Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator2->Accumulate({&ordinal, &t2}), IsOk());
  EXPECT_THAT(aggregator2->Accumulate({&ordinal, &t3}), IsOk());

  if (GetParam()) {
    auto factory = dynamic_cast<const OneDimBaseGroupingAggregatorFactory*>(
        GetAggregatorFactory(GetDefaultIntrinsic().uri).value());
    auto one_dim_base_aggregator1 =
        std::unique_ptr<OneDimBaseGroupingAggregator>(
            dynamic_cast<OneDimBaseGroupingAggregator*>(aggregator1.release()));
    auto state = std::move(*(one_dim_base_aggregator1)).ToProto();
    aggregator1 = factory->FromProto(GetDefaultIntrinsic(), state).value();
    auto one_dim_base_aggregator2 =
        std::unique_ptr<OneDimBaseGroupingAggregator>(
            dynamic_cast<OneDimBaseGroupingAggregator*>(aggregator2.release()));
    auto state2 = std::move(*(one_dim_base_aggregator2)).ToProto();
    aggregator2 = factory->FromProto(GetDefaultIntrinsic(), state2).value();
  }

  int aggregator2_num_inputs = aggregator2->GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(*aggregator2).Report().value()[0]);
  Tensor ordinals_for_merge(std::vector<int64_t>{0});
  EXPECT_THAT((dynamic_cast<OneDimGroupingAggregator<int32_t, int64_t>*>(
                   aggregator1.get()))
                  ->MergeTensors({&ordinals_for_merge, &aggregator2_result},
                                 aggregator2_num_inputs),
              IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  EXPECT_THAT(std::move(*aggregator1).Report(),
              IsOkAndHolds(ElementsAre(IsTensor<int64_t>({1}, {6}))));
}

TEST_P(GroupingFederatedSumTest, MergeSucceedsWithNonSharedOrdinals) {
  auto aggregator1 = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  auto aggregator2 = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor ordinal(std::vector<int64_t>{0});
  Tensor t1({1});
  Tensor t2({2});
  Tensor t3({3});
  EXPECT_THAT(aggregator1->Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator2->Accumulate({&ordinal, &t2}), IsOk());
  EXPECT_THAT(aggregator2->Accumulate({&ordinal, &t3}), IsOk());

  if (GetParam()) {
    auto factory = dynamic_cast<const OneDimBaseGroupingAggregatorFactory*>(
        GetAggregatorFactory(GetDefaultIntrinsic().uri).value());
    auto one_dim_base_aggregator1 =
        std::unique_ptr<OneDimBaseGroupingAggregator>(
            dynamic_cast<OneDimBaseGroupingAggregator*>(aggregator1.release()));
    auto state = std::move(*(one_dim_base_aggregator1)).ToProto();
    aggregator1 = factory->FromProto(GetDefaultIntrinsic(), state).value();
    auto one_dim_base_aggregator2 =
        std::unique_ptr<OneDimBaseGroupingAggregator>(
            dynamic_cast<OneDimBaseGroupingAggregator*>(aggregator2.release()));
    auto state2 = std::move(*(one_dim_base_aggregator2)).ToProto();
    aggregator2 = factory->FromProto(GetDefaultIntrinsic(), state2).value();
  }

  int aggregator2_num_inputs = aggregator2->GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(*aggregator2).Report().value()[0]);
  Tensor ordinals_for_merge(std::vector<int64_t>{2});
  EXPECT_THAT((dynamic_cast<OneDimGroupingAggregator<int32_t, int64_t>*>(
                   aggregator1.get()))
                  ->MergeTensors({&ordinals_for_merge, &aggregator2_result},
                                 aggregator2_num_inputs),
              IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  EXPECT_THAT(std::move(*aggregator1).Report(),
              IsOkAndHolds(ElementsAre(IsTensor<int64_t>({3}, {1, 0, 5}))));
}

TEST(GroupingFederatedSumTest, CreateWrongUri) {
  Intrinsic intrinsic = Intrinsic{"wrong_uri",
                                  {TensorSpec{"foo", DT_INT32, {}}},
                                  {TensorSpec{"foo_out", DT_INT32, {}}},
                                  {},
                                  {}};
  FCP_ASSERT_OK_AND_ASSIGN(auto factory, GetAggregatorFactory("GoogleSQL:sum"));
  EXPECT_THAT(factory->Create(intrinsic),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected intrinsic URI GoogleSQL:sum")));
}

TEST(GroupingFederatedSumTest, CreateUnsupportedNumberOfInputs) {
  Intrinsic intrinsic = Intrinsic{
      "GoogleSQL:sum",
      {TensorSpec{"foo", DT_INT32, {}}, TensorSpec{"bar", DT_INT32, {}}},
      {TensorSpec{"foo_out", DT_INT32, {}}},
      {},
      {}};
  EXPECT_THAT(CreateTensorAggregator(intrinsic),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Exactly one input is expected")));
}

TEST(GroupingFederatedSumTest, CreateUnsupportedEmptyIntrinsic) {
  FCP_ASSERT_OK_AND_ASSIGN(auto factory, GetAggregatorFactory("GoogleSQL:sum"));
  EXPECT_THAT(factory->Create(Intrinsic{"GoogleSQL:sum", {}, {}, {}, {}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Exactly one input is expected")));
}

TEST(GroupingFederatedSumTest, CreateUnsupportedNumberOfOutputs) {
  Intrinsic intrinsic{"GoogleSQL:sum",
                      {TensorSpec{"foo", DT_INT32, {}}},
                      {TensorSpec{"foo_out", DT_INT32, {}},
                       TensorSpec{"bar_out", DT_INT32, {}}},
                      {},
                      {}};
  EXPECT_THAT(CreateTensorAggregator(intrinsic),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Exactly one output tensor is expected")));
}

TEST(GroupingFederatedSumTest,
     CreateUnsupportedUnmatchingInputAndOutputDataType) {
  Intrinsic intrinsic{"GoogleSQL:sum",
                      {TensorSpec{"foo", DT_INT32, {}}},
                      {TensorSpec{"foo_out", DT_FLOAT, {}}},
                      {},
                      {}};

  EXPECT_THAT(CreateTensorAggregator(intrinsic),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Input and output tensors have mismatched "
                                 "dtypes: input tensor has dtype DT_INT32 and "
                                 "output tensor has dtype DT_FLOAT")));
}

TEST(GroupingFederatedSumTest, CreateUnsupportedUnmatchingInputAndOutputShape) {
  Intrinsic intrinsic = Intrinsic{"GoogleSQL:sum",
                                  {TensorSpec{"foo", DT_INT32, {1}}},
                                  {TensorSpec{"foo", DT_INT32, {2}}},
                                  {},
                                  {}};
  EXPECT_THAT(
      CreateTensorAggregator(intrinsic),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Input and output tensors have mismatched shapes")));
}

TEST(GroupingFederatedSumTest, CreateUnsupportedIntrinsicWithParameter) {
  Tensor tensor = Tensor::Create(DT_FLOAT, {2, 3},
                                 CreateTestData<float>({1, 2, 3, 4, 5, 6}))
                      .value();
  Intrinsic intrinsic{"GoogleSQL:sum",
                      {TensorSpec{"foo", DT_INT32, {8}}},
                      {TensorSpec{"foo_out", DT_INT32, {16}}},
                      {},
                      {}};
  intrinsic.parameters.push_back(std::move(tensor));
  EXPECT_THAT(CreateTensorAggregator(intrinsic),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No input parameters expected")));
}

TEST(GroupingFederatedSumTest, CreateUnsupportedNestedIntrinsic) {
  Intrinsic inner = Intrinsic{"GoogleSQL:sum",
                              {TensorSpec{"foo", DT_INT32, {8}}},
                              {TensorSpec{"foo_out", DT_INT32, {16}}},
                              {},
                              {}};
  Intrinsic intrinsic = Intrinsic{"GoogleSQL:sum",
                                  {TensorSpec{"bar", DT_INT32, {1}}},
                                  {TensorSpec{"bar_out", DT_INT32, {2}}},
                                  {},
                                  {}};
  intrinsic.nested_intrinsics.push_back(std::move(inner));
  EXPECT_THAT(CreateTensorAggregator(intrinsic),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Not expected to have inner aggregations")));
}

TEST(GroupingFederatedSumTest, CreateUnsupportedStringDataType) {
  Intrinsic intrinsic = Intrinsic{"GoogleSQL:sum",
                                  {TensorSpec{"foo", DT_STRING, {1}}},
                                  {TensorSpec{"foo_out", DT_STRING, {1}}},
                                  {},
                                  {}};
  EXPECT_THAT(CreateTensorAggregator(intrinsic),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("GroupingFederatedSum isn't supported for "
                                 "DT_STRING datatype")));
}

TEST(GroupingFederatedSumTest, Deserialize_Unimplemented) {
  EXPECT_THAT(DeserializeTensorAggregator(GetDefaultIntrinsic(), ""),
              StatusIs(absl::StatusCode::kUnimplemented));
}

INSTANTIATE_TEST_SUITE_P(
    GroupingFederatedSumTestInstantiation, GroupingFederatedSumTest,
    ::testing::ValuesIn<bool>({false, true}),
    [](const ::testing::TestParamInfo<GroupingFederatedSumTest::ParamType>&
           info) { return info.param ? "SaveIntermediateState" : "None"; });

}  // namespace
}  // namespace aggregation
}  // namespace fcp
