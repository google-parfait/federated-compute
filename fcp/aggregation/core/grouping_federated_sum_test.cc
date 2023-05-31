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
#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

using ::testing::Eq;
using testing::HasSubstr;
using ::testing::IsTrue;

Intrinsic GetDefaultIntrinsic() {
  // One "GoogleSQL:sum" intrinsic with a single int32 tensor of unknown size.
  return Intrinsic{"GoogleSQL:sum",
                   {TensorSpec{"foo", DT_INT32, {-1}}},
                   {TensorSpec{"foo_out", DT_INT32, {-1}}},
                   {},
                   {}};
}

TEST(GroupingFederatedSumTest, ScalarAggregation_Succeeds) {
  auto aggregator = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator->Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator->Accumulate({&ordinal, &t2}), IsOk());
  EXPECT_THAT(aggregator->Accumulate({&ordinal, &t3}), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({1}, {6}));
}

TEST(GroupingFederatedSumTest, DenseAggregation_Succeeds) {
  TensorShape shape{4};
  auto aggregator = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor ordinals =
      Tensor::Create(DT_INT64, shape, CreateTestData<int64_t>({0, 1, 2, 3}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t1}), IsOk());
  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t2}), IsOk());
  EXPECT_THAT(aggregator->Accumulate({&ordinals, &t3}), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor(shape, {14, 19, 23, 49}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(GroupingFederatedSumTest, Merge_Succeeds) {
  auto aggregator1 = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  auto aggregator2 = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator1->Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator2->Accumulate({&ordinal, &t2}), IsOk());
  EXPECT_THAT(aggregator2->Accumulate({&ordinal, &t3}), IsOk());

  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({1}, {6}));
}

TEST(GroupingFederatedSumTest, Create_WrongUri) {
  Intrinsic intrinsic = Intrinsic{"wrong_uri",
                                  {TensorSpec{"foo", DT_INT32, {}}},
                                  {TensorSpec{"foo_out", DT_INT32, {}}},
                                  {},
                                  {}};
  Status s =
      (*GetAggregatorFactory("GoogleSQL:sum"))->Create(intrinsic).status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Expected intrinsic URI GoogleSQL:sum"));
}

TEST(GroupingFederatedSumTest, Create_UnsupportedNumberOfInputs) {
  Intrinsic intrinsic = Intrinsic{
      "GoogleSQL:sum",
      {TensorSpec{"foo", DT_INT32, {}}, TensorSpec{"bar", DT_INT32, {}}},
      {TensorSpec{"foo_out", DT_INT32, {}}},
      {},
      {}};
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Exactly one input is expected"));
}

TEST(GroupingFederatedSumTest, Create_UnsupportedEmptyIntrinsic) {
  Status s = (*GetAggregatorFactory("GoogleSQL:sum"))
                 ->Create(Intrinsic{"GoogleSQL:sum", {}, {}, {}, {}})
                 .status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Exactly one input is expected"));
}

TEST(GroupingFederatedSumTest, Create_UnsupportedNumberOfOutputs) {
  Intrinsic intrinsic{"GoogleSQL:sum",
                      {TensorSpec{"foo", DT_INT32, {}}},
                      {TensorSpec{"foo_out", DT_INT32, {}},
                       TensorSpec{"bar_out", DT_INT32, {}}},
                      {},
                      {}};
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Exactly one output tensor is expected"));
}

TEST(GroupingFederatedSumTest,
     Create_UnsupportedUnmatchingInputAndOutputDataType) {
  Intrinsic intrinsic{"GoogleSQL:sum",
                      {TensorSpec{"foo", DT_INT32, {}}},
                      {TensorSpec{"foo_out", DT_FLOAT, {}}},
                      {},
                      {}};

  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("Input and output tensors have mismatched specs"));
}

TEST(GroupingFederatedSumTest,
     Create_UnsupportedUnmatchingInputAndOutputShape) {
  Intrinsic intrinsic = Intrinsic{"GoogleSQL:sum",
                                  {TensorSpec{"foo", DT_INT32, {1}}},
                                  {TensorSpec{"foo", DT_INT32, {2}}},
                                  {},
                                  {}};
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
}

TEST(GroupingFederatedSumTest, Create_UnsupportedIntrinsicWithParameter) {
  Tensor tensor = Tensor::Create(DT_FLOAT, {2, 3},
                                 CreateTestData<float>({1, 2, 3, 4, 5, 6}))
                      .value();
  Intrinsic intrinsic{"GoogleSQL:sum",
                      {TensorSpec{"foo", DT_INT32, {8}}},
                      {TensorSpec{"foo_out", DT_INT32, {16}}},
                      {},
                      {}};
  intrinsic.parameters.push_back(std::move(tensor));
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("No input parameters expected"));
}

TEST(GroupingFederatedSumTest, Create_UnsupportedNestedIntrinsic) {
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
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("Not expected to have inner aggregations"));
}

TEST(GroupingFederatedSumTest, Create_UnsupportedStringDataType) {
  Intrinsic intrinsic = Intrinsic{"GoogleSQL:sum",
                                  {TensorSpec{"foo", DT_STRING, {1}}},
                                  {TensorSpec{"foo_out", DT_STRING, {1}}},
                                  {},
                                  {}};
  Status s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      HasSubstr("GroupingFederatedSum isn't supported for DT_STRING datatype"));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
