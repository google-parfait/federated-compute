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
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "fcp/aggregation/core/federated_state.pb.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/tensor_spec.h"
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
using ::testing::IsTrue;

Intrinsic GetDefaultIntrinsic() {
  // One "federated_sum" intrinsic with a single scalar int32 tensor.
  return Intrinsic{"federated_sum",
                   {TensorSpec{"foo", DT_INT32, {}}},
                   {TensorSpec{"foo_out", DT_INT32, {}}},
                   {},
                   {}};
}

TEST(FederatedSumTest, ScalarAggregation_Succeeds) {
  auto aggregator = CreateTensorAggregator(GetDefaultIntrinsic()).value();
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
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({}, {6}));
}

TEST(FederatedSumTest, DenseAggregation_Succeeds) {
  Intrinsic federated_sum_intrinsic{"federated_sum",
                                    {TensorSpec{"foo", DT_INT32, {4}}},
                                    {TensorSpec{"foo_out", DT_INT32, {4}}},
                                    {},
                                    {}};
  auto aggregator = CreateTensorAggregator(federated_sum_intrinsic).value();
  Tensor t1({1, 3, 15, 27});
  Tensor t2({10, 5, 1, 2});
  Tensor t3({3, 11, 7, 20});
  EXPECT_THAT(aggregator->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t3), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator).Report();
  EXPECT_THAT(result, IsOk());

  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({4}, {14, 19, 23, 49}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(FederatedSumTest, Merge_Succeeds) {
  auto aggregator1 = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  auto aggregator2 = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator1->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator2->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator2->Accumulate(t3), IsOk());

  EXPECT_THAT(aggregator1->MergeWith(std::move(*aggregator2)), IsOk());
  EXPECT_THAT(aggregator1->CanReport(), IsTrue());
  EXPECT_THAT(aggregator1->GetNumInputs(), Eq(3));

  auto result = std::move(*aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({}, {6}));
}

TEST(FederatedSumTest, SerializeDeserialize_Succeeds) {
  auto aggregator = CreateTensorAggregator(GetDefaultIntrinsic()).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());

  auto serialized_state = std::move(*aggregator).Serialize();
  auto deserialized_aggregator =
      DeserializeTensorAggregator(GetDefaultIntrinsic(),
                                  serialized_state.value())
          .value();

  EXPECT_THAT(deserialized_aggregator->Accumulate(t3), IsOk());
  EXPECT_THAT(deserialized_aggregator->GetNumInputs(), Eq(3));
  EXPECT_THAT(deserialized_aggregator->CanReport(), IsTrue());

  auto result = std::move(*deserialized_aggregator).Report();
  // Verify the resulting tensor.
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({}, {6}));
}

TEST(FederatedSumTest, Serialize_Succeeds) {
  const TensorShape shape = {4};
  Intrinsic federated_sum_intrinsic{"federated_sum",
                                    {TensorSpec{"foo", DT_INT32, shape}},
                                    {TensorSpec{"foo_out", DT_INT32, shape}},
                                    {},
                                    {}};
  FCP_ASSERT_OK_AND_ASSIGN(auto aggregator,
                           CreateTensorAggregator(federated_sum_intrinsic));
  Tensor t1({1, 3, 15, 27});
  Tensor t2({10, 5, 1, 2});
  Tensor t3({3, 11, 7, 20});
  EXPECT_THAT(aggregator->Accumulate(t1), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t2), IsOk());
  EXPECT_THAT(aggregator->Accumulate(t3), IsOk());
  EXPECT_THAT(aggregator->CanReport(), IsTrue());
  EXPECT_THAT(aggregator->GetNumInputs(), Eq(3));

  auto serialized_state = std::move(*aggregator).Serialize();
  ASSERT_THAT(serialized_state, IsOk());

  FederatedSumAggregatorState aggregator_state;
  ASSERT_TRUE(aggregator_state.ParseFromString(serialized_state.value()));
  EXPECT_THAT(aggregator_state.num_inputs(), Eq(3));
  const int32_t* vector_data =
      reinterpret_cast<const int32_t*>(aggregator_state.vector_data().data());
  std::vector<int32_t> data(
      vector_data,
      vector_data + aggregator_state.vector_data().size() / sizeof(int32_t));
  EXPECT_EQ(data, std::vector<int32_t>({14, 19, 23, 49}));
}

TEST(FederatedSumTest, Create_WrongUri) {
  Intrinsic intrinsic{"wrong_uri",
                      {TensorSpec{"foo", DT_INT32, {}}},
                      {TensorSpec{"foo_out", DT_INT32, {}}},
                      {},
                      {}};

  FCP_ASSERT_OK_AND_ASSIGN(auto factory, GetAggregatorFactory("federated_sum"));
  auto s = factory->Create(intrinsic).status();
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s.message(), HasSubstr("Expected intrinsic URI federated_sum"));
}

TEST(FederatedSumTest, Create_UnsupportedNumberOfInputs) {
  Intrinsic intrinsic{
      "federated_sum",
      {TensorSpec{"foo", DT_INT32, {}}, TensorSpec{"bar", DT_INT32, {}}},
      {TensorSpec{"foo_out", DT_INT32, {}}},
      {},
      {}};

  auto s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s.message(), HasSubstr("Exactly one input is expected"));
}

TEST(FederatedSumTest, Create_UnsupportedEmptyIntrinsic) {
  FCP_ASSERT_OK_AND_ASSIGN(auto factory, GetAggregatorFactory("federated_sum"));
  auto s = factory->Create(Intrinsic{"federated_sum", {}, {}, {}, {}}).status();
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s.message(), HasSubstr("Exactly one input is expected"));
}

TEST(FederatedSumTest, Create_UnsupportedNumberOfOutputs) {
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec{"foo", DT_INT32, {}}},
                      {TensorSpec{"foo_out", DT_INT32, {}},
                       TensorSpec{"bar_out", DT_INT32, {}}},
                      {},
                      {}};

  auto s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s.message(), HasSubstr("Exactly one output tensor is expected"));
}

TEST(FederatedSumTest, Create_UnsupportedUnmatchingInputAndOutputDataType) {
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec{"foo", DT_INT32, {}}},
                      {TensorSpec{"foo_out", DT_FLOAT, {}}},
                      {},
                      {}};

  auto s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s.message(),
              HasSubstr("Input and output tensors have mismatched specs"));
}

TEST(FederatedSumTest, Create_UnsupportedUnmatchingInputAndOutputShape) {
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec{"foo", DT_INT32, {1}}},
                      {TensorSpec{"foo", DT_INT32, {2}}},
                      {},
                      {}};

  auto s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s.message(),
              HasSubstr("Input and output tensors have mismatched specs"));
}

TEST(FederatedSumTest, Create_UnsupportedIntrinsicWithParameter) {
  Tensor tensor = Tensor::Create(DT_FLOAT, {2, 3},
                                 CreateTestData<float>({1, 2, 3, 4, 5, 6}))
                      .value();
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec{"foo", DT_INT32, {1}}},
                      {TensorSpec{"foo", DT_INT32, {2}}},
                      {},
                      {}};
  intrinsic.parameters.push_back(std::move(tensor));

  auto s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s.message(), HasSubstr("Expected no parameters"));
}

TEST(FederatedSumTest, Create_UnsupportedNestedIntrinsic) {
  Intrinsic inner{"federated_sum",
                  {TensorSpec{"foo", DT_INT32, {8}}},
                  {TensorSpec{"foo_out", DT_INT32, {16}}},
                  {},
                  {}};
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec{"bar", DT_INT32, {1}}},
                      {TensorSpec{"bar_out", DT_INT32, {2}}},
                      {},
                      {}};
  intrinsic.nested_intrinsics.push_back(std::move(inner));

  auto s = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s.message(), HasSubstr("Expected no nested intrinsics"));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
