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

#include "fcp/aggregation/checkpoint/checkpoint_aggregator.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/agg_vector_aggregator.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/federated_state.pb.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/aggregation/testing/mocks.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/base/scheduler.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::ByMove;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::StrEq;
using ::testing::TestWithParam;

using CheckpointAggregatorTest = TestWithParam<bool>;

// Creates a single intrinsic with the given URI, inputs and outputs and no
// parameters or nested intrinsics.
Intrinsic MakeIntrinsic(std::string uri, std::vector<TensorSpec> inputs,
                        std::vector<TensorSpec> outputs) {
  return Intrinsic{
      std::move(uri), std::move(inputs), std::move(outputs), {}, {}};
}

std::vector<Intrinsic> SingleIntrinsic(Intrinsic intrinsic) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(std::move(intrinsic));
  return intrinsics;
}

std::vector<Intrinsic> DefaultIntrinsics() {
  // One "federated_sum" intrinsic with a single scalar int32 tensor.
  return SingleIntrinsic(MakeIntrinsic("federated_sum",
                                       {TensorSpec("foo", DT_INT32, {})},
                                       {TensorSpec("foo_out", DT_INT32, {})}));
}

std::vector<Intrinsic> DefaultFedSqlIntrinsics() {
  Intrinsic group_by =
      MakeIntrinsic("fedsql_group_by", {TensorSpec("key1", DT_FLOAT, {-1})},
                    {TensorSpec("key1_out", DT_FLOAT, {-1})});
  group_by.nested_intrinsics.push_back(
      MakeIntrinsic("GoogleSQL:sum", {TensorSpec("val1", DT_FLOAT, {-1})},
                    {TensorSpec("val1_out", DT_FLOAT, {-1})}));
  return SingleIntrinsic(std::move(group_by));
}

// Creates a CheckpointAggregator which owns the intrinsics.
std::unique_ptr<CheckpointAggregator> Create(
    std::vector<Intrinsic> intrinsics) {
  auto result = CheckpointAggregator::Create(std::move(intrinsics));
  ABSL_EXPECT_OK(result.status());
  return std::move(result.value());
}

std::unique_ptr<CheckpointAggregator> CreateWithDefaultIntrinsics() {
  return Create(DefaultIntrinsics());
}

std::unique_ptr<CheckpointAggregator> CreateWithDefaultFedSqlIntrinsics() {
  return Create(DefaultFedSqlIntrinsics());
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsSuccess) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {})},
                        {TensorSpec("foo_out", DT_INT32, {})},
                        {},
                        {}});
  ABSL_EXPECT_OK(CheckpointAggregator::Create(&intrinsics));
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsUnsupportedNumberOfInputs) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(
      {"federated_sum",
       {TensorSpec("foo", DT_INT32, {}), TensorSpec("bar", DT_INT32, {})},
       {TensorSpec("foo_out", DT_INT32, {})},
       {},
       {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsUnsupportedNumberOfOutputs) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {})},
                        {TensorSpec("foo_out", DT_INT32, {}),
                         TensorSpec("bar_out", DT_INT32, {})},
                        {},
                        {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsUnsupportedInputType) {
  std::vector<Intrinsic> intrinsics;
  Tensor parameter({42.f});
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec("foo", DT_INT32, {})},
                      {TensorSpec("foo_out", DT_INT32, {})},
                      {},
                      {}};
  intrinsic.parameters.push_back(std::move(parameter));
  intrinsics.push_back(std::move(intrinsic));
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsUnsupportedIntrinsicUri) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"unsupported_xyz",
                        {TensorSpec("foo", DT_INT32, {})},
                        {TensorSpec("foo_out", DT_INT32, {})},
                        {},
                        {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsUnsupportedInputSpec) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {-1})},
                        {TensorSpec("foo_out", DT_INT32, {})},
                        {},
                        {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest,
     CreateFromIntrinsicsMismatchingInputAndOutputDataType) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {})},
                        {TensorSpec("foo_out", DT_FLOAT, {})},
                        {},
                        {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest,
     CreateFromIntrinsicsMismatchingInputAndOutputShape) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {1})},
                        {TensorSpec("foo_out", DT_INT32, {2})},
                        {},
                        {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest, CreateOwnedIntrinsicsSuccess) {
  ABSL_EXPECT_OK(CheckpointAggregator::Create(DefaultIntrinsics()));
}

TEST(CheckpointAggregatorTest, CreateOwnedIntrinsicsUnsupportedNumberOfInputs) {
  EXPECT_THAT(
      CheckpointAggregator::Create(SingleIntrinsic(MakeIntrinsic(
          "federated_sum",
          {TensorSpec("foo", DT_INT32, {}), TensorSpec("bar", DT_INT32, {})},
          {TensorSpec("foo_out", DT_INT32, {})}))),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest,
     CreateOwnedIntrinsicsUnsupportedNumberOfOutputs) {
  EXPECT_THAT(CheckpointAggregator::Create(SingleIntrinsic(MakeIntrinsic(
                  "federated_sum", {TensorSpec("foo", DT_INT32, {})},
                  {TensorSpec("foo_out", DT_INT32, {}),
                   TensorSpec("bar_out", DT_INT32, {})}))),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest, CreateOwnedIntrinsicsUnsupportedInputType) {
  // A parameter is provided in place of the input tensor.
  Intrinsic intrinsic =
      MakeIntrinsic("federated_sum", {}, {TensorSpec("foo_out", DT_INT32, {})});
  intrinsic.parameters.push_back(Tensor({42.f}));
  EXPECT_THAT(
      CheckpointAggregator::Create(SingleIntrinsic(std::move(intrinsic))),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest, CreateOwnedIntrinsicsUnsupportedIntrinsicUri) {
  EXPECT_THAT(CheckpointAggregator::Create(SingleIntrinsic(MakeIntrinsic(
                  "unsupported_xyz", {TensorSpec("foo", DT_INT32, {})},
                  {TensorSpec("foo_out", DT_INT32, {})}))),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(CheckpointAggregatorTest, CreateOwnedIntrinsicsUnsupportedInputSpec) {
  EXPECT_THAT(CheckpointAggregator::Create(SingleIntrinsic(MakeIntrinsic(
                  "federated_sum", {TensorSpec("foo", DT_INT32, {-1})},
                  {TensorSpec("foo_out", DT_INT32, {})}))),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest,
     CreateOwnedIntrinsicsMismatchingInputAndOutputDataType) {
  EXPECT_THAT(CheckpointAggregator::Create(SingleIntrinsic(MakeIntrinsic(
                  "federated_sum", {TensorSpec("foo", DT_INT32, {})},
                  {TensorSpec("foo_out", DT_FLOAT, {})}))),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest,
     CreateOwnedIntrinsicsMismatchingInputAndOutputShape) {
  EXPECT_THAT(CheckpointAggregator::Create(SingleIntrinsic(MakeIntrinsic(
                  "federated_sum", {TensorSpec("foo", DT_INT32, {1})},
                  {TensorSpec("foo_out", DT_INT32, {2})}))),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(CheckpointAggregatorTest, CreateFromIntrinsicsAccumulateSuccess) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {})},
                        {TensorSpec("foo_out", DT_INT32, {})},
                        {},
                        {}});
  auto aggregator = CheckpointAggregator::Create(&intrinsics).value();

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator =
        CheckpointAggregator::Deserialize(&intrinsics, serialized_state)
            .value();
  }
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({2}));
  });
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));
}

TEST(CheckpointAggregatorTest, AccumulateMissingTensor) {
  auto aggregator = CreateWithDefaultIntrinsics();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo")))
      .WillOnce(Return(ByMove(absl::NotFoundError("Missing tensor foo"))));
  EXPECT_THAT(aggregator->Accumulate(parser),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(CheckpointAggregatorTest, AccumulateMismatchingTensor) {
  auto aggregator = CreateWithDefaultIntrinsics();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce([] {
    return Tensor::Create(DT_FLOAT, {}, CreateTestData({2.f}));
  });
  EXPECT_THAT(aggregator->Accumulate(parser),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest, AccumulateSuccess) {
  auto aggregator = CreateWithDefaultIntrinsics();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({2}));
  });
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));
}

TEST(CheckpointAggregatorTest, AccumulateAfterReport) {
  auto aggregator = CreateWithDefaultIntrinsics();

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {0})))
      .WillOnce(Return(absl::OkStatus()));
  ABSL_EXPECT_OK(aggregator->Report(builder));

  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({2}));
  });
  EXPECT_THAT(aggregator->Accumulate(parser),
              StatusIs(absl::StatusCode::kAborted));
}

TEST(CheckpointAggregatorTest, AccumulateAfterAbort) {
  auto aggregator = CreateWithDefaultIntrinsics();
  aggregator->Abort();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({2}));
  });
  EXPECT_THAT(aggregator->Accumulate(parser),
              StatusIs(absl::StatusCode::kAborted));
}

TEST_P(CheckpointAggregatorTest, ReportZeroInputs) {
  auto aggregator = CreateWithDefaultIntrinsics();

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    FCP_ASSERT_OK_AND_ASSIGN(
        aggregator, CheckpointAggregator::Deserialize(DefaultIntrinsics(),
                                                      serialized_state));
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {0})))
      .WillOnce(Return(absl::OkStatus()));
  absl::StatusOr<int> num_checkpoints_aggregated =
      aggregator->GetNumCheckpointsAggregated();
  ABSL_EXPECT_OK(num_checkpoints_aggregated);
  EXPECT_EQ(*num_checkpoints_aggregated, 0);
  ABSL_EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportOneInput) {
  auto aggregator = CreateWithDefaultIntrinsics();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({2}));
  });
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    FCP_ASSERT_OK_AND_ASSIGN(
        aggregator, CheckpointAggregator::Deserialize(DefaultIntrinsics(),
                                                      serialized_state));
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {2})))
      .WillOnce(Return(absl::OkStatus()));
  absl::StatusOr<int> num_checkpoints_aggregated =
      aggregator->GetNumCheckpointsAggregated();
  ABSL_EXPECT_OK(num_checkpoints_aggregated);
  EXPECT_EQ(*num_checkpoints_aggregated, 1);
  ABSL_EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportTwoInputs) {
  auto aggregator = CreateWithDefaultIntrinsics();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo")))
      .WillOnce(Invoke(
          [] { return Tensor::Create(DT_INT32, {}, CreateTestData({2})); }))
      .WillOnce(Invoke(
          [] { return Tensor::Create(DT_INT32, {}, CreateTestData({3})); }));
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    FCP_ASSERT_OK_AND_ASSIGN(
        aggregator, CheckpointAggregator::Deserialize(DefaultIntrinsics(),
                                                      serialized_state));
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {5})))
      .WillOnce(Return(absl::OkStatus()));
  absl::StatusOr<int> num_checkpoints_aggregated =
      aggregator->GetNumCheckpointsAggregated();
  ABSL_EXPECT_OK(num_checkpoints_aggregated);
  EXPECT_EQ(*num_checkpoints_aggregated, 2);
  ABSL_EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportMultipleTensors) {
  // Exercises the unowned intrinsics overloads of Create and Deserialize.
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(MakeIntrinsic("federated_sum",
                                     {TensorSpec("foo", DT_INT32, {3})},
                                     {TensorSpec("foo_out", DT_INT32, {3})}));
  intrinsics.push_back(
      MakeIntrinsic("federated_sum", {TensorSpec("bar", DT_FLOAT, {2, 2})},
                    {TensorSpec("bar_out", DT_FLOAT, {2, 2})}));
  FCP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CheckpointAggregator> aggregator,
                           CheckpointAggregator::Create(&intrinsics));

  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce([] {
    return Tensor::Create(DT_INT32, {3}, CreateTestData({1, 2, 3}));
  });
  EXPECT_CALL(parser, GetTensor(StrEq("bar"))).WillOnce([] {
    return Tensor::Create(DT_FLOAT, {2, 2},
                          CreateTestData({1.f, 2.f, 3.f, 4.f}));
  });
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator =
        CheckpointAggregator::Deserialize(&intrinsics, serialized_state)
            .value();
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({3}, {1, 2, 3})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("bar_out"),
                           IsTensor<float>({2, 2}, {1.f, 2.f, 3.f, 4.f})))
      .WillOnce(Return(absl::OkStatus()));
  absl::StatusOr<int> num_checkpoints_aggregated =
      aggregator->GetNumCheckpointsAggregated();
  ABSL_EXPECT_OK(num_checkpoints_aggregated);
  EXPECT_EQ(*num_checkpoints_aggregated, 1);
  ABSL_EXPECT_OK(aggregator->Report(builder));
}

TEST(CheckpointAggregatorTest, ReportAfterReport) {
  auto aggregator = CreateWithDefaultIntrinsics();

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {0})))
      .WillOnce(Return(absl::OkStatus()));
  ABSL_EXPECT_OK(aggregator->Report(builder));
  EXPECT_THAT(aggregator->Report(builder),
              StatusIs(absl::StatusCode::kAborted));
}

TEST(CheckpointAggregatorTest, ReportAfterAbort) {
  auto aggregator = CreateWithDefaultIntrinsics();
  aggregator->Abort();
  MockCheckpointBuilder builder;
  EXPECT_THAT(aggregator->Report(builder),
              StatusIs(absl::StatusCode::kAborted));
}

TEST(CheckpointAggregatorTest, GetNumCheckpointsAggregatedAfterReport) {
  auto aggregator = CreateWithDefaultIntrinsics();

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {0})))
      .WillOnce(Return(absl::OkStatus()));
  ABSL_EXPECT_OK(aggregator->Report(builder));
  EXPECT_THAT(aggregator->GetNumCheckpointsAggregated(),
              StatusIs(absl::StatusCode::kAborted));
}

TEST(CheckpointAggregatorTest, GetNumCheckpointsAggregatedAfterAbort) {
  auto aggregator = CreateWithDefaultIntrinsics();
  aggregator->Abort();
  EXPECT_THAT(aggregator->GetNumCheckpointsAggregated(),
              StatusIs(absl::StatusCode::kAborted));
}

// A fake aggregator that can never report.
class CannotReportAggregator : public AggVectorAggregator<int> {
 public:
  using AggVectorAggregator<int>::AggVectorAggregator;

  absl::StatusOr<std::string> Serialize() && override {
    return absl::UnimplementedError("Unimplemented");
  }

 private:
  void AggregateVector(const AggVector<int>& agg_vector) override {}
  bool CanReport() const override { return false; }
};

class CannotReportAggregatorFactory : public TensorAggregatorFactory {
 private:
  absl::StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    if (intrinsic.inputs[0].dtype() != DT_INT32) {
      return absl::InvalidArgumentError("Unsupported dtype: expected DT_INT32");
    }
    return std::make_unique<CannotReportAggregator>(
        intrinsic.inputs[0].dtype(), intrinsic.inputs[0].shape());
  }

  absl::StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    return absl::UnimplementedError("Unimplemented");
  }
};

TEST(CheckpointAggregatorTest, ReportWithFailedCanReportPrecondition) {
  CannotReportAggregatorFactory agg_factory;
  RegisterAggregatorFactory("foo1_aggregation", &agg_factory);
  // Intrinsics that use the CannotReportAggregatorFactory above.
  // This aggregation is supposed to always fail the CanReport call.
  auto aggregator = Create(SingleIntrinsic(
      MakeIntrinsic("foo1_aggregation", {TensorSpec("foo", DT_INT32, {})},
                    {TensorSpec("foo_out", DT_INT32, {})})));

  MockCheckpointBuilder builder;
  EXPECT_THAT(aggregator->Report(builder),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

// A fake aggregator that never increments the number of inputs aggregated.
class DoesNotIncrementNumInputsAggregator : public AggVectorAggregator<int> {
 public:
  using AggVectorAggregator<int>::AggVectorAggregator;

  absl::StatusOr<std::string> Serialize() && override {
    return absl::UnimplementedError("Unimplemented");
  }

 private:
  void AggregateVector(const AggVector<int>& agg_vector) override {}
  int GetNumInputs() const override { return 0; }
};

class DoesNotIncrementNumInputsAggregatorFactory
    : public TensorAggregatorFactory {
 private:
  absl::StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    if (intrinsic.inputs[0].dtype() != DT_INT32) {
      return absl::InvalidArgumentError("Unsupported dtype: expected DT_INT32");
    }
    return std::make_unique<DoesNotIncrementNumInputsAggregator>(
        intrinsic.inputs[0].dtype(), intrinsic.inputs[0].shape());
  }

  absl::StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    return absl::UnimplementedError("Unimplemented");
  }
};

TEST(CheckpointAggregatorTest, GetNumCheckpointsAggregatedFailsOnMismatch) {
  DoesNotIncrementNumInputsAggregatorFactory agg_factory;
  RegisterAggregatorFactory("does_not_increment_num_inputs_aggregation",
                            &agg_factory);
  // Intrinsics that use the DoesNotIncrementNumInputsAggregatorFactory
  // above. This aggregation always returns 0 for the number of inputs
  // aggregated, even after inputs have been accumulated, which will lead to a
  // mismatch with tensor aggregators that accurately count the number of inputs
  // aggregated.
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(
      MakeIntrinsic("does_not_increment_num_inputs_aggregation",
                    {TensorSpec("foo", DT_INT32, {3})},
                    {TensorSpec("foo_out", DT_INT32, {3})}));
  intrinsics.push_back(
      MakeIntrinsic("federated_sum", {TensorSpec("bar", DT_FLOAT, {2, 2})},
                    {TensorSpec("bar_out", DT_FLOAT, {2, 2})}));
  auto aggregator = Create(std::move(intrinsics));
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce([] {
    return Tensor::Create(DT_INT32, {3}, CreateTestData({1, 2, 3}));
  });
  EXPECT_CALL(parser, GetTensor(StrEq("bar"))).WillOnce([] {
    return Tensor::Create(DT_FLOAT, {2, 2},
                          CreateTestData({1.f, 2.f, 3.f, 4.f}));
  });
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));

  // The number of aggregated checkpoints is undefined because the inner
  // aggregators have aggregated different numbers of tensors.
  EXPECT_THAT(aggregator->GetNumCheckpointsAggregated(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_P(CheckpointAggregatorTest, ReportFedSqlZeroInputs) {
  // One intrinsic:
  //    fedsql_group_by with two grouping keys key1 and key2, only the first one
  //    of which should be output, and two inner GoogleSQL:sum intrinsics bar
  //    and baz operating on float tensors.
  // Exercises the unowned intrinsics overloads of Create and Deserialize.
  Intrinsic group_by = MakeIntrinsic("fedsql_group_by",
                                     {TensorSpec("key1", DT_STRING, {-1}),
                                      TensorSpec("key2", DT_STRING, {-1})},
                                     {TensorSpec("key1_out", DT_STRING, {-1}),
                                      TensorSpec("", DT_STRING, {-1})});
  group_by.nested_intrinsics.push_back(
      MakeIntrinsic("GoogleSQL:sum", {TensorSpec("bar", DT_FLOAT, {-1})},
                    {TensorSpec("bar_out", DT_FLOAT, {-1})}));
  group_by.nested_intrinsics.push_back(
      MakeIntrinsic("GoogleSQL:sum", {TensorSpec("baz", DT_FLOAT, {-1})},
                    {TensorSpec("baz_out", DT_FLOAT, {-1})}));
  std::vector<Intrinsic> intrinsics = SingleIntrinsic(std::move(group_by));
  FCP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CheckpointAggregator> aggregator,
                           CheckpointAggregator::Create(&intrinsics));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator =
        CheckpointAggregator::Deserialize(&intrinsics, serialized_state)
            .value();
  }

  MockCheckpointBuilder builder;
  // Verify that empty tensors are added to the result checkpoint.
  EXPECT_CALL(builder, Add(StrEq("key1_out"), IsTensor<string_view>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("bar_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("baz_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  absl::StatusOr<int> num_checkpoints_aggregated =
      aggregator->GetNumCheckpointsAggregated();
  ABSL_EXPECT_OK(num_checkpoints_aggregated);
  EXPECT_EQ(*num_checkpoints_aggregated, 0);
  ABSL_EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportFedSqlsOneInput) {
  auto aggregator = CreateWithDefaultFedSqlIntrinsics();

  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("key1"))).WillOnce([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({1.f, 2.f}));
  });
  EXPECT_CALL(parser, GetTensor(StrEq("val1"))).WillOnce([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({.1f, .2f}));
  });
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator = CheckpointAggregator::Deserialize(DefaultFedSqlIntrinsics(),
                                                   serialized_state)
                     .value();
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("key1_out"), IsTensor<float>({2}, {1.f, 2.f})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("val1_out"), IsTensor<float>({2}, {.1f, .2f})))
      .WillOnce(Return(absl::OkStatus()));
  absl::StatusOr<int> num_checkpoints_aggregated =
      aggregator->GetNumCheckpointsAggregated();
  ABSL_EXPECT_OK(num_checkpoints_aggregated);
  EXPECT_EQ(*num_checkpoints_aggregated, 1);
  ABSL_EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportFedSqlsTwoInputs) {
  auto aggregator = CreateWithDefaultFedSqlIntrinsics();

  MockCheckpointParser parser1;
  EXPECT_CALL(parser1, GetTensor(StrEq("key1"))).WillOnce([] {
    return Tensor::Create(DT_FLOAT, {3}, CreateTestData({1.f, 2.f, 3.f}));
  });
  EXPECT_CALL(parser1, GetTensor(StrEq("val1"))).WillOnce([] {
    return Tensor::Create(DT_FLOAT, {3}, CreateTestData({.1f, .2f, .3f}));
  });
  ABSL_EXPECT_OK(aggregator->Accumulate(parser1));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator = CheckpointAggregator::Deserialize(DefaultFedSqlIntrinsics(),
                                                   serialized_state)
                     .value();
  }

  MockCheckpointParser parser2;
  EXPECT_CALL(parser2, GetTensor(StrEq("key1"))).WillOnce([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({2.f, 4.f}));
  });
  EXPECT_CALL(parser2, GetTensor(StrEq("val1"))).WillOnce([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({.4f, .5f}));
  });
  ABSL_EXPECT_OK(aggregator->Accumulate(parser2));

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("key1_out"),
                           IsTensor<float>({4}, {1.f, 2.f, 3.f, 4.f})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("val1_out"),
                           IsTensor<float>({4}, {.1f, .6f, .3f, .5f})))
      .WillOnce(Return(absl::OkStatus()));
  absl::StatusOr<int> num_checkpoints_aggregated =
      aggregator->GetNumCheckpointsAggregated();
  ABSL_EXPECT_OK(num_checkpoints_aggregated);
  EXPECT_EQ(*num_checkpoints_aggregated, 2);
  ABSL_EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportFedSqlsEmptyInput) {
  auto aggregator = CreateWithDefaultFedSqlIntrinsics();

  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("key1"))).WillOnce([] {
    return Tensor::Create(DT_FLOAT, {0}, CreateTestData<float>({}));
  });
  EXPECT_CALL(parser, GetTensor(StrEq("val1"))).WillOnce([] {
    return Tensor::Create(DT_FLOAT, {0}, CreateTestData<float>({}));
  });
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator = CheckpointAggregator::Deserialize(DefaultFedSqlIntrinsics(),
                                                   serialized_state)
                     .value();
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("key1_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("val1_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));

  absl::StatusOr<int> num_checkpoints_aggregated =
      aggregator->GetNumCheckpointsAggregated();
  ABSL_EXPECT_OK(num_checkpoints_aggregated);
  EXPECT_EQ(*num_checkpoints_aggregated, 1);
  ABSL_EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, MergeSuccess) {
  auto aggregator1 = CreateWithDefaultIntrinsics();
  MockCheckpointParser parser1;
  EXPECT_CALL(parser1, GetTensor(StrEq("foo"))).WillOnce([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({5}));
  });
  ABSL_EXPECT_OK(aggregator1->Accumulate(parser1));

  auto aggregator2 = CreateWithDefaultIntrinsics();
  MockCheckpointParser parser2;
  EXPECT_CALL(parser2, GetTensor(StrEq("foo"))).WillOnce([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({7}));
  });
  ABSL_EXPECT_OK(aggregator2->Accumulate(parser2));

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize().value();
    aggregator1 = CheckpointAggregator::Deserialize(DefaultIntrinsics(),
                                                    serialized_state1)
                      .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize().value();
    aggregator2 = CheckpointAggregator::Deserialize(DefaultIntrinsics(),
                                                    serialized_state2)
                      .value();
  }

  ABSL_EXPECT_OK(aggregator1->MergeWith(std::move(*aggregator2)));

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize().value();
    aggregator1 = CheckpointAggregator::Deserialize(DefaultIntrinsics(),
                                                    serialized_state1)
                      .value();
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {12})))
      .WillOnce(Return(absl::OkStatus()));

  absl::StatusOr<int> num_checkpoints_aggregated =
      aggregator1->GetNumCheckpointsAggregated();
  ABSL_EXPECT_OK(num_checkpoints_aggregated);
  EXPECT_EQ(*num_checkpoints_aggregated, 2);
  ABSL_EXPECT_OK(aggregator1->Report(builder));
}

TEST(CheckpointAggregatorTest, MergeIncompatibleConfigs) {
  // TODO: b/316662605 - Add this test.
}

TEST(CheckpointAggregatorTest, ConcurrentAccumulationSuccess) {
  const int64_t kNumInputs = 10;
  auto aggregator = CreateWithDefaultIntrinsics();

  // The following block will repeatedly provide scalar int tensors with
  // incrementing values.
  std::atomic<int> tensor_value = 0;
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillRepeatedly([&] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({++tensor_value}));
  });

  // Schedule receiving inputs on 4 concurrent threads.
  auto scheduler = CreateThreadPoolScheduler(4);
  for (int64_t i = 0; i < kNumInputs; ++i) {
    scheduler->Schedule(
        [&]() { ABSL_EXPECT_OK(aggregator->Accumulate(parser)); });
  }
  scheduler->WaitUntilIdle();

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {55})))
      .WillOnce(Return(absl::OkStatus()));

  absl::StatusOr<int> num_checkpoints_aggregated =
      aggregator->GetNumCheckpointsAggregated();
  ABSL_EXPECT_OK(num_checkpoints_aggregated);
  EXPECT_EQ(*num_checkpoints_aggregated, kNumInputs);
  ABSL_EXPECT_OK(aggregator->Report(builder));
}

// A trivial test aggregator that delegates aggregation to a function.
class FunctionAggregator final : public AggVectorAggregator<int> {
 public:
  using Func = std::function<int(int, int)>;

  FunctionAggregator(DataType dtype, TensorShape shape, Func agg_function)
      : AggVectorAggregator<int>(dtype, shape), agg_function_(agg_function) {}

  absl::StatusOr<std::string> Serialize() && override {
    return absl::UnimplementedError("Unimplemented");
  }

 private:
  void AggregateVector(const AggVector<int>& agg_vector) override {
    for (auto [i, v] : agg_vector) {
      data()[i] = agg_function_(data()[i], v);
    }
  }

  const Func agg_function_;
};

// Factory for the FunctionAggregator.
class FunctionAggregatorFactory final : public TensorAggregatorFactory {
 public:
  explicit FunctionAggregatorFactory(FunctionAggregator::Func agg_function)
      : agg_function_(agg_function) {}

 private:
  absl::StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    if (intrinsic.inputs[0].dtype() != DT_INT32) {
      return absl::InvalidArgumentError("Unsupported dtype: expected DT_INT32");
    }
    return std::make_unique<FunctionAggregator>(intrinsic.inputs[0].dtype(),
                                                intrinsic.inputs[0].shape(),
                                                agg_function_);
  }

  absl::StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    return absl::UnimplementedError("Unimplemented");
  }

  const FunctionAggregator::Func agg_function_;
};

TEST(CheckpointAggregatorTest, ConcurrentAccumulationAbortWhileQueued) {
  const int64_t kNumInputs = 10;
  const int64_t kNumInputsBeforeBlocking = 3;

  // Notifies the aggregation to unblock;
  absl::Notification resume_aggregation_notification;
  absl::Notification aggregation_blocked_notification;
  std::atomic<int> agg_counter = 0;
  FunctionAggregatorFactory agg_factory([&](int a, int b) {
    if (++agg_counter > kNumInputsBeforeBlocking &&
        !aggregation_blocked_notification.HasBeenNotified()) {
      aggregation_blocked_notification.Notify();
      resume_aggregation_notification.WaitForNotification();
    }
    return a + b;
  });
  RegisterAggregatorFactory("foo2_aggregation", &agg_factory);

  auto aggregator = Create(SingleIntrinsic(
      MakeIntrinsic("foo2_aggregation", {TensorSpec("foo", DT_INT32, {})},
                    {TensorSpec("foo_out", DT_INT32, {})})));

  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillRepeatedly([&] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
  });

  // Schedule receiving inputs on 10 concurrent threads.
  auto scheduler = CreateThreadPoolScheduler(10);
  for (int64_t i = 0; i < kNumInputs; ++i) {
    scheduler->Schedule([&]() {
      EXPECT_THAT(aggregator->Accumulate(parser),
                  AnyOf(StatusIs(absl::StatusCode::kOk),
                        StatusIs(absl::StatusCode::kAborted)));
    });
  }

  aggregation_blocked_notification.WaitForNotification();
  aggregator->Abort();
  resume_aggregation_notification.Notify();
  scheduler->WaitUntilIdle();
}

TEST(CheckpointAggregatorTest, SerializeAfterReport) {
  auto aggregator = CreateWithDefaultIntrinsics();

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {0})))
      .WillOnce(Return(absl::OkStatus()));
  ABSL_EXPECT_OK(aggregator->Report(builder));
  EXPECT_THAT(std::move(*aggregator).Serialize(),
              StatusIs(absl::StatusCode::kAborted));
}

TEST(CheckpointAggregatorTest, SerializeAfterAbort) {
  auto aggregator = CreateWithDefaultIntrinsics();
  aggregator->Abort();
  EXPECT_THAT(std::move(*aggregator).Serialize(),
              StatusIs(absl::StatusCode::kAborted));
}

TEST(CheckpointAggregatorTest, DeserializeInvalidState) {
  std::string serialized_state = "invalid";
  EXPECT_THAT(
      CheckpointAggregator::Deserialize(DefaultIntrinsics(), serialized_state),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest, DeserializeFromIntrinsicsInvalidState) {
  std::vector<Intrinsic> intrinsics = DefaultIntrinsics();
  std::string serialized_state = "invalid";
  EXPECT_THAT(CheckpointAggregator::Deserialize(&intrinsics, serialized_state),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CheckpointAggregatorTest, DeserializeFromIntrinsicsMismatchedSize) {
  std::vector<Intrinsic> intrinsics = DefaultIntrinsics();
  intrinsics.pop_back();
  FCP_ASSERT_OK_AND_ASSIGN(auto aggregator,
                           CheckpointAggregator::Create(&intrinsics));
  FCP_ASSERT_OK_AND_ASSIGN(std::string serialized_state,
                           std::move(*aggregator).Serialize());
  std::vector<Intrinsic> default_intrinsics = DefaultIntrinsics();
  EXPECT_THAT(
      CheckpointAggregator::Deserialize(&default_intrinsics, serialized_state),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("The number of serialized aggregators doesn't match")));
}

TEST(CheckpointAggregatorTest, DeserializeOwnedIntrinsicsSuccess) {
  std::string serialized_state;
  {
    // The intrinsics are owned by the aggregator and go away with it.
    auto aggregator = CreateWithDefaultIntrinsics();
    MockCheckpointParser parser;
    EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce([] {
      return Tensor::Create(DT_INT32, {}, CreateTestData({3}));
    });
    ABSL_ASSERT_OK(aggregator->Accumulate(parser));
    FCP_ASSERT_OK_AND_ASSIGN(serialized_state,
                             std::move(*aggregator).Serialize());
  }

  FCP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CheckpointAggregator> aggregator,
      CheckpointAggregator::Deserialize(DefaultIntrinsics(),
                                        std::move(serialized_state)));
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({4}));
  });
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {7})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_THAT(aggregator->GetNumCheckpointsAggregated(), IsOkAndHolds(2));
  ABSL_EXPECT_OK(aggregator->Report(builder));
}

// A simple test aggregator that adds all values in the AggVector and puts the
// result into multiple partitions.
class TestPartitioningAggregator final : public AggVectorAggregator<int> {
 public:
  TestPartitioningAggregator(DataType dtype, TensorShape shape)
      : AggVectorAggregator<int>(dtype, shape) {}

  absl::StatusOr<std::string> Serialize() && override {
    FederatedSumAggregatorState aggregator_state;
    aggregator_state.set_num_inputs(GetNumInputs());
    *(aggregator_state.mutable_vector_data()) = vector_data().EncodeContent();
    return aggregator_state.SerializeAsString();
  }

  absl::StatusOr<std::vector<std::string>> Partition(int num_partitions) &&
      override {
    std::vector<std::string> partitions(num_partitions);
    auto serialized_state = std::move(*this).Serialize().value();
    for (int i = 0; i < num_partitions; ++i) {
      partitions[i] = serialized_state;
    }
    return partitions;
  }

 private:
  void AggregateVector(const AggVector<int>& agg_vector) override {
    for (auto [i, v] : agg_vector) {
      data()[i] = data()[i] + v;
    }
  }
};

// Factory for the TestPartitioningAggregator.
class TestPartitioningAggregatorFactory final : public TensorAggregatorFactory {
 public:
  TestPartitioningAggregatorFactory() = default;

 private:
  absl::StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    if (intrinsic.inputs[0].dtype() != DT_INT32) {
      return absl::InvalidArgumentError("Unsupported dtype: expected DT_INT32");
    }
    return std::make_unique<TestPartitioningAggregator>(
        intrinsic.inputs[0].dtype(), intrinsic.inputs[0].shape());
  }

  absl::StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    return absl::UnimplementedError("Unimplemented");
  }
};

TEST(CheckpointAggregatorTest, Partition) {
  RegisterAggregatorFactory("test_partitioning_aggregator",
                            new TestPartitioningAggregatorFactory());
  // Add 2 top-level aggregators.
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(MakeIntrinsic("test_partitioning_aggregator",
                                     {TensorSpec("foo", DT_INT32, {})},
                                     {TensorSpec("foo_out", DT_INT32, {})}));
  intrinsics.push_back(MakeIntrinsic("test_partitioning_aggregator",
                                     {TensorSpec("bar", DT_INT32, {})},
                                     {TensorSpec("bar_out", DT_INT32, {})}));
  auto aggregator = Create(std::move(intrinsics));
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillRepeatedly([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
  });
  EXPECT_CALL(parser, GetTensor(StrEq("bar"))).WillRepeatedly([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
  });
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));
  ABSL_EXPECT_OK(aggregator->Accumulate(parser));
  FCP_ASSERT_OK_AND_ASSIGN(std::vector<std::string> partitions,
                           std::move(*aggregator).Partition(3));

  EXPECT_EQ(partitions.size(), 3);
  TestPartitioningAggregator expected_aggregator(DT_INT32, {});
  FCP_ASSERT_OK_AND_ASSIGN(Tensor t,
                           Tensor::Create(DT_INT32, {}, CreateTestData({1})));
  ABSL_EXPECT_OK(expected_aggregator.Accumulate({&t}));
  ABSL_EXPECT_OK(expected_aggregator.Accumulate({&t}));
  ABSL_EXPECT_OK(expected_aggregator.Accumulate({&t}));
  std::string serialized_aggregator =
      std::move(expected_aggregator).Serialize().value();
  // All partitions should be equal.
  for (const auto& partition : partitions) {
    CheckpointAggregatorState state;
    EXPECT_TRUE(state.ParseFromString(partition));
    EXPECT_EQ(state.aggregators_size(), 2);
    EXPECT_EQ(state.aggregators(0), serialized_aggregator);
    EXPECT_EQ(state.aggregators(1), serialized_aggregator);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CheckpointAggregatorTestInstantiation, CheckpointAggregatorTest,
    ::testing::ValuesIn<bool>({false, true}),
    [](const ::testing::TestParamInfo<CheckpointAggregatorTest::ParamType>&
           info) { return info.param ? "SerializeDeserialize" : "None"; });

}  // namespace
}  // namespace aggregation
}  // namespace fcp
