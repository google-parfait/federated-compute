/*
 * Copyright 2020 Google LLC
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
#include "fcp/client/federated_task_environment.h"

#include <functional>
#include <memory>
#include <string>
#include <variant>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/base/platform.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace client {
namespace {

using ::google::internal::federatedml::v2::Checkpoint;
using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

static constexpr absl::Time kReferenceTime = absl::InfinitePast();

class FederatedTaskEnvironmentTest : public testing::Test {
 public:
  FederatedTaskEnvironmentTest() {}

 protected:
  void SetUp() override {
    task_env_ = std::make_unique<FederatedTaskEnvironment>(
        &mock_simple_task_environment_, &mock_federated_protocol_,
        &mock_log_manager_, &mock_event_publisher_, &mock_opstats_logger_,
        kReferenceTime, absl::ZeroDuration());
  }

  StrictMock<MockSimpleTaskEnvironment> mock_simple_task_environment_;
  StrictMock<MockLogManager> mock_log_manager_;
  StrictMock<MockEventPublisher> mock_event_publisher_;
  StrictMock<MockOpStatsLogger> mock_opstats_logger_;
  StrictMock<MockFederatedProtocol> mock_federated_protocol_;
  std::unique_ptr<TaskEnvironment> task_env_;
};

// Tests the case where Finish is called with PhaseOutcome::Interrupted. This
// must not lead to a call to the protocol or LogManager and must return OK.
TEST_F(FederatedTaskEnvironmentTest, TestFinishWhenInterrupted) {
  EXPECT_OK(task_env_->Finish(engine::PhaseOutcome::INTERRUPTED,
                              absl::InfiniteDuration(), {}));
}

// Matcher to test equality of two ComputationResults objects.
MATCHER_P(EqualsComputationResult, expected, "") {
  if (expected.get().size() != arg.size()) return false;
  for (const auto& [k, v] : expected.get()) {
    if (arg.count(k) != 1) return false;
    const auto& arg_v = arg.at(k);
    if (std::holds_alternative<TFCheckpoint>(arg_v)) {
      if (!std::holds_alternative<TFCheckpoint>(v)) return false;
      if (absl::get<TFCheckpoint>(v) != absl::get<TFCheckpoint>(arg_v))
        return false;
    } else if (std::holds_alternative<QuantizedTensor>(arg_v)) {
      if (!std::holds_alternative<QuantizedTensor>(v)) return false;
      const auto& v_q = absl::get<QuantizedTensor>(v);
      const auto& arg_q = absl::get<QuantizedTensor>(arg_v);
      if (v_q.values != arg_q.values) return false;
      if (v_q.bitwidth != arg_q.bitwidth) return false;
      if (v_q.dimensions != arg_q.dimensions) return false;
    } else {
      return false;
    }
  }
  return true;
}

// Tests the case where Finish is called without any prior call to Publish().
// This should work, but the checkpoint fed to the protocol should be empty.
TEST_F(FederatedTaskEnvironmentTest, TestFinishWithoutPublish) {
  ComputationResults empty_result;
  empty_result.emplace("tensorflow_checkpoint", "");
  EXPECT_CALL(
      mock_federated_protocol_,
      MockReportCompleted(EqualsComputationResult(std::cref(empty_result)),
                          std::vector<std::pair<std::string, double>>{},
                          absl::InfiniteDuration()))
      .WillOnce(Return(absl::UnimplementedError("")));
  EXPECT_CALL(mock_log_manager_,
              LogToLongHistogram(
                  HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME, 0, 0,
                  engine::DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
                  testing::Gt(0)));
  EXPECT_CALL(mock_log_manager_,
              LogToLongHistogram(
                  HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY, 0, 0,
                  engine::DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
                  testing::Ge(0)));

    EXPECT_CALL(mock_event_publisher_, PublishReportStarted(0));
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(opstats::OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED));
    EXPECT_CALL(mock_opstats_logger_, CommitToStorage())
        .WillOnce(Return(absl::OkStatus()));

  // NB once SecAgg is supported
  EXPECT_THAT(task_env_->Finish(engine::PhaseOutcome::COMPLETED,
                                absl::InfiniteDuration(), {}),
              fcp::IsCode(UNIMPLEMENTED));
}

TEST_F(FederatedTaskEnvironmentTest, TestFinishAndPublish) {
  // Expected ComputationResult.
  const std::string kTFCheckpointContent = "tf_checkpoint";
  ComputationResults result;
  result.emplace("tensorflow_checkpoint", kTFCheckpointContent);
  result.emplace("some_tensor", QuantizedTensor{{1L, 2L}, 1, {2, 1}});
  // Create a SecAgg Checkpoint file.
  Checkpoint checkpoint;
  Checkpoint::Aggregand secagg_aggregand;
  secagg_aggregand.mutable_quantized()->set_bitwidth(1);
  secagg_aggregand.mutable_quantized()->add_values(1L);
  secagg_aggregand.mutable_quantized()->add_values(2L);
  secagg_aggregand.mutable_quantized()->add_dimensions(2L);
  secagg_aggregand.mutable_quantized()->add_dimensions(1L);
  (*checkpoint.mutable_aggregands())["some_tensor"] = secagg_aggregand;
  const std::string secagg_file = fcp::TemporaryTestFile("secagg");
  EXPECT_THAT(
      fcp::WriteStringToFile(secagg_file, checkpoint.SerializeAsString()),
      fcp::IsCode(OK));
  // Create a TF checkpoint file.
  const std::string tf_file = fcp::TemporaryTestFile("tf");

  EXPECT_THAT(fcp::WriteStringToFile(tf_file, kTFCheckpointContent),
              fcp::IsCode(OK));

  EXPECT_CALL(mock_federated_protocol_,
              MockReportCompleted(EqualsComputationResult(std::cref(result)),
                                  std::vector<std::pair<std::string, double>>{},
                                  absl::InfiniteDuration()))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_log_manager_,
              LogToLongHistogram(
                  HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME, 0, 0,
                  engine::DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
                  testing::Gt(0)));
  EXPECT_CALL(mock_log_manager_,
              LogToLongHistogram(
                  HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY, 0, 0,
                  engine::DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
                  testing::Ge(0)));
  EXPECT_THAT(task_env_->PublishParameters(tf_file, secagg_file),
              fcp::IsCode(OK));

    EXPECT_CALL(mock_event_publisher_, PublishReportStarted(0));
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(opstats::OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED));
    EXPECT_CALL(mock_opstats_logger_, CommitToStorage())
        .WillOnce(Return(absl::OkStatus()));
    // Ensure PublishReportFinished is called with the protocol's up-to-date
    // values.
    EXPECT_CALL(mock_event_publisher_, PublishReportFinished(_, _, _))
        .WillOnce(DoAll([&](int64_t report_request_size_bytes,
                            int64_t chunking_layer_bytes_sent,
                            absl::Duration plan_duration) {
          EXPECT_EQ(report_request_size_bytes,
                    mock_federated_protocol_.report_request_size_bytes());
          EXPECT_EQ(chunking_layer_bytes_sent,
                    mock_federated_protocol_.chunking_layer_bytes_sent());
        }));
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(opstats::OperationalStats::Event::EVENT_KIND_UPLOAD_FINISHED));
  EXPECT_OK(task_env_->Finish(engine::PhaseOutcome::COMPLETED,
                              absl::InfiniteDuration(), {}));
}

TEST_F(FederatedTaskEnvironmentTest, TestFinishAndPublishWithError) {
  // Create a SecAgg Checkpoint file.
  Checkpoint checkpoint;
  Checkpoint::Aggregand secagg_aggregand;
  secagg_aggregand.mutable_quantized()->set_bitwidth(1);
  secagg_aggregand.mutable_quantized()->add_values(1L);
  (*checkpoint.mutable_aggregands())["some_tensor"] = secagg_aggregand;
  const std::string secagg_file = fcp::TemporaryTestFile("secagg");
  EXPECT_THAT(
      fcp::WriteStringToFile(secagg_file, checkpoint.SerializeAsString()),
      fcp::IsCode(OK));
  // Create a TF checkpoint file.
  const std::string tf_file = fcp::TemporaryTestFile("tf");
  const std::string kTFCheckpointContent = "tf_checkpoint";
  EXPECT_THAT(fcp::WriteStringToFile(tf_file, kTFCheckpointContent),
              fcp::IsCode(OK));

  // Create a checkpoint. We expect this one not to be sent to the protocol
  // even though we published it because we'll report an error.
  Checkpoint::Aggregand tf_aggregand;
  tf_aggregand.mutable_tensorflow_checkpoint()->set_checkpoint(
      kTFCheckpointContent);
  (*checkpoint.mutable_aggregands())["tensorflow_checkpoint"] = tf_aggregand;

  EXPECT_CALL(mock_federated_protocol_,
              MockReportNotCompleted(engine::PhaseOutcome::ERROR,
                                     absl::InfiniteDuration()))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_log_manager_,
              LogToLongHistogram(
                  HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME, 0, 0,
                  engine::DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
                  testing::Gt(0)));
  EXPECT_CALL(mock_log_manager_,
              LogToLongHistogram(
                  HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY, 0, 0,
                  engine::DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
                  testing::Ge(0)));
    EXPECT_CALL(mock_event_publisher_, PublishReportStarted(0));
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(opstats::OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED));
    EXPECT_CALL(mock_opstats_logger_, CommitToStorage())
        .WillOnce(Return(absl::OkStatus()));
    // Ensure PublishReportFinished is called with the protocol's up-to-date
    // values.
    EXPECT_CALL(mock_event_publisher_,
                PublishReportFinished(
                    MockFederatedProtocol::kPostReportNotCompletedStats
                        .report_size_bytes,
                    MockFederatedProtocol::kPostReportNotCompletedStats
                        .chunking_layer_bytes_sent,
                    _));
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(opstats::OperationalStats::Event::EVENT_KIND_UPLOAD_FINISHED));
  EXPECT_THAT(task_env_->PublishParameters(tf_file, secagg_file),
              fcp::IsCode(OK));
  EXPECT_OK(task_env_->Finish(engine::PhaseOutcome::ERROR,
                              absl::InfiniteDuration(), {}));
}

}  // anonymous namespace
}  // namespace client
}  // namespace fcp
