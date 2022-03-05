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
#include "fcp/client/phase_logger_impl.h"

#include <string>

#include "google/protobuf/util/time_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "fcp/client/test_helpers.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace client {
namespace {

using ::fcp::client::opstats::OperationalStats;
using ::google::internal::federatedml::v2::RetryWindow;
using ::google::protobuf::util::TimeUtil;
using ::testing::_;
using ::testing::Ge;
using ::testing::InSequence;
using ::testing::Return;
using ::testing::StrictMock;

const int64_t kBytesDownloaded = 200;
const int64_t kBytesUploaded = 100;
const int64_t kChunkingLayerBytesReceived = 100;
const int64_t kChunkingLayerBytesSent = 50;
const int64_t kReportSizeBytes = 15;
const int kTotalExampleCount = 10;
const int64_t kTotalExampleSizeBytes = 1000;

// Parameterize tests with
// 1) whether use per phase log;
// 2) whether log tf error message.
class PhaseLoggerImplTest
    : public testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    use_per_phase_logs_ = std::get<0>(GetParam());
    log_tensorflow_error_messages_ = std::get<1>(GetParam());
    EXPECT_CALL(mock_flags_, per_phase_logs())
        .WillRepeatedly(Return(use_per_phase_logs_));
    EXPECT_CALL(mock_flags_, log_tensorflow_error_messages())
        .WillRepeatedly(Return(log_tensorflow_error_messages_));
    phase_logger_ = std::make_unique<PhaseLoggerImpl>(
        &mock_event_publisher_, &mock_opstats_logger_, &mock_log_manager_,
        &mock_flags_);
  }

  void VerifyCounterLogged(HistogramCounters counter,
                           const testing::Matcher<int64_t>& matcher) {
    EXPECT_CALL(mock_log_manager_,
                LogToLongHistogram(counter, /*execution_index=*/0,
                                   /*epoch_index=*/0,
                                   engine::DataSourceType::DATASET, matcher));
  }

  StrictMock<MockLogManager> mock_log_manager_;
  StrictMock<MockEventPublisher> mock_event_publisher_;
  StrictMock<MockOpStatsLogger> mock_opstats_logger_;
  MockFlags mock_flags_;
  bool use_per_phase_logs_ = false;
  bool log_tensorflow_error_messages_ = false;
  std::unique_ptr<PhaseLoggerImpl> phase_logger_;
  NetworkStats network_stats_ = {
      .bytes_downloaded = kBytesDownloaded,
      .bytes_uploaded = kBytesUploaded,
      .chunking_layer_bytes_received = kChunkingLayerBytesReceived,
      .chunking_layer_bytes_sent = kChunkingLayerBytesSent,
      .report_size_bytes = kReportSizeBytes};
};

std::string GenerateTestName(
    const testing::TestParamInfo<PhaseLoggerImplTest::ParamType>& info) {
  std::string name = absl::StrCat(
      std::get<0>(info.param) ? "Per_phase_log" : "Legacy_log", "__",
      std::get<1>(info.param) ? "Log_tf_error_messages"
                              : "no_tf_error_messages");
  return name;
}

INSTANTIATE_TEST_SUITE_P(OldVsNewBehavior, PhaseLoggerImplTest,
                         testing::Combine(testing::Bool(), testing::Bool()),
                         GenerateTestName);

TEST_P(PhaseLoggerImplTest, UpdateRetryWindowAndNetworkStats) {
  RetryWindow retry_window;
  *retry_window.mutable_retry_token() = "retry_token";
  *retry_window.mutable_delay_max() = TimeUtil::HoursToDuration(48);
  *retry_window.mutable_delay_min() = TimeUtil::HoursToDuration(4);

  InSequence seq;
  EXPECT_CALL(mock_opstats_logger_, SetRetryWindow(EqualsProto(retry_window)));
  if (use_per_phase_logs_) {
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(kBytesDownloaded, kBytesUploaded,
                        kChunkingLayerBytesReceived, kChunkingLayerBytesSent));
  }

  phase_logger_->UpdateRetryWindowAndNetworkStats(retry_window, network_stats_);
}

TEST_P(PhaseLoggerImplTest, SetModelIdentifier) {
  std::string model_identifier = "model_identifier";
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(model_identifier));
  EXPECT_CALL(mock_log_manager_, SetModelIdentifier(model_identifier));

  phase_logger_->SetModelIdentifier(model_identifier);
}

TEST_P(PhaseLoggerImplTest, LogTaskNotStarted) {
  std::string error_message = "Client is not ready for training.";
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
                  error_message));
  phase_logger_->LogTaskNotStarted(error_message);
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalCheckInStarted) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(
            OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
  }
  phase_logger_->LogEligibilityEvalCheckInStarted();
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalCheckInIOError) {
  std::string error_message = "Network error";
  std::string expected_error_message = absl::StrCat(
      "Error during eligibility check-in: code: 3, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(
      mock_opstats_logger_,
      AddEventWithErrorMessage(OperationalStats::Event::EVENT_KIND_ERROR_IO,
                               expected_error_message));
  VerifyCounterLogged(
      HistogramCounters::TRAINING_FL_ELIGIBILITY_EVAL_CHECKIN_LATENCY, Ge(0));
  phase_logger_->LogEligibilityEvalCheckInIOError(
      absl::InvalidArgumentError(error_message), absl::Now());
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalCheckInClientInterrupted) {
  std::string error_message = "Client is not idle";
  std::string expected_error_message = absl::StrCat(
      "Error during eligibility check-in: code: 1, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
                  expected_error_message));
  VerifyCounterLogged(
      HistogramCounters::TRAINING_FL_ELIGIBILITY_EVAL_CHECKIN_LATENCY, Ge(0));

  phase_logger_->LogEligibilityEvalCheckInClientInterrupted(
      absl::CancelledError(error_message), absl::Now());
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalCheckInServerAborted) {
  std::string error_message = "Connection aborted by the server";
  std::string expected_error_message = absl::StrCat(
      "Error during eligibility check-in: code: 10, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_SERVER_ABORTED,
                  expected_error_message));
  VerifyCounterLogged(
      HistogramCounters::TRAINING_FL_ELIGIBILITY_EVAL_CHECKIN_LATENCY, Ge(0));

  phase_logger_->LogEligibilityEvalCheckInServerAborted(
      absl::AbortedError(error_message), absl::Now());
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalNotConfigured) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_event_publisher_,
                PublishEligibilityEvalNotConfigured(
                    kBytesDownloaded, kChunkingLayerBytesReceived, _));
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(OperationalStats::Event::EVENT_KIND_ELIGIBILITY_DISABLED));
  }
  VerifyCounterLogged(
      HistogramCounters::TRAINING_FL_ELIGIBILITY_EVAL_CHECKIN_LATENCY, Ge(0));

  phase_logger_->LogEligibilityEvalNotConfigured(network_stats_, absl::Now());
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalCheckInTurnedAway) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_event_publisher_,
                PublishEligibilityEvalRejected(kBytesDownloaded,
                                               kChunkingLayerBytesReceived, _));
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED));
  }
  VerifyCounterLogged(
      HistogramCounters::TRAINING_FL_ELIGIBILITY_EVAL_CHECKIN_LATENCY, Ge(0));

  phase_logger_->LogEligibilityEvalCheckInTurnedAway(network_stats_,
                                                     absl::Now());
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalCheckInInvalidPayloadError) {
  std::string error_message = "Cannot parse eligibility eval plan";
  InSequence seq;
  EXPECT_CALL(
      mock_log_manager_,
      LogDiag(
          ProdDiagCode::
              BACKGROUND_TRAINING_ELIGIBILITY_EVAL_FAILED_CANNOT_PARSE_PLAN));
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_ERROR_IO, error_message));
  VerifyCounterLogged(
      HistogramCounters::TRAINING_FL_ELIGIBILITY_EVAL_CHECKIN_LATENCY, Ge(0));

  phase_logger_->LogEligibilityEvalCheckInInvalidPayloadError(error_message,
                                                              absl::Now());
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalCheckInCompleted) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(OperationalStats::Event::EVENT_KIND_ELIGIBILITY_ENABLED));
    EXPECT_CALL(mock_event_publisher_,
                PublishEligibilityEvalPlanReceived(
                    kBytesDownloaded, kChunkingLayerBytesReceived, _));
  }
  VerifyCounterLogged(
      HistogramCounters::TRAINING_FL_ELIGIBILITY_EVAL_CHECKIN_LATENCY, Ge(0));

  phase_logger_->LogEligibilityEvalCheckInCompleted(network_stats_,
                                                    absl::Now());
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalComputationStarted) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_event_publisher_, PublishPlanExecutionStarted());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::
                             EVENT_KIND_ELIGIBILITY_COMPUTATION_STARTED));
  }

  phase_logger_->LogEligibilityEvalComputationStarted();
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalComputationInvalidArgument) {
  std::string error_message = "Invalid plan.";
  std::string expected_error_message = absl::StrCat(
      "Error during eligibility eval computation: code: 3, error: ",
      error_message);
  InSequence seq;
  EXPECT_CALL(
      mock_log_manager_,
      LogDiag(
          ProdDiagCode::BACKGROUND_TRAINING_FAILED_PLAN_FAILS_SANITY_CHECK));
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(
      mock_opstats_logger_,
      AddEventWithErrorMessage(OperationalStats::Event::EVENT_KIND_ERROR_IO,
                               expected_error_message));

  phase_logger_->LogEligibilityEvalComputationInvalidArgument(
      absl::InvalidArgumentError(error_message));
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalComputationExampleIteratorError) {
  std::string original_message = "Failed to create example iterator";
  absl::Status error_status = absl::InvalidArgumentError(original_message);
  std::string expected_error_message = absl::StrCat(
      "Error during eligibility eval computation: code: 3, error: ",
      original_message);
  if (use_per_phase_logs_) {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_,
                PublishExampleSelectorError(0, 0, 0, expected_error_message));
    EXPECT_CALL(mock_opstats_logger_,
                AddEventWithErrorMessage(
                    OperationalStats::Event::EVENT_KIND_ERROR_EXAMPLE_SELECTOR,
                    expected_error_message));
  }

  phase_logger_->LogEligibilityEvalComputationExampleIteratorError(
      error_status);
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalComputationTensorflowError) {
  std::string original_message = "Missing kernel for op NotExist";
  absl::Status error_status = absl::InvalidArgumentError(original_message);
  std::string expected_error_message = absl::StrCat(
      "Error during eligibility eval computation: code: 3, error: ");
  if (log_tensorflow_error_messages_) {
    absl::StrAppend(&expected_error_message, original_message);
  }
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_,
              PublishTensorFlowError(
                  /*execution_index=*/0, /*epoch_index=*/0, kTotalExampleCount,
                  expected_error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW,
                  expected_error_message));
  if (use_per_phase_logs_) {
    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_LATENCY, Ge(0));
    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_END_TIME, Ge(0));
  }
  phase_logger_->LogEligibilityEvalComputationTensorflowError(
      error_status, kTotalExampleCount, absl::Now() - absl::Minutes(2),
      absl::Now() - absl::Minutes(5));
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalComputationInterrupted) {
  std::string error_message = "Client is no longer idle";
  std::string expected_error_message = absl::StrCat(
      "Error during eligibility eval computation: code: 1, error: ",
      error_message);
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_event_publisher_,
                PublishInterruption(
                    /*execution_index=*/0, /*epoch_index=*/0,
                    kTotalExampleCount, kTotalExampleSizeBytes, _));
    EXPECT_CALL(mock_opstats_logger_,
                AddEventWithErrorMessage(
                    OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
                    expected_error_message));
    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_LATENCY, Ge(0));
    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_END_TIME, Ge(0));
  }

  phase_logger_->LogEligibilityEvalComputationInterrupted(
      absl::CancelledError(error_message), kTotalExampleCount,
      kTotalExampleSizeBytes, absl::Now() - absl::Minutes(2),
      absl::Now() - absl::Minutes(5));
}

TEST_P(PhaseLoggerImplTest, LogEligibilityEvalComputationCompleted) {
  absl::Time run_plan_start_time = absl::Now() - absl::Minutes(8);
  absl::Time reference_time = absl::Now() - absl::Minutes(9);
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_event_publisher_,
                PublishPlanCompleted(kTotalExampleCount, kTotalExampleSizeBytes,
                                     run_plan_start_time));
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::
                             EVENT_KIND_ELIGIBILITY_COMPUTATION_FINISHED));
    VerifyCounterLogged(HistogramCounters::TRAINING_OVERALL_EXAMPLE_SIZE,
                        kTotalExampleSizeBytes);
    VerifyCounterLogged(HistogramCounters::TRAINING_OVERALL_EXAMPLE_COUNT,
                        kTotalExampleCount);
    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_LATENCY, Ge(0));
    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_END_TIME, Ge(0));
  }

  phase_logger_->LogEligibilityEvalComputationCompleted(
      kTotalExampleCount, kTotalExampleSizeBytes, run_plan_start_time,
      reference_time);
}

TEST_P(PhaseLoggerImplTest, LogCheckInStarted) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
  }
  phase_logger_->LogCheckInStarted();
}

TEST_P(PhaseLoggerImplTest, LogCheckInIOError) {
  std::string error_message = "IO error";
  std::string expected_error_message =
      absl::StrCat("Error during check-in: code: 14, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(
      mock_opstats_logger_,
      AddEventWithErrorMessage(OperationalStats::Event::EVENT_KIND_ERROR_IO,
                               expected_error_message));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_LATENCY, Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_END_TIME, Ge(0));

  phase_logger_->LogCheckInIOError(absl::UnavailableError(error_message),
                                   absl::Now() - absl::Minutes(2),
                                   absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogCheckInClientInterrupted) {
  std::string error_message = "The client is no longer idle";
  std::string expected_error_message =
      absl::StrCat("Error during check-in: code: 1, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
                  expected_error_message));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_LATENCY, Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_END_TIME, Ge(0));

  phase_logger_->LogCheckInClientInterrupted(
      absl::CancelledError(error_message), absl::Now() - absl::Minutes(2),
      absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogCheckInServerAborted) {
  std::string error_message = "The request is aborted by the server";
  std::string expected_error_message =
      absl::StrCat("Error during check-in: code: 10, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_SERVER_ABORTED,
                  expected_error_message));

  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_LATENCY, Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_END_TIME, Ge(0));

  phase_logger_->LogCheckInServerAborted(absl::AbortedError(error_message),
                                         absl::Now() - absl::Minutes(2),
                                         absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogCheckInTurnedAway) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_event_publisher_, PublishRejected());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_REJECTED));
  }
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_LATENCY, Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_END_TIME, Ge(0));

  phase_logger_->LogCheckInTurnedAway(absl::Now() - absl::Minutes(2),
                                      absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogCheckInInvalidPayload) {
  std::string error_message = "Cannot parse plan";
  InSequence seq;
  EXPECT_CALL(
      mock_log_manager_,
      LogDiag(ProdDiagCode::BACKGROUND_TRAINING_FAILED_CANNOT_PARSE_PLAN));
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_ERROR_IO, error_message));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_LATENCY, Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_END_TIME, Ge(0));

  phase_logger_->LogCheckInInvalidPayload(error_message,
                                          absl::Now() - absl::Minutes(2),
                                          absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogCheckInCompleted) {
  std::string task_name = "my_task";
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_event_publisher_,
                PublishCheckinFinished(kBytesDownloaded,
                                       kChunkingLayerBytesReceived, _));
    EXPECT_CALL(mock_opstats_logger_,
                AddCheckinAcceptedEventWithTaskName(task_name));
  }
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_LATENCY, Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_CHECKIN_END_TIME, Ge(0));

  phase_logger_->LogCheckInCompleted(task_name, network_stats_,
                                     absl::Now() - absl::Minutes(2),
                                     absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogComputationStarted) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_event_publisher_, PublishPlanExecutionStarted());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED));
  }
  phase_logger_->LogComputationStarted();
}

TEST_P(PhaseLoggerImplTest, LogComputationInvalidArgument) {
  std::string error_message = "Unexpected input tensor";
  std::string expected_error_message =
      absl::StrCat("Error during computation: code: 3, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(
      mock_log_manager_,
      LogDiag(
          ProdDiagCode::BACKGROUND_TRAINING_FAILED_PLAN_FAILS_SANITY_CHECK));
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(
      mock_opstats_logger_,
      AddEventWithErrorMessage(OperationalStats::Event::EVENT_KIND_ERROR_IO,
                               expected_error_message));
  phase_logger_->LogComputationInvalidArgument(
      absl::InvalidArgumentError(error_message));
}

TEST_P(PhaseLoggerImplTest, LogComputationIOError) {
  std::string error_message = "IO error";
  std::string expected_error_message =
      absl::StrCat("Error during computation: code: 3, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(
      mock_opstats_logger_,
      AddEventWithErrorMessage(OperationalStats::Event::EVENT_KIND_ERROR_IO,
                               expected_error_message));
  phase_logger_->LogComputationIOError(
      absl::InvalidArgumentError(error_message));
}

TEST_P(PhaseLoggerImplTest, LogComputationExampleIteratorError) {
  std::string original_message = "Cannot create example iterator";
  absl::Status error_status = absl::InvalidArgumentError(original_message);
  std::string expected_error_message = absl::StrCat(
      "Error during computation: code: 3, error: ", original_message);
  if (use_per_phase_logs_) {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_,
                PublishExampleSelectorError(0, 0, 0, expected_error_message));
    EXPECT_CALL(mock_opstats_logger_,
                AddEventWithErrorMessage(
                    OperationalStats::Event::EVENT_KIND_ERROR_EXAMPLE_SELECTOR,
                    expected_error_message));
  }
  phase_logger_->LogComputationExampleIteratorError(error_status);
}

TEST_P(PhaseLoggerImplTest, LogComputationTensorflowError) {
  std::string original_message = "Missing op kernel NotExist";
  absl::Status error_status = absl::InvalidArgumentError(original_message);
  std::string expected_error_message =
      absl::StrCat("Error during computation: code: 3, error: ");
  if (log_tensorflow_error_messages_) {
    absl::StrAppend(&expected_error_message, original_message);
  }
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_,
              PublishTensorFlowError(
                  /*execution_index=*/0, /*epoch_index=*/0, kTotalExampleCount,
                  expected_error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW,
                  expected_error_message));
  if (use_per_phase_logs_) {
    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_LATENCY, Ge(0));
    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_END_TIME, Ge(0));
  }
  phase_logger_->LogComputationTensorflowError(error_status, kTotalExampleCount,
                                               absl::Now() - absl::Minutes(6),
                                               absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogComputationInterrupted) {
  absl::Time run_plan_start_time = absl::Now() - absl::Minutes(6);
  absl::Time reference_time = absl::Now() - absl::Minutes(8);
  std::string error_message = "Client is no longer idle.";
  std::string expected_error_message =
      absl::StrCat("Error during computation: code: 1, error: ", error_message);
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(
        mock_event_publisher_,
        PublishInterruption(
            /*execution_index=*/0, /*epoch_index=*/0, kTotalExampleCount,
            kTotalExampleSizeBytes, run_plan_start_time));
    EXPECT_CALL(mock_opstats_logger_,
                AddEventWithErrorMessage(
                    OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
                    expected_error_message));

    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_LATENCY, Ge(0));
    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_END_TIME, Ge(0));
  }

  phase_logger_->LogComputationInterrupted(
      absl::CancelledError(error_message), kTotalExampleCount,
      kTotalExampleSizeBytes, run_plan_start_time, reference_time);
}

TEST_P(PhaseLoggerImplTest, LogComputationCompleted) {
  absl::Time run_plan_start_time = absl::Now() - absl::Minutes(6);
  absl::Time reference_time = absl::Now() - absl::Minutes(8);
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_event_publisher_,
                PublishPlanCompleted(kTotalExampleCount, kTotalExampleSizeBytes,
                                     run_plan_start_time));
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED));
    VerifyCounterLogged(HistogramCounters::TRAINING_OVERALL_EXAMPLE_SIZE,
                        kTotalExampleSizeBytes);
    VerifyCounterLogged(HistogramCounters::TRAINING_OVERALL_EXAMPLE_COUNT,
                        kTotalExampleCount);
    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_LATENCY, Ge(0));
    VerifyCounterLogged(HistogramCounters::TRAINING_RUN_PHASE_END_TIME, Ge(0));
  }

  phase_logger_->LogComputationCompleted(kTotalExampleCount,
                                         kTotalExampleSizeBytes,
                                         run_plan_start_time, reference_time);
}

TEST_P(PhaseLoggerImplTest, LogResultUploadStartedOpStatsDbCommitSucceeds) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED));
    EXPECT_CALL(mock_opstats_logger_, CommitToStorage)
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(mock_event_publisher_, PublishReportStarted(0));
  }

  ASSERT_OK(phase_logger_->LogResultUploadStarted());
}

TEST_P(PhaseLoggerImplTest, LogResultUploadStartedOpStatsDbCommitFails) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED));
    EXPECT_CALL(mock_opstats_logger_, CommitToStorage)
        .WillOnce(Return(absl::InternalError("")));
  }

  absl::Status status = phase_logger_->LogResultUploadStarted();
  if (use_per_phase_logs_) {
    ASSERT_THAT(status, IsCode(absl::StatusCode::kInternal));
  } else {
    ASSERT_OK(status);
  }
}

TEST_P(PhaseLoggerImplTest, LogResultUploadIOError) {
  std::string error_message = "Network IO";
  std::string expected_error_message =
      absl::StrCat("Error reporting results: code: 14, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(
      mock_opstats_logger_,
      AddEventWithErrorMessage(OperationalStats::Event::EVENT_KIND_ERROR_IO,
                               expected_error_message));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY,
                      Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME,
                      Ge(0));
  phase_logger_->LogResultUploadIOError(absl::UnavailableError(error_message),
                                        absl::Now() - absl::Minutes(1),
                                        absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogResultUploadClientInterrupted) {
  std::string error_message = "Client is no longer idle";
  std::string expected_error_message =
      absl::StrCat("Error reporting results: code: 1, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
                  expected_error_message));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY,
                      Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME,
                      Ge(0));

  phase_logger_->LogResultUploadClientInterrupted(
      absl::CancelledError(error_message), absl::Now() - absl::Minutes(1),
      absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogResultUploadServerAborted) {
  std::string error_message = "Request is aborted by the server";
  std::string expected_error_message =
      absl::StrCat("Error reporting results: code: 10, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_SERVER_ABORTED,
                  expected_error_message));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY,
                      Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME,
                      Ge(0));

  phase_logger_->LogResultUploadServerAborted(absl::AbortedError(error_message),
                                              absl::Now() - absl::Minutes(1),
                                              absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogResultUploadCompleted) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(
        mock_event_publisher_,
        PublishReportFinished(kReportSizeBytes, kChunkingLayerBytesSent, _));
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_FINISHED));
  }
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY,
                      Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME,
                      Ge(0));

  phase_logger_->LogResultUploadCompleted(network_stats_,
                                          absl::Now() - absl::Minutes(1),
                                          absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogFailureUploadStartedOpstatsDbCommitSucceeds) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED));
    EXPECT_CALL(mock_opstats_logger_, CommitToStorage())
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(mock_event_publisher_, PublishReportStarted(0));
  }
  ASSERT_OK(phase_logger_->LogFailureUploadStarted());
}

TEST_P(PhaseLoggerImplTest, LogFailureUploadStartedOpstatsDbCommitFails) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED));
    EXPECT_CALL(mock_opstats_logger_, CommitToStorage())
        .WillOnce(Return(absl::InternalError("")));
  }
  absl::Status status = phase_logger_->LogFailureUploadStarted();
  if (use_per_phase_logs_) {
    ASSERT_THAT(status, IsCode(absl::StatusCode::kInternal));
  } else {
    ASSERT_OK(status);
  }
}

TEST_P(PhaseLoggerImplTest, LogFailureUploadIOError) {
  std::string error_message = "Network error.";
  std::string expected_error_message = absl::StrCat(
      "Error reporting computation failure: code: 14, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(
      mock_opstats_logger_,
      AddEventWithErrorMessage(OperationalStats::Event::EVENT_KIND_ERROR_IO,
                               expected_error_message));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY,
                      Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME,
                      Ge(0));
  phase_logger_->LogFailureUploadIOError(absl::UnavailableError(error_message),
                                         absl::Now() - absl::Minutes(1),
                                         absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogFailureUploadClientInterrupted) {
  std::string error_message = "The client is no longer idle";
  std::string expected_error_message = absl::StrCat(
      "Error reporting computation failure: code: 1, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
                  expected_error_message));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY,
                      Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME,
                      Ge(0));
  phase_logger_->LogFailureUploadClientInterrupted(
      absl::CancelledError(error_message), absl::Now() - absl::Minutes(1),
      absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogFailureUploadServerAborted) {
  std::string error_message = "Request is aborted by the server.";
  std::string expected_error_message = absl::StrCat(
      "Error reporting computation failure: code: 10, error: ", error_message);
  InSequence seq;
  EXPECT_CALL(mock_event_publisher_, PublishIoError(0, expected_error_message));
  EXPECT_CALL(mock_opstats_logger_,
              AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_SERVER_ABORTED,
                  expected_error_message));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY,
                      Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME,
                      Ge(0));
  phase_logger_->LogFailureUploadServerAborted(
      absl::AbortedError(error_message), absl::Now() - absl::Minutes(1),
      absl::Now() - absl::Minutes(8));
}

TEST_P(PhaseLoggerImplTest, LogFailureUploadCompleted) {
  InSequence seq;
  if (use_per_phase_logs_) {
    EXPECT_CALL(
        mock_event_publisher_,
        PublishReportFinished(kReportSizeBytes, kChunkingLayerBytesSent, _));
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_FINISHED));
  }
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY,
                      Ge(0));
  VerifyCounterLogged(HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME,
                      Ge(0));
  phase_logger_->LogFailureUploadCompleted(network_stats_,
                                           absl::Now() - absl::Minutes(1),
                                           absl::Now() - absl::Minutes(8));
}

}  // namespace
}  // namespace client
}  // namespace fcp
