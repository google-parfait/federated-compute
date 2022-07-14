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
#ifndef FCP_CLIENT_TEST_HELPERS_H_
#define FCP_CLIENT_TEST_HELPERS_H_

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/flags.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_db.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/phase_logger.h"
#include "fcp/client/secagg_event_publisher.h"
#include "fcp/client/simple_task_environment.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

namespace fcp {
namespace client {

class MockSecAggEventPublisher : public SecAggEventPublisher {
 public:
  MOCK_METHOD(void, PublishStateTransition,
              (::fcp::secagg::ClientState state, size_t last_sent_message_size,
               size_t last_received_message_size),
              (override));
  MOCK_METHOD(void, PublishError, (), (override));
  MOCK_METHOD(void, PublishAbort,
              (bool client_initiated, const std::string& error_message),
              (override));
  MOCK_METHOD(void, set_execution_session_id, (int64_t execution_session_id),
              (override));
};

class MockEventPublisher : public EventPublisher {
 public:
  MOCK_METHOD(void, PublishEligibilityEvalCheckin, (), (override));
  MOCK_METHOD(void, PublishEligibilityEvalPlanReceived,
              (int64_t bytes_downloaded,
               int64_t chunking_layer_bytes_downloaded,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalNotConfigured,
              (int64_t bytes_downloaded,
               int64_t chunking_layer_bytes_downloaded,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalRejected,
              (int64_t bytes_downloaded,
               int64_t chunking_layer_bytes_downloaded,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishCheckin, (), (override));
  MOCK_METHOD(void, PublishCheckinFinished,
              (int64_t bytes_downloaded,
               int64_t chunking_layer_bytes_downloaded,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishRejected, (), (override));
  MOCK_METHOD(void, PublishReportStarted, (int64_t report_size_bytes),
              (override));
  MOCK_METHOD(void, PublishReportFinished,
              (int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
               absl::Duration report_duration),
              (override));
  MOCK_METHOD(void, PublishPlanExecutionStarted, (), (override));
  MOCK_METHOD(void, PublishEpochStarted, (int execution_index, int epoch_index),
              (override));
  MOCK_METHOD(void, PublishTensorFlowError,
              (int execution_index, int epoch_index, int epoch_example_index,
               absl::string_view error_message),
              (override));
  MOCK_METHOD(void, PublishIoError,
              (int execution_index, absl::string_view error_message),
              (override));
  MOCK_METHOD(void, PublishExampleSelectorError,
              (int execution_index, int epoch_index, int epoch_example_index,
               absl::string_view error_message),
              (override));
  MOCK_METHOD(void, PublishInterruption,
              (int execution_index, int epoch_index, int epoch_example_index,
               int64_t total_example_size_bytes, absl::Time start_time),
              (override));
  MOCK_METHOD(void, PublishEpochCompleted,
              (int execution_index, int epoch_index, int epoch_example_index,
               int64_t epoch_example_size_bytes, absl::Time epoch_start_time),
              (override));
  MOCK_METHOD(void, PublishStats,
              (int execution_index, int epoch_index,
               (const absl::flat_hash_map<std::string, double>& stats)),
              (override));
  MOCK_METHOD(void, PublishPlanCompleted,
              (int total_example_count, int64_t total_example_size_bytes,
               absl::Time start_time),
              (override));
  MOCK_METHOD(void, SetModelIdentifier, (const std::string& model_identifier),
              (override));
  MOCK_METHOD(void, PublishTaskNotStarted, (absl::string_view error_message),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalCheckInIoError,
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
               absl::string_view error_message,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalCheckInClientInterrupted,
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
               absl::string_view error_message,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalCheckInServerAborted,
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
               absl::string_view error_message,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalCheckInErrorInvalidPayload,
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
               absl::string_view error_message,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalComputationStarted, (), (override));
  MOCK_METHOD(void, PublishEligibilityEvalComputationInvalidArgument,
              (absl::string_view error_message, int total_example_count,
               int64_t total_example_size_bytes,
               absl::Duration computation_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalComputationExampleIteratorError,
              (absl::string_view error_message, int total_example_count,
               int64_t total_example_size_bytes,
               absl::Duration computation_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalComputationTensorflowError,
              (int total_example_count, int64_t total_example_size_bytes,
               absl::string_view error_message,
               absl::Duration computation_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalComputationInterrupted,
              (int total_example_count, int64_t total_example_size_bytes,
               absl::string_view error_message,
               absl::Duration computation_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalComputationCompleted,
              (int total_example_count, int64_t total_example_size_bytes,
               absl::Duration computation_duration),
              (override));
  MOCK_METHOD(void, PublishCheckinIoError,
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
               absl::string_view error_message,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishCheckinClientInterrupted,
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
               absl::string_view error_message,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishCheckinServerAborted,
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
               absl::string_view error_message,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishCheckinInvalidPayload,
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
               absl::string_view error_message,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishRejected,
              (int64_t bytes_downloaded,
               int64_t chunking_layer_bytes_downloaded,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishCheckinFinishedV2,
              (int64_t bytes_downloaded,
               int64_t chunking_layer_bytes_downloaded,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishComputationStarted, (), (override));
  MOCK_METHOD(void, PublishComputationInvalidArgument,
              (absl::string_view error_message, int total_example_count,
               int64_t total_example_size_bytes,
               absl::Duration computation_duration),
              (override));
  MOCK_METHOD(void, PublishComputationIOError,
              (absl::string_view error_message, int total_example_count,
               int64_t total_example_size_bytes,
               absl::Duration computation_duration),
              (override));
  MOCK_METHOD(void, PublishComputationExampleIteratorError,
              (absl::string_view error_message, int total_example_count,
               int64_t total_example_size_bytes,
               absl::Duration computation_duration),
              (override));
  MOCK_METHOD(void, PublishComputationTensorflowError,
              (int total_example_count, int64_t total_example_size_bytes,
               absl::string_view error_message,
               absl::Duration computation_duration),
              (override));
  MOCK_METHOD(void, PublishComputationInterrupted,
              (int total_example_count, int64_t total_example_size_bytes,
               absl::string_view error_message,
               absl::Duration computation_duration),
              (override));
  MOCK_METHOD(void, PublishComputationCompleted,
              (int total_example_count, int64_t total_example_size_bytes,
               absl::Time start_time),
              (override));
  MOCK_METHOD(void, PublishResultUploadStarted, (), (override));
  MOCK_METHOD(void, PublishResultUploadIOError,
              (int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
               absl::string_view error_message, absl::Duration upload_duration),
              (override));
  MOCK_METHOD(void, PublishResultUploadClientInterrupted,
              (int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
               absl::string_view error_message, absl::Duration upload_duration),
              (override));
  MOCK_METHOD(void, PublishResultUploadServerAborted,
              (int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
               absl::string_view error_message, absl::Duration upload_duration),
              (override));
  MOCK_METHOD(void, PublishResultUploadCompleted,
              (int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
               absl::Duration upload_duration),
              (override));
  MOCK_METHOD(void, PublishFailureUploadStarted, (), (override));
  MOCK_METHOD(void, PublishFailureUploadIOError,
              (int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
               absl::string_view error_message, absl::Duration upload_duration),
              (override));
  MOCK_METHOD(void, PublishFailureUploadClientInterrupted,
              (int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
               absl::string_view error_message, absl::Duration upload_duration),
              (override));
  MOCK_METHOD(void, PublishFailureUploadServerAborted,
              (int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
               absl::string_view error_message, absl::Duration upload_duration),
              (override));
  MOCK_METHOD(void, PublishFailureUploadCompleted,
              (int64_t report_size_bytes, int64_t chunking_layer_bytes_snet,
               absl::Duration upload_duration),
              (override));

  SecAggEventPublisher* secagg_event_publisher() override {
    return &secagg_event_publisher_;
  }

 private:
  ::testing::NiceMock<MockSecAggEventPublisher> secagg_event_publisher_;
};

// A mock FederatedProtocol implementation, which keeps track of the stages in
// the protocol and returns a different set of network stats and RetryWindow for
// each stage, making it easier to write accurate assertions in unit tests.
class MockFederatedProtocol : public FederatedProtocol {
 public:
  constexpr static NetworkStats kPostEligibilityCheckinStats = {
      .bytes_downloaded = 100,
      .bytes_uploaded = 200,
      .chunking_layer_bytes_received = 300,
      .chunking_layer_bytes_sent = 400,
      .report_size_bytes = 0};
  constexpr static NetworkStats kPostCheckinStats = {
      .bytes_downloaded = 1000,
      .bytes_uploaded = 2000,
      .chunking_layer_bytes_received = 3000,
      .chunking_layer_bytes_sent = 4000,
      .report_size_bytes = 0};
  constexpr static NetworkStats kPostReportCompletedStats = {
      .bytes_downloaded = 10000,
      .bytes_uploaded = 20000,
      .chunking_layer_bytes_received = 30000,
      .chunking_layer_bytes_sent = 40000,
      .report_size_bytes = 555};
  constexpr static NetworkStats kPostReportNotCompletedStats = {
      .bytes_downloaded = 9999,
      .bytes_uploaded = 19999,
      .chunking_layer_bytes_received = 29999,
      .chunking_layer_bytes_sent = 39999,
      .report_size_bytes = 111};

  static google::internal::federatedml::v2::RetryWindow
  GetInitialRetryWindow() {
    google::internal::federatedml::v2::RetryWindow retry_window;
    retry_window.mutable_delay_min()->set_seconds(0L);
    retry_window.mutable_delay_max()->set_seconds(1L);
    *retry_window.mutable_retry_token() = "INITIAL";
    return retry_window;
  }

  static google::internal::federatedml::v2::RetryWindow
  GetPostEligibilityCheckinRetryWindow() {
    google::internal::federatedml::v2::RetryWindow retry_window;
    retry_window.mutable_delay_min()->set_seconds(100L);
    retry_window.mutable_delay_max()->set_seconds(101L);
    *retry_window.mutable_retry_token() = "POST_ELIGIBILITY";
    return retry_window;
  }

  static google::internal::federatedml::v2::RetryWindow
  GetPostCheckinRetryWindow() {
    google::internal::federatedml::v2::RetryWindow retry_window;
    retry_window.mutable_delay_min()->set_seconds(200L);
    retry_window.mutable_delay_max()->set_seconds(201L);
    *retry_window.mutable_retry_token() = "POST_CHECKIN";
    return retry_window;
  }

  static google::internal::federatedml::v2::RetryWindow
  GetPostReportCompletedRetryWindow() {
    google::internal::federatedml::v2::RetryWindow retry_window;
    retry_window.mutable_delay_min()->set_seconds(300L);
    retry_window.mutable_delay_max()->set_seconds(301L);
    *retry_window.mutable_retry_token() = "POST_REPORT_COMPLETED";
    return retry_window;
  }

  static google::internal::federatedml::v2::RetryWindow
  GetPostReportNotCompletedRetryWindow() {
    google::internal::federatedml::v2::RetryWindow retry_window;
    retry_window.mutable_delay_min()->set_seconds(400L);
    retry_window.mutable_delay_max()->set_seconds(401L);
    *retry_window.mutable_retry_token() = "POST_REPORT_NOT_COMPLETED";
    return retry_window;
  }

  explicit MockFederatedProtocol() {}

  // We override the real FederatedProtocol methods so that we can intercept the
  // progression of protocol stages, and expose dedicate gMock-overridable
  // methods for use in tests.
  absl::StatusOr<EligibilityEvalCheckinResult> EligibilityEvalCheckin() final {
    network_stats_ = kPostEligibilityCheckinStats;
    retry_window_ = GetPostEligibilityCheckinRetryWindow();
    return MockEligibilityEvalCheckin();
  };
  MOCK_METHOD(absl::StatusOr<EligibilityEvalCheckinResult>,
              MockEligibilityEvalCheckin, ());

  absl::StatusOr<CheckinResult> Checkin(
      const std::optional<
          ::google::internal::federatedml::v2::TaskEligibilityInfo>&
          task_eligibility_info) final {
    network_stats_ = kPostCheckinStats;
    retry_window_ = GetPostCheckinRetryWindow();
    return MockCheckin(task_eligibility_info);
  };
  MOCK_METHOD(absl::StatusOr<CheckinResult>, MockCheckin,
              (const std::optional<
                  ::google::internal::federatedml::v2::TaskEligibilityInfo>&
                   task_eligibility_info));

  absl::Status ReportCompleted(ComputationResults results,
                               absl::Duration plan_duration) final {
    network_stats_ = kPostReportCompletedStats;
    retry_window_ = GetPostReportCompletedRetryWindow();
    return MockReportCompleted(std::move(results), plan_duration);
  };
  MOCK_METHOD(absl::Status, MockReportCompleted,
              (ComputationResults results, absl::Duration plan_duration));

  absl::Status ReportNotCompleted(engine::PhaseOutcome phase_outcome,
                                  absl::Duration plan_duration) final {
    network_stats_ = kPostReportNotCompletedStats;
    retry_window_ = GetPostReportNotCompletedRetryWindow();
    return MockReportNotCompleted(phase_outcome, plan_duration);
  };
  MOCK_METHOD(absl::Status, MockReportNotCompleted,
              (engine::PhaseOutcome phase_outcome,
               absl::Duration plan_duration));

  ::google::internal::federatedml::v2::RetryWindow GetLatestRetryWindow()
      final {
    return retry_window_;
  }

  int64_t chunking_layer_bytes_sent() final {
    return network_stats_.chunking_layer_bytes_sent;
  }
  int64_t chunking_layer_bytes_received() final {
    return network_stats_.chunking_layer_bytes_received;
  }
  int64_t bytes_downloaded() final { return network_stats_.bytes_downloaded; };
  int64_t bytes_uploaded() final { return network_stats_.bytes_uploaded; };
  int64_t report_request_size_bytes() final {
    return network_stats_.report_size_bytes;
  };

 private:
  NetworkStats network_stats_;
  ::google::internal::federatedml::v2::RetryWindow retry_window_ =
      GetInitialRetryWindow();
};

class MockLogManager : public LogManager {
 public:
  MOCK_METHOD(void, LogDiag, (ProdDiagCode), (override));
  MOCK_METHOD(void, LogDiag, (DebugDiagCode), (override));
  MOCK_METHOD(void, LogToLongHistogram,
              (fcp::client::HistogramCounters, int, int,
               fcp::client::engine::DataSourceType, int64_t),
              (override));
  MOCK_METHOD(void, SetModelIdentifier, (const std::string&), (override));
};

class MockOpStatsLogger : public ::fcp::client::opstats::OpStatsLogger {
 public:
  MOCK_METHOD(void, AddCheckinAcceptedEventWithTaskName,
              (const std::string& task_name), (override));
  MOCK_METHOD(
      void, AddEvent,
      (::fcp::client::opstats::OperationalStats::Event::EventKind event),
      (override));
  MOCK_METHOD(void, AddEventWithErrorMessage,
              (::fcp::client::opstats::OperationalStats::Event::EventKind event,
               const std::string& error_message),
              (override));
  MOCK_METHOD(void, UpdateDatasetStats,
              (const std::string& collection_uri, int additional_example_count,
               int64_t additional_example_size_bytes),
              (override));
  MOCK_METHOD(void, SetNetworkStats,
              (int64_t bytes_downloaded, int64_t bytes_uploaded,
               int64_t chunking_layer_bytes_downloaded,
               int64_t chunking_layer_bytes_uploaded),
              (override));
  MOCK_METHOD(void, SetRetryWindow,
              (google::internal::federatedml::v2::RetryWindow retry_window),
              (override));
  MOCK_METHOD(::fcp::client::opstats::OpStatsDb*, GetOpStatsDb, (), (override));
  MOCK_METHOD(bool, IsOpStatsEnabled, (), (const override));
  MOCK_METHOD(absl::Status, CommitToStorage, (), (override));
};

class MockSimpleTaskEnvironment : public SimpleTaskEnvironment {
 public:
  MOCK_METHOD(std::string, GetBaseDir, (), (override));
  MOCK_METHOD((absl::StatusOr<std::unique_ptr<ExampleIterator>>),
              CreateExampleIterator,
              (const google::internal::federated::plan::ExampleSelector&
                   example_selector),
              (override));
  MOCK_METHOD((absl::StatusOr<std::unique_ptr<ExampleIterator>>),
              CreateExampleIterator,
              (const google::internal::federated::plan::ExampleSelector&
                   example_selector,
               const SelectorContext& selector_context),
              (override));
  MOCK_METHOD(std::unique_ptr<fcp::client::http::HttpClient>, CreateHttpClient,
              (), (override));
  MOCK_METHOD(bool, TrainingConditionsSatisfied, (), (override));
};

class MockExampleIterator : public ExampleIterator {
 public:
  MOCK_METHOD(absl::StatusOr<std::string>, Next, (), (override));
  MOCK_METHOD(void, Close, (), (override));
};

// An iterator that passes through each example in the dataset once.
class SimpleExampleIterator : public ExampleIterator {
 public:
  // Passes through each of the examples in the `Dataset.client_data.example`
  // field.
  explicit SimpleExampleIterator(
      google::internal::federated::plan::Dataset dataset);
  // Passes through each of the examples in the
  // `Dataset.client_data.selected_example.example` field, whose example
  // collection URI matches the provided `collection_uri`.
  SimpleExampleIterator(google::internal::federated::plan::Dataset dataset,
                        absl::string_view collection_uri);
  absl::StatusOr<std::string> Next() override;
  void Close() override {}

 protected:
  std::vector<std::string> examples_;
  int index_ = 0;
};

class MockFlags : public Flags {
 public:
  MOCK_METHOD(int64_t, condition_polling_period_millis, (), (const, override));
  MOCK_METHOD(int64_t, tf_execution_teardown_grace_period_millis, (),
              (const, override));
  MOCK_METHOD(int64_t, tf_execution_teardown_extended_period_millis, (),
              (const, override));
  MOCK_METHOD(int64_t, grpc_channel_deadline_seconds, (), (const, override));
  MOCK_METHOD(bool, log_tensorflow_error_messages, (), (const, override));
  MOCK_METHOD(bool, enable_opstats, (), (const, override));
  MOCK_METHOD(int64_t, opstats_ttl_days, (), (const, override));
  MOCK_METHOD(int64_t, opstats_db_size_limit_bytes, (), (const, override));
  MOCK_METHOD(int64_t, federated_training_transient_errors_retry_delay_secs, (),
              (const, override));
  MOCK_METHOD(float,
              federated_training_transient_errors_retry_delay_jitter_percent,
              (), (const, override));
  MOCK_METHOD(int64_t, federated_training_permanent_errors_retry_delay_secs, (),
              (const, override));
  MOCK_METHOD(float,
              federated_training_permanent_errors_retry_delay_jitter_percent,
              (), (const, override));
  MOCK_METHOD(std::vector<int32_t>, federated_training_permanent_error_codes,
              (), (const, override));
  MOCK_METHOD(bool, use_tflite_training, (), (const, override));
  MOCK_METHOD(bool, enable_grpc_with_http_resource_support, (),
              (const, override));
  MOCK_METHOD(bool, enable_grpc_with_eligibility_eval_http_resource_support, (),
              (const, override));
  MOCK_METHOD(bool, granular_per_phase_logs, (), (const, override));
  MOCK_METHOD(bool, ensure_dynamic_tensors_are_released, (), (const, override));
  MOCK_METHOD(int32_t, large_tensor_threshold_for_dynamic_allocation, (),
              (const, override));
  MOCK_METHOD(bool, disable_http_request_body_compression, (),
              (const, override));
  MOCK_METHOD(bool, use_http_federated_compute_protocol, (), (const, override));
  MOCK_METHOD(bool, enable_computation_id, (), (const, override));
  MOCK_METHOD(int32_t, waiting_period_sec_for_cancellation, (),
              (const, override));
};

// Helper methods for extracting opstats fields from TF examples.
std::string ExtractSingleString(const tensorflow::Example& example,
                                const char key[]);
google::protobuf::RepeatedPtrField<std::string> ExtractRepeatedString(
    const tensorflow::Example& example, const char key[]);
int64_t ExtractSingleInt64(const tensorflow::Example& example,
                           const char key[]);
google::protobuf::RepeatedField<int64_t> ExtractRepeatedInt64(
    const tensorflow::Example& example, const char key[]);

class MockOpStatsDb : public ::fcp::client::opstats::OpStatsDb {
 public:
  MOCK_METHOD(absl::StatusOr<::fcp::client::opstats::OpStatsSequence>, Read, (),
              (override));
  MOCK_METHOD(absl::Status, Transform,
              (std::function<void(::fcp::client::opstats::OpStatsSequence&)>),
              (override));
};

class MockPhaseLogger : public PhaseLogger {
 public:
  MOCK_METHOD(
      void, UpdateRetryWindowAndNetworkStats,
      (const ::google::internal::federatedml::v2::RetryWindow& retry_window,
       NetworkStats stats),
      (override));
  MOCK_METHOD(void, SetModelIdentifier, (absl::string_view model_identifier),
              (override));
  MOCK_METHOD(void, LogTaskNotStarted, (absl::string_view error_message),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalCheckInStarted, (), (override));
  MOCK_METHOD(void, LogEligibilityEvalCheckInIOError,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_eligibility_eval_checkin),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalCheckInInvalidPayloadError,
              (absl::string_view error_message, NetworkStats stats,
               absl::Time time_before_eligibility_eval_checkin),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalCheckInClientInterrupted,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_eligibility_eval_checkin),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalCheckInServerAborted,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_eligibility_eval_checkin),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalNotConfigured,
              (NetworkStats stats,
               absl::Time time_before_eligibility_eval_checkin),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalCheckInTurnedAway,
              (NetworkStats stats,
               absl::Time time_before_eligibility_eval_checkin),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalCheckInCompleted,
              (NetworkStats stats,
               absl::Time time_before_eligibility_eval_checkin),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalComputationStarted, (), (override));
  MOCK_METHOD(void, LogEligibilityEvalComputationInvalidArgument,
              (absl::Status error_status, int total_example_count,
               int64_t total_example_size_bytes,
               absl::Time run_plan_start_time),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalComputationExampleIteratorError,
              (absl::Status error_status, int total_example_count,
               int64_t total_example_size_bytes,
               absl::Time run_plan_start_time),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalComputationTensorflowError,
              (absl::Status error_status, int total_example_count,
               int64_t total_example_size_bytes, absl::Time run_plan_start_time,
               absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalComputationInterrupted,
              (absl::Status error_status, int total_example_count,
               int64_t total_example_size_bytes, absl::Time run_plan_start_time,
               absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogEligibilityEvalComputationCompleted,
              (int total_example_count, int64_t total_example_size_bytes,
               absl::Time run_plan_start_time, absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogCheckInStarted, (), (override));
  MOCK_METHOD(void, LogCheckInIOError,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_checkin, absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogCheckInInvalidPayload,
              (absl::string_view error_message, NetworkStats stats,
               absl::Time time_before_checkin, absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogCheckInClientInterrupted,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_checkin, absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogCheckInServerAborted,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_checkin, absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogCheckInTurnedAway,
              (NetworkStats stats, absl::Time time_before_checkin,
               absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogCheckInCompleted,
              (absl::string_view task_name, NetworkStats stats,
               absl::Time time_before_checkin, absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogComputationStarted, (), (override));
  MOCK_METHOD(void, LogComputationInvalidArgument,
              (absl::Status error_status, int total_example_count,
               int64_t total_example_size_bytes,
               absl::Time run_plan_start_time),
              (override));
  MOCK_METHOD(void, LogComputationExampleIteratorError,
              (absl::Status error_status, int total_example_count,
               int64_t total_example_size_bytes,
               absl::Time run_plan_start_time),
              (override));
  MOCK_METHOD(void, LogComputationIOError,
              (absl::Status error_status, int total_example_count,
               int64_t total_example_size_bytes,
               absl::Time run_plan_start_time),
              (override));
  MOCK_METHOD(void, LogComputationTensorflowError,
              (absl::Status error_status, int total_example_count,
               int64_t total_example_size_bytes, absl::Time run_plan_start_time,
               absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogComputationInterrupted,
              (absl::Status error_status, int total_example_count,
               int64_t total_example_size_bytes, absl::Time run_plan_start_time,
               absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogComputationCompleted,
              (int total_example_count, int64_t total_example_size_bytes,
               absl::Time run_plan_start_time, absl::Time reference_time),
              (override));
  MOCK_METHOD(absl::Status, LogResultUploadStarted, (), (override));
  MOCK_METHOD(void, LogResultUploadIOError,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_result_upload, absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogResultUploadClientInterrupted,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_result_upload, absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogResultUploadServerAborted,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_result_upload, absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogResultUploadCompleted,
              (NetworkStats stats, absl::Time time_before_result_upload,
               absl::Time reference_time),
              (override));
  MOCK_METHOD(absl::Status, LogFailureUploadStarted, (), (override));
  MOCK_METHOD(void, LogFailureUploadIOError,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_failure_upload,
               absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogFailureUploadClientInterrupted,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_failure_upload,
               absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogFailureUploadServerAborted,
              (absl::Status error_status, NetworkStats stats,
               absl::Time time_before_failure_upload,
               absl::Time reference_time),
              (override));
  MOCK_METHOD(void, LogFailureUploadCompleted,
              (NetworkStats stats, absl::Time time_before_result_upload,
               absl::Time reference_time),
              (override));
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_TEST_HELPERS_H_
