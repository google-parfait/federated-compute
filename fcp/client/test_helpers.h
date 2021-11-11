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

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/flags.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_db.h"
#include "fcp/client/opstats/opstats_logger.h"
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
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_downloaded,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalNotConfigured,
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_downloaded,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishEligibilityEvalRejected,
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_downloaded,
               absl::Duration download_duration),
              (override));
  MOCK_METHOD(void, PublishCheckin, (), (override));
  MOCK_METHOD(void, PublishCheckinFinished,
              (int64_t bytes_downloaded, int64_t chunking_layer_bytes_downloaded,
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

  SecAggEventPublisher* secagg_event_publisher() override {
    return &secagg_event_publisher_;
  }

 private:
  ::testing::NiceMock<MockSecAggEventPublisher> secagg_event_publisher_;
};

class MockFederatedProtocol : public FederatedProtocol {
 public:
  explicit MockFederatedProtocol() {}
  MOCK_METHOD(absl::StatusOr<EligibilityEvalCheckinResult>,
              EligibilityEvalCheckin, (), (override));
  MOCK_METHOD(absl::StatusOr<CheckinResult>, Checkin,
              (const absl::optional<
                  ::google::internal::federatedml::v2::TaskEligibilityInfo>&
                   task_eligibility_info),
              (override));
  MOCK_METHOD(::google::internal::federatedml::v2::RetryWindow,
              GetLatestRetryWindow, (), (override));

  MOCK_METHOD(absl::Status, ReportCompleted,
              (ComputationResults results,
               (const std::vector<std::pair<std::string, double>>& stats),
               absl::Duration plan_duration),
              (override));

  MOCK_METHOD(absl::Status, ReportNotCompleted,
              (engine::PhaseOutcome phase_outcome,
               absl::Duration plan_duration),
              (override));
  MOCK_METHOD(int64_t, chunking_layer_bytes_sent, (), (override));
  MOCK_METHOD(int64_t, chunking_layer_bytes_received, (), (override));
  MOCK_METHOD(int64_t, bytes_downloaded, (), (override));
  MOCK_METHOD(int64_t, bytes_uploaded, (), (override));
  MOCK_METHOD(int64_t, report_request_size_bytes, (), (override));
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

struct FlArtifacts {
  google::internal::federated::plan::ClientOnlyPlan plan;
  google::internal::federated::plan::Dataset dataset;
  std::string checkpoint;
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
  MOCK_METHOD(bool, opstats_enforce_singleton, (), (const, override));
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
  MOCK_METHOD(std::vector<int32_t>, federated_training_permanent_error_codes, (),
              (const, override));
  MOCK_METHOD(bool, commit_opstats_on_upload_started, (), (const, override));
  MOCK_METHOD(bool, record_earliest_trustworthy_time_for_opstats, (),
              (const, override));
  MOCK_METHOD(bool, per_phase_logs, (), (const, override));
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

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_TEST_HELPERS_H_
