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

#ifndef FCP_CLIENT_FAKE_EVENT_PUBLISHER_H_
#define FCP_CLIENT_FAKE_EVENT_PUBLISHER_H_

#include <string>

#include "absl/strings/str_split.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/secagg_event_publisher.h"

namespace fcp {
namespace client {

class SecAggEventPublisher;

// Macro to print log messages prefixed by ClassName::FunctionName, stripping
// namespaces before ClassName, if any.
#define FCP_CLIENT_LOG_FUNCTION_NAME                              \
  std::string _demangle_buf(1024, '\0');                          \
  size_t _demangle_buf_len = _demangle_buf.length();              \
  abi::__cxa_demangle(typeid(*this).name(), _demangle_buf.data(), \
                      &_demangle_buf_len, nullptr);               \
  FCP_LOG(INFO) << static_cast<std::vector<std::string>>(         \
                       absl::StrSplit(_demangle_buf, "::"))       \
                       .back()                                    \
                       .c_str()                                   \
                << "::" << __func__

// An implementation of the SecAggEventPublisher interface that logs calls to
// stderr.
class SecAggLoggingEventPublisher : public SecAggEventPublisher {
 public:
  explicit SecAggLoggingEventPublisher(bool quiet) : quiet_(quiet) {}

  void PublishStateTransition(::fcp::secagg::ClientState state,
                              size_t last_sent_message_size,
                              size_t last_received_message_size) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }
  void PublishError() override { FCP_CLIENT_LOG_FUNCTION_NAME; }
  void PublishAbort(bool client_initiated,
                    const std::string& error_message) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }
  void set_execution_session_id(int64_t execution_session_id) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

 private:
  const bool quiet_;
};

// An implementation of the EventPublisher interface that logs calls to stderr.
class FakeEventPublisher : public EventPublisher {
 public:
  // Logs all events to stderr.
  FakeEventPublisher() : FakeEventPublisher(/*quiet=*/false) {}
  // Logs only error and "client rejected" events to stderr.
  explicit FakeEventPublisher(bool quiet)
      : quiet_(quiet), secagg_event_publisher_(quiet) {}

  void PublishEligibilityEvalCheckin() override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishEligibilityEvalPlanReceived(int64_t, int64_t,
                                          absl::Duration) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishEligibilityEvalNotConfigured(int64_t, int64_t,
                                           absl::Duration) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishEligibilityEvalRejected(int64_t, int64_t,
                                      absl::Duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishCheckin() override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishCheckinFinished(int64_t, int64_t, absl::Duration) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishRejected() override { FCP_CLIENT_LOG_FUNCTION_NAME; }

  void PublishReportStarted(int64_t) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishReportFinished(int64_t, int64_t, absl::Duration) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishPlanExecutionStarted() override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishEpochStarted(int, int) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishTensorFlowError(int execution_index, int epoch_index, int,
                              absl::string_view error_message) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << "exec " << execution_index << ", epoch "
                                 << epoch_index << ": " << error_message;
  }

  void PublishIoError(int execution_index,
                      absl::string_view error_message) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << "exec " << execution_index << ": "
                                 << error_message;
  }

  void PublishExampleSelectorError(int execution_index, int epoch_index, int,
                                   absl::string_view error_message) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << "exec " << execution_index << ", epoch "
                                 << epoch_index << ": " << error_message;
  }

  void PublishInterruption(int, int, int, int64_t, absl::Time) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishEpochCompleted(int, int, int, int64_t, absl::Time) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishPlanCompleted(int total_example_count, int64_t,
                            absl::Time) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME << ":\ttotal_example_count="
                                 << total_example_count;
  }

  void PublishTaskNotStarted(absl::string_view error_message) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalCheckInIoError(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalCheckInClientInterrupted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalCheckInServerAborted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalCheckInErrorInvalidPayload(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalComputationStarted() override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishEligibilityEvalComputationInvalidArgument(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalComputationExampleIteratorError(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalComputationTensorflowError(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalComputationInterrupted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalComputationCompleted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishCheckinIoError(int64_t bytes_downloaded,
                             int64_t chunking_layer_bytes_received,
                             absl::string_view error_message,
                             absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishCheckinClientInterrupted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishCheckinServerAborted(int64_t bytes_downloaded,
                                   int64_t chunking_layer_bytes_received,
                                   absl::string_view error_message,
                                   absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishCheckinInvalidPayload(int64_t bytes_downloaded,
                                    int64_t chunking_layer_bytes_received,
                                    absl::string_view error_message,
                                    absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishRejected(int64_t bytes_downloaded,
                       int64_t chunking_layer_bytes_downloaded,
                       absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishCheckinFinishedV2(int64_t, int64_t, absl::Duration) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishComputationStarted() override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishComputationInvalidArgument(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishComputationIOError(absl::string_view error_message,
                                 int total_example_count,
                                 int64_t total_example_size_bytes,
                                 absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishComputationExampleIteratorError(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishComputationTensorflowError(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishComputationInterrupted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishComputationCompleted(int total_example_count, int64_t,
                                   absl::Time) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME << ":\ttotal_example_count="
                                 << total_example_count;
  }

  void PublishResultUploadStarted() override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishResultUploadIOError(int64_t report_size_bytes,
                                  int64_t chunking_layer_bytes_sent,
                                  absl::string_view error_message,
                                  absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishResultUploadClientInterrupted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishResultUploadServerAborted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishResultUploadCompleted(int64_t report_size_bytes,
                                    int64_t chunking_layer_bytes_sent,
                                    absl::Duration upload_duration) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishFailureUploadStarted() override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishFailureUploadIOError(int64_t report_size_bytes,
                                   int64_t chunking_layer_bytes_sent,
                                   absl::string_view error_message,
                                   absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishFailureUploadClientInterrupted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishFailureUploadServerAborted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishFailureUploadCompleted(int64_t report_size_bytes,
                                     int64_t chunking_layer_bytes_sent,
                                     absl::Duration upload_duration) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void SetModelIdentifier(const std::string& model_identifier) override {
    if (quiet_) return;
    FCP_CLIENT_LOG_FUNCTION_NAME << ":\n\t" << model_identifier;
  }

  SecAggEventPublisher* secagg_event_publisher() override {
    return &secagg_event_publisher_;
  }

 private:
  const bool quiet_;
  SecAggLoggingEventPublisher secagg_event_publisher_;
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_FAKE_EVENT_PUBLISHER_H_
