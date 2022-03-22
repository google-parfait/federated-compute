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

#include "fcp/base/monitoring.h"
#include "fcp/client/event_publisher.h"

namespace fcp {
namespace client {

class SecAggEventPublisher;

class FakeEventPublisher : public EventPublisher {
 public:
  void PublishEligibilityEvalCheckin() override {}

  void PublishEligibilityEvalPlanReceived(int64_t, int64_t,
                                          absl::Duration) override {}

  void PublishEligibilityEvalNotConfigured(int64_t, int64_t,
                                           absl::Duration) override {}

  void PublishEligibilityEvalRejected(int64_t, int64_t,
                                      absl::Duration) override {}

  void PublishCheckin() override {}

  void PublishCheckinFinished(int64_t bytes_downloaded,
                              int64_t chunking_layer_bytes_downloaded,
                              absl::Duration download_duration) override {}

  void PublishRejected() override {}

  void PublishReportStarted(int64_t report_size_bytes) override {}

  void PublishReportFinished(int64_t report_size_bytes,
                             int64_t chunking_layer_bytes_sent,
                             absl::Duration report_duration) override {}

  void PublishPlanExecutionStarted() override {}

  void PublishEpochStarted(int execution_index, int epoch_index) override {}

  void PublishTensorFlowError(int execution_index, int epoch_index,
                              int epoch_example_index,
                              absl::string_view error_message) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishIoError(int execution_index,
                      absl::string_view error_message) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishExampleSelectorError(int execution_index, int epoch_index,
                                   int epoch_example_index,
                                   absl::string_view error_message) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishInterruption(int execution_index, int epoch_index,
                           int epoch_example_index,
                           int64_t total_example_size_bytes,
                           absl::Time start_time) override {}

  void PublishEpochCompleted(int execution_index, int epoch_index,
                             int epoch_example_index,
                             int64_t epoch_example_size_bytes,
                             absl::Time epoch_start_time) override {}

  void PublishStats(
      int execution_index, int epoch_index,
      const absl::flat_hash_map<std::string, double>& stats) override {}

  void PublishPlanCompleted(int total_example_count,
                            int64_t total_example_size_bytes,
                            absl::Time start_time) override {}

  void PublishTaskNotStarted(absl::string_view error_message) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishEligibilityEvalCheckInIoError(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishEligibilityEvalCheckInClientInterrupted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishEligibilityEvalCheckInServerAborted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishEligibilityEvalCheckInErrorInvalidPayload(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishEligibilityEvalComputationStarted() override {}

  void PublishEligibilityEvalComputationInvalidArgument(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishEligibilityEvalComputationExampleIteratorError(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishEligibilityEvalComputationTensorflowError(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishEligibilityEvalComputationInterrupted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishEligibilityEvalComputationCompleted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {}

  void PublishCheckinIoError(int64_t bytes_downloaded,
                             int64_t chunking_layer_bytes_received,
                             absl::string_view error_message,
                             absl::Duration download_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishCheckinClientInterrupted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishCheckinServerAborted(int64_t bytes_downloaded,
                                   int64_t chunking_layer_bytes_received,
                                   absl::string_view error_message,
                                   absl::Duration download_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishCheckinInvalidPayload(int64_t bytes_downloaded,
                                    int64_t chunking_layer_bytes_received,
                                    absl::string_view error_message,
                                    absl::Duration download_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishRejected(int64_t bytes_downloaded,
                       int64_t chunking_layer_bytes_downloaded,
                       absl::Duration download_duration) override {}

  void PublishCheckinFinishedV2(int64_t bytes_downloaded,
                                int64_t chunking_layer_bytes_downloaded,
                                absl::Duration download_duration) override {}

  void PublishComputationInvalidArgument(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishComputationIOError(absl::string_view error_message,
                                 int total_example_count,
                                 int64_t total_example_size_bytes,
                                 absl::Duration computation_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishComputationExampleIteratorError(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishComputationTensorflowError(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishComputationInterrupted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishResultUploadStarted() override {}

  void PublishResultUploadIOError(int64_t report_size_bytes,
                                  int64_t chunking_layer_bytes_sent,
                                  absl::string_view error_message,
                                  absl::Duration upload_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishResultUploadClientInterrupted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishResultUploadServerAborted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishResultUploadCompleted(int64_t report_size_bytes,
                                    int64_t chunking_layer_bytes_sent,
                                    absl::Duration upload_duration) override {}

  void PublishFailureUploadStarted() override {}

  void PublishFailureUploadIOError(int64_t report_size_bytes,
                                   int64_t chunking_layer_bytes_sent,
                                   absl::string_view error_message,
                                   absl::Duration upload_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishFailureUploadClientInterrupted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishFailureUploadServerAborted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_LOG(ERROR) << error_message;
  }

  void PublishFailureUploadCompleted(int64_t report_size_bytes,
                                     int64_t chunking_layer_bytes_snet,
                                     absl::Duration upload_duration) override {}

  void SetModelIdentifier(const std::string& model_identifier) override {}

  SecAggEventPublisher* secagg_event_publisher() override { return nullptr; }
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_FAKE_EVENT_PUBLISHER_H_
