/*
 * Copyright 2019 Google LLC
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
#ifndef FCP_CLIENT_EVENT_PUBLISHER_H_
#define FCP_CLIENT_EVENT_PUBLISHER_H_

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace fcp {
namespace client {

class SecAggEventPublisher;

// An interface for publishing events that occur during training. This is a
// separate interface from LogManager because the reported events will typically
// be both reported to a cloud monitoring backend and to the Federated server as
// part of publishing results.
// All methods in here either succeed with OK, or fail with INVALID_ARGUMENT.
class EventPublisher {
 public:
  virtual ~EventPublisher() = default;

  // Publishes that the device is about to issue an eligibility eval check in
  // with the server.
  virtual void PublishEligibilityEvalCheckin() = 0;

  // Publishes that the device has finished its eligibility eval checkin with
  // the server, and received an eligibility eval plan, along with information
  // how much data was downloaded and how long that took.
  virtual void PublishEligibilityEvalPlanReceived(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_downloaded,
      absl::Duration download_duration) = 0;

  // Publishes that the server did not return an eligibility eval task to the
  // client, along with information how much data was downloaded and how long
  // that took.
  virtual void PublishEligibilityEvalNotConfigured(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_downloaded,
      absl::Duration download_duration) = 0;

  // Publishes that the server rejected the device's eligibility eval checkin,
  // along with information how much data was downloaded and how long that took.
  virtual void PublishEligibilityEvalRejected(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_downloaded,
      absl::Duration download_duration) = 0;

  // Publishes that the device is about to check in with the server.
  virtual void PublishCheckin() = 0;

  // Publishes that the device has finished checking in with the server, along
  // with information how much data was downloaded and how long that took.
  virtual void PublishCheckinFinished(int64_t bytes_downloaded,
                                      int64_t chunking_layer_bytes_downloaded,
                                      absl::Duration download_duration) = 0;

  // Publishes that the server rejected the device.
  virtual void PublishRejected() = 0;

  // Publishes that the device is about to report the results of a federated
  // computation to the server.
  virtual void PublishReportStarted(int64_t report_size_bytes) = 0;

  // Publishes that the device has successfully reported its results to the
  // server and received instructions on when to reconnect.
  virtual void PublishReportFinished(int64_t report_size_bytes,
                                     int64_t chunking_layer_bytes_sent,
                                     absl::Duration report_duration) = 0;

  // Publishes that plan execution has started.
  virtual void PublishPlanExecutionStarted() = 0;

  // Publishes that an epoch has started.
  virtual void PublishEpochStarted(int execution_index, int epoch_index) = 0;

  // Publishes a TensorFlow error that happened in the given ClientExecution.
  virtual void PublishTensorFlowError(int execution_index, int epoch_index,
                                      int epoch_example_index,
                                      absl::string_view error_message) = 0;

  // Publishes an I/O error (e.g. disk, network) that happened in the given
  // ClientExecution.
  virtual void PublishIoError(int execution_index,
                              absl::string_view error_message) = 0;

  // Publishes an ExampleSelector error from the given ClientExecution.
  virtual void PublishExampleSelectorError(int execution_index, int epoch_index,
                                           int epoch_example_index,
                                           absl::string_view error_message) = 0;

  // Publishes an interruption event for the given client execution.
  virtual void PublishInterruption(int execution_index, int epoch_index,
                                   int epoch_example_index,
                                   int64_t total_example_size_bytes,
                                   absl::Time start_time) = 0;

  // Publishes a completion event for the current epoch.
  virtual void PublishEpochCompleted(int execution_index, int epoch_index,
                                     int epoch_example_index,
                                     int64_t epoch_example_size_bytes,
                                     absl::Time epoch_start_time) = 0;

  // Publishes model statistics.
  virtual void PublishStats(
      int execution_index, int epoch_index,
      const absl::flat_hash_map<std::string, double>& stats) = 0;

  // Publishes an event that plan execution is complete.
  virtual void PublishPlanCompleted(int total_example_count,
                                    int64_t total_example_size_bytes,
                                    absl::Time start_time) = 0;
  // Publishes that the task didn't start.
  virtual void PublishTaskNotStarted(absl::string_view error_message) = 0;

  // Publish that an IO error was encountered during eligibility eval check-in.
  virtual void PublishEligibilityEvalCheckInIoError(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message, absl::Duration download_duration) = 0;
  // Publish that the eligibility eval check-in is interrupted by the client.
  virtual void PublishEligibilityEvalCheckInClientInterrupted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message, absl::Duration download_duration) = 0;
  // Publish that the eligibility eval check-in is aborted by the server.
  virtual void PublishEligibilityEvalCheckInServerAborted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message, absl::Duration download_duration) = 0;
  // Publish that the eligibility eval check-in returned an invalid payload.
  virtual void PublishEligibilityEvalCheckInErrorInvalidPayload(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message, absl::Duration download_duration) = 0;
  // Publish an eligibility eval task starts computation.
  virtual void PublishEligibilityEvalComputationStarted() = 0;
  // Publish that the eligibility eval task is invalid.
  virtual void PublishEligibilityEvalComputationInvalidArgument(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) = 0;
  // Publish an example iterator error occurred during eligibility eval task.
  virtual void PublishEligibilityEvalComputationExampleIteratorError(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) = 0;
  // Publish that a tensorflow error occurred during eligibility eval task.
  virtual void PublishEligibilityEvalComputationTensorflowError(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message, absl::Duration computation_duration) = 0;
  // Publish that the client has interrupted the eligibility eval task.
  virtual void PublishEligibilityEvalComputationInterrupted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message, absl::Duration computation_duration) = 0;
  // Publish an eligibility eval task finished.
  virtual void PublishEligibilityEvalComputationCompleted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::Duration computation_duration) = 0;
  // Publish an IO error occurred during regular check-in.
  virtual void PublishCheckinIoError(int64_t bytes_downloaded,
                                     int64_t chunking_layer_bytes_received,
                                     absl::string_view error_message,
                                     absl::Duration download_duration) = 0;
  // Publish that the client interrupted the regular check-in.
  virtual void PublishCheckinClientInterrupted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message, absl::Duration download_duration) = 0;
  // Publish that the server aborted the regular check-in.
  virtual void PublishCheckinServerAborted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message, absl::Duration download_duration) = 0;
  // Publish that an invalid payload was downloaded from the regular check-in.
  virtual void PublishCheckinInvalidPayload(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message, absl::Duration download_duration) = 0;
  // Publishes that the server rejected the device, also logs network stats and
  // duration.
  virtual void PublishRejected(int64_t bytes_downloaded,
                               int64_t chunking_layer_bytes_downloaded,
                               absl::Duration download_duration) = 0;

  // Publishes that the device has finished checking in with the server, along
  // with information how much data was downloaded and how long that took.
  virtual void PublishCheckinFinishedV2(int64_t bytes_downloaded,
                                        int64_t chunking_layer_bytes_downloaded,
                                        absl::Duration download_duration) = 0;
  // Publishes that plan execution has started.
  virtual void PublishComputationStarted() = 0;
  // Publish that the task is invalid.
  virtual void PublishComputationInvalidArgument(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) = 0;
  // Publish that an IO error occurred during computation.
  virtual void PublishComputationIOError(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) = 0;
  // Publish that an example iterator error occurred during computation.
  virtual void PublishComputationExampleIteratorError(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) = 0;
  // Publish that an tensorflow error occurred during computation.
  virtual void PublishComputationTensorflowError(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message, absl::Duration computation_duration) = 0;
  // Publish that the task computation is interrupted.
  virtual void PublishComputationInterrupted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message, absl::Duration computation_duration) = 0;
  // Publishes an event that plan execution is complete.
  virtual void PublishComputationCompleted(int total_example_count,
                                           int64_t total_example_size_bytes,
                                           absl::Time start_time) = 0;
  // Publish that the client starts to upload result.
  virtual void PublishResultUploadStarted() = 0;
  // Publish that an IO error occurred during result upload.
  virtual void PublishResultUploadIOError(int64_t report_size_bytes,
                                          int64_t chunking_layer_bytes_sent,
                                          absl::string_view error_message,
                                          absl::Duration upload_duration) = 0;
  // Publish that the client has interrupted the result upload.
  virtual void PublishResultUploadClientInterrupted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message, absl::Duration upload_duration) = 0;
  // Publish hat the server has aborted the result upload.
  virtual void PublishResultUploadServerAborted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message, absl::Duration upload_duration) = 0;
  // Publish that the result upload is completed.
  virtual void PublishResultUploadCompleted(int64_t report_size_bytes,
                                            int64_t chunking_layer_bytes_sent,
                                            absl::Duration upload_duration) = 0;
  // Publish that the task computation has failed, and the client starts to
  // upload the failure to the server.
  virtual void PublishFailureUploadStarted() = 0;
  // Publish that an IO error occurred during failure upload.
  virtual void PublishFailureUploadIOError(int64_t report_size_bytes,
                                           int64_t chunking_layer_bytes_sent,
                                           absl::string_view error_message,
                                           absl::Duration upload_duration) = 0;
  // Publish that the client has interrupted the failure upload.
  virtual void PublishFailureUploadClientInterrupted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message, absl::Duration upload_duration) = 0;
  // Publish that the server has aborted the failure upload.
  virtual void PublishFailureUploadServerAborted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message, absl::Duration upload_duration) = 0;
  // Publish that the failure upload completed.
  virtual void PublishFailureUploadCompleted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::Duration upload_duration) = 0;

  // After calling this function, all subsequently published events will be
  // annotated with the specified model_identifier. This value is typically
  // provided by the federated server and used on events resulting from
  // PublishEligibilityEvalCheckinFinished(), PublishCheckinFinished() and
  // later.
  //
  // Note that this method may be called multiple times with different values,
  // if over the course of a training session multiple models are executed.
  virtual void SetModelIdentifier(const std::string& model_identifier) = 0;

  // Returns a pointer to a publisher which records secure aggregation protocol
  // events.  The returned value must not be nullptr.
  virtual SecAggEventPublisher* secagg_event_publisher() = 0;
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_EVENT_PUBLISHER_H_
