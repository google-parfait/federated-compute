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
