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
#ifndef FCP_CLIENT_PHASE_LOGGER_H_
#define FCP_CLIENT_PHASE_LOGGER_H_

#include "absl/strings/string_view.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/protos/federated_api.pb.h"

namespace fcp {
namespace client {

struct NetworkStats {
  int64_t bytes_downloaded = 0;
  int64_t bytes_uploaded = 0;
  int64_t chunking_layer_bytes_received = 0;
  int64_t chunking_layer_bytes_sent = 0;
  int64_t report_size_bytes = 0;
};

class PhaseLogger {
 public:
  virtual ~PhaseLogger() = default;
  virtual void UpdateRetryWindowAndNetworkStats(
      const ::google::internal::federatedml::v2::RetryWindow& retry_window,
      NetworkStats stats) = 0;
  virtual void SetModelIdentifier(absl::string_view model_identifier) = 0;

  // Called when a run was started but immediately aborted.
  virtual void LogTaskNotStarted(absl::string_view error_message) = 0;

  // Eligibility eval check-in phase.
  // Called when an eligibility eval check-in starts.
  virtual void LogEligibilityEvalCheckInStarted() = 0;
  // Called when an IO error is encountered during eligibility eval check-in.
  virtual void LogEligibilityEvalCheckInIOError(
      absl::Status error_status, NetworkStats stats,
      absl::Time time_before_eligibility_eval_checkin) = 0;
  // Called when an invalid payload is received from the eligibility eval
  // check-in result.
  virtual void LogEligibilityEvalCheckInInvalidPayloadError(
      absl::string_view error_message, NetworkStats stats,
      absl::Time time_before_eligibility_eval_checkin) = 0;
  // Called when the eligibility eval check-in is interrupted by the client.
  virtual void LogEligibilityEvalCheckInClientInterrupted(
      absl::Status error_status, NetworkStats stats,
      absl::Time time_before_eligibility_eval_checkin) = 0;
  // Called when the eligibility eval check-in is aborted by the server.
  virtual void LogEligibilityEvalCheckInServerAborted(
      absl::Status error_status, NetworkStats stats,
      absl::Time time_before_eligibility_eval_checkin) = 0;
  // Called when eligibility eval is not configured.
  virtual void LogEligibilityEvalNotConfigured(
      NetworkStats stats, absl::Time time_before_eligibility_eval_checkin) = 0;
  // Called when eligibility eval check-in request is turned away by the server.
  virtual void LogEligibilityEvalCheckInTurnedAway(
      NetworkStats stats, absl::Time time_before_eligibility_eval_checkin) = 0;
  // Called when a valid eligibility eval plan is received.
  virtual void LogEligibilityEvalCheckInCompleted(
      NetworkStats stats, absl::Time time_before_eligibility_eval_checkin) = 0;

  // Eligibility eval computation phase.
  // Called when the eligibility eval computation starts.
  virtual void LogEligibilityEvalComputationStarted() = 0;
  // Called when the input parameters for the eligibility eval task are invalid.
  virtual void LogEligibilityEvalComputationInvalidArgument(
      absl::Status error_status, int total_example_count,
      int64_t total_example_size_bytes, absl::Time run_plan_start_time) = 0;
  // Called when an example store error happened during eligibility eval
  // computation.
  virtual void LogEligibilityEvalComputationExampleIteratorError(
      absl::Status error_status, int total_example_count,
      int64_t total_example_size_bytes, absl::Time run_plan_start_time) = 0;
  // Called when a tensorflow error happened during eligibiliity eval
  // computation.
  virtual void LogEligibilityEvalComputationTensorflowError(
      absl::Status error_status, int total_example_count,
      int64_t total_example_size_bytes, absl::Time run_plan_start_time,
      absl::Time reference_time) = 0;
  // Called when the eligibility eval computation is interrupted.
  virtual void LogEligibilityEvalComputationInterrupted(
      absl::Status error_status, int total_example_count,
      int64_t total_example_size_bytes, absl::Time run_plan_start_time,
      absl::Time reference_time) = 0;
  // Called when the eligibility eval computation is completed.
  virtual void LogEligibilityEvalComputationCompleted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::Time run_plan_start_time, absl::Time reference_time) = 0;

  // Check-in phase.
  // Called when a regular check-in starts.
  virtual void LogCheckInStarted() = 0;
  // Called when an IO error occurred during check-in.
  virtual void LogCheckInIOError(absl::Status error_status, NetworkStats stats,
                                 absl::Time time_before_checkin,
                                 absl::Time reference_time) = 0;
  // Called when an invalid payload is received from the check-in result.
  virtual void LogCheckInInvalidPayload(absl::string_view error_message,
                                        NetworkStats stats,
                                        absl::Time time_before_checkin,
                                        absl::Time reference_time) = 0;
  // Called when check-in is interrupted by the client.
  virtual void LogCheckInClientInterrupted(absl::Status error_status,
                                           NetworkStats stats,
                                           absl::Time time_before_checkin,
                                           absl::Time reference_time) = 0;
  // Called when check-in is aborted by the server.
  virtual void LogCheckInServerAborted(absl::Status error_status,
                                       NetworkStats stats,
                                       absl::Time time_before_checkin,
                                       absl::Time reference_time) = 0;
  // Called when the client's check-in request is turned away by the server.
  virtual void LogCheckInTurnedAway(NetworkStats stats,
                                    absl::Time time_before_checkin,
                                    absl::Time reference_time) = 0;
  // Called when check-in is completed.
  virtual void LogCheckInCompleted(absl::string_view task_name,
                                   NetworkStats stats,
                                   absl::Time time_before_checkin,
                                   absl::Time reference_time) = 0;

  // Computation phase.
  // Called when computation started.
  virtual void LogComputationStarted() = 0;
  // Called when the input parameters are invalid.
  virtual void LogComputationInvalidArgument(
      absl::Status error_status, int total_example_count,
      int64_t total_example_size_bytes, absl::Time run_plan_start_time) = 0;
  // Called when an example store error occurred during computation.
  virtual void LogComputationExampleIteratorError(
      absl::Status error_status, int total_example_count,
      int64_t total_example_size_bytes, absl::Time run_plan_start_time) = 0;
  // Called when an IO error happened during computation
  virtual void LogComputationIOError(absl::Status error_status,
                                     int total_example_count,
                                     int64_t total_example_size_bytes,
                                     absl::Time run_plan_start_time) = 0;
  // Called when a tensorflow error happened during computation.
  virtual void LogComputationTensorflowError(absl::Status error_status,
                                             int total_example_count,
                                             int64_t total_example_size_bytes,
                                             absl::Time run_plan_start_time,
                                             absl::Time reference_time) = 0;
  // Called when computation is interrupted.
  virtual void LogComputationInterrupted(absl::Status error_status,
                                         int total_example_count,
                                         int64_t total_example_size_bytes,
                                         absl::Time run_plan_start_time,
                                         absl::Time reference_time) = 0;
  // Called when computation is completed.
  virtual void LogComputationCompleted(int total_example_count,
                                       int64_t total_example_size_bytes,
                                       absl::Time run_plan_start_time,
                                       absl::Time reference_time) = 0;

  // Result upload phase. Result upload only happens when all the previous
  // phases succeed.
  // Called when result upload started.
  virtual absl::Status LogResultUploadStarted() = 0;
  // Called when an IO error occurred during result upload.
  virtual void LogResultUploadIOError(absl::Status error_status,
                                      NetworkStats stats,
                                      absl::Time time_before_result_upload,
                                      absl::Time reference_time) = 0;
  // Called when the result upload is interrupted by the client.
  virtual void LogResultUploadClientInterrupted(
      absl::Status error_status, NetworkStats stats,
      absl::Time time_before_result_upload, absl::Time reference_time) = 0;
  // Called when the result upload is aborted by the server.
  virtual void LogResultUploadServerAborted(
      absl::Status error_status, NetworkStats stats,
      absl::Time time_before_result_upload, absl::Time reference_time) = 0;
  // Called when result upload is completed.
  virtual void LogResultUploadCompleted(NetworkStats stats,
                                        absl::Time time_before_result_upload,
                                        absl::Time reference_time) = 0;

  // Failure upload phase. Failure upload only happens when any of the previous
  // phases failed.
  // Called when failure upload starts.
  virtual absl::Status LogFailureUploadStarted() = 0;
  // Called when an IO error occurred during failure upload.
  virtual void LogFailureUploadIOError(absl::Status error_status,
                                       NetworkStats stats,
                                       absl::Time time_before_failure_upload,
                                       absl::Time reference_time) = 0;
  // Called when the failure upload is interrupted by the client.
  virtual void LogFailureUploadClientInterrupted(
      absl::Status error_status, NetworkStats stats,
      absl::Time time_before_failure_upload, absl::Time reference_time) = 0;
  // Called when the failure upload is aborted by the server.
  virtual void LogFailureUploadServerAborted(
      absl::Status error_status, NetworkStats stats,
      absl::Time time_before_failure_upload, absl::Time reference_time) = 0;
  // Called when the failure upload is completed.
  virtual void LogFailureUploadCompleted(NetworkStats stats,
                                         absl::Time time_before_failure_upload,
                                         absl::Time reference_time) = 0;
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_PHASE_LOGGER_H_
