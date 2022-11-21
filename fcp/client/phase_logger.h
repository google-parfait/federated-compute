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
#include "fcp/client/stats.h"
#include "fcp/protos/federated_api.pb.h"

namespace fcp {
namespace client {

class PhaseLogger {
 public:
  virtual ~PhaseLogger() = default;
  virtual void UpdateRetryWindowAndNetworkStats(
      const ::google::internal::federatedml::v2::RetryWindow& retry_window,
      const NetworkStats& network_stats) = 0;
  virtual void SetModelIdentifier(absl::string_view model_identifier) = 0;

  // Called when a run was started but immediately aborted.
  virtual void LogTaskNotStarted(absl::string_view error_message) = 0;
  // Called when a run was started but the runtime failed to initialize a
  // noncritical component, and execution continue.
  virtual void LogNonfatalInitializationError(absl::Status error_status) = 0;
  // Called when a run was started but the runtime failed to initialize a
  // component, and execution was halted.
  virtual void LogFatalInitializationError(absl::Status error_status) = 0;

  // Eligibility eval check-in phase.
  // Called when an eligibility eval check-in starts.
  virtual void LogEligibilityEvalCheckinStarted() = 0;
  // Called when an IO error is encountered during eligibility eval check-in.
  virtual void LogEligibilityEvalCheckinIOError(
      absl::Status error_status, const NetworkStats& network_stats,
      absl::Time time_before_checkin) = 0;
  // Called when an invalid payload is received from the eligibility eval
  // check-in result.
  virtual void LogEligibilityEvalCheckinInvalidPayloadError(
      absl::string_view error_message, const NetworkStats& network_stats,
      absl::Time time_before_checkin) = 0;
  // Called when the eligibility eval check-in is interrupted by the client.
  virtual void LogEligibilityEvalCheckinClientInterrupted(
      absl::Status error_status, const NetworkStats& network_stats,
      absl::Time time_before_checkin) = 0;
  // Called when the eligibility eval check-in is aborted by the server.
  virtual void LogEligibilityEvalCheckinServerAborted(
      absl::Status error_status, const NetworkStats& network_stats,
      absl::Time time_before_checkin) = 0;
  // Called when eligibility eval is not configured.
  virtual void LogEligibilityEvalNotConfigured(
      const NetworkStats& network_stats, absl::Time time_before_checkin) = 0;
  // Called when eligibility eval check-in request is turned away by the server.
  virtual void LogEligibilityEvalCheckinTurnedAway(
      const NetworkStats& network_stats, absl::Time time_before_checkin) = 0;
  virtual void LogEligibilityEvalCheckinPlanUriReceived(
      const NetworkStats& network_stats, absl::Time time_before_checkin) = 0;
  // Called when a valid eligibility eval plan is received.
  virtual void LogEligibilityEvalCheckinCompleted(
      const NetworkStats& network_stats, absl::Time time_before_checkin,
      absl::Time time_before_plan_download) = 0;

  // Eligibility eval computation phase.
  // Called when the eligibility eval computation starts.
  virtual void LogEligibilityEvalComputationStarted() = 0;
  // Called when the input parameters for the eligibility eval task are invalid.
  virtual void LogEligibilityEvalComputationInvalidArgument(
      absl::Status error_status, const ExampleStats& example_stats,
      absl::Time run_plan_start_time) = 0;
  // Called when an example store error happened during eligibility eval
  // computation.
  virtual void LogEligibilityEvalComputationExampleIteratorError(
      absl::Status error_status, const ExampleStats& example_stats,
      absl::Time run_plan_start_time) = 0;
  // Called when a tensorflow error happened during eligibiliity eval
  // computation.
  virtual void LogEligibilityEvalComputationTensorflowError(
      absl::Status error_status, const ExampleStats& example_stats,
      absl::Time run_plan_start_time, absl::Time reference_time) = 0;
  // Called when the eligibility eval computation is interrupted.
  virtual void LogEligibilityEvalComputationInterrupted(
      absl::Status error_status, const ExampleStats& example_stats,
      absl::Time run_plan_start_time, absl::Time reference_time) = 0;
  // Called when the eligibility eval computation is completed.
  virtual void LogEligibilityEvalComputationCompleted(
      const ExampleStats& example_stats, absl::Time run_plan_start_time,
      absl::Time reference_time) = 0;

  // Check-in phase.
  // Called when a regular check-in starts.
  virtual void LogCheckinStarted() = 0;
  // Called when an IO error occurred during check-in.
  virtual void LogCheckinIOError(absl::Status error_status,
                                 const NetworkStats& network_stats,
                                 absl::Time time_before_checkin,
                                 absl::Time reference_time) = 0;
  // Called when an invalid payload is received from the check-in result.
  virtual void LogCheckinInvalidPayload(absl::string_view error_message,
                                        const NetworkStats& network_stats,
                                        absl::Time time_before_checkin,
                                        absl::Time reference_time) = 0;
  // Called when check-in is interrupted by the client.
  virtual void LogCheckinClientInterrupted(absl::Status error_status,
                                           const NetworkStats& network_stats,
                                           absl::Time time_before_checkin,
                                           absl::Time reference_time) = 0;
  // Called when check-in is aborted by the server.
  virtual void LogCheckinServerAborted(absl::Status error_status,
                                       const NetworkStats& network_stats,
                                       absl::Time time_before_checkin,
                                       absl::Time reference_time) = 0;
  // Called when the client's check-in request is turned away by the server.
  virtual void LogCheckinTurnedAway(const NetworkStats& network_stats,
                                    absl::Time time_before_checkin,
                                    absl::Time reference_time) = 0;
  virtual void LogCheckinPlanUriReceived(absl::string_view task_name,
                                         const NetworkStats& network_stats,
                                         absl::Time time_before_checkin) = 0;
  // Called when check-in is completed.
  virtual void LogCheckinCompleted(absl::string_view task_name,
                                   const NetworkStats& network_stats,
                                   absl::Time time_before_checkin,
                                   absl::Time time_before_plan_download,
                                   absl::Time reference_time) = 0;

  // Computation phase.
  // Called when computation started.
  virtual void LogComputationStarted() = 0;
  // Called when the input parameters are invalid.
  virtual void LogComputationInvalidArgument(
      absl::Status error_status, const ExampleStats& example_stats,
      const NetworkStats& network_stats, absl::Time run_plan_start_time) = 0;
  // Called when an example store error occurred during computation.
  virtual void LogComputationExampleIteratorError(
      absl::Status error_status, const ExampleStats& example_stats,
      const NetworkStats& network_stats, absl::Time run_plan_start_time) = 0;
  // Called when an IO error happened during computation
  virtual void LogComputationIOError(absl::Status error_status,
                                     const ExampleStats& example_stats,
                                     const NetworkStats& network_stats,
                                     absl::Time run_plan_start_time) = 0;
  // Called when a tensorflow error happened during computation.
  virtual void LogComputationTensorflowError(absl::Status error_status,
                                             const ExampleStats& example_stats,
                                             const NetworkStats& network_stats,
                                             absl::Time run_plan_start_time,
                                             absl::Time reference_time) = 0;
  // Called when computation is interrupted.
  virtual void LogComputationInterrupted(absl::Status error_status,
                                         const ExampleStats& example_stats,
                                         const NetworkStats& network_stats,
                                         absl::Time run_plan_start_time,
                                         absl::Time reference_time) = 0;
  // Called when computation is completed.
  virtual void LogComputationCompleted(const ExampleStats& example_stats,
                                       const NetworkStats& network_stats,
                                       absl::Time run_plan_start_time,
                                       absl::Time reference_time) = 0;

  // Result upload phase. Result upload only happens when all the previous
  // phases succeed.
  // Called when result upload started.
  virtual absl::Status LogResultUploadStarted() = 0;
  // Called when an IO error occurred during result upload.
  virtual void LogResultUploadIOError(absl::Status error_status,
                                      const NetworkStats& network_stats,
                                      absl::Time time_before_result_upload,
                                      absl::Time reference_time) = 0;
  // Called when the result upload is interrupted by the client.
  virtual void LogResultUploadClientInterrupted(
      absl::Status error_status, const NetworkStats& network_stats,
      absl::Time time_before_result_upload, absl::Time reference_time) = 0;
  // Called when the result upload is aborted by the server.
  virtual void LogResultUploadServerAborted(
      absl::Status error_status, const NetworkStats& network_stats,
      absl::Time time_before_result_upload, absl::Time reference_time) = 0;
  // Called when result upload is completed.
  virtual void LogResultUploadCompleted(const NetworkStats& network_stats,
                                        absl::Time time_before_result_upload,
                                        absl::Time reference_time) = 0;

  // Failure upload phase. Failure upload only happens when any of the previous
  // phases failed.
  // Called when failure upload starts.
  virtual absl::Status LogFailureUploadStarted() = 0;
  // Called when an IO error occurred during failure upload.
  virtual void LogFailureUploadIOError(absl::Status error_status,
                                       const NetworkStats& network_stats,
                                       absl::Time time_before_failure_upload,
                                       absl::Time reference_time) = 0;
  // Called when the failure upload is interrupted by the client.
  virtual void LogFailureUploadClientInterrupted(
      absl::Status error_status, const NetworkStats& network_stats,
      absl::Time time_before_failure_upload, absl::Time reference_time) = 0;
  // Called when the failure upload is aborted by the server.
  virtual void LogFailureUploadServerAborted(
      absl::Status error_status, const NetworkStats& network_stats,
      absl::Time time_before_failure_upload, absl::Time reference_time) = 0;
  // Called when the failure upload is completed.
  virtual void LogFailureUploadCompleted(const NetworkStats& network_stats,
                                         absl::Time time_before_failure_upload,
                                         absl::Time reference_time) = 0;
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_PHASE_LOGGER_H_
