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
#ifndef FCP_CLIENT_PHASE_LOGGER_IMPL_H_
#define FCP_CLIENT_PHASE_LOGGER_IMPL_H_

#include <string>

#include "absl/strings/string_view.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/flags.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/phase_logger.h"
#include "fcp/protos/federated_api.pb.h"

namespace fcp {
namespace client {

class PhaseLoggerImpl : public PhaseLogger {
 public:
  PhaseLoggerImpl(EventPublisher* event_publisher,
                  opstats::OpStatsLogger* opstats_logger,
                  LogManager* log_manager, const Flags* flags)
      : event_publisher_(event_publisher),
        opstats_logger_(opstats_logger),
        log_manager_(log_manager),
        use_per_phase_logs_(flags->per_phase_logs()),
        log_tensorflow_error_messages_(flags->log_tensorflow_error_messages()),
        granular_per_phase_logs_(flags->granular_per_phase_logs()) {}

  void UpdateRetryWindowAndNetworkStats(
      const ::google::internal::federatedml::v2::RetryWindow& retry_window,
      NetworkStats stats) override;
  void SetModelIdentifier(absl::string_view model_identifier) override;
  void LogTaskNotStarted(absl::string_view error_message) override;

  // Eligibility eval check-in phase.
  void LogEligibilityEvalCheckInStarted() override;
  void LogEligibilityEvalCheckInIOError(
      absl::Status error_status, NetworkStats stats,
      absl::Time time_before_eligibility_eval_checkin) override;
  void LogEligibilityEvalCheckInInvalidPayloadError(
      absl::string_view error_message, NetworkStats stats,
      absl::Time time_before_eligibility_eval_checkin) override;
  void LogEligibilityEvalCheckInClientInterrupted(
      absl::Status error_status, NetworkStats stats,
      absl::Time time_before_eligibility_eval_checkin) override;
  void LogEligibilityEvalCheckInServerAborted(
      absl::Status error_status, NetworkStats stats,
      absl::Time time_before_eligibility_eval_checkin) override;
  void LogEligibilityEvalNotConfigured(
      NetworkStats stats,
      absl::Time time_before_eligibility_eval_checkin) override;
  void LogEligibilityEvalCheckInTurnedAway(
      NetworkStats stats,
      absl::Time time_before_eligibility_eval_checkin) override;
  void LogEligibilityEvalCheckInCompleted(
      NetworkStats stats,
      absl::Time time_before_eligibility_eval_checkin) override;

  // Eligibility eval computation phase.
  void LogEligibilityEvalComputationStarted() override;
  void LogEligibilityEvalComputationInvalidArgument(
      absl::Status error_status, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Time run_plan_start_time) override;
  void LogEligibilityEvalComputationExampleIteratorError(
      absl::Status error_status, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Time run_plan_start_time) override;
  void LogEligibilityEvalComputationTensorflowError(
      absl::Status error_status, int total_example_count,
      int64_t total_example_size_bytes, absl::Time run_plan_start_time,
      absl::Time reference_time) override;
  void LogEligibilityEvalComputationInterrupted(
      absl::Status error_status, int total_example_count,
      int64_t total_example_size_bytes, absl::Time run_plan_start_time,
      absl::Time reference_time) override;
  void LogEligibilityEvalComputationCompleted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::Time run_plan_start_time, absl::Time reference_time) override;

  // Check-in phase.
  void LogCheckInStarted() override;
  void LogCheckInIOError(absl::Status error_status, NetworkStats stats,
                         absl::Time time_before_checkin,
                         absl::Time reference_time) override;
  void LogCheckInInvalidPayload(absl::string_view error_message,
                                NetworkStats stats,
                                absl::Time time_before_checkin,
                                absl::Time reference_time) override;
  void LogCheckInClientInterrupted(absl::Status error_status,
                                   NetworkStats stats,
                                   absl::Time time_before_checkin,
                                   absl::Time reference_time) override;
  void LogCheckInServerAborted(absl::Status error_status, NetworkStats stats,
                               absl::Time time_before_checkin,
                               absl::Time reference_time) override;
  void LogCheckInTurnedAway(NetworkStats stats, absl::Time time_before_checkin,
                            absl::Time reference_time) override;
  void LogCheckInCompleted(absl::string_view task_name, NetworkStats stats,
                           absl::Time time_before_checkin,
                           absl::Time reference_time) override;

  // Computation phase.
  void LogComputationStarted() override;
  void LogComputationInvalidArgument(absl::Status error_status,
                                     int total_example_count,
                                     int64_t total_example_size_bytes,
                                     absl::Time run_plan_start_time) override;
  void LogComputationExampleIteratorError(
      absl::Status error_status, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Time run_plan_start_time) override;
  void LogComputationIOError(absl::Status error_status, int total_example_count,
                             int64_t total_example_size_bytes,
                             absl::Time run_plan_start_time) override;
  void LogComputationTensorflowError(absl::Status error_status,
                                     int total_example_count,
                                     int64_t total_example_size_bytes,
                                     absl::Time run_plan_start_time,
                                     absl::Time reference_time) override;
  void LogComputationInterrupted(absl::Status error_status,
                                 int total_example_count,
                                 int64_t total_example_size_bytes,
                                 absl::Time run_plan_start_time,
                                 absl::Time reference_time) override;
  void LogComputationCompleted(int total_example_count,
                               int64_t total_example_size_bytes,
                               absl::Time run_plan_start_time,
                               absl::Time reference_time) override;

  absl::Status LogResultUploadStarted() override;
  void LogResultUploadIOError(absl::Status error_status, NetworkStats stats,
                              absl::Time time_before_result_upload,
                              absl::Time reference_time) override;
  void LogResultUploadClientInterrupted(absl::Status error_status,
                                        NetworkStats stats,
                                        absl::Time time_before_result_upload,
                                        absl::Time reference_time) override;
  void LogResultUploadServerAborted(absl::Status error_status,
                                    NetworkStats stats,
                                    absl::Time time_before_result_upload,
                                    absl::Time reference_time) override;
  void LogResultUploadCompleted(NetworkStats stats,
                                absl::Time time_before_result_upload,
                                absl::Time reference_time) override;

  // Failure upload phase.
  absl::Status LogFailureUploadStarted() override;
  void LogFailureUploadIOError(absl::Status error_status, NetworkStats stats,
                               absl::Time time_before_failure_upload,
                               absl::Time reference_time) override;
  void LogFailureUploadClientInterrupted(absl::Status error_status,
                                         NetworkStats stats,
                                         absl::Time time_before_failure_upload,
                                         absl::Time reference_time) override;
  void LogFailureUploadServerAborted(absl::Status error_status,
                                     NetworkStats stats,
                                     absl::Time time_before_failure_upload,
                                     absl::Time reference_time) override;
  void LogFailureUploadCompleted(NetworkStats stats,
                                 absl::Time time_before_failure_upload,
                                 absl::Time reference_time) override;

 private:
  void LogTimeSince(HistogramCounters histogram_counter,
                    absl::Time reference_time);
  void LogEligibilityEvalCheckInLatency(
      absl::Time time_before_eligibility_eval_checkin);
  void LogEligibilityEvalComputationLatency(absl::Time run_plan_start_time,
                                            absl::Time reference_time);
  void LogCheckInLatency(absl::Time time_before_checkin,
                         absl::Time reference_time);
  void LogComputationLatency(absl::Time run_plan_start_time,
                             absl::Time reference_time);
  void LogReportLatency(absl::Time time_before_report,
                        absl::Time reference_time);
  std::string GetErrorMessage(absl::Status error_status,
                              absl::string_view error_prefix,
                              bool keep_error_message);

  EventPublisher* event_publisher_;
  opstats::OpStatsLogger* opstats_logger_;
  LogManager* log_manager_;
  const bool use_per_phase_logs_;
  const bool log_tensorflow_error_messages_;
  const bool granular_per_phase_logs_;
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_PHASE_LOGGER_IMPL_H_
