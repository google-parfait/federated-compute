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
        log_tensorflow_error_messages_(flags->log_tensorflow_error_messages()) {
  }

  void UpdateRetryWindowAndNetworkStats(
      const ::google::internal::federatedml::v2::RetryWindow& retry_window,
      const NetworkStats& network_stats) override;
  void SetModelIdentifier(absl::string_view model_identifier) override;
  void LogTaskNotStarted(absl::string_view error_message) override;
  void LogNonfatalInitializationError(absl::Status error_status) override;
  void LogFatalInitializationError(absl::Status error_status) override;

  // Eligibility eval check-in phase.
  void LogEligibilityEvalCheckinStarted() override;
  void LogEligibilityEvalCheckinIOError(
      absl::Status error_status, const NetworkStats& network_stats,
      absl::Time time_before_checkin) override;
  void LogEligibilityEvalCheckinInvalidPayloadError(
      absl::string_view error_message, const NetworkStats& network_stats,
      absl::Time time_before_checkin) override;
  void LogEligibilityEvalCheckinClientInterrupted(
      absl::Status error_status, const NetworkStats& network_stats,
      absl::Time time_before_checkin) override;
  void LogEligibilityEvalCheckinServerAborted(
      absl::Status error_status, const NetworkStats& network_stats,
      absl::Time time_before_checkin) override;
  void LogEligibilityEvalNotConfigured(const NetworkStats& network_stats,
                                       absl::Time time_before_checkin) override;
  void LogEligibilityEvalCheckinTurnedAway(
      const NetworkStats& network_stats,
      absl::Time time_before_checkin) override;
  void LogEligibilityEvalCheckinPlanUriReceived(
      const NetworkStats& network_stats,
      absl::Time time_before_checkin) override;
  void LogEligibilityEvalCheckinCompleted(
      const NetworkStats& network_stats, absl::Time time_before_checkin,
      absl::Time time_before_plan_download) override;

  // Eligibility eval computation phase.
  void LogEligibilityEvalComputationStarted() override;
  void LogEligibilityEvalComputationInvalidArgument(
      absl::Status error_status, const ExampleStats& example_stats,
      absl::Time run_plan_start_time) override;
  void LogEligibilityEvalComputationExampleIteratorError(
      absl::Status error_status, const ExampleStats& example_stats,
      absl::Time run_plan_start_time) override;
  void LogEligibilityEvalComputationTensorflowError(
      absl::Status error_status, const ExampleStats& example_stats,
      absl::Time run_plan_start_time, absl::Time reference_time) override;
  void LogEligibilityEvalComputationInterrupted(
      absl::Status error_status, const ExampleStats& example_stats,
      absl::Time run_plan_start_time, absl::Time reference_time) override;
  void LogEligibilityEvalComputationCompleted(
      const ExampleStats& example_stats, absl::Time run_plan_start_time,
      absl::Time reference_time) override;

  // Check-in phase.
  void LogCheckinStarted() override;
  void LogCheckinIOError(absl::Status error_status,
                         const NetworkStats& network_stats,
                         absl::Time time_before_checkin,
                         absl::Time reference_time) override;
  void LogCheckinInvalidPayload(absl::string_view error_message,
                                const NetworkStats& network_stats,
                                absl::Time time_before_checkin,
                                absl::Time reference_time) override;
  void LogCheckinClientInterrupted(absl::Status error_status,
                                   const NetworkStats& network_stats,
                                   absl::Time time_before_checkin,
                                   absl::Time reference_time) override;
  void LogCheckinServerAborted(absl::Status error_status,
                               const NetworkStats& network_stats,
                               absl::Time time_before_checkin,
                               absl::Time reference_time) override;
  void LogCheckinTurnedAway(const NetworkStats& network_stats,
                            absl::Time time_before_checkin,
                            absl::Time reference_time) override;
  void LogCheckinPlanUriReceived(absl::string_view task_name,
                                 const NetworkStats& network_stats,
                                 absl::Time time_before_checkin) override;
  void LogCheckinCompleted(absl::string_view task_name,
                           const NetworkStats& network_stats,
                           absl::Time time_before_checkin,
                           absl::Time time_before_plan_download,
                           absl::Time reference_time) override;

  // Computation phase.
  void LogComputationStarted() override;
  void LogComputationInvalidArgument(absl::Status error_status,
                                     const ExampleStats& example_stats,
                                     const NetworkStats& network_stats,
                                     absl::Time run_plan_start_time) override;
  void LogComputationExampleIteratorError(
      absl::Status error_status, const ExampleStats& example_stats,
      const NetworkStats& network_stats,
      absl::Time run_plan_start_time) override;
  void LogComputationIOError(absl::Status error_status,
                             const ExampleStats& example_stats,
                             const NetworkStats& network_stats,
                             absl::Time run_plan_start_time) override;
  void LogComputationTensorflowError(absl::Status error_status,
                                     const ExampleStats& example_stats,
                                     const NetworkStats& network_stats,
                                     absl::Time run_plan_start_time,
                                     absl::Time reference_time) override;
  void LogComputationInterrupted(absl::Status error_status,
                                 const ExampleStats& example_stats,
                                 const NetworkStats& network_stats,
                                 absl::Time run_plan_start_time,
                                 absl::Time reference_time) override;
  void LogComputationCompleted(const ExampleStats& example_stats,
                               const NetworkStats& network_stats,
                               absl::Time run_plan_start_time,
                               absl::Time reference_time) override;

  absl::Status LogResultUploadStarted() override;
  void LogResultUploadIOError(absl::Status error_status,
                              const NetworkStats& network_stats,
                              absl::Time time_before_result_upload,
                              absl::Time reference_time) override;
  void LogResultUploadClientInterrupted(absl::Status error_status,
                                        const NetworkStats& network_stats,
                                        absl::Time time_before_result_upload,
                                        absl::Time reference_time) override;
  void LogResultUploadServerAborted(absl::Status error_status,
                                    const NetworkStats& network_stats,
                                    absl::Time time_before_result_upload,
                                    absl::Time reference_time) override;
  void LogResultUploadCompleted(const NetworkStats& network_stats,
                                absl::Time time_before_result_upload,
                                absl::Time reference_time) override;

  // Failure upload phase.
  absl::Status LogFailureUploadStarted() override;
  void LogFailureUploadIOError(absl::Status error_status,
                               const NetworkStats& network_stats,
                               absl::Time time_before_failure_upload,
                               absl::Time reference_time) override;
  void LogFailureUploadClientInterrupted(absl::Status error_status,
                                         const NetworkStats& network_stats,
                                         absl::Time time_before_failure_upload,
                                         absl::Time reference_time) override;
  void LogFailureUploadServerAborted(absl::Status error_status,
                                     const NetworkStats& network_stats,
                                     absl::Time time_before_failure_upload,
                                     absl::Time reference_time) override;
  void LogFailureUploadCompleted(const NetworkStats& network_stats,
                                 absl::Time time_before_failure_upload,
                                 absl::Time reference_time) override;

 private:
  void LogTimeSince(HistogramCounters histogram_counter,
                    absl::Time reference_time);
  void LogEligibilityEvalCheckinLatency(absl::Time time_before_checkin);
  void LogEligibilityEvalComputationLatency(absl::Time run_plan_start_time,
                                            absl::Time reference_time);
  void LogCheckinLatency(absl::Time time_before_checkin,
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
  const bool log_tensorflow_error_messages_;
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_PHASE_LOGGER_IMPL_H_
