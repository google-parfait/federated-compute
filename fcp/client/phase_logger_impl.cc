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
#include "fcp/client/phase_logger_impl.h"

#include <string>

#include "fcp/base/monitoring.h"

namespace fcp {
namespace client {
namespace {
constexpr absl::string_view kEligibilityCheckInErrorPrefix =
    "Error during eligibility check-in: ";
constexpr absl::string_view kEligibilityComputationErrorPrefix =
    "Error during eligibility eval computation: ";
constexpr absl::string_view kCheckInErrorPrefix = "Error during check-in: ";
constexpr absl::string_view kComputationErrorPrefix =
    "Error during computation: ";
constexpr absl::string_view kResultUploadErrorPrefix =
    "Error reporting results: ";
constexpr absl::string_view kFailureUploadErrorPrefix =
    "Error reporting computation failure: ";
}  // anonymous namespace

using ::fcp::client::opstats::OperationalStats;
using ::google::internal::federatedml::v2::RetryWindow;

void PhaseLoggerImpl::UpdateRetryWindowAndNetworkStats(
    const RetryWindow& retry_window, NetworkStats stats) {
  opstats_logger_->SetRetryWindow(retry_window);

  if (use_per_phase_logs_) {
    // Update the network stats.
    opstats_logger_->SetNetworkStats(
        stats.bytes_downloaded, stats.bytes_uploaded,
        stats.chunking_layer_bytes_received, stats.chunking_layer_bytes_sent);
  }
}

void PhaseLoggerImpl::SetModelIdentifier(absl::string_view model_identifier) {
  event_publisher_->SetModelIdentifier(std::string(model_identifier));
  log_manager_->SetModelIdentifier(std::string(model_identifier));
}

void PhaseLoggerImpl::LogTaskNotStarted(absl::string_view error_message) {
  if (granular_per_phase_logs_) {
    event_publisher_->PublishTaskNotStarted(error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_TRAIN_NOT_STARTED,
        std::string(error_message));
  } else {
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
        std::string(error_message));
  }
}

void PhaseLoggerImpl::LogEligibilityEvalCheckInStarted() {
  if (use_per_phase_logs_) {
    event_publisher_->PublishEligibilityEvalCheckin();
    opstats_logger_->AddEvent(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED);
  }
}

void PhaseLoggerImpl::LogEligibilityEvalCheckInIOError(
    absl::Status error_status, NetworkStats stats,
    absl::Time time_before_eligibility_eval_checkin) {
  std::string error_message =
      GetErrorMessage(error_status, kEligibilityCheckInErrorPrefix,
                      /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishEligibilityEvalCheckInIoError(
        stats.bytes_downloaded, stats.chunking_layer_bytes_received,
        error_message, absl::Now() - time_before_eligibility_eval_checkin);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_ERROR_IO,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, error_message);
  }
  LogEligibilityEvalCheckInLatency(time_before_eligibility_eval_checkin);
}

void PhaseLoggerImpl::LogEligibilityEvalCheckInClientInterrupted(
    absl::Status error_status, NetworkStats stats,
    absl::Time time_before_eligibility_eval_checkin) {
  std::string error_message =
      GetErrorMessage(error_status, kEligibilityCheckInErrorPrefix,
                      /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishEligibilityEvalCheckInClientInterrupted(
        stats.bytes_downloaded, stats.chunking_layer_bytes_received,
        error_message, absl::Now() - time_before_eligibility_eval_checkin);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::
            EVENT_KIND_ELIGIBILITY_CHECKIN_CLIENT_INTERRUPTED,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED, error_message);
  }
  LogEligibilityEvalCheckInLatency(time_before_eligibility_eval_checkin);
}

void PhaseLoggerImpl::LogEligibilityEvalCheckInServerAborted(
    absl::Status error_status, NetworkStats stats,
    absl::Time time_before_eligibility_eval_checkin) {
  std::string error_message =
      GetErrorMessage(error_status, kEligibilityCheckInErrorPrefix,
                      /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishEligibilityEvalCheckInServerAborted(
        stats.bytes_downloaded, stats.chunking_layer_bytes_received,
        error_message, absl::Now() - time_before_eligibility_eval_checkin);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_SERVER_ABORTED,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_SERVER_ABORTED, error_message);
  }
  LogEligibilityEvalCheckInLatency(time_before_eligibility_eval_checkin);
}

void PhaseLoggerImpl::LogEligibilityEvalNotConfigured(
    NetworkStats stats, absl::Time time_before_eligibility_eval_checkin) {
  if (use_per_phase_logs_) {
    event_publisher_->PublishEligibilityEvalNotConfigured(
        stats.bytes_downloaded, stats.chunking_layer_bytes_received,
        absl::Now() - time_before_eligibility_eval_checkin);
    opstats_logger_->AddEvent(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_DISABLED);
  }
  LogEligibilityEvalCheckInLatency(time_before_eligibility_eval_checkin);
}

void PhaseLoggerImpl::LogEligibilityEvalCheckInTurnedAway(
    NetworkStats stats, absl::Time time_before_eligibility_eval_checkin) {
  if (use_per_phase_logs_) {
    event_publisher_->PublishEligibilityEvalRejected(
        stats.bytes_downloaded, stats.chunking_layer_bytes_received,
        absl::Now() - time_before_eligibility_eval_checkin);
    opstats_logger_->AddEvent(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED);
  }
  LogEligibilityEvalCheckInLatency(time_before_eligibility_eval_checkin);
}

void PhaseLoggerImpl::LogEligibilityEvalCheckInInvalidPayloadError(
    absl::string_view error_message, NetworkStats stats,
    absl::Time time_before_eligibility_eval_checkin) {
  log_manager_->LogDiag(
      ProdDiagCode::
          BACKGROUND_TRAINING_ELIGIBILITY_EVAL_FAILED_CANNOT_PARSE_PLAN);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishEligibilityEvalCheckInErrorInvalidPayload(
        stats.bytes_downloaded, stats.chunking_layer_bytes_received,
        error_message, absl::Now() - time_before_eligibility_eval_checkin);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::
            EVENT_KIND_ELIGIBILITY_CHECKIN_ERROR_INVALID_PAYLOAD,
        std::string(error_message));
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO,
        std::string(error_message));
  }
  LogEligibilityEvalCheckInLatency(time_before_eligibility_eval_checkin);
}

void PhaseLoggerImpl::LogEligibilityEvalCheckInCompleted(
    NetworkStats stats, absl::Time time_before_eligibility_eval_checkin) {
  if (use_per_phase_logs_) {
    opstats_logger_->AddEvent(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_ENABLED);
    event_publisher_->PublishEligibilityEvalPlanReceived(
        stats.bytes_downloaded, stats.chunking_layer_bytes_received,
        absl::Now() - time_before_eligibility_eval_checkin);
  }
  LogEligibilityEvalCheckInLatency(time_before_eligibility_eval_checkin);
}

void PhaseLoggerImpl::LogEligibilityEvalComputationStarted() {
  if (use_per_phase_logs_) {
    if (granular_per_phase_logs_) {
      event_publisher_->PublishEligibilityEvalComputationStarted();
    } else {
      event_publisher_->PublishPlanExecutionStarted();
    }
    opstats_logger_->AddEvent(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_COMPUTATION_STARTED);
  }
}

void PhaseLoggerImpl::LogEligibilityEvalComputationInvalidArgument(
    absl::Status error_status, int total_example_count,
    int64_t total_example_size_bytes, absl::Time run_plan_start_time) {
  std::string error_message =
      GetErrorMessage(error_status, kEligibilityComputationErrorPrefix,
                      /* keep_error_message= */ true);
  log_manager_->LogDiag(
      ProdDiagCode::BACKGROUND_TRAINING_FAILED_PLAN_FAILS_SANITY_CHECK);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishEligibilityEvalComputationInvalidArgument(
        error_message, total_example_count, total_example_size_bytes,
        absl::Now() - run_plan_start_time);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::
            EVENT_KIND_ELIGIBILITY_COMPUTATION_ERROR_INVALID_ARGUMENT,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, error_message);
  }
}

void PhaseLoggerImpl::LogEligibilityEvalComputationExampleIteratorError(
    absl::Status error_status, int total_example_count,
    int64_t total_example_size_bytes, absl::Time run_plan_start_time) {
  if (use_per_phase_logs_) {
    std::string error_message =
        GetErrorMessage(error_status, kEligibilityComputationErrorPrefix,
                        /* keep_error_message= */ true);
    if (granular_per_phase_logs_) {
      event_publisher_->PublishEligibilityEvalComputationExampleIteratorError(
          error_message, total_example_count, total_example_size_bytes,
          absl::Now() - run_plan_start_time);
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::
              EVENT_KIND_ELIGIBILITY_COMPUTATION_ERROR_EXAMPLE_ITERATOR,
          error_message);
    } else {
      event_publisher_->PublishExampleSelectorError(0, 0, 0, error_message);
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_EXAMPLE_SELECTOR,
          error_message);
    }
  }
}

void PhaseLoggerImpl::LogEligibilityEvalComputationTensorflowError(
    absl::Status error_status, int total_example_count,
    int64_t total_example_size_bytes, absl::Time run_plan_start_time,
    absl::Time reference_time) {
  std::string error_message =
      GetErrorMessage(error_status, kEligibilityComputationErrorPrefix,
                      log_tensorflow_error_messages_);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishEligibilityEvalComputationTensorflowError(
        total_example_count, total_example_size_bytes, error_message,
        absl::Now() - run_plan_start_time);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::
            EVENT_KIND_ELIGIBILITY_COMPUTATION_ERROR_TENSORFLOW,
        error_message);
  } else {
    event_publisher_->PublishTensorFlowError(
        /*execution_index=*/0, /*epoch_index=*/0,
        /*epoch_example_index=*/total_example_count, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW, error_message);
  }
  if (use_per_phase_logs_) {
    LogEligibilityEvalComputationLatency(run_plan_start_time, reference_time);
  }
}

void PhaseLoggerImpl::LogEligibilityEvalComputationInterrupted(
    absl::Status error_status, int total_example_count,
    int64_t total_example_size_bytes, absl::Time run_plan_start_time,
    absl::Time reference_time) {
  if (use_per_phase_logs_) {
    std::string error_message =
        GetErrorMessage(error_status, kEligibilityComputationErrorPrefix,
                        /* keep_error_message= */ true);
    if (granular_per_phase_logs_) {
      event_publisher_->PublishEligibilityEvalComputationInterrupted(
          total_example_count, total_example_size_bytes, error_message,
          absl::Now() - run_plan_start_time);
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::
              EVENT_KIND_ELIGIBILITY_COMPUTATION_CLIENT_INTERRUPTED,
          error_message);
    } else {
      event_publisher_->PublishInterruption(
          /*execution_index=*/0, /*epoch_index=*/0, total_example_count,
          total_example_size_bytes, run_plan_start_time);
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
          error_message);
    }
    LogEligibilityEvalComputationLatency(run_plan_start_time, reference_time);
  }
}

void PhaseLoggerImpl::LogEligibilityEvalComputationCompleted(
    int total_example_count, int64_t total_example_size_bytes,
    absl::Time run_plan_start_time, absl::Time reference_time) {
  if (use_per_phase_logs_) {
    if (granular_per_phase_logs_) {
      event_publisher_->PublishEligibilityEvalComputationCompleted(
          total_example_count, total_example_size_bytes,
          absl::Now() - run_plan_start_time);
    } else {
      event_publisher_->PublishPlanCompleted(
          total_example_count, total_example_size_bytes, run_plan_start_time);
    }
    opstats_logger_->AddEvent(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_COMPUTATION_FINISHED);
    log_manager_->LogToLongHistogram(
        HistogramCounters::TRAINING_OVERALL_EXAMPLE_SIZE,
        total_example_size_bytes);
    log_manager_->LogToLongHistogram(
        HistogramCounters::TRAINING_OVERALL_EXAMPLE_COUNT, total_example_count);
    LogEligibilityEvalComputationLatency(run_plan_start_time, reference_time);
  }
}

void PhaseLoggerImpl::LogCheckInStarted() {
  if (use_per_phase_logs_) {
    // Log that we are about to check in with the server.
    event_publisher_->PublishCheckin();
    opstats_logger_->AddEvent(
        OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED);
  }
}

void PhaseLoggerImpl::LogCheckInIOError(absl::Status error_status,
                                        NetworkStats stats,
                                        absl::Time time_before_checkin,
                                        absl::Time reference_time) {
  std::string error_message = GetErrorMessage(error_status, kCheckInErrorPrefix,
                                              /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishCheckinIoError(
        stats.bytes_downloaded, stats.chunking_layer_bytes_received,
        error_message, absl::Now() - time_before_checkin);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_CHECKIN_ERROR_IO, error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, error_message);
  }
  LogCheckInLatency(time_before_checkin, reference_time);
}

void PhaseLoggerImpl::LogCheckInClientInterrupted(
    absl::Status error_status, NetworkStats stats,
    absl::Time time_before_checkin, absl::Time reference_time) {
  std::string error_message = GetErrorMessage(error_status, kCheckInErrorPrefix,
                                              /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishCheckinClientInterrupted(
        stats.bytes_downloaded, stats.chunking_layer_bytes_received,
        error_message, absl::Now() - time_before_checkin);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_CHECKIN_CLIENT_INTERRUPTED,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED, error_message);
  }
  LogCheckInLatency(time_before_checkin, reference_time);
}

void PhaseLoggerImpl::LogCheckInServerAborted(absl::Status error_status,
                                              NetworkStats stats,
                                              absl::Time time_before_checkin,
                                              absl::Time reference_time) {
  std::string error_message = GetErrorMessage(error_status, kCheckInErrorPrefix,
                                              /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishCheckinServerAborted(
        stats.bytes_downloaded, stats.chunking_layer_bytes_received,
        error_message, absl::Now() - time_before_checkin);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_CHECKIN_SERVER_ABORTED,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_SERVER_ABORTED, error_message);
  }
  LogCheckInLatency(time_before_checkin, reference_time);
}

void PhaseLoggerImpl::LogCheckInTurnedAway(NetworkStats stats,
                                           absl::Time time_before_checkin,
                                           absl::Time reference_time) {
  if (use_per_phase_logs_) {
    if (granular_per_phase_logs_) {
      event_publisher_->PublishRejected(stats.bytes_downloaded,
                                        stats.chunking_layer_bytes_received,
                                        absl::Now() - time_before_checkin);
    } else {
      event_publisher_->PublishRejected();
    }
    opstats_logger_->AddEvent(
        OperationalStats::Event::EVENT_KIND_CHECKIN_REJECTED);
  }
  LogCheckInLatency(time_before_checkin, reference_time);
}

void PhaseLoggerImpl::LogCheckInInvalidPayload(absl::string_view error_message,
                                               NetworkStats stats,
                                               absl::Time time_before_checkin,
                                               absl::Time reference_time) {
  log_manager_->LogDiag(
      ProdDiagCode::BACKGROUND_TRAINING_FAILED_CANNOT_PARSE_PLAN);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishCheckinInvalidPayload(
        stats.bytes_downloaded, stats.chunking_layer_bytes_received,
        error_message, absl::Now() - time_before_checkin);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_CHECKIN_ERROR_INVALID_PAYLOAD,
        std::string(error_message));
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO,
        std::string(error_message));
  }
  LogCheckInLatency(time_before_checkin, reference_time);
}

void PhaseLoggerImpl::LogCheckInCompleted(absl::string_view task_name,
                                          NetworkStats stats,
                                          absl::Time time_before_checkin,
                                          absl::Time reference_time) {
  if (use_per_phase_logs_) {
    if (granular_per_phase_logs_) {
      event_publisher_->PublishCheckinFinishedV2(
          stats.bytes_downloaded, stats.chunking_layer_bytes_received,
          absl::Now() - time_before_checkin);
    } else {
      event_publisher_->PublishCheckinFinished(
          stats.bytes_downloaded, stats.chunking_layer_bytes_received,
          absl::Now() - time_before_checkin);
    }
    opstats_logger_->AddCheckinAcceptedEventWithTaskName(
        std::string(task_name));
  }
  LogCheckInLatency(time_before_checkin, reference_time);
}

void PhaseLoggerImpl::LogComputationStarted() {
  if (use_per_phase_logs_) {
    if (granular_per_phase_logs_) {
      event_publisher_->PublishComputationStarted();
    } else {
      event_publisher_->PublishPlanExecutionStarted();
    }
    opstats_logger_->AddEvent(
        OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED);
  }
}

void PhaseLoggerImpl::LogComputationInvalidArgument(
    absl::Status error_status, int total_example_count,
    int64_t total_example_size_bytes, absl::Time run_plan_start_time) {
  std::string error_message =
      GetErrorMessage(error_status, kComputationErrorPrefix,
                      /* keep_error_message= */ true);
  log_manager_->LogDiag(
      ProdDiagCode::BACKGROUND_TRAINING_FAILED_PLAN_FAILS_SANITY_CHECK);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishComputationInvalidArgument(
        error_message, total_example_count, total_example_size_bytes,
        absl::Now() - run_plan_start_time);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_COMPUTATION_ERROR_INVALID_ARGUMENT,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, error_message);
  }
}

void PhaseLoggerImpl::LogComputationIOError(absl::Status error_status,
                                            int total_example_count,
                                            int64_t total_example_size_bytes,
                                            absl::Time run_plan_start_time) {
  std::string error_message =
      GetErrorMessage(error_status, kComputationErrorPrefix,
                      /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishComputationIOError(
        error_message, total_example_count, total_example_size_bytes,
        absl::Now() - run_plan_start_time);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_COMPUTATION_ERROR_IO,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, error_message);
  }
}

void PhaseLoggerImpl::LogComputationExampleIteratorError(
    absl::Status error_status, int total_example_count,
    int64_t total_example_size_bytes, absl::Time run_plan_start_time) {
  if (use_per_phase_logs_) {
    std::string error_message = GetErrorMessage(
        error_status, kComputationErrorPrefix, /* keep_error_message= */ true);
    if (granular_per_phase_logs_) {
      event_publisher_->PublishComputationExampleIteratorError(
          error_message, total_example_count, total_example_size_bytes,
          absl::Now() - run_plan_start_time);
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::
              EVENT_KIND_COMPUTATION_ERROR_EXAMPLE_ITERATOR,
          error_message);
    } else {
      event_publisher_->PublishExampleSelectorError(0, 0, 0, error_message);
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_EXAMPLE_SELECTOR,
          error_message);
    }
  }
}

void PhaseLoggerImpl::LogComputationTensorflowError(
    absl::Status error_status, int total_example_count,
    int64_t total_example_size_bytes, absl::Time run_plan_start_time,
    absl::Time reference_time) {
  std::string error_message = GetErrorMessage(
      error_status, kComputationErrorPrefix, log_tensorflow_error_messages_);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishComputationTensorflowError(
        total_example_count, total_example_size_bytes, error_message,
        absl::Now() - run_plan_start_time);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_COMPUTATION_ERROR_TENSORFLOW,
        error_message);
  } else {
    event_publisher_->PublishTensorFlowError(
        /*execution_index=*/0, /*epoch_index=*/0,
        /*epoch_example_index=*/total_example_count, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW, error_message);
  }
  if (use_per_phase_logs_) {
    LogComputationLatency(run_plan_start_time, reference_time);
  }
}

void PhaseLoggerImpl::LogComputationInterrupted(
    absl::Status error_status, int total_example_count,
    int64_t total_example_size_bytes, absl::Time run_plan_start_time,
    absl::Time reference_time) {
  if (use_per_phase_logs_) {
    std::string error_message =
        GetErrorMessage(error_status, kComputationErrorPrefix,
                        /* keep_error_message= */ true);
    if (granular_per_phase_logs_) {
      event_publisher_->PublishComputationInterrupted(
          total_example_count, total_example_size_bytes, error_message,
          absl::Now() - run_plan_start_time);
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::
              EVENT_KIND_ELIGIBILITY_COMPUTATION_CLIENT_INTERRUPTED,
          error_message);
    } else {
      event_publisher_->PublishInterruption(
          /*execution_index=*/0, /*epoch_index=*/0, total_example_count,
          total_example_size_bytes, run_plan_start_time);
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
          error_message);
    }
    LogComputationLatency(run_plan_start_time, reference_time);
  }
}

void PhaseLoggerImpl::LogComputationCompleted(int total_example_count,
                                              int64_t total_example_size_bytes,
                                              absl::Time run_plan_start_time,
                                              absl::Time reference_time) {
  if (use_per_phase_logs_) {
    if (granular_per_phase_logs_) {
      event_publisher_->PublishComputationCompleted(
          total_example_count, total_example_size_bytes, run_plan_start_time);
    } else {
      event_publisher_->PublishPlanCompleted(
          total_example_count, total_example_size_bytes, run_plan_start_time);
    }
    opstats_logger_->AddEvent(
        OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED);
    log_manager_->LogToLongHistogram(
        HistogramCounters::TRAINING_OVERALL_EXAMPLE_SIZE,
        total_example_size_bytes);
    log_manager_->LogToLongHistogram(
        HistogramCounters::TRAINING_OVERALL_EXAMPLE_COUNT, total_example_count);
    LogComputationLatency(run_plan_start_time, reference_time);
  }
}

absl::Status PhaseLoggerImpl::LogResultUploadStarted() {
  if (use_per_phase_logs_) {
    if (granular_per_phase_logs_) {
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED);
    } else {
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED);
    }
    // Commit the run data accumulated thus far to Opstats and fail if
    // something goes wrong.
    FCP_RETURN_IF_ERROR(opstats_logger_->CommitToStorage());
    if (granular_per_phase_logs_) {
      event_publisher_->PublishResultUploadStarted();
    } else {
      event_publisher_->PublishReportStarted(0);
    }
  }
  return absl::OkStatus();
}

void PhaseLoggerImpl::LogResultUploadIOError(
    absl::Status error_status, NetworkStats stats,
    absl::Time time_before_result_upload, absl::Time reference_time) {
  std::string error_message =
      GetErrorMessage(error_status, kResultUploadErrorPrefix,
                      /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishResultUploadIOError(
        stats.report_size_bytes, stats.chunking_layer_bytes_sent, error_message,
        absl::Now() - time_before_result_upload);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_ERROR_IO,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, error_message);
  }
  LogReportLatency(time_before_result_upload, reference_time);
}

void PhaseLoggerImpl::LogResultUploadClientInterrupted(
    absl::Status error_status, NetworkStats stats,
    absl::Time time_before_result_upload, absl::Time reference_time) {
  std::string error_message =
      GetErrorMessage(error_status, kResultUploadErrorPrefix,
                      /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishResultUploadClientInterrupted(
        stats.report_size_bytes, stats.chunking_layer_bytes_sent, error_message,
        absl::Now() - time_before_result_upload);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_CLIENT_INTERRUPTED,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED, error_message);
  }
  LogReportLatency(time_before_result_upload, reference_time);
}

void PhaseLoggerImpl::LogResultUploadServerAborted(
    absl::Status error_status, NetworkStats stats,
    absl::Time time_before_result_upload, absl::Time reference_time) {
  std::string error_message =
      GetErrorMessage(error_status, kResultUploadErrorPrefix,
                      /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishResultUploadServerAborted(
        stats.report_size_bytes, stats.chunking_layer_bytes_sent, error_message,
        absl::Now() - time_before_result_upload);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_SERVER_ABORTED,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_SERVER_ABORTED, error_message);
  }
  LogReportLatency(time_before_result_upload, reference_time);
}

void PhaseLoggerImpl::LogResultUploadCompleted(
    NetworkStats stats, absl::Time time_before_result_upload,
    absl::Time reference_time) {
  if (use_per_phase_logs_) {
    if (granular_per_phase_logs_) {
      event_publisher_->PublishResultUploadCompleted(
          stats.report_size_bytes, stats.chunking_layer_bytes_sent,
          absl::Now() - time_before_result_upload);
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_FINISHED);
    } else {
      event_publisher_->PublishReportFinished(
          stats.report_size_bytes, stats.chunking_layer_bytes_sent,
          absl::Now() - time_before_result_upload);
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_UPLOAD_FINISHED);
    }
  }
  LogReportLatency(time_before_result_upload, reference_time);
}

absl::Status PhaseLoggerImpl::LogFailureUploadStarted() {
  if (use_per_phase_logs_) {
    if (granular_per_phase_logs_) {
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_FAILURE_UPLOAD_STARTED);
    } else {
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED);
    }
    // Commit the run data accumulated thus far to Opstats and fail if
    // something goes wrong.
    FCP_RETURN_IF_ERROR(opstats_logger_->CommitToStorage());
    if (granular_per_phase_logs_) {
      event_publisher_->PublishFailureUploadStarted();
    } else {
      event_publisher_->PublishReportStarted(0);
    }
  }
  return absl::OkStatus();
}

void PhaseLoggerImpl::LogFailureUploadIOError(
    absl::Status error_status, NetworkStats stats,
    absl::Time time_before_failure_upload, absl::Time reference_time) {
  std::string error_message =
      GetErrorMessage(error_status, kFailureUploadErrorPrefix,
                      /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishFailureUploadIOError(
        stats.report_size_bytes, stats.chunking_layer_bytes_sent, error_message,
        absl::Now() - time_before_failure_upload);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_FAILURE_UPLOAD_ERROR_IO,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, error_message);
  }
  LogReportLatency(time_before_failure_upload, reference_time);
}

void PhaseLoggerImpl::LogFailureUploadClientInterrupted(
    absl::Status error_status, NetworkStats stats,
    absl::Time time_before_failure_upload, absl::Time reference_time) {
  std::string error_message =
      GetErrorMessage(error_status, kFailureUploadErrorPrefix,
                      /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishFailureUploadClientInterrupted(
        stats.report_size_bytes, stats.chunking_layer_bytes_sent, error_message,
        absl::Now() - time_before_failure_upload);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_FAILURE_UPLOAD_CLIENT_INTERRUPTED,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED, error_message);
  }
  LogReportLatency(time_before_failure_upload, reference_time);
}

void PhaseLoggerImpl::LogFailureUploadServerAborted(
    absl::Status error_status, NetworkStats stats,
    absl::Time time_before_failure_upload, absl::Time reference_time) {
  std::string error_message =
      GetErrorMessage(error_status, kFailureUploadErrorPrefix,
                      /* keep_error_message= */ true);
  if (granular_per_phase_logs_) {
    event_publisher_->PublishFailureUploadServerAborted(
        stats.report_size_bytes, stats.chunking_layer_bytes_sent, error_message,
        absl::Now() - time_before_failure_upload);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_FAILURE_UPLOAD_SERVER_ABORTED,
        error_message);
  } else {
    event_publisher_->PublishIoError(0, error_message);
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_SERVER_ABORTED, error_message);
  }
  LogReportLatency(time_before_failure_upload, reference_time);
}

void PhaseLoggerImpl::LogFailureUploadCompleted(
    NetworkStats stats, absl::Time time_before_failure_upload,
    absl::Time reference_time) {
  if (use_per_phase_logs_) {
    if (granular_per_phase_logs_) {
      event_publisher_->PublishFailureUploadCompleted(
          stats.report_size_bytes, stats.chunking_layer_bytes_sent,
          absl::Now() - time_before_failure_upload);
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_FAILURE_UPLOAD_FINISHED);
    } else {
      event_publisher_->PublishReportFinished(
          stats.report_size_bytes, stats.chunking_layer_bytes_sent,
          absl::Now() - time_before_failure_upload);
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_UPLOAD_FINISHED);
    }
  }
  LogReportLatency(time_before_failure_upload, reference_time);
}

void PhaseLoggerImpl::LogTimeSince(HistogramCounters histogram_counter,
                                   absl::Time reference_time) {
  absl::Duration duration = absl::Now() - reference_time;
  log_manager_->LogToLongHistogram(histogram_counter,
                                   absl::ToInt64Milliseconds(duration));
}

void PhaseLoggerImpl::LogEligibilityEvalCheckInLatency(
    absl::Time time_before_eligibility_eval_checkin) {
  LogTimeSince(HistogramCounters::TRAINING_FL_ELIGIBILITY_EVAL_CHECKIN_LATENCY,
               time_before_eligibility_eval_checkin);
}

void PhaseLoggerImpl::LogEligibilityEvalComputationLatency(
    absl::Time run_plan_start_time, absl::Time reference_time) {
  LogTimeSince(HistogramCounters::TRAINING_RUN_PHASE_LATENCY,
               run_plan_start_time);
  LogTimeSince(HistogramCounters::TRAINING_RUN_PHASE_END_TIME, reference_time);
}

void PhaseLoggerImpl::LogCheckInLatency(absl::Time time_before_checkin,
                                        absl::Time reference_time) {
  LogTimeSince(HistogramCounters::TRAINING_FL_CHECKIN_LATENCY,
               time_before_checkin);
  LogTimeSince(HistogramCounters::TRAINING_FL_CHECKIN_END_TIME, reference_time);
}

void PhaseLoggerImpl::LogComputationLatency(absl::Time run_plan_start_time,
                                            absl::Time reference_time) {
  LogTimeSince(HistogramCounters::TRAINING_RUN_PHASE_LATENCY,
               run_plan_start_time);
  LogTimeSince(HistogramCounters::TRAINING_RUN_PHASE_END_TIME, reference_time);
}

void PhaseLoggerImpl::LogReportLatency(absl::Time time_before_report,
                                       absl::Time reference_time) {
  LogTimeSince(HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY,
               time_before_report);
  LogTimeSince(HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME,
               reference_time);
}

std::string PhaseLoggerImpl::GetErrorMessage(absl::Status error_status,
                                             absl::string_view error_prefix,
                                             bool keep_error_message) {
  return absl::StrCat(error_prefix, "code: ", error_status.code(), ", error: ",
                      keep_error_message ? error_status.message() : "");
}

}  // namespace client
}  // namespace fcp
