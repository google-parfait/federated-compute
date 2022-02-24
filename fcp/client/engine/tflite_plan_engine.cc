/*
 * Copyright 2021 Google LLC
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
#include "fcp/client/engine/tflite_plan_engine.h"

#include <functional>
#include <string>
#include <utility>

#include "fcp/client/engine/plan_engine_helpers.h"
#include "fcp/client/engine/tflite_wrapper.h"
#include "fcp/protos/plan.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace fcp {
namespace client {
namespace engine {

using ::fcp::client::opstats::OperationalStats;
using ::google::internal::federated::plan::TensorflowSpec;

namespace {

PlanResult CreatePlanResultFromOutput(
    absl::StatusOr<std::vector<tensorflow::Tensor>> output,
    EventPublisher* event_publisher, opstats::OpStatsLogger* opstats_logger,
    LogManager* log_manager, const Flags* flags,
    std::function<void()> log_computation_finished,
    std::atomic<int>* total_example_count,
    std::atomic<int64_t>* total_example_size_bytes,
    const std::vector<std::string>& output_names,
    absl::Time run_plan_start_time, absl::Time reference_time) {
  switch (output.status().code()) {
    case absl::StatusCode::kOk: {
      if (!flags->per_phase_logs()) {
        event_publisher->PublishPlanCompleted(*total_example_count,
                                              *total_example_size_bytes,
                                              run_plan_start_time);
        log_computation_finished();
        log_manager->LogToLongHistogram(
            HistogramCounters::TRAINING_OVERALL_EXAMPLE_SIZE,
            *total_example_size_bytes);
        log_manager->LogToLongHistogram(
            HistogramCounters::TRAINING_OVERALL_EXAMPLE_COUNT,
            *total_example_count);
      }
      PlanResult plan_result(PlanOutcome::kSuccess, absl::OkStatus());
      plan_result.output_names = output_names;
      plan_result.output_tensors = std::move(*output);
      plan_result.total_example_count = *total_example_count;
      plan_result.total_example_size_bytes = *total_example_size_bytes;
      return plan_result;
    }
    case absl::StatusCode::kCancelled:
      if (!flags->per_phase_logs()) {
        event_publisher->PublishInterruption(
            /*execution_index=*/0, /*epoch_index=*/0, *total_example_count,
            *total_example_size_bytes, reference_time);
        opstats_logger->AddEventWithErrorMessage(
            OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
            std::string(output.status().message()));
      }
      return PlanResult(PlanOutcome::kInterrupted, std::move(output.status()));
    case absl::StatusCode::kInvalidArgument:
      if (!flags->per_phase_logs()) {
        event_publisher->PublishTensorFlowError(
            /*execution_index=*/0, /*epoch_index=*/0, *total_example_count,
            absl::StrCat("code: ", output.status().code(), ", error: ",
                         flags->log_tensorflow_error_messages()
                             ? output.status().message()
                             : ""));
        opstats_logger->AddEventWithErrorMessage(
            OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW,
            std::string(output.status().message()));
      }
      return PlanResult(PlanOutcome::kTensorflowError,
                        std::move(output.status()));
    default:
      FCP_LOG(FATAL) << "unexpected status code: " << output.status().code();
  }
  // Unreachable code.
  return PlanResult(PlanOutcome::kTensorflowError, absl::InternalError(""));
}
}  // namespace

PlanResult TfLitePlanEngine::RunPlan(
    const TensorflowSpec& tensorflow_spec, const std::string& model,
    std::unique_ptr<absl::flat_hash_map<std::string, std::string>> inputs,
    const std::vector<std::string>& output_names,
    absl::Time run_plan_start_time, absl::Time reference_time,
    std::function<void()> log_computation_started,
    std::function<void()> log_computation_finished,
    const SelectorContext& selector_context) {
  log_manager_->LogDiag(ProdDiagCode::BACKGROUND_TRAINING_TFLITE_ENGINE_USED);
  // Check that all inputs have corresponding TensorSpecProtos.
  absl::flat_hash_set<std::string> expected_input_tensor_names_set;
  for (auto it = inputs->begin(); it != inputs->end(); it++) {
    expected_input_tensor_names_set.insert(it->first);
  }
  absl::Status validity_checks = ValidateTensorflowSpec(
      tensorflow_spec, expected_input_tensor_names_set, output_names);
  if (!validity_checks.ok()) {
    FCP_LOG(ERROR) << validity_checks.message();
    if (!flags_->per_phase_logs()) {
      log_manager_->LogDiag(
          ProdDiagCode::BACKGROUND_TRAINING_FAILED_PLAN_FAILS_SANITY_CHECK);
      event_publisher_->PublishIoError(
          /*execution_index=*/0, validity_checks.message());
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_IO,
          std::string(validity_checks.message()));
    }
    return PlanResult(PlanOutcome::kInvalidArgument,
                      std::move(validity_checks));
  }
  std::atomic<int> total_example_count = 0;
  std::atomic<int64_t> total_example_size_bytes = 0;
  HostObjectRegistration host_registration = AddDatasetTokenToInputsForTfLite(
      [this, selector_context](
          const google::internal::federated::plan::ExampleSelector& selector) {
        return task_env_->CreateExampleIterator(selector, selector_context);
      },
      event_publisher_, log_manager_, opstats_logger_, inputs.get(),
      tensorflow_spec.dataset_token_tensor_name(), &total_example_count,
      &total_example_size_bytes);
  absl::StatusOr<std::unique_ptr<TfLiteWrapper>> tflite_wrapper =
      TfLiteWrapper::Create(
          model,
          [this]() {
            return task_env_->ShouldAbort(absl::Now(),
                                          timing_config_->polling_period);
          },
          *timing_config_, log_manager_, std::move(inputs));
  if (!tflite_wrapper.ok()) {
    if (!flags_->per_phase_logs()) {
      event_publisher_->PublishTensorFlowError(
          /*execution_index=*/0, /*epoch_index=*/0, /*epoch_example_index=*/0,
          absl::StrCat("code: ", tflite_wrapper.status().code(), ", error: ",
                       flags_->log_tensorflow_error_messages()
                           ? tflite_wrapper.status().message()
                           : ""));
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW,
          std::string(tflite_wrapper.status().message()));
    }
    return PlanResult(PlanOutcome::kTensorflowError, tflite_wrapper.status());
  }
  // Start running the plan.
  if (!flags_->per_phase_logs()) {
    event_publisher_->PublishPlanExecutionStarted();
    log_computation_started();
  }
  absl::StatusOr<std::vector<tensorflow::Tensor>> output =
      (*tflite_wrapper)->Run();
  PlanResult plan_result = CreatePlanResultFromOutput(
      std::move(output), event_publisher_, opstats_logger_, log_manager_,
      flags_, std::move(log_computation_finished), &total_example_count,
      &total_example_size_bytes, output_names, run_plan_start_time,
      reference_time);
  // Log timing info.
  if (!flags_->per_phase_logs()) {
    LogTimeSince(HistogramCounters::TRAINING_RUN_PHASE_LATENCY,
                 run_plan_start_time);
    LogTimeSince(HistogramCounters::TRAINING_RUN_PHASE_END_TIME,
                 reference_time);
  }
  return plan_result;
}

void TfLitePlanEngine::LogTimeSince(HistogramCounters histogram_counter,
                                    absl::Time reference_time) {
  absl::Duration duration = absl::Now() - reference_time;
  log_manager_->LogToLongHistogram(histogram_counter,
                                   absl::ToInt64Milliseconds(duration));
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
