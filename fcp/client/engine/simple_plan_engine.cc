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
#include "fcp/client/engine/simple_plan_engine.h"

#include "google/protobuf/any.pb.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/engine/plan_engine_helpers.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace fcp {
namespace client {
namespace engine {

using ::fcp::client::opstats::OperationalStats;
using ::fcp::client::opstats::OpStatsLogger;
using ::google::internal::federated::plan::TensorflowSpec;

namespace {

absl::Status ValidateTensorflowSpec(
    const TensorflowSpec& tensorflow_spec,
    const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
    const std::vector<std::string>& output_names) {
  // Check that all inputs have corresponding TensorSpecProtos.
  absl::flat_hash_set<std::string> expected_input_tensor_names_set;
  for (const std::pair<std::string, tensorflow::Tensor>& input : inputs) {
    expected_input_tensor_names_set.insert(input.first);
  }
  if (expected_input_tensor_names_set.size() !=
      tensorflow_spec.input_tensor_specs_size()) {
    return absl::InvalidArgumentError(
        "Unexpected number of input_tensor_specs");
  }

  for (const tensorflow::TensorSpecProto& it :
       tensorflow_spec.input_tensor_specs()) {
    if (!expected_input_tensor_names_set.count(it.name())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Missing expected TensorSpecProto for input ", it.name()));
    }
  }
  // Check that all outputs have corresponding TensorSpecProtos.
  absl::flat_hash_set<std::string> expected_output_tensor_names_set(
      output_names.begin(), output_names.end());
  if (expected_output_tensor_names_set.size() !=
      tensorflow_spec.output_tensor_specs_size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected number of output_tensor_specs: ",
                     expected_output_tensor_names_set.size(), " vs. ",
                     tensorflow_spec.output_tensor_specs_size()));
  }
  for (const tensorflow::TensorSpecProto& it :
       tensorflow_spec.output_tensor_specs()) {
    if (!expected_output_tensor_names_set.count(it.name())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Missing expected TensorSpecProto for output ", it.name()));
    }
  }

  return absl::OkStatus();
}

}  // namespace

SimplePlanEngine::SimplePlanEngine(
    SimpleTaskEnvironment* task_env, LogManager* log_manager,
    EventPublisher* event_publisher, OpStatsLogger* opstats_logger,
    const InterruptibleRunner::TimingConfig* timing_config, const Flags* flags)
    : task_env_(task_env),
      log_manager_(log_manager),
      event_publisher_(event_publisher),
      opstats_logger_(opstats_logger),
      timing_config_(timing_config),
      flags_(flags) {}

PlanResult SimplePlanEngine::RunPlan(
    const TensorflowSpec& tensorflow_spec, const std::string& graph,
    const ::google::protobuf::Any& config_proto,
    std::unique_ptr<std::vector<std::pair<std::string, tensorflow::Tensor>>>
        inputs,
    const std::vector<std::string>& output_names,
    absl::Time run_plan_start_time, absl::Time reference_time,
    std::function<void()> log_computation_started,
    std::function<void()> log_computation_finished,
    const SelectorContext& selector_context) {
  absl::Status validity_checks =
      ValidateTensorflowSpec(tensorflow_spec, *inputs, output_names);
  if (!validity_checks.ok()) {
    FCP_LOG(ERROR) << validity_checks.message();
    log_manager_->LogDiag(
        ProdDiagCode::BACKGROUND_TRAINING_FAILED_PLAN_FAILS_SANITY_CHECK);
    event_publisher_->PublishIoError(
        /*execution_index=*/0, validity_checks.message());
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO,
        std::string(validity_checks.message()));
    return PlanResult(PhaseOutcome::ERROR);
  }

  absl::StatusOr<std::unique_ptr<TensorFlowWrapper>> tf_wrapper_or =
      TensorFlowWrapper::Create(
          graph, config_proto,
          [this]() {
            return task_env_->ShouldAbort(absl::Now(),
                                          timing_config_->polling_period);
          },
          *timing_config_, log_manager_);
  if (!tf_wrapper_or.ok()) {
    event_publisher_->PublishTensorFlowError(
        /*execution_index=*/0, /*epoch_index=*/0, /*epoch_example_index=*/0,
        flags_->log_tensorflow_error_messages()
            ? tf_wrapper_or.status().message()
            : "");
    opstats_logger_->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW,
        std::string(tf_wrapper_or.status().message()));
    return PlanResult(PhaseOutcome::ERROR);
  }

  std::unique_ptr<TensorFlowWrapper> tf_wrapper =
      std::move(tf_wrapper_or.value());
  auto tf_result = RunPlanInternal(tf_wrapper.get(), tensorflow_spec,
                                   std::move(inputs), output_names,
                                   run_plan_start_time, log_computation_started,
                                   log_computation_finished, selector_context);
  FCP_CHECK(tf_wrapper->CloseAndRelease().ok());

  // Log timing info.
  LogTimeSince(HistogramCounters::TRAINING_RUN_PHASE_LATENCY,
               run_plan_start_time);
  LogTimeSince(HistogramCounters::TRAINING_RUN_PHASE_END_TIME, reference_time);

  switch (tf_result.status().code()) {
    case absl::StatusCode::kOk: {
      PlanResult plan_result(PhaseOutcome::COMPLETED);
      plan_result.output_names = output_names;
      plan_result.output_tensors = std::move(tf_result).value();
      return plan_result;
    }
    case absl::StatusCode::kCancelled:
      return PlanResult(PhaseOutcome::INTERRUPTED);
    case absl::StatusCode::kInvalidArgument:
      return PlanResult(PhaseOutcome::ERROR);
    default:
      FCP_LOG(FATAL) << "unexpected status code: " << tf_result.status().code();
  }
  // Unreachable, but clang doesn't get it.
  return PlanResult(PhaseOutcome::ERROR);
}

absl::StatusOr<std::vector<tensorflow::Tensor>>
SimplePlanEngine::RunPlanInternal(
    TensorFlowWrapper* tf_wrapper,
    const google::internal::federated::plan::TensorflowSpec& tensorflow_spec,
    std::unique_ptr<std::vector<std::pair<std::string, tensorflow::Tensor>>>
        inputs,
    const std::vector<std::string>& output_names,
    absl::Time run_plan_start_time,
    std::function<void()> log_computation_started,
    std::function<void()> log_computation_finished,
    const SelectorContext& selector_context) {
  // Populate input tensor vector
  std::atomic<int> total_example_count = 0;
  std::atomic<int64_t> total_example_size_bytes = 0;
  // AddDatasetTokenToInputs first registers a DatasetProvider with the global
  // ExternalDatasetProviderRegistry and then returns a HostObjectRegistration
  // object. Hold onto the HostObjectRegistration object since it de-registers
  // upon destruction.
  HostObjectRegistration host_registration = AddDatasetTokenToInputs(
      [this, selector_context](
          const google::internal::federated::plan::ExampleSelector& selector) {
          return task_env_->CreateExampleIterator(selector, selector_context);
      },
      event_publisher_, log_manager_, opstats_logger_, inputs.get(),
      tensorflow_spec.dataset_token_tensor_name(), &total_example_count,
      &total_example_size_bytes);

  std::vector<std::string> target_names;
  for (const std::string& target_node_name :
       tensorflow_spec.target_node_names()) {
    target_names.push_back(target_node_name);
  }

  // Start running the plan.
  event_publisher_->PublishPlanExecutionStarted();
  log_computation_started();

  absl::Time call_start_time = absl::Now();
  FCP_ASSIGN_OR_RETURN(
      auto result,
      RunTensorFlowInternal(tf_wrapper, *inputs, output_names, target_names,
                            &total_example_count, &total_example_size_bytes,
                            call_start_time));

  event_publisher_->PublishPlanCompleted(
      total_example_count, total_example_size_bytes, run_plan_start_time);
  log_computation_finished();
  log_manager_->LogToLongHistogram(
      HistogramCounters::TRAINING_OVERALL_EXAMPLE_SIZE,
      total_example_size_bytes);
  log_manager_->LogToLongHistogram(
      HistogramCounters::TRAINING_OVERALL_EXAMPLE_COUNT, total_example_count);

  return result;
}

absl::StatusOr<std::vector<tensorflow::Tensor>>
SimplePlanEngine::RunTensorFlowInternal(
    TensorFlowWrapper* tf_wrapper,
    const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
    const std::vector<std::string>& output_tensor_names,
    const std::vector<std::string>& target_node_names,
    std::atomic<int>* total_example_count,
    std::atomic<int64_t>* total_example_size_bytes, absl::Time start) {
  std::vector<tensorflow::Tensor> outputs;
  absl::Status status =
      tf_wrapper->Run(inputs, output_tensor_names, target_node_names, &outputs);
  switch (status.code()) {
    case absl::StatusCode::kCancelled:
      event_publisher_->PublishInterruption(
          /*execution_index=*/0, /*epoch_index=*/0, total_example_count->load(),
          total_example_size_bytes->load(), start);
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
          std::string(status.message()));
      return status;
    case absl::StatusCode::kInvalidArgument:
      event_publisher_->PublishTensorFlowError(
          /*execution_index=*/0, /*epoch_index=*/0, total_example_count->load(),
          flags_->log_tensorflow_error_messages() ? status.message() : "");
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW,
          std::string(status.message()));
      return status;
    case absl::StatusCode::kOutOfRange:
    case absl::StatusCode::kOk:
      break;
    default:
      FCP_CHECK_STATUS(status);
  }

  return outputs;
}

void SimplePlanEngine::LogTimeSince(HistogramCounters histogram_counter,
                                    absl::Time reference_time) {
  absl::Duration duration = absl::Now() - reference_time;
  log_manager_->LogToLongHistogram(histogram_counter,
                                   absl::ToInt64Milliseconds(duration));
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
