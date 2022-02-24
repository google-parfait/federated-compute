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
#include "fcp/client/lc_runner.h"

#include <map>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/platform.h"
#include "fcp/client/engine/plan_engine_helpers.h"
#include "fcp/client/engine/simple_plan_engine.h"

#ifdef FCP_CLIENT_SUPPORT_TFLITE
#include "fcp/client/engine/tflite_plan_engine.h"
#endif

#include "fcp/client/phase_logger_impl.h"
#include "fcp/client/selector_context.pb.h"
#include "fcp/protos/plan.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace fcp {
namespace client {

using ::fcp::client::opstats::OperationalStats;
using ::fcp::client::opstats::OpStatsLogger;
using ::google::internal::federated::plan::ClientOnlyPlan;
using ::google::internal::federated::plan::LocalComputeIORouter;

namespace {
std::unique_ptr<std::vector<std::pair<std::string, tensorflow::Tensor>>>
ConstructInputsForTensorflowSpecPlan(const LocalComputeIORouter& local_compute,
                                     const std::string& input_dir_uri,
                                     const std::string& output_dir_uri) {
  auto inputs = std::make_unique<
      std::vector<std::pair<std::string, tensorflow::Tensor>>>();
  tensorflow::Tensor input_dirpath(tensorflow::DT_STRING, {});
  input_dirpath.scalar<tensorflow::tstring>()() = input_dir_uri;
  inputs->push_back({local_compute.input_dir_tensor_name(), input_dirpath});
  tensorflow::Tensor output_dirpath(tensorflow::DT_STRING, {});
  output_dirpath.scalar<tensorflow::tstring>()() = output_dir_uri;
  inputs->push_back({local_compute.output_dir_tensor_name(), output_dirpath});
  return inputs;
}

#ifdef FCP_CLIENT_SUPPORT_TFLITE
std::unique_ptr<TfLiteInputs> ConstructInputsForTFLitePlan(
    const LocalComputeIORouter& local_compute, const std::string& input_dir_uri,
    const std::string& output_dir_uri) {
  auto inputs = std::make_unique<TfLiteInputs>();
  (*inputs)[local_compute.input_dir_tensor_name()] = input_dir_uri;
  (*inputs)[local_compute.output_dir_tensor_name()] = output_dir_uri;
  return inputs;
}
#endif

void LogComputationOutcome(engine::PlanResult plan_result,
                           PhaseLogger& phase_logger,
                           absl::Time run_plan_start_time,
                           absl::Time reference_time) {
  switch (plan_result.outcome) {
    case engine::PlanOutcome::kSuccess:
      phase_logger.LogComputationCompleted(plan_result.total_example_count,
                                           plan_result.total_example_size_bytes,
                                           run_plan_start_time, reference_time);
      break;
    case engine::PlanOutcome::kInterrupted:
      phase_logger.LogComputationInterrupted(
          plan_result.original_status, plan_result.total_example_count,
          plan_result.total_example_size_bytes, run_plan_start_time,
          reference_time);
      break;
    case engine::PlanOutcome::kInvalidArgument:
      phase_logger.LogComputationInvalidArgument(plan_result.original_status);
      break;
    case engine::PlanOutcome::kTensorflowError:
      phase_logger.LogComputationTensorflowError(
          std::move(plan_result.original_status),
          plan_result.total_example_count, run_plan_start_time, reference_time);
      break;
  }
}

absl::Status RunPlanWithTensorflowSpec(
    PhaseLogger& phase_logger, SimpleTaskEnvironment* env_deps,
    EventPublisher* event_publisher, LogManager* log_manager,
    OpStatsLogger* opstats_logger, const Flags* flags,
    const ClientOnlyPlan& client_plan, const std::string& input_dir_uri,
    const std::string& output_dir_uri,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time run_plan_start_time, const absl::Time reference_time,
    const SelectorContext& selector_context) {
  // Check that this is a TensorflowSpec-based plan for local computation.
  if (!client_plan.phase().has_tensorflow_spec()) {
    absl::Status error_status =
        absl::InvalidArgumentError("Plan without TensorflowSpec");
    phase_logger.LogComputationInvalidArgument(error_status);
    return error_status;
  }
  if (!client_plan.phase().has_local_compute() ||
      client_plan.phase().execution_size() > 0) {
    absl::Status error_status =
        absl::InvalidArgumentError("Invalid TensorflowSpec-based plan");
    phase_logger.LogComputationInvalidArgument(error_status);
    return error_status;
  }

  auto log_computation_started = [opstats_logger]() {
    opstats_logger->AddEvent(
        OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED);
  };
  auto log_computation_finished = [opstats_logger]() {
    opstats_logger->AddEvent(
        OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED);
  };

  // Run plan
  std::vector<std::string> output_names_unused;

#ifdef FCP_CLIENT_SUPPORT_TFLITE
  if (flags->use_tflite_training() && !client_plan.tflite_graph().empty()) {
    auto inputs = ConstructInputsForTFLitePlan(
        client_plan.phase().local_compute(), input_dir_uri, output_dir_uri);
    engine::TfLitePlanEngine plan_engine(env_deps, log_manager, event_publisher,
                                         opstats_logger, &timing_config, flags);
    engine::PlanResult plan_result = plan_engine.RunPlan(
        client_plan.phase().tensorflow_spec(), client_plan.tflite_graph(),
        std::move(inputs), output_names_unused, run_plan_start_time,
        reference_time, log_computation_started, log_computation_finished,
        selector_context);
    engine::PlanOutcome outcome = plan_result.outcome;
    LogComputationOutcome(std::move(plan_result), phase_logger,
                          run_plan_start_time, reference_time);
    return ConvertPlanOutcomeToStatus(outcome);
  }
#endif

  // Construct input tensors based on the values in the LocalComputeIORouter
  // message.
  auto inputs = ConstructInputsForTensorflowSpecPlan(
      client_plan.phase().local_compute(), input_dir_uri, output_dir_uri);
  engine::SimplePlanEngine plan_engine(env_deps, log_manager, event_publisher,
                                       opstats_logger, &timing_config, flags);
  engine::PlanResult plan_result = plan_engine.RunPlan(
      client_plan.phase().tensorflow_spec(), client_plan.graph(),
      client_plan.tensorflow_config_proto(), std::move(inputs),
      output_names_unused, run_plan_start_time, reference_time,
      log_computation_started, log_computation_finished, selector_context);
  engine::PlanOutcome outcome = plan_result.outcome;
  LogComputationOutcome(std::move(plan_result), phase_logger,
                        run_plan_start_time, reference_time);
  return ConvertPlanOutcomeToStatus(outcome);
}
}  // anonymous namespace

absl::Status RunLocalComputation(SimpleTaskEnvironment* env_deps,
                                 EventPublisher* event_publisher,
                                 LogManager* log_manager, const Flags* flags,
                                 const std::string& session_name,
                                 const std::string& plan_uri,
                                 const std::string& input_dir_uri,
                                 const std::string& output_dir_uri) {
  auto opstats_logger = engine::CreateOpStatsLogger(
      env_deps->GetBaseDir(), flags, log_manager, session_name,
      /*population_name=*/"");
  SelectorContext selector_context;
  selector_context.mutable_computation_properties()->set_session_name(
      session_name);
  *selector_context.mutable_computation_properties()->mutable_local_compute() =
      LocalComputation();
  PhaseLoggerImpl phase_logger(event_publisher, opstats_logger.get(),
                               log_manager, flags);
  return RunLocalComputation(phase_logger, env_deps, event_publisher,
                             log_manager, opstats_logger.get(), flags, plan_uri,
                             input_dir_uri, output_dir_uri, selector_context);
}

absl::Status RunLocalComputation(
    PhaseLogger& phase_logger, SimpleTaskEnvironment* env_deps,
    EventPublisher* event_publisher, LogManager* log_manager,
    OpStatsLogger* opstats_logger, const Flags* flags,
    const std::string& plan_uri, const std::string& input_dir_uri,
    const std::string& output_dir_uri,
    const SelectorContext& selector_context) {
  absl::Time reference_time = absl::Now();
  absl::Duration polling_period =
      absl::Milliseconds(flags->condition_polling_period_millis());
  // Check if the device conditions allow running a local computation.
  if (env_deps->ShouldAbort(reference_time, polling_period)) {
    std::string message =
        "Device conditions not satisfied, aborting local computation";
    FCP_LOG(INFO) << message;
    phase_logger.LogTaskNotStarted(message);
    return absl::CancelledError("");
  }
  fcp::client::InterruptibleRunner::TimingConfig timing_config = {
      .polling_period = polling_period,
      .graceful_shutdown_period = absl::Milliseconds(
          flags->tf_execution_teardown_grace_period_millis()),
      .extended_shutdown_period = absl::Milliseconds(
          flags->tf_execution_teardown_extended_period_millis()),
  };

  absl::Time run_plan_start_time = absl::Now();
  phase_logger.LogComputationStarted();

  absl::StatusOr<std::string> plan_str = fcp::ReadFileToString(plan_uri);
  if (!plan_str.ok()) {
    phase_logger.LogComputationIOError(plan_str.status());
    return plan_str.status();
  }

  ClientOnlyPlan plan;
  if (!plan.ParseFromString(*plan_str)) {
    absl::Status error_status =
        absl::InvalidArgumentError("could not parse received plan");
    phase_logger.LogComputationInvalidArgument(error_status);
    return error_status;
  }

  std::vector<std::string> output_names;
  std::vector<tensorflow::Tensor> output_tensors;
  return RunPlanWithTensorflowSpec(
      phase_logger, env_deps, event_publisher, log_manager, opstats_logger,
      flags, plan, input_dir_uri, output_dir_uri, timing_config,
      run_plan_start_time, reference_time, selector_context);
}

}  // namespace client
}  // namespace fcp
