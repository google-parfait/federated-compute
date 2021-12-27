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
#include "fcp/client/fl_runner.h"

#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/platform.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/engine/plan_engine.h"
#include "fcp/client/engine/plan_engine_helpers.h"
#include "fcp/client/engine/simple_plan_engine.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/federated_task_environment.h"
#include "fcp/client/files.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/fl_runner_internal.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/grpc_federated_protocol.h"
#include "fcp/client/histogram_counters.pb.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/selector_context.pb.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/opstats.pb.h"
#include "fcp/protos/plan.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace fcp {
namespace client {

using ::fcp::client::opstats::OperationalStats;
using ::fcp::client::opstats::OpStatsLogger;
using ::google::internal::federated::plan::AggregationConfig;
using ::google::internal::federated::plan::ClientOnlyPlan;
using ::google::internal::federated::plan::FederatedComputeEligibilityIORouter;
using ::google::internal::federated::plan::FederatedComputeIORouter;
using ::google::internal::federated::plan::TensorflowSpec;
using ::google::internal::federatedml::v2::RetryWindow;
using ::google::internal::federatedml::v2::TaskEligibilityInfo;

namespace {

template <typename T>
void AddValuesToQuantized(QuantizedTensor* quantized,
                          const tensorflow::Tensor& tensor) {
  auto flat_tensor = tensor.flat<T>();
  quantized->values.reserve(quantized->values.size() + flat_tensor.size());
  for (int i = 0; i < flat_tensor.size(); i++) {
    quantized->values.push_back(flat_tensor(i));
  }
}

struct PlanResultAndCheckpointFile {
  explicit PlanResultAndCheckpointFile(engine::PlanResult plan_result)
      : plan_result(std::move(plan_result)) {}
  engine::PlanResult plan_result;
  std::string checkpoint_file;

  PlanResultAndCheckpointFile(PlanResultAndCheckpointFile&&) = default;
  PlanResultAndCheckpointFile& operator=(PlanResultAndCheckpointFile&&) =
      default;

  // Disallow copy and assign.
  PlanResultAndCheckpointFile(const PlanResultAndCheckpointFile&) = delete;
  PlanResultAndCheckpointFile& operator=(const PlanResultAndCheckpointFile&) =
      delete;
};

absl::StatusOr<ComputationResults> CreateComputationResults(
    const TensorflowSpec& tensorflow_spec,
    PlanResultAndCheckpointFile plan_result_and_checkpoint_file) {
  const auto& [plan_result, checkpoint_file] = plan_result_and_checkpoint_file;
  ComputationResults computation_results;
  for (int i = 0; i < plan_result.output_names.size(); i++) {
    QuantizedTensor quantized;
    const auto& output_tensor = plan_result.output_tensors[i];
    switch (output_tensor.dtype()) {
      case tensorflow::DT_INT8:
        AddValuesToQuantized<int8_t>(&quantized, output_tensor);
        quantized.bitwidth = 7;
        break;
      case tensorflow::DT_UINT8:
        AddValuesToQuantized<uint8_t>(&quantized, output_tensor);
        quantized.bitwidth = 8;
        break;
      case tensorflow::DT_INT16:
        AddValuesToQuantized<int16_t>(&quantized, output_tensor);
        quantized.bitwidth = 15;
        break;
      case tensorflow::DT_UINT16:
        AddValuesToQuantized<uint16_t>(&quantized, output_tensor);
        quantized.bitwidth = 16;
        break;
      case tensorflow::DT_INT32:
        AddValuesToQuantized<int32_t>(&quantized, output_tensor);
        quantized.bitwidth = 31;
        break;
      case tensorflow::DT_INT64:
        AddValuesToQuantized<tensorflow::int64>(&quantized, output_tensor);
        quantized.bitwidth = 62;
        break;
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Tensor of type", tensorflow::DataType_Name(output_tensor.dtype()),
            "could not be converted to quantized value"));
    }
    computation_results[plan_result.output_names[i]] = std::move(quantized);
  }

  // Add dimensions to QuantizedTensors.
  for (const tensorflow::TensorSpecProto& tensor_spec :
       tensorflow_spec.output_tensor_specs()) {
    if (computation_results.find(tensor_spec.name()) !=
        computation_results.end()) {
      for (const tensorflow::TensorShapeProto_Dim& dim :
           tensor_spec.shape().dim()) {
        absl::get<QuantizedTensor>(computation_results[tensor_spec.name()])
            .dimensions.push_back(dim.size());
      }
    }
  }

  // Name of the TF checkpoint inside the aggregand map in the Checkpoint
  // protobuf. This field name is ignored by the server.
  if (!checkpoint_file.empty()) {
    FCP_ASSIGN_OR_RETURN(std::string tf_checkpoint,
                         fcp::ReadFileToString(checkpoint_file));
    computation_results[std::string(kTensorflowCheckpointAggregand)] =
        std::move(tf_checkpoint);
  }
  return computation_results;
}

std::unique_ptr<std::vector<std::pair<std::string, tensorflow::Tensor>>>
ConstructInputsForEligibilityEvalPlan(
    const FederatedComputeEligibilityIORouter& io_router,
    const std::string& checkpoint_input_filename) {
  auto inputs = std::make_unique<
      std::vector<std::pair<std::string, tensorflow::Tensor>>>();
  if (!io_router.input_filepath_tensor_name().empty()) {
    tensorflow::Tensor input_filepath(tensorflow::DT_STRING, {});
    input_filepath.scalar<tensorflow::tstring>()() = checkpoint_input_filename;
    inputs->push_back({io_router.input_filepath_tensor_name(), input_filepath});
  }
  return inputs;
}

engine::PlanResult RunEligibilityEvalPlanWithTensorflowSpec(
    SimpleTaskEnvironment* env_deps, EventPublisher* event_publisher,
    LogManager* log_manager, OpStatsLogger* opstats_logger, const Flags* flags,
    const ClientOnlyPlan& client_plan,
    const std::string& checkpoint_input_filename,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time run_plan_start_time, const absl::Time reference_time,
    const SelectorContext& eligibility_selector_context) {
  // Check that this is a TensorflowSpec-based plan for federated eligibility
  // computation.
  if (!client_plan.phase().has_tensorflow_spec() ||
      !client_plan.phase().has_federated_compute_eligibility() ||
      client_plan.phase().execution_size() > 0) {
    log_manager->LogDiag(
        ProdDiagCode::BACKGROUND_TRAINING_FAILED_PLAN_FAILS_SANITY_CHECK);
    std::string message = "Invalid eligibility eval plan";
    event_publisher->PublishIoError(0, message);
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
    return engine::PlanResult(engine::PhaseOutcome::ERROR);
  }

  const FederatedComputeEligibilityIORouter& io_router =
      client_plan.phase().federated_compute_eligibility();

  std::vector<std::string> output_names = {
      io_router.task_eligibility_info_tensor_name()};

  auto log_computation_started = [opstats_logger]() {
    opstats_logger->AddEvent(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_COMPUTATION_STARTED);
  };
  auto log_computation_finished = [opstats_logger]() {
    opstats_logger->AddEvent(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_COMPUTATION_FINISHED);
  };

  // Construct input tensors and output tensor names based on the values in the
  // FederatedComputeEligibilityIORouter message.
  auto inputs = ConstructInputsForEligibilityEvalPlan(
      io_router, checkpoint_input_filename);
  // Run plan and get a set of output tensors back.
  engine::SimplePlanEngine plan_engine(env_deps, log_manager, event_publisher,
                                       opstats_logger, &timing_config, flags);
  return plan_engine.RunPlan(
      client_plan.phase().tensorflow_spec(), client_plan.graph(),
      client_plan.tensorflow_config_proto(), std::move(inputs), output_names,
      run_plan_start_time, reference_time, log_computation_started,
      log_computation_finished, eligibility_selector_context);
}

// Validates the output tensors that resulted from executing the plan, and then
// parses the output into a TaskEligibilityInfo proto. Returns an error if
// validation or parsing failed.
absl::StatusOr<TaskEligibilityInfo> ParseEligibilityEvalPlanOutput(
    const std::vector<tensorflow::Tensor>& output_tensors) {
  auto output_size = output_tensors.size();
  if (output_size != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected number of output tensors: ", output_size));
  }
  auto output_elements = output_tensors[0].NumElements();
  if (output_elements != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unexpected number of output tensor elements: ", output_elements));
  }
  tensorflow::DataType output_type = output_tensors[0].dtype();
  if (output_type != tensorflow::DT_STRING) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected output tensor type: ", output_type));
  }

  // Extract the serialized TaskEligibilityInfo proto from the tensor and
  // parse it.
  // First, convert the output Tensor into a Scalar (= a TensorMap with 1
  // element), then use its operator() to access the actual data.
  const tensorflow::tstring& serialized_output =
      output_tensors[0].scalar<const tensorflow::tstring>()();
  TaskEligibilityInfo parsed_output;
  if (!parsed_output.ParseFromString(serialized_output)) {
    return absl::InvalidArgumentError("Could not parse output proto");
  }
  return parsed_output;
}

std::unique_ptr<std::vector<std::pair<std::string, tensorflow::Tensor>>>
ConstructInputsForTensorflowSpecPlan(
    const FederatedComputeIORouter& io_router,
    const std::string& checkpoint_input_filename,
    const std::string& checkpoint_output_filename) {
  auto inputs = std::make_unique<
      std::vector<std::pair<std::string, tensorflow::Tensor>>>();
  if (!io_router.input_filepath_tensor_name().empty()) {
    tensorflow::Tensor input_filepath(tensorflow::DT_STRING, {});
    input_filepath.scalar<tensorflow::tstring>()() = checkpoint_input_filename;
    inputs->push_back({io_router.input_filepath_tensor_name(), input_filepath});
  }

  if (!io_router.output_filepath_tensor_name().empty()) {
    tensorflow::Tensor output_filepath(tensorflow::DT_STRING, {});
    output_filepath.scalar<tensorflow::tstring>()() =
        checkpoint_output_filename;
    inputs->push_back(
        {io_router.output_filepath_tensor_name(), output_filepath});
  }

  return inputs;
}

absl::StatusOr<std::vector<std::string>> ConstructOutputsForTensorflowSpecPlan(
    EventPublisher* event_publisher, LogManager* log_manager,
    OpStatsLogger* opstats_logger, const FederatedComputeIORouter& io_router) {
  std::vector<std::string> output_names;
  for (const google::protobuf::MapPair<std::string, AggregationConfig>& it :
       io_router.aggregations()) {
    output_names.push_back(it.first);
    // The only aggregation type currently supported is secure aggregation.
    if (!it.second.has_secure_aggregation()) {
      log_manager->LogDiag(
          ProdDiagCode::BACKGROUND_TRAINING_FAILED_PLAN_FAILS_SANITY_CHECK);
      std::string message =
          "Unsupported aggregation type in TensorflowSpec-based plan";
      event_publisher->PublishIoError(0, message);
      opstats_logger->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
      return absl::InvalidArgumentError("");
    }
  }

  return output_names;
}

PlanResultAndCheckpointFile RunPlanWithTensorflowSpec(
    SimpleTaskEnvironment* env_deps, EventPublisher* event_publisher,
    Files* files, LogManager* log_manager, OpStatsLogger* opstats_logger,
    const Flags* flags, const ClientOnlyPlan& client_plan,
    const std::string& checkpoint_input_filename,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time run_plan_start_time, const absl::Time reference_time,
    const SelectorContext& selector_context) {
  PlanResultAndCheckpointFile result(
      (engine::PlanResult(engine::PhaseOutcome::ERROR)));

  // Check that this is a TensorflowSpec-based plan for federated computation.
  FCP_CHECK(client_plan.phase().has_tensorflow_spec());
  if (!client_plan.phase().has_federated_compute() ||
      client_plan.phase().execution_size() > 0) {
    log_manager->LogDiag(
        ProdDiagCode::BACKGROUND_TRAINING_FAILED_PLAN_FAILS_SANITY_CHECK);
    std::string message = "Invalid TensorflowSpec-based plan";
    event_publisher->PublishIoError(0, message);
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
    return result;
  }

  // Construct input tensors based on the values in the
  // FederatedComputeIORouter message and create a temporary file for the output
  // checkpoint if needed.
  absl::StatusOr<std::string> checkpoint_output_filename =
      files->CreateTempFile("output", ".ckp");
  if (!checkpoint_output_filename.ok()) {
    auto status = checkpoint_output_filename.status();
    auto message = absl::StrCat(
        "Could not create temporary output checkpoint file: code: ",
        status.code(), ", message: ", status.message());
    event_publisher->PublishIoError(0, message);
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
    return result;
  }
  auto inputs = ConstructInputsForTensorflowSpecPlan(
      client_plan.phase().federated_compute(), checkpoint_input_filename,
      *checkpoint_output_filename);

  // Get the output tensor names.
  absl::StatusOr<std::vector<std::string>> output_names =
      ConstructOutputsForTensorflowSpecPlan(
          event_publisher, log_manager, opstats_logger,
          client_plan.phase().federated_compute());
  if (!output_names.ok()) {
    return result;
  }

  auto log_computation_started = [opstats_logger]() {
    opstats_logger->AddEvent(
        OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED);
  };
  auto log_computation_finished = [opstats_logger]() {
    opstats_logger->AddEvent(
        OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED);
  };

  // Run plan and get a set of output tensors back.
  engine::PlanResult plan_result(engine::PhaseOutcome::ERROR);

  engine::SimplePlanEngine plan_engine(env_deps, log_manager, event_publisher,
                                       opstats_logger, &timing_config, flags);
  plan_result = plan_engine.RunPlan(
      client_plan.phase().tensorflow_spec(), client_plan.graph(),
      client_plan.tensorflow_config_proto(), std::move(inputs), *output_names,
      run_plan_start_time, reference_time, log_computation_started,
      log_computation_finished, selector_context);

  result.plan_result = std::move(plan_result);
  result.checkpoint_file = *checkpoint_output_filename;

  return result;
}

absl::Status ReportTensorflowSpecPlanResult(
    EventPublisher* event_publisher, LogManager* log_manager,
    FederatedProtocol* federated_protocol, OpStatsLogger* opstats_logger,
    const TensorflowSpec& tensorflow_spec,
    PlanResultAndCheckpointFile plan_result_and_checkpoint_file,
    absl::Time run_plan_start_time, absl::Time reference_time) {
  engine::PhaseOutcome outcome =
      plan_result_and_checkpoint_file.plan_result.outcome;
  const absl::Time before_report_time = absl::Now();
  absl::Status result = absl::InternalError("");
  if (outcome == engine::INTERRUPTED) {
    // If plan execution got interrupted, we do not report to the server, and
    // do not wait for a reply, but bail fast to get out of the way. We use
    // OK rather than ABORTED to ensure that the result isn't handled as if an
    // additional, new network error occurred. Instead, by using OK, we simply
    // act as if the reporting succeeded.
    result = absl::OkStatus();
  } else if (outcome == engine::COMPLETED) {
    absl::StatusOr<ComputationResults> computation_results_or =
        CreateComputationResults(tensorflow_spec,
                                 std::move(plan_result_and_checkpoint_file));
    if (!computation_results_or.ok()) {
      auto status = computation_results_or.status();
      auto message = absl::StrCat(
          "Unable to create computation results from TensorflowSpec-based plan "
          "outputs. code: ",
          status.code(), ", message: ", status.message());
      event_publisher->PublishTensorFlowError(
          /*execution_index=*/0, /*epoch_index=*/0, /*epoch_example_index=*/0,
          message);
      opstats_logger->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW, message);
      result = federated_protocol->ReportNotCompleted(
          engine::PhaseOutcome::ERROR,
          /*plan_duration=*/absl::Now() - run_plan_start_time);
    } else {
      result = federated_protocol->ReportCompleted(
          std::move(computation_results_or).value(),
          /*stats=*/std::vector<std::pair<std::string, double>>(),
          /*plan_duration=*/absl::Now() - run_plan_start_time);
    }
  } else {
    result = federated_protocol->ReportNotCompleted(
        engine::PhaseOutcome::ERROR,
        /*plan_duration=*/absl::Now() - run_plan_start_time);
  }
  const absl::Time after_report_time = absl::Now();
  log_manager->LogToLongHistogram(
      HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME,
      absl::ToInt64Milliseconds(after_report_time - reference_time));
  log_manager->LogToLongHistogram(
      HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY,
      absl::ToInt64Milliseconds(after_report_time - before_report_time));
  return result;
}

bool RunPlanWithExecutions(
    SimpleTaskEnvironment* env_deps, EventPublisher* event_publisher,
    Files* files, LogManager* log_manager, OpStatsLogger* opstats_logger,
    const Flags* flags, FederatedProtocol* federated_protocol,
    const ClientOnlyPlan& client_plan,
    const std::string& checkpoint_input_filename,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time reference_time) {
  // Create a task environment.
  auto env = std::make_unique<FederatedTaskEnvironment>(
      env_deps, federated_protocol, log_manager, event_publisher,
      opstats_logger, flags, reference_time,
      absl::Milliseconds(flags->condition_polling_period_millis()));

  // Run plan. The task environment will report results back to server via the
  // underlying federated protocol.
  engine::PlanEngine plan_engine;
  return plan_engine.RunPlan(
      env.get(), files, log_manager, event_publisher, opstats_logger,
      client_plan, checkpoint_input_filename, timing_config, reference_time,
      flags->log_tensorflow_error_messages());
}

// Updates the fields of `FLRunnerResult` that should always be updated after
// each interaction with the `FederatedProtocol` object.
void UpdateRetryWindowAndNetworkStats(FederatedProtocol& federated_protocol,
                                      OpStatsLogger* opstats_logger,
                                      FLRunnerResult& fl_runner_result) {
  // Update the result's retry window to the most recent one.
  auto retry_window = federated_protocol.GetLatestRetryWindow();
  *fl_runner_result.mutable_retry_window() = retry_window;
  opstats_logger->SetRetryWindow(retry_window);
}

// Writes the given checkpoint data to a newly created temporary file.
// Returns the filename if successful, or an error if the file could not be
// created, or if writing to the file failed.
absl::StatusOr<std::string> CreateInputCheckpointFile(
    Files* files, const std::string& checkpoint) {
  // Create the temporary checkpoint file.
  // Deletion of the file is left to the caller / the Files implementation.
  FCP_ASSIGN_OR_RETURN(absl::StatusOr<std::string> filename,
                       files->CreateTempFile("init", ".ckp"));
  // Write the checkpoint data to the file.
  std::fstream checkpoint_stream(*filename, std::ios_base::out);
  if (checkpoint_stream.fail() || !(checkpoint_stream << checkpoint).good()) {
    return absl::InvalidArgumentError("Failed to write to file");
  }
  checkpoint_stream.close();
  return filename;
}

// Issues an eligibility eval checkin request and executes the eligibility eval
// task if the server returns one.
//
// This function modifies the FLRunnerResult with values received over the
// course of the eligibility eval protocol interaction.
//
// Returns:
// - the TaskEligibilityInfo produced by the eligibility eval task, if the
//   server provided an eligibility eval task to run.
// - an std::nullopt if the server indicated that there is no eligibility eval
//   task configured for the population.
// - an INTERNAL error if the server rejects the client or another error occurs
//   that should abort the training run. The error will already have been logged
//   appropriately.
absl::StatusOr<std::optional<TaskEligibilityInfo>>
IssueEligibilityEvalCheckinAndRunPlan(
    SimpleTaskEnvironment* env_deps, EventPublisher* event_publisher,
    Files* files, LogManager* log_manager, OpStatsLogger* opstats_logger,
    const Flags* flags, FederatedProtocol* federated_protocol,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time reference_time, FLRunnerResult& fl_runner_result,
    const SelectorContext& selector_context) {
  absl::Time time_before_eligibility_eval_checkin = absl::Now();
  // Log that we are about to check in with the server.
  if (flags->per_phase_logs()) {
    event_publisher->PublishEligibilityEvalCheckin();
    opstats_logger->AddEvent(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED);
  }
  // Issue the eligibility eval checkin request.
  // The EligibilityEvalCheckin(...) method will call SetModelIdentifier() with
  // the name of whatever task it receives, ensuring that subsequent events will
  // be tagged with that identifier.
  absl::StatusOr<FederatedProtocol::EligibilityEvalCheckinResult>
      eligibility_checkin_result = federated_protocol->EligibilityEvalCheckin();
  UpdateRetryWindowAndNetworkStats(*federated_protocol, opstats_logger,
                                   fl_runner_result);

  log_manager->LogToLongHistogram(
      HistogramCounters::TRAINING_FL_ELIGIBILITY_EVAL_CHECKIN_LATENCY, 0, 0,
      engine::DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
      absl::ToInt64Milliseconds(absl::Now() -
                                time_before_eligibility_eval_checkin));

  if (!eligibility_checkin_result.ok()) {
    auto status = eligibility_checkin_result.status();
    auto message = absl::StrCat("Error during eligibility eval checkin: code: ",
                                status.code(), ", message: ", status.message());
    event_publisher->PublishIoError(0, message);
    engine::LogOpStatsNetworkErrors(
        opstats_logger, eligibility_checkin_result.status(), message);
    FCP_LOG(INFO) << message;
    return absl::InternalError("");
  }

  if (std::holds_alternative<FederatedProtocol::Rejection>(
          *eligibility_checkin_result)) {
    if (flags->per_phase_logs()) {
      event_publisher->PublishEligibilityEvalRejected(
          federated_protocol->bytes_downloaded(),
          federated_protocol->chunking_layer_bytes_received(),
          absl::Now() - time_before_eligibility_eval_checkin);
      opstats_logger->AddEvent(
          OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED);
    }
    // If the server explicitly rejected our request, then we must abort and
    // we must not proceed to the "checkin" phase below.
    FCP_LOG(INFO) << "Device rejected by server during eligibility eval "
                     "checkin; aborting";
    return absl::InternalError("");
  } else if (std::holds_alternative<FederatedProtocol::EligibilityEvalDisabled>(
                 *eligibility_checkin_result)) {
    if (flags->per_phase_logs()) {
      event_publisher->PublishEligibilityEvalNotConfigured(
          federated_protocol->bytes_downloaded(),
          federated_protocol->chunking_layer_bytes_received(),
          absl::Now() - time_before_eligibility_eval_checkin);
      opstats_logger->AddEvent(
          OperationalStats::Event::EVENT_KIND_ELIGIBILITY_DISABLED);
    }
    // If the server indicates that no eligibility eval task is configured for
    // the population then there is nothing more to do. We simply proceed to
    // the "checkin" phase below without providing it a TaskEligibilityInfo
    // proto.
    return std::nullopt;
  }

  // Run the eligibility eval task if the server returned one.
  auto eligibility_eval_payload =
      absl::get<FederatedProtocol::CheckinResultPayload>(
          *eligibility_checkin_result);
  auto eligibility_selector_context = SelectorContext(selector_context);
  *eligibility_selector_context.mutable_computation_properties()
       ->mutable_eligibility_eval() = EligibilityEvalComputation();

  absl::StatusOr<std::string> checkpoint_input_filename =
      CreateInputCheckpointFile(files, eligibility_eval_payload.checkpoint);
  if (!checkpoint_input_filename.ok()) {
    auto status = checkpoint_input_filename.status();
    auto message = absl::StrCat(
        "Failed to create eligibility eval checkpoint input file: code: ",
        status.code(), ", message: ", status.message());
    event_publisher->PublishIoError(0, message);
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
    FCP_LOG(ERROR) << message;
    return absl::InternalError("");
  }

  if (flags->per_phase_logs()) {
    opstats_logger->AddEvent(
        OperationalStats::Event::EVENT_KIND_ELIGIBILITY_ENABLED);
    event_publisher->PublishEligibilityEvalPlanReceived(
        federated_protocol->bytes_downloaded(),
        federated_protocol->chunking_layer_bytes_received(),
        absl::Now() - time_before_eligibility_eval_checkin);
  }

  absl::Time run_plan_start_time = absl::Now();
  engine::PlanResult plan_result = RunEligibilityEvalPlanWithTensorflowSpec(
      env_deps, event_publisher, log_manager, opstats_logger, flags,
      eligibility_eval_payload.client_only_plan, *checkpoint_input_filename,
      timing_config, run_plan_start_time, reference_time, selector_context);
  if (plan_result.outcome != engine::PhaseOutcome::COMPLETED) {
    // If eligibility eval plan execution failed then we can't proceed to
    // the checkin phase, since we'll have no TaskEligibilityInfo to
    // provide. All we can do is abort and reschedule. An error will already
    // have been published to EventPublisher by SimplePlanEngine.
    return absl::InternalError("");
  }
  absl::StatusOr<TaskEligibilityInfo> parsed_output =
      ParseEligibilityEvalPlanOutput(plan_result.output_tensors);
  if (!parsed_output.ok()) {
    auto status = parsed_output.status();
    auto message = absl::StrCat("Invalid eligibility eval plan output: code: ",
                                status.code(), ", message: ", status.message());
    event_publisher->PublishTensorFlowError(
        /*execution_index=*/0, /*epoch_index=*/0, /*epoch_example_index=*/0,
        message);
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW, message);
    FCP_LOG(ERROR) << message;
    return absl::InternalError("");
  }
  return std::move(*parsed_output);
}

}  // namespace

absl::StatusOr<FLRunnerResult> RunFederatedComputation(
    SimpleTaskEnvironment* env_deps, EventPublisher* event_publisher,
    Files* files, LogManager* log_manager, const Flags* flags,
    const std::string& federated_service_uri, const std::string& api_key,
    const std::string& test_cert_path, const std::string& session_name,
    const std::string& population_name, const std::string& retry_token,
    const std::string& client_version,
    const std::string& attestation_measurement) {
  auto opstats_logger =
      engine::CreateOpStatsLogger(env_deps->GetBaseDir(), flags, log_manager,
                                  session_name, population_name);

  absl::Time reference_time = absl::Now();
  FLRunnerResult fl_runner_result;
  fcp::client::InterruptibleRunner::TimingConfig timing_config = {
      .polling_period =
          absl::Milliseconds(flags->condition_polling_period_millis()),
      .graceful_shutdown_period = absl::Milliseconds(
          flags->tf_execution_teardown_grace_period_millis()),
      .extended_shutdown_period = absl::Milliseconds(
          flags->tf_execution_teardown_extended_period_millis()),
  };

  auto should_abort_protocol_callback = [&env_deps, &timing_config]() -> bool {
    // Return the Status if failed, or the negated value if successful.
    return env_deps->ShouldAbort(absl::Now(), timing_config.polling_period);
  };

  // Check in with the server to either retrieve a plan + initial checkpoint, or
  // get rejected with a RetryWindow.
  auto grpc_channel_deadline = flags->grpc_channel_deadline_seconds();
  if (grpc_channel_deadline <= 0) {
    grpc_channel_deadline = 600;
    FCP_LOG(INFO) << "Using default channel deadline of "
                  << grpc_channel_deadline << " seconds.";
  }
  SelectorContext federated_selector_context;
  federated_selector_context.mutable_computation_properties()->set_session_name(
      session_name);
  FederatedComputation federated_computation;
  federated_computation.set_population_name(population_name);
  *federated_selector_context.mutable_computation_properties()
       ->mutable_federated() = federated_computation;

  SelectorContext eligibility_selector_context;
  eligibility_selector_context.mutable_computation_properties()
      ->set_session_name(session_name);
  EligibilityEvalComputation eligibility_eval_computation;
  eligibility_eval_computation.set_population_name(population_name);
  *eligibility_selector_context.mutable_computation_properties()
       ->mutable_eligibility_eval() = eligibility_eval_computation;

  GrpcFederatedProtocol federated_protocol(
      event_publisher, log_manager, opstats_logger.get(), flags,
      federated_service_uri, api_key, test_cert_path, population_name,
      retry_token, client_version, attestation_measurement,
      should_abort_protocol_callback, timing_config, grpc_channel_deadline);
  return RunFederatedComputation(
      env_deps, event_publisher, files, log_manager, opstats_logger.get(),
      flags, &federated_protocol, timing_config, reference_time,
      eligibility_selector_context, federated_selector_context);
}

absl::StatusOr<FLRunnerResult> RunFederatedComputation(
    SimpleTaskEnvironment* env_deps, EventPublisher* event_publisher,
    Files* files, LogManager* log_manager,
    ::fcp::client::opstats::OpStatsLogger* opstats_logger, const Flags* flags,
    FederatedProtocol* federated_protocol,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time reference_time,
    const SelectorContext& eligibility_selector_context,
    const SelectorContext& federated_selector_context) {
  // Construct a default FLRunnerResult that reflects an unsuccessful training
  // attempt and which uses RetryWindow corresponding to transient errors (if
  // the flag is on).
  // This is what will be returned if we have to bail early, before we've
  // received a RetryWindow from the server.
  FLRunnerResult fl_runner_result;
  fl_runner_result.set_contribution_result(FLRunnerResult::FAIL);
  // Before we even check whether we should abort right away, update the retry
  // window. That way we will use the most appropriate retry window we have
  // available (an implementation detail of FederatedProtocol, but generally a
  // 'transient error' retry window based on the provided flag values) in case
  // we do need to abort.
  UpdateRetryWindowAndNetworkStats(*federated_protocol, opstats_logger,
                                   fl_runner_result);

  // Check if the device conditions allow for checking in with the server
  // and running a federated computation. If not, bail early with the
  // transient error retry window.
  bool should_abort =
      env_deps->ShouldAbort(absl::Now(), timing_config.polling_period);
  if (should_abort) {
    std::string message =
        "Device conditions not satisfied, aborting federated computation";
    FCP_LOG(INFO) << message;
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED, message);
    return fl_runner_result;
  }

  std::optional<TaskEligibilityInfo> task_eligibility_info = std::nullopt;
  // Note that this method will update fl_runner_result's fields with values
  // received over the course of the eligibility eval protocol interaction.
  absl::StatusOr<std::optional<TaskEligibilityInfo>> eligibility_eval_result =
      IssueEligibilityEvalCheckinAndRunPlan(
          env_deps, event_publisher, files, log_manager, opstats_logger, flags,
          federated_protocol, timing_config, reference_time, fl_runner_result,
          eligibility_selector_context);
  if (!eligibility_eval_result.ok()) {
    return fl_runner_result;
  }
  task_eligibility_info = std::move(*eligibility_eval_result);

  absl::Time time_before_checkin = absl::Now();
  // Clear the model identifier before check-in, to ensure that the any prior
  // eligibility eval task name isn't used any longer.
  event_publisher->SetModelIdentifier("");
  log_manager->SetModelIdentifier("");
  if (flags->per_phase_logs()) {
    // Log that we are about to check in with the server.
    event_publisher->PublishCheckin();
    opstats_logger->AddEvent(
        OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED);
  }

  // The Checkin(...) method will call SetModelIdentifier() with the name of
  // whatever task it receives, ensuring that subsequent events will be tagged
  // with that identifier.
  absl::StatusOr<FederatedProtocol::CheckinResult> checkin_result =
      federated_protocol->Checkin(task_eligibility_info);
  UpdateRetryWindowAndNetworkStats(*federated_protocol, opstats_logger,
                                   fl_runner_result);
  log_manager->LogToLongHistogram(
      HistogramCounters::TRAINING_FL_CHECKIN_END_TIME, 0, 0,
      engine::DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
      absl::ToInt64Milliseconds(absl::Now() - reference_time));
  log_manager->LogToLongHistogram(
      HistogramCounters::TRAINING_FL_CHECKIN_LATENCY, 0, 0,
      engine::DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
      absl::ToInt64Milliseconds(absl::Now() - time_before_checkin));
  if (!checkin_result.ok()) {
    auto status = checkin_result.status();
    auto message = absl::StrCat("Error during checkin: code: ", status.code(),
                                ", message: ", status.message());
    event_publisher->PublishIoError(0, message);
    engine::LogOpStatsNetworkErrors(opstats_logger, checkin_result.status(),
                                    message);
    FCP_LOG(INFO) << message;
    return fl_runner_result;
  }

  // Server rejected us? Return the fl_runner_results as-is.
  if (std::holds_alternative<FederatedProtocol::Rejection>(*checkin_result)) {
    if (flags->per_phase_logs()) {
      event_publisher->PublishRejected();
      opstats_logger->AddEvent(
          OperationalStats::Event::EVENT_KIND_CHECKIN_REJECTED);
    }
    FCP_LOG(INFO) << "Device rejected by server during checkin; aborting";
    return fl_runner_result;
  }

  auto acceptance =
      absl::get<FederatedProtocol::CheckinResultPayload>(*checkin_result);
  absl::StatusOr<std::string> checkpoint_input_filename =
      CreateInputCheckpointFile(files, acceptance.checkpoint);
  if (!checkpoint_input_filename.ok()) {
    auto status = checkpoint_input_filename.status();
    auto message = absl::StrCat(
        "Failed to create checkpoint input file: code: ", status.code(),
        ", message: ", status.message());
    event_publisher->PublishIoError(0, message);
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
    FCP_LOG(ERROR) << message;
    return fl_runner_result;
  }
  if (flags->per_phase_logs()) {
    event_publisher->PublishCheckinFinished(
        federated_protocol->bytes_downloaded(),
        federated_protocol->chunking_layer_bytes_received(),
        absl::Now() - time_before_checkin);
    opstats_logger->AddCheckinAcceptedEventWithTaskName(acceptance.task_name);
  }

  SelectorContext federated_selector_context_with_task_name =
      federated_selector_context;
  federated_selector_context_with_task_name.mutable_computation_properties()
      ->mutable_federated()
      ->set_task_name(acceptance.task_name);

  const auto& federated_compute_io_router =
      acceptance.client_only_plan.phase().federated_compute();
  const bool has_simpleagg_tensors =
      !federated_compute_io_router.output_filepath_tensor_name().empty();
  bool all_aggregations_are_secagg = true;
  for (const auto& aggregation : federated_compute_io_router.aggregations()) {
    all_aggregations_are_secagg &= aggregation.second.protocol_config_case() ==
                                   AggregationConfig::kSecureAggregation;
  }
  if (!has_simpleagg_tensors && all_aggregations_are_secagg) {
    federated_selector_context_with_task_name.mutable_computation_properties()
        ->mutable_federated()
        ->mutable_secure_aggregation()
        ->set_minimum_clients_in_server_visible_aggregate(
            acceptance.minimum_clients_in_server_visible_aggregate);
  } else {
    // Has an output checkpoint, so some tensors must be simply aggregated.
    *(federated_selector_context_with_task_name
          .mutable_computation_properties()
          ->mutable_federated()
          ->mutable_simple_aggregation()) = SimpleAggregation();
  }

  RetryWindow report_retry_window;
  if (acceptance.client_only_plan.phase().has_tensorflow_spec()) {
    absl::Time run_plan_start_time = absl::Now();
    PlanResultAndCheckpointFile plan_result_and_checkpoint_file =
        RunPlanWithTensorflowSpec(
            env_deps, event_publisher, files, log_manager, opstats_logger,
            flags, acceptance.client_only_plan, *checkpoint_input_filename,
            timing_config, run_plan_start_time, reference_time,
            federated_selector_context_with_task_name);

    absl::Time upload_start = absl::Now();
    if (flags->per_phase_logs()) {
      opstats_logger->AddEvent(
          OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED);
      // Commit the run data accumulated thus far to Opstats and fail if
      // something goes wrong.
      FCP_RETURN_IF_ERROR(opstats_logger->CommitToStorage());
      event_publisher->PublishReportStarted(0);
    }
    auto outcome = plan_result_and_checkpoint_file.plan_result.outcome;
    absl::Status report_result = ReportTensorflowSpecPlanResult(
        event_publisher, log_manager, federated_protocol, opstats_logger,
        acceptance.client_only_plan.phase().tensorflow_spec(),
        std::move(plan_result_and_checkpoint_file), run_plan_start_time,
        reference_time);
    if (!report_result.ok()) {
      // If the report to the server failed, log an error.
      auto message =
          absl::StrCat("Error reporting results: code: ", report_result.code(),
                       ", message: ", report_result.message());
      event_publisher->PublishIoError(0, message);
      engine::LogOpStatsNetworkErrors(opstats_logger, report_result, message);
      FCP_LOG(INFO) << message;
    } else {
      if (flags->per_phase_logs()) {
        event_publisher->PublishReportFinished(
            federated_protocol->report_request_size_bytes(),
            federated_protocol->chunking_layer_bytes_sent(),
            absl::Now() - upload_start);
        opstats_logger->AddEvent(
            OperationalStats::Event::EVENT_KIND_UPLOAD_FINISHED);
      }
      if (outcome == engine::COMPLETED) {
        // Only if training succeeded *and* reporting succeeded do we consider
        // the device to have contributed successfully.
        fl_runner_result.set_contribution_result(FLRunnerResult::SUCCESS);
      }
    }

  } else {
    if (RunPlanWithExecutions(
            env_deps, event_publisher, files, log_manager, opstats_logger,
            flags, federated_protocol, acceptance.client_only_plan,
            *checkpoint_input_filename, timing_config, reference_time)) {
      // Only if RunPlanWithExecutions returns true, indicating that both
      // training *and* reporting succeeded, do we consider the device to have
      // contributed successfully.
      fl_runner_result.set_contribution_result(FLRunnerResult::SUCCESS);
    }
  }

  // Update the FLRunnerResult fields one more time to account for the "Report"
  // protocol interaction.
  UpdateRetryWindowAndNetworkStats(*federated_protocol, opstats_logger,
                                   fl_runner_result);

  return fl_runner_result;
}

FLRunnerTensorflowSpecResult RunPlanWithTensorflowSpecForTesting(
    SimpleTaskEnvironment* env_deps, EventPublisher* event_publisher,
    Files* files, LogManager* log_manager, const Flags* flags,
    const ClientOnlyPlan& client_plan,
    const std::string& checkpoint_input_filename,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time run_plan_start_time, const absl::Time reference_time) {
  FLRunnerTensorflowSpecResult result;
  result.set_outcome(engine::PhaseOutcome::ERROR);
  engine::PlanResult plan_result(engine::PhaseOutcome::ERROR);
  auto opstats_logger =
      engine::CreateOpStatsLogger(env_deps->GetBaseDir(), flags, log_manager,
                                  /*session_name=*/"", /*population_name=*/"");
  SelectorContext selector_context;
  if (client_plan.phase().has_federated_compute()) {
    // Regular TensorflowSpec-based plans.
    PlanResultAndCheckpointFile plan_result_and_checkpoint_file =
        RunPlanWithTensorflowSpec(
            env_deps, event_publisher, files, log_manager, opstats_logger.get(),
            flags, client_plan, checkpoint_input_filename, timing_config,
            run_plan_start_time, reference_time, selector_context);
    result.set_checkpoint_output_filename(
        plan_result_and_checkpoint_file.checkpoint_file);
    plan_result = std::move(plan_result_and_checkpoint_file.plan_result);
  } else if (client_plan.phase().has_federated_compute_eligibility()) {
    // Eligibility eval plans.
    plan_result = RunEligibilityEvalPlanWithTensorflowSpec(
        env_deps, event_publisher, log_manager, opstats_logger.get(), flags,
        client_plan, checkpoint_input_filename, timing_config,
        run_plan_start_time, reference_time, selector_context);
  } else {
    // This branch shouldn't be taken, unless we add an additional type of
    // TensorflowSpec-based plan in the future. We return a readable error so
    // that when such new plan types *are* added, they result in clear
    // compatibility test failures when such plans are erroneously targeted at
    // old releases that don't support them yet.
    event_publisher->PublishIoError(0, "Unsupported TensorflowSpec-based plan");
    return result;
  }

  // Copy output tensors into the result proto.
  result.set_outcome(plan_result.outcome);
  if (plan_result.outcome == engine::COMPLETED) {
    for (int i = 0; i < plan_result.output_names.size(); i++) {
      tensorflow::TensorProto output_tensor_proto;
      plan_result.output_tensors[i].AsProtoField(&output_tensor_proto);
      (*result.mutable_output_tensors())[plan_result.output_names[i]] =
          std::move(output_tensor_proto);
    }
  }

  return result;
}

}  // namespace client
}  // namespace fcp
