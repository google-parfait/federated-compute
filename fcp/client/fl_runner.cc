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
#include "fcp/client/engine/plan_engine_helpers.h"
#include "fcp/client/engine/simple_plan_engine.h"

#ifdef FCP_CLIENT_SUPPORT_TFLITE
#include "fcp/client/engine/tflite_plan_engine.h"
#endif

#include "fcp/client/event_publisher.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/federated_protocol_util.h"
#include "fcp/client/files.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/fl_runner_internal.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/grpc_federated_protocol.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/phase_logger_impl.h"
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

// Parses a proto from either an std::string or an absl::Cord. This allows the
// proto data to be provided in either format.
template <typename MessageT>
bool ParseFromStringOrCord(MessageT& proto,
                           std::variant<std::string, absl::Cord> data) {
  if (std::holds_alternative<std::string>(data)) {
    return proto.ParseFromString(std::get<std::string>(data));
  } else {
    return proto.ParseFromString(std::string(std::get<absl::Cord>(data)));
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
    const PlanResultAndCheckpointFile& plan_result_and_checkpoint_file) {
  const auto& [plan_result, checkpoint_file] = plan_result_and_checkpoint_file;
  if (plan_result.outcome != engine::PlanOutcome::kSuccess) {
    return absl::InvalidArgumentError("Computation failed.");
  }
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

#ifdef FCP_CLIENT_SUPPORT_TFLITE
std::unique_ptr<TfLiteInputs> ConstructTfLiteInputsForEligibilityEvalPlan(
    const FederatedComputeEligibilityIORouter& io_router,
    const std::string& checkpoint_input_filename) {
  auto inputs = std::make_unique<TfLiteInputs>();
  if (!io_router.input_filepath_tensor_name().empty()) {
    (*inputs)[io_router.input_filepath_tensor_name()] =
        checkpoint_input_filename;
  }
  return inputs;
}
#endif

NetworkStats GetNetworkStats(FederatedProtocol* federated_protocol) {
  return {.bytes_downloaded = federated_protocol->bytes_downloaded(),
          .bytes_uploaded = federated_protocol->bytes_uploaded(),
          .chunking_layer_bytes_received =
              federated_protocol->chunking_layer_bytes_received(),
          .chunking_layer_bytes_sent =
              federated_protocol->chunking_layer_bytes_sent(),
          .report_size_bytes = federated_protocol->report_request_size_bytes()};
}

// Updates the fields of `FLRunnerResult` that should always be updated after
// each interaction with the `FederatedProtocol` object.
void UpdateRetryWindowAndNetworkStats(FederatedProtocol& federated_protocol,
                                      PhaseLogger& phase_logger,
                                      FLRunnerResult& fl_runner_result) {
  // Update the result's retry window to the most recent one.
  auto retry_window = federated_protocol.GetLatestRetryWindow();
  *fl_runner_result.mutable_retry_window() = retry_window;
  phase_logger.UpdateRetryWindowAndNetworkStats(
      retry_window, GetNetworkStats(&federated_protocol));
}

engine::PlanResult RunEligibilityEvalPlanWithTensorflowSpec(
    SimpleTaskEnvironment* env_deps, LogManager* log_manager,
    OpStatsLogger* opstats_logger, const Flags* flags,
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
    return engine::PlanResult(
        engine::PlanOutcome::kInvalidArgument,
        absl::InvalidArgumentError("Invalid eligibility eval plan"));
  }
  const FederatedComputeEligibilityIORouter& io_router =
      client_plan.phase().federated_compute_eligibility();

  std::vector<std::string> output_names = {
      io_router.task_eligibility_info_tensor_name()};

#ifdef FCP_CLIENT_SUPPORT_TFLITE
  if (flags->use_tflite_training() && !client_plan.tflite_graph().empty()) {
    std::unique_ptr<TfLiteInputs> tflite_inputs =
        ConstructTfLiteInputsForEligibilityEvalPlan(io_router,
                                                    checkpoint_input_filename);
    engine::TfLitePlanEngine plan_engine(env_deps, log_manager, opstats_logger,
                                         flags, &timing_config);
    return plan_engine.RunPlan(
        client_plan.phase().tensorflow_spec(), client_plan.tflite_graph(),
        std::move(tflite_inputs), output_names, eligibility_selector_context);
  }
#endif

  // Construct input tensors and output tensor names based on the values in the
  // FederatedComputeEligibilityIORouter message.
  auto inputs = ConstructInputsForEligibilityEvalPlan(
      io_router, checkpoint_input_filename);
  // Run plan and get a set of output tensors back.
  engine::SimplePlanEngine plan_engine(env_deps, log_manager, opstats_logger,
                                       &timing_config);
  return plan_engine.RunPlan(
      client_plan.phase().tensorflow_spec(), client_plan.graph(),
      client_plan.tensorflow_config_proto(), std::move(inputs), output_names,
      eligibility_selector_context);
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

#ifdef FCP_CLIENT_SUPPORT_TFLITE
std::unique_ptr<TfLiteInputs> ConstructTFLiteInputsForTensorflowSpecPlan(
    const FederatedComputeIORouter& io_router,
    const std::string& checkpoint_input_filename,
    const std::string& checkpoint_output_filename) {
  auto inputs = std::make_unique<TfLiteInputs>();
  if (!io_router.input_filepath_tensor_name().empty()) {
    (*inputs)[io_router.input_filepath_tensor_name()] =
        checkpoint_input_filename;
  }

  if (!io_router.output_filepath_tensor_name().empty()) {
    (*inputs)[io_router.output_filepath_tensor_name()] =
        checkpoint_output_filename;
  }

  return inputs;
}
#endif

absl::StatusOr<std::vector<std::string>> ConstructOutputsForTensorflowSpecPlan(
    const FederatedComputeIORouter& io_router) {
  std::vector<std::string> output_names;
  for (const google::protobuf::MapPair<std::string, AggregationConfig>& it :
       io_router.aggregations()) {
    output_names.push_back(it.first);
    // The only aggregation type currently supported is secure aggregation.
    if (!it.second.has_secure_aggregation()) {
      return absl::InvalidArgumentError(
          "Unsupported aggregation type in TensorflowSpec-based plan");
    }
  }

  return output_names;
}

absl::StatusOr<std::vector<std::string>> ConstructOutputsWithDeterministicOrder(
    const TensorflowSpec& tensorflow_spec,
    const FederatedComputeIORouter& io_router) {
  std::vector<std::string> output_names;
  // The order of output tensor names should match the order in TensorflowSpec.
  for (const auto& output_tensor_spec : tensorflow_spec.output_tensor_specs()) {
    std::string tensor_name = output_tensor_spec.name();
    if (!io_router.aggregations().contains(tensor_name) ||
        !io_router.aggregations().at(tensor_name).has_secure_aggregation()) {
      return absl::InvalidArgumentError(
          "Output tensor is missing in AggregationConfig, or has unsupported "
          "aggregation type.");
    }
    output_names.push_back(tensor_name);
  }

  return output_names;
}

PlanResultAndCheckpointFile RunPlanWithTensorflowSpec(
    SimpleTaskEnvironment* env_deps, LogManager* log_manager,
    OpStatsLogger* opstats_logger, const Flags* flags,
    const ClientOnlyPlan& client_plan,
    const std::string& checkpoint_input_filename,
    const std::string& checkpoint_output_filename,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const SelectorContext& selector_context) {
  if (!client_plan.phase().has_tensorflow_spec()) {
    return PlanResultAndCheckpointFile(engine::PlanResult(
        engine::PlanOutcome::kInvalidArgument,
        absl::InvalidArgumentError("Plan must include TensorflowSpec.")));
  }
  if (!client_plan.phase().has_federated_compute() ||
      client_plan.phase().execution_size() > 0) {
    return PlanResultAndCheckpointFile(engine::PlanResult(
        engine::PlanOutcome::kInvalidArgument,
        absl::InvalidArgumentError("Invalid TensorflowSpec-based plan")));
  }

  // Get the output tensor names.
  absl::StatusOr<std::vector<std::string>> output_names;
  if (flags->use_tflite_training() || flags->deterministic_output_order()) {
    output_names = ConstructOutputsWithDeterministicOrder(
        client_plan.phase().tensorflow_spec(),
        client_plan.phase().federated_compute());
  } else {
    output_names = ConstructOutputsForTensorflowSpecPlan(
        client_plan.phase().federated_compute());
  }
  if (!output_names.ok()) {
    return PlanResultAndCheckpointFile(engine::PlanResult(
        engine::PlanOutcome::kInvalidArgument, output_names.status()));
  }

  // Run plan and get a set of output tensors back.
#ifdef FCP_CLIENT_SUPPORT_TFLITE
  if (flags->use_tflite_training() && !client_plan.tflite_graph().empty()) {
    std::unique_ptr<TfLiteInputs> tflite_inputs =
        ConstructTFLiteInputsForTensorflowSpecPlan(
            client_plan.phase().federated_compute(), checkpoint_input_filename,
            checkpoint_output_filename);
    engine::TfLitePlanEngine plan_engine(env_deps, log_manager, opstats_logger,
                                         flags, &timing_config);
    engine::PlanResult plan_result = plan_engine.RunPlan(
        client_plan.phase().tensorflow_spec(), client_plan.tflite_graph(),
        std::move(tflite_inputs), *output_names, selector_context);
    PlanResultAndCheckpointFile result(std::move(plan_result));
    result.checkpoint_file = checkpoint_output_filename;

    return result;
  }
#endif

  // Construct input tensors based on the values in the
  // FederatedComputeIORouter message and create a temporary file for the output
  // checkpoint if needed.
  auto inputs = ConstructInputsForTensorflowSpecPlan(
      client_plan.phase().federated_compute(), checkpoint_input_filename,
      checkpoint_output_filename);
  engine::SimplePlanEngine plan_engine(env_deps, log_manager, opstats_logger,
                                       &timing_config);
  engine::PlanResult plan_result = plan_engine.RunPlan(
      client_plan.phase().tensorflow_spec(), client_plan.graph(),
      client_plan.tensorflow_config_proto(), std::move(inputs), *output_names,
      selector_context);

  PlanResultAndCheckpointFile result(std::move(plan_result));
  result.checkpoint_file = checkpoint_output_filename;

  return result;
}

void LogEligibilityEvalComputationOutcome(
    PhaseLogger& phase_logger, engine::PlanResult plan_result,
    const absl::Status& eligibility_info_parsing_status,
    absl::Time run_plan_start_time, absl::Time reference_time) {
  switch (plan_result.outcome) {
    case engine::PlanOutcome::kSuccess: {
      if (eligibility_info_parsing_status.ok()) {
        phase_logger.LogEligibilityEvalComputationCompleted(
            plan_result.total_example_count,
            plan_result.total_example_size_bytes, run_plan_start_time,
            reference_time);
      } else {
        phase_logger.LogEligibilityEvalComputationTensorflowError(
            eligibility_info_parsing_status, plan_result.total_example_count,
            plan_result.total_example_size_bytes, run_plan_start_time,
            reference_time);
        FCP_LOG(ERROR) << eligibility_info_parsing_status.message();
      }
      break;
    }
    case engine::PlanOutcome::kInterrupted:
      phase_logger.LogEligibilityEvalComputationInterrupted(
          plan_result.original_status, plan_result.total_example_count,
          plan_result.total_example_size_bytes, run_plan_start_time,
          reference_time);
      break;
    case engine::PlanOutcome::kInvalidArgument:
      phase_logger.LogEligibilityEvalComputationInvalidArgument(
          plan_result.original_status, plan_result.total_example_count,
          plan_result.total_example_size_bytes, run_plan_start_time);
      break;
    case engine::PlanOutcome::kTensorflowError:
      phase_logger.LogEligibilityEvalComputationTensorflowError(
          plan_result.original_status, plan_result.total_example_count,
          plan_result.total_example_size_bytes, run_plan_start_time,
          reference_time);
      break;
    case engine::PlanOutcome::kExampleIteratorError:
      phase_logger.LogEligibilityEvalComputationExampleIteratorError(
          plan_result.original_status, plan_result.total_example_count,
          plan_result.total_example_size_bytes, run_plan_start_time);
      break;
  }
}

void LogComputationOutcome(const engine::PlanResult& plan_result,
                           absl::Status computation_results_parsing_status,
                           PhaseLogger& phase_logger,
                           absl::Time run_plan_start_time,
                           absl::Time reference_time) {
  switch (plan_result.outcome) {
    case engine::PlanOutcome::kSuccess: {
      int total_example_count = plan_result.total_example_count;
      int64_t total_example_size_bytes = plan_result.total_example_size_bytes;
      if (computation_results_parsing_status.ok()) {
        phase_logger.LogComputationCompleted(
            total_example_count, total_example_size_bytes, run_plan_start_time,
            reference_time);
      } else {
        phase_logger.LogComputationTensorflowError(
            computation_results_parsing_status, total_example_count,
            total_example_size_bytes, run_plan_start_time, reference_time);
      }
      break;
    }
    case engine::PlanOutcome::kInterrupted:
      phase_logger.LogComputationInterrupted(
          plan_result.original_status, plan_result.total_example_count,
          plan_result.total_example_size_bytes, run_plan_start_time,
          reference_time);
      break;
    case engine::PlanOutcome::kInvalidArgument:
      phase_logger.LogComputationInvalidArgument(
          plan_result.original_status, plan_result.total_example_count,
          plan_result.total_example_size_bytes, run_plan_start_time);
      break;
    case engine::PlanOutcome::kTensorflowError:
      phase_logger.LogComputationTensorflowError(
          plan_result.original_status, plan_result.total_example_count,
          plan_result.total_example_size_bytes, run_plan_start_time,
          reference_time);
      break;
    case engine::PlanOutcome::kExampleIteratorError:
      phase_logger.LogComputationExampleIteratorError(
          plan_result.original_status, plan_result.total_example_count,
          plan_result.total_example_size_bytes, run_plan_start_time);
      break;
  }
}

void LogResultUploadStatus(PhaseLogger& phase_logger, absl::Status result,
                           NetworkStats stats,
                           absl::Time time_before_result_upload,
                           absl::Time reference_time) {
  if (result.ok()) {
    phase_logger.LogResultUploadCompleted(stats, time_before_result_upload,
                                          reference_time);
  } else {
    auto message =
        absl::StrCat("Error reporting results: code: ", result.code(),
                     ", message: ", result.message());
    FCP_LOG(INFO) << message;
    if (result.code() == absl::StatusCode::kAborted) {
      phase_logger.LogResultUploadServerAborted(
          result, stats, time_before_result_upload, reference_time);
    } else if (result.code() == absl::StatusCode::kCancelled) {
      phase_logger.LogResultUploadClientInterrupted(
          result, stats, time_before_result_upload, reference_time);
    } else {
      phase_logger.LogResultUploadIOError(
          result, stats, time_before_result_upload, reference_time);
    }
  }
}

void LogFailureUploadStatus(PhaseLogger& phase_logger, absl::Status result,
                            NetworkStats stats,
                            absl::Time time_before_failure_upload,
                            absl::Time reference_time) {
  if (result.ok()) {
    phase_logger.LogFailureUploadCompleted(stats, time_before_failure_upload,
                                           reference_time);
  } else {
    auto message = absl::StrCat("Error reporting computation failure: code: ",
                                result.code(), ", message: ", result.message());
    FCP_LOG(INFO) << message;
    if (result.code() == absl::StatusCode::kAborted) {
      phase_logger.LogFailureUploadServerAborted(
          result, stats, time_before_failure_upload, reference_time);
    } else if (result.code() == absl::StatusCode::kCancelled) {
      phase_logger.LogFailureUploadClientInterrupted(
          result, stats, time_before_failure_upload, reference_time);
    } else {
      phase_logger.LogFailureUploadIOError(
          result, stats, time_before_failure_upload, reference_time);
    }
  }
}

absl::Status ReportTensorflowSpecPlanResult(
    FederatedProtocol* federated_protocol, PhaseLogger& phase_logger,
    absl::StatusOr<ComputationResults> computation_results,
    absl::Time run_plan_start_time, absl::Time reference_time) {
  const absl::Time before_report_time = absl::Now();
  absl::Status result = absl::InternalError("");
  if (computation_results.ok()) {
    FCP_RETURN_IF_ERROR(phase_logger.LogResultUploadStarted());
    result = federated_protocol->ReportCompleted(
        std::move(*computation_results),
        /*plan_duration=*/absl::Now() - run_plan_start_time);
    LogResultUploadStatus(phase_logger, result,
                          GetNetworkStats(federated_protocol),
                          before_report_time, reference_time);
  } else {
    FCP_RETURN_IF_ERROR(phase_logger.LogFailureUploadStarted());
    result = federated_protocol->ReportNotCompleted(
        engine::PhaseOutcome::ERROR,
        /*plan_duration=*/absl::Now() - run_plan_start_time);
    LogFailureUploadStatus(phase_logger, result,
                           GetNetworkStats(federated_protocol),
                           before_report_time, reference_time);
  }
  return result;
}

// Writes the given data to the stream, and returns true if successful and false
// if not.
bool WriteStringOrCordToFstream(
    std::fstream& stream, const std::variant<std::string, absl::Cord>& data) {
  if (stream.fail()) {
    return false;
  }
  if (std::holds_alternative<std::string>(data)) {
    return (stream << std::get<std::string>(data)).good();
  }
  for (absl::string_view chunk : std::get<absl::Cord>(data).Chunks()) {
    if (!(stream << chunk).good()) {
      return false;
    }
  }
  return true;
}

// Writes the given checkpoint data to a newly created temporary file.
// Returns the filename if successful, or an error if the file could not be
// created, or if writing to the file failed.
absl::StatusOr<std::string> CreateInputCheckpointFile(
    Files* files, const std::variant<std::string, absl::Cord>& checkpoint) {
  // Create the temporary checkpoint file.
  // Deletion of the file is left to the caller / the Files implementation.
  FCP_ASSIGN_OR_RETURN(absl::StatusOr<std::string> filename,
                       files->CreateTempFile("init", ".ckp"));
  // Write the checkpoint data to the file.
  std::fstream checkpoint_stream(*filename, std::ios_base::out);
  if (!WriteStringOrCordToFstream(checkpoint_stream, checkpoint)) {
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
    SimpleTaskEnvironment* env_deps, PhaseLogger& phase_logger, Files* files,
    LogManager* log_manager, OpStatsLogger* opstats_logger, const Flags* flags,
    FederatedProtocol* federated_protocol,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time reference_time, FLRunnerResult& fl_runner_result,
    const SelectorContext& selector_context) {
  absl::Time time_before_eligibility_eval_checkin = absl::Now();
  // Log that we are about to check in with the server.
  phase_logger.LogEligibilityEvalCheckInStarted();
  // Issue the eligibility eval checkin request.
  // The EligibilityEvalCheckin(...) method will call SetModelIdentifier() with
  // the name of whatever task it receives, ensuring that subsequent events will
  // be tagged with that identifier.
  absl::StatusOr<FederatedProtocol::EligibilityEvalCheckinResult>
      eligibility_checkin_result = federated_protocol->EligibilityEvalCheckin();
  UpdateRetryWindowAndNetworkStats(*federated_protocol, phase_logger,
                                   fl_runner_result);

  // It's a bit unfortunate that we have to inspect the checkin_result and
  // extract the model identifier here rather than further down the function,
  // but this ensures that the histograms below will have the right model
  // identifier attached (and we want to also emit the histograms even if we
  // have failed/rejected checkin outcomes).
  if (eligibility_checkin_result.ok() &&
      std::holds_alternative<FederatedProtocol::EligibilityEvalTask>(
          *eligibility_checkin_result)) {
    auto model_identifier = std::get<FederatedProtocol::EligibilityEvalTask>(
                                *eligibility_checkin_result)
                                .execution_id;
    phase_logger.SetModelIdentifier(model_identifier);
  }

  if (!eligibility_checkin_result.ok()) {
    auto status = eligibility_checkin_result.status();
    auto message = absl::StrCat("Error during eligibility eval checkin: code: ",
                                status.code(), ", message: ", status.message());
    if (status.code() == absl::StatusCode::kAborted) {
      phase_logger.LogEligibilityEvalCheckInServerAborted(
          status, GetNetworkStats(federated_protocol),
          time_before_eligibility_eval_checkin);
    } else if (status.code() == absl::StatusCode::kCancelled) {
      phase_logger.LogEligibilityEvalCheckInClientInterrupted(
          status, GetNetworkStats(federated_protocol),
          time_before_eligibility_eval_checkin);
    } else if (!status.ok()) {
      phase_logger.LogEligibilityEvalCheckInIOError(
          status, GetNetworkStats(federated_protocol),
          time_before_eligibility_eval_checkin);
    }
    FCP_LOG(INFO) << message;
    return absl::InternalError("");
  }

  if (std::holds_alternative<FederatedProtocol::Rejection>(
          *eligibility_checkin_result)) {
    phase_logger.LogEligibilityEvalCheckInTurnedAway(
        GetNetworkStats(federated_protocol),
        time_before_eligibility_eval_checkin);
    // If the server explicitly rejected our request, then we must abort and
    // we must not proceed to the "checkin" phase below.
    FCP_LOG(INFO) << "Device rejected by server during eligibility eval "
                     "checkin; aborting";
    return absl::InternalError("");
  } else if (std::holds_alternative<FederatedProtocol::EligibilityEvalDisabled>(
                 *eligibility_checkin_result)) {
    phase_logger.LogEligibilityEvalNotConfigured(
        GetNetworkStats(federated_protocol),
        time_before_eligibility_eval_checkin);
    // If the server indicates that no eligibility eval task is configured for
    // the population then there is nothing more to do. We simply proceed to
    // the "checkin" phase below without providing it a TaskEligibilityInfo
    // proto.
    return std::nullopt;
  }

  // Parse and run the eligibility eval task if the server returned one.
  auto eligibility_eval_task =
      absl::get<FederatedProtocol::EligibilityEvalTask>(
          *eligibility_checkin_result);

  ClientOnlyPlan plan;
  if (!ParseFromStringOrCord(plan, eligibility_eval_task.payloads.plan)) {
    auto message = "Failed to parse received eligibility eval plan";
    phase_logger.LogEligibilityEvalCheckInInvalidPayloadError(
        message, GetNetworkStats(federated_protocol),
        time_before_eligibility_eval_checkin);
    FCP_LOG(ERROR) << message;
    return absl::InternalError("");
  }

  absl::StatusOr<std::string> checkpoint_input_filename =
      CreateInputCheckpointFile(files,
                                eligibility_eval_task.payloads.checkpoint);
  if (!checkpoint_input_filename.ok()) {
    auto status = checkpoint_input_filename.status();
    auto message = absl::StrCat(
        "Failed to create eligibility eval checkpoint input file: code: ",
        status.code(), ", message: ", status.message());
    phase_logger.LogEligibilityEvalCheckInIOError(
        status, GetNetworkStats(federated_protocol),
        time_before_eligibility_eval_checkin);
    FCP_LOG(ERROR) << message;
    return absl::InternalError("");
  }

  phase_logger.LogEligibilityEvalCheckInCompleted(
      GetNetworkStats(federated_protocol),
      time_before_eligibility_eval_checkin);

  absl::Time run_plan_start_time = absl::Now();
  phase_logger.LogEligibilityEvalComputationStarted();
  engine::PlanResult plan_result = RunEligibilityEvalPlanWithTensorflowSpec(
      env_deps, log_manager, opstats_logger, flags, plan,
      *checkpoint_input_filename, timing_config, run_plan_start_time,
      reference_time, selector_context);
  absl::StatusOr<TaskEligibilityInfo> task_eligibility_info;
  if (plan_result.outcome == engine::PlanOutcome::kSuccess) {
    task_eligibility_info =
        ParseEligibilityEvalPlanOutput(plan_result.output_tensors);
  }
  LogEligibilityEvalComputationOutcome(phase_logger, std::move(plan_result),
                                       task_eligibility_info.status(),
                                       run_plan_start_time, reference_time);
  return task_eligibility_info;
}

struct CheckinResult {
  std::string task_name;
  ClientOnlyPlan plan;
  int32_t minimum_clients_in_server_visible_aggregate;
  std::string checkpoint_input_filename;
};
absl::StatusOr<CheckinResult> IssueCheckin(
    PhaseLogger& phase_logger, LogManager* log_manager, Files* files,
    FederatedProtocol* federated_protocol,
    std::optional<TaskEligibilityInfo> task_eligibility_info,
    absl::Time reference_time, const std::string& population_name,
    FLRunnerResult& fl_runner_result) {
  absl::Time time_before_checkin = absl::Now();
  // Clear the model identifier before check-in, to ensure that the any prior
  // eligibility eval task name isn't used any longer.
  phase_logger.SetModelIdentifier("");
  phase_logger.LogCheckInStarted();

  // The Checkin(...) method will call SetModelIdentifier() with the name of
  // whatever task it receives, ensuring that subsequent events will be tagged
  // with that identifier.
  absl::StatusOr<FederatedProtocol::CheckinResult> checkin_result =
      federated_protocol->Checkin(task_eligibility_info);
  UpdateRetryWindowAndNetworkStats(*federated_protocol, phase_logger,
                                   fl_runner_result);

  // It's a bit unfortunate that we have to inspect the checkin_result and
  // extract the model identifier here rather than further down the function,
  // but this ensures that the histograms below will have the right model
  // identifier attached (and we want to also emit the histograms even if we
  // have failed/rejected checkin outcomes).
  if (checkin_result.ok() &&
      std::holds_alternative<FederatedProtocol::TaskAssignment>(
          *checkin_result)) {
    auto model_identifier =
        std::get<FederatedProtocol::TaskAssignment>(*checkin_result)
            .aggregation_session_id;
    phase_logger.SetModelIdentifier(model_identifier);
  }

  if (!checkin_result.ok()) {
    auto status = checkin_result.status();
    auto message = absl::StrCat("Error during checkin: code: ", status.code(),
                                ", message: ", status.message());
    if (status.code() == absl::StatusCode::kAborted) {
      phase_logger.LogCheckInServerAborted(status,
                                           GetNetworkStats(federated_protocol),
                                           time_before_checkin, reference_time);
    } else if (status.code() == absl::StatusCode::kCancelled) {
      phase_logger.LogCheckInClientInterrupted(
          status, GetNetworkStats(federated_protocol), time_before_checkin,
          reference_time);
    } else if (!status.ok()) {
      phase_logger.LogCheckInIOError(status,
                                     GetNetworkStats(federated_protocol),
                                     time_before_checkin, reference_time);
    }
    FCP_LOG(INFO) << message;
    return status;
  }

  // Server rejected us? Return the fl_runner_results as-is.
  if (std::holds_alternative<FederatedProtocol::Rejection>(*checkin_result)) {
    phase_logger.LogCheckInTurnedAway(GetNetworkStats(federated_protocol),
                                      time_before_checkin, reference_time);
    FCP_LOG(INFO) << "Device rejected by server during checkin; aborting";
    return absl::InternalError("Device rejected by server.");
  }

  auto task_assignment =
      absl::get<FederatedProtocol::TaskAssignment>(*checkin_result);

  ClientOnlyPlan plan;
  if (!ParseFromStringOrCord(plan, task_assignment.payloads.plan)) {
    auto message = "Failed to parse received plan";
    phase_logger.LogCheckInInvalidPayload(message,
                                          GetNetworkStats(federated_protocol),
                                          time_before_checkin, reference_time);
    FCP_LOG(ERROR) << message;
    return absl::InternalError("");
  }

  int32_t minimum_clients_in_server_visible_aggregate = 0;
  if (task_assignment.sec_agg_info.has_value()) {
    auto minimum_number_of_participants =
        plan.phase().minimum_number_of_participants();
    if (task_assignment.sec_agg_info->expected_number_of_clients <
        minimum_number_of_participants) {
      return absl::InternalError(
          "expectedNumberOfClients was less than Plan's "
          "minimumNumberOfParticipants.");
    }
    minimum_clients_in_server_visible_aggregate =
        task_assignment.sec_agg_info
            ->minimum_clients_in_server_visible_aggregate;
  }

  absl::StatusOr<std::string> checkpoint_input_filename =
      CreateInputCheckpointFile(files, task_assignment.payloads.checkpoint);
  if (!checkpoint_input_filename.ok()) {
    auto status = checkpoint_input_filename.status();
    auto message = absl::StrCat(
        "Failed to create checkpoint input file: code: ", status.code(),
        ", message: ", status.message());
    phase_logger.LogCheckInIOError(status, GetNetworkStats(federated_protocol),
                                   time_before_checkin, reference_time);
    FCP_LOG(ERROR) << message;
    return status;
  }
  std::string task_name = ExtractTaskNameFromAggregationSessionId(
      task_assignment.aggregation_session_id, population_name, *log_manager);
  phase_logger.LogCheckInCompleted(task_name,
                                   GetNetworkStats(federated_protocol),
                                   time_before_checkin, reference_time);
  return CheckinResult{
      .task_name = std::move(task_name),
      .plan = std::move(plan),
      .minimum_clients_in_server_visible_aggregate =
          minimum_clients_in_server_visible_aggregate,
      .checkpoint_input_filename = std::move(*checkpoint_input_filename)};
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

  std::unique_ptr<::fcp::client::http::HttpClient> http_client =
      flags->enable_grpc_with_http_resource_support()
          ? env_deps->CreateHttpClient()
          : nullptr;

  GrpcFederatedProtocol federated_protocol(
      event_publisher, log_manager, flags, http_client.get(),
      federated_service_uri, api_key, test_cert_path, population_name,
      retry_token, client_version, attestation_measurement,
      should_abort_protocol_callback, timing_config, grpc_channel_deadline);
  PhaseLoggerImpl phase_logger(event_publisher, opstats_logger.get(),
                               log_manager, flags);
  return RunFederatedComputation(env_deps, phase_logger, event_publisher, files,
                                 log_manager, opstats_logger.get(), flags,
                                 &federated_protocol, timing_config,
                                 reference_time, session_name, population_name);
}

absl::StatusOr<FLRunnerResult> RunFederatedComputation(
    SimpleTaskEnvironment* env_deps, PhaseLogger& phase_logger,
    EventPublisher* event_publisher, Files* files, LogManager* log_manager,
    OpStatsLogger* opstats_logger, const Flags* flags,
    FederatedProtocol* federated_protocol,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time reference_time, const std::string& session_name,
    const std::string& population_name) {
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
  UpdateRetryWindowAndNetworkStats(*federated_protocol, phase_logger,
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
    phase_logger.LogTaskNotStarted(message);
    return fl_runner_result;
  }

  // Note that this method will update fl_runner_result's fields with values
  // received over the course of the eligibility eval protocol interaction.
  absl::StatusOr<std::optional<TaskEligibilityInfo>> eligibility_eval_result =
      IssueEligibilityEvalCheckinAndRunPlan(
          env_deps, phase_logger, files, log_manager, opstats_logger, flags,
          federated_protocol, timing_config, reference_time, fl_runner_result,
          eligibility_selector_context);
  if (!eligibility_eval_result.ok()) {
    return fl_runner_result;
  }

  auto checkin_result =
      IssueCheckin(phase_logger, log_manager, files, federated_protocol,
                   std::move(*eligibility_eval_result), reference_time,
                   population_name, fl_runner_result);

  if (!checkin_result.ok()) {
    return fl_runner_result;
  }

  SelectorContext federated_selector_context_with_task_name =
      federated_selector_context;
  federated_selector_context_with_task_name.mutable_computation_properties()
      ->mutable_federated()
      ->set_task_name(checkin_result->task_name);

  const auto& federated_compute_io_router =
      checkin_result->plan.phase().federated_compute();
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
            checkin_result->minimum_clients_in_server_visible_aggregate);
  } else {
    // Has an output checkpoint, so some tensors must be simply aggregated.
    *(federated_selector_context_with_task_name
          .mutable_computation_properties()
          ->mutable_federated()
          ->mutable_simple_aggregation()) = SimpleAggregation();
  }

  RetryWindow report_retry_window;
  phase_logger.LogComputationStarted();
  absl::Time run_plan_start_time = absl::Now();
  absl::StatusOr<std::string> checkpoint_output_filename =
      files->CreateTempFile("output", ".ckp");
  if (!checkpoint_output_filename.ok()) {
    auto status = checkpoint_output_filename.status();
    auto message = absl::StrCat(
        "Could not create temporary output checkpoint file: code: ",
        status.code(), ", message: ", status.message());
    phase_logger.LogComputationIOError(status, /*total_example_count=*/0,
                                       /*total_example_size_bytes=*/0,
                                       run_plan_start_time);
    return fl_runner_result;
  }
  PlanResultAndCheckpointFile plan_result_and_checkpoint_file =
      RunPlanWithTensorflowSpec(env_deps, log_manager, opstats_logger, flags,
                                checkin_result->plan,
                                checkin_result->checkpoint_input_filename,
                                *checkpoint_output_filename, timing_config,
                                federated_selector_context_with_task_name);
  auto outcome = plan_result_and_checkpoint_file.plan_result.outcome;
  absl::StatusOr<ComputationResults> computation_results;
  if (outcome == engine::PlanOutcome::kSuccess) {
    computation_results =
        CreateComputationResults(checkin_result->plan.phase().tensorflow_spec(),
                                 plan_result_and_checkpoint_file);
  }
  LogComputationOutcome(plan_result_and_checkpoint_file.plan_result,
                        computation_results.status(), phase_logger,
                        run_plan_start_time, reference_time);
  absl::Status report_result = ReportTensorflowSpecPlanResult(
      federated_protocol, phase_logger, std::move(computation_results),
      run_plan_start_time, reference_time);
  if (outcome == engine::PlanOutcome::kSuccess && report_result.ok()) {
    // Only if training succeeded *and* reporting succeeded do we consider
    // the device to have contributed successfully.
    fl_runner_result.set_contribution_result(FLRunnerResult::SUCCESS);
  }

  // Update the FLRunnerResult fields one more time to account for the "Report"
  // protocol interaction.
  UpdateRetryWindowAndNetworkStats(*federated_protocol, phase_logger,
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
  engine::PlanResult plan_result(engine::PlanOutcome::kTensorflowError,
                                 absl::UnknownError(""));
  auto opstats_logger =
      engine::CreateOpStatsLogger(env_deps->GetBaseDir(), flags, log_manager,
                                  /*session_name=*/"", /*population_name=*/"");
  PhaseLoggerImpl phase_logger(event_publisher, opstats_logger.get(),
                               log_manager, flags);
  SelectorContext selector_context;
  phase_logger.LogComputationStarted();
  if (client_plan.phase().has_federated_compute()) {
    absl::StatusOr<std::string> checkpoint_output_filename =
        files->CreateTempFile("output", ".ckp");
    if (!checkpoint_output_filename.ok()) {
      phase_logger.LogComputationIOError(
          checkpoint_output_filename.status(), /*total_example_count=*/0,
          /*total_example_size_bytes=*/0, run_plan_start_time);
      return result;
    }
    // Regular TensorflowSpec-based plans.
    PlanResultAndCheckpointFile plan_result_and_checkpoint_file =
        RunPlanWithTensorflowSpec(env_deps, log_manager, opstats_logger.get(),
                                  flags, client_plan, checkpoint_input_filename,
                                  *checkpoint_output_filename, timing_config,
                                  selector_context);
    result.set_checkpoint_output_filename(
        plan_result_and_checkpoint_file.checkpoint_file);
    plan_result = std::move(plan_result_and_checkpoint_file.plan_result);
  } else if (client_plan.phase().has_federated_compute_eligibility()) {
    // Eligibility eval plans.
    plan_result = RunEligibilityEvalPlanWithTensorflowSpec(
        env_deps, log_manager, opstats_logger.get(), flags, client_plan,
        checkpoint_input_filename, timing_config, run_plan_start_time,
        reference_time, selector_context);
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
  result.set_outcome(
      engine::ConvertPlanOutcomeToPhaseOutcome(plan_result.outcome));
  if (plan_result.outcome == engine::PlanOutcome::kSuccess) {
    for (int i = 0; i < plan_result.output_names.size(); i++) {
      tensorflow::TensorProto output_tensor_proto;
      plan_result.output_tensors[i].AsProtoField(&output_tensor_proto);
      (*result.mutable_output_tensors())[plan_result.output_names[i]] =
          std::move(output_tensor_proto);
    }
    phase_logger.LogComputationCompleted(plan_result.total_example_count,
                                         plan_result.total_example_size_bytes,
                                         run_plan_start_time, reference_time);
  } else {
    phase_logger.LogComputationTensorflowError(
        plan_result.original_status, plan_result.total_example_count,
        plan_result.total_example_size_bytes, run_plan_start_time,
        reference_time);
  }

  return result;
}

}  // namespace client
}  // namespace fcp
