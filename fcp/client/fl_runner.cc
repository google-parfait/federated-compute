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

#include <fcntl.h>

#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "fcp/base/clock.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/platform.h"
#include "fcp/client/cache/file_backed_resource_cache.h"
#include "fcp/client/cache/resource_cache.h"
#include "fcp/client/engine/common.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/engine/example_iterator_factory.h"
#include "fcp/client/engine/example_query_plan_engine.h"
#include "fcp/client/engine/plan_engine_helpers.h"
#include "fcp/client/opstats/opstats_utils.h"
#include "fcp/client/parsing_utils.h"

#ifdef FCP_CLIENT_SUPPORT_TFMOBILE
#include "fcp/client/engine/simple_plan_engine.h"
#endif

#include "fcp/client/engine/tflite_plan_engine.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/federated_protocol_util.h"
#include "fcp/client/files.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/http/http_federated_protocol.h"

#ifdef FCP_CLIENT_SUPPORT_GRPC
#include "fcp/client/grpc_federated_protocol.h"
#endif

#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_example_store.h"
#include "fcp/client/phase_logger_impl.h"
#include "fcp/client/secagg_runner.h"
#include "fcp/client/selector_context.pb.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/opstats.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/protos/population_eligibility_spec.pb.h"
#include "openssl/digest.h"
#include "openssl/evp.h"
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
using ::google::internal::federated::plan::PopulationEligibilitySpec;
using ::google::internal::federated::plan::TensorflowSpec;
using ::google::internal::federatedml::v2::RetryWindow;
using ::google::internal::federatedml::v2::TaskEligibilityInfo;

using TfLiteInputs = absl::flat_hash_map<std::string, std::string>;

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

std::string ComputeSHA256FromStringOrCord(
    std::variant<std::string, absl::Cord> data) {
  std::unique_ptr<EVP_MD_CTX, void (*)(EVP_MD_CTX*)> mdctx(EVP_MD_CTX_create(),
                                                           EVP_MD_CTX_destroy);
  FCP_CHECK(EVP_DigestInit_ex(mdctx.get(), EVP_sha256(), nullptr));

  std::string plan_str;
  if (std::holds_alternative<std::string>(data)) {
    plan_str = std::get<std::string>(data);
  } else {
    plan_str = std::string(std::get<absl::Cord>(data));
  }

  FCP_CHECK(EVP_DigestUpdate(mdctx.get(), plan_str.c_str(), sizeof(int)));
  const int hash_len = 32;  // 32 bytes for SHA-256.
  uint8_t computation_id_bytes[hash_len];
  FCP_CHECK(EVP_DigestFinal_ex(mdctx.get(), computation_id_bytes, nullptr));

  return std::string(reinterpret_cast<char const*>(computation_id_bytes),
                     hash_len);
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

// Creates computation results. The method checks for SecAgg tensors only if
// `tensorflow_spec != nullptr`.
absl::StatusOr<ComputationResults> CreateComputationResults(
    const TensorflowSpec* tensorflow_spec,
    const PlanResultAndCheckpointFile& plan_result_and_checkpoint_file) {
  const auto& [plan_result, checkpoint_file] = plan_result_and_checkpoint_file;
  if (plan_result.outcome != engine::PlanOutcome::kSuccess) {
    return absl::InvalidArgumentError("Computation failed.");
  }
  ComputationResults computation_results;
  if (tensorflow_spec != nullptr) {
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
          return absl::InvalidArgumentError(
              absl::StrCat("Tensor of type",
                           tensorflow::DataType_Name(output_tensor.dtype()),
                           "could not be converted to quantized value"));
      }
      computation_results[plan_result.output_names[i]] = std::move(quantized);
    }

    // Add dimensions to QuantizedTensors.
    for (const tensorflow::TensorSpecProto& tensor_spec :
         tensorflow_spec->output_tensor_specs()) {
      if (computation_results.find(tensor_spec.name()) !=
          computation_results.end()) {
        for (const tensorflow::TensorShapeProto_Dim& dim :
             tensor_spec.shape().dim()) {
          std::get<QuantizedTensor>(computation_results[tensor_spec.name()])
              .dimensions.push_back(dim.size());
        }
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

#ifdef FCP_CLIENT_SUPPORT_TFMOBILE
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
#endif

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

// Returns the cumulative network stats (those incurred up until this point in
// time).
//
// The `FederatedSelectManager` object may be null, if it is know that there has
// been no network usage from it yet.
NetworkStats GetCumulativeNetworkStats(
    FederatedProtocol* federated_protocol,
    FederatedSelectManager* fedselect_manager) {
  NetworkStats result = federated_protocol->GetNetworkStats();
  if (fedselect_manager != nullptr) {
    result = result + fedselect_manager->GetNetworkStats();
  }
  return result;
}

// Returns the newly incurred network stats since the previous snapshot of stats
// (the `reference_point` argument).
NetworkStats GetNetworkStatsSince(FederatedProtocol* federated_protocol,
                                  FederatedSelectManager* fedselect_manager,
                                  const NetworkStats& reference_point) {
  return GetCumulativeNetworkStats(federated_protocol, fedselect_manager) -
         reference_point;
}

// Updates the fields of `FLRunnerResult` that should always be updated after
// each interaction with the `FederatedProtocol` or `FederatedSelectManager`
// objects.
//
// The `FederatedSelectManager` object may be null, if it is know that there has
// been no network usage from it yet.
void UpdateRetryWindowAndNetworkStats(FederatedProtocol& federated_protocol,
                                      FederatedSelectManager* fedselect_manager,
                                      PhaseLogger& phase_logger,
                                      FLRunnerResult& fl_runner_result) {
  // Update the result's retry window to the most recent one.
  auto retry_window = federated_protocol.GetLatestRetryWindow();
  RetryInfo retry_info;
  *retry_info.mutable_retry_token() = retry_window.retry_token();
  *retry_info.mutable_minimum_delay() = retry_window.delay_min();
  *fl_runner_result.mutable_retry_info() = retry_info;
  phase_logger.UpdateRetryWindowAndNetworkStats(
      retry_window,
      GetCumulativeNetworkStats(&federated_protocol, fedselect_manager));
}

// Creates an ExampleIteratorFactory that routes queries to the
// SimpleTaskEnvironment::CreateExampleIterator() method.
std::unique_ptr<engine::ExampleIteratorFactory>
CreateSimpleTaskEnvironmentIteratorFactory(
    SimpleTaskEnvironment* task_env, const SelectorContext& selector_context) {
  return std::make_unique<engine::FunctionalExampleIteratorFactory>(
      /*can_handle_func=*/
      [](const google::internal::federated::plan::ExampleSelector&) {
        // The SimpleTaskEnvironment-based ExampleIteratorFactory should
        // be the catch-all factory that is able to handle all queries
        // that no other ExampleIteratorFactory is able to handle.
        return true;
      },
      /*create_iterator_func=*/
      [task_env, selector_context](
          const google::internal::federated::plan::ExampleSelector&
              example_selector) {
        return task_env->CreateExampleIterator(example_selector,
                                               selector_context);
      },
      /*should_collect_stats=*/true);
}

engine::PlanResult RunEligibilityEvalPlanWithTensorflowSpec(
    std::vector<engine::ExampleIteratorFactory*> example_iterator_factories,
    std::function<bool()> should_abort, LogManager* log_manager,
    OpStatsLogger* opstats_logger, const Flags* flags,
    const ClientOnlyPlan& client_plan,
    const std::string& checkpoint_input_filename,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time run_plan_start_time, const absl::Time reference_time) {
  // Check that this is a TensorflowSpec-based plan for federated eligibility
  // computation.
  if (!client_plan.phase().has_tensorflow_spec() ||
      !client_plan.phase().has_federated_compute_eligibility()) {
    return engine::PlanResult(
        engine::PlanOutcome::kInvalidArgument,
        absl::InvalidArgumentError("Invalid eligibility eval plan"));
  }
  const FederatedComputeEligibilityIORouter& io_router =
      client_plan.phase().federated_compute_eligibility();

  std::vector<std::string> output_names = {
      io_router.task_eligibility_info_tensor_name()};

  if (!client_plan.tflite_graph().empty()) {
    log_manager->LogDiag(
        ProdDiagCode::BACKGROUND_TRAINING_TFLITE_MODEL_INCLUDED);
  }

  if (flags->use_tflite_training() && !client_plan.tflite_graph().empty()) {
    std::unique_ptr<TfLiteInputs> tflite_inputs =
        ConstructTfLiteInputsForEligibilityEvalPlan(io_router,
                                                    checkpoint_input_filename);
    engine::TfLitePlanEngine plan_engine(example_iterator_factories,
                                         should_abort, log_manager,
                                         opstats_logger, flags, &timing_config);
    return plan_engine.RunPlan(client_plan.phase().tensorflow_spec(),
                               client_plan.tflite_graph(),
                               std::move(tflite_inputs), output_names);
  }

#ifdef FCP_CLIENT_SUPPORT_TFMOBILE
  // Construct input tensors and output tensor names based on the values in the
  // FederatedComputeEligibilityIORouter message.
  auto inputs = ConstructInputsForEligibilityEvalPlan(
      io_router, checkpoint_input_filename);
  // Run plan and get a set of output tensors back.
  engine::SimplePlanEngine plan_engine(
      example_iterator_factories, should_abort, log_manager, opstats_logger,
      &timing_config, flags->support_constant_tf_inputs());
  return plan_engine.RunPlan(
      client_plan.phase().tensorflow_spec(), client_plan.graph(),
      client_plan.tensorflow_config_proto(), std::move(inputs), output_names);
#else
  return engine::PlanResult(
      engine::PlanOutcome::kTensorflowError,
      absl::InternalError("No eligibility eval plan engine enabled"));
#endif
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

#ifdef FCP_CLIENT_SUPPORT_TFMOBILE
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
#endif

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
    std::vector<engine::ExampleIteratorFactory*> example_iterator_factories,
    std::function<bool()> should_abort, LogManager* log_manager,
    OpStatsLogger* opstats_logger, const Flags* flags,
    const ClientOnlyPlan& client_plan,
    const std::string& checkpoint_input_filename,
    const std::string& checkpoint_output_filename,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config) {
  if (!client_plan.phase().has_tensorflow_spec()) {
    return PlanResultAndCheckpointFile(engine::PlanResult(
        engine::PlanOutcome::kInvalidArgument,
        absl::InvalidArgumentError("Plan must include TensorflowSpec.")));
  }
  if (!client_plan.phase().has_federated_compute()) {
    return PlanResultAndCheckpointFile(engine::PlanResult(
        engine::PlanOutcome::kInvalidArgument,
        absl::InvalidArgumentError("Invalid TensorflowSpec-based plan")));
  }

  // Get the output tensor names.
  absl::StatusOr<std::vector<std::string>> output_names;
  output_names = ConstructOutputsWithDeterministicOrder(
      client_plan.phase().tensorflow_spec(),
      client_plan.phase().federated_compute());
  if (!output_names.ok()) {
    return PlanResultAndCheckpointFile(engine::PlanResult(
        engine::PlanOutcome::kInvalidArgument, output_names.status()));
  }

  // Run plan and get a set of output tensors back.
  if (flags->use_tflite_training() && !client_plan.tflite_graph().empty()) {
    std::unique_ptr<TfLiteInputs> tflite_inputs =
        ConstructTFLiteInputsForTensorflowSpecPlan(
            client_plan.phase().federated_compute(), checkpoint_input_filename,
            checkpoint_output_filename);
    engine::TfLitePlanEngine plan_engine(example_iterator_factories,
                                         should_abort, log_manager,
                                         opstats_logger, flags, &timing_config);
    engine::PlanResult plan_result = plan_engine.RunPlan(
        client_plan.phase().tensorflow_spec(), client_plan.tflite_graph(),
        std::move(tflite_inputs), *output_names);
    PlanResultAndCheckpointFile result(std::move(plan_result));
    result.checkpoint_file = checkpoint_output_filename;

    return result;
  }

#ifdef FCP_CLIENT_SUPPORT_TFMOBILE
  // Construct input tensors based on the values in the
  // FederatedComputeIORouter message and create a temporary file for the output
  // checkpoint if needed.
  auto inputs = ConstructInputsForTensorflowSpecPlan(
      client_plan.phase().federated_compute(), checkpoint_input_filename,
      checkpoint_output_filename);
  engine::SimplePlanEngine plan_engine(
      example_iterator_factories, should_abort, log_manager, opstats_logger,
      &timing_config, flags->support_constant_tf_inputs());
  engine::PlanResult plan_result = plan_engine.RunPlan(
      client_plan.phase().tensorflow_spec(), client_plan.graph(),
      client_plan.tensorflow_config_proto(), std::move(inputs), *output_names);

  PlanResultAndCheckpointFile result(std::move(plan_result));
  result.checkpoint_file = checkpoint_output_filename;

  return result;
#else
  return PlanResultAndCheckpointFile(
      engine::PlanResult(engine::PlanOutcome::kTensorflowError,
                         absl::InternalError("No plan engine enabled")));
#endif
}

PlanResultAndCheckpointFile RunPlanWithExampleQuerySpec(
    std::vector<engine::ExampleIteratorFactory*> example_iterator_factories,
    OpStatsLogger* opstats_logger, const Flags* flags,
    const ClientOnlyPlan& client_plan,
    const std::string& checkpoint_output_filename) {
  if (!client_plan.phase().has_example_query_spec()) {
    return PlanResultAndCheckpointFile(engine::PlanResult(
        engine::PlanOutcome::kInvalidArgument,
        absl::InvalidArgumentError("Plan must include ExampleQuerySpec")));
  }
  if (!flags->enable_example_query_plan_engine()) {
    // Example query plan received while the flag is off.
    return PlanResultAndCheckpointFile(engine::PlanResult(
        engine::PlanOutcome::kInvalidArgument,
        absl::InvalidArgumentError(
            "Example query plan received while the flag is off")));
  }
  if (!client_plan.phase().has_federated_example_query()) {
    return PlanResultAndCheckpointFile(engine::PlanResult(
        engine::PlanOutcome::kInvalidArgument,
        absl::InvalidArgumentError("Invalid ExampleQuerySpec-based plan")));
  }
  for (const auto& example_query :
       client_plan.phase().example_query_spec().example_queries()) {
    for (auto const& [vector_name, spec] :
         example_query.output_vector_specs()) {
      const auto& aggregations =
          client_plan.phase().federated_example_query().aggregations();
      if ((aggregations.find(vector_name) == aggregations.end()) ||
          !aggregations.at(vector_name).has_tf_v1_checkpoint_aggregation()) {
        return PlanResultAndCheckpointFile(engine::PlanResult(
            engine::PlanOutcome::kInvalidArgument,
            absl::InvalidArgumentError("Output vector is missing in "
                                       "AggregationConfig, or has unsupported "
                                       "aggregation type.")));
      }
    }
  }

  engine::ExampleQueryPlanEngine plan_engine(example_iterator_factories,
                                             opstats_logger);
  engine::PlanResult plan_result = plan_engine.RunPlan(
      client_plan.phase().example_query_spec(), checkpoint_output_filename);
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
            plan_result.example_stats, run_plan_start_time, reference_time);
      } else {
        phase_logger.LogEligibilityEvalComputationTensorflowError(
            eligibility_info_parsing_status, plan_result.example_stats,
            run_plan_start_time, reference_time);
        FCP_LOG(ERROR) << eligibility_info_parsing_status.message();
      }
      break;
    }
    case engine::PlanOutcome::kInterrupted:
      phase_logger.LogEligibilityEvalComputationInterrupted(
          plan_result.original_status, plan_result.example_stats,
          run_plan_start_time, reference_time);
      break;
    case engine::PlanOutcome::kInvalidArgument:
      phase_logger.LogEligibilityEvalComputationInvalidArgument(
          plan_result.original_status, plan_result.example_stats,
          run_plan_start_time);
      break;
    case engine::PlanOutcome::kTensorflowError:
      phase_logger.LogEligibilityEvalComputationTensorflowError(
          plan_result.original_status, plan_result.example_stats,
          run_plan_start_time, reference_time);
      break;
    case engine::PlanOutcome::kExampleIteratorError:
      phase_logger.LogEligibilityEvalComputationExampleIteratorError(
          plan_result.original_status, plan_result.example_stats,
          run_plan_start_time);
      break;
  }
}

void LogComputationOutcome(const engine::PlanResult& plan_result,
                           absl::Status computation_results_parsing_status,
                           PhaseLogger& phase_logger,
                           const NetworkStats& network_stats,
                           absl::Time run_plan_start_time,
                           absl::Time reference_time) {
  switch (plan_result.outcome) {
    case engine::PlanOutcome::kSuccess: {
      if (computation_results_parsing_status.ok()) {
        phase_logger.LogComputationCompleted(plan_result.example_stats,
                                             network_stats, run_plan_start_time,
                                             reference_time);
      } else {
        phase_logger.LogComputationTensorflowError(
            computation_results_parsing_status, plan_result.example_stats,
            network_stats, run_plan_start_time, reference_time);
      }
      break;
    }
    case engine::PlanOutcome::kInterrupted:
      phase_logger.LogComputationInterrupted(
          plan_result.original_status, plan_result.example_stats, network_stats,
          run_plan_start_time, reference_time);
      break;
    case engine::PlanOutcome::kInvalidArgument:
      phase_logger.LogComputationInvalidArgument(
          plan_result.original_status, plan_result.example_stats, network_stats,
          run_plan_start_time);
      break;
    case engine::PlanOutcome::kTensorflowError:
      phase_logger.LogComputationTensorflowError(
          plan_result.original_status, plan_result.example_stats, network_stats,
          run_plan_start_time, reference_time);
      break;
    case engine::PlanOutcome::kExampleIteratorError:
      phase_logger.LogComputationExampleIteratorError(
          plan_result.original_status, plan_result.example_stats, network_stats,
          run_plan_start_time);
      break;
  }
}

void LogResultUploadStatus(PhaseLogger& phase_logger, absl::Status result,
                           const NetworkStats& network_stats,
                           absl::Time time_before_result_upload,
                           absl::Time reference_time) {
  if (result.ok()) {
    phase_logger.LogResultUploadCompleted(
        network_stats, time_before_result_upload, reference_time);
  } else {
    auto message =
        absl::StrCat("Error reporting results: code: ", result.code(),
                     ", message: ", result.message());
    FCP_LOG(INFO) << message;
    if (result.code() == absl::StatusCode::kAborted) {
      phase_logger.LogResultUploadServerAborted(
          result, network_stats, time_before_result_upload, reference_time);
    } else if (result.code() == absl::StatusCode::kCancelled) {
      phase_logger.LogResultUploadClientInterrupted(
          result, network_stats, time_before_result_upload, reference_time);
    } else {
      phase_logger.LogResultUploadIOError(
          result, network_stats, time_before_result_upload, reference_time);
    }
  }
}

void LogFailureUploadStatus(PhaseLogger& phase_logger, absl::Status result,
                            const NetworkStats& network_stats,
                            absl::Time time_before_failure_upload,
                            absl::Time reference_time) {
  if (result.ok()) {
    phase_logger.LogFailureUploadCompleted(
        network_stats, time_before_failure_upload, reference_time);
  } else {
    auto message = absl::StrCat("Error reporting computation failure: code: ",
                                result.code(), ", message: ", result.message());
    FCP_LOG(INFO) << message;
    if (result.code() == absl::StatusCode::kAborted) {
      phase_logger.LogFailureUploadServerAborted(
          result, network_stats, time_before_failure_upload, reference_time);
    } else if (result.code() == absl::StatusCode::kCancelled) {
      phase_logger.LogFailureUploadClientInterrupted(
          result, network_stats, time_before_failure_upload, reference_time);
    } else {
      phase_logger.LogFailureUploadIOError(
          result, network_stats, time_before_failure_upload, reference_time);
    }
  }
}

absl::Status ReportPlanResult(
    FederatedProtocol* federated_protocol, PhaseLogger& phase_logger,
    absl::StatusOr<ComputationResults> computation_results,
    absl::Time run_plan_start_time, absl::Time reference_time) {
  const absl::Time before_report_time = absl::Now();

  // Note that the FederatedSelectManager shouldn't be active anymore during the
  // reporting of results, so we don't bother passing it to
  // GetNetworkStatsSince.
  //
  // We must return only stats that cover the report phase for the log events
  // below.
  const NetworkStats before_report_stats =
      GetCumulativeNetworkStats(federated_protocol,
                                /*fedselect_manager=*/nullptr);
  absl::Status result = absl::InternalError("");
  if (computation_results.ok()) {
    FCP_RETURN_IF_ERROR(phase_logger.LogResultUploadStarted());
    result = federated_protocol->ReportCompleted(
        std::move(*computation_results),
        /*plan_duration=*/absl::Now() - run_plan_start_time, std::nullopt);
    LogResultUploadStatus(
        phase_logger, result,
        GetNetworkStatsSince(federated_protocol, /*fedselect_manager=*/nullptr,
                             before_report_stats),
        before_report_time, reference_time);
  } else {
    FCP_RETURN_IF_ERROR(phase_logger.LogFailureUploadStarted());
    result = federated_protocol->ReportNotCompleted(
        engine::PhaseOutcome::ERROR,
        /*plan_duration=*/absl::Now() - run_plan_start_time, std::nullopt);
    LogFailureUploadStatus(
        phase_logger, result,
        GetNetworkStatsSince(federated_protocol, /*fedselect_manager=*/nullptr,
                             before_report_stats),
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

absl::StatusOr<std::optional<TaskEligibilityInfo>> RunEligibilityEvalPlan(
    const FederatedProtocol::EligibilityEvalTask& eligibility_eval_task,
    std::vector<engine::ExampleIteratorFactory*> example_iterator_factories,
    std::function<bool()> should_abort, PhaseLogger& phase_logger, Files* files,
    LogManager* log_manager, OpStatsLogger* opstats_logger, const Flags* flags,
    FederatedProtocol* federated_protocol,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time reference_time, const absl::Time time_before_checkin,
    const absl::Time time_before_plan_download,
    const NetworkStats& network_stats) {
  ClientOnlyPlan plan;
  if (!ParseFromStringOrCord(plan, eligibility_eval_task.payloads.plan)) {
    auto message = "Failed to parse received eligibility eval plan";
    phase_logger.LogEligibilityEvalCheckinInvalidPayloadError(
        message, network_stats, time_before_plan_download);

    FCP_LOG(ERROR) << message;
    return absl::InternalError(message);
  }

  absl::StatusOr<std::string> checkpoint_input_filename =
      CreateInputCheckpointFile(files,
                                eligibility_eval_task.payloads.checkpoint);
  if (!checkpoint_input_filename.ok()) {
    auto status = checkpoint_input_filename.status();
    auto message = absl::StrCat(
        "Failed to create eligibility eval checkpoint input file: code: ",
        status.code(), ", message: ", status.message());
    phase_logger.LogEligibilityEvalCheckinIOError(status, network_stats,
                                                  time_before_plan_download);
    FCP_LOG(ERROR) << message;
    return absl::InternalError("");
  }

  phase_logger.LogEligibilityEvalCheckinCompleted(network_stats,
                                                  /*time_before_checkin=*/
                                                  time_before_checkin,
                                                  /*time_before_plan_download=*/
                                                  time_before_plan_download);

  absl::Time run_plan_start_time = absl::Now();
  phase_logger.LogEligibilityEvalComputationStarted();
  engine::PlanResult plan_result = RunEligibilityEvalPlanWithTensorflowSpec(
      example_iterator_factories, should_abort, log_manager, opstats_logger,
      flags, plan, *checkpoint_input_filename, timing_config,
      run_plan_start_time, reference_time);
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

struct EligibilityEvalResult {
  std::optional<TaskEligibilityInfo> task_eligibility_info;
  std::vector<std::string> task_names_for_multiple_task_assignments;
};

// Create an EligibilityEvalResult from a TaskEligibilityInfo and
// PopulationEligibilitySpec.  If both population_spec and task_eligibility_info
// are present, the returned EligibilityEvalResult will contain a
// TaskEligibilityInfo which only contains the tasks for single task assignment,
// and a vector of task names for multiple task assignment.
EligibilityEvalResult CreateEligibilityEvalResult(
    const std::optional<TaskEligibilityInfo>& task_eligibility_info,
    const std::optional<PopulationEligibilitySpec>& population_spec) {
  EligibilityEvalResult result;
  if (population_spec.has_value() && task_eligibility_info.has_value()) {
    absl::flat_hash_set<std::string> task_names_for_multiple_task_assignments;
    for (const auto& task_info : population_spec.value().task_info()) {
      if (task_info.task_assignment_mode() ==
          PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE) {
        task_names_for_multiple_task_assignments.insert(task_info.task_name());
      }
    }
    TaskEligibilityInfo single_task_assignment_eligibility_info;
    single_task_assignment_eligibility_info.set_version(
        task_eligibility_info.value().version());
    for (const auto& task_weight :
         task_eligibility_info.value().task_weights()) {
      if (task_names_for_multiple_task_assignments.contains(
              task_weight.task_name())) {
        result.task_names_for_multiple_task_assignments.push_back(
            task_weight.task_name());
      } else {
        *single_task_assignment_eligibility_info.mutable_task_weights()->Add() =
            task_weight;
      }
    }
    result.task_eligibility_info = single_task_assignment_eligibility_info;
  } else {
    result.task_eligibility_info = task_eligibility_info;
  }
  return result;
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
absl::StatusOr<EligibilityEvalResult> IssueEligibilityEvalCheckinAndRunPlan(
    std::vector<engine::ExampleIteratorFactory*> example_iterator_factories,
    std::function<bool()> should_abort, PhaseLogger& phase_logger, Files* files,
    LogManager* log_manager, OpStatsLogger* opstats_logger, const Flags* flags,
    FederatedProtocol* federated_protocol,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time reference_time, FLRunnerResult& fl_runner_result) {
  const absl::Time time_before_checkin = absl::Now();
  const NetworkStats network_stats_before_checkin =
      GetCumulativeNetworkStats(federated_protocol,
                                /*fedselect_manager=*/nullptr);

  // These fields will, after a successful checkin that resulted in an EET being
  // received, contain the time at which the EET plan/checkpoint URIs were
  // received (but not yet downloaded), as well as the cumulative network stats
  // at that point, allowing us to separately calculate how long it took to then
  // download the actual payloads.
  absl::Time time_before_plan_download = time_before_checkin;
  NetworkStats network_stats_before_plan_download =
      network_stats_before_checkin;

  // Log that we are about to check in with the server.
  phase_logger.LogEligibilityEvalCheckinStarted();

  // Issue the eligibility eval checkin request (providing a callback that will
  // be called when an EET is assigned to the task but before its
  // plan/checkpoint URIs have actually been downloaded).
  bool plan_uris_received_callback_called = false;
  std::function<void(const FederatedProtocol::EligibilityEvalTask&)>
      plan_uris_received_callback =
          [&time_before_plan_download, &network_stats_before_plan_download,
           &time_before_checkin, &network_stats_before_checkin,
           &federated_protocol, &phase_logger,
           &plan_uris_received_callback_called](
              const FederatedProtocol::EligibilityEvalTask& task) {
            // When the plan URIs have been received, we already know the name
            // of the task we have been assigned, so let's tell the PhaseLogger.
            phase_logger.SetModelIdentifier(task.execution_id);

            // We also should log a corresponding log event.
            phase_logger.LogEligibilityEvalCheckinPlanUriReceived(
                GetNetworkStatsSince(federated_protocol,
                                     /*fedselect_manager=*/nullptr,
                                     network_stats_before_checkin),
                time_before_checkin);

            // And we must take a snapshot of the current time & network stats,
            // so we can distinguish between the duration/network stats incurred
            // for the checkin request vs. the actual downloading of the
            // plan/checkpoint resources.
            time_before_plan_download = absl::Now();
            network_stats_before_plan_download =
                GetCumulativeNetworkStats(federated_protocol,
                                          /*fedselect_manager=*/nullptr);
            plan_uris_received_callback_called = true;
          };
  absl::StatusOr<FederatedProtocol::EligibilityEvalCheckinResult>
      eligibility_checkin_result = federated_protocol->EligibilityEvalCheckin(
          plan_uris_received_callback);
  UpdateRetryWindowAndNetworkStats(*federated_protocol,
                                   /*fedselect_manager=*/nullptr, phase_logger,
                                   fl_runner_result);

  // It's a bit unfortunate that we have to inspect the checkin_result and
  // extract the model identifier here rather than further down the function,
  // but this ensures that the histograms below will have the right model
  // identifier attached (and we want to also emit the histograms even if we
  // have failed/rejected checkin outcomes).
  if (eligibility_checkin_result.ok() &&
      std::holds_alternative<FederatedProtocol::EligibilityEvalTask>(
          *eligibility_checkin_result)) {
    // Make sure that if we received an EligibilityEvalTask, then the callback
    // should have already been called by this point by the protocol (ensuring
    // that SetModelIdentifier has been called etc.).
    FCP_CHECK(plan_uris_received_callback_called);
  }

  if (!eligibility_checkin_result.ok()) {
    auto status = eligibility_checkin_result.status();
    auto message = absl::StrCat("Error during eligibility eval checkin: code: ",
                                status.code(), ", message: ", status.message());
    if (status.code() == absl::StatusCode::kAborted) {
      phase_logger.LogEligibilityEvalCheckinServerAborted(
          status,
          GetNetworkStatsSince(federated_protocol,
                               /*fedselect_manager=*/nullptr,
                               network_stats_before_plan_download),
          time_before_plan_download);
    } else if (status.code() == absl::StatusCode::kCancelled) {
      phase_logger.LogEligibilityEvalCheckinClientInterrupted(
          status,
          GetNetworkStatsSince(federated_protocol,
                               /*fedselect_manager=*/nullptr,
                               network_stats_before_plan_download),
          time_before_plan_download);
    } else if (!status.ok()) {
      phase_logger.LogEligibilityEvalCheckinIOError(
          status,
          GetNetworkStatsSince(federated_protocol,
                               /*fedselect_manager=*/nullptr,
                               network_stats_before_plan_download),
          time_before_plan_download);
    }
    FCP_LOG(INFO) << message;
    return absl::InternalError("");
  }

  EligibilityEvalResult result;

  if (std::holds_alternative<FederatedProtocol::Rejection>(
          *eligibility_checkin_result)) {
    phase_logger.LogEligibilityEvalCheckinTurnedAway(
        GetNetworkStatsSince(federated_protocol, /*fedselect_manager=*/nullptr,
                             network_stats_before_checkin),
        time_before_checkin);
    // If the server explicitly rejected our request, then we must abort and
    // we must not proceed to the "checkin" phase below.
    FCP_LOG(INFO) << "Device rejected by server during eligibility eval "
                     "checkin; aborting";
    return absl::InternalError("");
  } else if (std::holds_alternative<FederatedProtocol::EligibilityEvalDisabled>(
                 *eligibility_checkin_result)) {
    phase_logger.LogEligibilityEvalNotConfigured(
        GetNetworkStatsSince(federated_protocol, /*fedselect_manager=*/nullptr,
                             network_stats_before_checkin),
        time_before_checkin);
    // If the server indicates that no eligibility eval task is configured for
    // the population then there is nothing more to do. We simply proceed to
    // the "checkin" phase below without providing it a TaskEligibilityInfo
    // proto.
    result.task_eligibility_info = std::nullopt;
    return result;
  }

  auto eligibility_eval_task =
      absl::get<FederatedProtocol::EligibilityEvalTask>(
          *eligibility_checkin_result);
  // Parse and run the eligibility eval task if the server returned one.
  // Now we have a EligibilityEvalTask, if an error happens, we will report to
  // the server via the ReportEligibilityEvalError.
  absl::StatusOr<std::optional<TaskEligibilityInfo>> task_eligibility_info =
      RunEligibilityEvalPlan(
          eligibility_eval_task, example_iterator_factories, should_abort,
          phase_logger, files, log_manager, opstats_logger, flags,
          federated_protocol, timing_config, reference_time,
          /*time_before_checkin=*/time_before_checkin,
          /*time_before_plan_download=*/time_before_plan_download,
          GetNetworkStatsSince(federated_protocol,
                               /*fedselect_manager=*/nullptr,
                               network_stats_before_plan_download));
  if (!task_eligibility_info.ok()) {
    // Note that none of the PhaseLogger methods will reflect the very little
    // amount of network usage the will be incurred by this protocol request.
    // We consider this to be OK to keep things simple, and because this should
    // use such a limited amount of network bandwidth. Do note that the network
    // usage *will* be correctly reported in the OpStats database.
    federated_protocol->ReportEligibilityEvalError(
        absl::Status(task_eligibility_info.status().code(),
                     "Failed to compute eligibility info"));
    UpdateRetryWindowAndNetworkStats(*federated_protocol,
                                     /*fedselect_manager=*/nullptr,
                                     phase_logger, fl_runner_result);
    return task_eligibility_info.status();
  }
  return CreateEligibilityEvalResult(
      *task_eligibility_info,
      eligibility_eval_task.population_eligibility_spec);
}

struct CheckinResult {
  std::string task_name;
  ClientOnlyPlan plan;
  int32_t minimum_clients_in_server_visible_aggregate;
  std::string checkpoint_input_filename;
  std::string computation_id;
  std::string federated_select_uri_template;
};
absl::StatusOr<CheckinResult> IssueCheckin(
    PhaseLogger& phase_logger, LogManager* log_manager, Files* files,
    FederatedProtocol* federated_protocol,
    std::optional<TaskEligibilityInfo> task_eligibility_info,
    absl::Time reference_time, const std::string& population_name,
    FLRunnerResult& fl_runner_result, const Flags* flags) {
  absl::Time time_before_checkin = absl::Now();
  // We must return only stats that cover the check in phase for the log
  // events below.
  const NetworkStats network_stats_before_checkin =
      GetCumulativeNetworkStats(federated_protocol,
                                /*fedselect_manager=*/nullptr);

  // These fields will, after a successful checkin that resulted in a task being
  // assigned, contain the time at which the task plan/checkpoint URIs were
  // received (but not yet downloaded), as well as the cumulative network stats
  // at that point, allowing us to separately calculate how long it took to then
  // download the actual payloads.
  absl::Time time_before_plan_download = time_before_checkin;
  NetworkStats network_stats_before_plan_download =
      network_stats_before_checkin;

  // Clear the model identifier before check-in, to ensure that the any prior
  // eligibility eval task name isn't used any longer.
  phase_logger.SetModelIdentifier("");
  phase_logger.LogCheckinStarted();

  std::string task_name;

  // Issue the checkin request (providing a callback that will be called when an
  // EET is assigned to the task but before its plan/checkpoint URIs have
  // actually been downloaded).
  bool plan_uris_received_callback_called = false;
  std::function<void(const FederatedProtocol::TaskAssignment&)>
      plan_uris_received_callback =
          [&time_before_plan_download, &network_stats_before_plan_download,
           &time_before_checkin, &network_stats_before_checkin, &task_name,
           &federated_protocol, &population_name, &log_manager, &phase_logger,
           &plan_uris_received_callback_called](
              const FederatedProtocol::TaskAssignment& task_assignment) {
            // When the plan URIs have been received, we already know the name
            // of the task we have been assigned, so let's tell the PhaseLogger.
            auto model_identifier = task_assignment.aggregation_session_id;
            phase_logger.SetModelIdentifier(model_identifier);

            // We also should log a corresponding log event.
            task_name = ExtractTaskNameFromAggregationSessionId(
                model_identifier, population_name, *log_manager);
            phase_logger.LogCheckinPlanUriReceived(
                task_name,
                GetNetworkStatsSince(federated_protocol,
                                     /*fedselect_manager=*/nullptr,
                                     network_stats_before_checkin),
                time_before_checkin);

            // And we must take a snapshot of the current time & network stats,
            // so we can distinguish between the duration/network stats incurred
            // for the checkin request vs. the actual downloading of the
            // plan/checkpoint resources.
            time_before_plan_download = absl::Now();
            network_stats_before_plan_download = GetCumulativeNetworkStats(
                federated_protocol, /*fedselect_manager=*/nullptr);
            plan_uris_received_callback_called = true;
          };
  absl::StatusOr<FederatedProtocol::CheckinResult> checkin_result =
      federated_protocol->Checkin(task_eligibility_info,
                                  plan_uris_received_callback);
  UpdateRetryWindowAndNetworkStats(*federated_protocol,
                                   /*fedselect_manager=*/nullptr, phase_logger,
                                   fl_runner_result);

  // It's a bit unfortunate that we have to inspect the checkin_result and
  // extract the model identifier here rather than further down the function,
  // but this ensures that the histograms below will have the right model
  // identifier attached (and we want to also emit the histograms even if we
  // have failed/rejected checkin outcomes).
  if (checkin_result.ok() &&
      std::holds_alternative<FederatedProtocol::TaskAssignment>(
          *checkin_result)) {
    // Make sure that if we received a TaskAssignment, then the callback should
    // have already been called by this point by the protocol (ensuring that
    // SetModelIdentifier has been called etc.).
    FCP_CHECK(plan_uris_received_callback_called);
  }

  if (!checkin_result.ok()) {
    auto status = checkin_result.status();
    auto message = absl::StrCat("Error during checkin: code: ", status.code(),
                                ", message: ", status.message());
    if (status.code() == absl::StatusCode::kAborted) {
      phase_logger.LogCheckinServerAborted(
          status,
          GetNetworkStatsSince(federated_protocol,
                               /*fedselect_manager=*/nullptr,
                               network_stats_before_plan_download),
          time_before_plan_download, reference_time);
    } else if (status.code() == absl::StatusCode::kCancelled) {
      phase_logger.LogCheckinClientInterrupted(
          status,
          GetNetworkStatsSince(federated_protocol,
                               /*fedselect_manager=*/nullptr,
                               network_stats_before_plan_download),
          time_before_plan_download, reference_time);
    } else if (!status.ok()) {
      phase_logger.LogCheckinIOError(
          status,
          GetNetworkStatsSince(federated_protocol,
                               /*fedselect_manager=*/nullptr,
                               network_stats_before_plan_download),
          time_before_plan_download, reference_time);
    }
    FCP_LOG(INFO) << message;
    return status;
  }

  // Server rejected us? Return the fl_runner_results as-is.
  if (std::holds_alternative<FederatedProtocol::Rejection>(*checkin_result)) {
    phase_logger.LogCheckinTurnedAway(
        GetNetworkStatsSince(federated_protocol, /*fedselect_manager=*/nullptr,
                             network_stats_before_checkin),
        time_before_checkin, reference_time);
    FCP_LOG(INFO) << "Device rejected by server during checkin; aborting";
    return absl::InternalError("Device rejected by server.");
  }

  auto task_assignment =
      absl::get<FederatedProtocol::TaskAssignment>(*checkin_result);

  ClientOnlyPlan plan;
  auto plan_bytes = task_assignment.payloads.plan;
  if (!ParseFromStringOrCord(plan, plan_bytes)) {
    auto message = "Failed to parse received plan";
    phase_logger.LogCheckinInvalidPayload(
        message,
        GetNetworkStatsSince(federated_protocol, /*fedselect_manager=*/nullptr,
                             network_stats_before_plan_download),
        time_before_plan_download, reference_time);
    FCP_LOG(ERROR) << message;
    return absl::InternalError("");
  }

  std::string computation_id;
  if (flags->enable_computation_id()) {
    computation_id = ComputeSHA256FromStringOrCord(plan_bytes);
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

  absl::StatusOr<std::string> checkpoint_input_filename = "";
  // Example query plan does not have an input checkpoint.
  if (!plan.phase().has_example_query_spec()) {
    checkpoint_input_filename =
        CreateInputCheckpointFile(files, task_assignment.payloads.checkpoint);
    if (!checkpoint_input_filename.ok()) {
      auto status = checkpoint_input_filename.status();
      auto message = absl::StrCat(
          "Failed to create checkpoint input file: code: ", status.code(),
          ", message: ", status.message());
      phase_logger.LogCheckinIOError(
          status,
          GetNetworkStatsSince(federated_protocol,
                               /*fedselect_manager=*/nullptr,
                               network_stats_before_plan_download),
          time_before_plan_download, reference_time);
      FCP_LOG(ERROR) << message;
      return status;
    }
  }
  phase_logger.LogCheckinCompleted(
      task_name,
      GetNetworkStatsSince(federated_protocol, /*fedselect_manager=*/nullptr,
                           network_stats_before_plan_download),
      /*time_before_checkin=*/time_before_checkin,
      /*time_before_plan_download=*/time_before_plan_download, reference_time);
  return CheckinResult{
      .task_name = std::move(task_name),
      .plan = std::move(plan),
      .minimum_clients_in_server_visible_aggregate =
          minimum_clients_in_server_visible_aggregate,
      .checkpoint_input_filename = std::move(*checkpoint_input_filename),
      .computation_id = std::move(computation_id),
      .federated_select_uri_template =
          task_assignment.federated_select_uri_template};
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

  PhaseLoggerImpl phase_logger(event_publisher, opstats_logger.get(),
                               log_manager, flags);

  // If there was an error initializing OpStats, opstats_logger will be a no-op
  // implementation and execution will be allowed to continue.
  if (!opstats_logger->GetInitStatus().ok()) {
    // This will only happen if OpStats is enabled and there was an error in
    // initialization.
    phase_logger.LogNonfatalInitializationError(
        opstats_logger->GetInitStatus());
  }

  Clock* clock = Clock::RealClock();
  std::unique_ptr<cache::ResourceCache> resource_cache;
  if (flags->max_resource_cache_size_bytes() > 0) {
    // Anything that goes wrong in FileBackedResourceCache::Create is a
    // programmer error.
    absl::StatusOr<std::unique_ptr<cache::ResourceCache>>
        resource_cache_internal = cache::FileBackedResourceCache::Create(
            env_deps->GetBaseDir(), env_deps->GetCacheDir(), log_manager, clock,
            flags->max_resource_cache_size_bytes());
    if (!resource_cache_internal.ok()) {
      auto resource_init_failed_status = absl::Status(
          resource_cache_internal.status().code(),
          absl::StrCat("Failed to initialize FileBackedResourceCache: ",
                       resource_cache_internal.status().ToString()));
      if (flags->resource_cache_initialization_error_is_fatal()) {
        phase_logger.LogFatalInitializationError(resource_init_failed_status);
        return resource_init_failed_status;
      }
      // We log an error but otherwise proceed as if the cache was disabled.
      phase_logger.LogNonfatalInitializationError(resource_init_failed_status);
    } else {
      resource_cache = std::move(*resource_cache_internal);
    }
  }

  std::unique_ptr<::fcp::client::http::HttpClient> http_client =
      flags->enable_grpc_with_http_resource_support() ||
              flags->use_http_federated_compute_protocol()
          ? env_deps->CreateHttpClient()
          : nullptr;

  std::unique_ptr<FederatedProtocol> federated_protocol;
  if (flags->use_http_federated_compute_protocol()) {
    log_manager->LogDiag(ProdDiagCode::HTTP_FEDERATED_PROTOCOL_USED);
    // Verify the entry point uri starts with "https://" or "http://localhost".
    // Note "http://localhost" is allowed for testing purpose.
    if (!(absl::StartsWith(federated_service_uri, "https://") ||
          absl::StartsWith(federated_service_uri, "http://localhost"))) {
      return absl::InvalidArgumentError("The entry point uri is invalid.");
    }
    federated_protocol = std::make_unique<http::HttpFederatedProtocol>(
        clock, log_manager, flags, http_client.get(),
        std::make_unique<SecAggRunnerFactoryImpl>(),
        event_publisher->secagg_event_publisher(), federated_service_uri,
        api_key, population_name, retry_token, client_version,
        attestation_measurement, should_abort_protocol_callback, absl::BitGen(),
        timing_config, resource_cache.get());
  } else {
#ifdef FCP_CLIENT_SUPPORT_GRPC
    // Check in with the server to either retrieve a plan + initial checkpoint,
    // or get rejected with a RetryWindow.
    auto grpc_channel_deadline = flags->grpc_channel_deadline_seconds();
    if (grpc_channel_deadline <= 0) {
      grpc_channel_deadline = 600;
      FCP_LOG(INFO) << "Using default channel deadline of "
                    << grpc_channel_deadline << " seconds.";
    }

    federated_protocol = std::make_unique<GrpcFederatedProtocol>(
        event_publisher, log_manager,
        std::make_unique<SecAggRunnerFactoryImpl>(), flags, http_client.get(),
        federated_service_uri, api_key, test_cert_path, population_name,
        retry_token, client_version, attestation_measurement,
        should_abort_protocol_callback, timing_config, grpc_channel_deadline,
        resource_cache.get());
#else
    return absl::InternalError("No FederatedProtocol enabled.");
#endif
  }
  std::unique_ptr<FederatedSelectManager> federated_select_manager;
  if (http_client != nullptr && flags->enable_federated_select()) {
    federated_select_manager = std::make_unique<HttpFederatedSelectManager>(
        log_manager, files, http_client.get(), should_abort_protocol_callback,
        timing_config);
  } else {
    federated_select_manager =
        std::make_unique<DisabledFederatedSelectManager>(log_manager);
  }
  return RunFederatedComputation(env_deps, phase_logger, event_publisher, files,
                                 log_manager, opstats_logger.get(), flags,
                                 federated_protocol.get(),
                                 federated_select_manager.get(), timing_config,
                                 reference_time, session_name, population_name);
}

absl::StatusOr<FLRunnerResult> RunFederatedComputation(
    SimpleTaskEnvironment* env_deps, PhaseLogger& phase_logger,
    EventPublisher* event_publisher, Files* files, LogManager* log_manager,
    OpStatsLogger* opstats_logger, const Flags* flags,
    FederatedProtocol* federated_protocol,
    FederatedSelectManager* fedselect_manager,
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
  UpdateRetryWindowAndNetworkStats(*federated_protocol, fedselect_manager,
                                   phase_logger, fl_runner_result);

  // Check if the device conditions allow for checking in with the server
  // and running a federated computation. If not, bail early with the
  // transient error retry window.
  std::function<bool()> should_abort = [env_deps, &timing_config]() {
    return env_deps->ShouldAbort(absl::Now(), timing_config.polling_period);
  };
  if (should_abort()) {
    std::string message =
        "Device conditions not satisfied, aborting federated computation";
    FCP_LOG(INFO) << message;
    phase_logger.LogTaskNotStarted(message);
    return fl_runner_result;
  }

  // Eligibility eval plans can use example iterators from the
  // SimpleTaskEnvironment and those reading the OpStats DB.
  opstats::OpStatsExampleIteratorFactory opstats_example_iterator_factory(
      opstats_logger, log_manager,
      flags->opstats_last_successful_contribution_criteria());
  std::unique_ptr<engine::ExampleIteratorFactory>
      env_eligibility_example_iterator_factory =
          CreateSimpleTaskEnvironmentIteratorFactory(
              env_deps, eligibility_selector_context);
  std::vector<engine::ExampleIteratorFactory*>
      eligibility_example_iterator_factories{
          &opstats_example_iterator_factory,
          env_eligibility_example_iterator_factory.get()};

  // Note that this method will update fl_runner_result's fields with values
  // received over the course of the eligibility eval protocol interaction.
  absl::StatusOr<EligibilityEvalResult> eligibility_eval_result =
      IssueEligibilityEvalCheckinAndRunPlan(
          eligibility_example_iterator_factories, should_abort, phase_logger,
          files, log_manager, opstats_logger, flags, federated_protocol,
          timing_config, reference_time, fl_runner_result);
  if (!eligibility_eval_result.ok()) {
    return fl_runner_result;
  }

  auto checkin_result =
      IssueCheckin(phase_logger, log_manager, files, federated_protocol,
                   std::move(eligibility_eval_result->task_eligibility_info),
                   reference_time, population_name, fl_runner_result, flags);

  if (!checkin_result.ok()) {
    return fl_runner_result;
  }

  SelectorContext federated_selector_context_with_task_name =
      federated_selector_context;
  federated_selector_context_with_task_name.mutable_computation_properties()
      ->mutable_federated()
      ->set_task_name(checkin_result->task_name);
  if (flags->enable_computation_id()) {
    federated_selector_context_with_task_name.mutable_computation_properties()
        ->mutable_federated()
        ->set_computation_id(checkin_result->computation_id);
  }
  if (checkin_result->plan.phase().has_example_query_spec()) {
    federated_selector_context_with_task_name.mutable_computation_properties()
        ->set_example_iterator_output_format(
            ::fcp::client::QueryTimeComputationProperties::
                EXAMPLE_QUERY_RESULT);
  }

  // Include the last successful contribution timestamp in the SelectorContext.
  const auto& opstats_db = opstats_logger->GetOpStatsDb();
  if (opstats_db != nullptr) {
    absl::StatusOr<opstats::OpStatsSequence> data = opstats_db->Read();
    if (data.ok()) {
      std::optional<google::protobuf::Timestamp>
          last_successful_contribution_time =
              opstats::GetLastSuccessfulContributionTime(
                  *data, checkin_result->task_name);
      if (last_successful_contribution_time.has_value()) {
        *(federated_selector_context_with_task_name
              .mutable_computation_properties()
              ->mutable_federated()
              ->mutable_historical_context()
              ->mutable_last_successful_contribution_time()) =
            *last_successful_contribution_time;
      }
    }
  }

  if (checkin_result->plan.phase().has_example_query_spec()) {
    // Example query plan only supports simple agg for now.
    *(federated_selector_context_with_task_name
          .mutable_computation_properties()
          ->mutable_federated()
          ->mutable_simple_aggregation()) = SimpleAggregation();
  } else {
    const auto& federated_compute_io_router =
        checkin_result->plan.phase().federated_compute();
    const bool has_simpleagg_tensors =
        !federated_compute_io_router.output_filepath_tensor_name().empty();
    bool all_aggregations_are_secagg = true;
    for (const auto& aggregation : federated_compute_io_router.aggregations()) {
      all_aggregations_are_secagg &=
          aggregation.second.protocol_config_case() ==
          AggregationConfig::kSecureAggregation;
    }
    if (!has_simpleagg_tensors && all_aggregations_are_secagg) {
      federated_selector_context_with_task_name
          .mutable_computation_properties()
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
  }

  RetryWindow report_retry_window;
  phase_logger.LogComputationStarted();
  absl::Time run_plan_start_time = absl::Now();
  NetworkStats run_plan_start_network_stats =
      GetCumulativeNetworkStats(federated_protocol, fedselect_manager);
  absl::StatusOr<std::string> checkpoint_output_filename =
      files->CreateTempFile("output", ".ckp");
  if (!checkpoint_output_filename.ok()) {
    auto status = checkpoint_output_filename.status();
    auto message = absl::StrCat(
        "Could not create temporary output checkpoint file: code: ",
        status.code(), ", message: ", status.message());
    phase_logger.LogComputationIOError(
        status, ExampleStats(),
        GetNetworkStatsSince(federated_protocol, fedselect_manager,
                             run_plan_start_network_stats),
        run_plan_start_time);
    return fl_runner_result;
  }

  // Regular plans can use example iterators from the SimpleTaskEnvironment,
  // those reading the OpStats DB, or those serving Federated Select slices.
  std::unique_ptr<engine::ExampleIteratorFactory> env_example_iterator_factory =
      CreateSimpleTaskEnvironmentIteratorFactory(
          env_deps, federated_selector_context_with_task_name);
  std::unique_ptr<::fcp::client::engine::ExampleIteratorFactory>
      fedselect_example_iterator_factory =
          fedselect_manager->CreateExampleIteratorFactoryForUriTemplate(
              checkin_result->federated_select_uri_template);
  std::vector<engine::ExampleIteratorFactory*> example_iterator_factories{
      fedselect_example_iterator_factory.get(),
      &opstats_example_iterator_factory, env_example_iterator_factory.get()};

  PlanResultAndCheckpointFile plan_result_and_checkpoint_file =
      checkin_result->plan.phase().has_example_query_spec()
          ? RunPlanWithExampleQuerySpec(
                example_iterator_factories, opstats_logger, flags,
                checkin_result->plan, *checkpoint_output_filename)
          : RunPlanWithTensorflowSpec(
                example_iterator_factories, should_abort, log_manager,
                opstats_logger, flags, checkin_result->plan,
                checkin_result->checkpoint_input_filename,
                *checkpoint_output_filename, timing_config);
  // Update the FLRunnerResult fields to account for any network usage during
  // the execution of the plan (e.g. due to Federated Select slices having been
  // fetched).
  UpdateRetryWindowAndNetworkStats(*federated_protocol, fedselect_manager,
                                   phase_logger, fl_runner_result);
  auto outcome = plan_result_and_checkpoint_file.plan_result.outcome;
  absl::StatusOr<ComputationResults> computation_results;
  if (outcome == engine::PlanOutcome::kSuccess) {
    computation_results = CreateComputationResults(
        checkin_result->plan.phase().has_example_query_spec()
            ? nullptr
            : &checkin_result->plan.phase().tensorflow_spec(),
        plan_result_and_checkpoint_file);
  }
  LogComputationOutcome(
      plan_result_and_checkpoint_file.plan_result, computation_results.status(),
      phase_logger,
      GetNetworkStatsSince(federated_protocol, fedselect_manager,
                           run_plan_start_network_stats),
      run_plan_start_time, reference_time);
  absl::Status report_result = ReportPlanResult(
      federated_protocol, phase_logger, std::move(computation_results),
      run_plan_start_time, reference_time);
  if (outcome == engine::PlanOutcome::kSuccess && report_result.ok()) {
    // Only if training succeeded *and* reporting succeeded do we consider
    // the device to have contributed successfully.
    fl_runner_result.set_contribution_result(FLRunnerResult::SUCCESS);
  }

  // Update the FLRunnerResult fields one more time to account for the "Report"
  // protocol interaction.
  UpdateRetryWindowAndNetworkStats(*federated_protocol, fedselect_manager,
                                   phase_logger, fl_runner_result);

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
  std::function<bool()> should_abort = [env_deps, &timing_config]() {
    return env_deps->ShouldAbort(absl::Now(), timing_config.polling_period);
  };

  auto opstats_logger =
      engine::CreateOpStatsLogger(env_deps->GetBaseDir(), flags, log_manager,
                                  /*session_name=*/"", /*population_name=*/"");
  PhaseLoggerImpl phase_logger(event_publisher, opstats_logger.get(),
                               log_manager, flags);

  // Regular plans can use example iterators from the SimpleTaskEnvironment,
  // those reading the OpStats DB, or those serving Federated Select slices.
  // However, we don't provide a Federated Select-specific example iterator
  // factory. That way, the Federated Select slice queries will be forwarded
  // to SimpleTaskEnvironment, which can handle them by providing
  // test-specific slices if they want to.
  //
  // Eligibility eval plans can only use iterators from the
  // SimpleTaskEnvironment and those reading the OpStats DB.
  opstats::OpStatsExampleIteratorFactory opstats_example_iterator_factory(
      opstats_logger.get(), log_manager,
      flags->opstats_last_successful_contribution_criteria());
  std::unique_ptr<engine::ExampleIteratorFactory> env_example_iterator_factory =
      CreateSimpleTaskEnvironmentIteratorFactory(env_deps, SelectorContext());
  std::vector<engine::ExampleIteratorFactory*> example_iterator_factories{
      &opstats_example_iterator_factory, env_example_iterator_factory.get()};

  phase_logger.LogComputationStarted();
  if (client_plan.phase().has_federated_compute()) {
    absl::StatusOr<std::string> checkpoint_output_filename =
        files->CreateTempFile("output", ".ckp");
    if (!checkpoint_output_filename.ok()) {
      phase_logger.LogComputationIOError(
          checkpoint_output_filename.status(), ExampleStats(),
          // Empty network stats, since no network protocol is actually used
          // in this method.
          NetworkStats(), run_plan_start_time);
      return result;
    }
    // Regular TensorflowSpec-based plans.
    PlanResultAndCheckpointFile plan_result_and_checkpoint_file =
        RunPlanWithTensorflowSpec(example_iterator_factories, should_abort,
                                  log_manager, opstats_logger.get(), flags,
                                  client_plan, checkpoint_input_filename,
                                  *checkpoint_output_filename, timing_config);
    result.set_checkpoint_output_filename(
        plan_result_and_checkpoint_file.checkpoint_file);
    plan_result = std::move(plan_result_and_checkpoint_file.plan_result);
  } else if (client_plan.phase().has_federated_compute_eligibility()) {
    // Eligibility eval plans.
    plan_result = RunEligibilityEvalPlanWithTensorflowSpec(
        example_iterator_factories, should_abort, log_manager,
        opstats_logger.get(), flags, client_plan, checkpoint_input_filename,
        timing_config, run_plan_start_time, reference_time);
  } else {
    // This branch shouldn't be taken, unless we add an additional type of
    // TensorflowSpec-based plan in the future. We return a readable error so
    // that when such new plan types *are* added, they result in clear
    // compatibility test failures when such plans are erroneously targeted at
    // old releases that don't support them yet.
    event_publisher->PublishIoError("Unsupported TensorflowSpec-based plan");
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
    phase_logger.LogComputationCompleted(
        plan_result.example_stats,
        // Empty network stats, since no network protocol is actually used in
        // this method.
        NetworkStats(), run_plan_start_time, reference_time);
  } else {
    phase_logger.LogComputationTensorflowError(
        plan_result.original_status, plan_result.example_stats, NetworkStats(),
        run_plan_start_time, reference_time);
  }

  return result;
}

}  // namespace client
}  // namespace fcp
