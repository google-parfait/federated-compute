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
#include <openssl/base.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "fcp/base/clock.h"
#include "fcp/base/digest.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/platform.h"
#include "fcp/client/cache/file_backed_resource_cache.h"
#include "fcp/client/cache/resource_cache.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/eligibility_decider.h"
#include "fcp/client/engine/common.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/engine/example_iterator_factory.h"
#include "fcp/client/engine/example_query_plan_engine.h"
#include "fcp/client/engine/plan_engine_helpers.h"
#include "fcp/client/engine/tflite_plan_engine.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/example_iterator_query_recorder.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/federated_select.h"
#include "fcp/client/files.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/http_federated_protocol.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_example_store.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/opstats/opstats_utils.h"
#include "fcp/client/parsing_utils.h"
#include "fcp/client/phase_logger.h"
#include "fcp/client/phase_logger_impl.h"
#include "fcp/client/secagg_runner.h"
#include "fcp/client/selector_context.pb.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/client/stats.h"
#include "fcp/client/task_result_info.pb.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/opstats.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/protos/population_eligibility_spec.pb.h"
#include "openssl/digest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/struct.pb.h"

#ifdef FCP_CLIENT_SUPPORT_TFMOBILE
#include "fcp/client/engine/simple_plan_engine.h"
#endif

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
  // The name of the output checkpoint file. Empty if the plan did not produce
  // an output checkpoint.
  std::string checkpoint_filename;

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
    const PlanResultAndCheckpointFile& plan_result_and_checkpoint_file,
    const Flags* flags) {
  const auto& [plan_result, checkpoint_filename] =
      plan_result_and_checkpoint_file;
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

  if (flags->enable_lightweight_client_report_wire_format()) {
    // Reads string from federated_compute_checkpoint
    if (plan_result.federated_compute_checkpoint.empty()) {
      return absl::InvalidArgumentError("Empty federate compute checkpoint");
    }
    computation_results[kFederatedComputeCheckpoint] =
        std::move(plan_result.federated_compute_checkpoint);
  } else {
    // Name of the TF checkpoint inside the aggregand map in the Checkpoint
    // protobuf. This field name is ignored by the server.
    if (!checkpoint_filename.empty()) {
      FCP_ASSIGN_OR_RETURN(std::string tf_checkpoint,
                           fcp::ReadFileToString(checkpoint_filename));
      computation_results[std::string(kTensorflowCheckpointAggregand)] =
          std::move(tf_checkpoint);
    }
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

// A helper class for running TensorFlowSpec Eligibility Eval Plans.
class EetPlanRunnerImpl : public EetPlanRunner {
 public:
  explicit EetPlanRunnerImpl(
      const std::function<engine::PlanResult(
          std::vector<engine::ExampleIteratorFactory*>)>& run_plan_func,
      const std::function<absl::StatusOr<TaskEligibilityInfo>(
          std::vector<tensorflow::Tensor>)>& parse_output_func)
      : run_plan_func_(run_plan_func), parse_output_func_(parse_output_func) {}

  engine::PlanResult RunPlan(std::vector<engine::ExampleIteratorFactory*>
                                 example_iterator_factories) override {
    return run_plan_func_(example_iterator_factories);
  }

  absl::StatusOr<TaskEligibilityInfo> ParseOutput(
      const std::vector<tensorflow::Tensor>& output_tensors) override {
    return parse_output_func_(output_tensors);
  }

  ~EetPlanRunnerImpl() override = default;

 private:
  std::function<engine::PlanResult(
      std::vector<engine::ExampleIteratorFactory*>)>
      run_plan_func_;
  std::function<absl::StatusOr<TaskEligibilityInfo>(
      std::vector<tensorflow::Tensor>)>
      parse_output_func_;
};

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
    SimpleTaskEnvironment* task_env, const SelectorContext& selector_context,
    PhaseLogger* phase_logger, const Flags* flags,
    bool should_log_collection_first_access_time) {
  return std::make_unique<engine::FunctionalExampleIteratorFactory>(
      /*can_handle_func=*/
      [](const google::internal::federated::plan::ExampleSelector&) {
        // The SimpleTaskEnvironment-based ExampleIteratorFactory should
        // be the catch-all factory that is able to handle all queries
        // that no other ExampleIteratorFactory is able to handle.
        return true;
      },
      /*create_iterator_func=*/
      [task_env, selector_context, should_log_collection_first_access_time,
       phase_logger,
       flags](const google::internal::federated::plan::ExampleSelector&
                  example_selector) {
        if (should_log_collection_first_access_time &&
            flags->log_collection_first_access_time()) {
          phase_logger->MaybeLogCollectionFirstAccessTime(
              example_selector.collection_uri());
        }
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

  const bool tflite_model_included = !client_plan.tflite_graph().empty();
  if (tflite_model_included) {
    log_manager->LogDiag(
        ProdDiagCode::BACKGROUND_TRAINING_TFLITE_MODEL_INCLUDED);
  }
  if (flags->use_tflite_training() && tflite_model_included) {
    std::unique_ptr<TfLiteInputs> tflite_inputs =
        ConstructTfLiteInputsForEligibilityEvalPlan(io_router,
                                                    checkpoint_input_filename);
    engine::TfLitePlanEngine plan_engine(
        example_iterator_factories, should_abort, log_manager, opstats_logger,
        flags, /*example_iterator_query_recorder=*/nullptr, &timing_config);
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
      /*example_iterator_query_recorder=*/nullptr, &timing_config);
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
    ExampleIteratorQueryRecorder* example_iterator_query_recorder,
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

  const bool tflite_model_included = !client_plan.tflite_graph().empty();
  if (tflite_model_included) {
    log_manager->LogDiag(
        ProdDiagCode::BACKGROUND_TRAINING_TFLITE_MODEL_INCLUDED);
  }
  // Run plan and get a set of output tensors back.
  if (flags->use_tflite_training() && tflite_model_included) {
    std::unique_ptr<TfLiteInputs> tflite_inputs =
        ConstructTFLiteInputsForTensorflowSpecPlan(
            client_plan.phase().federated_compute(), checkpoint_input_filename,
            checkpoint_output_filename);
    engine::TfLitePlanEngine plan_engine(
        example_iterator_factories, should_abort, log_manager, opstats_logger,
        flags, example_iterator_query_recorder, &timing_config);
    engine::PlanResult plan_result = plan_engine.RunPlan(
        client_plan.phase().tensorflow_spec(), client_plan.tflite_graph(),
        std::move(tflite_inputs), *output_names);
    PlanResultAndCheckpointFile result(std::move(plan_result));
    result.checkpoint_filename = checkpoint_output_filename;

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
      example_iterator_query_recorder, &timing_config);
  engine::PlanResult plan_result = plan_engine.RunPlan(
      client_plan.phase().tensorflow_spec(), client_plan.graph(),
      client_plan.tensorflow_config_proto(), std::move(inputs), *output_names);

  PlanResultAndCheckpointFile result(std::move(plan_result));
  result.checkpoint_filename = checkpoint_output_filename;

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
    ExampleIteratorQueryRecorder* example_iterator_query_recorder,
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
                                             opstats_logger, flags,
                                             example_iterator_query_recorder);
  engine::PlanResult plan_result = plan_engine.RunPlan(
      client_plan.phase().example_query_spec(), checkpoint_output_filename);
  PlanResultAndCheckpointFile result(std::move(plan_result));
  result.checkpoint_filename = checkpoint_output_filename;
  return result;
}

void LogNativeEligibilityEvalComputationOutcome(PhaseLogger& phase_logger,
                                                absl::Status status,
                                                absl::Time run_plan_start_time,
                                                absl::Time reference_time) {
  // The only possible error statuses we can get when evaluating a native eet
  // is INTERNAL from failing to read Opstats, or CANCELLED or INVALID_ARGUMENT
  // from an ExampleIterator.
  if (status.ok()) {
    phase_logger.LogEligibilityEvalComputationCompleted(
        ExampleStats(), run_plan_start_time, reference_time);
  } else if (absl::IsCancelled(status)) {
    phase_logger.LogEligibilityEvalComputationInterrupted(
        status, ExampleStats(), run_plan_start_time, reference_time);
  } else if (absl::IsInvalidArgument(status)) {
    phase_logger.LogEligibilityEvalComputationInvalidArgument(
        status, ExampleStats(), run_plan_start_time);
  } else {
    phase_logger.LogEligibilityEvalComputationIOError(
        status, ExampleStats(), run_plan_start_time, reference_time);
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
    absl::Time run_plan_start_time, absl::Time reference_time,
    std::optional<std::string> aggregation_session_id) {
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
        /*plan_duration=*/absl::Now() - run_plan_start_time,
        aggregation_session_id);
    LogResultUploadStatus(
        phase_logger, result,
        GetNetworkStatsSince(federated_protocol, /*fedselect_manager=*/nullptr,
                             before_report_stats),
        before_report_time, reference_time);
  } else {
    FCP_RETURN_IF_ERROR(phase_logger.LogFailureUploadStarted());
    result = federated_protocol->ReportNotCompleted(
        engine::PhaseOutcome::ERROR,
        /*plan_duration=*/absl::Now() - run_plan_start_time,
        aggregation_session_id);
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

absl::StatusOr<std::optional<TaskEligibilityInfo>> ComputeNativeEligibility(
    const PopulationEligibilitySpec& population_eligibility_spec,
    LogManager& log_manager, PhaseLogger& phase_logger,
    OpStatsLogger* opstats_logger, Clock& clock,
    std::vector<engine::ExampleIteratorFactory*> example_iterator_factories,
    EetPlanRunner& eet_plan_runner, const Flags* flags) {
  FCP_ASSIGN_OR_RETURN(opstats::OpStatsSequence opstats_sequence,
                       opstats_logger->GetOpStatsDb()->Read());

  FCP_ASSIGN_OR_RETURN(
      TaskEligibilityInfo task_eligibility_info,
      ComputeEligibility(population_eligibility_spec, log_manager, phase_logger,
                         opstats_sequence, clock, example_iterator_factories,
                         eet_plan_runner, flags));

  if (task_eligibility_info.task_weights_size() == 0) {
    // Eligibility could not be decided.
    return std::nullopt;
  }

  // Eligibility successfully computed via native eligibility!
  log_manager.LogDiag(
      ProdDiagCode::ELIGIBILITY_EVAL_NATIVE_COMPUTATION_SUCCESS);
  return task_eligibility_info;
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
    const NetworkStats& network_stats, Clock& clock) {
  ClientOnlyPlan plan;
  if (!ParseFromStringOrCord(plan, eligibility_eval_task.payloads.plan)) {
    auto message = "Failed to parse received eligibility eval plan";
    phase_logger.LogEligibilityEvalCheckinInvalidPayloadError(
        message, network_stats, time_before_plan_download);

    FCP_LOG(ERROR) << message;
    return absl::InternalError(message);
  }

  // TODO: b/325189386 - Remove checkpoint usage as native eligibility eval does
  // not require the checkpoint anymore.
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

  absl::Time run_computation_start_time = absl::Now();
  phase_logger.LogEligibilityEvalComputationStarted();

  absl::StatusOr<std::optional<TaskEligibilityInfo>>
      native_task_eligibility_info = std::nullopt;

  // population_eligibility_spec should always be set at this point, but just in
  // case we somehow get a legacy EET with no spec, we can skip evaluating this
  // and opt the client out of all tasks.
  if (eligibility_eval_task.population_eligibility_spec.has_value()) {
    std::function<engine::PlanResult(
        std::vector<engine::ExampleIteratorFactory*>)>
        run_plan_func = [&should_abort, &log_manager, &opstats_logger, &flags,
                         &plan, &checkpoint_input_filename, &timing_config,
                         &run_computation_start_time, &reference_time](
                            std::vector<engine::ExampleIteratorFactory*>
                                override_iterator_factories) {
          return RunEligibilityEvalPlanWithTensorflowSpec(
              override_iterator_factories, should_abort, log_manager,
              opstats_logger, flags, plan, *checkpoint_input_filename,
              timing_config, run_computation_start_time, reference_time);
        };

    std::function<absl::StatusOr<TaskEligibilityInfo>(
        std::vector<tensorflow::Tensor>)>
        parse_output_func = [](std::vector<tensorflow::Tensor> output_tensors) {
          return ParseEligibilityEvalPlanOutput(output_tensors);
        };

    EetPlanRunnerImpl eet_plan_runner(run_plan_func, parse_output_func);

    // TODO(team): Return ExampleStats out of the NEET engine so they can
    // be measured.
    native_task_eligibility_info = ComputeNativeEligibility(
        eligibility_eval_task.population_eligibility_spec.value(), *log_manager,
        phase_logger, opstats_logger, clock, example_iterator_factories,
        eet_plan_runner, flags);
  }

  LogNativeEligibilityEvalComputationOutcome(
      phase_logger, native_task_eligibility_info.status(),
      run_computation_start_time, reference_time);

  return native_task_eligibility_info;
}

struct EligibilityEvalResult {
  std::optional<TaskEligibilityInfo> task_eligibility_info;
  std::vector<std::string> task_names_for_multiple_task_assignments;
  bool population_supports_single_task_assignment = false;
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
  std::vector<std::string> task_names_for_multiple_task_assignments;
  if (population_spec.has_value()) {
    for (const auto& task_info : population_spec.value().task_info()) {
      if (task_info.task_assignment_mode() ==
          PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE) {
        task_names_for_multiple_task_assignments.push_back(
            task_info.task_name());
      } else if (task_info.task_assignment_mode() ==
                     PopulationEligibilitySpec::TaskInfo::
                         TASK_ASSIGNMENT_MODE_SINGLE ||
                 task_info.task_assignment_mode() ==
                     PopulationEligibilitySpec::TaskInfo::
                         TASK_ASSIGNMENT_MODE_UNSPECIFIED) {
        result.population_supports_single_task_assignment = true;
      }
    }
  } else {
    // If the server didn't return a PopulationEligibilitySpec, single task
    // assignment is enabled by default.
    result.population_supports_single_task_assignment = true;
  }

  if (task_eligibility_info.has_value()) {
    TaskEligibilityInfo single_task_assignment_eligibility_info;
    single_task_assignment_eligibility_info.set_version(
        task_eligibility_info.value().version());
    for (const auto& task_weight :
         task_eligibility_info.value().task_weights()) {
      const std::string& task_name = task_weight.task_name();
      if (task_weight.weight() > 0 &&
          std::find(task_names_for_multiple_task_assignments.begin(),
                    task_names_for_multiple_task_assignments.end(),
                    task_name) !=
              task_names_for_multiple_task_assignments.end()) {
        result.task_names_for_multiple_task_assignments.push_back(task_name);
      } else {
        *single_task_assignment_eligibility_info.mutable_task_weights()->Add() =
            task_weight;
      }
    }
    result.task_eligibility_info = single_task_assignment_eligibility_info;
  } else {
    result.task_eligibility_info = task_eligibility_info;
    result.task_names_for_multiple_task_assignments =
        std::move(task_names_for_multiple_task_assignments);
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
    const absl::Time reference_time, FLRunnerResult& fl_runner_result,
    Clock& clock) {
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
    auto eligibility_eval_disabled =
        std::get<FederatedProtocol::EligibilityEvalDisabled>(
            *eligibility_checkin_result);
    return CreateEligibilityEvalResult(
        /*task_eligibility_info=*/std::nullopt,
        eligibility_eval_disabled.population_eligibility_spec);
  }

  auto eligibility_eval_task = std::get<FederatedProtocol::EligibilityEvalTask>(
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
                               network_stats_before_plan_download),
          clock);
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
  std::string aggregation_session_id;
};

absl::StatusOr<CheckinResult> CreateCheckinResultFromTaskAssignment(
    const FederatedProtocol::TaskAssignment& task_assignment, Files* files,
    const std::function<void(absl::string_view, absl::string_view)>&
        log_invalid_payload_error,
    const std::function<void(absl::string_view, const absl::Status&)>&
        log_io_error,
    const Flags* flags) {
  ClientOnlyPlan plan;
  auto plan_bytes = task_assignment.payloads.plan;
  if (!ParseFromStringOrCord(plan, plan_bytes)) {
    auto message = "Failed to parse received plan";
    log_invalid_payload_error(task_assignment.task_name, message);
    FCP_LOG(ERROR) << message;
    return absl::InvalidArgumentError(message);
  }

  std::string computation_id;
  if (flags->enable_computation_id()) {
    if (flags->use_correct_sha256_impl_for_computation_id()) {
      if (std::holds_alternative<std::string>(plan_bytes)) {
        computation_id = ComputeSHA256(std::get<std::string>(plan_bytes));
      } else {
        computation_id = ComputeSHA256(std::get<absl::Cord>(plan_bytes));
      }
    } else {
      computation_id = ComputeSHA256FromStringOrCord(plan_bytes);
    }
  }

  int32_t minimum_clients_in_server_visible_aggregate = 0;
  if (task_assignment.sec_agg_info.has_value()) {
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
      log_io_error(task_assignment.task_name, status);
      FCP_LOG(ERROR) << message;
      return status;
    }
  }
  return CheckinResult{
      .task_name = task_assignment.task_name,
      .plan = std::move(plan),
      .minimum_clients_in_server_visible_aggregate =
          minimum_clients_in_server_visible_aggregate,
      .checkpoint_input_filename = std::move(*checkpoint_input_filename),
      .computation_id = std::move(computation_id),
      .federated_select_uri_template =
          task_assignment.federated_select_uri_template,
      .aggregation_session_id = task_assignment.aggregation_session_id};
}

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

  // Issue the checkin request (providing a callback that will be called when an
  // EET is assigned to the task but before its plan/checkpoint URIs have
  // actually been downloaded).
  bool plan_uris_received_callback_called = false;
  std::function<void(const FederatedProtocol::TaskAssignment&)>
      plan_uris_received_callback =
          [&time_before_plan_download, &network_stats_before_plan_download,
           &time_before_checkin, &network_stats_before_checkin,
           &federated_protocol, &phase_logger,
           &plan_uris_received_callback_called](
              const FederatedProtocol::TaskAssignment& task_assignment) {
            // When the plan URIs have been received, we already know the name
            // of the task we have been assigned, so let's tell the PhaseLogger.
            phase_logger.SetModelIdentifier(task_assignment.task_name);

            // We also should log a corresponding log event.
            phase_logger.LogCheckinPlanUriReceived(
                task_assignment.task_name,
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
      std::get<FederatedProtocol::TaskAssignment>(*checkin_result);
  std::function<void(absl::string_view, const absl::Status&)> log_io_error =
      [&phase_logger, &federated_protocol, &network_stats_before_plan_download,
       &time_before_plan_download,
       &reference_time](absl::string_view unused, const absl::Status& status) {
        phase_logger.LogCheckinIOError(
            status,
            GetNetworkStatsSince(federated_protocol,
                                 /*fedselect_manager=*/nullptr,
                                 network_stats_before_plan_download),
            time_before_plan_download, reference_time);
      };
  std::function<void(absl::string_view, absl::string_view)>
      log_invalid_payload_error = [&phase_logger, &federated_protocol,
                                   &network_stats_before_plan_download,
                                   &time_before_plan_download,
                                   &reference_time](absl::string_view unused,
                                                    absl::string_view message) {
        phase_logger.LogCheckinInvalidPayload(
            message,
            GetNetworkStatsSince(federated_protocol,
                                 /*fedselect_manager=*/nullptr,
                                 network_stats_before_plan_download),
            time_before_plan_download, reference_time);
      };
  absl::StatusOr<CheckinResult> result = CreateCheckinResultFromTaskAssignment(
      task_assignment, files, log_invalid_payload_error, log_io_error, flags);
  if (result.ok()) {
    // Only log the current index of the MinimumSeparationPolicy if the flag
    // `enable_minimum_separation_policy` is true and the `min_sep_policy_index`
    // is present in the `client_persisted_data`.
    std::optional<int64_t> min_sep_policy_index = std::nullopt;
    if (flags->enable_minimum_separation_policy() &&
        result.value().plan.has_client_persisted_data() &&
        result.value()
            .plan.client_persisted_data()
            .has_min_sep_policy_index()) {
      min_sep_policy_index =
          result.value().plan.client_persisted_data().min_sep_policy_index();
    }
    phase_logger.LogCheckinCompleted(
        result->task_name,
        GetNetworkStatsSince(federated_protocol,
                             /*fedselect_manager=*/nullptr,
                             network_stats_before_plan_download),
        /*time_before_checkin=*/time_before_checkin,
        /*time_before_plan_download=*/time_before_plan_download, reference_time,
        min_sep_policy_index);
  }
  return result;
}

SelectorContext FillSelectorContextWithTaskLevelDetails(
    const absl::StatusOr<CheckinResult>& checkin_result,
    const SelectorContext& federated_selector_context,
    OpStatsLogger* opstats_logger, const Flags* flags) {
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
      std::optional<
          absl::flat_hash_map<std::string, google::protobuf::Timestamp>>
          collection_first_access_times =
              opstats::GetPreviousCollectionFirstAccessTimeMap(
                  *data, checkin_result->task_name);
      if (collection_first_access_times.has_value()) {
        federated_selector_context_with_task_name
            .mutable_computation_properties()
            ->mutable_federated()
            ->mutable_historical_context()
            ->mutable_collection_first_access_times()
            ->insert(collection_first_access_times->begin(),
                     collection_first_access_times->end());
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
  return federated_selector_context_with_task_name;
}

// Issues a multiple task assignment request and tries to fetch payloads for all
// assigned tasks. Returns a vector of CheckinResult for the assigned tasks for
// which payloads were successfully fetched, or an empty vector if no tasks were
// assigned at all or if the request failed.
std::vector<CheckinResult> IssueMultipleTaskAssignments(
    const std::vector<std::string>& task_names, PhaseLogger& phase_logger,
    LogManager* log_manager, Files* files,
    FederatedProtocol* federated_protocol, FLRunnerResult& fl_runner_result,
    const std::string& population_name, absl::Time reference_time,
    const Flags* flags) {
  absl::Time time_before_multiple_task_assignments = absl::Now();
  // We must return only stats that cover the check in phase for the log
  // events below.
  const NetworkStats network_stats_before_multiple_task_assignments =
      GetCumulativeNetworkStats(federated_protocol,
                                /*fedselect_manager=*/nullptr);

  // These fields will, after a successful checkin that resulted in a task being
  // assigned, contain the time at which the task plan/checkpoint URIs were
  // received (but not yet downloaded), as well as the cumulative network stats
  // at that point, allowing us to separately calculate how long it took to then
  // download the actual payloads.
  absl::Time time_before_multiple_plans_download =
      time_before_multiple_task_assignments;
  NetworkStats network_stats_before_multiple_plans_download =
      network_stats_before_multiple_task_assignments;

  // Clear the model identifier before check-in, to ensure that any prior
  // eligibility eval task name isn't used any longer.
  phase_logger.SetModelIdentifier("");
  phase_logger.LogMultipleTaskAssignmentsStarted();

  // Issue the multiple task assignments request (providing a callback that will
  // be called when an EET is assigned to the task but before its
  // plan/checkpoint URIs have actually been downloaded).
  bool multiple_tasks_uris_received_callback_called = false;
  size_t task_cnt = task_names.size();
  std::function<void(size_t)> multiple_tasks_uris_received_callback =
      [&time_before_multiple_plans_download,
       &network_stats_before_multiple_plans_download,
       &time_before_multiple_task_assignments,
       &network_stats_before_multiple_task_assignments, &federated_protocol,
       &phase_logger, &multiple_tasks_uris_received_callback_called,
       task_cnt](int32_t pending_retrieval_task_assignment_cnt) {
        NetworkStats plan_uri_received_network_stats = GetNetworkStatsSince(
            federated_protocol,
            /*fedselect_manager=*/nullptr,
            network_stats_before_multiple_task_assignments);
        if (pending_retrieval_task_assignment_cnt < task_cnt) {
          phase_logger.LogMultipleTaskAssignmentsPlanUriPartialReceived(
              plan_uri_received_network_stats,
              time_before_multiple_task_assignments);
        } else {
          phase_logger.LogMultipleTaskAssignmentsPlanUriReceived(
              plan_uri_received_network_stats,
              time_before_multiple_task_assignments);
        }

        // And we must take a snapshot of the current time & network stats,
        // so we can distinguish between the duration/network stats incurred
        // for the checkin request vs. the actual downloading of the
        // plan/checkpoint resources.
        time_before_multiple_plans_download = absl::Now();
        network_stats_before_multiple_plans_download =
            GetCumulativeNetworkStats(federated_protocol,
                                      /*fedselect_manager=*/nullptr);
        multiple_tasks_uris_received_callback_called = true;
      };
  absl::StatusOr<FederatedProtocol::MultipleTaskAssignments>
      multiple_task_assignments =
          federated_protocol->PerformMultipleTaskAssignments(
              task_names, multiple_tasks_uris_received_callback);
  UpdateRetryWindowAndNetworkStats(*federated_protocol,
                                   /*fedselect_manager=*/nullptr, phase_logger,
                                   fl_runner_result);

  std::vector<CheckinResult> checkin_results;
  // If the overall result is not OK, it means that the task assignment request
  // failed completely without receiving any assignments and there's nothing
  // else for us to do.
  if (!multiple_task_assignments.ok()) {
    auto status = multiple_task_assignments.status();
    auto message = absl::StrCat(
        "Error during multiple task assignments: code: ", status.code(),
        ", message: ", status.message());
    if (status.code() == absl::StatusCode::kAborted) {
      phase_logger.LogMultipleTaskAssignmentsServerAborted(
          status,
          GetNetworkStatsSince(federated_protocol,
                               /*fedselect_manager=*/nullptr,
                               network_stats_before_multiple_plans_download),
          time_before_multiple_plans_download, reference_time);
    } else if (status.code() == absl::StatusCode::kCancelled) {
      phase_logger.LogMultipleTaskAssignmentsClientInterrupted(
          status,
          GetNetworkStatsSince(federated_protocol,
                               /*fedselect_manager=*/nullptr,
                               network_stats_before_multiple_plans_download),
          time_before_multiple_plans_download, reference_time);
    } else if (!status.ok()) {
      phase_logger.LogMultipleTaskAssignmentsIOError(
          status,
          GetNetworkStatsSince(federated_protocol,
                               /*fedselect_manager=*/nullptr,
                               network_stats_before_multiple_plans_download),
          time_before_multiple_plans_download, reference_time);
    }
    FCP_LOG(INFO) << message;
    return checkin_results;
  }

  if (multiple_task_assignments->task_assignments.empty()) {
    phase_logger.LogMultipleTaskAssignmentsTurnedAway(
        GetNetworkStatsSince(federated_protocol, /*fedselect_manager=*/nullptr,
                             network_stats_before_multiple_task_assignments),
        time_before_multiple_task_assignments, reference_time);
    FCP_LOG(INFO)
        << "Device issued multiple task assignments request, but zero task "
           "were assigned by server.";
    return checkin_results;
  }

  // Make sure that if we received at least one TaskAssignment, then the
  // callback should have already been called by this point by the protocol.
  FCP_CHECK(multiple_tasks_uris_received_callback_called);

  std::function<void(absl::string_view, absl::string_view)>
      log_invalid_payload_error = [&phase_logger](absl::string_view task_name,
                                                  absl::string_view message) {
        phase_logger.SetModelIdentifier(task_name);
        phase_logger.LogMultipleTaskAssignmentsInvalidPayload(message);
        // We reset the model identifier right away because the next log event
        // may not belong to the same task.
        phase_logger.SetModelIdentifier("");
      };
  std::function<void(absl::string_view, const absl::Status&)> log_io_error =
      [&phase_logger](absl::string_view task_name, const absl::Status& status) {
        phase_logger.SetModelIdentifier(task_name);
        phase_logger.LogMultipleTaskAssignmentsPayloadIOError(status);
        // We reset the model identifier right away because the next log event
        // may not belong to the same task.
        phase_logger.SetModelIdentifier("");
      };
  for (const auto& [task_name, task_assignment] :
       multiple_task_assignments->task_assignments) {
    if (!task_assignment.ok()) {
      // If the task assignment is not OK, it indicates that a task assignment
      // was received but that fetching the task's payloads failed.
      log_io_error(task_name, task_assignment.status());
      continue;
    }

    auto result = CreateCheckinResultFromTaskAssignment(
        *task_assignment, files, log_invalid_payload_error, log_io_error,
        flags);
    if (result.ok()) {
      checkin_results.push_back(*std::move(result));
    }
  }

  // If all the artifacts for the assigned tasks are retrieved successfully, we
  // consider multiple task assignments completed even the number of assigned
  // tasks is smaller than the number of tasks we originally requested.
  if (checkin_results.size() ==
      multiple_task_assignments->task_assignments.size()) {
    phase_logger.LogMultipleTaskAssignmentsCompleted(
        GetNetworkStatsSince(federated_protocol, /*fedselect_manager=*/nullptr,
                             network_stats_before_multiple_plans_download),
        /*time_before_multiple_task_assignments=*/
        time_before_multiple_task_assignments,
        /*time_before_plan_download=*/time_before_multiple_plans_download,
        reference_time);
  } else {
    phase_logger.LogMultipleTaskAssignmentsPartialCompleted(
        GetNetworkStatsSince(federated_protocol, /*fedselect_manager=*/nullptr,
                             network_stats_before_multiple_plans_download),
        /*time_before_multiple_task_assignments=*/
        time_before_multiple_task_assignments,
        /*time_before_plan_download=*/time_before_multiple_plans_download,
        reference_time);
  }
  return checkin_results;
}

struct RunPlanResults {
  engine::PlanOutcome outcome;
  absl::StatusOr<ComputationResults> computation_results;
  absl::Time run_plan_start_time;
};

RunPlanResults RunComputation(
    const absl::StatusOr<CheckinResult>& checkin_result,
    const SelectorContext& selector_context_with_task_details,
    SimpleTaskEnvironment* env_deps, PhaseLogger& phase_logger, Files* files,
    LogManager* log_manager, OpStatsLogger* opstats_logger, const Flags* flags,
    FederatedProtocol* federated_protocol,
    FederatedSelectManager* fedselect_manager,
    engine::ExampleIteratorFactory* opstats_example_iterator_factory,
    ExampleIteratorQueryRecorder* example_iterator_query_recorder,
    FLRunnerResult& fl_runner_result, const std::function<bool()>& should_abort,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time reference_time) {
  RetryWindow report_retry_window;
  phase_logger.LogComputationStarted(checkin_result->task_name);
  absl::Time run_plan_start_time = absl::Now();
  NetworkStats run_plan_start_network_stats =
      GetCumulativeNetworkStats(federated_protocol, fedselect_manager);

  std::string checkpoint_output_filename;
  bool task_requires_output_checkpoint = true;
  // A task does not require an output checkpoint if it is a lightweight task
  // and the new client report format is enabled, or if all of the outputs are
  // aggregated with secagg.
  if (flags->enable_lightweight_client_report_wire_format() &&
      checkin_result->plan.phase().has_example_query_spec()) {
    task_requires_output_checkpoint = false;
  } else if (checkin_result->plan.phase().has_federated_compute() &&
             checkin_result->plan.phase()
                 .federated_compute()
                 .output_filepath_tensor_name()
                 .empty()) {
    task_requires_output_checkpoint = false;
  }
  if (task_requires_output_checkpoint) {
    absl::StatusOr<std::string> output_filename =
        files->CreateTempFile("output", ".ckp");
    if (!output_filename.ok()) {
      auto status = output_filename.status();
      auto message = absl::StrCat(
          "Could not create temporary output checkpoint file: code: ",
          status.code(), ", message: ", status.message());
      phase_logger.LogComputationIOError(
          status, ExampleStats(),
          GetNetworkStatsSince(federated_protocol, fedselect_manager,
                               run_plan_start_network_stats),
          run_plan_start_time);
      return RunPlanResults{
          .outcome = engine::PlanOutcome::kInvalidArgument,
          .computation_results = absl::Status(status.code(), message),
          .run_plan_start_time = run_plan_start_time};
    }
    checkpoint_output_filename = *output_filename;
  }

  // Regular plans can use example iterators from the SimpleTaskEnvironment,
  // those reading the OpStats DB, or those serving Federated Select slices.
  // This iterator factory is used by the task to query the environment's
  // example store. We log first access time here to implement example-level
  // sampling without replacement for the environment.
  std::unique_ptr<engine::ExampleIteratorFactory> env_example_iterator_factory =
      CreateSimpleTaskEnvironmentIteratorFactory(
          env_deps, selector_context_with_task_details, &phase_logger, flags,
          /*should_log_collection_first_access_time=*/true);
  std::unique_ptr<::fcp::client::engine::ExampleIteratorFactory>
      fedselect_example_iterator_factory =
          fedselect_manager->CreateExampleIteratorFactoryForUriTemplate(
              checkin_result->federated_select_uri_template);
  std::vector<engine::ExampleIteratorFactory*> example_iterator_factories{
      fedselect_example_iterator_factory.get(),
      opstats_example_iterator_factory, env_example_iterator_factory.get()};

  PlanResultAndCheckpointFile plan_result_and_checkpoint_file =
      checkin_result->plan.phase().has_example_query_spec()
          ? RunPlanWithExampleQuerySpec(
                example_iterator_factories, opstats_logger, flags,
                example_iterator_query_recorder, checkin_result->plan,
                checkpoint_output_filename)
          : RunPlanWithTensorflowSpec(
                example_iterator_factories, should_abort, log_manager,
                opstats_logger, flags, example_iterator_query_recorder,
                checkin_result->plan, checkin_result->checkpoint_input_filename,
                checkpoint_output_filename, timing_config);
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
        plan_result_and_checkpoint_file, flags);
  }
  LogComputationOutcome(
      plan_result_and_checkpoint_file.plan_result, computation_results.status(),
      phase_logger,
      GetNetworkStatsSince(federated_protocol, fedselect_manager,
                           run_plan_start_network_stats),
      run_plan_start_time, reference_time);
  return RunPlanResults{.outcome = outcome,
                        .computation_results = std::move(computation_results),
                        .run_plan_start_time = run_plan_start_time};
}

std::vector<std::string> HandleMultipleTaskAssignments(
    const std::vector<CheckinResult>& multiple_task_assignments,
    const SelectorContext& federated_selector_context,
    SimpleTaskEnvironment* env_deps, PhaseLogger& phase_logger, Files* files,
    LogManager* log_manager, OpStatsLogger* opstats_logger, const Flags* flags,
    FederatedProtocol* federated_protocol,
    FederatedSelectManager* fedselect_manager,
    engine::ExampleIteratorFactory* opstats_example_iterator_factory,
    FLRunnerResult& fl_runner_result, const std::function<bool()>& should_abort,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time reference_time) {
  std::vector<std::string> successful_task_names;
  // We will try to run and report each task's results, even if one of those
  // steps fails for one of the tasks
  for (const auto& task_assignment : multiple_task_assignments) {
    phase_logger.SetModelIdentifier(task_assignment.task_name);
    SelectorContext selector_context_with_task_details =
        FillSelectorContextWithTaskLevelDetails(
            task_assignment, federated_selector_context, opstats_logger, flags);
    std::unique_ptr<ExampleIteratorQueryRecorder>
        example_iterator_query_recorder = nullptr;
    if (flags->enable_native_example_query_recording()) {
      example_iterator_query_recorder =
          std::make_unique<ExampleIteratorQueryRecorderImpl>(
              selector_context_with_task_details);
    }
    RunPlanResults run_plan_results = RunComputation(
        task_assignment, selector_context_with_task_details, env_deps,
        phase_logger, files, log_manager, opstats_logger, flags,
        federated_protocol, fedselect_manager, opstats_example_iterator_factory,
        example_iterator_query_recorder.get(), fl_runner_result, should_abort,
        timing_config, reference_time);
    absl::Status report_result =
        ReportPlanResult(federated_protocol, phase_logger,
                         std::move(run_plan_results.computation_results),
                         run_plan_results.run_plan_start_time, reference_time,
                         task_assignment.aggregation_session_id);
    TaskResultInfo task_result_info;
    if (run_plan_results.outcome == engine::PlanOutcome::kSuccess &&
        report_result.ok()) {
      // Only if training succeeded *and* reporting succeeded do we consider
      // the device to have contributed successfully.
      successful_task_names.push_back(task_assignment.task_name);
      task_result_info.set_result(true);
    } else {
      task_result_info.set_result(false);
    }
    if (flags->enable_native_example_query_recording() &&
        example_iterator_query_recorder) {
      *task_result_info.mutable_example_iterator_queries() =
          example_iterator_query_recorder->StopRecordingAndGetQueries();
    }
    env_deps->OnTaskCompleted(std::move(task_result_info));
  }
  return successful_task_names;
}

}  // namespace

absl::StatusOr<FLRunnerResult> RunFederatedComputation(
    SimpleTaskEnvironment* env_deps, EventPublisher* event_publisher,
    Files* files, LogManager* log_manager, const Flags* flags,
    const std::string& federated_service_uri, const std::string& api_key,
    const std::string& test_cert_path, const std::string& session_name,
    const std::string& population_name, const std::string& retry_token,
    const std::string& client_version,
    const std::string& client_attestation_measurement) {
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

  // Verify the entry point uri starts with "https://" or "http://localhost".
  // Note "http://localhost" is allowed for testing purpose.
  if (!(absl::StartsWith(federated_service_uri, "https://") ||
        absl::StartsWith(federated_service_uri, "http://localhost"))) {
    return absl::InvalidArgumentError("The entry point uri is invalid.");
  }

  std::unique_ptr<::fcp::client::http::HttpClient> http_client =
      env_deps->CreateHttpClient();
  std::unique_ptr<FederatedProtocol> federated_protocol =
      std::make_unique<http::HttpFederatedProtocol>(
          clock, log_manager, flags, http_client.get(),
          std::make_unique<SecAggRunnerFactoryImpl>(),
          event_publisher->secagg_event_publisher(), resource_cache.get(),
          env_deps->CreateAttestationVerifier(), federated_service_uri, api_key,
          population_name, retry_token, client_version,
          client_attestation_measurement, should_abort_protocol_callback,
          absl::BitGen(), timing_config);

  std::unique_ptr<FederatedSelectManager> federated_select_manager;
  if (flags->enable_federated_select()) {
    federated_select_manager = std::make_unique<HttpFederatedSelectManager>(
        log_manager, files, http_client.get(), should_abort_protocol_callback,
        timing_config);
  } else {
    federated_select_manager =
        std::make_unique<DisabledFederatedSelectManager>(log_manager);
  }
  return RunFederatedComputation(
      env_deps, phase_logger, event_publisher, files, log_manager,
      opstats_logger.get(), flags, federated_protocol.get(),
      federated_select_manager.get(), timing_config, reference_time,
      session_name, population_name, *clock);
}

absl::StatusOr<FLRunnerResult> RunFederatedComputation(
    SimpleTaskEnvironment* env_deps, PhaseLogger& phase_logger,
    EventPublisher* event_publisher, Files* files, LogManager* log_manager,
    OpStatsLogger* opstats_logger, const Flags* flags,
    FederatedProtocol* federated_protocol,
    FederatedSelectManager* fedselect_manager,
    const fcp::client::InterruptibleRunner::TimingConfig& timing_config,
    const absl::Time reference_time, const std::string& session_name,
    const std::string& population_name, Clock& clock) {
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
      opstats_logger, log_manager);
  // This iterator factory is used by the task to query the environment's
  // example store for eligibility, and thus does not log first access time
  // since we do not implement example-level SWOR for eligibility.
  std::unique_ptr<engine::ExampleIteratorFactory>
      env_eligibility_example_iterator_factory =
          CreateSimpleTaskEnvironmentIteratorFactory(
              env_deps, eligibility_selector_context, &phase_logger, flags,
              /*should_log_collection_first_access_time=*/false);
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
          timing_config, reference_time, fl_runner_result, clock);
  if (!eligibility_eval_result.ok()) {
    return fl_runner_result;
  }

  size_t expected_num_tasks = 0;
  std::vector<std::string> successful_task_names;

  // Run multiple task assignments first if enabled.
  if (flags->http_protocol_supports_multiple_task_assignments() &&
      !eligibility_eval_result->task_names_for_multiple_task_assignments
           .empty()) {
    expected_num_tasks = eligibility_eval_result
                             ->task_names_for_multiple_task_assignments.size();
    std::vector<CheckinResult> multiple_task_assignments =
        IssueMultipleTaskAssignments(
            eligibility_eval_result->task_names_for_multiple_task_assignments,
            phase_logger, log_manager, files, federated_protocol,
            fl_runner_result, population_name, reference_time, flags);
    successful_task_names = HandleMultipleTaskAssignments(
        multiple_task_assignments, federated_selector_context, env_deps,
        phase_logger, files, log_manager, opstats_logger, flags,
        federated_protocol, fedselect_manager,
        &opstats_example_iterator_factory, fl_runner_result, should_abort,
        timing_config, reference_time);
  }

  if (eligibility_eval_result->population_supports_single_task_assignment) {
    // Run single task assignment.

    // We increment expected_num_tasks because we expect the server to assign a
    // task.
    expected_num_tasks++;
    auto checkin_result =
        IssueCheckin(phase_logger, log_manager, files, federated_protocol,
                     std::move(eligibility_eval_result->task_eligibility_info),
                     reference_time, population_name, fl_runner_result, flags);

    if (checkin_result.ok()) {
      SelectorContext selector_context_with_task_details =
          FillSelectorContextWithTaskLevelDetails(checkin_result,
                                                  federated_selector_context,
                                                  opstats_logger, flags);
      std::unique_ptr<ExampleIteratorQueryRecorder>
          example_iterator_query_recorder = nullptr;
      if (flags->enable_native_example_query_recording()) {
        example_iterator_query_recorder =
            std::make_unique<ExampleIteratorQueryRecorderImpl>(
                selector_context_with_task_details);
      }

      auto run_plan_results = RunComputation(
          checkin_result, selector_context_with_task_details, env_deps,
          phase_logger, files, log_manager, opstats_logger, flags,
          federated_protocol, fedselect_manager,
          &opstats_example_iterator_factory,
          example_iterator_query_recorder.get(), fl_runner_result, should_abort,
          timing_config, reference_time);

      absl::Status report_result = ReportPlanResult(
          federated_protocol, phase_logger,
          std::move(run_plan_results.computation_results),
          run_plan_results.run_plan_start_time, reference_time, std::nullopt);
      TaskResultInfo task_result_info;
      if (run_plan_results.outcome == engine::PlanOutcome::kSuccess &&
          report_result.ok()) {
        // Only if training succeeded *and* reporting succeeded do we consider
        // the device to have contributed successfully.
        successful_task_names.push_back(checkin_result->task_name);
        task_result_info.set_result(true);
      } else {
        task_result_info.set_result(false);
      }
      if (flags->enable_native_example_query_recording() &&
          example_iterator_query_recorder) {
        *task_result_info.mutable_example_iterator_queries() =
            example_iterator_query_recorder->StopRecordingAndGetQueries();
      }
      env_deps->OnTaskCompleted(std::move(task_result_info));
    }
  }

  if (!successful_task_names.empty()) {
    if (successful_task_names.size() == expected_num_tasks) {
      fl_runner_result.set_contribution_result(FLRunnerResult::SUCCESS);
    } else {
      fl_runner_result.set_contribution_result(FLRunnerResult::PARTIAL);
    }
  }
  fl_runner_result.mutable_contributed_task_names()->Add(
      successful_task_names.begin(), successful_task_names.end());
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
    const absl::Time run_plan_start_time, const absl::Time reference_time,
    std::optional<PopulationEligibilitySpec> population_eligibility_spec) {
  FLRunnerTensorflowSpecResult result;
  result.set_outcome(engine::PhaseOutcome::ERROR);
  engine::PlanResult plan_result(engine::PlanOutcome::kTensorflowError,
                                 absl::UnknownError(""));
  std::function<bool()> should_abort = [env_deps, &timing_config]() {
    return env_deps->ShouldAbort(absl::Now(), timing_config.polling_period);
  };
  Clock* clock = Clock::RealClock();

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
      opstats_logger.get(), log_manager);
  std::unique_ptr<engine::ExampleIteratorFactory> env_example_iterator_factory =
      CreateSimpleTaskEnvironmentIteratorFactory(env_deps, SelectorContext(),
                                                 &phase_logger, flags, true);
  std::vector<engine::ExampleIteratorFactory*> example_iterator_factories{
      &opstats_example_iterator_factory, env_example_iterator_factory.get()};

  phase_logger.LogComputationStarted("");
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
                                  /*example_iterator_query_recorder=*/nullptr,
                                  client_plan, checkpoint_input_filename,
                                  *checkpoint_output_filename, timing_config);
    result.set_checkpoint_output_filename(
        plan_result_and_checkpoint_file.checkpoint_filename);
    plan_result = std::move(plan_result_and_checkpoint_file.plan_result);
  } else if (client_plan.phase().has_federated_compute_eligibility()) {
    if (population_eligibility_spec.has_value()) {
      std::function<engine::PlanResult(
          std::vector<engine::ExampleIteratorFactory*>)>
          run_plan_func =
              [&should_abort, &log_manager, &opstats_logger, &flags,
               &client_plan, &checkpoint_input_filename, &timing_config,
               &run_plan_start_time,
               &reference_time](std::vector<engine::ExampleIteratorFactory*>
                                    override_iterator_factories) {
                return RunEligibilityEvalPlanWithTensorflowSpec(
                    override_iterator_factories, should_abort, log_manager,
                    opstats_logger.get(), flags, client_plan,
                    checkpoint_input_filename, timing_config,
                    run_plan_start_time, reference_time);
              };

      std::function<absl::StatusOr<TaskEligibilityInfo>(
          std::vector<tensorflow::Tensor>)>
          parse_output_func =
              [](std::vector<tensorflow::Tensor> output_tensors) {
                return ParseEligibilityEvalPlanOutput(output_tensors);
              };

      EetPlanRunnerImpl eet_plan_runner(run_plan_func, parse_output_func);

      absl::StatusOr<std::optional<TaskEligibilityInfo>>
          native_task_eligibility_info = ComputeNativeEligibility(
              population_eligibility_spec.value(), *log_manager, phase_logger,
              opstats_logger.get(), *clock, example_iterator_factories,
              eet_plan_runner, flags);

      if (native_task_eligibility_info.ok()) {
        plan_result =
            engine::PlanResult(engine::PlanOutcome::kSuccess, absl::OkStatus());
      } else {
        plan_result = engine::PlanResult(engine::PlanOutcome::kTensorflowError,
                                         native_task_eligibility_info.status());
      }
    } else {
      // Legacy eligibility eval plan.
      plan_result = RunEligibilityEvalPlanWithTensorflowSpec(
          example_iterator_factories, should_abort, log_manager,
          opstats_logger.get(), flags, client_plan, checkpoint_input_filename,
          timing_config, run_plan_start_time, reference_time);
    }

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
