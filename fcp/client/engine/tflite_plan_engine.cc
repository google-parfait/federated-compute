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
#include <vector>

#include "fcp/client/engine/plan_engine_helpers.h"
#include "fcp/client/engine/tflite_wrapper.h"
#include "fcp/protos/plan.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace fcp {
namespace client {
namespace engine {

using ::google::internal::federated::plan::TensorflowSpec;

namespace {

PlanResult CreatePlanResultFromOutput(
    absl::StatusOr<OutputTensors> output, std::atomic<int>* total_example_count,
    std::atomic<int64_t>* total_example_size_bytes,
    absl::Status example_iterator_status) {
  switch (output.status().code()) {
    case absl::StatusCode::kOk: {
      PlanResult plan_result(PlanOutcome::kSuccess, absl::OkStatus());
      plan_result.output_names = std::move(output->output_tensor_names);
      plan_result.output_tensors = std::move(output->output_tensors);
      plan_result.example_stats = {
          .example_count = *total_example_count,
          .example_size_bytes = *total_example_size_bytes};
      return plan_result;
    }
    case absl::StatusCode::kCancelled:
      return PlanResult(PlanOutcome::kInterrupted, std::move(output.status()));
    case absl::StatusCode::kInvalidArgument:
      return CreateComputationErrorPlanResult(example_iterator_status,
                                              output.status());
    default:
      FCP_LOG(FATAL) << "unexpected status code: " << output.status().code();
  }
  // Unreachable code.
  return PlanResult(PlanOutcome::kTensorflowError, absl::InternalError(""));
}

TfLiteInterpreterOptions CreateOptions(const Flags& flags) {
  return TfLiteInterpreterOptions{
      .ensure_dynamic_tensors_are_released =
          flags.ensure_dynamic_tensors_are_released(),
      .large_tensor_threshold_for_dynamic_allocation =
          flags.large_tensor_threshold_for_dynamic_allocation(),
      .disable_delegate_clustering =
          flags.disable_tflite_delegate_clustering()};
}
}  // namespace

PlanResult TfLitePlanEngine::RunPlan(
    const TensorflowSpec& tensorflow_spec, const std::string& model,
    std::unique_ptr<absl::flat_hash_map<std::string, std::string>> inputs,
    const std::vector<std::string>& output_names) {
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
    return PlanResult(PlanOutcome::kInvalidArgument,
                      std::move(validity_checks));
  }
  std::atomic<int> total_example_count = 0;
  std::atomic<int64_t> total_example_size_bytes = 0;
  ExampleIteratorStatus example_iterator_status;
  HostObjectRegistration host_registration = AddDatasetTokenToInputsForTfLite(
      example_iterator_factories_, opstats_logger_, inputs.get(),
      tensorflow_spec.dataset_token_tensor_name(), &total_example_count,
      &total_example_size_bytes, &example_iterator_status);
  absl::StatusOr<std::unique_ptr<TfLiteWrapper>> tflite_wrapper =
      TfLiteWrapper::Create(model, should_abort_, *timing_config_, log_manager_,
                            std::move(inputs), output_names,
                            CreateOptions(flags_),
                            flags_.num_threads_for_tflite());
  if (!tflite_wrapper.ok()) {
    return PlanResult(PlanOutcome::kTensorflowError, tflite_wrapper.status());
  }
  // Start running the plan.
  absl::StatusOr<OutputTensors> output = (*tflite_wrapper)->Run();
  PlanResult plan_result = CreatePlanResultFromOutput(
      std::move(output), &total_example_count, &total_example_size_bytes,
      example_iterator_status.GetStatus());
  return plan_result;
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
