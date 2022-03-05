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
#include "fcp/client/engine/common.h"

#include <string>

#include "fcp/base/monitoring.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace fcp {
namespace client {
namespace engine {

using ::google::internal::federated::plan::TensorflowSpec;

PlanResult::PlanResult(PlanOutcome outcome, absl::Status status)
    : outcome(outcome), original_status(std::move(status)) {
  if (outcome == PlanOutcome::kSuccess) {
    FCP_CHECK(original_status.ok());
  }
}

absl::Status ValidateTensorflowSpec(
    const TensorflowSpec& tensorflow_spec,
    const absl::flat_hash_set<std::string>& expected_input_tensor_names_set,
    const std::vector<std::string>& output_names) {
  // Check that all inputs have corresponding TensorSpecProtos.
  if (expected_input_tensor_names_set.size() !=
      tensorflow_spec.input_tensor_specs_size()) {
    return absl::InvalidArgumentError(
        "Unexpected number of input_tensor_specs");
  }

  for (const tensorflow::TensorSpecProto& it :
       tensorflow_spec.input_tensor_specs()) {
    if (!expected_input_tensor_names_set.contains(it.name())) {
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

PhaseOutcome ConvertPlanOutcomeToPhaseOutcome(PlanOutcome plan_outcome) {
  switch (plan_outcome) {
    case PlanOutcome::kSuccess:
      return PhaseOutcome::COMPLETED;
    case PlanOutcome::kInterrupted:
      return PhaseOutcome::INTERRUPTED;
    case PlanOutcome::kTensorflowError:
    case PlanOutcome::kInvalidArgument:
    case PlanOutcome::kExampleIteratorError:
      return PhaseOutcome::ERROR;
  }
}

absl::Status ConvertPlanOutcomeToStatus(PlanOutcome outcome) {
  switch (outcome) {
    case PlanOutcome::kSuccess:
      return absl::OkStatus();
    case PlanOutcome::kTensorflowError:
    case PlanOutcome::kInvalidArgument:
    case PlanOutcome::kExampleIteratorError:
      return absl::InternalError("");
    case PlanOutcome::kInterrupted:
      return absl::CancelledError("");
  }
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
