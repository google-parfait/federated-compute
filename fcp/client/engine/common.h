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
#ifndef FCP_CLIENT_ENGINE_COMMON_H_
#define FCP_CLIENT_ENGINE_COMMON_H_

#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/engine/example_iterator_factory.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/stats.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace engine {

enum class PlanOutcome {
  kSuccess,
  // A TensorFlow error occurred.
  kTensorflowError,
  // Computation was interrupted.
  kInterrupted,
  // The input parameters are invalid.
  kInvalidArgument,
  // An example iterator error occurred.
  kExampleIteratorError,
};

// The result of a call to `SimplePlanEngine::RunPlan` or
// `TfLitePlanEngine::RunPlan`.
struct PlanResult {
  explicit PlanResult(PlanOutcome outcome, absl::Status status);

  // The outcome of the plan execution.
  PlanOutcome outcome;
  // The secagg tensors from the plan execution.
  absl::flat_hash_map<std::string, QuantizedTensor> secagg_tensor_map;
  // Only set if 'outcome' is 'kSuccess' and the federated compute wire format
  // is enabled, otherwise this is empty.
  absl::Cord federated_compute_checkpoint;
  // When the outcome is `kSuccess`, the status is ok. Otherwise, this status
  // contain the original error status which leads to the PlanOutcome.
  absl::Status original_status;
  ::fcp::client::ExampleStats example_stats;
  // Only set if the plan is an eligibility eval plan.
  absl::StatusOr<google::internal::federatedml::v2::TaskEligibilityInfo>
      task_eligibility_info;

  PlanResult(PlanResult&&) = default;
  PlanResult& operator=(PlanResult&&) = default;

  // Disallow copy and assign.
  PlanResult(const PlanResult&) = delete;
  PlanResult& operator=(const PlanResult&) = delete;
};

// Validates that the input tensors match what's inside the TensorflowSpec.
absl::Status ValidateTensorflowSpec(
    const google::internal::federated::plan::TensorflowSpec& tensorflow_spec,
    const absl::flat_hash_set<std::string>& expected_input_tensor_names_set,
    const std::vector<std::string>& output_names);

PhaseOutcome ConvertPlanOutcomeToPhaseOutcome(PlanOutcome plan_outcome);

absl::Status ConvertPlanOutcomeToStatus(engine::PlanOutcome outcome);

// Tracks whether any example iterator encountered an error during the
// computation (a single computation may use multiple iterators), either during
// creation of the iterator or during one of the iterations.
// This class is thread-safe.
class ExampleIteratorStatus {
 public:
  void SetStatus(absl::Status status) ABSL_LOCKS_EXCLUDED(mu_);
  absl::Status GetStatus() ABSL_LOCKS_EXCLUDED(mu_);

 private:
  absl::Status status_ ABSL_GUARDED_BY(mu_) = absl::OkStatus();
  mutable absl::Mutex mu_;
};

// Finds a suitable example iterator factory out of provided factories based on
// the provided selector.
ExampleIteratorFactory* FindExampleIteratorFactory(
    const google::internal::federated::plan::ExampleSelector& selector,
    std::vector<ExampleIteratorFactory*> example_iterator_factories);

}  // namespace engine
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_ENGINE_COMMON_H_
