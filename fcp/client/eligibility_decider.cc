/*
 * Copyright 2023 Google LLC
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

#include "fcp/client/eligibility_decider.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace fcp::client {

using ::google::internal::federated::plan::EligibilityPolicyEvalSpec;
using ::google::internal::federated::plan::PopulationEligibilitySpec;
using ::google::internal::federatedml::v2::TaskEligibilityInfo;
using ::google::internal::federatedml::v2::TaskWeight;

namespace {

// All policy kinds are currently unimplemented.
// TODO(team): Implement data availability.
const int32_t kDataAvailabilityImplementationVersion = -1;
// TODO(team): Implement sampling without replacement.
const int32_t kSworImplementationVersion = -1;
// TODO(team): Implement TF custom policies.
const int32_t kTfCustomPolicyImplementationVersion = -1;

}  // namespace

absl::StatusOr<TaskEligibilityInfo> ComputeEligibility(
    const PopulationEligibilitySpec& population_eligibility_spec,
    LogManager* log_manager) {
  // Initialize map of task name -> eligibility. This will be converted into a
  // TaskEligibilityInfo at the end.
  absl::flat_hash_map<std::string, bool> overall_eligibility;
  for (const auto& task_info : population_eligibility_spec.task_info()) {
    overall_eligibility[task_info.task_name()] = true;
  }

  // Initialize the TaskEligibilityInfo to return. If eligibility cannot be
  // decided, i.e. due to insufficient implementations, we'll return this
  // unfilled.
  TaskEligibilityInfo eligibility_result;

  // Initialize map of policy name -> task names that use that policy, and check
  // that the implementation versions for each policy are supported by the
  // client. If they are not, return nullopt.
  //
  // (At this point we support no policies, so if we use any at all, this will
  // return nullopt).
  absl::flat_hash_map<std::string, std::vector<std::string>>
      policy_name_to_task_names;
  for (int policy_index = 0;
       policy_index < population_eligibility_spec.eligibility_policies_size();
       policy_index++) {
    const EligibilityPolicyEvalSpec& policy_spec =
        population_eligibility_spec.eligibility_policies(policy_index);

    // TODO(team): After the three base policy kinds (da, swor, tf
    // custom) have been implemented, change this to mark the client ineligible
    // for tasks that do not implement the specified policy, instead of
    // returning a completely empty TaskEligibilityInfo.
    switch (policy_spec.policy_type_case()) {
      case EligibilityPolicyEvalSpec::PolicyTypeCase::kDataAvailabilityPolicy:
        if (kDataAvailabilityImplementationVersion <
            policy_spec.min_version()) {
          return eligibility_result;
        }
        break;
      case EligibilityPolicyEvalSpec::PolicyTypeCase::kSworPolicy:
        if (kSworImplementationVersion < policy_spec.min_version()) {
          return eligibility_result;
        }
        break;
      case EligibilityPolicyEvalSpec::PolicyTypeCase::kTfCustomPolicy:
        if (kTfCustomPolicyImplementationVersion < policy_spec.min_version()) {
          return eligibility_result;
        }
        break;
      default:
        // Unknown policy type! This can happen if a new policy type has been
        // added on the server, but is not yet implemented in the client. This
        // should be updated similarly when the client supports the three base
        // policy kinds.
        log_manager->LogDiag(
            ProdDiagCode::ELIGIBILITY_EVAL_UNEXPECTED_POLICY_KIND);
        return eligibility_result;
    }

    std::vector<std::string> task_names;
    for (const auto& task_info : population_eligibility_spec.task_info()) {
      for (int i : task_info.eligibility_policy_indices()) {
        if (policy_index == i) {
          task_names.push_back(task_info.task_name());
        }
      }
    }
    policy_name_to_task_names[policy_spec.name()] = task_names;
  }

  // TODO(team): Evaluate the policies!

  // Now, the overall_eligibility map is true/false depending on if the client
  // is eligible for the task. Convert the overall_eligibility map into a
  // TaskEligibilityInfo.
  eligibility_result.set_version(1);
  for (const auto& task_eligibility : overall_eligibility) {
    TaskWeight* task_weight = eligibility_result.mutable_task_weights()->Add();
    task_weight->set_task_name(task_eligibility.first);
    task_weight->set_weight(task_eligibility.second ? 1.0f : 0);
  }
  return eligibility_result;
}

}  // namespace fcp::client
