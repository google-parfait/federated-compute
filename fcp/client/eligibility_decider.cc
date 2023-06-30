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

#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "fcp/base/time_util.h"
#include "fcp/client/opstats/opstats_utils.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/protos/population_eligibility_spec.pb.h"
#include "re2/re2.h"

namespace fcp::client {

using ::google::internal::federated::plan::DataAvailabilityPolicy;
using ::google::internal::federated::plan::EligibilityPolicyEvalSpec;
using ::google::internal::federated::plan::ExampleSelector;
using ::google::internal::federated::plan::PopulationEligibilitySpec;
using ::google::internal::federated::plan::SamplingWithoutReplacementPolicy;
using ::google::internal::federatedml::v2::TaskEligibilityInfo;
using ::google::internal::federatedml::v2::TaskWeight;

namespace {

const int32_t kDataAvailabilityImplementationVersion = 1;
const int32_t kSworImplementationVersion = 1;
// TODO(team): Implement TF custom policies.
const int32_t kTfCustomPolicyImplementationVersion = -1;

bool ExecutionOutsideSworPeriod(
    absl::Time now, absl::Duration min_period,
    google::protobuf::Timestamp last_successful_contribution_time) {
  absl::Time last_successful_contribution_timestamp =
      fcp::TimeUtil::ConvertProtoToAbslTime(last_successful_contribution_time);
  absl::Duration period_between_last_successful_contribution =
      now - last_successful_contribution_timestamp;
  return min_period < period_between_last_successful_contribution;
}

// Computes sampling without replacement eligibility for the given set of tasks
// based on the execution history in opstats.
absl::flat_hash_set<std::string> ComputePerTaskSworEligibility(
    const SamplingWithoutReplacementPolicy& swor_policy,
    const absl::flat_hash_set<std::string>& task_names,
    const opstats::OpStatsSequence& opstats_sequence, Clock& clock) {
  // First, check that the period we're looking for does not exceed the opstats
  // db's trustworthiness timestamp. This timestamp tracks when the opstats db
  // was initialized, so if the swor period covers a time span beyond this
  // timestamp, it can not be guaranteed that the client did not successfully
  // participate within the period.
  absl::Duration min_period =
      fcp::TimeUtil::ConvertProtoToAbslDuration(swor_policy.min_period());
  absl::Time now = clock.Now();
  absl::Duration trustworthiness_period =
      now - fcp::TimeUtil::ConvertProtoToAbslTime(
                opstats_sequence.earliest_trustworthy_time());
  if (trustworthiness_period < min_period) {
    // No tasks eligible, return empty set
    return {};
  }

  // For group swor policies, since the period is consistent across all tasks,
  // and all tasks are in the same group, any previous successful contribution
  // matching the group_regex will opt out all tasks.
  if (!swor_policy.group_regex().empty()) {
    RE2 compiled_pattern(swor_policy.group_regex());
    if (!compiled_pattern.ok()) {
      // Bug in the group pattern, opt out all tasks. This should never happen
      // with the existing toolkit apis.
      return {};
    }
    std::optional<google::protobuf::Timestamp>
        last_successful_contribution_time =
            opstats::GetLastSuccessfulContributionTimeForPattern(
                opstats_sequence, compiled_pattern);

    // If there's no contribution by a task in the group, or if there are no
    // tasks in the group with an execution within the period, all tasks are
    // eligible.
    if (!last_successful_contribution_time.has_value() ||
        ExecutionOutsideSworPeriod(now, min_period,
                                   last_successful_contribution_time.value())) {
      return task_names;
    } else {
      return {};
    }
  }

  absl::flat_hash_set<std::string> eligibility_results;
  for (const std::string& task_name : task_names) {
    // TODO(team) Instead of using GetLastSuccessfulContributionTime
    // directly, consider passing in an opstats utils wrapper that can be
    // mocked.
    std::optional<google::protobuf::Timestamp>
        last_successful_contribution_time =
            opstats::GetLastSuccessfulContributionTime(opstats_sequence,
                                                       task_name);

    // If there was no last successful contribution time, we've never executed
    // this task.
    if (!last_successful_contribution_time.has_value()) {
      eligibility_results.insert(task_name);
      continue;
    }

    if (ExecutionOutsideSworPeriod(now, min_period,
                                   last_successful_contribution_time.value())) {
      // Most recent execution was outside the period, so we are eligible
      eligibility_results.insert(task_name);
      continue;
    }
  }

  return eligibility_results;
}

engine::ExampleIteratorFactory* FindExampleIteratorFactory(
    const ExampleSelector& selector,
    std::vector<engine::ExampleIteratorFactory*> example_iterator_factories) {
  for (engine::ExampleIteratorFactory* factory : example_iterator_factories) {
    // All ExampleIteratorFactories should be valid and never null.
    if (factory->CanHandle(selector)) {
      return factory;
    }
  }
  return nullptr;
}

// Returns true if the client has enough data to satisfy the policy, otherwise
// returns false. Returns an error status for exceptional cases.
absl::StatusOr<bool> ComputeDataAvailabilityEligibility(
    const DataAvailabilityPolicy& data_availability_policy,
    std::vector<engine::ExampleIteratorFactory*> example_iterator_factories) {
  ExampleSelector selector = data_availability_policy.selector();
  engine::ExampleIteratorFactory* iterator_factory =
      FindExampleIteratorFactory(selector, example_iterator_factories);

  if (iterator_factory == nullptr) {
    // The client does not host an iterator factory that can handle the given
    // selector.
    return absl::InvalidArgumentError(
        absl::StrCat("No iterator factory that can handle selector with uri ",
                     selector.collection_uri()));
  }

  FCP_ASSIGN_OR_RETURN(std::unique_ptr<ExampleIterator> iterator,
                       iterator_factory->CreateExampleIterator(selector));
  int32_t min_example_count = data_availability_policy.min_example_count();
  int examples_returned = 0;
  while (examples_returned < min_example_count) {
    absl::StatusOr<std::string> example = iterator->Next();
    // Next() returns the example bytestring on success, and:
    //  - CANCELLED if the call got interrupted.
    //  - INVALID_ARGUMENT if some other error occurred, e.g. I/O.
    //  - OUT_OF_RANGE if the end of the iterator was reached.
    //
    // Only on success and OUT_OF_RANGE should we continue execution.
    if (example.ok()) {
      examples_returned++;
      continue;
    }
    if (absl::IsOutOfRange(example.status())) {
      // No more examples, and the client did not return a sufficient amount to
      // satisfy the policy.
      return false;
    }
    // By here, the example is not ok, and the status is unexpected.
    return example.status();
  }

  // Enough examples returned!
  return true;
}

}  // namespace

absl::StatusOr<TaskEligibilityInfo> ComputeEligibility(
    const PopulationEligibilitySpec& population_eligibility_spec,
    LogManager& log_manager, const opstats::OpStatsSequence& opstats_sequence,
    Clock& clock,
    std::vector<engine::ExampleIteratorFactory*> example_iterator_factories) {
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
  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>
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
        log_manager.LogDiag(
            ProdDiagCode::ELIGIBILITY_EVAL_UNEXPECTED_POLICY_KIND);
        return eligibility_result;
    }

    absl::flat_hash_set<std::string> task_names;
    for (const auto& task_info : population_eligibility_spec.task_info()) {
      for (int i : task_info.eligibility_policy_indices()) {
        if (policy_index == i) {
          task_names.insert(task_info.task_name());
        }
      }
    }
    policy_name_to_task_names[policy_spec.name()] = task_names;
  }

  // Initialize the list of task names to compute eligibility for. Task names
  // will be removed from this list as policy computations mark them as
  // ineligible. For each policy computation, we only evaluate the intersection
  // of the tasks that use that policy and this list. This avoids computing
  // eligibility for tasks which have already been marked ineligible from
  // previous computations.
  absl::flat_hash_set<std::string> eligible_tasks;
  for (const auto& task_info : population_eligibility_spec.task_info()) {
    eligible_tasks.insert(task_info.task_name());
  }

  // For each policy:
  // 1. If eligible tasks is now empty due to the previous iteration, quit
  // early.
  // 2. Get task names from policy_name_to_task_names.
  // 3. Intersect with eligible_tasks to get the set of tasks names that we
  // should use to compute eligibility for this policy.
  // 4. Call compute x policy, get back a set of task names. This set will only
  // contain tasks that are still eligible according to the policy computation
  // we just did.
  // 5. Remove all tasks from eligible_tasks that are not eligible according to
  // the policy computation.

  for (const EligibilityPolicyEvalSpec& policy_spec :
       population_eligibility_spec.eligibility_policies()) {
    // Check if there are no more eligible tasks due to previous policy
    // computations. If so, there's no point in continuing to evaluate policies.
    if (eligible_tasks.empty()) {
      break;
    }

    // The task names this policy applies to.
    absl::flat_hash_set<std::string> policy_task_names =
        policy_name_to_task_names.at(policy_spec.name());
    // Remove those tasks that already aren't eligible anymore anyway.
    absl::erase_if(policy_task_names,
                   [&eligible_tasks](const std::string& policy_task_name) {
                     return !eligible_tasks.contains(policy_task_name);
                   });

    // If policy_task_names is now empty, there's no need to compute eligibility
    // for this policy.
    if (policy_task_names.empty()) {
      continue;
    }

    absl::flat_hash_set<std::string> eligible_policy_task_names;
    switch (policy_spec.policy_type_case()) {
      case EligibilityPolicyEvalSpec::PolicyTypeCase::kSworPolicy:
        eligible_policy_task_names = ComputePerTaskSworEligibility(
            policy_spec.swor_policy(), policy_task_names, opstats_sequence,
            clock);
        break;
      case EligibilityPolicyEvalSpec::PolicyTypeCase::kDataAvailabilityPolicy:
        // Need to wrap initialization of local variables in a case statement.
        {
          FCP_ASSIGN_OR_RETURN(bool data_is_available,
                               ComputeDataAvailabilityEligibility(
                                   policy_spec.data_availability_policy(),
                                   example_iterator_factories));
          if (data_is_available) {
            eligible_policy_task_names = policy_task_names;
          }
        }
        break;
      default:
        // Should never happen, because we pre-filtered above based on
        // policy kind.
        return absl::InternalError(
            absl::StrCat("Unexpected policy kind during eval: ",
                         policy_spec.policy_type_case()));
    }

    // Merge computed eligible tasks with overall eligible tasks. If a task name
    // is in policy_task_names but not in eligible_policy_task_names,
    // the runtime is not eligible for that task.
    for (const std::string& task_name : policy_task_names) {
      if (!eligible_policy_task_names.contains(task_name)) {
        eligible_tasks.erase(task_name);
      }
    }
  }

  // Now, eligible_tasks only contains task names that are eligible for all of
  // their applied policies.
  eligibility_result.set_version(1);
  for (const auto& task_info : population_eligibility_spec.task_info()) {
    TaskWeight* task_weight = eligibility_result.mutable_task_weights()->Add();
    task_weight->set_task_name(task_info.task_name());
    task_weight->set_weight(
        eligible_tasks.contains(task_info.task_name()) ? 1.0f : 0);
  }
  return eligibility_result;
}

}  // namespace fcp::client
