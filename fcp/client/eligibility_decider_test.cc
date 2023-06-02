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

#include "google/protobuf/duration.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/population_eligibility_spec.pb.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace client {

using ::google::internal::federated::plan::EligibilityPolicyEvalSpec;
using ::google::internal::federated::plan::PopulationEligibilitySpec;
using ::testing::NiceMock;

PopulationEligibilitySpec GenNoPoliciesSpec(int num_tasks) {
  PopulationEligibilitySpec spec;
  for (int i = 0; i < num_tasks; ++i) {
    PopulationEligibilitySpec::TaskInfo* task_info =
        spec.mutable_task_info()->Add();
    task_info->set_task_name(absl::StrCat("task_", i));
    task_info->set_task_assignment_mode(
        PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  }
  return spec;
}

class EligibilityDeciderTest : public testing::Test {
 protected:
  NiceMock<MockLogManager> mock_log_manager_;
};

TEST_F(EligibilityDeciderTest, NoPoliciesEligibleForAllTasks) {
  int num_tasks = 4;
  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(GenNoPoliciesSpec(num_tasks), &mock_log_manager_);

  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), num_tasks);

  for (const auto& task_weight : eligibility_result->task_weights()) {
    ASSERT_EQ(task_weight.weight(), 1.0f);
  }
}

TEST_F(EligibilityDeciderTest, SworPolicyReturnsNullOpt) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* swor_spec =
      spec.mutable_eligibility_policies()->Add();
  swor_spec->set_name("swor_policy_5_seconds");
  swor_spec->set_min_version(1);
  swor_spec->mutable_swor_policy()->mutable_min_period()->set_seconds(5);

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name(absl::StrCat("single_task_1"));
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  // Result should be ok, but because sampling without replacement is
  // unimplemented, we get an empty TaskEligibilityInfo.
  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, &mock_log_manager_);
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 0);
}

TEST_F(EligibilityDeciderTest, DataAvailabilityPolicyReturnsNullOpt) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* da_spec =
      spec.mutable_eligibility_policies()->Add();
  da_spec->set_name("da_policy_3_examples");
  da_spec->set_min_version(1);
  da_spec->mutable_data_availability_policy()->set_min_example_count(3);

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name(absl::StrCat("single_task_1"));
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  // Result should be ok, but because data availability is  unimplemented, we
  // get an empty TaskEligibilityInfo.
  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, &mock_log_manager_);
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 0);
}

TEST_F(EligibilityDeciderTest, TfCustomPolicyReturnsNullOpt) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* tf_spec =
      spec.mutable_eligibility_policies()->Add();
  tf_spec->set_name("tf_custom_policy");
  tf_spec->set_min_version(1);
  *tf_spec->mutable_tf_custom_policy()->mutable_arguments() =
      "hi hello how are you";

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name(absl::StrCat("single_task_1"));
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  // Result should be ok, but because TF custom policies are unimplemented, we
  // get an empty TaskEligibilityInfo.
  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, &mock_log_manager_);
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 0);
}

}  // namespace client
}  // namespace fcp
