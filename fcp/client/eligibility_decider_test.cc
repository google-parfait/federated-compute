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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "google/protobuf/timestamp.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "fcp/base/simulated_clock.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/population_eligibility_spec.pb.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace client {

using ::google::internal::federated::plan::EligibilityPolicyEvalSpec;
using ::google::internal::federated::plan::PopulationEligibilitySpec;
using ::testing::NiceMock;
using ::testing::Return;

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

opstats::OperationalStats::Event CreateOpstatsEvent(
    opstats::OperationalStats::Event::EventKind event_kind,
    int64_t event_time_seconds) {
  opstats::OperationalStats::Event event;
  event.set_event_type(event_kind);
  google::protobuf::Timestamp t;
  t.set_seconds(event_time_seconds);
  *event.mutable_timestamp() = t;
  return event;
}

std::unique_ptr<engine::ExampleIteratorFactory> SetUpExampleIteratorFactory(
    int num_examples) {
  return std::make_unique<engine::FunctionalExampleIteratorFactory>(
      [num_examples](
          const google::internal::federated::plan::ExampleSelector& selector) {
        std::vector<const char*> examples;
        for (int i = 0; i < num_examples; i++) {
          examples.push_back("(｡◕‿◕｡)");
        }
        return std::make_unique<SimpleExampleIterator>(examples);
      });
}

class EligibilityDeciderTest : public testing::Test {
 protected:
  NiceMock<MockLogManager> mock_log_manager_;
  SimulatedClock clock_;
  std::vector<engine::ExampleIteratorFactory*> example_iterator_factories_;
};

opstats::OpStatsSequence GenOpstatsSequence() { return {}; }

TEST_F(EligibilityDeciderTest, NoPoliciesEligibleForAllTasks) {
  int num_tasks = 4;
  absl::StatusOr<TaskEligibilityInfo> eligibility_result = ComputeEligibility(
      GenNoPoliciesSpec(num_tasks), mock_log_manager_, GenOpstatsSequence(),
      clock_, {SetUpExampleIteratorFactory(0).get()});

  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), num_tasks);

  for (const auto& task_weight : eligibility_result->task_weights()) {
    ASSERT_EQ(task_weight.weight(), 1.0f);
  }
}

TEST_F(EligibilityDeciderTest, SworPolicyIsEligible) {
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

  opstats::OpStatsSequence opstats_sequence;
  // Trustworthy since epoch time
  opstats_sequence.mutable_earliest_trustworthy_time()->set_seconds(0);

  // Set the clock to epoch + 5 seconds. When we evaluate the swor policy, we'll
  // look back 5 seconds into the past and see that there are no previous
  // opstats entries for this task.
  clock_.AdvanceTime(absl::Seconds(5));

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, opstats_sequence, clock_,
                         {SetUpExampleIteratorFactory(0).get()});
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 1);
  // Eligible according to swor.
  ASSERT_EQ(eligibility_result->task_weights().at(0).weight(), 1.0f);
}

TEST_F(EligibilityDeciderTest, SworPolicyIsNotEligible) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* swor_spec =
      spec.mutable_eligibility_policies()->Add();
  swor_spec->set_name("swor_policy_5_seconds");
  swor_spec->set_min_version(1);
  swor_spec->mutable_swor_policy()->mutable_min_period()->set_seconds(5);

  std::string task_name = "single_task_1";
  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name(absl::StrCat(task_name));
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  opstats::OperationalStats stats;
  stats.set_task_name(task_name);
  int64_t upload_started_time_sec = 4;
  stats.mutable_events()->Add(CreateOpstatsEvent(
      opstats::OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED,
      upload_started_time_sec));
  opstats::OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(stats);
  // Trustworthy since epoch time
  opstats_sequence.mutable_earliest_trustworthy_time()->set_seconds(0);

  // Set the clock to epoch + 5 seconds. When we evaluate the swor policy, we'll
  // look back 5 seconds into the past and see our upload started at epoch + 4
  // seconds, and thus be ineligible.
  clock_.AdvanceTime(absl::Seconds(5));

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, opstats_sequence, clock_,
                         {SetUpExampleIteratorFactory(0).get()});
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 1);
  // Ineligible according to swor.
  ASSERT_EQ(eligibility_result->task_weights().at(0).weight(), 0);
}

// Tests that a task is marked ineligible if any of its policies consider it
// ineligible, even if all other policies consider it eligible.
TEST_F(EligibilityDeciderTest, IsNotEligibleIfIneligibleForAtLeastOnePolicy) {
  // In the real world, we would never have two swor policies on one task,
  // because the upper bound is the only one that matters. However it is the
  // only policy we have implemented at the moment.
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* swor_spec =
      spec.mutable_eligibility_policies()->Add();
  swor_spec->set_name("swor_policy_5_seconds");
  swor_spec->set_min_version(1);
  swor_spec->mutable_swor_policy()->mutable_min_period()->set_seconds(5);

  EligibilityPolicyEvalSpec* swor_spec2 =
      spec.mutable_eligibility_policies()->Add();
  swor_spec2->set_name("swor_policy_1_second");
  swor_spec2->set_min_version(1);
  swor_spec2->mutable_swor_policy()->mutable_min_period()->set_seconds(1);

  std::string task_name = "single_task_1";
  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name(absl::StrCat(task_name));
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);
  task_info->mutable_eligibility_policy_indices()->Add(1);

  opstats::OperationalStats stats;
  stats.set_task_name(task_name);
  int64_t upload_started_time_sec = 1;
  stats.mutable_events()->Add(CreateOpstatsEvent(
      opstats::OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED,
      upload_started_time_sec));
  opstats::OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(stats);
  // Trustworthy since epoch time
  opstats_sequence.mutable_earliest_trustworthy_time()->set_seconds(0);

  // Set the clock to epoch + 5 seconds. When we evaluate the swor policy, we'll
  // look back 5 seconds into the past and see our upload started at epoch + 1
  // seconds and thus be ineligible for 5 second swor, even though we are
  // eligible with one second swor.
  clock_.AdvanceTime(absl::Seconds(5));

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, opstats_sequence, clock_,
                         {SetUpExampleIteratorFactory(0).get()});
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 1);
  // Ineligible according to swor.
  ASSERT_EQ(eligibility_result->task_weights().at(0).weight(), 0);
}

TEST_F(EligibilityDeciderTest, DataAvailabilityPolicyIsEligible) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* da_spec =
      spec.mutable_eligibility_policies()->Add();
  da_spec->set_name("da_policy_3_examples");
  da_spec->set_min_version(1);
  da_spec->mutable_data_availability_policy()->set_min_example_count(3);
  *da_spec->mutable_data_availability_policy()
       ->mutable_selector()
       ->mutable_collection_uri() = "app:/padam_padam";

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name(absl::StrCat("single_task_1"));
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(5).get()});
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 1);
  ASSERT_EQ(eligibility_result->task_weights().at(0).weight(), 1.0f);
}

TEST_F(EligibilityDeciderTest, DataAvailabilityPolicyIsNotEligible) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* da_spec =
      spec.mutable_eligibility_policies()->Add();
  da_spec->set_name("da_policy_3_examples");
  da_spec->set_min_version(1);
  da_spec->mutable_data_availability_policy()->set_min_example_count(3);
  *da_spec->mutable_data_availability_policy()
       ->mutable_selector()
       ->mutable_collection_uri() = "app:/padam_padam";

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name(absl::StrCat("single_task_1"));
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(2).get()});
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 1);
  ASSERT_EQ(eligibility_result->task_weights().at(0).weight(), 0.0f);
}

TEST_F(EligibilityDeciderTest, DataAvailabilityPolicyComputationError) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* da_spec =
      spec.mutable_eligibility_policies()->Add();
  da_spec->set_name("da_policy_3_examples");
  da_spec->set_min_version(1);
  da_spec->mutable_data_availability_policy()->set_min_example_count(3);
  *da_spec->mutable_data_availability_policy()
       ->mutable_selector()
       ->mutable_collection_uri() = "app:/padam_padam";

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name(absl::StrCat("single_task_1"));
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  auto mock_iterator = std::make_unique<MockExampleIterator>();
  EXPECT_CALL(*mock_iterator, Next())
      .WillRepeatedly(Return(absl::InternalError("Oh no :(((")));

  auto example_iterator_factory =
      std::make_unique<engine::FunctionalExampleIteratorFactory>(
          [&mock_iterator](
              const google::internal::federated::plan::ExampleSelector&
                  selector) { return std::move(mock_iterator); });

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {example_iterator_factory.get()});
  EXPECT_THAT(eligibility_result.status(), IsCode(absl::StatusCode::kInternal));
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
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(0).get()});
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 0);
}

}  // namespace client
}  // namespace fcp
