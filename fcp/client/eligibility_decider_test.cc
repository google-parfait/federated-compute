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
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "google/protobuf/timestamp.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "fcp/base/simulated_clock.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/engine/common.h"
#include "fcp/client/engine/example_iterator_factory.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/population_eligibility_spec.pb.h"
#include "fcp/testing/testing.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace fcp {
namespace client {

using ::google::internal::federated::plan::EligibilityPolicyEvalSpec;
using ::google::internal::federated::plan::ExampleSelector;
using ::google::internal::federated::plan::PopulationEligibilitySpec;
using ::testing::_;
using ::testing::DoAll;
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

class MockEetPlanRunner : public EetPlanRunner {
 public:
  MOCK_METHOD(
      engine::PlanResult, RunPlan,
      (std::vector<engine::ExampleIteratorFactory*> example_iterator_factories),
      (override));
  MOCK_METHOD(absl::StatusOr<TaskEligibilityInfo>, ParseOutput,
              (const std::vector<tensorflow::Tensor>& output_tensors),
              (override));
};

class EligibilityDeciderTest : public testing::Test {
 protected:
  NiceMock<MockLogManager> mock_log_manager_;
  SimulatedClock clock_;
  std::vector<engine::ExampleIteratorFactory*> example_iterator_factories_;
  NiceMock<MockEetPlanRunner> mock_eet_plan_runner_;
  MockFlags mock_flags_;
};

opstats::OpStatsSequence GenOpstatsSequence() { return {}; }

TEST_F(EligibilityDeciderTest, NoPoliciesEligibleForAllTasks) {
  int num_tasks = 4;
  absl::StatusOr<TaskEligibilityInfo> eligibility_result = ComputeEligibility(
      GenNoPoliciesSpec(num_tasks), mock_log_manager_, GenOpstatsSequence(),
      clock_, {SetUpExampleIteratorFactory(0).get()}, false,
      mock_eet_plan_runner_, &mock_flags_);

  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), num_tasks);

  for (const auto& task_weight : eligibility_result->task_weights()) {
    ASSERT_EQ(task_weight.weight(), 1.0f);
  }
}

TEST_F(EligibilityDeciderTest, NoPolicyTypeLogsError) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* unset_spec =
      spec.mutable_eligibility_policies()->Add();
  unset_spec->set_name("I_have_no_policy_kind_case");
  unset_spec->set_min_version(1);

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name("single_task_1");
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  EXPECT_CALL(mock_log_manager_,
              LogDiag(ProdDiagCode::ELIGIBILITY_EVAL_UNEXPECTED_POLICY_KIND));

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(0).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
  ASSERT_OK(eligibility_result);
  ASSERT_TRUE(eligibility_result->task_weights().empty());
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
  task_info->set_task_name("single_task_1");
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
                         {SetUpExampleIteratorFactory(0).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
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
  task_info->set_task_name(task_name);
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
                         {SetUpExampleIteratorFactory(0).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 1);
  // Ineligible according to swor.
  ASSERT_EQ(eligibility_result->task_weights().at(0).weight(), 0);
}

TEST_F(EligibilityDeciderTest, GroupSworPolicyIsEligible) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* swor_spec =
      spec.mutable_eligibility_policies()->Add();
  swor_spec->set_name("group_swor_policy_5_seconds");
  swor_spec->set_min_version(1);
  swor_spec->mutable_swor_policy()->mutable_min_period()->set_seconds(5);
  swor_spec->mutable_swor_policy()->set_group_regex("(.*\b)padam_padam(\b.*)");

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name("single_task_1.padam_padam");
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
                         {SetUpExampleIteratorFactory(0).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 1);
  // Eligible according to group swor.
  ASSERT_EQ(eligibility_result->task_weights().at(0).weight(), 1.0f);
}

TEST_F(EligibilityDeciderTest, GroupSworPolicyIsNotEligible) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* swor_spec =
      spec.mutable_eligibility_policies()->Add();
  swor_spec->set_name("group_swor_policy_5_seconds");
  swor_spec->set_min_version(1);
  swor_spec->mutable_swor_policy()->mutable_min_period()->set_seconds(5);
  swor_spec->mutable_swor_policy()->set_group_regex("(.*\\b)group_swor(\\b.*)");

  std::string task_name = "single_task_1.group_swor";
  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name(task_name);
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  opstats::OperationalStats stats;
  stats.set_task_name("my_other_task_in_the_same_group.group_swor");
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
                         {SetUpExampleIteratorFactory(0).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
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
  task_info->set_task_name(task_name);
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
                         {SetUpExampleIteratorFactory(0).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
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
  task_info->set_task_name("single_task_1");
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(5).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
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
  task_info->set_task_name("single_task_1");
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(2).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
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
  task_info->set_task_name("single_task_1");
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
                         {example_iterator_factory.get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
  EXPECT_THAT(eligibility_result.status(), IsCode(absl::StatusCode::kInternal));
}

TEST_F(EligibilityDeciderTest, TfCustomPolicyReturnsEmptyTaskEligibilityInfo) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* tf_spec =
      spec.mutable_eligibility_policies()->Add();
  tf_spec->set_name("tf_custom_policy");
  tf_spec->set_min_version(1);

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name("single_task_1");
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  // Result should be ok, but because TF custom policies are unimplemented, we
  // get an empty TaskEligibilityInfo.
  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(0).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 0);
}

TEST_F(EligibilityDeciderTest, TfCustomPolicyEnabledRunsSuccessfully) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* tf_spec =
      spec.mutable_eligibility_policies()->Add();
  tf_spec->set_name("tf_custom_policy");
  tf_spec->set_min_version(1);
  *tf_spec->mutable_tf_custom_policy()->mutable_arguments() =
      "hi hello how are you";

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name("single_task_1");
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  engine::PlanResult plan_result(engine::PlanOutcome::kSuccess,
                                 absl::OkStatus());
  plan_result.output_tensors = {};

  EXPECT_CALL(mock_eet_plan_runner_, RunPlan(_))
      .WillOnce(Return(std::move(plan_result)));

  TaskEligibilityInfo tf_custom_policy_output;
  tf_custom_policy_output.set_version(1);
  auto* task_weight = tf_custom_policy_output.add_task_weights();
  task_weight->set_task_name("single_task_1");
  task_weight->set_weight(1.0f);

  EXPECT_CALL(mock_eet_plan_runner_, ParseOutput(_))
      .WillOnce(Return(tf_custom_policy_output));

  // Result should match the response of our TfCustomPolicy output since we have
  // no other policies.
  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(0).get()}, true,
                         mock_eet_plan_runner_, &mock_flags_);
  ASSERT_OK(eligibility_result);
  EXPECT_THAT(*eligibility_result, EqualsProto(tf_custom_policy_output));
}

TEST_F(EligibilityDeciderTest, TfCustomPolicyPreparesNeetContextIterator) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* tf_spec =
      spec.mutable_eligibility_policies()->Add();
  std::string policy_name = "tf_custom_policy";
  tf_spec->set_name(policy_name);
  tf_spec->set_min_version(1);
  *tf_spec->mutable_tf_custom_policy()->mutable_arguments() =
      "hi hello how are you";

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  std::string task_name = "single_task_1";
  task_info->set_task_name(task_name);
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  engine::PlanResult plan_result(engine::PlanOutcome::kSuccess,
                                 absl::OkStatus());
  plan_result.output_tensors = {};

  // We do some asserts inside of our mock RunPlan, as we will not be able to
  // capture these variables and test them outside of the scope of
  // ComputeEligibility.
  EXPECT_CALL(mock_eet_plan_runner_, RunPlan(_))
      .WillOnce(DoAll(
          [&](std::vector<engine::ExampleIteratorFactory*> factory_pointers) {
            // iterator_factories should contain both our base iterator and our
            // neet context iterator. They should also be in order of growing
            // scope, as the implementation will scan through them from
            // beginning to end. The neet context iterator factory should be
            // first, as it specifically handles the neet context uri, then our
            // generic iterator factory should be second, as it handles all
            // possible uris.

            ExampleSelector neet_selector;
            *neet_selector.mutable_collection_uri() =
                "internal:/eligibility_context";
            ExampleSelector my_examples_selector;
            *my_examples_selector.mutable_collection_uri() = "app:/rain_on_me";

            ASSERT_EQ(factory_pointers.size(), 2);
            auto eligibility_context_factory = factory_pointers[0];
            ASSERT_TRUE(eligibility_context_factory->CanHandle(neet_selector));
            ASSERT_FALSE(
                eligibility_context_factory->CanHandle(my_examples_selector));
            ASSERT_TRUE(factory_pointers[1]->CanHandle(my_examples_selector));

            // Get all examples from the neet context iterator
            std::vector<std::string> eligibility_context_example;
            absl::StatusOr<std::unique_ptr<ExampleIterator>>
                eligibility_context_iterator =
                    eligibility_context_factory->CreateExampleIterator(
                        neet_selector);
            ASSERT_OK(eligibility_context_iterator);
            absl::StatusOr<std::string> next_example =
                (*eligibility_context_iterator)->Next();
            while (next_example.ok()) {
              eligibility_context_example.push_back(*next_example);
              next_example = (*eligibility_context_iterator)->Next();
            }
            ASSERT_EQ(eligibility_context_example.size(), 1);
            tensorflow::Example example;
            ASSERT_TRUE(
                example.ParseFromString(eligibility_context_example[0]));
            ASSERT_EQ(example.features()
                          .feature()
                          .at("policy_name")
                          .bytes_list()
                          .value(0),
                      policy_name);
            ASSERT_EQ(example.features()
                          .feature()
                          .at("task_names")
                          .bytes_list()
                          .value_size(),
                      1);
            ASSERT_EQ(example.features()
                          .feature()
                          .at("task_names")
                          .bytes_list()
                          .value(0),
                      task_name);
          },
          Return(std::move(plan_result))));

  TaskEligibilityInfo tf_custom_policy_output;
  tf_custom_policy_output.set_version(1);
  auto* task_weight = tf_custom_policy_output.add_task_weights();
  task_weight->set_task_name("single_task_1");
  task_weight->set_weight(1.0f);

  EXPECT_CALL(mock_eet_plan_runner_, ParseOutput(_))
      .WillOnce(Return(tf_custom_policy_output));

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(0).get()}, true,
                         mock_eet_plan_runner_, &mock_flags_);

  ASSERT_OK(eligibility_result);
}

TEST_F(EligibilityDeciderTest, TfCustomPolicyEnabledPlanOutcomeFailure) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* tf_spec =
      spec.mutable_eligibility_policies()->Add();
  tf_spec->set_name("tf_custom_policy");
  tf_spec->set_min_version(1);
  *tf_spec->mutable_tf_custom_policy()->mutable_arguments() =
      "hi hello how are you";

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name("single_task_1");
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  auto execution_error = absl::InternalError("oh no!!");
  engine::PlanResult plan_result(engine::PlanOutcome::kTensorflowError,
                                 execution_error);

  EXPECT_CALL(mock_eet_plan_runner_, RunPlan(_))
      .WillOnce(Return(std::move(plan_result)));

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(0).get()}, true,
                         mock_eet_plan_runner_, &mock_flags_);
  EXPECT_THAT(eligibility_result.status(), IsCode(absl::StatusCode::kInternal));
}

TEST_F(EligibilityDeciderTest, TfCustomPolicyEnabledParseOutputsFailure) {
  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* tf_spec =
      spec.mutable_eligibility_policies()->Add();
  tf_spec->set_name("tf_custom_policy");
  tf_spec->set_min_version(1);
  *tf_spec->mutable_tf_custom_policy()->mutable_arguments() =
      "hi hello how are you";

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name("single_task_1");
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  engine::PlanResult plan_result(engine::PlanOutcome::kSuccess,
                                 absl::OkStatus());
  plan_result.output_tensors = {};

  EXPECT_CALL(mock_eet_plan_runner_, RunPlan(_))
      .WillOnce(Return(std::move(plan_result)));

  EXPECT_CALL(mock_eet_plan_runner_, ParseOutput(_))
      .WillOnce(Return(absl::InternalError("cripes!")));

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(0).get()}, true,
                         mock_eet_plan_runner_, &mock_flags_);
  EXPECT_THAT(eligibility_result.status(), IsCode(absl::StatusCode::kInternal));
}

TEST_F(EligibilityDeciderTest, EligibleForAllPolicyTypes) {
  PopulationEligibilitySpec spec;

  std::string task_name = "single_task_1";

  EligibilityPolicyEvalSpec* swor_spec =
      spec.mutable_eligibility_policies()->Add();
  swor_spec->set_name("swor_policy_5_seconds");
  swor_spec->set_min_version(1);
  swor_spec->mutable_swor_policy()->mutable_min_period()->set_seconds(5);

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

  // Set the clock to epoch + 10 seconds. When we evaluate the swor policy,
  // we'll look back 5 seconds into the past and see our upload started at epoch
  // + 1 seconds, and thus be eligible.
  clock_.AdvanceTime(absl::Seconds(10));

  EligibilityPolicyEvalSpec* da_spec =
      spec.mutable_eligibility_policies()->Add();
  da_spec->set_name("da_policy_3_examples");
  da_spec->set_min_version(1);
  da_spec->mutable_data_availability_policy()->set_min_example_count(3);
  *da_spec->mutable_data_availability_policy()
       ->mutable_selector()
       ->mutable_collection_uri() = "app:/bad_idea_right";

  EligibilityPolicyEvalSpec* tf_spec =
      spec.mutable_eligibility_policies()->Add();
  tf_spec->set_name("tf_custom_policy");
  tf_spec->set_min_version(1);
  *tf_spec->mutable_tf_custom_policy()->mutable_arguments() =
      "hi hello how are you";

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name(task_name);
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);
  task_info->mutable_eligibility_policy_indices()->Add(1);
  task_info->mutable_eligibility_policy_indices()->Add(2);

  engine::PlanResult plan_result(engine::PlanOutcome::kSuccess,
                                 absl::OkStatus());
  plan_result.output_tensors = {};

  EXPECT_CALL(mock_eet_plan_runner_, RunPlan(_))
      .WillOnce(Return(std::move(plan_result)));

  TaskEligibilityInfo tf_custom_policy_output;
  tf_custom_policy_output.set_version(1);
  auto* task_weight = tf_custom_policy_output.add_task_weights();
  task_weight->set_task_name(task_name);
  task_weight->set_weight(1.0f);

  EXPECT_CALL(mock_eet_plan_runner_, ParseOutput(_))
      .WillOnce(Return(tf_custom_policy_output));
  // Result should match the response of our TfCustomPolicy output since we have
  // a single task eligible for all policies.
  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(5).get()}, true,
                         mock_eet_plan_runner_, &mock_flags_);
  ASSERT_OK(eligibility_result);
  EXPECT_THAT(*eligibility_result, EqualsProto(tf_custom_policy_output));
}

TEST_F(EligibilityDeciderTest, TwoTasksOneEligibleForAllOneNot) {
  PopulationEligibilitySpec spec;

  std::string task_name = "single_task_1";
  std::string task_name2 = "single_task_2";

  EligibilityPolicyEvalSpec* swor_spec =
      spec.mutable_eligibility_policies()->Add();
  swor_spec->set_name("swor_policy_5_seconds");
  swor_spec->set_min_version(1);
  swor_spec->mutable_swor_policy()->mutable_min_period()->set_seconds(5);

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

  // Set the clock to epoch + 10 seconds. When we evaluate the swor policy,
  // we'll look back 5 seconds into the past and see our upload started at epoch
  // + 1 seconds, and thus be eligible.
  clock_.AdvanceTime(absl::Seconds(10));

  EligibilityPolicyEvalSpec* da_spec =
      spec.mutable_eligibility_policies()->Add();
  da_spec->set_name("da_policy_3_examples");
  da_spec->set_min_version(1);
  da_spec->mutable_data_availability_policy()->set_min_example_count(3);
  *da_spec->mutable_data_availability_policy()
       ->mutable_selector()
       ->mutable_collection_uri() = "app:/super_shy";

  EligibilityPolicyEvalSpec* tf_spec =
      spec.mutable_eligibility_policies()->Add();
  tf_spec->set_name("tf_custom_policy");
  tf_spec->set_min_version(1);
  *tf_spec->mutable_tf_custom_policy()->mutable_arguments() =
      "hi hello how are you";

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name(task_name);
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);
  task_info->mutable_eligibility_policy_indices()->Add(1);
  task_info->mutable_eligibility_policy_indices()->Add(2);

  PopulationEligibilitySpec::TaskInfo* task_info2 =
      spec.mutable_task_info()->Add();
  task_info2->set_task_name(task_name2);
  task_info2->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_MULTIPLE);
  task_info2->mutable_eligibility_policy_indices()->Add(0);
  task_info2->mutable_eligibility_policy_indices()->Add(1);
  task_info2->mutable_eligibility_policy_indices()->Add(2);

  engine::PlanResult plan_result(engine::PlanOutcome::kSuccess,
                                 absl::OkStatus());
  plan_result.output_tensors = {};

  EXPECT_CALL(mock_eet_plan_runner_, RunPlan(_))
      .WillOnce(Return(std::move(plan_result)));

  TaskEligibilityInfo tf_custom_policy_output;
  tf_custom_policy_output.set_version(1);
  auto* task_weight = tf_custom_policy_output.add_task_weights();
  task_weight->set_task_name(task_name);
  task_weight->set_weight(1.0f);
  auto* task_weight2 = tf_custom_policy_output.add_task_weights();
  task_weight2->set_task_name(task_name2);
  task_weight2->set_weight(0.0f);

  EXPECT_CALL(mock_eet_plan_runner_, ParseOutput(_))
      .WillOnce(Return(tf_custom_policy_output));
  // Result should match the response of our TfCustomPolicy output since we have
  // a single task eligible for all policies.
  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, GenOpstatsSequence(), clock_,
                         {SetUpExampleIteratorFactory(5).get()}, true,
                         mock_eet_plan_runner_, &mock_flags_);
  ASSERT_OK(eligibility_result);
  EXPECT_THAT(*eligibility_result, EqualsProto(tf_custom_policy_output));
}

TEST_F(EligibilityDeciderTest, MinSepPolicyEnabledTaskNotExecutedIsEligible) {
  EXPECT_CALL(mock_flags_, enable_minimum_separation_policy())
      .WillRepeatedly(Return(true));

  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* min_sep_spec =
      spec.mutable_eligibility_policies()->Add();
  min_sep_spec->set_name("min_sep_policy_3_rounds");
  min_sep_spec->set_min_version(1);
  min_sep_spec->mutable_min_sep_policy()->set_current_index(5);
  min_sep_spec->mutable_min_sep_policy()->set_minimum_separation(3);

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name("single_task_1");
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_SINGLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  opstats::OpStatsSequence opstats_sequence;

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, opstats_sequence, clock_,
                         {SetUpExampleIteratorFactory(0).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 1);
  ASSERT_EQ(eligibility_result->task_weights().at(0).weight(), 1.0f);
}

TEST_F(EligibilityDeciderTest,
       MinSepPolicyEnabledIndexSepGreaterThanOrEqualToMinSepIsEligible) {
  EXPECT_CALL(mock_flags_, enable_minimum_separation_policy())
      .WillRepeatedly(Return(true));

  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* min_sep_spec =
      spec.mutable_eligibility_policies()->Add();
  min_sep_spec->set_name("min_sep_policy_3_rounds");
  min_sep_spec->set_min_version(1);
  min_sep_spec->mutable_min_sep_policy()->set_current_index(5);
  min_sep_spec->mutable_min_sep_policy()->set_minimum_separation(3);

  // The task's current index separation equals to the minimum separation.
  PopulationEligibilitySpec::TaskInfo* task_info1 =
      spec.mutable_task_info()->Add();
  task_info1->set_task_name("single_task_1");
  task_info1->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_SINGLE);
  task_info1->mutable_eligibility_policy_indices()->Add(0);

  // The task's current index separation is greater than the minimum separation.
  PopulationEligibilitySpec::TaskInfo* task_info2 =
      spec.mutable_task_info()->Add();
  task_info2->set_task_name("single_task_2");
  task_info2->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_SINGLE);
  task_info2->mutable_eligibility_policy_indices()->Add(0);

  opstats::OpStatsSequence opstats_sequence;

  opstats::OperationalStats stats1;
  stats1.set_task_name(task_info1->task_name());
  stats1.mutable_min_sep_policy_current_index()->insert(
      {min_sep_spec->name(), 2});
  stats1.mutable_events()->Add(CreateOpstatsEvent(
      opstats::OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED, 1));
  *opstats_sequence.add_opstats() = std::move(stats1);

  opstats::OperationalStats stats2;
  stats2.set_task_name(task_info2->task_name());
  stats2.mutable_min_sep_policy_current_index()->insert(
      {min_sep_spec->name(), 1});
  stats2.mutable_events()->Add(CreateOpstatsEvent(
      opstats::OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED, 1));
  *opstats_sequence.add_opstats() = std::move(stats2);

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, opstats_sequence, clock_,
                         {SetUpExampleIteratorFactory(0).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 2);
  ASSERT_EQ(eligibility_result->task_weights().at(0).weight(), 1.0f);
  ASSERT_EQ(eligibility_result->task_weights().at(1).weight(), 1.0f);
}

TEST_F(EligibilityDeciderTest,
       MinSepPolicyEnabledIndexSepLessThanMinSepIsNotEligible) {
  EXPECT_CALL(mock_flags_, enable_minimum_separation_policy())
      .WillRepeatedly(Return(true));

  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* min_sep_spec =
      spec.mutable_eligibility_policies()->Add();
  min_sep_spec->set_name("min_swor_policy_3_rounds");
  min_sep_spec->set_min_version(1);
  min_sep_spec->mutable_min_sep_policy()->set_current_index(5);
  min_sep_spec->mutable_min_sep_policy()->set_minimum_separation(3);

  PopulationEligibilitySpec::TaskInfo* task_info =
      spec.mutable_task_info()->Add();
  task_info->set_task_name("single_task_1");
  task_info->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_SINGLE);
  task_info->mutable_eligibility_policy_indices()->Add(0);

  opstats::OpStatsSequence opstats_sequence;
  opstats::OperationalStats stats;
  stats.set_task_name(task_info->task_name());
  stats.mutable_min_sep_policy_current_index()->insert(
      {min_sep_spec->name(), 3});
  stats.mutable_events()->Add(CreateOpstatsEvent(
      opstats::OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED,
      1000));
  *opstats_sequence.add_opstats() = std::move(stats);

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, opstats_sequence, clock_,
                         {SetUpExampleIteratorFactory(0).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 1);
  ASSERT_EQ(eligibility_result->task_weights().at(0).weight(), 0.0f);
}

TEST_F(EligibilityDeciderTest, MinSepPolicyDisabledIsAlwaysNotEligible) {
  EXPECT_CALL(mock_flags_, enable_minimum_separation_policy())
      .WillRepeatedly(Return(false));

  PopulationEligibilitySpec spec;

  EligibilityPolicyEvalSpec* min_sep_spec =
      spec.mutable_eligibility_policies()->Add();
  min_sep_spec->set_name("min_swor_policy_3_rounds");
  min_sep_spec->set_min_version(1);
  min_sep_spec->mutable_min_sep_policy()->set_current_index(5);
  min_sep_spec->mutable_min_sep_policy()->set_minimum_separation(3);

  // The task's current index separation equals to the minimum separation.
  PopulationEligibilitySpec::TaskInfo* task_info1 =
      spec.mutable_task_info()->Add();
  task_info1->set_task_name("single_task_1");
  task_info1->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_SINGLE);
  task_info1->mutable_eligibility_policy_indices()->Add(0);

  // The task's current index separation is greater than the minimum separation.
  PopulationEligibilitySpec::TaskInfo* task_info2 =
      spec.mutable_task_info()->Add();
  task_info2->set_task_name("single_task_2");
  task_info2->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_SINGLE);
  task_info2->mutable_eligibility_policy_indices()->Add(0);

  // The task's current index separation is less than the minimum separation.
  PopulationEligibilitySpec::TaskInfo* task_info3 =
      spec.mutable_task_info()->Add();
  task_info3->set_task_name("single_task_3");
  task_info3->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_SINGLE);
  task_info3->mutable_eligibility_policy_indices()->Add(0);

  // The task was never executed.
  PopulationEligibilitySpec::TaskInfo* task_info4 =
      spec.mutable_task_info()->Add();
  task_info4->set_task_name("single_task_4");
  task_info4->set_task_assignment_mode(
      PopulationEligibilitySpec::TaskInfo::TASK_ASSIGNMENT_MODE_SINGLE);
  task_info4->mutable_eligibility_policy_indices()->Add(0);

  opstats::OpStatsSequence opstats_sequence;

  opstats::OperationalStats stats1;
  stats1.set_task_name(task_info1->task_name());
  stats1.mutable_min_sep_policy_current_index()->insert(
      {min_sep_spec->name(), 2});
  stats1.mutable_events()->Add(CreateOpstatsEvent(
      opstats::OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED, 1));
  *opstats_sequence.add_opstats() = std::move(stats1);

  opstats::OperationalStats stats2;
  stats2.set_task_name(task_info2->task_name());
  stats2.mutable_min_sep_policy_current_index()->insert(
      {min_sep_spec->name(), 1});
  stats2.mutable_events()->Add(CreateOpstatsEvent(
      opstats::OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED, 1));
  *opstats_sequence.add_opstats() = std::move(stats2);

  opstats::OperationalStats stats3;
  stats3.set_task_name(task_info3->task_name());
  stats3.mutable_min_sep_policy_current_index()->insert(
      {min_sep_spec->name(), 3});
  stats3.mutable_events()->Add(CreateOpstatsEvent(
      opstats::OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED, 1));
  *opstats_sequence.add_opstats() = std::move(stats3);

  absl::StatusOr<TaskEligibilityInfo> eligibility_result =
      ComputeEligibility(spec, mock_log_manager_, opstats_sequence, clock_,
                         {SetUpExampleIteratorFactory(0).get()}, false,
                         mock_eet_plan_runner_, &mock_flags_);
  ASSERT_OK(eligibility_result);
  ASSERT_EQ(eligibility_result->task_weights_size(), 4);
  // All tasks should be not eligible as the minimum separation policy is
  // disabled.
  for (const auto& task_weight : eligibility_result->task_weights()) {
    ASSERT_EQ(task_weight.weight(), 0.0f);
  }
}

}  // namespace client
}  // namespace fcp
