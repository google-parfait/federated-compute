// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "fcp/client/privatelogger/tests/mock_task_contribution_tracker_setup.h"
#include "fcp/client/privatelogger/tests/task_contribution_tracker_mocks.h"
#include "fcp/client/privatelogger/tests/task_contribution_tracker_test_scenarios.h"

namespace fcp::client::privatelogger {

using ::absl_testing::IsOk;
using ::testing::UnorderedElementsAreArray;

template <typename Tracker, typename Setup>
struct TrackerTestConfig {
  using TrackerType = Tracker;
  using SetupType = Setup;
};

template <typename Config>
class TaskContributionTrackerTest : public ::testing::Test {
 protected:
  using TrackerType = typename Config::TrackerType;
  using SetupType = typename Config::SetupType;

  std::unique_ptr<TrackerType> tracker_ = std::make_unique<TrackerType>();
  SetupType setup_;
};

using TrackerImplementations =
    ::testing::Types<TrackerTestConfig<MockTaskContributionTracker,
                                       MockTaskContributionTrackerSetup>>;
TYPED_TEST_SUITE(TaskContributionTrackerTest, TrackerImplementations);

TYPED_TEST(TaskContributionTrackerTest, SuccessScenario) {
  const std::string log_source = "source_1";
  const std::string task_name = "task_1";
  const std::vector<int64_t> ids = {1, 2, 3};

  this->setup_.setup(kScenarioTrackerSuccess, *this->tracker_);

  EXPECT_THAT(this->tracker_->CommitContributions(log_source, task_name, ids),
              IsOk());

  auto result = this->tracker_->GetContributedIds(log_source, task_name);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(*result, UnorderedElementsAreArray(ids));
}

TYPED_TEST(TaskContributionTrackerTest, RollbackScenario) {
  const std::string log_source = "source_1";
  const std::string task_name = "task_1";
  const std::vector<int64_t> ids = {1, 2, 3};

  this->setup_.setup(kScenarioTrackerRollback, *this->tracker_);

  EXPECT_THAT(this->tracker_->CommitContributions(log_source, task_name, ids),
              IsOk());

  EXPECT_THAT(this->tracker_->RollbackContributions(log_source, task_name),
              IsOk());

  auto result = this->tracker_->GetContributedIds(log_source, task_name);
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE((*result).empty());
}

TYPED_TEST(TaskContributionTrackerTest, MultipleTasksScenario) {
  const std::string log_source = "source_1";

  this->setup_.setup(kScenarioTrackerMultipleTasks, *this->tracker_);

  EXPECT_THAT(this->tracker_->CommitContributions(log_source, "task_A", {1}),
              IsOk());

  EXPECT_THAT(this->tracker_->CommitContributions(log_source, "task_B", {2}),
              IsOk());

  auto result_a = this->tracker_->GetContributedIds(log_source, "task_A");
  ASSERT_THAT(result_a, IsOk());
  EXPECT_THAT(*result_a, UnorderedElementsAreArray({1}));

  auto result_b = this->tracker_->GetContributedIds(log_source, "task_B");
  ASSERT_THAT(result_b, IsOk());
  EXPECT_THAT(*result_b, UnorderedElementsAreArray({2}));
}

TYPED_TEST(TaskContributionTrackerTest, MultipleLogSourcesScenario) {
  const std::string task_name = "task_1";

  this->setup_.setup(kScenarioTrackerMultipleLogSources, *this->tracker_);

  EXPECT_THAT(this->tracker_->CommitContributions("source_1", task_name, {1}),
              IsOk());

  EXPECT_THAT(this->tracker_->CommitContributions("source_2", task_name, {2}),
              IsOk());

  auto result_1 = this->tracker_->GetContributedIds("source_1", task_name);
  ASSERT_THAT(result_1, IsOk());
  EXPECT_THAT(*result_1, UnorderedElementsAreArray({1}));

  auto result_2 = this->tracker_->GetContributedIds("source_2", task_name);
  ASSERT_THAT(result_2, IsOk());
  EXPECT_THAT(*result_2, UnorderedElementsAreArray({2}));
}

TYPED_TEST(TaskContributionTrackerTest, CumulativeCommitsScenario) {
  const std::string log_source = "source_1";
  const std::string task_name = "task_1";

  this->setup_.setup(kScenarioTrackerCumulativeCommits, *this->tracker_);

  EXPECT_THAT(this->tracker_->CommitContributions(log_source, task_name, {1}),
              IsOk());

  EXPECT_THAT(this->tracker_->CommitContributions(log_source, task_name, {2}),
              IsOk());

  auto result = this->tracker_->GetContributedIds(log_source, task_name);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(*result, UnorderedElementsAreArray({1, 2}));
}

TYPED_TEST(TaskContributionTrackerTest, EmptyCommitScenario) {
  const std::string log_source = "source_1";
  const std::string task_name = "task_1";

  this->setup_.setup(kScenarioTrackerEmptyCommit, *this->tracker_);

  EXPECT_THAT(this->tracker_->CommitContributions(log_source, task_name, {}),
              IsOk());

  auto result = this->tracker_->GetContributedIds(log_source, task_name);
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE((*result).empty());
}

TYPED_TEST(TaskContributionTrackerTest, SameIdDifferentTaskRollback) {
  const std::string log_source = "source_1";

  this->setup_.setup(kScenarioTrackerSameIdDifferentTaskRollback,
                     *this->tracker_);

  EXPECT_THAT(this->tracker_->CommitContributions(log_source, "task_A", {1}),
              IsOk());

  EXPECT_THAT(this->tracker_->CommitContributions(log_source, "task_B", {1}),
              IsOk());

  EXPECT_THAT(this->tracker_->RollbackContributions(log_source, "task_A"),
              IsOk());

  auto result_a = this->tracker_->GetContributedIds(log_source, "task_A");
  ASSERT_THAT(result_a, IsOk());
  EXPECT_TRUE((*result_a).empty());

  auto result_b = this->tracker_->GetContributedIds(log_source, "task_B");
  ASSERT_THAT(result_b, IsOk());
  EXPECT_THAT(*result_b, UnorderedElementsAreArray({1}));
}

}  // namespace fcp::client::privatelogger
