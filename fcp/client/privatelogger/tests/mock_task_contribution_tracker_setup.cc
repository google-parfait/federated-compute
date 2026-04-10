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

#include "fcp/client/privatelogger/tests/mock_task_contribution_tracker_setup.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "fcp/client/privatelogger/tests/task_contribution_tracker_mocks.h"
#include "fcp/client/privatelogger/tests/task_contribution_tracker_test_scenarios.h"

namespace fcp::client::privatelogger {

using ::testing::Return;

MockTaskContributionTrackerSetup::MockTaskContributionTrackerSetup() {
  scenario_setups_.emplace(
      kScenarioTrackerSuccess,
      [this](MockTaskContributionTracker& tracker) { SetupSuccess(tracker); });
  scenario_setups_.emplace(
      kScenarioTrackerRollback,
      [this](MockTaskContributionTracker& tracker) { SetupRollback(tracker); });
  scenario_setups_.emplace(kScenarioTrackerMultipleTasks,
                           [this](MockTaskContributionTracker& tracker) {
                             SetupMultipleTasks(tracker);
                           });
  scenario_setups_.emplace(kScenarioTrackerMultipleLogSources,
                           [this](MockTaskContributionTracker& tracker) {
                             SetupMultipleLogSources(tracker);
                           });
  scenario_setups_.emplace(kScenarioTrackerCumulativeCommits,
                           [this](MockTaskContributionTracker& tracker) {
                             SetupCumulativeCommits(tracker);
                           });
  scenario_setups_.emplace(kScenarioTrackerEmptyCommit,
                           [this](MockTaskContributionTracker& tracker) {
                             SetupEmptyCommit(tracker);
                           });
  scenario_setups_.emplace(kScenarioTrackerSameIdDifferentTaskRollback,
                           [this](MockTaskContributionTracker& tracker) {
                             SetupSameIdDifferentTaskRollback(tracker);
                           });
}

void MockTaskContributionTrackerSetup::setup(
    const std::string& scenario_name, MockTaskContributionTracker& tracker) {
  auto it = scenario_setups_.find(scenario_name);
  if (it != scenario_setups_.end()) {
    // Found: call the setup function
    it->second(tracker);
  }
  // Not found: NO-OP.
}

void MockTaskContributionTrackerSetup::SetupSuccess(
    MockTaskContributionTracker& tracker) {
  EXPECT_CALL(tracker, CommitContributions("source_1", "task_1",
                                           std::vector<int64_t>{1, 2, 3}))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, GetContributedIds("source_1", "task_1"))
      .WillOnce(Return(std::vector<int64_t>{1, 2, 3}));
}

void MockTaskContributionTrackerSetup::SetupRollback(
    MockTaskContributionTracker& tracker) {
  EXPECT_CALL(tracker, CommitContributions("source_1", "task_1",
                                           std::vector<int64_t>{1, 2, 3}))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, RollbackContributions("source_1", "task_1"))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, GetContributedIds("source_1", "task_1"))
      .WillOnce(Return(std::vector<int64_t>{}));
}

void MockTaskContributionTrackerSetup::SetupMultipleTasks(
    MockTaskContributionTracker& tracker) {
  EXPECT_CALL(tracker, CommitContributions("source_1", "task_A",
                                           std::vector<int64_t>{1}))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, CommitContributions("source_1", "task_B",
                                           std::vector<int64_t>{2}))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, GetContributedIds("source_1", "task_A"))
      .WillOnce(Return(std::vector<int64_t>{1}));
  EXPECT_CALL(tracker, GetContributedIds("source_1", "task_B"))
      .WillOnce(Return(std::vector<int64_t>{2}));
}

void MockTaskContributionTrackerSetup::SetupMultipleLogSources(
    MockTaskContributionTracker& tracker) {
  EXPECT_CALL(tracker, CommitContributions("source_1", "task_1",
                                           std::vector<int64_t>{1}))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, CommitContributions("source_2", "task_1",
                                           std::vector<int64_t>{2}))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, GetContributedIds("source_1", "task_1"))
      .WillOnce(Return(std::vector<int64_t>{1}));
  EXPECT_CALL(tracker, GetContributedIds("source_2", "task_1"))
      .WillOnce(Return(std::vector<int64_t>{2}));
}

void MockTaskContributionTrackerSetup::SetupCumulativeCommits(
    MockTaskContributionTracker& tracker) {
  EXPECT_CALL(tracker, CommitContributions("source_1", "task_1",
                                           std::vector<int64_t>{1}))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, CommitContributions("source_1", "task_1",
                                           std::vector<int64_t>{2}))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, GetContributedIds("source_1", "task_1"))
      .WillOnce(Return(std::vector<int64_t>{1, 2}));
}

void MockTaskContributionTrackerSetup::SetupEmptyCommit(
    MockTaskContributionTracker& tracker) {
  EXPECT_CALL(tracker,
              CommitContributions("source_1", "task_1", std::vector<int64_t>{}))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, GetContributedIds("source_1", "task_1"))
      .WillOnce(Return(std::vector<int64_t>{}));
}

void MockTaskContributionTrackerSetup::SetupSameIdDifferentTaskRollback(
    MockTaskContributionTracker& tracker) {
  EXPECT_CALL(tracker, CommitContributions("source_1", "task_A",
                                           std::vector<int64_t>{1}))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, CommitContributions("source_1", "task_B",
                                           std::vector<int64_t>{1}))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(tracker, RollbackContributions("source_1", "task_A"))
      .WillOnce(Return(absl::OkStatus()));

  EXPECT_CALL(tracker, GetContributedIds("source_1", "task_A"))
      .WillOnce(Return(std::vector<int64_t>{}));
  EXPECT_CALL(tracker, GetContributedIds("source_1", "task_B"))
      .WillOnce(Return(std::vector<int64_t>{1}));
}

}  // namespace fcp::client::privatelogger
