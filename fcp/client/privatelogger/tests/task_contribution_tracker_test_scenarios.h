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

#ifndef FCP_CLIENT_PRIVATELOGGER_TESTS_TASK_CONTRIBUTION_TRACKER_TEST_SCENARIOS_H_
#define FCP_CLIENT_PRIVATELOGGER_TESTS_TASK_CONTRIBUTION_TRACKER_TEST_SCENARIOS_H_

#include <concepts>
#include <string>

#include "fcp/client/privatelogger/task_contribution_tracker.h"

namespace fcp::client::privatelogger {

// Names of the scenarios for TaskContributionTracker.
inline constexpr char kScenarioTrackerSuccess[] = "tracker_success";
inline constexpr char kScenarioTrackerRollback[] = "tracker_rollback";
inline constexpr char kScenarioTrackerMultipleTasks[] =
    "tracker_multiple_tasks";
inline constexpr char kScenarioTrackerMultipleLogSources[] =
    "tracker_multiple_log_sources";
inline constexpr char kScenarioTrackerCumulativeCommits[] =
    "tracker_cumulative_commits";
inline constexpr char kScenarioTrackerEmptyCommit[] = "tracker_empty_commit";
inline constexpr char kScenarioTrackerSameIdDifferentTaskRollback[] =
    "tracker_same_id_different_task_rollback";

template <typename T>
concept TaskContributionTrackerImpl =
    std::derived_from<T, TaskContributionTracker>;

template <TaskContributionTrackerImpl T>
class TaskContributionTrackerTestSetup {
 public:
  virtual ~TaskContributionTrackerTestSetup() = default;
  virtual void setup(const std::string& scenario_name, T& tracker) = 0;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_TESTS_TASK_CONTRIBUTION_TRACKER_TEST_SCENARIOS_H_
