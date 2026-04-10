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

#ifndef FCP_CLIENT_PRIVATELOGGER_TESTS_MOCK_TASK_CONTRIBUTION_TRACKER_SETUP_H_
#define FCP_CLIENT_PRIVATELOGGER_TESTS_MOCK_TASK_CONTRIBUTION_TRACKER_SETUP_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "fcp/client/privatelogger/tests/task_contribution_tracker_mocks.h"
#include "fcp/client/privatelogger/tests/task_contribution_tracker_test_scenarios.h"

namespace fcp::client::privatelogger {

class MockTaskContributionTrackerSetup
    : public TaskContributionTrackerTestSetup<MockTaskContributionTracker> {
 public:
  MockTaskContributionTrackerSetup();
  void setup(const std::string& scenario_name,
             MockTaskContributionTracker& tracker) override;

 private:
  void SetupSuccess(MockTaskContributionTracker& tracker);
  void SetupRollback(MockTaskContributionTracker& tracker);
  void SetupMultipleTasks(MockTaskContributionTracker& tracker);
  void SetupMultipleLogSources(MockTaskContributionTracker& tracker);
  void SetupCumulativeCommits(MockTaskContributionTracker& tracker);
  void SetupEmptyCommit(MockTaskContributionTracker& tracker);
  void SetupSameIdDifferentTaskRollback(MockTaskContributionTracker& tracker);

  absl::flat_hash_map<
      std::string, absl::AnyInvocable<void(MockTaskContributionTracker&) const>>
      scenario_setups_;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_TESTS_MOCK_TASK_CONTRIBUTION_TRACKER_SETUP_H_
