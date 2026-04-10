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

#ifndef FCP_CLIENT_PRIVATELOGGER_TESTS_TASK_CONTRIBUTION_TRACKER_MOCKS_H_
#define FCP_CLIENT_PRIVATELOGGER_TESTS_TASK_CONTRIBUTION_TRACKER_MOCKS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/client/privatelogger/task_contribution_tracker.h"

namespace fcp::client::privatelogger {

class MockTaskContributionTracker : public TaskContributionTracker {
 public:
  MOCK_METHOD(absl::Status, CommitContributions,
              (const std::string& log_source, const std::string& task_name,
               const std::vector<int64_t>& entry_ids),
              (override));
  MOCK_METHOD(absl::Status, RollbackContributions,
              (const std::string& log_source, const std::string& task_name),
              (override));
  MOCK_METHOD(absl::StatusOr<std::vector<int64_t>>, GetContributedIds,
              (const std::string& log_source, const std::string& task_name),
              (override));
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_TESTS_TASK_CONTRIBUTION_TRACKER_MOCKS_H_
