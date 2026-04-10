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

#ifndef FCP_CLIENT_PRIVATELOGGER_TASK_CONTRIBUTION_TRACKER_H_
#define FCP_CLIENT_PRIVATELOGGER_TASK_CONTRIBUTION_TRACKER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace fcp::client::privatelogger {

// Manages records of entry contributions to tasks, ensuring that the same entry
// is not re-contributed to the same task.
//
// The typical lifecycle of a TaskContributionTracker is as follows:
// 1. GetContributedIds is used to filter already contributed entries from being
//    uploaded.
// 2. CommitContributions is called after the new entries are selected but
//    before the data is uploaded.
// 3. If the upload fails, RollbackContributions is called to undo the committed
//    contributions for the failed upload.
class TaskContributionTracker {
 public:
  virtual ~TaskContributionTracker() = default;

  // Retrieves IDs of entries that have already been committed for the task.
  virtual absl::StatusOr<std::vector<int64_t>> GetContributedIds(
      const std::string& log_source, const std::string& task_name) = 0;

  // Marks the entries as contributed to the given log source and task.
  virtual absl::Status CommitContributions(
      const std::string& log_source, const std::string& task_name,
      const std::vector<int64_t>& entry_ids) = 0;

  // Rolls back committed contributions for the given task and log source.
  virtual absl::Status RollbackContributions(const std::string& log_source,
                                             const std::string& task_name) = 0;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_TASK_CONTRIBUTION_TRACKER_H_
