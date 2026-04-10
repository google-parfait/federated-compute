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

#ifndef FCP_CLIENT_PRIVATELOGGER_PRIVATE_LOGGER_INTERNAL_H_
#define FCP_CLIENT_PRIVATELOGGER_PRIVATE_LOGGER_INTERNAL_H_

#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace fcp::client::privatelogger {

// Interface for Federated Compute integration to fetch and manage logs
// contributed to an FCP task.
class PrivateLoggerInternal {
 public:
  virtual ~PrivateLoggerInternal() = default;

  // Fetches data for a specific FCP task and marks it as 'pending'
  // contribution. Returns a list of serialized log entries (e.g. as bytes).
  virtual absl::StatusOr<std::vector<std::string>> GetAndCommitData(
      const std::string& task_name) ABSL_LOCKS_EXCLUDED(mutex_) = 0;

  // Rolls back the 'pending' status for contributions made for a specific task.
  // Called if the FCP task fails.
  virtual absl::Status RollbackContribution(const std::string& task_name)
      ABSL_LOCKS_EXCLUDED(mutex_) = 0;

 protected:
  mutable absl::Mutex mutex_;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_PRIVATE_LOGGER_INTERNAL_H_
