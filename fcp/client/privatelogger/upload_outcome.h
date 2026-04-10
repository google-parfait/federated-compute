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

#ifndef FCP_CLIENT_PRIVATELOGGER_UPLOAD_OUTCOME_H_
#define FCP_CLIENT_PRIVATELOGGER_UPLOAD_OUTCOME_H_

#include <string>

namespace fcp::client::privatelogger {

// Represents the outcome of an upload attempt for a single task.
struct UploadOutcome {
  // The status of the upload.
  enum class Status {
    // Data was likely contributed to the server.
    kContributed,
    // The upload failed, and no data conclusively reached the server.
    kNotContributed,
  };

  // The name of the upload task that was executed.
  std::string task_name;

  // The result of the upload attempt.
  Status status;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_UPLOAD_OUTCOME_H_
