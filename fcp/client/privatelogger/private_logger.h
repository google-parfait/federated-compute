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

#ifndef FCP_CLIENT_PRIVATELOGGER_PRIVATE_LOGGER_H_
#define FCP_CLIENT_PRIVATELOGGER_PRIVATE_LOGGER_H_

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "fcp/client/privatelogger/private_logger_internal.h"
#include "google/protobuf/message_lite.h"

namespace fcp::client::privatelogger {

// Interface for logging data to an on-device store and then uploading it to a
// private log source via Federated Compute.
// This class is thread-safe.
class PrivateLogger : public PrivateLoggerInternal {
 public:
  ~PrivateLogger() override = default;

  // Logs a new entry (as a proto message) to the on-device store.
  // This operation is typically performed asynchronously.
  virtual absl::Status Log(const google::protobuf::MessageLite& event)
      ABSL_LOCKS_EXCLUDED(mutex_) = 0;

  // Triggers an upload of the locally stored data to the private log source.
  // This operation is blocking.
  virtual absl::Status Upload() ABSL_LOCKS_EXCLUDED(mutex_) = 0;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_PRIVATE_LOGGER_H_
