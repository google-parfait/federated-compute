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

#ifndef FCP_CLIENT_PRIVATELOGGER_PRIVATE_LOGGER_CONNECTOR_H_
#define FCP_CLIENT_PRIVATELOGGER_PRIVATE_LOGGER_CONNECTOR_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "fcp/client/privatelogger/data_provider.h"
#include "fcp/client/privatelogger/upload_outcome.h"

namespace fcp::client::privatelogger {

// Interface for the platform-specific connector that bridges the PrivateLogger
// to the underlying federated compute protocol implementation.
class PrivateLoggerConnector {
 public:
  virtual ~PrivateLoggerConnector() = default;

  // Initiates an upload.
  //
  // @param private_log_source_name The target private log source.
  // @param data_provider A provider that allows fetching data from the logger.
  // @return A vector of outcomes for each task that was executed, or a status
  //     on failure.
  virtual absl::StatusOr<std::vector<UploadOutcome>> Upload(
      const std::string& private_log_source_name,
      DataProvider& data_provider) = 0;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_PRIVATE_LOGGER_CONNECTOR_H_
