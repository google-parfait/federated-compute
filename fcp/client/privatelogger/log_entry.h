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

#ifndef FCP_CLIENT_PRIVATELOGGER_LOG_ENTRY_H_
#define FCP_CLIENT_PRIVATELOGGER_LOG_ENTRY_H_

#include <string>

namespace fcp::client::privatelogger {

// Represents a single entry that is passed for private logging.
struct LogEntry {
  // The serialized value of the entry.
  std::string value;

  // The creation timestamp of the entry, as a string formatted according to
  // ISO-8601 with the device's local time-zone offset.
  // For example: "2016-09-12T08:00:00.123-07:00".
  std::string timestamp;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_LOG_ENTRY_H_
