/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FCP_CLIENT_PRIVATELOGGER_COMMON_LOG_ENTRY_H_
#define FCP_CLIENT_PRIVATELOGGER_COMMON_LOG_ENTRY_H_

#include <string>

#include "absl/time/time.h"

namespace fcp::client::privatelogger {

struct LogEntry {
  absl::Time timestamp;
  std::string value;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_COMMON_LOG_ENTRY_H_
