/*
 * Copyright 2017 Google LLC
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

#include "fcp/base/monitoring.h"

#include <string>

#include "fcp/base/base_name.h"

namespace fcp {

namespace internal {

#ifdef FCP_BAREMETAL
// Provides safe static initialization of the default global logger instance.
// TODO(team): Improve the logger registration mechanism.
Logger*& GetGlobalLogger() {
  static Logger* global_logger = new Logger();
  return global_logger;
}

Logger* logger() { return GetGlobalLogger(); }
void set_logger(Logger* logger) { GetGlobalLogger() = logger; }

void Logger::Log(const char* file, int line, LogSeverity severity,
                 const char* message) {
  // By default we just ignore all logs. Another Logger can be registered to do
  // something with them, however.
}
#endif  // FCP_BAREMETAL

StatusBuilder::StatusBuilder(StatusCode code, const char* file, int line)
    : file_(file), line_(line), code_(code), message_() {}

StatusBuilder::StatusBuilder(StatusBuilder const& other)
    : file_(other.file_),
      line_(other.line_),
      code_(other.code_),
      message_(other.message_.str()) {}

StatusBuilder::operator Status() {
  auto message_str = message_.str();
  if (code_ != OK) {
    StringStream status_message;
    status_message << "(at " << BaseName(file_) << ":" << line_ << message_str;
    message_str = status_message.str();
  }
  return Status(code_, message_str);
}

}  // namespace internal

}  // namespace fcp
