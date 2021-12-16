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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h> /* for abort() */
#include <string.h>

#ifdef __ANDROID__
#include <android/log.h>
#endif

#include <cstring>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "fcp/base/platform.h"

namespace fcp {

namespace internal {

namespace {
#ifdef __ANDROID__
constexpr char kAndroidLogTag[] = "fcp";

int AndroidLogLevel(absl::LogSeverity severity) {
  switch (severity) {
    case absl::LogSeverity::kFatal:
      return ANDROID_LOG_FATAL;
    case absl::LogSeverity::kError:
      return ANDROID_LOG_ERROR;
    case absl::LogSeverity::kWarning:
      return ANDROID_LOG_WARN;
    default:
      return ANDROID_LOG_INFO;
  }
}
#endif
}  // namespace

Logger* logger = new Logger();

void Logger::Log(const char* file, int line, absl::LogSeverity severity,
                 const char* message) {
  auto base_file_name = BaseName(file);
#ifdef __ANDROID__
  bool log_to_logcat = true;
#ifdef NDEBUG
  // We don't log INFO logs on Android if this is a production build, since
  // they're too verbose. We can't just log them at ANDROID_LOG_VERBOSE either,
  // since then they'd still show up in the logcat unless we first check
  // __android_log_is_loggable, but that function isn't available until Android
  // API level 30. So to keep things simple we only log warnings or above,
  // unless this is a debug build.
  log_to_logcat = severity != absl::LogSeverity::kInfo;
#endif
  if (log_to_logcat) {
    int level = AndroidLogLevel(severity);
    __android_log_print(level, kAndroidLogTag, "%c %s:%d %s\n",
                        absl::LogSeverityName(severity)[0],
                        base_file_name.c_str(), line, message);
  }
#endif
  // Note that on Android we print both to logcat *and* stderr. This allows
  // tests to use ASSERT_DEATH to test for fatal error messages, among other
  // uses.
  absl::FPrintF(stderr, "%c %s:%d %s\n", absl::LogSeverityName(severity)[0],
                base_file_name, line, message);
}

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
    message_str = absl::StrCat("(at ", BaseName(file_), ":",
                               std::to_string(line_), ") ", message_str);
    if (log_severity_ != kNoLog) {
      logger->Log(file_, line_, log_severity_,
                  absl::StrCat("[", code_, "] ", message_str).c_str());
      if (log_severity_ == absl::LogSeverity::kFatal) {
        abort();
      }
    }
  }
  return Status(code_, message_str);
}

}  // namespace internal

}  // namespace fcp
