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

#include <stdlib.h> /* for abort() */

#include <string>

#ifndef FCP_BAREMETAL
#include <stdarg.h>
#include <stdio.h>

#ifdef __ANDROID__
#include <android/log.h>
#endif

#include <cstring>

#include "absl/strings/str_format.h"
#endif  // FCP_BAREMETAL

#include "fcp/base/base_name.h"

namespace fcp {

namespace internal {

namespace {
#ifdef __ANDROID__
constexpr char kAndroidLogTag[] = "fcp";

int AndroidLogLevel(LogSeverity severity) {
  switch (severity) {
    case LogSeverity::kFatal:
      return ANDROID_LOG_FATAL;
    case LogSeverity::kError:
      return ANDROID_LOG_ERROR;
    case LogSeverity::kWarning:
      return ANDROID_LOG_WARN;
    default:
      return ANDROID_LOG_INFO;
  }
}
#endif

}  // namespace

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
#ifndef FCP_BAREMETAL
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
  log_to_logcat = severity != LogSeverity::kInfo;
#endif  // NDEBUG
  if (log_to_logcat) {
    int level = AndroidLogLevel(severity);
    __android_log_print(level, kAndroidLogTag, "%c %s:%d %s\n",
                        absl::LogSeverityName(severity)[0],
                        base_file_name.c_str(), line, message);
  }
#endif  // __ANDROID__
  // Note that on Android we print both to logcat *and* stderr. This allows
  // tests to use ASSERT_DEATH to test for fatal error messages, among other
  // uses.
  absl::FPrintF(stderr, "%c %s:%d %s\n", absl::LogSeverityName(severity)[0],
                base_file_name, line, message);
#endif  // FCP_BAREMETAL
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
    StringStream status_message;
    status_message << "(at " << BaseName(file_) << ":" << line_ << message_str;
    message_str = status_message.str();
    if (log_severity_ != kNoLog) {
      StringStream log_message;
      log_message << "[" << code_ << "] " << message_str;
      logger()->Log(file_, line_, log_severity_, log_message.str().c_str());
      if (log_severity_ == LogSeverity::kFatal) {
        abort();
      }
    }
  }
  return Status(code_, message_str);
}

}  // namespace internal

}  // namespace fcp
