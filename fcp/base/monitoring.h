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
#ifndef FCP_BASE_MONITORING_H_
#define FCP_BASE_MONITORING_H_

#include "absl/base/optimization.h"
#include "absl/log/absl_log.h"

namespace fcp {

// General Definitions
// ===================

// Logging and Assertions
// ======================

/**
 * Defines a subset of Google style logging. Use FCP_LOG(INFO),
 * FCP_LOG(WARNING), FCP_LOG(ERROR) or FCP_LOG(FATAL) to stream log messages.
 *
 * These macros should be preferred over using Absl's log macros directly, since
 * they are compatible with the different environments some of the FCP code is
 * used in. They also reduce the logging verbosity on Android, to ensure
 * that INFO or VLOG logs are not logged to logcat unless
 * FCP_VERBOSE_ANDROID_LOGCAT is defined at build time.
 *
 * Example:
 *
 *     FCP_LOG(INFO) << "some info log";
 *     FCP_VLOG(1) << "some verbose log";
 */

#define FCP_LOG(severity) _FCP_LOG_##severity
#define FCP_LOG_IF(severity, condition) _FCP_LOG_IF_##severity(condition)
// An FCP_VLOG is also defined (below), but note that these log statements may
// be evaluated regardless of whether the binary's verbosity level is set high
// enough for them to be included in the actual logs, so be careful when using
// it with expensive-to-evaluate log statements.

#if !defined(__ANDROID__)
// On regular (non-Android) builds we forward all logs to Absl as-is.
#define _FCP_LOG_INFO ABSL_LOG(INFO)
#define _FCP_LOG_WARNING ABSL_LOG(WARNING)
#define _FCP_LOG_ERROR ABSL_LOG(ERROR)
#define _FCP_LOG_FATAL ABSL_LOG(FATAL)
#define _FCP_LOG_IF_INFO(condition) ABSL_LOG_IF(INFO, condition)
#define _FCP_LOG_IF_WARNING(condition) ABSL_LOG_IF(WARNING, condition)
#define _FCP_LOG_IF_ERROR(condition) ABSL_LOG_IF(ERROR, condition)
#define _FCP_LOG_IF_FATAL(condition) ABSL_LOG_IF(FATAL, condition)
#define FCP_VLOG(verbosity) ABSL_VLOG(verbosity)

#endif  // !defined(__ANDROID__)

#if defined(__ANDROID__)
// On Android we prepend "fcp: " to all logs, since Absl will set the log tag
// for all process-wide logs to a generic "native". Prefixing by "fcp" helps us
// find FCP-related logs in the logcat more easily.

#define _FCP_LOG_WARNING ABSL_LOG(WARNING) << "fcp: "
#define _FCP_LOG_ERROR ABSL_LOG(ERROR) << "fcp: "
#define _FCP_LOG_FATAL ABSL_LOG(FATAL) << "fcp: "
#define _FCP_LOG_IF_WARNING(condition) \
  ABSL_LOG_IF(WARNING, condition) << "fcp: "
#define _FCP_LOG_IF_ERROR(condition) ABSL_LOG_IF(ERROR, condition) << "fcp: "
#define _FCP_LOG_IF_FATAL(condition) ABSL_LOG_IF(FATAL, condition) << "fcp: "

// On Android we also, by default, do not log INFO level logs (or more verbose
// VLOGs) to Absl (which in turn would log them to logcat). Only if
// FCP_VERBOSE_ANDROID_LOGCAT is defined do we log those logs. If that is not
// defined, then the logs will be stripped out by the linker due to our use of
// ABSL_LOG_IF(..., false).

#ifdef FCP_VERBOSE_ANDROID_LOGCAT
#define _FCP_LOG_INFO ABSL_LOG(INFO) << "fcp: "
#define _FCP_LOG_IF_INFO(condition) ABSL_LOG_IF(INFO, condition) << "fcp: "
#define FCP_VLOG(verbosity) ABSL_VLOG(verbosity) << "fcp: "
#else
#define _FCP_LOG_INFO ABSL_LOG_IF(INFO, false)
#define _FCP_LOG_IF_INFO(condition) ABSL_LOG_IF(INFO, false)
#define FCP_VLOG(verbosity) ABSL_LOG_IF(INFO, false)
#endif  // FCP_VERBOSE_ANDROID_LOGCAT

#endif  // defined(__ANDROID__)

/**
 * Check that the condition holds, otherwise die. Any additional messages can
 * be streamed into the invocation. Example:
 *
 *     FCP_CHECK(condition) << "stuff went wrong";
 */
#define FCP_CHECK(condition)                          \
  FCP_LOG_IF(FATAL, ABSL_PREDICT_FALSE(!(condition))) \
      << ("Check failed: " #condition ". ")

/**
 * Check that the expression generating a status code is OK, otherwise die.
 * Any additional messages can be streamed into the invocation.
 */
#define FCP_CHECK_STATUS(status)                                      \
  for (auto __check_status = (status);                                \
       __check_status.code() != ::absl::StatusCode::kOk;)             \
  FCP_LOG_IF(FATAL, __check_status.code() != ::absl::StatusCode::kOk) \
      << "status not OK: " << __check_status

}  // namespace fcp

#endif  // FCP_BASE_MONITORING_H_
