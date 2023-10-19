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

#ifdef FCP_BAREMETAL
#include <string>
#include <utility>

#include "fcp/base/new.h"
#include "fcp/base/string_stream.h"
#else
#include <sstream>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#endif  // FCP_BAREMETAL

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
 * used in. E.g. they support being used in 'bare metal' environments that don't
 * support Absl. They also reduce the logging verbosity on Android, to ensure
 * that INFO or VLOG logs are not logged to logcat unless
 * FCP_VERBOSE_ANDROID_LOGCAT is defined at build time.
 *
 * Example:
 *
 *     FCP_LOG(INFO) << "some info log";
 *     FCP_VLOG(1) << "some verbose log";
 */

#define FCP_LOG(severity) _FCP_LOG_##severity
#ifndef FCP_BAREMETAL
#define FCP_LOG_IF(severity, condition) _FCP_LOG_IF_##severity(condition)
#endif  // FCP_BAREMETAL
// An FCP_VLOG is also defined (below), but note that these log statements may
// be evaluated regardless of whether the binary's verbosity level is set high
// enough for them to be included in the actual logs, so be careful when using
// it with expensive-to-evaluate log statements.

#if !defined(__ANDROID__) && !defined(FCP_BAREMETAL)
// On regular (non-Android, non-bare metal) builds we forward all logs to Absl
// as-is.
#define _FCP_LOG_INFO ABSL_LOG(INFO)
#define _FCP_LOG_WARNING ABSL_LOG(WARNING)
#define _FCP_LOG_ERROR ABSL_LOG(ERROR)
#define _FCP_LOG_FATAL ABSL_LOG(FATAL)
#define _FCP_LOG_IF_INFO(condition) ABSL_LOG_IF(INFO, condition)
#define _FCP_LOG_IF_WARNING(condition) ABSL_LOG_IF(WARNING, condition)
#define _FCP_LOG_IF_ERROR(condition) ABSL_LOG_IF(ERROR, condition)
#define _FCP_LOG_IF_FATAL(condition) ABSL_LOG_IF(FATAL, condition)
#define FCP_VLOG(verbosity) ABSL_LOG(INFO).WithVerbosity(verbosity)

#elif defined(__ANDROID__)
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
#define FCP_VLOG(verbosity) ABSL_LOG(INFO).WithVerbosity(verbosity) << "fcp: "
#else
#define _FCP_LOG_INFO ABSL_LOG_IF(INFO, false)
#define _FCP_LOG_IF_INFO(condition) ABSL_LOG_IF(INFO, false)
#define FCP_VLOG(verbosity) ABSL_LOG_IF(INFO, false)
#endif  // FCP_VERBOSE_ANDROID_LOGCAT

#elif defined(FCP_BAREMETAL)
#define _FCP_LOG_INFO \
  ::fcp::internal::LogMessage(__FILE__, __LINE__, ::fcp::LogSeverity::kInfo)
#define _FCP_LOG_WARNING \
  ::fcp::internal::LogMessage(__FILE__, __LINE__, ::fcp::LogSeverity::kWarning)
#define _FCP_LOG_ERROR \
  ::fcp::internal::LogMessage(__FILE__, __LINE__, ::fcp::LogSeverity::kError)
#define _FCP_LOG_FATAL \
  ::fcp::internal::LogMessage(__FILE__, __LINE__, ::fcp::LogSeverity::kFatal)
#define FCP_LOG_IF(severity, condition) \
  !(condition) ? (void)0                \
               : ::fcp::internal::LogMessageVoidify() & _FCP_LOG_##severity
#endif  // FCP_BAREMETAL

#ifdef FCP_BAREMETAL
#define FCP_PREDICT_FALSE(x) (x)
#define FCP_PREDICT_TRUE(x) (x)
#else
#define FCP_PREDICT_FALSE(x) ABSL_PREDICT_FALSE(x)
#define FCP_PREDICT_TRUE(x) ABSL_PREDICT_TRUE(x)
#endif

/**
 * Check that the condition holds, otherwise die. Any additional messages can
 * be streamed into the invocation. Example:
 *
 *     FCP_CHECK(condition) << "stuff went wrong";
 */
#define FCP_CHECK(condition)                         \
  FCP_LOG_IF(FATAL, FCP_PREDICT_FALSE(!(condition))) \
      << ("Check failed: " #condition ". ")

/**
 * Check that the expression generating a status code is OK, otherwise die.
 * Any additional messages can be streamed into the invocation.
 */
#define FCP_CHECK_STATUS(status)                                     \
  for (auto __check_status = (status);                               \
       __check_status.code() != ::fcp::StatusCode::kOk;)             \
  FCP_LOG_IF(FATAL, __check_status.code() != ::fcp::StatusCode::kOk) \
      << "status not OK: " << __check_status

#ifdef FCP_BAREMETAL
// Bare Metal Logging Implementation Details
// =========================================
// When building for bare metal, we cannot use any existing logging library
// directly. Instead, we introduce a Logger class of which one global instance
// can be registered to handle log messages.

enum class LogSeverity : int {
  kInfo = 0,
  kWarning = 1,
  kError = 2,
  kFatal = 3,
};

namespace internal {

/**
 * An object which implements a logger sink. The default sink sends log
 * messages to stderr.
 */
class Logger {
 public:
  virtual ~Logger() = default;

  /**
   * Basic log function.
   *
   * @param file The name of the associated file.
   * @param line  The line in this file.
   * @param severity  Severity of the log message.
   * @param message  The message to log.
   */
  virtual void Log(const char* file, int line, LogSeverity severity,
                   const char* message);
};

/**
 * Gets/sets the active logger object.
 */
Logger* logger();
void set_logger(Logger* logger);

/**
 * Object allowing to construct a log message by streaming into it. This is
 * used by the macro LOG(severity).
 */
class LogMessage {
 public:
  LogMessage(const char* file, int line, LogSeverity severity)
      : file_(file), line_(line), severity_(severity) {}

  template <typename T>
  LogMessage& operator<<(const T& x) {
    message_ << x;
    return *this;
  }

  ~LogMessage() {
    logger()->Log(file_, line_, severity_, message_.str().c_str());
    if (severity_ == LogSeverity::kFatal) {
      abort();
    }
  }

 private:
  const char* file_;
  int line_;
  LogSeverity severity_;
  StringStream message_;
};

/**
 * This class is used to cast a LogMessage instance to void within a ternary
 * expression in the expansion of FCP_LOG_IF.  The cast is necessary so
 * that the types of the expressions on either side of the : match, and the &
 * operator is used because its precedence is lower than << but higher than
 * ?:.
 */
class LogMessageVoidify {
 public:
  void operator&(LogMessage&) {}  // NOLINT
};
}  // namespace internal
#endif  // FCP_BAREMETAL

// Status and StatusOr
// ===================

/**
 * Constructor for a status. A status message can be streamed into it. This
 * captures the current file and line position and includes it into the status
 * message if the status code is not OK.
 *
 * Use as in:
 *
 *   FCP_STATUS(OK);                // signal success
 *   FCP_STATUS(code) << message;   // signal failure
 *
 * FCP_STATUS can be used in places which either expect a Status or a
 * StatusOr<T>.
 */
#define FCP_STATUS(code) \
  ::fcp::internal::MakeStatusBuilder(code, __FILE__, __LINE__)

#ifdef FCP_BAREMETAL
#define FCP_MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define FCP_MUST_USE_RESULT ABSL_MUST_USE_RESULT
#endif

#ifdef FCP_BAREMETAL
// The bare-metal implementation doesn't depend on Abseil library and
// provides its own implementations of StatusCode, Status and StatusOr that are
// source code compatible with absl::StatusCode, absl::Status,
// and absl::StatusOr.

// See absl::StatusCode for details.
enum StatusCode : int {
  kOk = 0,
  kCancelled = 1,
  kUnknown = 2,
  kInvalidArgument = 3,
  kDeadlineExceeded = 4,
  kNotFound = 5,
  kAlreadyExists = 6,
  kPermissionDenied = 7,
  kResourceExhausted = 8,
  kFailedPrecondition = 9,
  kAborted = 10,
  kOutOfRange = 11,
  kUnimplemented = 12,
  kInternal = 13,
  kUnavailable = 14,
  kDataLoss = 15,
  kUnauthenticated = 16,
};

class FCP_MUST_USE_RESULT Status final {
 public:
  Status() : code_(StatusCode::kOk) {}
  Status(StatusCode code, const std::string& message)
      : code_(code), message_(message) {}

  // Status is copyable and moveable.
  Status(const Status&) = default;
  Status& operator=(const Status&) = default;
  Status(Status&&) = default;
  Status& operator=(Status&&) = default;

  // Tests whether this status is OK.
  bool ok() const { return code_ == StatusCode::kOk; }

  // Gets this status code.
  StatusCode code() const { return code_; }

  // Gets this status message.
  const std::string& message() const { return message_; }

 private:
  StatusCode code_;
  std::string message_;
};

template <typename T>
class FCP_MUST_USE_RESULT StatusOr final {
 public:
  // Default constructor initializes StatusOr with kUnknown code.
  explicit StatusOr() : StatusOr(StatusCode::kUnknown) {}

  // Constructs a StatusOr from a failed status. The passed status must not be
  // OK. This constructor is expected to be implicitly called.
  StatusOr(Status status)  // NOLINT
      : status_(std::move(status)) {
    FCP_CHECK(!this->status().ok());
  }

  // Constructs a StatusOr from a status code.
  explicit StatusOr(StatusCode code) : StatusOr(code, "") {}

  // Constructs a StatusOr from a status code and a message.
  StatusOr(StatusCode code, const std::string& message)
      : status_(Status(code, message)) {
    FCP_CHECK(!this->status().ok());
  }

  // Construct a StatusOr from a value.
  StatusOr(const T& value)  // NOLINT
      : value_(value) {}

  // Construct a StatusOr from an R-value.
  StatusOr(T&& value)  // NOLINT
      : value_(std::move(value)) {}

  // StatusOr is copyable and moveable.
  StatusOr(const StatusOr& other) : status_(other.status_) {
    if (ok()) {
      AssignValue(other.value_);
    }
  }

  StatusOr(StatusOr&& other) : status_(std::move(other.status_)) {
    if (ok()) {
      AssignValue(std::move(other.value_));
    }
  }

  StatusOr& operator=(const StatusOr& other) {
    if (this != &other) {
      ClearValue();
      if (other.ok()) {
        AssignValue(other.value_);
      }
      status_ = other.status_;
    }
    return *this;
  }

  StatusOr& operator=(StatusOr&& other) {
    if (this != &other) {
      ClearValue();
      if (other.ok()) {
        AssignValue(std::move(other.value_));
      }
      status_ = std::move(other.status_);
    }
    return *this;
  }

  ~StatusOr() { ClearValue(); }

  // Tests whether this StatusOr is OK and has a value.
  bool ok() const { return status_.ok(); }

  // Returns the status.
  const Status& status() const { return status_; }

  // Returns the value if the StatusOr is OK.
  const T& value() const& {
    CheckOk();
    return value_;
  }
  T& value() & {
    CheckOk();
    return value_;
  }
  T&& value() && {
    CheckOk();
    return std::move(value_);
  }

  // Operator *
  const T& operator*() const& { return value(); }
  T& operator*() & { return value(); }
  T&& operator*() && { return std::move(value()); }

  // Operator ->
  const T* operator->() const { return &value(); }
  T* operator->() { return &value(); }

  // Used to explicitly ignore a StatusOr (avoiding unused-result warnings).
  void Ignore() const {}

 private:
  void CheckOk() const { FCP_CHECK(ok()) << "StatusOr has no value"; }

  // This is used to assign the value in place without invoking the assignment
  // operator. Using the assignment operator wouldn't work in case the value_
  // wasn't previously initialized. For example the value_ object might try
  // to clear its previous value.
  template <typename Arg>
  void AssignValue(Arg&& arg) {
    new (&unused_) T(std::forward<Arg>(arg));
  }

  // Destroy the current value if it was initialized.
  void ClearValue() {
    if (ok()) value_.~T();
  }

  Status status_;

  // Using the union allows to avoid initializing the value_ field when
  // StatusOr is constructed with Status.
  struct Unused {};
  union {
    Unused unused_;
    T value_;
  };
};

#else

// By default absl::Status and absl::StatusOr classes are used.
using Status = absl::Status;
using StatusCode = absl::StatusCode;
template <typename T>
using StatusOr = absl::StatusOr<T>;

#endif  // FCP_BAREMETAL

constexpr auto OK = StatusCode::kOk;
constexpr auto CANCELLED = StatusCode::kCancelled;
constexpr auto UNKNOWN = StatusCode::kUnknown;
constexpr auto INVALID_ARGUMENT = StatusCode::kInvalidArgument;
constexpr auto DEADLINE_EXCEEDED = StatusCode::kDeadlineExceeded;
constexpr auto NOT_FOUND = StatusCode::kNotFound;
constexpr auto ALREADY_EXISTS = StatusCode::kAlreadyExists;
constexpr auto PERMISSION_DENIED = StatusCode::kPermissionDenied;
constexpr auto RESOURCE_EXHAUSTED = StatusCode::kResourceExhausted;
constexpr auto FAILED_PRECONDITION = StatusCode::kFailedPrecondition;
constexpr auto ABORTED = StatusCode::kAborted;
constexpr auto OUT_OF_RANGE = StatusCode::kOutOfRange;
constexpr auto UNIMPLEMENTED = StatusCode::kUnimplemented;
constexpr auto INTERNAL = StatusCode::kInternal;
constexpr auto UNAVAILABLE = StatusCode::kUnavailable;
constexpr auto DATA_LOSS = StatusCode::kDataLoss;
constexpr auto UNAUTHENTICATED = StatusCode::kUnauthenticated;

namespace internal {
/** Functions to assist with FCP_RETURN_IF_ERROR() */
inline const Status AsStatus(const Status& status) { return status; }
template <typename T>
inline const Status AsStatus(const StatusOr<T>& status_or) {
  return status_or.status();
}
}  // namespace internal

/**
 * Macro which allows to check for a Status (or StatusOr) and return from the
 * current method if not OK. Example:
 *
 *     Status DoSomething() {
 *       FCP_RETURN_IF_ERROR(Step1());
 *       FCP_RETURN_IF_ERROR(Step2ReturningStatusOr().status());
 *       return FCP_STATUS(OK);
 *     }
 */
#define FCP_RETURN_IF_ERROR(expr)                             \
  do {                                                        \
    ::fcp::Status __status = ::fcp::internal::AsStatus(expr); \
    if (__status.code() != ::fcp::StatusCode::kOk) {          \
      return (__status);                                      \
    }                                                         \
  } while (false)

/**
 * Macro which allows to check for a StatusOr and return it's status if not OK,
 * otherwise assign the value in the StatusOr to variable or declaration. Usage:
 *
 *     StatusOr<bool> DoSomething() {
 *       FCP_ASSIGN_OR_RETURN(auto value, TryComputeSomething());
 *       if (!value) {
 *         FCP_ASSIGN_OR_RETURN(value, TryComputeSomethingElse());
 *       }
 *       return value;
 *     }
 */
#define FCP_ASSIGN_OR_RETURN(lhs, expr) \
  _FCP_ASSIGN_OR_RETURN_1(              \
      _FCP_ASSIGN_OR_RETURN_CONCAT(statusor_for_aor, __LINE__), lhs, expr)

#define _FCP_ASSIGN_OR_RETURN_1(statusor, lhs, expr) \
  auto statusor = (expr);                            \
  if (!statusor.ok()) {                              \
    return statusor.status();                        \
  }                                                  \
  lhs = std::move(statusor).value()

// See https://goo.gl/x3iba2 for the reason of this construction.
#define _FCP_ASSIGN_OR_RETURN_CONCAT(x, y) \
  _FCP_ASSIGN_OR_RETURN_CONCAT_INNER(x, y)
#define _FCP_ASSIGN_OR_RETURN_CONCAT_INNER(x, y) x##y

// Status Implementation Details
// =============================

namespace internal {

/**
 * Helper class which allows to construct a status with message by streaming
 * into it. Implicitly converts to Status and StatusOr so can be used as a drop
 * in replacement when those types are expected.
 */
class FCP_MUST_USE_RESULT StatusBuilder {
 public:
  /** Construct a StatusBuilder from status code. */
  StatusBuilder(StatusCode code, const char* file, int line);

  /**
   * Copy constructor for status builder. Most of the time not needed because of
   * copy ellision. */
  StatusBuilder(StatusBuilder const& other);

  /** Return true if the constructed status will be OK. */
  inline bool ok() const { return code_ == OK; }

  /** Returns the code of the constructed status. */
  inline StatusCode code() const { return code_; }

  /** Stream into status message of this builder. */
  template <typename T>
  StatusBuilder& operator<<(T x) {
    message_ << x;
    return *this;
  }

  /** Implicit conversion to Status. */
  operator Status();  // NOLINT

  /** Implicit conversion to StatusOr. */
  template <typename T>
  inline operator StatusOr<T>() {  // NOLINT
    return StatusOr<T>(static_cast<Status>(*this));
  }

 private:
  const char* const file_;
  const int line_;
  const StatusCode code_;

#ifndef FCP_BAREMETAL
  using StringStream = std::ostringstream;
#endif  // FCP_BAREMETAL
  StringStream message_;
};

inline StatusBuilder MakeStatusBuilder(StatusCode code, const char* file,
                                       int line) {
  return StatusBuilder(code, file, line);
}

}  // namespace internal
}  // namespace fcp

#endif  // FCP_BASE_MONITORING_H_
