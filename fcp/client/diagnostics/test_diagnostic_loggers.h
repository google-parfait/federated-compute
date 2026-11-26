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
#ifndef FCP_CLIENT_DIAGNOSTICS_TEST_DIAGNOSTIC_LOGGERS_H_
#define FCP_CLIENT_DIAGNOSTICS_TEST_DIAGNOSTIC_LOGGERS_H_

#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"

// DiagnosticLoggers for tests.
namespace fcp::client::diagnostics {

// Records the logged codes. Thread-safe.
class RecordingDiagnosticLogger : public DiagnosticLogger {
 public:
  void LogCode(DiagnosticCode code) override {
    absl::MutexLock lock(mutex_);
    codes_.push_back(code);
  }

  // The codes logged so far, in order.
  std::vector<DiagnosticCode> codes() const {
    absl::MutexLock lock(mutex_);
    return codes_;
  }

 private:
  mutable absl::Mutex mutex_;
  std::vector<DiagnosticCode> codes_ ABSL_GUARDED_BY(mutex_);
};

// A shared NoOpDiagnosticLogger, for calls whose codes a test doesn't check.
inline DiagnosticLogger& NoDiagnostics() {
  static NoOpDiagnosticLogger* const logger = new NoOpDiagnosticLogger();
  return *logger;
}

}  // namespace fcp::client::diagnostics

#endif  // FCP_CLIENT_DIAGNOSTICS_TEST_DIAGNOSTIC_LOGGERS_H_
