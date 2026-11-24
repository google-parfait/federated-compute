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
#ifndef FCP_CLIENT_DIAGNOSTICS_DIAGNOSTIC_LOGGER_H_
#define FCP_CLIENT_DIAGNOSTICS_DIAGNOSTIC_LOGGER_H_

namespace fcp::client::diagnostics {

// Codes reported to a DiagnosticLogger. They carry no context, so they must
// be as detailed as possible: some platforms (e.g. Chrome) only see the
// integer value, and the absl::Status details (message, payloads) aren't
// available there. Rules:
//
// - Every distinct failure cause gets its own code. Never reuse a generic
//   code for a new cause: add a new value instead. A generic "unclassified"
//   error code is only a last resort, for failures no code path classified.
// - Values have 4 digits and are grouped by kind: info codes (kInfo...) from
//   1000, successes (kSuccess...) from 2000, and errors (kError...) from 3000,
//   each numbered sequentially. A new value is appended at the end of its
//   kind's range.
// - Values are stable once released: never renumber, and never reuse removed
//   values.
// - Names start with kSuccess for successes, kInfo for normal events that are
//   neither successes nor errors, and kError for errors.
//
// Callers log at most one cause code per failure, where it happens.
enum class DiagnosticCode : int {
  // Never logged; the zero value.
  kUnknown = 0,
};

// Receives DiagnosticCodes, e.g. to forward them to the platform's
// monitoring. Must be thread-safe if codes are logged from several threads.
// Codes are plain integers; a richer event API may be added later.
class DiagnosticLogger {
 public:
  virtual ~DiagnosticLogger() = default;

  virtual void LogCode(DiagnosticCode code) = 0;
};

// A DiagnosticLogger that drops every code.
class NoOpDiagnosticLogger : public DiagnosticLogger {
 public:
  void LogCode(DiagnosticCode code) override {}
};

}  // namespace fcp::client::diagnostics

#endif  // FCP_CLIENT_DIAGNOSTICS_DIAGNOSTIC_LOGGER_H_
