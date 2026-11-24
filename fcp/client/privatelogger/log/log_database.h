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
#ifndef FCP_CLIENT_PRIVATELOGGER_LOG_LOG_DATABASE_H_
#define FCP_CLIENT_PRIVATELOGGER_LOG_LOG_DATABASE_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"
#include "fcp/client/privatelogger/common/log_entry.h"

namespace fcp::client::privatelogger {

// A log entry together with the ID the database assigned to it.
struct StoredLogEntry {
  int64_t id;
  LogEntry entry;
};

// The entries GetAndMarkUncontributedEntries() selected.
struct SelectedLogEntries {
  std::vector<StoredLogEntry> entries;
  // Whether the byte budget ended the selection, i.e. at least one more
  // selectable entry was left out.
  bool byte_limit_reached = false;
};

// Stores log entries and, per workload set, which of them were contributed to
// it.
//
// The write methods take the caller's DiagnosticLogger, to which they log the
// causes they detect themselves (e.g. kErrorSqliteTransactionOpen); the caller
// logs its operation's code only if no cause was logged.
class LogDatabase {
 public:
  LogDatabase() = default;
  virtual ~LogDatabase() = default;

  virtual absl::Status InsertEntry(const LogEntry& entry,
                                   diagnostics::DiagnosticLogger& logger) = 0;

  // In one transaction, selects the oldest (by timestamp, then insertion
  // order) entries newer than `cutoff_time` that aren't contributed to
  // `workload_set_name`, and marks them as contributed. Entries are taken in
  // that order until the next one would bring the sum of the value sizes over
  // `max_bytes` (> 0); the first entry is always taken, even if it's larger.
  // The others stay uncontributed for a later call.
  virtual absl::StatusOr<SelectedLogEntries> GetAndMarkUncontributedEntries(
      absl::string_view workload_set_name, absl::Time cutoff_time,
      int64_t max_bytes, diagnostics::DiagnosticLogger& logger) = 0;

  // Removes the contribution records of `entry_ids` for `workload_set_name`,
  // so that GetAndMarkUncontributedEntries() selects those entries again.
  virtual absl::Status DeleteContributions(
      absl::string_view workload_set_name, absl::Span<const int64_t> entry_ids,
      diagnostics::DiagnosticLogger& logger) = 0;

  // Deletes the entries with a timestamp at or before `cutoff_time` (so
  // exactly those GetAndMarkUncontributedEntries() no longer selects) and
  // their contribution records. Returns how many of the deleted entries were
  // never contributed to any workload set.
  virtual absl::StatusOr<int64_t> DeleteEntriesOlderThan(
      absl::Time cutoff_time, diagnostics::DiagnosticLogger& logger) = 0;

  // Returns the number of entries with a timestamp after `cutoff_time`, i.e.
  // those GetAndMarkUncontributedEntries() may still select for some workload
  // set.
  virtual absl::StatusOr<int64_t> CountEntriesNewerThan(
      absl::Time cutoff_time) = 0;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_LOG_LOG_DATABASE_H_
