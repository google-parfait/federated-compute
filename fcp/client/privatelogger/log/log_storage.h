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
#ifndef FCP_CLIENT_PRIVATELOGGER_LOG_LOG_STORAGE_H_
#define FCP_CLIENT_PRIVATELOGGER_LOG_LOG_STORAGE_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"
#include "fcp/client/privatelogger/common/log_entry.h"

namespace fcp::client::privatelogger {

// How urgently LogStorage::CleanUp() should delete expired entries.
enum class CleanupMode {
  // A cheap, frequent call, e.g. on every Log(): the storage may skip the
  // cleanup most of the time, so that expired entries are deleted eventually
  // without a database write on every call.
  kPeriodic,
  // Clean up now, e.g. when the logger is created or after an upload.
  kImmediate,
};

// The entries LogStorage::GetLogsToUpload() selected for one workload set,
// already marked as contributed to it. If the upload succeeds, the caller just
// drops the batch; otherwise RollBack() makes the entries selectable again.
//
// Delivery is at most once: the entries are marked as contributed when they're
// selected, so if a batch is lost without RollBack() (a crash, a forgotten
// rollback, or move-assigning another batch over one that wasn't rolled back),
// its entries are never offered to that workload set again.
//
// A batch must not outlive the LogStorage that created it: RollBack() uses
// that storage.
class UploadBatch {
 public:
  // Gets the caller's DiagnosticLogger, for the causes the storage detects.
  using RollbackCallback =
      absl::AnyInvocable<absl::Status(diagnostics::DiagnosticLogger&) &&>;

  UploadBatch() = default;
  UploadBatch(std::vector<LogEntry> entries, RollbackCallback rollback,
              bool byte_limit_reached = false)
      : entries_(std::move(entries)),
        rollback_(std::move(rollback)),
        byte_limit_reached_(byte_limit_reached) {}

  UploadBatch(const UploadBatch&) = delete;
  UploadBatch& operator=(const UploadBatch&) = delete;

  // Moving leaves `other` empty: no entries, and its RollBack() does nothing
  // (a moved-from absl::AnyInvocable isn't guaranteed to be empty).
  UploadBatch(UploadBatch&& other) noexcept
      : entries_(std::exchange(other.entries_, {})),
        rollback_(std::exchange(other.rollback_, nullptr)),
        byte_limit_reached_(std::exchange(other.byte_limit_reached_, false)) {}
  UploadBatch& operator=(UploadBatch&& other) noexcept {
    entries_ = std::exchange(other.entries_, {});
    rollback_ = std::exchange(other.rollback_, nullptr);
    byte_limit_reached_ = std::exchange(other.byte_limit_reached_, false);
    return *this;
  }

  ~UploadBatch() = default;

  absl::Span<const LogEntry> entries() const { return entries_; }
  bool empty() const { return entries_.empty(); }
  // Whether the byte budget ended the selection: at least one more entry that
  // could have been selected was left for a later batch.
  bool byte_limit_reached() const { return byte_limit_reached_; }

  // Unmarks the entries as contributed to the workload set. Only the first
  // call does anything; later calls return OK.
  absl::Status RollBack(diagnostics::DiagnosticLogger& logger) {
    if (rollback_ == nullptr) {
      return absl::OkStatus();
    }
    return std::exchange(rollback_, nullptr)(logger);
  }

 private:
  std::vector<LogEntry> entries_;
  RollbackCallback rollback_;
  bool byte_limit_reached_ = false;
};

// Stores log entries and tracks which have been contributed to which workload
// set. Implementations must be thread-safe: Log() and CleanUp() may be called
// from any thread while another thread selects entries (GetLogsToUpload()) and
// rolls them back, and GetLogsToUpload() may be called concurrently, for the
// same or different workload sets. GetLogsToUpload() must select and mark the
// entries as one atomic step, so that an entry logged concurrently ends up in
// either this batch or a later one, and no entry is in two batches for the same
// workload set.
class LogStorage {
 public:
  LogStorage() = default;
  virtual ~LogStorage() = default;

  // The methods that write take the caller's DiagnosticLogger, to which an
  // implementation logs the causes it detects itself (e.g.
  // kErrorSqliteTransactionOpen); the caller logs its operation's code only
  // if no cause was logged.
  virtual absl::Status Log(const LogEntry& input,
                           diagnostics::DiagnosticLogger& logger) = 0;

  // Selects the oldest entries logged after `cutoff_time` (the start of the
  // retention window) and not yet contributed to `workload_set_name`, and
  // marks them as contributed. Entries are taken oldest first while the sum of
  // their value sizes stays within `max_bytes` (> 0), but always at least one,
  // so that an entry larger than the budget can't block the others forever.
  // The rest stay uncontributed for a later call; the batch's
  // byte_limit_reached() says whether there are any. The returned batch,
  // including its RollBack(), must not outlive this storage.
  virtual absl::StatusOr<UploadBatch> GetLogsToUpload(
      absl::string_view workload_set_name, absl::Time cutoff_time,
      int64_t max_bytes, diagnostics::DiagnosticLogger& logger) = 0;

  // Deletes the entries logged at or before `cutoff_time` (outside the
  // retention window) and their contribution records. kPeriodic may skip
  // this most of the time; kImmediate must do it now. Returns how many of the
  // deleted entries were never contributed to any workload set (0 if nothing
  // was deleted, including when kPeriodic skipped the cleanup), so that the
  // caller can report entries that expired without being uploaded.
  virtual absl::StatusOr<int64_t> CleanUp(
      CleanupMode mode, absl::Time cutoff_time,
      diagnostics::DiagnosticLogger& logger) = 0;

  // Returns the number of entries logged after `cutoff_time`, i.e. all entries
  // in the retention window, including those already contributed to every
  // workload set. Which workload sets exist is only known later, so this is an
  // upper bound of what an upload could take; 0 means there's nothing to
  // upload.
  virtual absl::StatusOr<int64_t> CountPendingEntries(
      absl::Time cutoff_time) = 0;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_LOG_LOG_STORAGE_H_
