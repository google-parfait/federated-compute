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
#include "fcp/client/privatelogger/log/database_log_storage.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"
#include "fcp/client/privatelogger/common/log_entry.h"
#include "fcp/client/privatelogger/log/log_database.h"
#include "fcp/client/privatelogger/log/log_storage.h"

namespace fcp::client::privatelogger {
namespace {

constexpr int kPeriodicCleanupInterval = 100;

}  // namespace

DatabaseLogStorage::DatabaseLogStorage(
    std::unique_ptr<LogDatabase> log_database)
    : log_database_(std::move(log_database)) {
  FCP_CHECK(log_database_ != nullptr);
}

absl::Status DatabaseLogStorage::Log(const LogEntry& input,
                                     diagnostics::DiagnosticLogger& logger) {
  return log_database_->InsertEntry(input, logger);
}

absl::StatusOr<UploadBatch> DatabaseLogStorage::GetLogsToUpload(
    absl::string_view workload_set_name, absl::Time cutoff_time,
    int64_t max_bytes, diagnostics::DiagnosticLogger& logger) {
  ABSL_ASSIGN_OR_RETURN(SelectedLogEntries selected,
                        log_database_->GetAndMarkUncontributedEntries(
                            workload_set_name, cutoff_time, max_bytes, logger));
  std::vector<StoredLogEntry>& stored_entries = selected.entries;
  if (stored_entries.empty()) {
    // Nothing was marked, so there's nothing to roll back.
    return UploadBatch();
  }

  std::vector<LogEntry> entries;
  std::vector<int64_t> entry_ids;
  entries.reserve(stored_entries.size());
  entry_ids.reserve(stored_entries.size());
  for (StoredLogEntry& stored_entry : stored_entries) {
    entry_ids.push_back(stored_entry.id);
    entries.push_back(std::move(stored_entry.entry));
  }

  return UploadBatch(
      std::move(entries),
      // A weak_ptr instead of `this`: see log_database_.
      [database = std::weak_ptr<LogDatabase>(log_database_),
       workload_set = std::string(workload_set_name),
       ids = std::move(entry_ids)](
          diagnostics::DiagnosticLogger& rollback_logger) -> absl::Status {
        std::shared_ptr<LogDatabase> log_database = database.lock();
        if (log_database == nullptr) {
          return absl::FailedPreconditionError(
              "The LogStorage of this batch was destroyed");
        }
        return log_database->DeleteContributions(workload_set, ids,
                                                 rollback_logger);
      },
      selected.byte_limit_reached);
}

absl::StatusOr<int64_t> DatabaseLogStorage::CleanUp(
    CleanupMode mode, absl::Time cutoff_time,
    diagnostics::DiagnosticLogger& logger) {
  switch (mode) {
    case CleanupMode::kPeriodic:
      // Concurrent calls may occasionally clean up twice in a row; that's
      // harmless.
      if (calls_since_last_cleanup_.fetch_add(1, std::memory_order_relaxed) <
          kPeriodicCleanupInterval - 1) {
        return 0;
      }
      [[fallthrough]];
    case CleanupMode::kImmediate:
      calls_since_last_cleanup_.store(0, std::memory_order_relaxed);
      break;
  }
  return log_database_->DeleteEntriesOlderThan(cutoff_time, logger);
}

absl::StatusOr<int64_t> DatabaseLogStorage::CountPendingEntries(
    absl::Time cutoff_time) {
  return log_database_->CountEntriesNewerThan(cutoff_time);
}

}  // namespace fcp::client::privatelogger
