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
#ifndef FCP_CLIENT_PRIVATELOGGER_LOG_DATABASE_LOG_STORAGE_H_
#define FCP_CLIENT_PRIVATELOGGER_LOG_DATABASE_LOG_STORAGE_H_

#include <atomic>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"
#include "fcp/client/privatelogger/common/log_entry.h"
#include "fcp/client/privatelogger/log/log_database.h"
#include "fcp/client/privatelogger/log/log_storage.h"

namespace fcp::client::privatelogger {

// A LogStorage on a LogDatabase, which must be thread-safe.
class DatabaseLogStorage : public LogStorage {
 public:
  // `log_database` must not be null; a null one is a programming error and
  // crashes here rather than on first use.
  explicit DatabaseLogStorage(std::unique_ptr<LogDatabase> log_database);
  ~DatabaseLogStorage() override = default;

  absl::Status Log(const LogEntry& input,
                   diagnostics::DiagnosticLogger& logger) override;

  absl::StatusOr<UploadBatch> GetLogsToUpload(
      absl::string_view workload_set_name, absl::Time cutoff_time,
      int64_t max_bytes, diagnostics::DiagnosticLogger& logger) override;

  absl::StatusOr<int64_t> CleanUp(
      CleanupMode mode, absl::Time cutoff_time,
      diagnostics::DiagnosticLogger& logger) override;
  absl::StatusOr<int64_t> CountPendingEntries(absl::Time cutoff_time) override;

 private:
  std::shared_ptr<LogDatabase> log_database_;
  std::atomic<int> calls_since_last_cleanup_ = 0;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_LOG_DATABASE_LOG_STORAGE_H_
