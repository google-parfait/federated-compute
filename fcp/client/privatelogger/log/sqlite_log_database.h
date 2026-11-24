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
#ifndef FCP_CLIENT_PRIVATELOGGER_LOG_SQLITE_LOG_DATABASE_H_
#define FCP_CLIENT_PRIVATELOGGER_LOG_SQLITE_LOG_DATABASE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"
#include "fcp/client/privatelogger/common/log_entry.h"
#include "fcp/client/privatelogger/log/log_database.h"

struct sqlite3;

namespace fcp::client::privatelogger {

// A LogDatabase backed by two SQLite tables. Thread-safe.
//
// Their names start with `table_prefix` (empty by default), so that several
// instances can share one database. All columns are NOT NULL.
//
//   <prefix>log_entries: one row per logged entry.
//   +------------------+---------+------------------------------------------+
//   | id               | INTEGER | PRIMARY KEY AUTOINCREMENT (never reused) |
//   | timestamp_millis | INTEGER | LogEntry::timestamp, in Unix millis      |
//   | value            | BLOB    | LogEntry::value                          |
//   +------------------+---------+------------------------------------------+
//   INDEX idx_<prefix>log_entries_timestamp_millis ON (timestamp_millis)
//
//   <prefix>contributed_entries: one row per workload set an entry was
//   contributed to (marked by GetAndMarkUncontributedEntries(), unmarked by
//   DeleteContributions()).
//   +-------------------+---------+-----------------------------------------+
//   | workload_set_name | TEXT    | PRIMARY KEY (workload_set_name,         |
//   | entry_id          | INTEGER |              entry_id)                  |
//   |                   |         | entry_id -> <prefix>log_entries.id      |
//   +-------------------+---------+-----------------------------------------+
//   INDEX idx_<prefix>contributed_entries_entry_id ON (entry_id)
//
//   log_entries.id 1 ---- 0..n contributed_entries.entry_id
//
// entry_id isn't a declared foreign key; DeleteEntriesOlderThan() deletes the
// contribution records of the entries it deletes, in the same transaction. The
// two indexes let it find both by the expired rows only.
class SqliteLogDatabase : public LogDatabase {
 public:
  // Opens (or creates) a SQLite database at `db_path` (use ":memory:" for an
  // in-memory database) and initializes its schema using `table_prefix`.
  // The returned instance owns the `sqlite3` connection and closes it on
  // destruction. The connection waits up to 5 seconds for another
  // connection's lock on the file before failing with SQLITE_BUSY. To keep
  // other data in the same file, prefer opening one connection yourself and
  // passing it to every user (Create(sqlite3*)): that avoids lock contention
  // between connections altogether.
  static absl::StatusOr<std::unique_ptr<SqliteLogDatabase>> Create(
      absl::string_view db_path, absl::string_view table_prefix = "");

  // Wraps an existing open `sqlite3` connection and initializes its schema
  // using `table_prefix`. Multiple `SqliteLogDatabase` instances may share the
  // same `sqlite3*` pointer (typically with distinct `table_prefix` values);
  // access to the shared connection is synchronized automatically. Does not
  // take ownership of `db`; the caller must keep `db` alive for the lifetime of
  // all `SqliteLogDatabase` instances using it and close it afterwards. The
  // connection's settings are left as they are: the caller is responsible for
  // its busy timeout (sqlite3_busy_timeout()) if other connections use the
  // same file. The caller must not keep its own transactions open on `db`
  // while instances use it: the shared lock serializes single SQLite calls,
  // not the caller's transactions, so every write checks that no transaction
  // is open and otherwise fails with FAILED_PRECONDITION instead of joining
  // it. The write methods also log kErrorSqliteTransactionOpen to their
  // logger; Create() has no logger, so its schema setup only returns the
  // error.
  static absl::StatusOr<std::unique_ptr<SqliteLogDatabase>> Create(
      sqlite3* db, absl::string_view table_prefix = "");

  ~SqliteLogDatabase() override;

  SqliteLogDatabase(const SqliteLogDatabase&) = delete;
  SqliteLogDatabase& operator=(const SqliteLogDatabase&) = delete;

  absl::Status InsertEntry(const LogEntry& entry,
                           diagnostics::DiagnosticLogger& logger) override;

  absl::StatusOr<SelectedLogEntries> GetAndMarkUncontributedEntries(
      absl::string_view workload_set_name, absl::Time cutoff_time,
      int64_t max_bytes, diagnostics::DiagnosticLogger& logger) override;

  absl::Status DeleteContributions(
      absl::string_view workload_set_name, absl::Span<const int64_t> entry_ids,
      diagnostics::DiagnosticLogger& logger) override;

  absl::StatusOr<int64_t> DeleteEntriesOlderThan(
      absl::Time cutoff_time, diagnostics::DiagnosticLogger& logger) override;

  absl::StatusOr<int64_t> CountEntriesNewerThan(
      absl::Time cutoff_time) override;

 private:
  SqliteLogDatabase(sqlite3* db, bool owns_db,
                    std::shared_ptr<absl::Mutex> mutex,
                    absl::string_view table_prefix);

  absl::Status InitializeSchema() ABSL_EXCLUSIVE_LOCKS_REQUIRED(*mutex_);

  std::shared_ptr<absl::Mutex> mutex_;
  sqlite3* db_ ABSL_GUARDED_BY(*mutex_) = nullptr;
  const bool owns_db_;
  const std::string log_entries_table_;
  const std::string contributed_entries_table_;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_LOG_SQLITE_LOG_DATABASE_H_
