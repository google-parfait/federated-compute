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
#include "fcp/client/privatelogger/log/sqlite_log_database.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "fcp/client/confidentialcompute/storage/sqlite_connection_mutex.h"
#include "fcp/client/confidentialcompute/storage/sqlite_util.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"
#include "fcp/client/privatelogger/common/log_entry.h"
#include "fcp/client/privatelogger/log/log_database.h"
#include "sqlite3.h"

namespace fcp::client::privatelogger {
namespace {

using ::fcp::client::confidentialcompute::internal::BindBlob;
using ::fcp::client::confidentialcompute::internal::BindInt64;
using ::fcp::client::confidentialcompute::internal::BindText;
using ::fcp::client::confidentialcompute::internal::CheckNoTransactionOpen;
using ::fcp::client::confidentialcompute::internal::ColumnBlob;
using ::fcp::client::confidentialcompute::internal::ExecuteSql;
using ::fcp::client::confidentialcompute::internal::GetOrCreateConnectionMutex;
using ::fcp::client::confidentialcompute::internal::OpenSqliteDatabase;
using ::fcp::client::confidentialcompute::internal::PrepareStatement;
using ::fcp::client::confidentialcompute::internal::ScopedSqliteConnectionLock;
using ::fcp::client::confidentialcompute::internal::ScopedSqliteStmt;
using ::fcp::client::confidentialcompute::internal::ScopedSqliteTransaction;
using ::fcp::client::confidentialcompute::internal::SqliteErrorStatus;
using ::fcp::client::confidentialcompute::internal::ValidateTablePrefix;

}  // namespace

absl::StatusOr<std::unique_ptr<SqliteLogDatabase>> SqliteLogDatabase::Create(
    absl::string_view db_path, absl::string_view table_prefix) {
  ABSL_RETURN_IF_ERROR(ValidateTablePrefix(table_prefix));

  // Owned connections get kSqliteBusyTimeout.
  ABSL_ASSIGN_OR_RETURN(sqlite3 * db, OpenSqliteDatabase(db_path));

  auto database = absl::WrapUnique(new SqliteLogDatabase(
      db, /*owns_db=*/true, std::make_shared<absl::Mutex>(), table_prefix));
  {
    absl::MutexLock lock(*database->mutex_);
    ScopedSqliteConnectionLock conn_lock(database->db_);
    ABSL_RETURN_IF_ERROR(database->InitializeSchema());
  }
  return database;
}

absl::StatusOr<std::unique_ptr<SqliteLogDatabase>> SqliteLogDatabase::Create(
    sqlite3* db, absl::string_view table_prefix) {
  if (db == nullptr) {
    return absl::InvalidArgumentError(
        "SQLite database connection must not be null");
  }
  ABSL_RETURN_IF_ERROR(ValidateTablePrefix(table_prefix));

  auto database = absl::WrapUnique(new SqliteLogDatabase(
      db, /*owns_db=*/false, GetOrCreateConnectionMutex(db), table_prefix));
  {
    absl::MutexLock lock(*database->mutex_);
    ScopedSqliteConnectionLock conn_lock(database->db_);
    ABSL_RETURN_IF_ERROR(database->InitializeSchema());
  }
  return database;
}

SqliteLogDatabase::SqliteLogDatabase(sqlite3* db, bool owns_db,
                                     std::shared_ptr<absl::Mutex> mutex,
                                     absl::string_view table_prefix)
    : mutex_(std::move(mutex)),
      db_(db),
      owns_db_(owns_db),
      log_entries_table_(absl::StrCat(table_prefix, "log_entries")),
      contributed_entries_table_(
          absl::StrCat(table_prefix, "contributed_entries")) {}

SqliteLogDatabase::~SqliteLogDatabase() {
  absl::MutexLock lock(*mutex_);
  if (owns_db_ && db_ != nullptr) {
    sqlite3_close_v2(db_);
  }
  db_ = nullptr;
}

absl::Status SqliteLogDatabase::InitializeSchema() {
  const std::string create_schema_sql = absl::StrCat(
      "CREATE TABLE IF NOT EXISTS ", log_entries_table_, R"sql( (
        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        timestamp_millis INTEGER NOT NULL,
        value BLOB NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_)sql",
      log_entries_table_, "_timestamp_millis ON ", log_entries_table_,
      R"sql( (timestamp_millis);
      CREATE TABLE IF NOT EXISTS )sql",
      contributed_entries_table_, R"sql( (
        workload_set_name TEXT NOT NULL,
        entry_id INTEGER NOT NULL,
        PRIMARY KEY (workload_set_name, entry_id)
      );
      CREATE INDEX IF NOT EXISTS idx_)sql",
      contributed_entries_table_, "_entry_id ON ", contributed_entries_table_,
      " (entry_id);");
  // Create() has no DiagnosticLogger: the error is returned to the caller
  // creating the database.
  diagnostics::NoOpDiagnosticLogger no_diagnostics;
  ABSL_RETURN_IF_ERROR(CheckNoTransactionOpen(db_, no_diagnostics));
  // One transaction, so that the schema is created completely or not at all.
  ABSL_ASSIGN_OR_RETURN(ScopedSqliteTransaction transaction,
                        ScopedSqliteTransaction::Begin(db_));
  ABSL_RETURN_IF_ERROR(ExecuteSql(db_, create_schema_sql.c_str()));
  return transaction.Commit();
}

absl::Status SqliteLogDatabase::InsertEntry(
    const LogEntry& entry, diagnostics::DiagnosticLogger& logger) {
  absl::MutexLock lock(*mutex_);
  ScopedSqliteConnectionLock conn_lock(db_);
  // A single INSERT would silently join an open transaction of the client.
  ABSL_RETURN_IF_ERROR(CheckNoTransactionOpen(db_, logger));
  const std::string insert_sql =
      absl::StrCat("INSERT INTO ", log_entries_table_,
                   " (timestamp_millis, value) VALUES (?1, ?2);");
  ABSL_ASSIGN_OR_RETURN(ScopedSqliteStmt stmt,
                        PrepareStatement(db_, insert_sql));
  ABSL_RETURN_IF_ERROR(
      BindInt64(db_, stmt.get(), 1, absl::ToUnixMillis(entry.timestamp)));
  ABSL_RETURN_IF_ERROR(BindBlob(db_, stmt.get(), 2, entry.value));

  if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
    return SqliteErrorStatus(db_, "Failed to insert log entry");
  }
  return absl::OkStatus();
}

absl::StatusOr<SelectedLogEntries>
SqliteLogDatabase::GetAndMarkUncontributedEntries(
    absl::string_view workload_set_name, absl::Time cutoff_time,
    int64_t max_bytes, diagnostics::DiagnosticLogger& logger) {
  if (max_bytes <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("max_bytes must be positive, got ", max_bytes));
  }
  absl::MutexLock lock(*mutex_);
  ScopedSqliteConnectionLock conn_lock(db_);
  ABSL_RETURN_IF_ERROR(CheckNoTransactionOpen(db_, logger));
  ABSL_ASSIGN_OR_RETURN(ScopedSqliteTransaction transaction,
                        ScopedSqliteTransaction::Begin(db_));

  SelectedLogEntries selected;
  std::vector<StoredLogEntry>& results = selected.entries;
  {
    // Walks the candidates oldest first and stops at the first one that
    // doesn't fit in the budget, all inside this transaction. SQLite reads
    // each row's value when stepping to it, so the value of the first entry
    // that doesn't fit is read too, but it isn't copied.
    const std::string select_sql = absl::StrCat(
        "SELECT entry.id, entry.timestamp_millis, entry.value FROM ",
        log_entries_table_,
        " entry WHERE entry.timestamp_millis > ?1 AND NOT EXISTS ("
        "SELECT 1 FROM ",
        contributed_entries_table_,
        " contribution WHERE contribution.workload_set_name = ?2 AND "
        "contribution.entry_id = entry.id"
        ") ORDER BY entry.timestamp_millis ASC, entry.id ASC;");
    ABSL_ASSIGN_OR_RETURN(ScopedSqliteStmt select_stmt,
                          PrepareStatement(db_, select_sql));
    ABSL_RETURN_IF_ERROR(
        BindInt64(db_, select_stmt.get(), 1, absl::ToUnixMillis(cutoff_time)));
    ABSL_RETURN_IF_ERROR(
        BindText(db_, select_stmt.get(), 2, workload_set_name));

    int64_t selected_bytes = 0;
    int rc;
    while ((rc = sqlite3_step(select_stmt.get())) == SQLITE_ROW) {
      // Values are stored as BLOBs, so this is their size in bytes.
      const int64_t value_size = sqlite3_column_bytes(select_stmt.get(), 2);
      // Always take the first entry, so that one larger than the budget
      // doesn't block the queue.
      if (!results.empty() && selected_bytes + value_size > max_bytes) {
        selected.byte_limit_reached = true;
        rc = SQLITE_DONE;
        break;
      }
      selected_bytes += value_size;
      const int64_t id = sqlite3_column_int64(select_stmt.get(), 0);
      const int64_t timestamp_millis =
          sqlite3_column_int64(select_stmt.get(), 1);
      results.push_back(StoredLogEntry{
          .id = id,
          .entry =
              LogEntry{
                  .timestamp = absl::FromUnixMillis(timestamp_millis),
                  .value = ColumnBlob(select_stmt.get(), 2),
              },
      });
    }
    if (rc != SQLITE_DONE) {
      return SqliteErrorStatus(db_,
                               "Failed to query uncontributed log entries");
    }
  }

  if (!results.empty()) {
    const std::string insert_contribution_sql =
        absl::StrCat("INSERT OR IGNORE INTO ", contributed_entries_table_,
                     " (workload_set_name, entry_id) VALUES (?1, ?2);");
    ABSL_ASSIGN_OR_RETURN(ScopedSqliteStmt insert_stmt,
                          PrepareStatement(db_, insert_contribution_sql));
    ABSL_RETURN_IF_ERROR(
        BindText(db_, insert_stmt.get(), 1, workload_set_name));
    for (const StoredLogEntry& entry : results) {
      ABSL_RETURN_IF_ERROR(BindInt64(db_, insert_stmt.get(), 2, entry.id));
      if (sqlite3_step(insert_stmt.get()) != SQLITE_DONE) {
        return SqliteErrorStatus(db_,
                                 "Failed to mark log entry as contributed");
      }
      sqlite3_reset(insert_stmt.get());
    }
  }

  ABSL_RETURN_IF_ERROR(transaction.Commit());
  return selected;
}

absl::Status SqliteLogDatabase::DeleteContributions(
    absl::string_view workload_set_name, absl::Span<const int64_t> entry_ids,
    diagnostics::DiagnosticLogger& logger) {
  if (entry_ids.empty()) {
    return absl::OkStatus();
  }

  absl::MutexLock lock(*mutex_);
  ScopedSqliteConnectionLock conn_lock(db_);
  ABSL_RETURN_IF_ERROR(CheckNoTransactionOpen(db_, logger));
  ABSL_ASSIGN_OR_RETURN(ScopedSqliteTransaction transaction,
                        ScopedSqliteTransaction::Begin(db_));
  {
    const std::string delete_contribution_sql =
        absl::StrCat("DELETE FROM ", contributed_entries_table_,
                     " WHERE workload_set_name = ?1 AND entry_id = ?2;");
    ABSL_ASSIGN_OR_RETURN(ScopedSqliteStmt stmt,
                          PrepareStatement(db_, delete_contribution_sql));
    ABSL_RETURN_IF_ERROR(BindText(db_, stmt.get(), 1, workload_set_name));
    for (const int64_t entry_id : entry_ids) {
      ABSL_RETURN_IF_ERROR(BindInt64(db_, stmt.get(), 2, entry_id));
      if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
        return SqliteErrorStatus(db_, "Failed to delete contribution record");
      }
      sqlite3_reset(stmt.get());
    }
  }
  return transaction.Commit();
}

absl::StatusOr<int64_t> SqliteLogDatabase::DeleteEntriesOlderThan(
    absl::Time cutoff_time, diagnostics::DiagnosticLogger& logger) {
  absl::MutexLock lock(*mutex_);
  ScopedSqliteConnectionLock conn_lock(db_);
  ABSL_RETURN_IF_ERROR(CheckNoTransactionOpen(db_, logger));
  ABSL_ASSIGN_OR_RETURN(ScopedSqliteTransaction transaction,
                        ScopedSqliteTransaction::Begin(db_));

  const int64_t cutoff_millis = absl::ToUnixMillis(cutoff_time);

  // Each statement finds the expired entries through the timestamp index and
  // their contribution records through the entry_id index, so the cost is
  // proportional to the number of expired rows, not to the table sizes.

  // Count the expired entries that were never contributed to any workload set.
  int64_t never_contributed = 0;
  {
    const std::string count_sql = absl::StrCat(
        "SELECT COUNT(*) FROM ", log_entries_table_,
        " entry WHERE entry.timestamp_millis <= ?1 AND NOT EXISTS ("
        "SELECT 1 FROM ",
        contributed_entries_table_,
        " contribution WHERE contribution.entry_id = entry.id);");
    ABSL_ASSIGN_OR_RETURN(ScopedSqliteStmt count_stmt,
                          PrepareStatement(db_, count_sql));
    ABSL_RETURN_IF_ERROR(BindInt64(db_, count_stmt.get(), 1, cutoff_millis));
    if (sqlite3_step(count_stmt.get()) != SQLITE_ROW) {
      return SqliteErrorStatus(
          db_, "Failed to count expired uncontributed log entries");
    }
    never_contributed = sqlite3_column_int64(count_stmt.get(), 0);
  }

  // Delete the contribution records of the expired entries, while the entries
  // still exist to find them.
  {
    const std::string delete_contributions_sql =
        absl::StrCat("DELETE FROM ", contributed_entries_table_,
                     " WHERE entry_id IN (SELECT id FROM ", log_entries_table_,
                     " WHERE timestamp_millis <= ?1);");
    ABSL_ASSIGN_OR_RETURN(ScopedSqliteStmt delete_contributions_stmt,
                          PrepareStatement(db_, delete_contributions_sql));
    ABSL_RETURN_IF_ERROR(
        BindInt64(db_, delete_contributions_stmt.get(), 1, cutoff_millis));
    if (sqlite3_step(delete_contributions_stmt.get()) != SQLITE_DONE) {
      return SqliteErrorStatus(db_,
                               "Failed to delete expired contribution records");
    }
  }

  // Delete the expired entries.
  {
    const std::string delete_entries_sql = absl::StrCat(
        "DELETE FROM ", log_entries_table_, " WHERE timestamp_millis <= ?1;");
    ABSL_ASSIGN_OR_RETURN(ScopedSqliteStmt delete_entries_stmt,
                          PrepareStatement(db_, delete_entries_sql));
    ABSL_RETURN_IF_ERROR(
        BindInt64(db_, delete_entries_stmt.get(), 1, cutoff_millis));
    if (sqlite3_step(delete_entries_stmt.get()) != SQLITE_DONE) {
      return SqliteErrorStatus(db_, "Failed to delete expired log entries");
    }
  }

  ABSL_RETURN_IF_ERROR(transaction.Commit());
  return never_contributed;
}

absl::StatusOr<int64_t> SqliteLogDatabase::CountEntriesNewerThan(
    absl::Time cutoff_time) {
  absl::MutexLock lock(*mutex_);
  ScopedSqliteConnectionLock conn_lock(db_);
  const std::string count_sql =
      absl::StrCat("SELECT COUNT(*) FROM ", log_entries_table_,
                   " WHERE timestamp_millis > ?1;");
  ABSL_ASSIGN_OR_RETURN(ScopedSqliteStmt stmt,
                        PrepareStatement(db_, count_sql));
  ABSL_RETURN_IF_ERROR(
      BindInt64(db_, stmt.get(), 1, absl::ToUnixMillis(cutoff_time)));
  if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
    return SqliteErrorStatus(db_, "Failed to count log entries");
  }
  return sqlite3_column_int64(stmt.get(), 0);
}

}  // namespace fcp::client::privatelogger
