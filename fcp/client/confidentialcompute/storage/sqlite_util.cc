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
#include "fcp/client/confidentialcompute/storage/sqlite_util.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"
#include "sqlite3.h"

namespace fcp::client::confidentialcompute::internal {

void SqliteStmtDeleter::operator()(sqlite3_stmt* stmt) const {
  sqlite3_finalize(stmt);
}

absl::StatusOr<sqlite3*> OpenSqliteDatabase(absl::string_view path) {
  // sqlite3_open_v2() takes a C string, which would silently cut the path at
  // the first NUL.
  if (absl::StrContains(path, '\0')) {
    return absl::InvalidArgumentError(
        "SQLite database path must not contain NUL characters");
  }
  sqlite3* db = nullptr;
  const std::string path_str(path);
  const int flags =
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX;
  if (sqlite3_open_v2(path_str.c_str(), &db, flags, nullptr) != SQLITE_OK) {
    absl::Status status =
        SqliteErrorStatus(db, "Failed to open SQLite database");
    if (db != nullptr) {
      sqlite3_close_v2(db);
    }
    return status;
  }
  if (sqlite3_busy_timeout(db, static_cast<int>(absl::ToInt64Milliseconds(
                                   kSqliteBusyTimeout))) != SQLITE_OK) {
    absl::Status status =
        SqliteErrorStatus(db, "Failed to set the SQLite busy timeout");
    sqlite3_close_v2(db);
    return status;
  }
  return db;
}

absl::Status CheckNoTransactionOpen(sqlite3* db,
                                    diagnostics::DiagnosticLogger& logger) {
  if (db == nullptr) {
    return absl::InvalidArgumentError("SQLite connection is null");
  }
  if (sqlite3_get_autocommit(db) == 0) {
    logger.LogCode(diagnostics::DiagnosticCode::kErrorSqliteTransactionOpen);
    return absl::FailedPreconditionError(
        "A transaction is already open on the shared SQLite connection");
  }
  return absl::OkStatus();
}

absl::Status ValidateTablePrefix(absl::string_view table_prefix) {
  for (const char c : table_prefix) {
    if (!absl::ascii_isalnum(static_cast<unsigned char>(c)) && c != '_') {
      return absl::InvalidArgumentError(
          "Table prefix must contain only alphanumeric characters and '_'");
    }
  }
  if (!table_prefix.empty() &&
      absl::ascii_isdigit(static_cast<unsigned char>(table_prefix.front()))) {
    return absl::InvalidArgumentError(
        "Table prefix must not start with a digit");
  }
  if (absl::StartsWithIgnoreCase(table_prefix, "sqlite_")) {
    return absl::InvalidArgumentError(
        "Table prefix must not start with 'sqlite_', which SQLite reserves for "
        "its internal tables");
  }
  return absl::OkStatus();
}

absl::Status SqliteErrorStatus(sqlite3* db, absl::string_view message) {
  if (db == nullptr) {
    return absl::InternalError(message);
  }
  return absl::InternalError(absl::StrCat(message, ": ", sqlite3_errmsg(db)));
}

absl::Status CheckSqliteSize(size_t size, absl::string_view what) {
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::OutOfRangeError(
        absl::StrCat("SQLite ", what, " too large: ", size, " bytes"));
  }
  return absl::OkStatus();
}

absl::StatusOr<ScopedSqliteStmt> PrepareStatement(sqlite3* db,
                                                  absl::string_view sql) {
  if (db == nullptr) {
    return absl::InvalidArgumentError("SQLite connection is null");
  }
  ABSL_RETURN_IF_ERROR(CheckSqliteSize(sql.size(), "SQL statement"));
  sqlite3_stmt* raw_stmt = nullptr;
  if (sqlite3_prepare_v2(db, sql.data(), static_cast<int>(sql.size()),
                         &raw_stmt, nullptr) != SQLITE_OK) {
    sqlite3_finalize(raw_stmt);
    return SqliteErrorStatus(db, "Failed to prepare SQLite statement");
  }
  // SQLite prepares no statement, successfully, for SQL that is empty or
  // contains only whitespace or comments.
  if (raw_stmt == nullptr) {
    return absl::InvalidArgumentError("SQL statement is empty");
  }
  return ScopedSqliteStmt(raw_stmt);
}

std::string ColumnText(sqlite3_stmt* stmt, int index) {
  const unsigned char* data = sqlite3_column_text(stmt, index);
  // NULL for a NULL value (or if SQLite ran out of memory). Otherwise the size
  // is >= 0.
  if (data == nullptr) {
    return "";
  }
  const int size = sqlite3_column_bytes(stmt, index);
  return std::string(reinterpret_cast<const char*>(data),
                     static_cast<size_t>(size));
}

std::string ColumnBlob(sqlite3_stmt* stmt, int index) {
  const void* data = sqlite3_column_blob(stmt, index);
  // NULL for a NULL or zero-length value (or if SQLite ran out of memory).
  // Otherwise the size is > 0.
  if (data == nullptr) {
    return "";
  }
  const int size = sqlite3_column_bytes(stmt, index);
  return std::string(static_cast<const char*>(data), static_cast<size_t>(size));
}

absl::Status BindText(sqlite3* db, sqlite3_stmt* stmt, int index,
                      absl::string_view value) {
  ABSL_RETURN_IF_ERROR(CheckSqliteSize(value.size(), "text parameter"));
  const char* data = value.empty() ? "" : value.data();
  if (sqlite3_bind_text(stmt, index, data, static_cast<int>(value.size()),
                        SQLITE_TRANSIENT) != SQLITE_OK) {
    return SqliteErrorStatus(db, "Failed to bind text parameter");
  }
  return absl::OkStatus();
}

absl::Status BindBlob(sqlite3* db, sqlite3_stmt* stmt, int index,
                      absl::string_view value) {
  ABSL_RETURN_IF_ERROR(CheckSqliteSize(value.size(), "blob parameter"));
  const char* data = value.empty() ? "" : value.data();
  if (sqlite3_bind_blob(stmt, index, data, static_cast<int>(value.size()),
                        SQLITE_TRANSIENT) != SQLITE_OK) {
    return SqliteErrorStatus(db, "Failed to bind blob parameter");
  }
  return absl::OkStatus();
}

absl::Status BindInt64(sqlite3* db, sqlite3_stmt* stmt, int index,
                       int64_t value) {
  if (sqlite3_bind_int64(stmt, index, value) != SQLITE_OK) {
    return SqliteErrorStatus(db, "Failed to bind int64 parameter");
  }
  return absl::OkStatus();
}

absl::Status ExecuteSql(sqlite3* db, const char* sql) {
  if (db == nullptr) {
    return absl::InvalidArgumentError("SQLite connection is null");
  }
  char* error_msg = nullptr;
  if (sqlite3_exec(db, sql, nullptr, nullptr, &error_msg) != SQLITE_OK) {
    std::string error_str =
        error_msg != nullptr ? error_msg : sqlite3_errmsg(db);
    sqlite3_free(error_msg);
    return absl::InternalError(
        absl::StrCat("SQLite execution failed: ", error_str));
  }
  return absl::OkStatus();
}

namespace {

void Rollback(sqlite3* db) {
  // SQLite may already have rolled back after some errors (e.g. SQLITE_FULL).
  if (sqlite3_get_autocommit(db) == 0) {
    ExecuteSql(db, "ROLLBACK;").IgnoreError();
  }
}

}  // namespace

absl::StatusOr<ScopedSqliteTransaction> ScopedSqliteTransaction::Begin(
    sqlite3* db) {
  ABSL_RETURN_IF_ERROR(ExecuteSql(db, "BEGIN IMMEDIATE;"));
  return ScopedSqliteTransaction(db);
}

ScopedSqliteTransaction::ScopedSqliteTransaction(
    ScopedSqliteTransaction&& other) noexcept
    : db_(std::exchange(other.db_, nullptr)) {}

ScopedSqliteTransaction::~ScopedSqliteTransaction() {
  if (db_ != nullptr) {
    Rollback(db_);
  }
}

absl::Status ScopedSqliteTransaction::Commit() {
  if (db_ == nullptr) {
    return absl::FailedPreconditionError("The transaction is already over");
  }
  sqlite3* const db = std::exchange(db_, nullptr);
  absl::Status status = ExecuteSql(db, "COMMIT;");
  if (!status.ok()) {
    Rollback(db);
  }
  return status;
}

}  // namespace fcp::client::confidentialcompute::internal
