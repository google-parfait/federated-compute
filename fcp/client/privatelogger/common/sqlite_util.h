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
#ifndef FCP_CLIENT_PRIVATELOGGER_COMMON_SQLITE_UTIL_H_
#define FCP_CLIENT_PRIVATELOGGER_COMMON_SQLITE_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"

struct sqlite3;
struct sqlite3_stmt;

// SQLite helpers shared by PrivateLogger's SQLite-backed stores. Not part of
// the public API.
namespace fcp::client::privatelogger::internal {

struct SqliteStmtDeleter {
  void operator()(sqlite3_stmt* stmt) const;
};

using ScopedSqliteStmt = std::unique_ptr<sqlite3_stmt, SqliteStmtDeleter>;

// How long a connection opened by OpenSqliteDatabase() waits for a lock held
// by another connection to the same file (e.g. a second store on the same
// file, or another process) before an operation fails with SQLITE_BUSY.
// Connections passed in by the client keep their own busy timeout.
inline constexpr absl::Duration kSqliteBusyTimeout = absl::Seconds(5);

// Opens (creating it if needed) the database at `path` read-write, in
// serialized threading mode (SQLITE_OPEN_FULLMUTEX), with kSqliteBusyTimeout.
// The caller owns the returned connection and closes it with
// sqlite3_close_v2(). Returns INVALID_ARGUMENT if `path` contains a NUL
// character.
absl::StatusOr<sqlite3*> OpenSqliteDatabase(absl::string_view path);

// Returns FAILED_PRECONDITION, and logs kErrorSqliteTransactionOpen to
// `logger`, if a transaction is open on `db` (sqlite3_get_autocommit() is 0),
// which on a connection shared with the client means the client's own. Call it
// before every write, holding the connection's locks, so that a write never
// joins (and a failure never rolls back) the client's transaction: the
// connection mutex serializes single SQLite calls, not the client's
// transactions. Returns INVALID_ARGUMENT, without logging, if `db` is null.
absl::Status CheckNoTransactionOpen(sqlite3* db,
                                    diagnostics::DiagnosticLogger& logger);

// Returns INVALID_ARGUMENT unless `table_prefix` can start an unquoted SQLite
// table name: it may contain only ASCII letters, digits and '_', must not
// start with a digit, and must not start with "sqlite_" (in any case), which
// SQLite reserves for its internal tables. The empty prefix is valid. The
// stores append a name starting with a lowercase letter to the prefix (e.g.
// "log_entries"), so every valid prefix gives a valid table name.
absl::Status ValidateTablePrefix(absl::string_view table_prefix);

// Returns an INTERNAL error with `message` and the connection's last SQLite
// error message (if `db` isn't null).
absl::Status SqliteErrorStatus(sqlite3* db, absl::string_view message);

// Returns OUT_OF_RANGE (naming `what`) if `size` doesn't fit the int sizes of
// SQLite's API, else OK. Used by PrepareStatement() and the Bind functions.
absl::Status CheckSqliteSize(size_t size, absl::string_view what);

// Prepares the first SQL statement in `sql`. Returns INVALID_ARGUMENT if `sql`
// contains no statement (it's empty or only whitespace or comments), and
// INTERNAL with SQLite's error message if preparing fails.
absl::StatusOr<ScopedSqliteStmt> PrepareStatement(sqlite3* db,
                                                  absl::string_view sql);

// Returns column `index` of the current row of `stmt` as text or bytes
// (empty for NULL).
std::string ColumnText(sqlite3_stmt* stmt, int index);
std::string ColumnBlob(sqlite3_stmt* stmt, int index);

// Bind `value` to the 1-based parameter `index` of `stmt`. Text and blob
// values are copied. Values larger than SQLite's int sizes allow fail with
// OUT_OF_RANGE instead of being truncated.
absl::Status BindText(sqlite3* db, sqlite3_stmt* stmt, int index,
                      absl::string_view value);
absl::Status BindBlob(sqlite3* db, sqlite3_stmt* stmt, int index,
                      absl::string_view value);
absl::Status BindInt64(sqlite3* db, sqlite3_stmt* stmt, int index,
                       int64_t value);

// Runs the SQL statements in `sql` (NUL-terminated, since sqlite3_exec() has
// no length parameter). Returns INTERNAL with SQLite's error message if one
// fails.
absl::Status ExecuteSql(sqlite3* db, const char* sql);

// A write transaction on a connection: Begin() starts it (BEGIN IMMEDIATE),
// Commit() commits it, and destroying it without a successful Commit() rolls it
// back. Use it while holding the connection's locks, after
// CheckNoTransactionOpen(), and destroy it before releasing them; statements
// prepared inside it should be destroyed first (declare them after it).
class ScopedSqliteTransaction {
 public:
  // Returns INTERNAL if BEGIN IMMEDIATE fails, e.g. because a transaction is
  // already open (which is left as it is) or the database is locked.
  static absl::StatusOr<ScopedSqliteTransaction> Begin(sqlite3* db);

  ScopedSqliteTransaction(ScopedSqliteTransaction&& other) noexcept;
  ScopedSqliteTransaction& operator=(ScopedSqliteTransaction&&) = delete;
  ScopedSqliteTransaction(const ScopedSqliteTransaction&) = delete;
  ScopedSqliteTransaction& operator=(const ScopedSqliteTransaction&) = delete;

  // Rolls back, ignoring errors, unless committed.
  ~ScopedSqliteTransaction();

  // Commits. If COMMIT fails, rolls back and returns its error (INTERNAL).
  // Either way the transaction is over: calling Commit() again returns
  // FAILED_PRECONDITION.
  absl::Status Commit();

 private:
  explicit ScopedSqliteTransaction(sqlite3* db) : db_(db) {}

  // Null once the transaction is over (or moved from).
  sqlite3* db_;
};

}  // namespace fcp::client::privatelogger::internal

#endif  // FCP_CLIENT_PRIVATELOGGER_COMMON_SQLITE_UTIL_H_
