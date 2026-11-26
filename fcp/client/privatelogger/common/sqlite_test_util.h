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
#ifndef FCP_CLIENT_PRIVATELOGGER_COMMON_SQLITE_TEST_UTIL_H_
#define FCP_CLIENT_PRIVATELOGGER_COMMON_SQLITE_TEST_UTIL_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"

struct sqlite3;

// SQLite helpers for tests. The functions report errors as test failures.
namespace fcp::client::privatelogger::internal {

struct SqliteCloser {
  void operator()(sqlite3* db) const;
};

// A connection closed (sqlite3_close_v2()) on destruction.
using ScopedSqlite = std::unique_ptr<sqlite3, SqliteCloser>;

// Opens the database at `path` with OpenSqliteDatabase(). Returns null (and
// records a non-fatal failure) if that fails, so callers should
// ASSERT_NE(db, nullptr) before using it.
ScopedSqlite OpenSqliteForTest(absl::string_view path);

// Opens a new, empty in-memory database with OpenSqliteDatabase(). Returns null
// like OpenSqliteForTest() if that fails.
ScopedSqlite OpenInMemory();

// Runs the SQL statements in `sql` with ExecuteSql().
void Exec(sqlite3* db, const std::string& sql);

// Whether a transaction is open on `db`.
bool TransactionOpen(sqlite3* db);

// Returns the number of rows in `table`, or -1 if counting fails.
int64_t CountRows(sqlite3* db, absl::string_view table);

}  // namespace fcp::client::privatelogger::internal

#endif  // FCP_CLIENT_PRIVATELOGGER_COMMON_SQLITE_TEST_UTIL_H_
