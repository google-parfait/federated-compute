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
#include "fcp/client/privatelogger/common/sqlite_test_util.h"

#include <cstdint>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "fcp/client/privatelogger/common/sqlite_util.h"
#include "sqlite3.h"

namespace fcp::client::privatelogger::internal {

using ::absl_testing::IsOk;

void SqliteCloser::operator()(sqlite3* db) const { sqlite3_close_v2(db); }

ScopedSqlite OpenSqliteForTest(absl::string_view path) {
  absl::StatusOr<sqlite3*> db = OpenSqliteDatabase(path);
  EXPECT_THAT(db, IsOk()) << path;
  return ScopedSqlite(db.ok() ? *db : nullptr);
}

ScopedSqlite OpenInMemory() { return OpenSqliteForTest(":memory:"); }

void Exec(sqlite3* db, const std::string& sql) {
  EXPECT_THAT(ExecuteSql(db, sql.c_str()), IsOk()) << sql;
}

bool TransactionOpen(sqlite3* db) { return sqlite3_get_autocommit(db) == 0; }

int64_t CountRows(sqlite3* db, absl::string_view table) {
  absl::StatusOr<ScopedSqliteStmt> stmt =
      PrepareStatement(db, absl::StrCat("SELECT COUNT(*) FROM ", table, ";"));
  EXPECT_THAT(stmt, IsOk());
  if (!stmt.ok() || sqlite3_step(stmt->get()) != SQLITE_ROW) {
    ADD_FAILURE() << "Counting the rows of " << table << " failed";
    return -1;
  }
  return sqlite3_column_int64(stmt->get(), 0);
}

}  // namespace fcp::client::privatelogger::internal
