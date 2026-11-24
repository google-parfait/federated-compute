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
#include "fcp/client/privatelogger/common/sqlite_util.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"
#include "fcp/testing/testing.h"
#include "sqlite3.h"

namespace fcp::client::privatelogger::internal {
namespace {

using ::fcp::client::diagnostics::DiagnosticCode;
using ::fcp::client::diagnostics::DiagnosticLogger;

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;

struct SqliteCloser {
  void operator()(sqlite3* db) const { sqlite3_close_v2(db); }
};
using ScopedSqlite = std::unique_ptr<sqlite3, SqliteCloser>;

ScopedSqlite Open(absl::string_view path) {
  absl::StatusOr<sqlite3*> db = OpenSqliteDatabase(path);
  EXPECT_THAT(db, IsOk());
  return ScopedSqlite(db.ok() ? *db : nullptr);
}

void Exec(sqlite3* db, const std::string& sql) {
  char* error = nullptr;
  ASSERT_EQ(sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &error), SQLITE_OK)
      << (error != nullptr ? error : "");
}

TEST(SqliteUtilTest, BindAndReadRoundTrip) {
  ScopedSqlite db = Open(":memory:");
  Exec(db.get(),
       "CREATE TABLE t (text_value TEXT, int_value INTEGER, "
       "blob_value BLOB);");
  struct Row {
    std::string text;
    int64_t integer;
    std::string blob;
  };
  const Row rows[] = {
      {"text", 42, "blob"},
      {"", std::numeric_limits<int64_t>::min(), ""},
      {std::string("a\0b", 3), std::numeric_limits<int64_t>::max(),
       std::string("\0\1\2", 3)},
  };
  for (const Row& row : rows) {
    FCP_ASSERT_OK_AND_ASSIGN(
        ScopedSqliteStmt insert,
        PrepareStatement(db.get(), "INSERT INTO t VALUES (?1, ?2, ?3);"));
    ASSERT_THAT(BindText(db.get(), insert.get(), 1, row.text), IsOk());
    ASSERT_THAT(BindInt64(db.get(), insert.get(), 2, row.integer), IsOk());
    ASSERT_THAT(BindBlob(db.get(), insert.get(), 3, row.blob), IsOk());
    ASSERT_EQ(sqlite3_step(insert.get()), SQLITE_DONE);
  }

  FCP_ASSERT_OK_AND_ASSIGN(
      ScopedSqliteStmt select,
      PrepareStatement(db.get(),
                       "SELECT text_value, int_value, blob_value "
                       "FROM t ORDER BY rowid;"));
  for (const Row& row : rows) {
    ASSERT_EQ(sqlite3_step(select.get()), SQLITE_ROW);
    EXPECT_EQ(ColumnText(select.get(), 0), row.text);
    EXPECT_EQ(sqlite3_column_int64(select.get(), 1), row.integer);
    EXPECT_EQ(ColumnBlob(select.get(), 2), row.blob);
  }
  EXPECT_EQ(sqlite3_step(select.get()), SQLITE_DONE);
}

TEST(SqliteUtilTest, NullColumnsReadAsEmpty) {
  ScopedSqlite db = Open(":memory:");
  FCP_ASSERT_OK_AND_ASSIGN(ScopedSqliteStmt select,
                           PrepareStatement(db.get(), "SELECT NULL, NULL;"));
  ASSERT_EQ(sqlite3_step(select.get()), SQLITE_ROW);
  EXPECT_EQ(ColumnText(select.get(), 0), "");
  EXPECT_EQ(ColumnBlob(select.get(), 1), "");
}

TEST(SqliteUtilTest, EmptyColumnsReadAsEmpty) {
  ScopedSqlite db = Open(":memory:");
  FCP_ASSERT_OK_AND_ASSIGN(ScopedSqliteStmt select,
                           PrepareStatement(db.get(), "SELECT '', X'';"));
  ASSERT_EQ(sqlite3_step(select.get()), SQLITE_ROW);
  EXPECT_EQ(ColumnText(select.get(), 0), "");
  EXPECT_EQ(ColumnBlob(select.get(), 1), "");
}

TEST(SqliteUtilTest, SizeCheckRejectsSizesAboveIntMax) {
  // The Bind functions and PrepareStatement() use this check; testing it
  // directly avoids allocating 2 GiB.
  const size_t int_max = static_cast<size_t>(std::numeric_limits<int>::max());
  EXPECT_THAT(CheckSqliteSize(0, "blob parameter"), IsOk());
  EXPECT_THAT(CheckSqliteSize(int_max, "blob parameter"), IsOk());
  EXPECT_THAT(CheckSqliteSize(int_max + 1, "blob parameter"),
              StatusIs(absl::StatusCode::kOutOfRange,
                       HasSubstr("blob parameter too large")));
}

// Edge cases of valid prefixes.
constexpr absl::string_view kValidTablePrefixes[] = {
    "", "abc", "ABC_123_", "_", "_1", "sqlite", "SQLite", "my_sqlite_"};

TEST(SqliteUtilTest, ValidateTablePrefix) {
  for (absl::string_view prefix : kValidTablePrefixes) {
    EXPECT_THAT(ValidateTablePrefix(prefix), IsOk()) << prefix;
  }
  for (absl::string_view prefix :
       {"a-b", "a b", "a;", "a.b", "a'b", "\xc3\xa4"}) {
    EXPECT_THAT(ValidateTablePrefix(prefix),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("only alphanumeric")))
        << prefix;
  }
  for (absl::string_view prefix : {"1", "1abc", "9_"}) {
    EXPECT_THAT(ValidateTablePrefix(prefix),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("must not start with a digit")))
        << prefix;
  }
  for (absl::string_view prefix : {"sqlite_", "SQLITE_", "Sqlite_x"}) {
    EXPECT_THAT(
        ValidateTablePrefix(prefix),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("sqlite_")))
        << prefix;
  }
}

TEST(SqliteUtilTest, ValidTablePrefixesGiveValidTableNames) {
  ScopedSqlite db = Open(":memory:");
  for (absl::string_view prefix : kValidTablePrefixes) {
    Exec(db.get(),
         absl::StrCat("CREATE TABLE ", prefix, "log_entries (x INTEGER);",
                      "DROP TABLE ", prefix, "log_entries;"));
  }
  // What the digit and "sqlite_" checks prevent.
  for (absl::string_view prefix : {"1", "sqlite_", "SQLITE_"}) {
    const std::string sql =
        absl::StrCat("CREATE TABLE ", prefix, "log_entries (x INTEGER);");
    EXPECT_NE(sqlite3_exec(db.get(), sql.c_str(), nullptr, nullptr, nullptr),
              SQLITE_OK)
        << prefix;
  }
}

TEST(SqliteUtilTest, SqliteErrorStatusIncludesTheSqliteMessage) {
  ScopedSqlite db = Open(":memory:");
  absl::StatusOr<ScopedSqliteStmt> stmt =
      PrepareStatement(db.get(), "NOT VALID SQL");
  EXPECT_THAT(stmt, StatusIs(absl::StatusCode::kInternal,
                             AllOf(HasSubstr("Failed to prepare"),
                                   HasSubstr("syntax error"))));
  EXPECT_THAT(SqliteErrorStatus(nullptr, "no connection"),
              StatusIs(absl::StatusCode::kInternal, "no connection"));
}

TEST(SqliteUtilTest, PrepareStatementRejectsSqlWithoutAStatement) {
  ScopedSqlite db = Open(":memory:");
  for (absl::string_view sql : {"", "   \n\t", "-- comment", "/* comment */"}) {
    EXPECT_THAT(PrepareStatement(db.get(), sql),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << "SQL: '" << sql << "'";
  }
}

TEST(SqliteUtilTest, OpenSqliteDatabaseSetsTheBusyTimeout) {
  ScopedSqlite db = Open(":memory:");
  FCP_ASSERT_OK_AND_ASSIGN(ScopedSqliteStmt stmt,
                           PrepareStatement(db.get(), "PRAGMA busy_timeout;"));
  ASSERT_EQ(sqlite3_step(stmt.get()), SQLITE_ROW);
  EXPECT_EQ(sqlite3_column_int64(stmt.get(), 0),
            absl::ToInt64Milliseconds(kSqliteBusyTimeout));
  EXPECT_EQ(absl::ToInt64Milliseconds(kSqliteBusyTimeout), 5000);
}

TEST(SqliteUtilTest, OpenSqliteDatabaseFailsForAnUnopenablePath) {
  EXPECT_THAT(OpenSqliteDatabase("/nonexistent/directory/db.sqlite"),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(SqliteUtilTest, OpenSqliteDatabaseRejectsPathsWithNul) {
  // Would otherwise open "<TempDir>/nul.db".
  const std::string path =
      absl::StrCat(testing::TempDir(), "/nul.db", std::string(1, '\0'), "x");
  EXPECT_THAT(OpenSqliteDatabase(path),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OpenSqliteDatabase(std::string(1, '\0')),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SqliteUtilTest, SecondConnectionWaitsForTheLock) {
  const std::string path = absl::StrCat(testing::TempDir(), "/busy.db");
  ScopedSqlite holder = Open(path);
  ScopedSqlite waiter = Open(path);
  Exec(holder.get(), "CREATE TABLE IF NOT EXISTS t (x INTEGER);");

  // `holder` keeps a write transaction open for a while; `waiter`'s write
  // waits for it instead of failing with SQLITE_BUSY.
  constexpr absl::Duration kHoldTime = absl::Milliseconds(100);
  Exec(holder.get(), "BEGIN IMMEDIATE;");
  std::thread release([&] {
    absl::SleepFor(kHoldTime);
    Exec(holder.get(), "COMMIT;");
  });
  const absl::Time start = absl::Now();
  Exec(waiter.get(), "INSERT INTO t VALUES (1);");
  const absl::Duration waited = absl::Now() - start;
  release.join();
  // It had to wait (the lock was held for kHoldTime after the thread
  // started), but not for the whole timeout.
  EXPECT_GE(waited, kHoldTime / 2);
  EXPECT_LT(waited, kSqliteBusyTimeout);
}

// Records the logged codes. Used from one thread only.
struct RecordingDiagnosticLogger : public DiagnosticLogger {
  void LogCode(DiagnosticCode code) override { codes.push_back(code); }

  std::vector<DiagnosticCode> codes;
};

TEST(SqliteUtilTest, CheckNoTransactionOpen) {
  ScopedSqlite db = Open(":memory:");
  RecordingDiagnosticLogger logger;
  EXPECT_THAT(CheckNoTransactionOpen(db.get(), logger), IsOk());
  EXPECT_THAT(logger.codes, IsEmpty());

  // Inside a transaction: refused, with the code logged exactly once.
  Exec(db.get(), "BEGIN;");
  EXPECT_THAT(CheckNoTransactionOpen(db.get(), logger),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("transaction is already open")));
  EXPECT_THAT(logger.codes,
              ElementsAre(DiagnosticCode::kErrorSqliteTransactionOpen));

  // After the commit it's fine again, and nothing more is logged.
  Exec(db.get(), "COMMIT;");
  EXPECT_THAT(CheckNoTransactionOpen(db.get(), logger), IsOk());
  EXPECT_THAT(logger.codes, SizeIs(1));
}

TEST(SqliteUtilTest, CheckNoTransactionOpenRejectsANullConnection) {
  RecordingDiagnosticLogger logger;
  EXPECT_THAT(CheckNoTransactionOpen(nullptr, logger),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(logger.codes, IsEmpty());
}

}  // namespace
}  // namespace fcp::client::privatelogger::internal
