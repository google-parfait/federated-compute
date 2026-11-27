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
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "fcp/client/confidentialcompute/storage/sqlite_test_util.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"
#include "fcp/client/diagnostics/test_diagnostic_loggers.h"
#include "fcp/testing/testing.h"
#include "sqlite3.h"

namespace fcp::client::confidentialcompute::internal {
namespace {

using ::fcp::client::diagnostics::DiagnosticCode;
using ::fcp::client::diagnostics::RecordingDiagnosticLogger;

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;

TEST(SqliteUtilTest, BindAndReadRoundTrip) {
  ScopedSqlite db = OpenInMemory();
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
  ScopedSqlite db = OpenInMemory();
  FCP_ASSERT_OK_AND_ASSIGN(ScopedSqliteStmt select,
                           PrepareStatement(db.get(), "SELECT NULL, NULL;"));
  ASSERT_EQ(sqlite3_step(select.get()), SQLITE_ROW);
  EXPECT_EQ(ColumnText(select.get(), 0), "");
  EXPECT_EQ(ColumnBlob(select.get(), 1), "");
}

TEST(SqliteUtilTest, EmptyColumnsReadAsEmpty) {
  ScopedSqlite db = OpenInMemory();
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
  ScopedSqlite db = OpenInMemory();
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
  ScopedSqlite db = OpenInMemory();
  absl::StatusOr<ScopedSqliteStmt> stmt =
      PrepareStatement(db.get(), "NOT VALID SQL");
  EXPECT_THAT(stmt, StatusIs(absl::StatusCode::kInternal,
                             AllOf(HasSubstr("Failed to prepare"),
                                   HasSubstr("syntax error"))));
  EXPECT_THAT(SqliteErrorStatus(nullptr, "no connection"),
              StatusIs(absl::StatusCode::kInternal, "no connection"));
}

TEST(SqliteUtilTest, PrepareStatementRejectsSqlWithoutAStatement) {
  ScopedSqlite db = OpenInMemory();
  for (absl::string_view sql : {"", "   \n\t", "-- comment", "/* comment */"}) {
    EXPECT_THAT(PrepareStatement(db.get(), sql),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << "SQL: '" << sql << "'";
  }
}

TEST(SqliteUtilTest, PrepareStatementRejectsANullConnection) {
  EXPECT_THAT(PrepareStatement(nullptr, "SELECT 1;"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SqliteUtilTest, OpenSqliteDatabaseSetsTheBusyTimeout) {
  ScopedSqlite db = OpenInMemory();
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
  ScopedSqlite holder = OpenSqliteForTest(path);
  ScopedSqlite waiter = OpenSqliteForTest(path);
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

TEST(SqliteUtilTest, CheckNoTransactionOpen) {
  ScopedSqlite db = OpenInMemory();
  RecordingDiagnosticLogger logger;
  EXPECT_THAT(CheckNoTransactionOpen(db.get(), logger), IsOk());
  EXPECT_THAT(logger.codes(), IsEmpty());

  // Inside a transaction: refused, with the code logged exactly once.
  Exec(db.get(), "BEGIN;");
  EXPECT_THAT(CheckNoTransactionOpen(db.get(), logger),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("transaction is already open")));
  EXPECT_THAT(logger.codes(),
              ElementsAre(DiagnosticCode::kErrorSqliteTransactionOpen));

  // After the commit it's fine again, and nothing more is logged.
  Exec(db.get(), "COMMIT;");
  EXPECT_THAT(CheckNoTransactionOpen(db.get(), logger), IsOk());
  EXPECT_THAT(logger.codes(), SizeIs(1));
}

TEST(SqliteUtilTest, CheckNoTransactionOpenRejectsANullConnection) {
  RecordingDiagnosticLogger logger;
  EXPECT_THAT(CheckNoTransactionOpen(nullptr, logger),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(logger.codes(), IsEmpty());
}

TEST(ScopedSqliteTransactionTest, CommitKeepsTheChanges) {
  ScopedSqlite db = OpenInMemory();
  Exec(db.get(), "CREATE TABLE t (x INTEGER);");
  {
    FCP_ASSERT_OK_AND_ASSIGN(ScopedSqliteTransaction transaction,
                             ScopedSqliteTransaction::Begin(db.get()));
    EXPECT_TRUE(TransactionOpen(db.get()));
    Exec(db.get(), "INSERT INTO t VALUES (1);");
    EXPECT_THAT(transaction.Commit(), IsOk());
    EXPECT_FALSE(TransactionOpen(db.get()));
    // The transaction is over.
    EXPECT_THAT(transaction.Commit(),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
  EXPECT_EQ(CountRows(db.get(), "t"), 1);
}

TEST(ScopedSqliteTransactionTest, DestructionWithoutCommitRollsBack) {
  ScopedSqlite db = OpenInMemory();
  Exec(db.get(), "CREATE TABLE t (x INTEGER);");
  {
    FCP_ASSERT_OK_AND_ASSIGN(ScopedSqliteTransaction transaction,
                             ScopedSqliteTransaction::Begin(db.get()));
    Exec(db.get(), "INSERT INTO t VALUES (1);");
    // A moved-from transaction does nothing on destruction.
    ScopedSqliteTransaction moved = std::move(transaction);
    EXPECT_TRUE(TransactionOpen(db.get()));
  }
  EXPECT_FALSE(TransactionOpen(db.get()));
  EXPECT_EQ(CountRows(db.get(), "t"), 0);
}

TEST(ScopedSqliteTransactionTest, FailedCommitRollsBack) {
  ScopedSqlite db = OpenInMemory();
  // A deferred foreign key violation makes COMMIT fail, leaving the
  // transaction open.
  Exec(db.get(),
       "PRAGMA foreign_keys = ON;"
       "CREATE TABLE parent (id INTEGER PRIMARY KEY);"
       "CREATE TABLE child (parent_id INTEGER REFERENCES parent (id) "
       "DEFERRABLE INITIALLY DEFERRED);");
  FCP_ASSERT_OK_AND_ASSIGN(ScopedSqliteTransaction transaction,
                           ScopedSqliteTransaction::Begin(db.get()));
  Exec(db.get(), "INSERT INTO child VALUES (1);");
  EXPECT_THAT(transaction.Commit(),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("FOREIGN KEY")));
  EXPECT_FALSE(TransactionOpen(db.get()));
  EXPECT_EQ(CountRows(db.get(), "child"), 0);
}

TEST(ScopedSqliteTransactionTest, BeginFailsInsideATransactionAndKeepsIt) {
  ScopedSqlite db = OpenInMemory();
  Exec(db.get(), "CREATE TABLE t (x INTEGER);");
  Exec(db.get(), "BEGIN; INSERT INTO t VALUES (1);");
  EXPECT_THAT(ScopedSqliteTransaction::Begin(db.get()),
              StatusIs(absl::StatusCode::kInternal));
  // The open transaction is left alone.
  EXPECT_TRUE(TransactionOpen(db.get()));
  Exec(db.get(), "COMMIT;");
  EXPECT_EQ(CountRows(db.get(), "t"), 1);
}

TEST(ScopedSqliteTransactionTest, BeginRejectsANullConnection) {
  EXPECT_THAT(ScopedSqliteTransaction::Begin(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// Creates table `t`, on which inserting x = 2 makes SQLite roll back the whole
// transaction (RAISE(ROLLBACK)), as it does by itself after some errors.
void CreateTableThatRollsBackOnTwo(sqlite3* db) {
  Exec(db,
       "CREATE TABLE t (x INTEGER);"
       "CREATE TRIGGER roll_back BEFORE INSERT ON t WHEN NEW.x = 2 "
       "BEGIN SELECT RAISE(ROLLBACK, 'rolled back by SQLite'); END;");
}

// A sqlite3_trace_v2() callback that counts the ROLLBACK statements run.
int CountRollbackStatements(unsigned type, void* context, void* statement,
                            void* sql) {
  if (type == SQLITE_TRACE_STMT &&
      absl::StartsWithIgnoreCase(static_cast<const char*>(sql), "ROLLBACK")) {
    ++*static_cast<int*>(context);
  }
  return 0;
}

TEST(ScopedSqliteTransactionTest, DestructionAfterSqliteRolledBackDoesNothing) {
  ScopedSqlite db = OpenInMemory();
  CreateTableThatRollsBackOnTwo(db.get());
  int rollback_statements = 0;
  ASSERT_EQ(sqlite3_trace_v2(db.get(), SQLITE_TRACE_STMT,
                             &CountRollbackStatements, &rollback_statements),
            SQLITE_OK);
  {
    FCP_ASSERT_OK_AND_ASSIGN(ScopedSqliteTransaction transaction,
                             ScopedSqliteTransaction::Begin(db.get()));
    Exec(db.get(), "INSERT INTO t VALUES (1);");
    EXPECT_THAT(ExecuteSql(db.get(), "INSERT INTO t VALUES (2);"),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("rolled back by SQLite")));
    EXPECT_FALSE(TransactionOpen(db.get()));
  }
  // The destructor saw that no transaction was open and ran no ROLLBACK.
  EXPECT_EQ(rollback_statements, 0);
  EXPECT_FALSE(TransactionOpen(db.get()));
  EXPECT_EQ(CountRows(db.get(), "t"), 0);

  // The counting works: a transaction still open is rolled back.
  {
    FCP_ASSERT_OK_AND_ASSIGN(ScopedSqliteTransaction transaction,
                             ScopedSqliteTransaction::Begin(db.get()));
  }
  EXPECT_EQ(rollback_statements, 1);
  ASSERT_EQ(sqlite3_trace_v2(db.get(), 0, nullptr, nullptr), SQLITE_OK);
}

TEST(ScopedSqliteTransactionTest, CommitAfterSqliteRolledBackFails) {
  ScopedSqlite db = OpenInMemory();
  CreateTableThatRollsBackOnTwo(db.get());
  FCP_ASSERT_OK_AND_ASSIGN(ScopedSqliteTransaction transaction,
                           ScopedSqliteTransaction::Begin(db.get()));
  Exec(db.get(), "INSERT INTO t VALUES (1);");
  ASSERT_THAT(ExecuteSql(db.get(), "INSERT INTO t VALUES (2);"),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(transaction.Commit(), StatusIs(absl::StatusCode::kInternal));
  EXPECT_FALSE(TransactionOpen(db.get()));
  EXPECT_EQ(CountRows(db.get(), "t"), 0);
}

TEST(SqliteUtilTest, ExecuteSqlRunsAllStatementsAndReportsErrors) {
  ScopedSqlite db = OpenInMemory();
  EXPECT_THAT(ExecuteSql(db.get(),
                         "CREATE TABLE t (x INTEGER);"
                         "INSERT INTO t VALUES (1);"
                         "INSERT INTO t VALUES (2);"),
              IsOk());
  EXPECT_EQ(CountRows(db.get(), "t"), 2);
  EXPECT_THAT(ExecuteSql(db.get(), "NOT VALID SQL"),
              StatusIs(absl::StatusCode::kInternal,
                       AllOf(HasSubstr("SQLite execution failed"),
                             HasSubstr("syntax error"))));
  EXPECT_THAT(ExecuteSql(nullptr, "SELECT 1;"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace fcp::client::confidentialcompute::internal
