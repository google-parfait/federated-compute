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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"
#include "fcp/client/diagnostics/test_diagnostic_loggers.h"
#include "fcp/client/privatelogger/common/log_entry.h"
#include "fcp/client/privatelogger/common/sqlite_test_util.h"
#include "fcp/client/privatelogger/log/log_database.h"
#include "fcp/testing/testing.h"
#include "sqlite3.h"

namespace fcp::client::privatelogger {
namespace {

using ::fcp::client::diagnostics::DiagnosticCode;
using ::fcp::client::diagnostics::NoDiagnostics;
using ::fcp::client::diagnostics::RecordingDiagnosticLogger;
using ::fcp::client::privatelogger::internal::CountRows;
using ::fcp::client::privatelogger::internal::Exec;
using ::fcp::client::privatelogger::internal::OpenInMemory;
using ::fcp::client::privatelogger::internal::ScopedSqlite;
using ::fcp::client::privatelogger::internal::TransactionOpen;

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;

std::vector<std::string> Values(const SelectedLogEntries& selected) {
  std::vector<std::string> values;
  for (const StoredLogEntry& entry : selected.entries) {
    values.push_back(entry.entry.value);
  }
  return values;
}

std::vector<int64_t> Ids(const SelectedLogEntries& selected) {
  std::vector<int64_t> ids;
  for (const StoredLogEntry& entry : selected.entries) {
    ids.push_back(entry.id);
  }
  return ids;
}

class SqliteLogDatabaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FCP_ASSERT_OK_AND_ASSIGN(db_, SqliteLogDatabase::Create(":memory:"));
  }

  void Insert(absl::string_view value, absl::Time timestamp) {
    ASSERT_THAT(db_->InsertEntry(LogEntry{.timestamp = timestamp,
                                          .value = std::string(value)},
                                 NoDiagnostics()),
                IsOk());
  }

  absl::StatusOr<SelectedLogEntries> Select(
      absl::string_view set, int64_t max_bytes,
      absl::Time cutoff = absl::InfinitePast()) {
    return db_->GetAndMarkUncontributedEntries(set, cutoff, max_bytes,
                                               NoDiagnostics());
  }

  const absl::Time now_ = absl::FromUnixSeconds(1700000000);
  std::unique_ptr<SqliteLogDatabase> db_;
};

TEST_F(SqliteLogDatabaseTest, SelectsOldestFirstWithinTheByteBudget) {
  // Inserted out of timestamp order; each value is 2 bytes.
  Insert("bb", now_ - absl::Minutes(2));
  Insert("cc", now_ - absl::Minutes(1));
  Insert("aa", now_ - absl::Minutes(3));
  Insert("dd", now_);

  // 5 bytes fit two entries (4 bytes), not three (6).
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/5));
  EXPECT_THAT(Values(selected), ElementsAre("aa", "bb"));
  EXPECT_TRUE(selected.byte_limit_reached);

  // The rest stays uncontributed for the next call. Exactly filling the
  // budget is fine, and nothing is left.
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set", /*max_bytes=*/4));
  EXPECT_THAT(Values(selected), ElementsAre("cc", "dd"));
  EXPECT_FALSE(selected.byte_limit_reached);
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set", /*max_bytes=*/4));
  EXPECT_THAT(selected.entries, IsEmpty());
  EXPECT_FALSE(selected.byte_limit_reached);
}

TEST_F(SqliteLogDatabaseTest, TakesAtLeastOneEntryEvenIfLargerThanTheBudget) {
  Insert("larger than the budget", now_ - absl::Minutes(1));
  Insert("x", now_);
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/3));
  EXPECT_THAT(Values(selected), ElementsAre("larger than the budget"));
  EXPECT_TRUE(selected.byte_limit_reached);
  // The next call gets the rest.
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set", /*max_bytes=*/3));
  EXPECT_THAT(Values(selected), ElementsAre("x"));
  EXPECT_FALSE(selected.byte_limit_reached);
}

TEST_F(SqliteLogDatabaseTest, StopsAtTheFirstEntryThatDoesNotFit) {
  // Strictly oldest first: a smaller, newer entry doesn't jump the queue.
  Insert("aa", now_ - absl::Minutes(3));
  Insert("bbbbbbbb", now_ - absl::Minutes(2));
  Insert("c", now_ - absl::Minutes(1));
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/4));
  EXPECT_THAT(Values(selected), ElementsAre("aa"));
  EXPECT_TRUE(selected.byte_limit_reached);
}

TEST_F(SqliteLogDatabaseTest, EmptyValuesCountAsZeroBytes) {
  Insert("", now_ - absl::Minutes(2));
  Insert("", now_ - absl::Minutes(1));
  Insert("a", now_);
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/1));
  EXPECT_THAT(Values(selected), ElementsAre("", "", "a"));
  EXPECT_FALSE(selected.byte_limit_reached);
}

TEST_F(SqliteLogDatabaseTest, ValuesRoundTripExactly) {
  // Binary values: embedded NUL bytes, bytes that aren't valid UTF-8, and a
  // value larger than SQLite's default page size.
  const std::string with_nul("a\0b", 3);
  const std::string not_utf8 = "\xff\xfe\x80";
  const std::string large(100000, 'x');
  Insert(with_nul, now_ - absl::Minutes(2));
  Insert(not_utf8, now_ - absl::Minutes(1));
  Insert(large, now_);
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/1000000));
  EXPECT_THAT(Values(selected), ElementsAre(with_nul, not_utf8, large));
}

TEST_F(SqliteLogDatabaseTest, EqualTimestampsAreSelectedInInsertionOrder) {
  // Same size, and inserted in reverse lexical order, so that neither size nor
  // value can explain the order.
  Insert("b", now_);
  Insert("a", now_);
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set1", /*max_bytes=*/1));
  EXPECT_THAT(Values(selected), ElementsAre("b"));
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set2", /*max_bytes=*/10));
  EXPECT_THAT(Values(selected), ElementsAre("b", "a"));
}

TEST_F(SqliteLogDatabaseTest, BudgetAppliesPerWorkloadSet) {
  Insert("a", now_ - absl::Minutes(1));
  Insert("b", now_);
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set1", /*max_bytes=*/1));
  EXPECT_THAT(Values(selected), ElementsAre("a"));
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set2", /*max_bytes=*/10));
  EXPECT_THAT(Values(selected), ElementsAre("a", "b"));
}

TEST_F(SqliteLogDatabaseTest, RolledBackEntriesAreSelectedAgainFirst) {
  Insert("a", now_ - absl::Minutes(1));
  Insert("b", now_);
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/1));
  ASSERT_THAT(Values(selected), ElementsAre("a"));
  ASSERT_THAT(db_->DeleteContributions("set", {selected.entries[0].id},
                                       NoDiagnostics()),
              IsOk());
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set", /*max_bytes=*/1));
  EXPECT_THAT(Values(selected), ElementsAre("a"));
}

TEST_F(SqliteLogDatabaseTest, DeleteContributionsUnmarksAllGivenEntries) {
  Insert("a", now_ - absl::Minutes(2));
  Insert("b", now_ - absl::Minutes(1));
  Insert("c", now_);
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/100));
  ASSERT_THAT(Values(selected), ElementsAre("a", "b", "c"));
  // Unmark a and c, keep b marked.
  ASSERT_THAT(db_->DeleteContributions(
                  "set", {selected.entries[0].id, selected.entries[2].id},
                  NoDiagnostics()),
              IsOk());
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set", /*max_bytes=*/100));
  EXPECT_THAT(Values(selected), ElementsAre("a", "c"));
}

TEST_F(SqliteLogDatabaseTest, DeleteContributionsWithNoIdsDoesNothing) {
  Insert("a", now_);
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/100));
  ASSERT_THAT(Values(selected), ElementsAre("a"));
  ASSERT_THAT(db_->DeleteContributions("set", {}, NoDiagnostics()), IsOk());
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set", /*max_bytes=*/100));
  EXPECT_THAT(Values(selected), IsEmpty());
}

TEST_F(SqliteLogDatabaseTest, DeleteContributionsOnlyAffectsItsWorkloadSet) {
  Insert("a", now_);
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set1", /*max_bytes=*/100));
  ASSERT_THAT(Values(selected), ElementsAre("a"));
  const std::vector<int64_t> ids = Ids(selected);
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set2", /*max_bytes=*/100));
  ASSERT_THAT(Values(selected), ElementsAre("a"));

  ASSERT_THAT(db_->DeleteContributions("set1", ids, NoDiagnostics()), IsOk());
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set1", /*max_bytes=*/100));
  EXPECT_THAT(Values(selected), ElementsAre("a"));
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set2", /*max_bytes=*/100));
  EXPECT_THAT(Values(selected), IsEmpty());
}

TEST_F(SqliteLogDatabaseTest, CutoffExcludesEntriesAtOrBeforeIt) {
  const absl::Time cutoff = now_ - absl::Hours(1);
  Insert("before", cutoff - absl::Milliseconds(1));
  Insert("at", cutoff);
  Insert("after", cutoff + absl::Milliseconds(1));
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/100, cutoff));
  EXPECT_THAT(Values(selected), ElementsAre("after"));
  EXPECT_FALSE(selected.byte_limit_reached);
}

TEST_F(SqliteLogDatabaseTest, DeletesEntriesAtOrBeforeTheCutoff) {
  const absl::Time cutoff = now_ - absl::Hours(1);
  Insert("before", cutoff - absl::Milliseconds(1));
  Insert("at", cutoff);
  Insert("after", cutoff + absl::Milliseconds(1));
  // Contributed entries are deleted as well.
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/100));
  ASSERT_THAT(selected.entries, SizeIs(3));
  ASSERT_THAT(db_->DeleteEntriesOlderThan(cutoff, NoDiagnostics()), IsOk());

  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("other", /*max_bytes=*/100));
  EXPECT_THAT(Values(selected), ElementsAre("after"));
}

TEST_F(SqliteLogDatabaseTest, CleanupCountsNeverContributedEntries) {
  const absl::Time cutoff = now_ - absl::Hours(1);
  Insert("contributed", cutoff - absl::Milliseconds(2));
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/1));
  ASSERT_THAT(Values(selected), ElementsAre("contributed"));
  Insert("never contributed 1", cutoff - absl::Milliseconds(1));
  Insert("never contributed 2", cutoff);
  Insert("kept", cutoff + absl::Milliseconds(1));
  EXPECT_THAT(db_->DeleteEntriesOlderThan(cutoff, NoDiagnostics()),
              IsOkAndHolds(2));
  // Nothing left to delete.
  EXPECT_THAT(db_->DeleteEntriesOlderThan(cutoff, NoDiagnostics()),
              IsOkAndHolds(0));
}

TEST_F(SqliteLogDatabaseTest, CountsEntriesNewerThanTheCutoff) {
  const absl::Time cutoff = now_ - absl::Hours(1);
  EXPECT_THAT(db_->CountEntriesNewerThan(cutoff), IsOkAndHolds(0));
  Insert("at", cutoff);
  Insert("after", cutoff + absl::Milliseconds(1));
  Insert("later", now_);
  // Contributed entries still count: other workload sets may take them.
  ASSERT_THAT(Select("set", /*max_bytes=*/100), IsOk());
  EXPECT_THAT(db_->CountEntriesNewerThan(cutoff), IsOkAndHolds(2));
}

TEST_F(SqliteLogDatabaseTest, NonPositiveMaxBytesIsAnError) {
  Insert("a", now_);
  EXPECT_THAT(Select("set", /*max_bytes=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Nothing was marked.
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected,
                           Select("set", /*max_bytes=*/1));
  EXPECT_THAT(Values(selected), ElementsAre("a"));
}

// The prefix rules are tested in sqlite_util_test; this only checks that both
// Create() overloads validate the prefix.
TEST(SqliteLogDatabaseCreateTest, ValidatesTablePrefix) {
  EXPECT_THAT(SqliteLogDatabase::Create(":memory:", "bad-prefix"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  ScopedSqlite connection = OpenInMemory();
  EXPECT_THAT(SqliteLogDatabase::Create(connection.get(), "bad-prefix"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SqliteLogDatabaseCreateTest, RejectsNullConnection) {
  EXPECT_THAT(SqliteLogDatabase::Create(static_cast<sqlite3*>(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SqliteLogDatabaseCreateTest,
     InstancesWithDistinctPrefixesShareAConnection) {
  ScopedSqlite connection = OpenInMemory();
  FCP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SqliteLogDatabase> first,
                           SqliteLogDatabase::Create(connection.get(), "a_"));
  FCP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SqliteLogDatabase> second,
                           SqliteLogDatabase::Create(connection.get(), "b_"));
  const absl::Time now = absl::FromUnixSeconds(1700000000);
  ASSERT_THAT(first->InsertEntry(LogEntry{.timestamp = now, .value = "a"},
                                 NoDiagnostics()),
              IsOk());
  EXPECT_THAT(first->CountEntriesNewerThan(absl::InfinitePast()),
              IsOkAndHolds(1));
  EXPECT_THAT(second->CountEntriesNewerThan(absl::InfinitePast()),
              IsOkAndHolds(0));
}

TEST(SqliteLogDatabaseSharedConnectionTest,
     WritesFailWhileTheClientHasATransactionOpen) {
  ScopedSqlite connection = OpenInMemory();
  FCP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SqliteLogDatabase> db,
                           SqliteLogDatabase::Create(connection.get()));
  const absl::Time now = absl::FromUnixSeconds(1700000000);
  ASSERT_THAT(db->InsertEntry(LogEntry{.timestamp = now, .value = "a"},
                              NoDiagnostics()),
              IsOk());

  Exec(connection.get(), "BEGIN;");
  RecordingDiagnosticLogger logger;
  // Each refused write logs the code exactly once.
  EXPECT_THAT(db->InsertEntry(LogEntry{.timestamp = now, .value = "b"}, logger),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(logger.codes(), SizeIs(1));
  EXPECT_THAT(db->GetAndMarkUncontributedEntries("set", absl::InfinitePast(),
                                                 /*max_bytes=*/100, logger),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(logger.codes(), SizeIs(2));
  EXPECT_THAT(db->DeleteContributions("set", {1}, logger),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(logger.codes(), SizeIs(3));
  EXPECT_THAT(db->DeleteEntriesOlderThan(absl::InfiniteFuture(), logger),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(logger.codes(),
              ElementsAre(DiagnosticCode::kErrorSqliteTransactionOpen,
                          DiagnosticCode::kErrorSqliteTransactionOpen,
                          DiagnosticCode::kErrorSqliteTransactionOpen,
                          DiagnosticCode::kErrorSqliteTransactionOpen));
  // The schema setup is a write too.
  EXPECT_THAT(SqliteLogDatabase::Create(connection.get(), "other_"),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  // Reads are fine.
  EXPECT_THAT(db->CountEntriesNewerThan(absl::InfinitePast()), IsOkAndHolds(1));
  Exec(connection.get(), "COMMIT;");

  // Nothing was inserted, marked or deleted, and writes work again.
  FCP_ASSERT_OK_AND_ASSIGN(
      SelectedLogEntries selected,
      db->GetAndMarkUncontributedEntries("set", absl::InfinitePast(),
                                         /*max_bytes=*/100, logger));
  ASSERT_THAT(selected.entries, SizeIs(1));
  EXPECT_EQ(selected.entries[0].entry.value, "a");
  EXPECT_THAT(logger.codes(), SizeIs(4));
}

TEST(SqliteLogDatabaseCleanupTest, DeletesOnlyTheExpiredContributions) {
  ScopedSqlite connection = OpenInMemory();
  FCP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SqliteLogDatabase> db,
                           SqliteLogDatabase::Create(connection.get(), "t_"));
  const absl::Time now = absl::FromUnixSeconds(1700000000);
  const absl::Time cutoff = now - absl::Hours(1);
  ASSERT_THAT(db->InsertEntry(LogEntry{.timestamp = cutoff, .value = "expired"},
                              NoDiagnostics()),
              IsOk());
  ASSERT_THAT(db->InsertEntry(LogEntry{.timestamp = now, .value = "kept"},
                              NoDiagnostics()),
              IsOk());
  ASSERT_THAT(
      db->GetAndMarkUncontributedEntries("set", absl::InfinitePast(),
                                         /*max_bytes=*/100, NoDiagnostics()),
      IsOkAndHolds(Field(&SelectedLogEntries::entries, SizeIs(2))));

  EXPECT_THAT(db->DeleteEntriesOlderThan(cutoff, NoDiagnostics()),
              IsOkAndHolds(0));

  // Only the retained entry's contribution record is left, so it isn't
  // selected again for that set.
  EXPECT_EQ(CountRows(connection.get(), "t_contributed_entries"), 1);
  FCP_ASSERT_OK_AND_ASSIGN(
      SelectedLogEntries selected,
      db->GetAndMarkUncontributedEntries("set", absl::InfinitePast(),
                                         /*max_bytes=*/100, NoDiagnostics()));
  EXPECT_THAT(selected.entries, IsEmpty());
  FCP_ASSERT_OK_AND_ASSIGN(selected, db->GetAndMarkUncontributedEntries(
                                         "other", absl::InfinitePast(),
                                         /*max_bytes=*/100, NoDiagnostics()));
  EXPECT_THAT(Values(selected), ElementsAre("kept"));
}

// Makes one statement fail in the middle of a write transaction, using a
// trigger on the shared connection, to check that the transaction is rolled
// back.
class SqliteLogDatabaseRollbackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FCP_ASSERT_OK_AND_ASSIGN(
        db_, SqliteLogDatabase::Create(connection_.get(), "t_"));
  }

  void Insert(absl::string_view value, absl::Time timestamp) {
    ASSERT_THAT(db_->InsertEntry(LogEntry{.timestamp = timestamp,
                                          .value = std::string(value)},
                                 NoDiagnostics()),
                IsOk());
  }

  absl::StatusOr<SelectedLogEntries> Select(absl::string_view set) {
    return db_->GetAndMarkUncontributedEntries(set, absl::InfinitePast(),
                                               /*max_bytes=*/100,
                                               NoDiagnostics());
  }

  // Makes every `event` (INSERT or DELETE) on `table` fail, or only those
  // for which the SQL condition `when` (on NEW or OLD) holds.
  void FailOn(absl::string_view event, absl::string_view table,
              absl::string_view when = "") {
    const std::string sql =
        "CREATE TRIGGER fail BEFORE " + std::string(event) + " ON " +
        std::string(table) +
        (when.empty() ? "" : " WHEN " + std::string(when)) +
        " BEGIN SELECT RAISE(ABORT, 'injected failure'); END;";
    Exec(connection_.get(), sql);
  }

  void StopFailing() { Exec(connection_.get(), "DROP TRIGGER fail;"); }

  // The id of the last entry Insert() added.
  int64_t LastInsertedId() {
    return sqlite3_last_insert_rowid(connection_.get());
  }

  const absl::Time now_ = absl::FromUnixSeconds(1700000000);
  ScopedSqlite connection_ = OpenInMemory();
  std::unique_ptr<SqliteLogDatabase> db_;
};

TEST_F(SqliteLogDatabaseRollbackTest, FailedMarkingRollsBack) {
  Insert("a", now_ - absl::Minutes(1));
  Insert("b", now_);
  FailOn("INSERT", "t_contributed_entries",
         "NEW.entry_id = " + std::to_string(LastInsertedId()));
  // The selection works and "a" is marked; marking "b" fails.
  EXPECT_THAT(Select("set"), StatusIs(absl::StatusCode::kInternal,
                                      HasSubstr("injected failure")));
  EXPECT_FALSE(TransactionOpen(connection_.get()));
  EXPECT_THAT(db_->CountEntriesNewerThan(absl::InfinitePast()),
              IsOkAndHolds(2));

  // Nothing was marked.
  StopFailing();
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected, Select("set"));
  EXPECT_THAT(Values(selected), ElementsAre("a", "b"));
}

TEST_F(SqliteLogDatabaseRollbackTest, FailedDeleteContributionsRollsBack) {
  Insert("a", now_ - absl::Minutes(1));
  Insert("b", now_);
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected, Select("set"));
  ASSERT_THAT(selected.entries, SizeIs(2));
  const int64_t first_id = selected.entries[0].id;
  const int64_t second_id = selected.entries[1].id;
  // Deleting the first record works; deleting the second fails.
  FailOn("DELETE", "t_contributed_entries",
         "OLD.entry_id = " + std::to_string(second_id));
  EXPECT_THAT(
      db_->DeleteContributions("set", {first_id, second_id}, NoDiagnostics()),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("injected failure")));
  EXPECT_FALSE(TransactionOpen(connection_.get()));

  // Both records are still there.
  StopFailing();
  FCP_ASSERT_OK_AND_ASSIGN(selected, Select("set"));
  EXPECT_THAT(selected.entries, IsEmpty());
}

TEST_F(SqliteLogDatabaseRollbackTest, FailedCleanupRollsBack) {
  const absl::Time cutoff = now_ - absl::Hours(1);
  Insert("expired", cutoff);
  Insert("kept", now_);
  // Gives the expired entry a contribution record. Cleanup deletes it first,
  // then fails deleting the entry.
  ASSERT_THAT(Select("set"), IsOk());
  FailOn("DELETE", "t_log_entries");
  EXPECT_THAT(
      db_->DeleteEntriesOlderThan(cutoff, NoDiagnostics()),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("injected failure")));
  EXPECT_FALSE(TransactionOpen(connection_.get()));

  // The expired entry and its contribution record are still there.
  EXPECT_THAT(db_->CountEntriesNewerThan(absl::InfinitePast()),
              IsOkAndHolds(2));
  FCP_ASSERT_OK_AND_ASSIGN(SelectedLogEntries selected, Select("set"));
  EXPECT_THAT(selected.entries, IsEmpty());

  StopFailing();
  EXPECT_THAT(db_->DeleteEntriesOlderThan(cutoff, NoDiagnostics()),
              IsOkAndHolds(0));
  EXPECT_THAT(db_->CountEntriesNewerThan(absl::InfinitePast()),
              IsOkAndHolds(1));
}

}  // namespace
}  // namespace fcp::client::privatelogger
