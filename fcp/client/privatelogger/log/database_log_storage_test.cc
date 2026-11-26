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
#include "fcp/client/privatelogger/log/database_log_storage.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "fcp/client/diagnostics/diagnostic_logger.h"
#include "fcp/client/diagnostics/test_diagnostic_loggers.h"
#include "fcp/client/privatelogger/common/log_entry.h"
#include "fcp/client/privatelogger/common/sqlite_test_util.h"
#include "fcp/client/privatelogger/log/log_database.h"
#include "fcp/client/privatelogger/log/log_storage.h"
#include "fcp/client/privatelogger/log/sqlite_log_database.h"
#include "fcp/testing/testing.h"

namespace fcp::client::privatelogger {
namespace {

using ::fcp::client::diagnostics::DiagnosticCode;
using ::fcp::client::diagnostics::DiagnosticLogger;
using ::fcp::client::diagnostics::NoDiagnostics;
using ::fcp::client::diagnostics::RecordingDiagnosticLogger;
using ::fcp::client::privatelogger::internal::Exec;
using ::fcp::client::privatelogger::internal::OpenInMemory;
using ::fcp::client::privatelogger::internal::ScopedSqlite;

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;

std::vector<std::string> Values(const UploadBatch& batch) {
  std::vector<std::string> values;
  for (const LogEntry& entry : batch.entries()) {
    values.push_back(entry.value);
  }
  return values;
}

constexpr absl::string_view kSet = "set";
const absl::Time kNow = absl::FromUnixSeconds(1700000000);
const absl::Time kCutoff = kNow - absl::Hours(1);

class DatabaseLogStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FCP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SqliteLogDatabase> database,
                             SqliteLogDatabase::Create(":memory:"));
    storage_ = std::make_unique<DatabaseLogStorage>(std::move(database));
  }

  void Log(absl::string_view value, absl::Time timestamp) {
    ASSERT_THAT(storage_->Log(LogEntry{.timestamp = timestamp,
                                       .value = std::string(value)},
                              NoDiagnostics()),
                IsOk());
  }

  absl::StatusOr<UploadBatch> GetLogs(absl::string_view set = kSet,
                                      int64_t max_bytes = 100) {
    return storage_->GetLogsToUpload(set, kCutoff, max_bytes, NoDiagnostics());
  }

  std::unique_ptr<DatabaseLogStorage> storage_;
};

TEST_F(DatabaseLogStorageTest, GetLogsToUploadReturnsTheLoggedEntries) {
  Log("b", kNow - absl::Minutes(1));
  Log("a", kNow - absl::Minutes(2));
  Log("expired", kCutoff);
  FCP_ASSERT_OK_AND_ASSIGN(UploadBatch batch, GetLogs());
  EXPECT_FALSE(batch.empty());
  EXPECT_FALSE(batch.byte_limit_reached());
  ASSERT_THAT(batch.entries(), SizeIs(2));
  EXPECT_EQ(batch.entries()[0].timestamp, kNow - absl::Minutes(2));
  EXPECT_EQ(batch.entries()[0].value, "a");
  EXPECT_EQ(batch.entries()[1].timestamp, kNow - absl::Minutes(1));
  EXPECT_EQ(batch.entries()[1].value, "b");
}

TEST_F(DatabaseLogStorageTest, EmptyStorageGivesAnEmptyBatch) {
  FCP_ASSERT_OK_AND_ASSIGN(UploadBatch batch, GetLogs());
  EXPECT_TRUE(batch.empty());
  EXPECT_FALSE(batch.byte_limit_reached());
  EXPECT_THAT(batch.RollBack(NoDiagnostics()), IsOk());
}

TEST_F(DatabaseLogStorageTest, ByteLimitLeavesTheRestForTheNextBatch) {
  Log("aa", kNow - absl::Minutes(2));
  Log("bb", kNow - absl::Minutes(1));
  FCP_ASSERT_OK_AND_ASSIGN(UploadBatch batch, GetLogs(kSet, /*max_bytes=*/3));
  EXPECT_THAT(Values(batch), ElementsAre("aa"));
  EXPECT_TRUE(batch.byte_limit_reached());
  FCP_ASSERT_OK_AND_ASSIGN(batch, GetLogs(kSet, /*max_bytes=*/3));
  EXPECT_THAT(Values(batch), ElementsAre("bb"));
  EXPECT_FALSE(batch.byte_limit_reached());
}

TEST_F(DatabaseLogStorageTest, DroppedBatchStaysContributed) {
  Log("a", kNow);
  {
    FCP_ASSERT_OK_AND_ASSIGN(UploadBatch batch, GetLogs());
    EXPECT_THAT(Values(batch), ElementsAre("a"));
  }
  FCP_ASSERT_OK_AND_ASSIGN(UploadBatch batch, GetLogs());
  EXPECT_TRUE(batch.empty());
  // Other workload sets still get it.
  FCP_ASSERT_OK_AND_ASSIGN(batch, GetLogs("other"));
  EXPECT_THAT(Values(batch), ElementsAre("a"));
}

TEST_F(DatabaseLogStorageTest, RolledBackBatchIsSelectedAgain) {
  Log("a", kNow - absl::Minutes(1));
  Log("b", kNow);
  FCP_ASSERT_OK_AND_ASSIGN(UploadBatch batch, GetLogs());
  // A batch survives moves, including its rollback.
  UploadBatch moved = std::move(batch);
  ASSERT_THAT(Values(moved), ElementsAre("a", "b"));
  ASSERT_THAT(moved.RollBack(NoDiagnostics()), IsOk());
  // Rolling back again does nothing.
  EXPECT_THAT(moved.RollBack(NoDiagnostics()), IsOk());

  FCP_ASSERT_OK_AND_ASSIGN(batch, GetLogs());
  EXPECT_THAT(Values(batch), ElementsAre("a", "b"));
}

TEST_F(DatabaseLogStorageTest, RollbackOnlyAffectsItsWorkloadSet) {
  Log("a", kNow);
  FCP_ASSERT_OK_AND_ASSIGN(UploadBatch first, GetLogs("first"));
  FCP_ASSERT_OK_AND_ASSIGN(UploadBatch second, GetLogs("second"));
  ASSERT_THAT(first.RollBack(NoDiagnostics()), IsOk());
  FCP_ASSERT_OK_AND_ASSIGN(UploadBatch batch, GetLogs("first"));
  EXPECT_THAT(Values(batch), ElementsAre("a"));
  FCP_ASSERT_OK_AND_ASSIGN(batch, GetLogs("second"));
  EXPECT_TRUE(batch.empty());
}

TEST_F(DatabaseLogStorageTest, NonPositiveMaxBytesIsAnError) {
  Log("a", kNow);
  EXPECT_THAT(GetLogs(kSet, /*max_bytes=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetLogs(kSet, /*max_bytes=*/-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(DatabaseLogStorageTest, RollbackAfterCleanUpDeletedTheEntriesIsOk) {
  Log("a", kNow);
  FCP_ASSERT_OK_AND_ASSIGN(UploadBatch batch, GetLogs());
  ASSERT_THAT(Values(batch), ElementsAre("a"));
  ASSERT_THAT(storage_->CleanUp(CleanupMode::kImmediate, kNow, NoDiagnostics()),
              IsOkAndHolds(0));
  // There's nothing left to unmark.
  EXPECT_THAT(batch.RollBack(NoDiagnostics()), IsOk());
  EXPECT_THAT(storage_->CountPendingEntries(absl::InfinitePast()),
              IsOkAndHolds(0));
}

TEST_F(DatabaseLogStorageTest, CountPendingEntriesCountsTheRetentionWindow) {
  EXPECT_THAT(storage_->CountPendingEntries(kCutoff), IsOkAndHolds(0));
  Log("expired", kCutoff);
  Log("a", kNow);
  Log("b", kNow);
  // Contributed entries still count: other workload sets may take them.
  ASSERT_THAT(GetLogs(), IsOk());
  EXPECT_THAT(storage_->CountPendingEntries(kCutoff), IsOkAndHolds(2));
}

TEST_F(DatabaseLogStorageTest, ImmediateCleanUpDeletesExpiredEntries) {
  Log("contributed", kCutoff - absl::Minutes(1));
  FCP_ASSERT_OK_AND_ASSIGN(
      UploadBatch batch,
      storage_->GetLogsToUpload(kSet, absl::InfinitePast(), /*max_bytes=*/1,
                                NoDiagnostics()));
  ASSERT_THAT(Values(batch), ElementsAre("contributed"));
  Log("never contributed", kCutoff);
  Log("kept", kNow);
  // Returns the number of deleted entries that were never contributed.
  EXPECT_THAT(
      storage_->CleanUp(CleanupMode::kImmediate, kCutoff, NoDiagnostics()),
      IsOkAndHolds(1));
  EXPECT_THAT(storage_->CountPendingEntries(absl::InfinitePast()),
              IsOkAndHolds(1));
  EXPECT_THAT(
      storage_->CleanUp(CleanupMode::kImmediate, kCutoff, NoDiagnostics()),
      IsOkAndHolds(0));
}

TEST_F(DatabaseLogStorageTest, PeriodicCleanUpRunsOnEvery100thCall) {
  Log("expired", kCutoff);
  for (int i = 1; i < 100; ++i) {
    ASSERT_THAT(
        storage_->CleanUp(CleanupMode::kPeriodic, kCutoff, NoDiagnostics()),
        IsOkAndHolds(0));
  }
  EXPECT_THAT(storage_->CountPendingEntries(absl::InfinitePast()),
              IsOkAndHolds(1));
  EXPECT_THAT(
      storage_->CleanUp(CleanupMode::kPeriodic, kCutoff, NoDiagnostics()),
      IsOkAndHolds(1));
  EXPECT_THAT(storage_->CountPendingEntries(absl::InfinitePast()),
              IsOkAndHolds(0));
}

TEST_F(DatabaseLogStorageTest, ImmediateCleanUpRestartsThePeriodicCount) {
  for (int i = 1; i < 99; ++i) {
    ASSERT_THAT(
        storage_->CleanUp(CleanupMode::kPeriodic, kCutoff, NoDiagnostics()),
        IsOkAndHolds(0));
  }
  ASSERT_THAT(
      storage_->CleanUp(CleanupMode::kImmediate, kCutoff, NoDiagnostics()),
      IsOkAndHolds(0));
  Log("expired", kCutoff);
  // The next periodic cleanup is 100 calls after the immediate one.
  for (int i = 1; i < 100; ++i) {
    ASSERT_THAT(
        storage_->CleanUp(CleanupMode::kPeriodic, kCutoff, NoDiagnostics()),
        IsOkAndHolds(0));
  }
  EXPECT_THAT(
      storage_->CleanUp(CleanupMode::kPeriodic, kCutoff, NoDiagnostics()),
      IsOkAndHolds(1));
}

// A LogDatabase whose methods all fail with `status_`, recording the
// DiagnosticLogger each write got.
class FailingLogDatabase : public LogDatabase {
 public:
  explicit FailingLogDatabase(absl::Status status)
      : status_(std::move(status)) {}

  absl::Status InsertEntry(const LogEntry& entry,
                           DiagnosticLogger& logger) override {
    loggers_.push_back(&logger);
    return status_;
  }
  absl::StatusOr<SelectedLogEntries> GetAndMarkUncontributedEntries(
      absl::string_view workload_set_name, absl::Time cutoff_time,
      int64_t max_bytes, DiagnosticLogger& logger) override {
    loggers_.push_back(&logger);
    return status_;
  }
  absl::Status DeleteContributions(absl::string_view workload_set_name,
                                   absl::Span<const int64_t> entry_ids,
                                   DiagnosticLogger& logger) override {
    loggers_.push_back(&logger);
    return status_;
  }
  absl::StatusOr<int64_t> DeleteEntriesOlderThan(
      absl::Time cutoff_time, DiagnosticLogger& logger) override {
    loggers_.push_back(&logger);
    return status_;
  }
  absl::StatusOr<int64_t> CountEntriesNewerThan(
      absl::Time cutoff_time) override {
    return status_;
  }

  const std::vector<DiagnosticLogger*>& loggers() const { return loggers_; }

 private:
  const absl::Status status_;
  std::vector<DiagnosticLogger*> loggers_;
};

TEST(DatabaseLogStorageErrorTest,
     PeriodicCleanUpReturnsTheErrorOfThe100thCall) {
  const absl::Status error = absl::InternalError("database failed");
  auto database = std::make_unique<FailingLogDatabase>(error);
  FailingLogDatabase* database_ptr = database.get();
  DatabaseLogStorage storage(std::move(database));
  for (int i = 1; i < 100; ++i) {
    ASSERT_THAT(
        storage.CleanUp(CleanupMode::kPeriodic, kCutoff, NoDiagnostics()),
        IsOkAndHolds(0));
  }
  EXPECT_THAT(database_ptr->loggers(), IsEmpty());
  EXPECT_EQ(storage.CleanUp(CleanupMode::kPeriodic, kCutoff, NoDiagnostics())
                .status(),
            error);
}

TEST(DatabaseLogStorageErrorTest, PropagatesDatabaseErrorsWithoutLogging) {
  const absl::Status error = absl::InternalError("database failed");
  auto database = std::make_unique<FailingLogDatabase>(error);
  FailingLogDatabase* database_ptr = database.get();
  DatabaseLogStorage storage(std::move(database));
  RecordingDiagnosticLogger logger;

  EXPECT_EQ(storage.Log(LogEntry{.timestamp = kNow, .value = "a"}, logger),
            error);
  EXPECT_EQ(storage.GetLogsToUpload(kSet, kCutoff, /*max_bytes=*/100, logger)
                .status(),
            error);
  EXPECT_EQ(storage.CleanUp(CleanupMode::kImmediate, kCutoff, logger).status(),
            error);
  EXPECT_EQ(storage.CountPendingEntries(kCutoff).status(), error);

  // The caller's logger was passed through to every write, and the storage
  // logged nothing itself: logging the error is the caller's job.
  EXPECT_THAT(database_ptr->loggers(), ElementsAre(&logger, &logger, &logger));
  EXPECT_THAT(logger.codes(), IsEmpty());
}

TEST(DatabaseLogStorageErrorTest, DatabaseCausesAreLoggedExactlyOnce) {
  ScopedSqlite connection = OpenInMemory();
  FCP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SqliteLogDatabase> database,
                           SqliteLogDatabase::Create(connection.get()));
  DatabaseLogStorage storage(std::move(database));
  ASSERT_THAT(
      storage.Log(LogEntry{.timestamp = kNow, .value = "a"}, NoDiagnostics()),
      IsOk());
  FCP_ASSERT_OK_AND_ASSIGN(
      UploadBatch batch,
      storage.GetLogsToUpload(kSet, kCutoff, /*max_bytes=*/100,
                              NoDiagnostics()));
  ASSERT_THAT(Values(batch), ElementsAre("a"));

  // With a client transaction open, each write is refused and the cause is
  // logged once, by the database, to the logger of the call.
  Exec(connection.get(), "BEGIN;");
  RecordingDiagnosticLogger logger;
  EXPECT_THAT(storage.Log(LogEntry{.timestamp = kNow, .value = "b"}, logger),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(logger.codes(), SizeIs(1));
  EXPECT_THAT(
      storage.GetLogsToUpload("other", kCutoff, /*max_bytes=*/100, logger),
      StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(logger.codes(), SizeIs(2));
  EXPECT_THAT(storage.CleanUp(CleanupMode::kImmediate, kCutoff, logger),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(logger.codes(), SizeIs(3));
  // The rollback logs to the logger it's given, not the one of the selection.
  RecordingDiagnosticLogger rollback_logger;
  EXPECT_THAT(batch.RollBack(rollback_logger),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(logger.codes(),
              ElementsAre(DiagnosticCode::kErrorSqliteTransactionOpen,
                          DiagnosticCode::kErrorSqliteTransactionOpen,
                          DiagnosticCode::kErrorSqliteTransactionOpen));
  EXPECT_THAT(rollback_logger.codes(),
              ElementsAre(DiagnosticCode::kErrorSqliteTransactionOpen));
  Exec(connection.get(), "COMMIT;");
}

TEST(DatabaseLogStorageDeathTest, NullDatabaseCrashesAtConstruction) {
  EXPECT_DEATH(DatabaseLogStorage(nullptr), "");
}

TEST(DatabaseLogStorageLifetimeTest, RollbackAfterTheStorageIsGoneFails) {
  FCP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SqliteLogDatabase> database,
                           SqliteLogDatabase::Create(":memory:"));
  auto storage = std::make_unique<DatabaseLogStorage>(std::move(database));
  ASSERT_THAT(
      storage->Log(LogEntry{.timestamp = kNow, .value = "a"}, NoDiagnostics()),
      IsOk());
  FCP_ASSERT_OK_AND_ASSIGN(
      UploadBatch batch,
      storage->GetLogsToUpload(kSet, kCutoff, /*max_bytes=*/100,
                               NoDiagnostics()));
  ASSERT_THAT(Values(batch), ElementsAre("a"));

  // Not allowed, but mitigated: an error instead of a use-after-free.
  storage.reset();
  EXPECT_THAT(batch.RollBack(NoDiagnostics()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(UploadBatchTest, RollBackCallsTheCallbackOnceWithTheGivenLogger) {
  int calls = 0;
  DiagnosticLogger* rollback_logger = nullptr;
  UploadBatch batch(
      {LogEntry{.timestamp = kNow, .value = "a"}},
      [&](DiagnosticLogger& logger) {
        ++calls;
        rollback_logger = &logger;
        return absl::InternalError("rollback failed");
      },
      /*byte_limit_reached=*/true);
  EXPECT_THAT(Values(batch), ElementsAre("a"));
  EXPECT_TRUE(batch.byte_limit_reached());

  RecordingDiagnosticLogger logger;
  EXPECT_THAT(batch.RollBack(logger),
              StatusIs(absl::StatusCode::kInternal, "rollback failed"));
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(rollback_logger, &logger);
  // Later calls do nothing.
  EXPECT_THAT(batch.RollBack(logger), IsOk());
  EXPECT_EQ(calls, 1);
}

TEST(UploadBatchTest, EmptyAndMovedFromBatchesRollBackNothing) {
  UploadBatch empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_FALSE(empty.byte_limit_reached());
  EXPECT_THAT(empty.RollBack(NoDiagnostics()), IsOk());

  int calls = 0;
  UploadBatch original(
      {LogEntry{.timestamp = kNow, .value = "a"}},
      [&](DiagnosticLogger&) {
        ++calls;
        return absl::OkStatus();
      },
      /*byte_limit_reached=*/true);
  UploadBatch moved = std::move(original);
  EXPECT_TRUE(original.empty());  // NOLINT(bugprone-use-after-move)
  EXPECT_FALSE(original.byte_limit_reached());
  EXPECT_THAT(original.RollBack(NoDiagnostics()), IsOk());
  EXPECT_EQ(calls, 0);

  UploadBatch assigned;
  assigned = std::move(moved);
  EXPECT_THAT(
      moved.RollBack(NoDiagnostics()),  // NOLINT(bugprone-use-after-move)
      IsOk());
  EXPECT_EQ(calls, 0);
  EXPECT_THAT(Values(assigned), ElementsAre("a"));
  EXPECT_THAT(assigned.RollBack(NoDiagnostics()), IsOk());
  EXPECT_EQ(calls, 1);
}

TEST(DatabaseLogStorageConcurrencyTest,
     EveryEntryIsSelectedOncePerWorkloadSet) {
  FCP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SqliteLogDatabase> database,
                           SqliteLogDatabase::Create(":memory:"));
  DatabaseLogStorage storage(std::move(database));
  constexpr int kWriters = 4;
  constexpr int kEntriesPerWriter = 50;
  // Small batches, so that the selections interleave with the writes.
  constexpr int64_t kMaxBytes = 16;

  // Adds the values of `batch` to `selected`.
  auto record = [](const UploadBatch& batch,
                   std::map<std::string, int>& selected) {
    for (const LogEntry& entry : batch.entries()) {
      ++selected[entry.value];
    }
  };

  // While the writers log, the reader selects for "rolled back" (and rolls
  // back) and for "kept" (and keeps the batches).
  std::atomic<bool> writers_done = false;
  std::map<std::string, int> kept;
  std::thread reader([&] {
    while (!writers_done.load(std::memory_order_acquire)) {
      absl::StatusOr<UploadBatch> batch = storage.GetLogsToUpload(
          "rolled back", kCutoff, kMaxBytes, NoDiagnostics());
      ASSERT_THAT(batch, IsOk());
      ASSERT_THAT(batch->RollBack(NoDiagnostics()), IsOk());
      batch =
          storage.GetLogsToUpload("kept", kCutoff, kMaxBytes, NoDiagnostics());
      ASSERT_THAT(batch, IsOk());
      record(*batch, kept);
    }
  });
  std::vector<std::thread> writers;
  for (int writer = 0; writer < kWriters; ++writer) {
    writers.emplace_back([&storage, writer] {
      for (int i = 0; i < kEntriesPerWriter; ++i) {
        ASSERT_THAT(storage.Log(LogEntry{.timestamp = kNow,
                                         .value = std::to_string(writer) + "-" +
                                                  std::to_string(i)},
                                NoDiagnostics()),
                    IsOk());
      }
    });
  }
  for (std::thread& writer : writers) {
    writer.join();
  }
  writers_done.store(true, std::memory_order_release);
  reader.join();

  // Selects everything left for `set`.
  auto drain = [&](absl::string_view set,
                   std::map<std::string, int>& selected) {
    while (true) {
      FCP_ASSERT_OK_AND_ASSIGN(
          UploadBatch batch,
          storage.GetLogsToUpload(set, kCutoff, kMaxBytes, NoDiagnostics()));
      if (batch.empty()) {
        return;
      }
      record(batch, selected);
    }
  };
  std::map<std::string, int> rolled_back;
  drain("rolled back", rolled_back);
  drain("kept", kept);

  std::map<std::string, int> expected;
  for (int writer = 0; writer < kWriters; ++writer) {
    for (int i = 0; i < kEntriesPerWriter; ++i) {
      expected[std::to_string(writer) + "-" + std::to_string(i)] = 1;
    }
  }
  EXPECT_EQ(kept, expected);
  EXPECT_EQ(rolled_back, expected);
}

}  // namespace
}  // namespace fcp::client::privatelogger
