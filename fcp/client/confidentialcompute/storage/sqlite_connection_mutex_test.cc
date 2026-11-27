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
#include "fcp/client/confidentialcompute/storage/sqlite_connection_mutex.h"

#include <memory>
#include <thread>  // NOLINT(build/c++11)

#include "gtest/gtest.h"
#include "absl/synchronization/mutex.h"
#include "fcp/client/confidentialcompute/storage/sqlite_test_util.h"
#include "sqlite3.h"

namespace fcp::client::confidentialcompute::internal {
namespace {

TEST(SqliteConnectionMutexTest, SameConnectionGetsTheSameMutex) {
  ScopedSqlite db = OpenInMemory();
  std::shared_ptr<absl::Mutex> first = GetOrCreateConnectionMutex(db.get());
  std::shared_ptr<absl::Mutex> second = GetOrCreateConnectionMutex(db.get());
  EXPECT_EQ(first, second);
}

TEST(SqliteConnectionMutexTest, DifferentConnectionsGetDifferentMutexes) {
  ScopedSqlite db_a = OpenInMemory();
  ScopedSqlite db_b = OpenInMemory();
  std::shared_ptr<absl::Mutex> a = GetOrCreateConnectionMutex(db_a.get());
  std::shared_ptr<absl::Mutex> b = GetOrCreateConnectionMutex(db_b.get());
  EXPECT_NE(a, b);
}

TEST(SqliteConnectionMutexTest, MutexIsReleasedWhenTheLastUserIsGone) {
  ScopedSqlite db = OpenInMemory();
  std::shared_ptr<absl::Mutex> mutex = GetOrCreateConnectionMutex(db.get());
  std::weak_ptr<absl::Mutex> weak = mutex;
  mutex.reset();
  // The registry doesn't keep it alive.
  EXPECT_TRUE(weak.expired());
  // A later user gets a new, working mutex.
  std::shared_ptr<absl::Mutex> again = GetOrCreateConnectionMutex(db.get());
  ASSERT_NE(again, nullptr);
  absl::MutexLock lock(*again);
}

TEST(SqliteConnectionMutexTest, ScopedLockHoldsTheConnectionMutex) {
  ScopedSqlite db = OpenInMemory();
  sqlite3_mutex* connection_mutex = sqlite3_db_mutex(db.get());
  ASSERT_NE(connection_mutex, nullptr);
  // Tries the connection mutex from another thread (it's recursive, so
  // trying on this thread would always succeed).
  auto try_from_other_thread = [connection_mutex] {
    bool acquired = false;
    std::thread thread([&] {
      if (sqlite3_mutex_try(connection_mutex) == SQLITE_OK) {
        acquired = true;
        sqlite3_mutex_leave(connection_mutex);
      }
    });
    thread.join();
    return acquired;
  };
  {
    ScopedSqliteConnectionLock lock(db.get());
    EXPECT_FALSE(try_from_other_thread());
  }
  EXPECT_TRUE(try_from_other_thread());
}

TEST(SqliteConnectionMutexTest, ScopedLockAcceptsNull) {
  ScopedSqliteConnectionLock lock(nullptr);
}

}  // namespace
}  // namespace fcp::client::confidentialcompute::internal
