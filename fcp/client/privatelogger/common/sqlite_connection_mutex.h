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
#ifndef FCP_CLIENT_PRIVATELOGGER_COMMON_SQLITE_CONNECTION_MUTEX_H_
#define FCP_CLIENT_PRIVATELOGGER_COMMON_SQLITE_CONNECTION_MUTEX_H_

#include <memory>

#include "absl/synchronization/mutex.h"

struct sqlite3;
struct sqlite3_mutex;

namespace fcp::client::privatelogger {

// Returns a mutex shared by everything in this library that wraps the same
// `sqlite3*` connection (e.g. two stores sharing one connection), so that
// their connection-level transactions can't interleave even if the connection
// wasn't opened with SQLITE_OPEN_FULLMUTEX. The mutex lives as long as someone
// holds the returned pointer.
std::shared_ptr<absl::Mutex> GetOrCreateConnectionMutex(sqlite3* db);

// Holds the SQLite connection's own recursive mutex (if SQLite was built with
// one) for its lifetime, so that a SQLite call made on the same `sqlite3*`
// connection outside this library, e.g. by another thread of the client, can't
// run in the middle of one of our operations. It serializes single calls, not
// the client's transactions: a transaction the client opens spans many calls
// and stays open between them, and a write of ours would join it. So every
// write first checks that no transaction is open (CheckNoTransactionOpen() in
// sqlite_util.h), and the client must not keep a transaction open on a
// connection it shares with this library. `db` may be null.
class ScopedSqliteConnectionLock {
 public:
  explicit ScopedSqliteConnectionLock(sqlite3* db);
  ~ScopedSqliteConnectionLock();

  ScopedSqliteConnectionLock(const ScopedSqliteConnectionLock&) = delete;
  ScopedSqliteConnectionLock& operator=(const ScopedSqliteConnectionLock&) =
      delete;

 private:
  sqlite3_mutex* mutex_;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_COMMON_SQLITE_CONNECTION_MUTEX_H_
