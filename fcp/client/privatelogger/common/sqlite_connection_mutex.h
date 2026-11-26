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

// Holds the SQLite connection's own mutex (sqlite3_db_mutex()) for its
// lifetime, so that the client's SQLite calls on the same connection from other
// threads can't run in the middle of one of our operations. Does nothing if
// the connection has no mutex (see https://www.sqlite.org/threadsafe.html) or
// `db` is null.
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
