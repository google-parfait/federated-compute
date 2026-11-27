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

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "sqlite3.h"

namespace fcp::client::confidentialcompute::internal {

std::shared_ptr<absl::Mutex> GetOrCreateConnectionMutex(sqlite3* db) {
  static absl::NoDestructor<absl::Mutex> registry_mutex;
  static absl::NoDestructor<
      absl::flat_hash_map<sqlite3*, std::weak_ptr<absl::Mutex>>>
      registry ABSL_GUARDED_BY(*registry_mutex);

  absl::MutexLock lock(*registry_mutex);
  absl::erase_if(*registry,
                 [](const auto& entry) { return entry.second.expired(); });
  std::weak_ptr<absl::Mutex>& weak_mu = (*registry)[db];
  if (std::shared_ptr<absl::Mutex> mu = weak_mu.lock()) {
    return mu;
  }
  auto mu = std::make_shared<absl::Mutex>();
  weak_mu = mu;
  return mu;
}

ScopedSqliteConnectionLock::ScopedSqliteConnectionLock(sqlite3* db)
    : mutex_(db != nullptr ? sqlite3_db_mutex(db) : nullptr) {
  sqlite3_mutex_enter(mutex_);
}

ScopedSqliteConnectionLock::~ScopedSqliteConnectionLock() {
  sqlite3_mutex_leave(mutex_);
}

}  // namespace fcp::client::confidentialcompute::internal
