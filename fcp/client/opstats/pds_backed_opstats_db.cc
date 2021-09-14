/*
 * Copyright 2021 Google LLC
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
#include "fcp/client/opstats/pds_backed_opstats_db.h"

#include <sys/file.h>

#include <filesystem>

#include "google/protobuf/util/time_util.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/log_manager.h"
#include "protostore/file-storage.h"
#include "protostore/proto-data-store.h"

namespace fcp {
namespace client {
namespace opstats {
namespace {

using ::google::protobuf::util::TimeUtil;

ABSL_CONST_INIT absl::Mutex file_lock_mutex(absl::kConstInit);

absl::flat_hash_set<std::string>* GetFilesInUseSet() {
  // Create the heap allocated static set only once, never call d'tor.
  // See: go/totw/110
  static absl::flat_hash_set<std::string>* files_in_use =
      new absl::flat_hash_set<std::string>();
  return files_in_use;
}

absl::StatusOr<int> AcquireFileLock(const std::string& db_path,
                                    LogManager& log_manager) {
  absl::WriterMutexLock lock(&file_lock_mutex);
  // If the underlying file is already in the hash set, it means another
  // instance of OpStatsDb is using it, and we'll return an error.
  absl::flat_hash_set<std::string>* files_in_use = GetFilesInUseSet();
  if (!files_in_use->insert(db_path).second) {
    log_manager.LogDiag(ProdDiagCode::OPSTATS_MULTIPLE_DB_INSTANCE_DETECTED);
    return absl::InternalError(
        "Another instance is already using the underlying database file.");
  }
  // Create a new file descriptor.
  // Create the file if it doesn't exist, set permission to 0644.
  int fd = open(db_path.c_str(), O_CREAT | O_RDWR,
                S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fd < 0) {
    files_in_use->erase(db_path);
    log_manager.LogDiag(ProdDiagCode::OPSTATS_FAILED_TO_OPEN_FILE);
    return absl::InternalError(absl::StrCat("Failed to open file: ", db_path));
  }
  // Acquire exclusive lock on the file in a non-blocking mode.
  // flock(2) applies lock on the file object in the open file table, so it can
  // apply lock across different processes.  Within a process, flock doesn't
  // necessarily guarantee synchronization across multiple threads.
  // See:https://man7.org/linux/man-pages/man2/flock.2.html
  if (flock(fd, LOCK_EX | LOCK_NB) < 0) {
    files_in_use->erase(db_path);
    close(fd);
    log_manager.LogDiag(ProdDiagCode::OPSTATS_MULTIPLE_DB_INSTANCE_DETECTED);
    return absl::InternalError(
        "Failed to acquire file lock on the underlying database file.");
  }
  return fd;
}

void ReleaseFileLock(const std::string& db_path, int fd) {
  absl::WriterMutexLock lock(&file_lock_mutex);
  GetFilesInUseSet()->erase(db_path);
  FCP_CHECK(fd >= 0);
  // File lock is released when the descriptor is closed.
  close(fd);
}

// Returns the data in the db, or an error from the read operation.
absl::StatusOr<OpStatsSequence> ReadInternal(
    protostore::ProtoDataStore<OpStatsSequence>& db, LogManager& log_manager) {
  absl::StatusOr<const OpStatsSequence*> data = db.Read();
  if (data.ok()) {
    return *data.value();
  } else {
    log_manager.LogDiag(ProdDiagCode::OPSTATS_READ_FAILED);
    return absl::InternalError(
        absl::StrCat("Failed to read from database, with error message: ",
                     data.status().message()));
  }
}

// Overwrites the db to contain an empty OpStatsSequence message.
absl::Status ResetInternal(protostore::ProtoDataStore<OpStatsSequence>& db,
                           LogManager& log_manager) {
  absl::Status reset_status = db.Write(absl::make_unique<OpStatsSequence>());
  if (!reset_status.ok()) {
    log_manager.LogDiag(ProdDiagCode::OPSTATS_RESET_FAILED);
    return absl::InternalError(
        absl::StrCat("Failed to reset the database, with error message: ",
                     reset_status.code()));
  }
  return absl::OkStatus();
}

absl::Time GetLastUpdateTime(const OperationalStats& operational_stats) {
  if (operational_stats.events().empty()) {
    return absl::InfinitePast();
  }
  return absl::FromUnixSeconds(TimeUtil::TimestampToSeconds(
      operational_stats.events().rbegin()->timestamp()));
}

void RemoveOutdatedData(OpStatsSequence& data, absl::Duration ttl) {
  absl::Time earliest_accepted_time = absl::Now() - ttl;
  auto* op_stats = data.mutable_opstats();
  op_stats->erase(
      std::remove_if(op_stats->begin(), op_stats->end(),
                     [earliest_accepted_time](const OperationalStats& data) {
                       return GetLastUpdateTime(data) < earliest_accepted_time;
                     }),
      op_stats->end());
}

void PruneOldDataUntilBelowSizeLimit(OpStatsSequence& data,
                                     const int64_t max_size_bytes,
                                     LogManager& log_manager) {
  int64_t current_size = data.ByteSizeLong();
  auto& op_stats = *(data.mutable_opstats());
  if (current_size > max_size_bytes) {
    int64_t num_pruned_entries = 0;
    auto it = op_stats.begin();
    absl::Time earliest_event_time = absl::InfinitePast();
    // The OperationalStats are sorted by time from earliest to latest, so we'll
    // remove from the start.
    while (current_size > max_size_bytes && it != op_stats.end()) {
      if (earliest_event_time == absl::InfinitePast()) {
        earliest_event_time = GetLastUpdateTime(*it);
      }
      num_pruned_entries++;
      // Note that the size of an OperationalStats is smaller than the size
      // impact it has on the OpStatsSequence. We are being conservative here.
      current_size -= it->ByteSizeLong();
      it++;
    }
    op_stats.erase(op_stats.begin(), it);
    log_manager.LogToLongHistogram(
        HistogramCounters::OPSTATS_NUM_PRUNED_ENTRIES, num_pruned_entries);
    log_manager.LogToLongHistogram(
        HistogramCounters::OPSTATS_OLDEST_PRUNED_ENTRY_TENURE_HOURS,
        absl::ToInt64Hours(absl::Now() - earliest_event_time));
  }
  log_manager.LogToLongHistogram(HistogramCounters::OPSTATS_DB_SIZE_BYTES,
                                 current_size);
  log_manager.LogToLongHistogram(HistogramCounters::OPSTATS_DB_NUM_ENTRIES,
                                 op_stats.size());
}

}  // anonymous namespace

absl::StatusOr<std::unique_ptr<OpStatsDb>> PdsBackedOpStatsDb::Create(
    const std::string& base_dir, absl::Duration ttl, LogManager& log_manager,
    int64_t max_size_bytes, bool enforce_singleton) {
  std::filesystem::path path(base_dir);
  if (!path.is_absolute()) {
    log_manager.LogDiag(ProdDiagCode::OPSTATS_INVALID_FILE_PATH);
    return absl::InvalidArgumentError(
        absl::StrCat("The provided path: ", base_dir,
                     "is invalid. The path must start with \"/\""));
  }
  path /= kParentDir;
  std::error_code error;
  std::filesystem::create_directories(path, error);
  if (error.value() != 0) {
    log_manager.LogDiag(ProdDiagCode::OPSTATS_PARENT_DIR_CREATION_FAILED);
    return absl::InternalError(
        absl::StrCat("Failed to create directory ", path.generic_string()));
  }
  path /= kDbFileName;
  std::function<void()> lock_releaser;
  auto file_storage = absl::make_unique<protostore::FileStorage>();
  std::unique_ptr<protostore::ProtoDataStore<OpStatsSequence>> pds;
  bool should_initiate;
  if (enforce_singleton) {
    std::string db_path = path.generic_string();
    FCP_ASSIGN_OR_RETURN(int fd, AcquireFileLock(db_path, log_manager));
    lock_releaser = [db_path, fd]() { ReleaseFileLock(db_path, fd); };
    pds = absl::make_unique<protostore::ProtoDataStore<OpStatsSequence>>(
        *file_storage, db_path);
    absl::StatusOr<int64_t> file_size = file_storage->GetFileSize(path);
    if (!file_size.ok()) {
      lock_releaser();
      return file_size.status();
    }
    // If the size of the underlying file is zero, it means this is the first
    // time we create the database.
    should_initiate = file_size.value() == 0;
  } else {
    // Lock releaser is no-op because we didn't acquire the lock in this branch.
    lock_releaser = []() {};
    pds = absl::make_unique<protostore::ProtoDataStore<OpStatsSequence>>(
        *file_storage, path.generic_string());
    // If the underlying file doesn't exist, it means this is the first time we
    // create the database.
    should_initiate = !std::filesystem::exists(path);
  }

  // If this is the first time we create the OpStatsDb, we want to create an
  // empty database.
  if (should_initiate) {
    absl::Status write_status =
        pds->Write(absl::make_unique<OpStatsSequence>());
    if (!write_status.ok()) {
      lock_releaser();
      return write_status;
    }
  }
  return absl::WrapUnique(
      new PdsBackedOpStatsDb(std::move(pds), std::move(file_storage), ttl,
                             log_manager, max_size_bytes, lock_releaser));
}

PdsBackedOpStatsDb::~PdsBackedOpStatsDb() { lock_releaser_(); }

absl::StatusOr<OpStatsSequence> PdsBackedOpStatsDb::Read() {
  absl::WriterMutexLock lock(&mutex_);
  auto data_or = ReadInternal(*db_, log_manager_);
  if (!data_or.ok()) {
    // Try resetting after a failed read.
    auto reset_status = ResetInternal(*db_, log_manager_);
  }
  return data_or;
}

absl::Status PdsBackedOpStatsDb::Transform(
    std::function<void(OpStatsSequence&)> func) {
  absl::WriterMutexLock lock(&mutex_);
  OpStatsSequence data;
  auto data_or = ReadInternal(*db_, log_manager_);
  if (!data_or.ok()) {
    // Try resetting after a failed read.
    FCP_RETURN_IF_ERROR(ResetInternal(*db_, log_manager_));
  } else {
    data = std::move(data_or).value();
    RemoveOutdatedData(data, ttl_);
  }
  func(data);
  PruneOldDataUntilBelowSizeLimit(data, max_size_bytes_, log_manager_);

  absl::Status status =
      db_->Write(absl::make_unique<OpStatsSequence>(std::move(data)));
  if (!status.ok()) {
    log_manager_.LogDiag(ProdDiagCode::OPSTATS_WRITE_FAILED);
  }
  return status;
}

}  // namespace opstats
}  // namespace client
}  // namespace fcp
