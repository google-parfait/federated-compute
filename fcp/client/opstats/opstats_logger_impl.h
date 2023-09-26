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
#ifndef FCP_CLIENT_OPSTATS_OPSTATS_LOGGER_IMPL_H_
#define FCP_CLIENT_OPSTATS_OPSTATS_LOGGER_IMPL_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "fcp/client/flags.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_db.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/stats.h"
#include "fcp/protos/opstats.pb.h"

namespace fcp {
namespace client {
namespace opstats {

// An implementation of OpStatsLogger backed by a database.
class OpStatsLoggerImpl : public OpStatsLogger {
 public:
  // Creates a logger backed by an actual database. Populates the internal
  // message with the provided session and population names.
  OpStatsLoggerImpl(std::unique_ptr<OpStatsDb> db, LogManager* log_manager,
                    const Flags* flags, const std::string& session_name,
                    const std::string& population_name);

  // Commits the cumulative message to the db.
  ~OpStatsLoggerImpl() override;

  // Adds an event and the given task name to the cumulative internal message,
  // in a single transaction.
  void AddEventAndSetTaskName(const std::string& task_name,
                              OperationalStats::Event::EventKind event)
      ABSL_LOCKS_EXCLUDED(mutex_) override;

  // Adds an event to the cumulative internal message.
  void AddEvent(OperationalStats::Event::EventKind event)
      ABSL_LOCKS_EXCLUDED(mutex_) override;

  // Adds an event and corresponding error message to the cumulative internal
  // message.
  void AddEventWithErrorMessage(OperationalStats::Event::EventKind event,
                                const std::string& error_message) override;

  // Updates info associated with a dataset created for a given collection
  // in the cumulative internal message. If this is called multiple times
  // for the same collection, the example counts and sizes will be
  // aggregated in the underlying submessage.
  void UpdateDatasetStats(const std::string& collection_uri,
                          int additional_example_count,
                          int64_t additional_example_size_bytes)
      ABSL_LOCKS_EXCLUDED(mutex_) override;

  // Record the first access time of a dataset created for a given collection.
  void RecordCollectionFirstAccessTime(absl::string_view collection_uri,
                                       absl::Time first_access_time)
      ABSL_LOCKS_EXCLUDED(mutex_) override;

  // Adds network stats, replacing any old stats for the run, to the cumulative
  // internal message.
  void SetNetworkStats(const NetworkStats& network_stats)
      ABSL_LOCKS_EXCLUDED(mutex_) override;

  // Sets the retry window, replacing any old retry window, in the cumulative
  // internal message. Any retry token in the retry window message is dropped.
  void SetRetryWindow(
      google::internal::federatedml::v2::RetryWindow retry_window)
      ABSL_LOCKS_EXCLUDED(mutex_) override;

  // Get the underlying opstats database.
  OpStatsDb* GetOpStatsDb() override { return db_.get(); }

  // Whether opstats is enabled. An instance of this class should only ever be
  // created when opstats is enabled.
  bool IsOpStatsEnabled() const override { return true; }

  // Syncs all logged events to storage.
  absl::Status CommitToStorage() override;

  // Returns the task name of the currently executing task. Only returns a valid
  // task name if called after `AddEventAndSetTaskName` is called.
  std::string GetCurrentTaskName() override;

  // Start logging stats for a new Phase.
  void StartLoggingForPhase(OperationalStats::PhaseStats::Phase phase) override;

  // Stop logging stats for the current Phase.
  void StopLoggingForTheCurrentPhase() override;

 private:
  // Helper for adding a new event of the specified kind to the cumulative
  // message being stored in this class.
  void AddNewEventToStats(OperationalStats::Event::EventKind kind)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  // Helper for adding a new event of the specified kind to the current
  // PhaseStats message being stored in this class.
  void AddNewEventToCurrentPhaseStats(OperationalStats::Event::EventKind kind)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Cumulative message storing information about this run.
  absl::flat_hash_map<std::string, absl::Time> collection_first_access_time_map_
      ABSL_GUARDED_BY(mutex_);
  OperationalStats stats_ ABSL_GUARDED_BY(mutex_);
  bool already_committed_ ABSL_GUARDED_BY(mutex_) = false;
  std::unique_ptr<OpStatsDb> db_;
  LogManager* log_manager_;
  OperationalStats::PhaseStats current_phase_stats_ ABSL_GUARDED_BY(mutex_);
  const bool use_phase_stats_;
  NetworkStats accumulated_network_stats_ ABSL_GUARDED_BY(mutex_);
  absl::Mutex mutex_;
};

}  // namespace opstats
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_OPSTATS_OPSTATS_LOGGER_IMPL_H_
