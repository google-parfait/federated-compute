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
#include "fcp/client/opstats/opstats_logger_impl.h"

#include <string>
#include <utility>

#include "google/protobuf/util/time_util.h"
#include "fcp/base/time_util.h"
#include "fcp/client/flags.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_db.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/opstats.pb.h"

namespace fcp {
namespace client {
namespace opstats {

using ::google::internal::federatedml::v2::RetryWindow;

OpStatsLoggerImpl::OpStatsLoggerImpl(std::unique_ptr<OpStatsDb> db,
                                     LogManager* log_manager,
                                     const Flags* flags,
                                     const std::string& session_name,
                                     const std::string& population_name)
    : db_(std::move(db)), log_manager_(log_manager) {
  log_manager_->LogDiag(DebugDiagCode::TRAINING_OPSTATS_ENABLED);
  log_manager_->LogDiag(ProdDiagCode::OPSTATS_DB_COMMIT_EXPECTED);

  // Setup the OperationalStats message for the new run.
  stats_.set_session_name(session_name);
  stats_.set_population_name(population_name);
}

OpStatsLoggerImpl::~OpStatsLoggerImpl() {
  // We're in the dtor, we don't care about what CommitToStorage returns.
  auto status = CommitToStorage();
}

void OpStatsLoggerImpl::AddEventAndSetTaskName(
    const std::string& task_name, OperationalStats::Event::EventKind event) {
  absl::MutexLock lock(&mutex_);
  AddNewEventToStats(event);
  stats_.set_task_name(task_name);
}

void OpStatsLoggerImpl::AddEvent(OperationalStats::Event::EventKind event) {
  absl::MutexLock lock(&mutex_);
  AddNewEventToStats(event);
}

void OpStatsLoggerImpl::AddEventWithErrorMessage(
    OperationalStats::Event::EventKind event,
    const std::string& error_message) {
  absl::MutexLock lock(&mutex_);
  AddNewEventToStats(event);
  // Don't replace an existing error message.
  if (stats_.error_message().empty()) {
    stats_.set_error_message(error_message);
  }
}

void OpStatsLoggerImpl::UpdateDatasetStats(
    const std::string& collection_uri, int additional_example_count,
    int64_t additional_example_size_bytes) {
  absl::MutexLock lock(&mutex_);
  auto& dataset_stats = (*stats_.mutable_dataset_stats())[collection_uri];
  dataset_stats.set_num_examples_read(dataset_stats.num_examples_read() +
                                      additional_example_count);
  dataset_stats.set_num_bytes_read(dataset_stats.num_bytes_read() +
                                   additional_example_size_bytes);
}

void OpStatsLoggerImpl::SetNetworkStats(const NetworkStats& network_stats) {
  absl::MutexLock lock(&mutex_);
  stats_.set_chunking_layer_bytes_downloaded(network_stats.bytes_downloaded);
  stats_.set_chunking_layer_bytes_uploaded(network_stats.bytes_uploaded);
  *stats_.mutable_network_duration() =
      TimeUtil::ConvertAbslToProtoDuration(network_stats.network_duration);
}

void OpStatsLoggerImpl::SetRetryWindow(RetryWindow retry_window) {
  absl::MutexLock lock(&mutex_);
  retry_window.clear_retry_token();
  *stats_.mutable_retry_window() = std::move(retry_window);
}

void OpStatsLoggerImpl::AddNewEventToStats(
    OperationalStats::Event::EventKind kind) {
  auto new_event = stats_.add_events();
  new_event->set_event_type(kind);
  *new_event->mutable_timestamp() = google::protobuf::util::TimeUtil::GetCurrentTime();
}

absl::Status OpStatsLoggerImpl::CommitToStorage() {
  absl::MutexLock lock(&mutex_);
  log_manager_->LogDiag(ProdDiagCode::OPSTATS_DB_COMMIT_ATTEMPTED);
  const absl::Time before_commit_time = absl::Now();
  auto status = already_committed_
                    ? db_->Transform([stats = &stats_](OpStatsSequence& data) {
                        // Check if opstats on disk somehow got cleared between
                        // the first commit and now, and handle appropriately.
                        // This can happen e.g. if the ttl for the opstats db
                        // is incorrectly configured to have a very low ttl,
                        // causing the entire history to be lost as part of the
                        // update.
                        if (data.opstats_size() == 0) {
                          *data.add_opstats() = *stats;
                        } else {
                          *data.mutable_opstats(data.opstats_size() - 1) =
                              *stats;
                        }
                      })
                    : db_->Transform([stats = &stats_](OpStatsSequence& data) {
                        *data.add_opstats() = *stats;
                      });
  const absl::Time after_commit_time = absl::Now();
  log_manager_->LogToLongHistogram(
      HistogramCounters::TRAINING_OPSTATS_COMMIT_LATENCY,
      absl::ToInt64Milliseconds(after_commit_time - before_commit_time));
  already_committed_ = true;
  return status;
}

std::string OpStatsLoggerImpl::GetCurrentTaskName() {
  absl::MutexLock lock(&mutex_);
  return stats_.task_name();
}

}  // namespace opstats
}  // namespace client
}  // namespace fcp
