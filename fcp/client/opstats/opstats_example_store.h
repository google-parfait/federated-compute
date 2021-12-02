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
#ifndef FCP_CLIENT_OPSTATS_OPSTATS_EXAMPLE_STORE_H_
#define FCP_CLIENT_OPSTATS_OPSTATS_EXAMPLE_STORE_H_

#include <string>
#include <utility>

#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_db.h"
#include "fcp/client/simple_task_environment.h"

namespace fcp {
namespace client {
namespace opstats {
inline static constexpr char kOpStatsCollectionUri[] = "internal:/opstats";
inline static constexpr char kPopulationName[] = "population_name";
inline static constexpr char kSessionName[] = "session_name";
inline static constexpr char kTaskName[] = "task_name";
inline static constexpr char kEventsEventType[] = "events-event_type";
inline static constexpr char kEventsTimestampMillis[] = "events-timestamp";
inline static constexpr char kDatasetStatsUri[] = "dataset_stats-uri";
inline static constexpr char kDatasetStatsNumExamplesRead[] =
    "dataset_stats-num_examples_read";
inline static constexpr char kDatasetStatsNumBytesRead[] =
    "dataset_stats-num_bytes_read";
inline static constexpr char kErrorMessage[] = "error_message";
inline static constexpr char kRetryWindowDelayMinMillis[] =
    "retry_window-delay_min";
inline static constexpr char kRetryWindowDelayMaxMillis[] =
    "retry_window-delay_max";
inline static constexpr char kBytesDownloaded[] = "bytes_downloaded";
inline static constexpr char kBytesUploaded[] = "bytes_uploaded";
inline static constexpr char kChunkingLayerBytesDownloaded[] =
    "chunking_layer_bytes_downloaded";
inline static constexpr char kChunkingLayerBytesUploaded[] =
    "chunking_layer_bytes_uploaded";
inline static constexpr char kEarliestTrustWorthyTimeMillis[] =
    "earliest_trustworthy_time";

class OpStatsExampleIterator : public ExampleIterator {
 public:
  explicit OpStatsExampleIterator(std::vector<OperationalStats> op_stats,
                                  int64_t earliest_trustworthy_time)
      : next_(0),
        data_(std::move(op_stats)),
        earliest_trustworthy_time_millis_(earliest_trustworthy_time) {}
  absl::StatusOr<std::string> Next() override;
  void Close() override;

 private:
  // The index for the next OperationalStats to be used.
  int next_;
  std::vector<OperationalStats> data_;
  int64_t earliest_trustworthy_time_millis_;
};

absl::StatusOr<std::unique_ptr<ExampleIterator>> CreateExampleIterator(
    const google::internal::federated::plan::ExampleSelector& example_selector,
    OpStatsDb& op_stats_db, LogManager& log_manager);

}  // namespace opstats
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_OPSTATS_OPSTATS_EXAMPLE_STORE_H_
