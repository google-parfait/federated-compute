/*
 * Copyright 2022 Google LLC
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
#ifndef FCP_CLIENT_FEDERATED_PROTOCOL_UTIL_H_
#define FCP_CLIENT_FEDERATED_PROTOCOL_UTIL_H_

#include <string>

#include "google/protobuf/duration.pb.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "fcp/client/log_manager.h"
#include "fcp/protos/federated_api.pb.h"

namespace fcp {
namespace client {

// Utility methods likely shared by FederatedProtocol implementations.

// Picks an absolute retry time by picking a retry delay from the range
// specified by the RetryWindow, and then adding it to the current timestamp.
absl::Time PickRetryTimeFromRange(const ::google::protobuf::Duration& min_delay,
                                  const ::google::protobuf::Duration& max_delay,
                                  absl::BitGen& bit_gen);

// Picks a retry delay and encodes it as a zero-width RetryWindow (where
// delay_min and delay_max are set to the same value), from a given target delay
// and a configured amount of jitter.
::google::internal::federatedml::v2::RetryWindow
GenerateRetryWindowFromTargetDelay(absl::Duration target_delay,
                                   double jitter_percent,
                                   absl::BitGen& bit_gen);

// Converts the given absl::Time to a zero-width RetryWindow (where
// delay_min and delay_max are set to the same value), by converting the target
// retry time to a delay relative to the current timestamp.
::google::internal::federatedml::v2::RetryWindow
GenerateRetryWindowFromRetryTime(absl::Time retry_time);

// Extracts a task name from an aggregation session ID (in the HTTP protocol) or
// a phase ID (in the gRPC protocol), both of which are expected to adhere to
// the following format: "population_name/task_name#round_id.shard_id".
//
// Returns the `session_id` string unmodified if it does not match that format.
// A diag code will be logged to the `LogManager` in this case.
std::string ExtractTaskNameFromAggregationSessionId(
    const std::string& session_id, const std::string& population_name,
    fcp::client::LogManager& log_manager);

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_FEDERATED_PROTOCOL_UTIL_H_
