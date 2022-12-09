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

#ifndef FCP_CLIENT_OPSTATS_OPSTATS_UTILS_H_
#define FCP_CLIENT_OPSTATS_OPSTATS_UTILS_H_

#include <optional>
#include <string>

#include "google/protobuf/timestamp.pb.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/opstats/opstats_db.h"

namespace fcp {
namespace client {
namespace opstats {

// Returns an optional containing an OperationalStats of the last time the
// runtime successfully contributed to a task with the given task name,
// otherwise returns an empty optional.
std::optional<OperationalStats> GetLastSuccessfulContribution(
    OpStatsSequence& data, const std::string& task_name);

// Returns an optional containing the timestamp of the last time the runtime
// successfully contributed to a task with the given task name, otherwise
// returns an empty optional.
std::optional<google::protobuf::Timestamp> GetLastSuccessfulContributionTime(
    OpStatsSequence& data, const std::string& task_name);

}  // namespace opstats
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_OPSTATS_OPSTATS_UTILS_H_
