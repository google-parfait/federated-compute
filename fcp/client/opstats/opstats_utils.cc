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

#include "fcp/client/opstats/opstats_utils.h"

#include <algorithm>
#include <optional>
#include <string>

#include "google/protobuf/timestamp.pb.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/opstats/opstats_db.h"
#include "fcp/protos/opstats.pb.h"

namespace fcp {
namespace client {
namespace opstats {

std::optional<google::protobuf::Timestamp> GetLastSuccessfulContributionTime(
    OpStatsSequence& data, const std::string& task_name) {
  std::optional<OperationalStats> last_successful_entry =
      GetLastSuccessfulContribution(data, task_name);
  if (!last_successful_entry.has_value()) {
    return std::nullopt;
  }

  auto upload_started = std::find_if(
      last_successful_entry->events().begin(),
      last_successful_entry->events().end(),
      [](const OperationalStats::Event& event) {
        return event.event_type() ==
               OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED;
      });
  if (upload_started == last_successful_entry->events().end()) {
    // For last_successful_entry to have a value, it must have had an
    // EVENT_KIND_RESULT_UPLOAD_STARTED event, so we should never reach this.
    return std::nullopt;
  }

  return upload_started->timestamp();
}

std::optional<OperationalStats> GetLastSuccessfulContribution(
    OpStatsSequence& data, const std::string& task_name) {
  for (auto it = data.opstats().rbegin(); it != data.opstats().rend(); ++it) {
    const OperationalStats& opstats_entry = *it;
    bool upload_started = false;
    bool upload_aborted = false;
    if (opstats_entry.task_name() != task_name) {
      continue;
    }
    for (const auto& event : opstats_entry.events()) {
      if (event.event_type() ==
          OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED) {
        upload_started = true;
      }
      if (event.event_type() ==
          OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_SERVER_ABORTED) {
        upload_aborted = true;
      }
    }
    if (upload_started && !upload_aborted) {
      return opstats_entry;
    }
  }
  return std::nullopt;
}

}  // namespace opstats
}  // namespace client
}  // namespace fcp
