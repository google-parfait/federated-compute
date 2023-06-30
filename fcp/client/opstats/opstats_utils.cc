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
#include <functional>
#include <optional>
#include <string>

#include "google/protobuf/timestamp.pb.h"
#include "absl/strings/string_view.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/opstats/opstats_db.h"
#include "fcp/protos/opstats.pb.h"

namespace fcp {
namespace client {
namespace opstats {
namespace {

std::optional<OperationalStats> GetLastSuccessfulContributionForPredicate(
    const OpStatsSequence& data,
    std::function<bool(const OperationalStats&)> predicate) {
  for (auto it = data.opstats().rbegin(); it != data.opstats().rend(); ++it) {
    const OperationalStats& opstats_entry = *it;
    bool upload_started = false;
    bool upload_aborted = false;
    if (!predicate(opstats_entry)) {
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

// We use the timestamp of the RESULT_UPLOAD_STARTED event for the contribution
// time.
std::optional<google::protobuf::Timestamp> GetContributionTimeForEntry(
    const OperationalStats& entry) {
  auto upload_started = std::find_if(
      entry.events().begin(), entry.events().end(),
      [](const OperationalStats::Event& event) {
        return event.event_type() ==
               OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED;
      });
  if (upload_started == entry.events().end()) {
    // For last_successful_entry to have a value, it must have had an
    // EVENT_KIND_RESULT_UPLOAD_STARTED event, so we should never reach this.
    return std::nullopt;
  }

  return upload_started->timestamp();
}

}  // anonymous namespace

std::optional<OperationalStats> GetLastSuccessfulContribution(
    const OpStatsSequence& data, absl::string_view task_name) {
  return GetLastSuccessfulContributionForPredicate(
      data, [task_name](const OperationalStats& opstats_entry) {
        return opstats_entry.task_name() == task_name;
      });
}

std::optional<google::protobuf::Timestamp> GetLastSuccessfulContributionTime(
    const OpStatsSequence& data, absl::string_view task_name) {
  std::optional<OperationalStats> last_successful_entry =
      GetLastSuccessfulContribution(data, task_name);

  if (!last_successful_entry.has_value()) {
    return std::nullopt;
  }

  return GetContributionTimeForEntry(*last_successful_entry);
}

std::optional<google::protobuf::Timestamp>
GetLastSuccessfulContributionTimeForPattern(const OpStatsSequence& data,
                                            const RE2& compiled_pattern) {
  std::optional<OperationalStats> last_successful_entry =
      GetLastSuccessfulContributionForPredicate(
          data, [&compiled_pattern](const OperationalStats& opstats_entry) {
            return RE2::FullMatch(opstats_entry.task_name(), compiled_pattern);
          });

  if (!last_successful_entry.has_value()) {
    return std::nullopt;
  }

  return GetContributionTimeForEntry(*last_successful_entry);
}

}  // namespace opstats
}  // namespace client
}  // namespace fcp
