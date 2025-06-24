/*
 * Copyright 2025 Google LLC
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
#include "fcp/client/engine/privacy_id_utils.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/civil_time.h"
#include "fcp/base/digest.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/selection_criteria.pb.h"
#include "fcp/protos/confidentialcompute/windowing_schedule.pb.h"

namespace fcp {
namespace client {
namespace engine {

namespace {

using ::fcp::confidentialcompute::WindowingSchedule;

// Remove the timezone modifier from the event time. Event_time must be in the
// format YYYY-MM-DDTHH:MM:SS[+-]HH:MM, and the result will be in the format
// YYYY-MM-DDTHH:MM:SS.
absl::StatusOr<std::string> RemoveTimezoneFromEventTime(
    absl::string_view event_time) {
  if (event_time.length() != 25) {
    return absl::InvalidArgumentError(
        "Invalid event time format: incorrect length");
  }
  // Basic check for the presence of 'T' and timezone sign.
  if (event_time[10] != 'T' ||
      (event_time[19] != '+' && event_time[19] != '-')) {
    return absl::InvalidArgumentError(
        "Invalid event time format: missing T or timezone modifier");
  }
  return std::string(event_time.substr(0, 19));
}

// Template helper function to calculate the window start for any CivilTime
// type.
template <typename CivilTimeType>
absl::StatusOr<absl::CivilSecond> CalculateWindowStart(
    const WindowingSchedule::CivilTimeWindowSchedule& schedule,
    absl::CivilSecond event_civil_second) {
  const auto& start_date = schedule.start_date();
  const auto& start_time = schedule.start_time();
  CivilTimeType schedule_start(start_date.year(), start_date.month(),
                               start_date.day(), start_time.hours(),
                               start_time.minutes(), start_time.seconds());
  CivilTimeType event_civil_time(event_civil_second);

  if (event_civil_time < schedule_start) {
    return absl::InvalidArgumentError(
        "Event time is before the schedule start");
  }
  // Calculate the number of windows that elapsed between the schedule start
  // and the event time.
  absl::civil_diff_t diff = event_civil_time - schedule_start;
  int64_t window_size = schedule.size().size();
  absl::civil_diff_t window_index = diff / window_size;
  CivilTimeType window_start = schedule_start + window_index * window_size;
  return absl::CivilSecond(window_start);
}

absl::StatusOr<WindowingSchedule::CivilTimeWindowSchedule>
ValidateCivilTimeWindowSchedule(
    const fedsql::PrivacyIdConfig& privacy_id_config) {
  if (!privacy_id_config.has_windowing_schedule() ||
      !privacy_id_config.windowing_schedule()
           .has_civil_time_window_schedule()) {
    return absl::InvalidArgumentError("Missing CivilTimeWindowSchedule");
  }
  const auto& schedule =
      privacy_id_config.windowing_schedule().civil_time_window_schedule();

  if (schedule.time_zone_scheme() !=
      WindowingSchedule::CivilTimeWindowSchedule::IGNORE) {
    return absl::UnimplementedError(
        "Only IGNORE time zone scheme is supported");
  }
  if (schedule.size().size() != schedule.shift().size() ||
      schedule.size().unit() != schedule.shift().unit()) {
    return absl::UnimplementedError("Only tumbling windows are supported");
  }
  return schedule;
}

}  // namespace

absl::StatusOr<absl::CivilSecond> GetPrivacyIdTimeWindowStart(
    const fedsql::PrivacyIdConfig& privacy_id_config,
    absl::string_view event_time) {
  FCP_ASSIGN_OR_RETURN(WindowingSchedule::CivilTimeWindowSchedule schedule,
                       ValidateCivilTimeWindowSchedule(privacy_id_config));

  FCP_ASSIGN_OR_RETURN(std::string event_time_without_timezone,
                       RemoveTimezoneFromEventTime(event_time));
  absl::CivilSecond event_civil_second;
  if (!absl::ParseCivilTime(event_time_without_timezone, &event_civil_second)) {
    return absl::InvalidArgumentError("Invalid event time format");
  }

  const auto& start_date = schedule.start_date();

  switch (schedule.size().unit()) {
    case WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::HOURS: {
      const auto& start_time = schedule.start_time();
      if (start_time.minutes() != 0 || start_time.seconds() != 0 ||
          start_time.nanos() != 0) {
        return absl::InvalidArgumentError(
            "Start time minutes, seconds, and nanos must be 0 for HOUR "
            "windows");
      }
      return CalculateWindowStart<absl::CivilHour>(schedule,
                                                   event_civil_second);
    }
    case WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS:
      if (schedule.has_start_time()) {
        return absl::InvalidArgumentError(
            "Start time must be unset for DAY windows");
      }
      return CalculateWindowStart<absl::CivilDay>(schedule, event_civil_second);
    case WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::MONTHS:
      if (start_date.day() != 0 || schedule.has_start_time()) {
        return absl::InvalidArgumentError(
            "Start date day and start time must be unset for MONTH windows");
      }
      return CalculateWindowStart<absl::CivilMonth>(schedule,
                                                    event_civil_second);
    case WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::YEARS:
      if (start_date.month() != 0 || start_date.day() != 0 ||
          schedule.has_start_time()) {
        return absl::InvalidArgumentError(
            "Start date month, day, and time must be unset for YEAR windows");
      }
      return CalculateWindowStart<absl::CivilYear>(schedule,
                                                   event_civil_second);
    default:
      return absl::UnimplementedError("Unsupported windowing schedule unit");
  }
}

absl::StatusOr<std::string> GetPrivacyId(absl::string_view source_id,
                                         absl::CivilSecond window_start) {
  std::string hash = ComputeSHA256(
      absl::StrCat(source_id, absl::FormatCivilTime(window_start)));
  if (hash.size() < 16) {
    return absl::InternalError("SHA256 hash is too short");
  }
  return hash.substr(0, 16);
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
