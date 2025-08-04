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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/civil_time.h"
#include "fcp/base/digest.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/protos/confidentialcompute/selection_criteria.pb.h"
#include "fcp/protos/confidentialcompute/windowing_schedule.pb.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace engine {

namespace {

using ::fcp::confidentialcompute::WindowingSchedule;
using ::fedsql::PrivacyIdConfig;
using ::google::internal::federated::plan::ExampleQuerySpec;

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

template <typename ValuesType>
void CopySelectedValues(const ValuesType& source, ValuesType* dest,
                        const std::vector<int>& indices) {
  for (int index : indices) {
    dest->add_value(source.value(index));
  }
}

absl::StatusOr<ExampleQueryResult> CreateExampleQueryResultFromIndices(
    const ExampleQueryResult& original_result,
    const std::vector<int>& indices) {
  ExampleQueryResult new_result;
  for (const auto& [column_name, values] :
       original_result.vector_data().vectors()) {
    ExampleQueryResult::VectorData::Values new_values;
    switch (values.values_case()) {
      case ExampleQueryResult::VectorData::Values::kInt32Values:
        CopySelectedValues(values.int32_values(),
                           new_values.mutable_int32_values(), indices);
        break;
      case ExampleQueryResult::VectorData::Values::kInt64Values:
        CopySelectedValues(values.int64_values(),
                           new_values.mutable_int64_values(), indices);
        break;
      case ExampleQueryResult::VectorData::Values::kStringValues:
        CopySelectedValues(values.string_values(),
                           new_values.mutable_string_values(), indices);
        break;
      case ExampleQueryResult::VectorData::Values::kBoolValues:
        CopySelectedValues(values.bool_values(),
                           new_values.mutable_bool_values(), indices);
        break;
      case ExampleQueryResult::VectorData::Values::kFloatValues:
        CopySelectedValues(values.float_values(),
                           new_values.mutable_float_values(), indices);
        break;
      case ExampleQueryResult::VectorData::Values::kDoubleValues:
        CopySelectedValues(values.double_values(),
                           new_values.mutable_double_values(), indices);
        break;
      case ExampleQueryResult::VectorData::Values::kBytesValues:
        CopySelectedValues(values.bytes_values(),
                           new_values.mutable_bytes_values(), indices);
        break;
      case ExampleQueryResult::VectorData::Values::VALUES_NOT_SET:
        return absl::InvalidArgumentError(
            "Invalid values type in the example query result");
    }
    (*new_result.mutable_vector_data()->mutable_vectors())[column_name] =
        new_values;
  }
  return new_result;
}

// Adds the privacy ID to the example query result.
absl::Status AddPrivacyIdColumn(absl::string_view privacy_id,
                                ExampleQueryResult& example_query_result) {
  if (example_query_result.vector_data().vectors().contains(
          confidential_compute::kPrivacyIdColumnName)) {
    return absl::InvalidArgumentError(
        "Privacy ID column cannot already exist in the example query result");
  }
  ExampleQueryResult::VectorData::Values privacy_id_values;
  privacy_id_values.mutable_bytes_values()->add_value(std::string(privacy_id));
  (*example_query_result.mutable_vector_data()
        ->mutable_vectors())[confidential_compute::kPrivacyIdColumnName] =
      std::move(privacy_id_values);
  return absl::OkStatus();
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

absl::StatusOr<std::vector<SplitResults>> SplitResultsByPrivacyId(
    const std::vector<
        std::pair<ExampleQuerySpec::ExampleQuery, ExampleQueryResult>>&
        structured_example_query_results,
    const PrivacyIdConfig& privacy_id_config, absl::string_view source_id) {
  std::vector<SplitResults> all_split_results;

  for (const auto& [example_query, example_query_result] :
       structured_example_query_results) {
    auto it = example_query_result.vector_data().vectors().find(
        confidential_compute::kEventTimeColumnName);
    if (it == example_query_result.vector_data().vectors().end()) {
      return absl::InvalidArgumentError(
          "Required column not found in the example query result: " +
          std::string(confidential_compute::kEventTimeColumnName));
    }
    const ExampleQueryResult::VectorData::Values& event_time_values =
        it->second;
    if (!event_time_values.has_string_values()) {
      return absl::InvalidArgumentError(
          "Unexpected data type for event time column, expected string.");
    }

    // Build a map of privacy ID to the indices of the rows with that privacy
    // ID.
    absl::flat_hash_map<std::string, std::vector<int>> privacy_id_to_indices;
    for (int i = 0; i < event_time_values.string_values().value_size(); ++i) {
      const std::string& event_time_str =
          event_time_values.string_values().value(i);
      FCP_ASSIGN_OR_RETURN(
          absl::CivilSecond window_start,
          GetPrivacyIdTimeWindowStart(privacy_id_config, event_time_str));
      FCP_ASSIGN_OR_RETURN(std::string privacy_id,
                           GetPrivacyId(source_id, window_start));
      privacy_id_to_indices[privacy_id].push_back(i);
    }

    SplitResults split_results;
    split_results.example_query = example_query;
    for (const auto& [privacy_id, indices] : privacy_id_to_indices) {
      PerPrivacyIdResult per_privacy_id_result;
      per_privacy_id_result.privacy_id = privacy_id;
      FCP_ASSIGN_OR_RETURN(
          ExampleQueryResult per_privacy_id_example_query_result,
          CreateExampleQueryResultFromIndices(example_query_result, indices));

      FCP_RETURN_IF_ERROR(
          AddPrivacyIdColumn(privacy_id, per_privacy_id_example_query_result));

      per_privacy_id_result.example_query_result =
          std::move(per_privacy_id_example_query_result);

      split_results.per_privacy_id_results.push_back(
          std::move(per_privacy_id_result));
    }
    all_split_results.push_back(std::move(split_results));
  }
  return all_split_results;
}
}  // namespace engine
}  // namespace client
}  // namespace fcp
