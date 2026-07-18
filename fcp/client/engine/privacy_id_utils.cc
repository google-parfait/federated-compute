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

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "google/type/datetime.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/civil_time.h"
#include "fcp/base/digest.h"
#include "fcp/client/event_time_range.pb.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/confidentialcompute/time_window_utilities.h"
#include "fcp/protos/confidentialcompute/windowing_schedule.pb.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace engine {

namespace {

using ::fcp::confidentialcompute::WindowingSchedule;
using ::google::internal::federated::plan::ExampleQuerySpec;
using ::google::internal::federated::plan::PrivacyIdConfig;

constexpr int kPrivacyIdLength = 16;

absl::StatusOr<WindowingSchedule::CivilTimeWindowSchedule>
VerifyConfigHasCivilTimeWindowSchedule(
    const PrivacyIdConfig& privacy_id_config) {
  if (!privacy_id_config.has_windowing_schedule() ||
      !privacy_id_config.windowing_schedule()
           .has_civil_time_window_schedule()) {
    return absl::InvalidArgumentError("Missing CivilTimeWindowSchedule");
  }
  return privacy_id_config.windowing_schedule().civil_time_window_schedule();
}

template <typename ValuesType>
void CopySelectedValues(const ValuesType& source, ValuesType* dest,
                        const std::vector<int>& indices) {
  for (int index : indices) {
    dest->add_value(source.value(index));
  }
}

google::type::DateTime ConvertCivilHourToDateTime(absl::CivilHour civil_hour) {
  google::type::DateTime date_time;
  date_time.set_year(static_cast<int32_t>(civil_hour.year()));
  date_time.set_month(civil_hour.month());
  date_time.set_day(civil_hour.day());
  date_time.set_hours(civil_hour.hour());
  return date_time;
}

struct ExampleQueryResultSelection {
  std::vector<int> indices;
  // Use CivilHour as the finest granularity since this range will be stored in
  // the unencrypted blob header.
  absl::CivilHour min_event_time = absl::CivilHour::max();
  absl::CivilHour max_event_time = absl::CivilHour::min();
};

absl::StatusOr<ExampleQueryResult> CreateExampleQueryResultFromSelection(
    const ExampleQueryResult& original_result,
    const ExampleQueryResultSelection& selection) {
  ExampleQueryResult new_result;
  for (const auto& [column_name, values] :
       original_result.vector_data().vectors()) {
    ExampleQueryResult::VectorData::Values new_values;
    switch (values.values_case()) {
      case ExampleQueryResult::VectorData::Values::kInt32Values:
        CopySelectedValues(values.int32_values(),
                           new_values.mutable_int32_values(),
                           selection.indices);
        break;
      case ExampleQueryResult::VectorData::Values::kInt64Values:
        CopySelectedValues(values.int64_values(),
                           new_values.mutable_int64_values(),
                           selection.indices);
        break;
      case ExampleQueryResult::VectorData::Values::kStringValues:
        CopySelectedValues(values.string_values(),
                           new_values.mutable_string_values(),
                           selection.indices);
        break;
      case ExampleQueryResult::VectorData::Values::kBoolValues:
        CopySelectedValues(values.bool_values(),
                           new_values.mutable_bool_values(), selection.indices);
        break;
      case ExampleQueryResult::VectorData::Values::kFloatValues:
        CopySelectedValues(values.float_values(),
                           new_values.mutable_float_values(),
                           selection.indices);
        break;
      case ExampleQueryResult::VectorData::Values::kDoubleValues:
        CopySelectedValues(values.double_values(),
                           new_values.mutable_double_values(),
                           selection.indices);
        break;
      case ExampleQueryResult::VectorData::Values::kBytesValues:
        CopySelectedValues(values.bytes_values(),
                           new_values.mutable_bytes_values(),
                           selection.indices);
        break;
      case ExampleQueryResult::VectorData::Values::VALUES_NOT_SET:
        return absl::InvalidArgumentError(
            "Invalid values type in the example query result");
    }
    (*new_result.mutable_vector_data()->mutable_vectors())[column_name] =
        new_values;
  }
  new_result.mutable_stats()->set_output_rows_count(
      static_cast<int32_t>(selection.indices.size()));
  if (!selection.indices.empty()) {
    *new_result.mutable_stats()
         ->mutable_cross_query_event_time_range()
         ->mutable_start_event_time() =
        ConvertCivilHourToDateTime(selection.min_event_time);
    *new_result.mutable_stats()
         ->mutable_cross_query_event_time_range()
         ->mutable_end_event_time() =
        ConvertCivilHourToDateTime(selection.max_event_time);
  }
  return new_result;
}

absl::StatusOr<std::string> GetPrivacyId(absl::string_view source_id) {
  std::string hash = ComputeSHA256(source_id);
  if (hash.size() < kPrivacyIdLength) {
    return absl::InternalError("SHA256 hash is too short");
  }
  return hash.substr(0, kPrivacyIdLength);
}

absl::StatusOr<std::string> GetPrivacyId(absl::string_view source_id,
                                         absl::CivilSecond window_start) {
  return GetPrivacyId(
      absl::StrCat(source_id, absl::FormatCivilTime(window_start)));
}

// Splits rows into groups by time window, assigning each group a rotating
// privacy ID derived from (source_id, window_start). Each row's event time
// determines which window it falls into, and rows in the same window share the
// same privacy ID. The resulting SplitResults contains one PerPrivacyIdResult
// per window, each with the subset of rows for that window and the event time
// range spanning those rows.
absl::StatusOr<SplitResults> SplitResultsByWindowedPrivacyId(
    ExampleQuerySpec::ExampleQuery example_query,
    const ExampleQueryResult& example_query_result,
    const PrivacyIdConfig& privacy_id_config,
    const ExampleQueryResult::VectorData::Values& event_time_values,
    absl::string_view source_id) {
  // For each row, parse the event time, determine the time window it belongs
  // to, and compute the privacy ID for that window. Group row indices by
  // privacy ID, tracking the min/max event times per group for the event time
  // range metadata.
  absl::flat_hash_map<std::string, ExampleQueryResultSelection>
      privacy_id_selections;
  for (int i = 0; i < event_time_values.string_values().value_size(); ++i) {
    const std::string& event_time_str =
        event_time_values.string_values().value(i);
    ABSL_ASSIGN_OR_RETURN(
        absl::CivilSecond event_civil_second,
        confidentialcompute::ConvertEventTimeToCivilSecond(event_time_str));

    ABSL_ASSIGN_OR_RETURN(
        WindowingSchedule::CivilTimeWindowSchedule schedule,
        VerifyConfigHasCivilTimeWindowSchedule(privacy_id_config));
    ABSL_ASSIGN_OR_RETURN(absl::CivilSecond window_start,
                          GetTimeWindowStart(schedule, event_civil_second));
    ABSL_ASSIGN_OR_RETURN(std::string privacy_id,
                          GetPrivacyId(source_id, window_start));

    ExampleQueryResultSelection& selection = privacy_id_selections[privacy_id];
    selection.indices.push_back(i);
    selection.min_event_time =
        std::min(absl::CivilHour(event_civil_second), selection.min_event_time);
    selection.max_event_time =
        std::max(absl::CivilHour(event_civil_second), selection.max_event_time);
  }

  // Build a PerPrivacyIdResult for each window by selecting the rows that
  // belong to that window from the original result.
  SplitResults split_results = {.example_query = std::move(example_query)};
  split_results.per_privacy_id_results.reserve(privacy_id_selections.size());
  for (const auto& [privacy_id, selection] : privacy_id_selections) {
    PerPrivacyIdResult per_privacy_id_result;
    per_privacy_id_result.privacy_id = privacy_id;
    ABSL_ASSIGN_OR_RETURN(
        ExampleQueryResult per_privacy_id_example_query_result,
        CreateExampleQueryResultFromSelection(example_query_result, selection));

    per_privacy_id_result.example_query_result =
        std::move(per_privacy_id_example_query_result);

    split_results.per_privacy_id_results.push_back(
        std::move(per_privacy_id_result));
  }
  return split_results;
}

// Assigns all rows a single non-rotating privacy ID derived from only the
// source_id (no time component). Since all rows share the same privacy ID,
// there is no need to split or re-index the rows — the entire
// ExampleQueryResult is copied directly. The event time range is computed from
// the min/max event times across all rows.
absl::StatusOr<SplitResults> SplitResultsByNonRotatingPrivacyId(
    ExampleQuerySpec::ExampleQuery example_query,
    const ExampleQueryResult& example_query_result,
    const ExampleQueryResult::VectorData::Values& event_time_values,
    absl::string_view source_id) {
  ABSL_ASSIGN_OR_RETURN(std::string privacy_id, GetPrivacyId(source_id));

  SplitResults split_results = {.example_query = std::move(example_query)};
  if (example_query_result.stats().output_rows_count() == 0) {
    return split_results;
  }

  // Copy the entire result since all rows belong to the same privacy ID.
  PerPrivacyIdResult per_privacy_id_result;
  per_privacy_id_result.privacy_id = privacy_id;
  per_privacy_id_result.example_query_result = example_query_result;

  // Compute the event time range across all rows. Event times are truncated to
  // CivilHour granularity because this range is stored in the unencrypted blob
  // header.
  absl::CivilHour min_time = absl::CivilHour::max();
  absl::CivilHour max_time = absl::CivilHour::min();
  for (int i = 0; i < event_time_values.string_values().value_size(); ++i) {
    ABSL_ASSIGN_OR_RETURN(absl::CivilSecond event_civil_second,
                          confidentialcompute::ConvertEventTimeToCivilSecond(
                              event_time_values.string_values().value(i)));
    min_time = std::min(absl::CivilHour(event_civil_second), min_time);
    max_time = std::max(absl::CivilHour(event_civil_second), max_time);
  }
  *per_privacy_id_result.example_query_result.mutable_stats()
       ->mutable_cross_query_event_time_range()
       ->mutable_start_event_time() = ConvertCivilHourToDateTime(min_time);
  *per_privacy_id_result.example_query_result.mutable_stats()
       ->mutable_cross_query_event_time_range()
       ->mutable_end_event_time() = ConvertCivilHourToDateTime(max_time);

  split_results.per_privacy_id_results.push_back(
      std::move(per_privacy_id_result));
  return split_results;
}

}  // namespace

absl::StatusOr<SplitResults> SplitResultsByPrivacyId(
    ExampleQuerySpec::ExampleQuery example_query,
    const ExampleQueryResult& example_query_result,
    const PrivacyIdConfig& privacy_id_config, absl::string_view source_id,
    bool enable_privacy_id_v2) {
  // Find the event time column, which is required for both rotating and
  // non-rotating privacy IDs. The column name has a query-specific prefix, so
  // we match by suffix.
  const ExampleQueryResult::VectorData::Values* event_time_values = nullptr;

  for (const auto& [column_name, values] :
       example_query_result.vector_data().vectors()) {
    if (absl::EndsWith(column_name,
                       confidential_compute::kEventTimeColumnName)) {
      if (event_time_values != nullptr) {
        return absl::InvalidArgumentError(
            absl::StrCat("Multiple columns found ending with ",
                         confidential_compute::kEventTimeColumnName));
      }
      event_time_values = &values;
    }
  }

  if (event_time_values == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Required column ending with %s not found in the example query result",
        confidential_compute::kEventTimeColumnName));
  }

  if (!event_time_values->has_string_values()) {
    return absl::InvalidArgumentError(
        "Unexpected data type for event time column, expected string.");
  }
  if (example_query_result.vector_data().vectors().contains(
          confidential_compute::kPrivacyIdColumnName)) {
    return absl::InvalidArgumentError(
        "Privacy ID column cannot already exist in the example query result");
  }

  // Allow non-rotating privacy IDs when the flag is enabled and the windowing
  // schedule is missing.
  if (enable_privacy_id_v2 && !privacy_id_config.has_windowing_schedule()) {
    return SplitResultsByNonRotatingPrivacyId(std::move(example_query),
                                              example_query_result,
                                              *event_time_values, source_id);
  }
  // When the windowing schedule is present, this computes rotating privacy IDs.
  // When missing, this returns an error (preserving the original behavior when
  // the flag is off).
  return SplitResultsByWindowedPrivacyId(
      std::move(example_query), example_query_result, privacy_id_config,
      *event_time_values, source_id);
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
