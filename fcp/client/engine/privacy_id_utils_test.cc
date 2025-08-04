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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/civil_time.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/protos/confidentialcompute/selection_criteria.pb.h"
#include "fcp/protos/confidentialcompute/windowing_schedule.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace client {
namespace engine {
namespace {

using ::fcp::confidentialcompute::WindowingSchedule;
using ::fedsql::PrivacyIdConfig;
using ::google::internal::federated::plan::ExampleQuerySpec;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

MATCHER_P2(StatusIs, expected_code, message_matcher, "") {
  return ExplainMatchResult(IsCode(expected_code), arg, result_listener) &&
         ExplainMatchResult(message_matcher, arg.status().message(),
                            result_listener);
}

MATCHER_P(IsOkAndHolds, m, "") {
  return testing::ExplainMatchResult(IsOk(), arg, result_listener) &&
         testing::ExplainMatchResult(m, arg.value(), result_listener);
}

MATCHER_P2(PerPrivacyIdResultIs, privacy_id, example_query_result, "") {
  return ExplainMatchResult(testing::Eq(privacy_id), arg.privacy_id,
                            result_listener) &&
         ExplainMatchResult(EqualsProto(example_query_result),
                            arg.example_query_result, result_listener);
}

PrivacyIdConfig CreatePrivacyIdConfig(
    int window_size,
    WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::Unit unit,
    int start_year, int start_month, int start_day,
    std::optional<int> start_hours = std::nullopt,
    std::optional<int> start_minutes = std::nullopt,
    std::optional<int> start_seconds = std::nullopt) {
  PrivacyIdConfig config;
  auto* schedule =
      config.mutable_windowing_schedule()->mutable_civil_time_window_schedule();
  schedule->mutable_size()->set_size(window_size);
  schedule->mutable_size()->set_unit(unit);
  schedule->mutable_shift()->set_size(window_size);
  schedule->mutable_shift()->set_unit(unit);
  if (start_year != 0) schedule->mutable_start_date()->set_year(start_year);
  if (start_month != 0) schedule->mutable_start_date()->set_month(start_month);
  if (start_day != 0) schedule->mutable_start_date()->set_day(start_day);
  if (start_hours.has_value())
    schedule->mutable_start_time()->set_hours(*start_hours);
  if (start_minutes.has_value())
    schedule->mutable_start_time()->set_minutes(*start_minutes);
  if (start_seconds.has_value())
    schedule->mutable_start_time()->set_seconds(*start_seconds);
  schedule->set_time_zone_scheme(
      WindowingSchedule::CivilTimeWindowSchedule::IGNORE);
  return config;
}

TEST(PrivacyIdUtilsTest, GetWindowStartHourly) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/6,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::HOURS,
      /*start_year=*/2024, /*start_month=*/1,
      /*start_day=*/1, /*start_hours=*/0);
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-01T13:05:00+00:00"),
              IsOkAndHolds(absl::CivilSecond(2024, 1, 1, 12, 0, 0)));
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-01T00:00:00+00:00"),
              IsOkAndHolds(absl::CivilSecond(2024, 1, 1, 0, 0, 0)));
}

TEST(PrivacyIdUtilsTest, GetWindowStartDaily) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/1);
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-05T13:05:00+00:00"),
              IsOkAndHolds(absl::CivilSecond(2024, 1, 5, 0, 0, 0)));
}

TEST(PrivacyIdUtilsTest, GetWindowStartMonthly) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::MONTHS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/0);
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-03-05T13:05:00+00:00"),
              IsOkAndHolds(absl::CivilSecond(2024, 3, 1, 0, 0, 0)));
}

TEST(PrivacyIdUtilsTest, GetWindowStartYearly) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::YEARS,
      /*start_year=*/2024, /*start_month=*/0, /*start_day=*/0);
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2025-03-05T13:05:00+00:00"),
              IsOkAndHolds(absl::CivilSecond(2025, 1, 1, 0, 0, 0)));
}

TEST(PrivacyIdUtilsTest, GetWindowStartTimezoneIgnored) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      6, WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::HOURS, 2024, 1,
      1, /*start_hours=*/0);
  // The timezone modifier should be ignored, so +00:00 and -08:00 should yield
  // the same window start.
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-01T13:05:00+00:00"),
              IsOkAndHolds(absl::CivilSecond(2024, 1, 1, 12, 0, 0)));
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-01T13:05:00-08:00"),
              IsOkAndHolds(absl::CivilSecond(2024, 1, 1, 12, 0, 0)));
}

TEST(PrivacyIdUtilsTest, GetWindowStartEventBeforeStart) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/10);
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-05T13:05:00+00:00"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Event time is before the schedule start")));
}

TEST(PrivacyIdUtilsTest, InvalidEventTimeFormat) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/1);
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-05T13:05:00"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid event time format")));
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-05 13:05:00+00:00"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid event time format")));
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-05T13:05:00Z"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid event time format")));
}

TEST(PrivacyIdUtilsTest, MissingWindowSchedule) {
  PrivacyIdConfig config;
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-01T13:05:00+00:00"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Missing CivilTimeWindowSchedule")));
}

TEST(PrivacyIdUtilsTest, NonTumblingWindow) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/1);
  config.mutable_windowing_schedule()
      ->mutable_civil_time_window_schedule()
      ->mutable_shift()
      ->set_size(2);
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-01T13:05:00+00:00"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Only tumbling windows are supported")));
}

TEST(PrivacyIdUtilsTest, InvalidStartTimeForHourWindow) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::HOURS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/1,
      /*start_hours=*/0,
      /*start_minutes=*/1, /*start_seconds=*/0);  // Minutes not 0
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-01T13:05:00+00:00"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Start time minutes, seconds, and nanos must "
                                 "be 0 for HOUR windows")));
}

TEST(PrivacyIdUtilsTest, InvalidStartTimeForDayWindow) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/1,
      /*start_hours=*/1,
      /*start_minutes=*/0, /*start_seconds=*/0);  // Hours not 0
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-01T13:05:00+00:00"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Start time must be unset for DAY windows")));
}

TEST(PrivacyIdUtilsTest, InvalidStartTimeForMonthWindow) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::MONTHS,
      /*start_year=*/2024, /*start_month=*/1,
      /*start_day=*/2);  // Day not 0
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-01T13:05:00+00:00"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Start date day and start time must be unset "
                                 "for MONTH windows")));
}

TEST(PrivacyIdUtilsTest, InvalidStartTimeForYearWindow) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::YEARS,
      /*start_year=*/2024, /*start_month=*/2,
      /*start_day=*/0);  // Month not 0
  EXPECT_THAT(GetPrivacyIdTimeWindowStart(config, "2024-01-01T13:05:00+00:00"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Start date month, day, and time must be "
                                 "unset for YEAR windows")));
}

TEST(PrivacyIdUtilsTest, GetPrivacyId) {
  absl::string_view source_id = "test_source";
  absl::CivilSecond window_start(2024, 5, 15, 10, 0, 0);

  absl::StatusOr<std::string> id1 = GetPrivacyId(source_id, window_start);
  ASSERT_OK(id1);
  absl::StatusOr<std::string> id2 = GetPrivacyId(source_id, window_start);
  EXPECT_EQ(*id1, *id2);

  absl::CivilSecond different_window(2024, 5, 15, 11, 0, 0);
  absl::StatusOr<std::string> id3 = GetPrivacyId(source_id, different_window);
  EXPECT_NE(*id1, *id3);

  absl::StatusOr<std::string> id4 = GetPrivacyId("other_source", window_start);
  EXPECT_NE(*id1, *id4);
}

ExampleQuerySpec::ExampleQuery CreateExampleQuery() {
  return PARSE_TEXT_PROTO(R"pb(
    example_selector { collection_uri: "app:/test_collection" }
  )pb");
}

ExampleQueryResult CreateExampleQueryResult(
    const std::vector<std::string>& event_times,
    const std::vector<int64_t>& values) {
  ExampleQueryResult result;
  auto& vectors = *result.mutable_vector_data()->mutable_vectors();

  ExampleQueryResult::VectorData::Values event_time_values;
  event_time_values.mutable_string_values()->mutable_value()->Add(
      event_times.begin(), event_times.end());
  vectors[confidential_compute::kEventTimeColumnName] = event_time_values;

  ExampleQueryResult::VectorData::Values data_values;
  data_values.mutable_int64_values()->mutable_value()->Add(values.begin(),
                                                           values.end());
  vectors["test_data"] = data_values;
  return result;
}

ExampleQueryResult AddPrivacyIdColumn(ExampleQueryResult result,
                                      const std::string privacy_id) {
  ExampleQueryResult::VectorData::Values privacy_id_values;
  privacy_id_values.mutable_bytes_values()->add_value(privacy_id);
  (*result.mutable_vector_data()
        ->mutable_vectors())[confidential_compute::kPrivacyIdColumnName] =
      privacy_id_values;
  return result;
}

TEST(PrivacyIdUtilsTest, SplitResultsByPrivacyIdSingleQuery) {
  // With a window size of 1 day and rows with event times from Jan 1, 2024
  // to Jan 2, 2024, expect the rows to be split into two separate results,
  // one for Jan 1, 2024 and one for Jan 2, 2024.
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/1);
  std::vector<std::pair<ExampleQuerySpec::ExampleQuery, ExampleQueryResult>>
      results;
  results.push_back({CreateExampleQuery(),
                     CreateExampleQueryResult({"2024-01-01T10:00:00+00:00",
                                               "2024-01-02T12:00:00+00:00",
                                               "2024-01-01T08:00:00-09:00"},
                                              {1, 2, 3})});

  absl::StatusOr<std::vector<SplitResults>> split_results =
      SplitResultsByPrivacyId(results, config, "test_source");
  ASSERT_OK(split_results);
  ASSERT_EQ(split_results->size(), 1);

  absl::StatusOr<std::string> privacy_id_1 =
      GetPrivacyId("test_source", absl::CivilDay(2024, 1, 1));
  ASSERT_OK(privacy_id_1);
  absl::StatusOr<std::string> privacy_id_2 =
      GetPrivacyId("test_source", absl::CivilDay(2024, 1, 2));
  ASSERT_OK(privacy_id_2);

  EXPECT_THAT(
      (*split_results)[0].per_privacy_id_results,
      UnorderedElementsAre(
          PerPrivacyIdResultIs(
              *privacy_id_1,
              AddPrivacyIdColumn(
                  CreateExampleQueryResult({"2024-01-01T10:00:00+00:00",
                                            "2024-01-01T08:00:00-09:00"},
                                           {1, 3}),
                  *privacy_id_1)),
          PerPrivacyIdResultIs(
              *privacy_id_2,
              AddPrivacyIdColumn(
                  CreateExampleQueryResult({"2024-01-02T12:00:00+00:00"}, {2}),
                  *privacy_id_2))));
}

TEST(PrivacyIdUtilsTest, SplitResultsByPrivacyIdMultipleQueries) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/1);
  std::vector<std::pair<ExampleQuerySpec::ExampleQuery, ExampleQueryResult>>
      results;
  //  With a privacy ID window size of 1 day and rows with event times from
  //  Jan 1, 2024 to Jan 3, 2024, expect the rows to be split into three
  //  separate results, one for each day
  results.push_back({CreateExampleQuery(),
                     CreateExampleQueryResult({"2024-01-01T10:00:00+00:00",
                                               "2024-01-02T12:00:00+00:00",
                                               "2024-01-01T08:00:00-09:00"},
                                              {1, 2, 3})});
  //  With a privacy ID window size of 1 day and rows with event times from
  //  Jan 2, 2024 to Jan 3, 2024, expect the rows to be split into two
  //  separate results
  results.push_back({CreateExampleQuery(),
                     CreateExampleQueryResult({"2024-01-03T10:00:00+00:00",
                                               "2024-01-02T12:00:00+00:00"},
                                              {4, 5})});

  absl::StatusOr<std::vector<SplitResults>> split_results =
      SplitResultsByPrivacyId(results, config, "test_source");
  ASSERT_OK(split_results);
  ASSERT_EQ(split_results->size(), 2);

  absl::StatusOr<std::string> privacy_id_1 =
      GetPrivacyId("test_source", absl::CivilDay(2024, 1, 1));
  ASSERT_OK(privacy_id_1);
  absl::StatusOr<std::string> privacy_id_2 =
      GetPrivacyId("test_source", absl::CivilDay(2024, 1, 2));
  ASSERT_OK(privacy_id_2);
  absl::StatusOr<std::string> privacy_id_3 =
      GetPrivacyId("test_source", absl::CivilDay(2024, 1, 3));
  ASSERT_OK(privacy_id_3);

  EXPECT_THAT(
      (*split_results)[0].per_privacy_id_results,
      UnorderedElementsAre(
          PerPrivacyIdResultIs(
              *privacy_id_1,
              AddPrivacyIdColumn(
                  CreateExampleQueryResult({"2024-01-01T10:00:00+00:00",
                                            "2024-01-01T08:00:00-09:00"},
                                           {1, 3}),
                  *privacy_id_1)),
          PerPrivacyIdResultIs(
              *privacy_id_2,
              AddPrivacyIdColumn(
                  CreateExampleQueryResult({"2024-01-02T12:00:00+00:00"}, {2}),
                  *privacy_id_2))));

  EXPECT_THAT(
      (*split_results)[1].per_privacy_id_results,
      UnorderedElementsAre(
          PerPrivacyIdResultIs(
              *privacy_id_3,
              AddPrivacyIdColumn(
                  CreateExampleQueryResult({"2024-01-03T10:00:00+00:00"}, {4}),
                  *privacy_id_3)),
          PerPrivacyIdResultIs(
              *privacy_id_2,
              AddPrivacyIdColumn(
                  CreateExampleQueryResult({"2024-01-02T12:00:00+00:00"}, {5}),
                  *privacy_id_2))));
}

TEST(PrivacyIdUtilsTest, SplitResultsByPrivacyIdNoEventTimeColumn) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/1);
  std::vector<std::pair<ExampleQuerySpec::ExampleQuery, ExampleQueryResult>>
      results;
  ExampleQueryResult query_result =
      CreateExampleQueryResult({"2024-01-01T12:00:00+00:00"}, {1});
  query_result.mutable_vector_data()->mutable_vectors()->erase(
      confidential_compute::kEventTimeColumnName);
  results.push_back({CreateExampleQuery(), query_result});

  EXPECT_THAT(SplitResultsByPrivacyId(results, config, "test_source"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Required column not found")));
}

TEST(PrivacyIdUtilsTest, SplitResultsByPrivacyIdWrongEventTimeType) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/1);
  std::vector<std::pair<ExampleQuerySpec::ExampleQuery, ExampleQueryResult>>
      results;
  ExampleQueryResult query_result =
      CreateExampleQueryResult({"to be replaced"}, {1});
  // Replace string event time with invalid event time type int64
  (*query_result.mutable_vector_data()
        ->mutable_vectors())[confidential_compute::kEventTimeColumnName]
      .mutable_int64_values()
      ->add_value(12345);
  results.push_back({CreateExampleQuery(), query_result});

  EXPECT_THAT(
      SplitResultsByPrivacyId(results, config, "test_source"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Unexpected data type for event time column, expected string")));
}

TEST(PrivacyIdUtilsTest, SplitResultsByPrivacyIdInvalidEventTimeFormat) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/1);
  std::vector<std::pair<ExampleQuerySpec::ExampleQuery, ExampleQueryResult>>
      results;
  results.push_back({CreateExampleQuery(),
                     CreateExampleQueryResult({"invalid event time"}, {1})});

  EXPECT_THAT(SplitResultsByPrivacyId(results, config, "test_source"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid event time format")));
}

TEST(PrivacyIdUtilsTest, SplitResultsByPrivacyIdAlreadyHasPrivacyId) {
  PrivacyIdConfig config = CreatePrivacyIdConfig(
      /*window_size=*/1,
      WindowingSchedule::CivilTimeWindowSchedule::TimePeriod::DAYS,
      /*start_year=*/2024, /*start_month=*/1, /*start_day=*/1);
  std::vector<std::pair<ExampleQuerySpec::ExampleQuery, ExampleQueryResult>>
      results;
  results.push_back({CreateExampleQuery(),
                     AddPrivacyIdColumn(CreateExampleQueryResult(
                                            {"2024-01-01T10:00:00+00:00"}, {1}),
                                        "existing_id")});

  EXPECT_THAT(
      SplitResultsByPrivacyId(results, config, "test_source"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Privacy ID column cannot already exist in the "
                         "example query result")));
}

}  // namespace
}  // namespace engine
}  // namespace client
}  // namespace fcp
