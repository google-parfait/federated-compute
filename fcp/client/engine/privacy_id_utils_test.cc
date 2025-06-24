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

#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/civil_time.h"
#include "fcp/protos/confidentialcompute/selection_criteria.pb.h"
#include "fcp/protos/confidentialcompute/windowing_schedule.pb.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace client {
namespace engine {
namespace {

using ::fcp::confidentialcompute::WindowingSchedule;
using ::fedsql::PrivacyIdConfig;
using ::testing::HasSubstr;

MATCHER_P2(StatusIs, expected_code, message_matcher, "") {
  return ExplainMatchResult(IsCode(expected_code), arg, result_listener) &&
         ExplainMatchResult(message_matcher, arg.status().message(),
                            result_listener);
}

MATCHER_P(IsOkAndHolds, m, "") {
  return testing::ExplainMatchResult(IsOk(), arg, result_listener) &&
         testing::ExplainMatchResult(m, arg.value(), result_listener);
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

}  // namespace
}  // namespace engine
}  // namespace client
}  // namespace fcp
