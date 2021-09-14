/*
 * Copyright 2017 Google LLC
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

#include "fcp/base/monitoring.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace fcp {

namespace {

using ::testing::MatchesRegex;
using ::testing::Not;

MATCHER(IsOk, "") { return arg.ok(); }

MATCHER_P(IsOkAndHolds, m, "") {
  return testing::ExplainMatchResult(IsOk(), arg, result_listener) &&
         testing::ExplainMatchResult(m, arg.value(), result_listener);
}

TEST(MonitoringTest, LogInfo) {
  testing::internal::CaptureStderr();
  FCP_LOG(INFO) << "info log of something happening";
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("I.*info log of something happening\n"));
}

TEST(MonitoringTest, LogWarning) {
  testing::internal::CaptureStderr();
  FCP_LOG(WARNING) << "warning log of something happening";
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("W.*warning log of something happening\n"));
}

TEST(MonitoringTest, LogError) {
  testing::internal::CaptureStderr();
  FCP_LOG(ERROR) << "error log of something happening";
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("E.*error log of something happening\n"));
}

TEST(MonitoringDeathTest, LogFatal) {
  ASSERT_DEATH({ FCP_LOG(FATAL) << "fatal log"; }, "fatal log");
}

TEST(MonitoringTest, StatusBuilderLogInfo) {
  testing::internal::CaptureStderr();
  Status status = (FCP_STATUS(ABORTED) << "something happened").LogInfo();
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("I.*something happened\n"));
}

TEST(MonitoringTest, StatusBuilderLogWarning) {
  testing::internal::CaptureStderr();
  Status status = (FCP_STATUS(ABORTED) << "something happened").LogWarning();
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("W.*something happened\n"));
}

TEST(MonitoringTest, StatusBuilderLogError) {
  testing::internal::CaptureStderr();
  Status status = (FCP_STATUS(ABORTED) << "something happened").LogError();
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("E.*something happened\n"));
}

TEST(MonitoringDeathTest, StatusBuilderLogFatal) {
  ASSERT_DEATH(
      {
        Status status =
            (FCP_STATUS(ABORTED) << "something happened").LogFatal();
      },
      "something happened");
}

TEST(MonitoringTest, LogIfTrue) {
  testing::internal::CaptureStderr();
  FCP_LOG_IF(INFO, true) << "some log";
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("I.*some log\n"));
}

TEST(MonitoringTest, LogIfFalse) {
  testing::internal::CaptureStderr();
  FCP_LOG_IF(INFO, false) << "some log";
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_EQ(output, "");
}

TEST(MonitoringTest, CheckSucceeds) { FCP_CHECK(1 < 2); }

TEST(MonitoringDeathTest, CheckFails) {
  ASSERT_DEATH({ FCP_CHECK(1 < 0); }, "Check failed: 1 < 0.");
}

TEST(MonitoringTest, StatusOr) {
  StatusOr<int> fail_status = FCP_STATUS(ABORTED) << "operation aborted";
  ASSERT_FALSE(fail_status.ok());
  ASSERT_EQ(fail_status.status().code(), ABORTED);
  // TODO(team): Port StatusIs matcher to avoid casting message(),
  // which is string_view, to std::string.
  ASSERT_THAT(fail_status.status().message(),
              MatchesRegex(".*operation aborted"));
}

TEST(MonitoringTest, StatusBuilder) {
  ASSERT_FALSE(FCP_STATUS(ABORTED).ok());
  ASSERT_EQ(FCP_STATUS(ABORTED).code(), ABORTED);
}

TEST(MonitoringTest, FcpReturnIfError) {
  ASSERT_THAT(
      []() -> StatusOr<int> {
        Status fail_status = FCP_STATUS(ABORTED);
        FCP_RETURN_IF_ERROR(fail_status);
        return 0;
      }(),
      Not(IsOk()));
  ASSERT_THAT(
      []() -> StatusOr<int> {
        FCP_RETURN_IF_ERROR(Status());
        return 0;
      }(),
      IsOkAndHolds(0));

  ASSERT_THAT(
      []() -> StatusOr<int> {
        StatusOr<int> fail_statusor = FCP_STATUS(ABORTED);
        FCP_RETURN_IF_ERROR(fail_statusor);
        return 0;
      }(),
      Not(IsOk()));
  ASSERT_THAT(
      []() -> StatusOr<int> {
        FCP_RETURN_IF_ERROR(StatusOr<int>(0));
        return 0;
      }(),
      IsOkAndHolds(0));
}

}  // namespace
}  // namespace fcp
