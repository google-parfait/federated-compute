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

#include <stdio.h>

#include <array>
#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/log_severity.h"
#include "absl/strings/str_format.h"
#include "fcp/base/base_name.h"

namespace fcp {
namespace {

using ::testing::Eq;
using ::testing::MatchesRegex;
using ::testing::Not;

MATCHER(IsOk, "") { return arg.ok(); }

MATCHER_P(IsOkAndHolds, m, "") {
  return testing::ExplainMatchResult(IsOk(), arg, result_listener) &&
         testing::ExplainMatchResult(m, arg.value(), result_listener);
}

class BaremetalLogger final : public internal::Logger {
 public:
  void Log(const char* file, int line, LogSeverity severity,
           const char* message) override {
    absl::FPrintF(stderr, "%c %s:%d %s\n",
                  absl::LogSeverityName(static_cast<absl::LogSeverity>(
                      static_cast<int>(severity)))[0],
                  BaseName(file), line, message);
  }
};

class MonitoringTest : public ::testing::TestWithParam<bool> {
 public:
  void SetUp() override {
    if (replace_logger_) {
      prev_logger_ = internal::logger();
      internal::set_logger(&logger_);
    }
  }
  void TearDown() override {
    if (replace_logger_) {
      internal::set_logger(prev_logger_);
    }
  }

 private:
  const bool replace_logger_ = GetParam();
  internal::Logger* prev_logger_ = nullptr;
  BaremetalLogger logger_;
};

#ifdef FCP_BAREMETAL
INSTANTIATE_TEST_SUITE_P(Baremetal, MonitoringTest, testing::Values(true));
#else
INSTANTIATE_TEST_SUITE_P(Base, MonitoringTest, testing::Values(false));
#endif

TEST_P(MonitoringTest, LogInfo) {
  testing::internal::CaptureStderr();
  FCP_LOG(INFO) << "info log of something happening";
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("I.*info log of something happening\n"));
}

TEST_P(MonitoringTest, LogWarning) {
  testing::internal::CaptureStderr();
  FCP_LOG(WARNING) << "warning log of something happening";
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("W.*warning log of something happening\n"));
}

TEST_P(MonitoringTest, LogError) {
  testing::internal::CaptureStderr();
  FCP_LOG(ERROR) << "error log of something happening";
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("E.*error log of something happening\n"));
}

TEST_P(MonitoringTest, LogFatal) {
  ASSERT_DEATH({ FCP_LOG(FATAL) << "fatal log"; }, "fatal log");
}

TEST_P(MonitoringTest, StatusBuilderLogInfo) {
  testing::internal::CaptureStderr();
  Status status = (FCP_STATUS(ABORTED) << "something happened").LogInfo();
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("I.*something happened\n"));
}

TEST_P(MonitoringTest, StatusBuilderLogWarning) {
  testing::internal::CaptureStderr();
  Status status = (FCP_STATUS(ABORTED) << "something happened").LogWarning();
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("W.*something happened\n"));
}

TEST_P(MonitoringTest, StatusBuilderLogError) {
  testing::internal::CaptureStderr();
  Status status = (FCP_STATUS(ABORTED) << "something happened").LogError();
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("E.*something happened\n"));
}

TEST_P(MonitoringTest, StatusBuilderLogFatal) {
  ASSERT_DEATH(
      {
        Status status =
            (FCP_STATUS(ABORTED) << "something happened").LogFatal();
      },
      "something happened");
}

TEST_P(MonitoringTest, LogIfTrue) {
  testing::internal::CaptureStderr();
  FCP_LOG_IF(INFO, true) << "some log";
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(output, MatchesRegex("I.*some log\n"));
}

TEST_P(MonitoringTest, LogIfFalse) {
  testing::internal::CaptureStderr();
  FCP_LOG_IF(INFO, false) << "some log";
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_EQ(output, "");
}

TEST_P(MonitoringTest, CheckSucceeds) { FCP_CHECK(1 < 2); }

TEST_P(MonitoringTest, CheckFails) {
  ASSERT_DEATH({ FCP_CHECK(1 < 0); }, "Check failed: 1 < 0.");
}

TEST_P(MonitoringTest, StatusOr) {
  StatusOr<int> default_constructed_status;
  ASSERT_FALSE(default_constructed_status.ok());
  ASSERT_EQ(default_constructed_status.status().code(), UNKNOWN);

  StatusOr<int> fail_status = FCP_STATUS(ABORTED) << "operation aborted";
  ASSERT_FALSE(fail_status.ok());
  ASSERT_EQ(fail_status.status().code(), ABORTED);
  // TODO(team): Port StatusIs matcher to avoid casting message(),
  // which is string_view, to std::string.
  ASSERT_THAT(fail_status.status().message(),
              MatchesRegex(".*operation aborted"));
}

TEST_P(MonitoringTest, StatusOrCopyAssignment) {
  StatusOr<int> fail_status = FCP_STATUS(ABORTED) << "operation aborted";
  StatusOr<int> copy_of_fail_status(fail_status);
  ASSERT_FALSE(copy_of_fail_status.ok());
  ASSERT_EQ(copy_of_fail_status.status().code(), ABORTED);
  ASSERT_THAT(copy_of_fail_status.status().message(),
              MatchesRegex(".*operation aborted"));

  StatusOr<int> ok_status = 42;
  StatusOr<int> copy_of_ok_status(ok_status);
  ASSERT_THAT(copy_of_ok_status, IsOkAndHolds(Eq(42)));
  ASSERT_EQ(copy_of_ok_status.value(), 42);
}

TEST_P(MonitoringTest, StatusOrMoveAssignment) {
  StatusOr<std::unique_ptr<std::string>> fail_status = FCP_STATUS(ABORTED)
                                                       << "operation aborted";
  StatusOr<std::unique_ptr<std::string>> moved_fail_status(
      std::move(fail_status));
  ASSERT_FALSE(moved_fail_status.ok());
  ASSERT_EQ(moved_fail_status.status().code(), ABORTED);
  ASSERT_THAT(moved_fail_status.status().message(),
              MatchesRegex(".*operation aborted"));

  auto value = std::make_unique<std::string>("foobar");
  StatusOr<std::unique_ptr<std::string>> ok_status = std::move(value);
  StatusOr<std::unique_ptr<std::string>> moved_ok_status(std::move(ok_status));
  ASSERT_TRUE(moved_ok_status.ok());
  ASSERT_EQ(*moved_ok_status.value(), "foobar");
}

TEST_P(MonitoringTest, StatusOrCopying) {
  StatusOr<int> fail_status = FCP_STATUS(ABORTED) << "operation aborted";
  StatusOr<int> copy_of_status = fail_status;
  ASSERT_FALSE(copy_of_status.ok());
  ASSERT_EQ(copy_of_status.status().code(), ABORTED);
  ASSERT_THAT(copy_of_status.status().message(),
              MatchesRegex(".*operation aborted"));

  StatusOr<int> ok_status = 42;
  copy_of_status = ok_status;
  ASSERT_THAT(copy_of_status, IsOkAndHolds(Eq(42)));
  ASSERT_EQ(copy_of_status.value(), 42);
}

TEST_P(MonitoringTest, StatusOrMoving) {
  StatusOr<std::unique_ptr<std::string>> fail_status = FCP_STATUS(ABORTED)
                                                       << "operation aborted";
  StatusOr<std::unique_ptr<std::string>> moved_status = std::move(fail_status);
  ASSERT_FALSE(moved_status.ok());
  ASSERT_EQ(moved_status.status().code(), ABORTED);
  ASSERT_THAT(moved_status.status().message(),
              MatchesRegex(".*operation aborted"));

  auto value = std::make_unique<std::string>("foobar");
  StatusOr<std::unique_ptr<std::string>> ok_status = std::move(value);
  moved_status = std::move(ok_status);
  ASSERT_TRUE(moved_status.ok());
  ASSERT_EQ(*moved_status.value(), "foobar");
}

TEST_P(MonitoringTest, StatusBuilder) {
  ASSERT_FALSE(FCP_STATUS(ABORTED).ok());
  ASSERT_EQ(FCP_STATUS(ABORTED).code(), ABORTED);
}

TEST_P(MonitoringTest, FcpReturnIfError) {
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
