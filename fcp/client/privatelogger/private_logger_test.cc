// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "fcp/client/privatelogger/tests/mock_private_logger_setup.h"
#include "fcp/client/privatelogger/tests/private_logger_mocks.h"
#include "fcp/client/privatelogger/tests/test_scenarios.h"
#include "fcp/client/selector_context.pb.h"

namespace fcp::client::privatelogger {

using ::absl_testing::IsOk;
using ::testing::Not;

template <typename Logger, typename Setup>
struct TestConfig {
  using LoggerType = Logger;
  using SetupType = Setup;
};

template <typename Config>
class ParametrizedPrivateLoggerTest : public ::testing::Test {
 protected:
  using LoggerType = typename Config::LoggerType;
  using SetupType = typename Config::SetupType;

  std::unique_ptr<LoggerType> logger_ = std::make_unique<LoggerType>();
  SetupType setup_;
};

using Implementations =
    ::testing::Types<TestConfig<MockPrivateLogger, MockPrivateLoggerSetup>>;
TYPED_TEST_SUITE(ParametrizedPrivateLoggerTest, Implementations);

TYPED_TEST(ParametrizedPrivateLoggerTest, SuccessSingleTask) {
  this->setup_.setup(kScenarioSuccessSingleTask, *this->logger_);

  fcp::client::SelectorContext msg;
  msg.mutable_computation_properties()->set_session_name("session");

  EXPECT_THAT(this->logger_->Log(msg), IsOk());
  EXPECT_THAT(this->logger_->Upload(), IsOk());
  auto data = this->logger_->GetAndCommitData("task_1");
  ASSERT_THAT(data, IsOk());
  EXPECT_EQ((*data).size(), 1);
}

TYPED_TEST(ParametrizedPrivateLoggerTest, SuccessMultipleTasks) {
  this->setup_.setup(kScenarioSuccessMultipleTasks, *this->logger_);

  fcp::client::SelectorContext msg;
  msg.mutable_computation_properties()->set_session_name("session");

  EXPECT_THAT(this->logger_->Log(msg), IsOk());
  EXPECT_THAT(this->logger_->Log(msg), IsOk());
  EXPECT_THAT(this->logger_->Log(msg), IsOk());
  EXPECT_THAT(this->logger_->Log(msg), IsOk());
  EXPECT_THAT(this->logger_->Upload(), IsOk());
  EXPECT_THAT(this->logger_->GetAndCommitData("task_A"), IsOk());
  EXPECT_THAT(this->logger_->GetAndCommitData("task_B"), IsOk());
}

TYPED_TEST(ParametrizedPrivateLoggerTest, UploadFailure) {
  this->setup_.setup(kScenarioUploadFailure, *this->logger_);

  fcp::client::SelectorContext msg;
  msg.mutable_computation_properties()->set_session_name("session");

  EXPECT_THAT(this->logger_->Log(msg), IsOk());
  EXPECT_THAT(this->logger_->Upload(), Not(IsOk()));
  EXPECT_THAT(this->logger_->RollbackContribution("task_1"), IsOk());
}

TYPED_TEST(ParametrizedPrivateLoggerTest, SequentialUploads) {
  this->setup_.setup(kScenarioSequentialUploads, *this->logger_);

  fcp::client::SelectorContext msg;
  msg.mutable_computation_properties()->set_session_name("session");

  EXPECT_THAT(this->logger_->Log(msg), IsOk());
  EXPECT_THAT(this->logger_->Upload(), IsOk());
  EXPECT_THAT(this->logger_->GetAndCommitData("task_1"), IsOk());

  EXPECT_THAT(this->logger_->Log(msg), IsOk());
  EXPECT_THAT(this->logger_->Upload(), IsOk());
  EXPECT_THAT(this->logger_->GetAndCommitData("task_1"), IsOk());
}

}  // namespace fcp::client::privatelogger
