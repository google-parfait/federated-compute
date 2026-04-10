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
#include "fcp/client/privatelogger/tests/mock_private_logger_internal_setup.h"
#include "fcp/client/privatelogger/tests/private_logger_internal_mocks.h"
#include "fcp/client/privatelogger/tests/private_logger_internal_test_scenarios.h"

namespace fcp::client::privatelogger {

using ::absl_testing::IsOk;
using ::testing::Not;

template <PrivateLoggerInternalImpl Logger, typename Setup>
struct TestConfig {
  using LoggerType = Logger;
  using SetupType = Setup;
};

template <typename Config>
class ParametrizedPrivateLoggerInternalTest : public ::testing::Test {
 protected:
  using LoggerType = typename Config::LoggerType;
  using SetupType = typename Config::SetupType;

  std::unique_ptr<LoggerType> logger_ = std::make_unique<LoggerType>();
  SetupType setup_;
};

using Implementations = ::testing::Types<
    TestConfig<MockPrivateLoggerInternal, MockPrivateLoggerInternalSetup>>;
TYPED_TEST_SUITE(ParametrizedPrivateLoggerInternalTest, Implementations);

TYPED_TEST(ParametrizedPrivateLoggerInternalTest, GetAndCommitDataSuccess) {
  this->setup_.setup(kScenarioInternalSuccess, *this->logger_);

  auto result = this->logger_->GetAndCommitData("test_task");
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ((*result).size(), 1);
}

TYPED_TEST(ParametrizedPrivateLoggerInternalTest, RollbackContributionSuccess) {
  this->setup_.setup(kScenarioInternalSuccess, *this->logger_);

  auto status = this->logger_->RollbackContribution("test_task");
  EXPECT_THAT(status, IsOk());
}

TYPED_TEST(ParametrizedPrivateLoggerInternalTest, GetAndCommitDataFailure) {
  this->setup_.setup(kScenarioInternalFailure, *this->logger_);

  auto result = this->logger_->GetAndCommitData("test_task");
  EXPECT_THAT(result, Not(IsOk()));
}

TYPED_TEST(ParametrizedPrivateLoggerInternalTest, RollbackContributionFailure) {
  this->setup_.setup(kScenarioInternalFailure, *this->logger_);

  auto status = this->logger_->RollbackContribution("test_task");
  EXPECT_THAT(status, Not(IsOk()));
}

}  // namespace fcp::client::privatelogger
