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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "fcp/client/privatelogger/tests/data_provider_mocks.h"
#include "fcp/client/privatelogger/tests/mock_private_logger_connector_setup.h"
#include "fcp/client/privatelogger/tests/private_logger_connector_mocks.h"
#include "fcp/client/privatelogger/tests/private_logger_connector_test_scenarios.h"
#include "fcp/client/privatelogger/upload_outcome.h"

namespace fcp::client::privatelogger {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

template <typename Connector, typename Setup>
struct ConnectorTestConfig {
  using ConnectorType = Connector;
  using SetupType = Setup;
};

template <typename Config>
class PrivateLoggerConnectorTest : public ::testing::Test {
 protected:
  using ConnectorType = typename Config::ConnectorType;
  using SetupType = typename Config::SetupType;

  std::unique_ptr<ConnectorType> connector_ = std::make_unique<ConnectorType>();
  SetupType setup_;
  MockDataProvider mock_data_provider_;
};

using ConnectorImplementations =
    ::testing::Types<ConnectorTestConfig<MockPrivateLoggerConnector,
                                         MockPrivateLoggerConnectorSetup>>;
TYPED_TEST_SUITE(PrivateLoggerConnectorTest, ConnectorImplementations);

TYPED_TEST(PrivateLoggerConnectorTest, SuccessScenario) {
  this->setup_.setup(kScenarioConnectorSuccess, *this->connector_);

  auto result = this->connector_->Upload("source_1", this->mock_data_provider_);

  ASSERT_THAT(result, IsOk());
  ASSERT_EQ((*result).size(), 1);
  EXPECT_EQ((*result)[0].task_name, "task_1");
  EXPECT_EQ((*result)[0].status, UploadOutcome::Status::kContributed);
}

TYPED_TEST(PrivateLoggerConnectorTest, MultipleTasksScenario) {
  this->setup_.setup(kScenarioConnectorMultipleTasks, *this->connector_);

  auto result = this->connector_->Upload("source_1", this->mock_data_provider_);

  ASSERT_THAT(result, IsOk());
  ASSERT_EQ((*result).size(), 2);
  EXPECT_EQ((*result)[0].task_name, "task_A");
  EXPECT_EQ((*result)[1].task_name, "task_B");
}

TYPED_TEST(PrivateLoggerConnectorTest, PartialFailureScenario) {
  this->setup_.setup(kScenarioConnectorPartialFailure, *this->connector_);

  auto result = this->connector_->Upload("source_1", this->mock_data_provider_);

  ASSERT_THAT(result, IsOk());
  ASSERT_EQ((*result).size(), 2);
  EXPECT_EQ((*result)[0].status, UploadOutcome::Status::kContributed);
  EXPECT_EQ((*result)[1].status, UploadOutcome::Status::kNotContributed);
}

TYPED_TEST(PrivateLoggerConnectorTest, DataProviderErrorScenario) {
  this->setup_.setup(kScenarioConnectorDataProviderError, *this->connector_);

  auto result = this->connector_->Upload("source_1", this->mock_data_provider_);

  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInternal,
                               HasSubstr("DataProvider error")));
}

TYPED_TEST(PrivateLoggerConnectorTest, TotalFailureScenario) {
  this->setup_.setup(kScenarioConnectorTotalFailure, *this->connector_);

  auto result = this->connector_->Upload("source_1", this->mock_data_provider_);

  EXPECT_THAT(result, StatusIs(absl::StatusCode::kUnavailable,
                               HasSubstr("FCP service unavailable")));
}

TYPED_TEST(PrivateLoggerConnectorTest, EmptyDataScenario) {
  this->setup_.setup(kScenarioConnectorEmptyData, *this->connector_);

  auto result = this->connector_->Upload("source_1", this->mock_data_provider_);

  ASSERT_THAT(result, IsOk());
  EXPECT_FALSE((*result).empty());
}

}  // namespace fcp::client::privatelogger
