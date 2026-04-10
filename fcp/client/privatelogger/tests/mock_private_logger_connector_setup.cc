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

#include "fcp/client/privatelogger/tests/mock_private_logger_connector_setup.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "fcp/client/privatelogger/tests/private_logger_connector_mocks.h"
#include "fcp/client/privatelogger/tests/private_logger_connector_test_scenarios.h"
#include "fcp/client/privatelogger/upload_outcome.h"

namespace fcp::client::privatelogger {

using ::testing::_;
using ::testing::Return;

MockPrivateLoggerConnectorSetup::MockPrivateLoggerConnectorSetup() {
  scenario_setups_.emplace(kScenarioConnectorSuccess,
                           [this](MockPrivateLoggerConnector& connector) {
                             SetupConnectorSuccess(connector);
                           });
  scenario_setups_.emplace(kScenarioConnectorMultipleTasks,
                           [this](MockPrivateLoggerConnector& connector) {
                             SetupConnectorMultipleTasks(connector);
                           });
  scenario_setups_.emplace(kScenarioConnectorPartialFailure,
                           [this](MockPrivateLoggerConnector& connector) {
                             SetupConnectorPartialFailure(connector);
                           });
  scenario_setups_.emplace(kScenarioConnectorDataProviderError,
                           [this](MockPrivateLoggerConnector& connector) {
                             SetupConnectorDataProviderError(connector);
                           });
  scenario_setups_.emplace(kScenarioConnectorTotalFailure,
                           [this](MockPrivateLoggerConnector& connector) {
                             SetupConnectorTotalFailure(connector);
                           });
  scenario_setups_.emplace(kScenarioConnectorEmptyData,
                           [this](MockPrivateLoggerConnector& connector) {
                             SetupConnectorEmptyData(connector);
                           });
}

void MockPrivateLoggerConnectorSetup::setup(
    const std::string& scenario_name, MockPrivateLoggerConnector& connector) {
  auto it = scenario_setups_.find(scenario_name);
  if (it != scenario_setups_.end()) {
    it->second(connector);
  }
}

void MockPrivateLoggerConnectorSetup::SetupConnectorSuccess(
    MockPrivateLoggerConnector& mock_connector) {
  std::vector<UploadOutcome> outcomes = {
      {.task_name = "task_1", .status = UploadOutcome::Status::kContributed}};
  EXPECT_CALL(mock_connector, Upload("source_1", _)).WillOnce(Return(outcomes));
}

void MockPrivateLoggerConnectorSetup::SetupConnectorMultipleTasks(
    MockPrivateLoggerConnector& mock_connector) {
  std::vector<UploadOutcome> outcomes = {
      {.task_name = "task_A", .status = UploadOutcome::Status::kContributed},
      {.task_name = "task_B", .status = UploadOutcome::Status::kContributed}};
  EXPECT_CALL(mock_connector, Upload("source_1", _)).WillOnce(Return(outcomes));
}

void MockPrivateLoggerConnectorSetup::SetupConnectorPartialFailure(
    MockPrivateLoggerConnector& mock_connector) {
  std::vector<UploadOutcome> outcomes = {
      {.task_name = "task_A", .status = UploadOutcome::Status::kContributed},
      {.task_name = "task_B",
       .status = UploadOutcome::Status::kNotContributed}};
  EXPECT_CALL(mock_connector, Upload("source_1", _)).WillOnce(Return(outcomes));
}

void MockPrivateLoggerConnectorSetup::SetupConnectorDataProviderError(
    MockPrivateLoggerConnector& mock_connector) {
  EXPECT_CALL(mock_connector, Upload("source_1", _))
      .WillOnce(Return(absl::InternalError("DataProvider error")));
}

void MockPrivateLoggerConnectorSetup::SetupConnectorTotalFailure(
    MockPrivateLoggerConnector& mock_connector) {
  EXPECT_CALL(mock_connector, Upload("source_1", _))
      .WillOnce(Return(absl::UnavailableError("FCP service unavailable")));
}

void MockPrivateLoggerConnectorSetup::SetupConnectorEmptyData(
    MockPrivateLoggerConnector& mock_connector) {
  std::vector<UploadOutcome> outcomes = {
      {.task_name = "task_1", .status = UploadOutcome::Status::kContributed}};
  EXPECT_CALL(mock_connector, Upload("source_1", _)).WillOnce(Return(outcomes));
}

}  // namespace fcp::client::privatelogger
