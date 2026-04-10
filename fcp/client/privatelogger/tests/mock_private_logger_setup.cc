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

#include "fcp/client/privatelogger/tests/mock_private_logger_setup.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "fcp/client/privatelogger/tests/private_logger_mocks.h"
#include "fcp/client/privatelogger/tests/private_logger_test_scenarios.h"

namespace fcp::client::privatelogger {

using ::testing::_;
using ::testing::Return;

MockPrivateLoggerSetup::MockPrivateLoggerSetup() {
  scenario_setups_.emplace(
      kScenarioSuccessSingleTask,
      [this](MockPrivateLogger& logger) { SetupSuccessSingleTask(logger); });
  scenario_setups_.emplace(
      kScenarioSuccessMultipleTasks,
      [this](MockPrivateLogger& logger) { SetupSuccessMultipleTasks(logger); });
  scenario_setups_.emplace(
      kScenarioUploadFailure,
      [this](MockPrivateLogger& logger) { SetupUploadFailure(logger); });
  scenario_setups_.emplace(
      kScenarioSequentialUploads,
      [this](MockPrivateLogger& logger) { SetupSequentialUploads(logger); });
}

void MockPrivateLoggerSetup::setup(const std::string& scenario_name,
                                   MockPrivateLogger& logger) {
  auto it = scenario_setups_.find(scenario_name);
  if (it != scenario_setups_.end()) {
    // Found: call the setup function
    it->second(logger);
  }
  // Not found: NO-OP.
}

void MockPrivateLoggerSetup::SetupSuccessSingleTask(
    MockPrivateLogger& mock_logger) {
  EXPECT_CALL(mock_logger, Log(_)).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_logger, Upload()).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_logger, GetAndCommitData("task_1"))
      .WillOnce(Return(std::vector<std::string>{"data"}));
  EXPECT_CALL(mock_logger, RollbackContribution(_)).Times(0);
}

void MockPrivateLoggerSetup::SetupSuccessMultipleTasks(
    MockPrivateLogger& mock_logger) {
  EXPECT_CALL(mock_logger, Log(_))
      .Times(4)
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(mock_logger, Upload()).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_logger, GetAndCommitData("task_A"))
      .WillOnce(Return(std::vector<std::string>{"data_A"}));
  EXPECT_CALL(mock_logger, GetAndCommitData("task_B"))
      .WillOnce(Return(std::vector<std::string>{"data_B"}));
  EXPECT_CALL(mock_logger, RollbackContribution(_)).Times(0);
}

void MockPrivateLoggerSetup::SetupUploadFailure(
    MockPrivateLogger& mock_logger) {
  EXPECT_CALL(mock_logger, Log(_)).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_logger, Upload())
      .WillOnce(Return(absl::InternalError("Upload failed")));
  EXPECT_CALL(mock_logger, RollbackContribution("task_1"))
      .WillOnce(Return(absl::OkStatus()));
}

void MockPrivateLoggerSetup::SetupSequentialUploads(
    MockPrivateLogger& mock_logger) {
  EXPECT_CALL(mock_logger, Log(_))
      .Times(2)
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(mock_logger, Upload())
      .Times(2)
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(mock_logger, GetAndCommitData("task_1"))
      .Times(2)
      .WillRepeatedly(Return(std::vector<std::string>{"data"}));
  EXPECT_CALL(mock_logger, RollbackContribution(_)).Times(0);
}

}  // namespace fcp::client::privatelogger
