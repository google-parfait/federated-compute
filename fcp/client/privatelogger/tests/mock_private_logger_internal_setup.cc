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

#include "fcp/client/privatelogger/tests/mock_private_logger_internal_setup.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "fcp/client/privatelogger/tests/private_logger_internal_mocks.h"
#include "fcp/client/privatelogger/tests/private_logger_internal_test_scenarios.h"

namespace fcp::client::privatelogger {

using ::testing::Return;

MockPrivateLoggerInternalSetup::MockPrivateLoggerInternalSetup() {
  scenario_setups_.emplace(
      kScenarioInternalSuccess,
      [this](MockPrivateLoggerInternal& mock) { SetupInternalSuccess(mock); });
  scenario_setups_.emplace(
      kScenarioInternalFailure,
      [this](MockPrivateLoggerInternal& mock) { SetupInternalFailure(mock); });
}

void MockPrivateLoggerInternalSetup::setup(
    const std::string& scenario_name, MockPrivateLoggerInternal& mock_logger) {
  auto it = scenario_setups_.find(scenario_name);
  if (it != scenario_setups_.end()) {
    // Found: call the setup function
    it->second(mock_logger);
  }
  // Not found: NO-OP.
}

void MockPrivateLoggerInternalSetup::SetupInternalSuccess(
    MockPrivateLoggerInternal& mock_logger) {
  EXPECT_CALL(mock_logger, GetAndCommitData(::testing::_))
      .WillRepeatedly(Return(std::vector<std::string>{"data"}));
  EXPECT_CALL(mock_logger, RollbackContribution(::testing::_))
      .WillRepeatedly(Return(absl::OkStatus()));
}

void MockPrivateLoggerInternalSetup::SetupInternalFailure(
    MockPrivateLoggerInternal& mock_logger) {
  EXPECT_CALL(mock_logger, GetAndCommitData(::testing::_))
      .WillRepeatedly(Return(absl::InternalError("error")));
  EXPECT_CALL(mock_logger, RollbackContribution(::testing::_))
      .WillRepeatedly(Return(absl::InternalError("error")));
}

}  // namespace fcp::client::privatelogger
