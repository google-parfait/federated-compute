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

#include "fcp/client/privatelogger/tests/mock_data_provider_setup.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/client/privatelogger/log_entry.h"
#include "fcp/client/privatelogger/tests/data_provider_mocks.h"
#include "fcp/client/privatelogger/tests/data_provider_test_scenarios.h"
#include "fcp/client/selector_context.pb.h"

namespace fcp::client::privatelogger {

using ::testing::_;

MockDataProviderSetup::MockDataProviderSetup() {
  scenario_setups_.emplace(
      kScenarioDataProviderSuccessEmpty,
      [this](MockDataProvider& provider) { SetupSuccessEmpty(provider); });
  scenario_setups_.emplace(kScenarioDataProviderSuccessSingleEntry,
                           [this](MockDataProvider& provider) {
                             SetupSuccessSingleEntry(provider);
                           });
  scenario_setups_.emplace(kScenarioDataProviderSuccessMultipleEntries,
                           [this](MockDataProvider& provider) {
                             SetupSuccessMultipleEntries(provider);
                           });
  scenario_setups_.emplace(
      kScenarioDataProviderFailureInternal,
      [this](MockDataProvider& provider) { SetupFailureInternal(provider); });
  scenario_setups_.emplace(
      kScenarioDataProviderTaskSpecific,
      [this](MockDataProvider& provider) { SetupTaskSpecific(provider); });
}

void MockDataProviderSetup::setup(const std::string& scenario_name,
                                  MockDataProvider& data_provider) {
  auto it = scenario_setups_.find(scenario_name);
  if (it != scenario_setups_.end()) {
    // Found: call the setup function
    it->second(data_provider);
  }
  // Not found: NO-OP.
}

void MockDataProviderSetup::SetupSuccessEmpty(
    MockDataProvider& mock_data_provider) {
  EXPECT_CALL(mock_data_provider, GetData(_)).WillOnce([]() {
    return std::vector<LogEntry>();
  });
}

void MockDataProviderSetup::SetupSuccessSingleEntry(
    MockDataProvider& mock_data_provider) {
  EXPECT_CALL(mock_data_provider, GetData(_)).WillOnce([]() {
    return std::vector<LogEntry>{
        LogEntry{.value = "data_1", .timestamp = "ts_1"}};
  });
}

void MockDataProviderSetup::SetupSuccessMultipleEntries(
    MockDataProvider& mock_data_provider) {
  EXPECT_CALL(mock_data_provider, GetData(_)).WillOnce([]() {
    return std::vector<LogEntry>{
        LogEntry{.value = "data_1", .timestamp = "ts_1"},
        LogEntry{.value = "data_2", .timestamp = "ts_2"},
    };
  });
}

void MockDataProviderSetup::SetupFailureInternal(
    MockDataProvider& mock_data_provider) {
  EXPECT_CALL(mock_data_provider, GetData(_)).WillOnce([]() {
    return absl::InternalError("Internal error");
  });
}

void MockDataProviderSetup::SetupTaskSpecific(
    MockDataProvider& mock_data_provider) {
  EXPECT_CALL(mock_data_provider, GetData(_))
      .WillRepeatedly([](const fcp::client::SelectorContext& context)
                          -> absl::StatusOr<std::vector<LogEntry>> {
        std::string task_name =
            context.computation_properties().federated().task_name();
        if (task_name == "task_A") {
          return std::vector<LogEntry>{
              LogEntry{.value = "data_A", .timestamp = "ts_A"}};
        } else if (task_name == "task_B") {
          return std::vector<LogEntry>{
              LogEntry{.value = "data_B", .timestamp = "ts_B"}};
        }
        return std::vector<LogEntry>();
      });
}

}  // namespace fcp::client::privatelogger
