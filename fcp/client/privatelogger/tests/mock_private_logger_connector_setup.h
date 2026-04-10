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

#ifndef FCP_CLIENT_PRIVATELOGGER_TESTS_MOCK_PRIVATE_LOGGER_CONNECTOR_SETUP_H_
#define FCP_CLIENT_PRIVATELOGGER_TESTS_MOCK_PRIVATE_LOGGER_CONNECTOR_SETUP_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "fcp/client/privatelogger/tests/private_logger_connector_mocks.h"
#include "fcp/client/privatelogger/tests/private_logger_connector_test_scenarios.h"

namespace fcp::client::privatelogger {

class MockPrivateLoggerConnectorSetup
    : public PrivateLoggerConnectorTestSetup<MockPrivateLoggerConnector> {
 public:
  MockPrivateLoggerConnectorSetup();
  void setup(const std::string& scenario_name,
             MockPrivateLoggerConnector& connector) override;

 private:
  void SetupConnectorSuccess(MockPrivateLoggerConnector& mock_connector);
  void SetupConnectorMultipleTasks(MockPrivateLoggerConnector& mock_connector);
  void SetupConnectorPartialFailure(MockPrivateLoggerConnector& mock_connector);
  void SetupConnectorDataProviderError(
      MockPrivateLoggerConnector& mock_connector);
  void SetupConnectorTotalFailure(MockPrivateLoggerConnector& mock_connector);
  void SetupConnectorEmptyData(MockPrivateLoggerConnector& mock_connector);

  absl::flat_hash_map<
      std::string, absl::AnyInvocable<void(MockPrivateLoggerConnector&) const>>
      scenario_setups_;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_TESTS_MOCK_PRIVATE_LOGGER_CONNECTOR_SETUP_H_
