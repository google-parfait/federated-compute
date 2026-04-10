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

#ifndef FCP_CLIENT_PRIVATELOGGER_TESTS_PRIVATE_LOGGER_CONNECTOR_TEST_SCENARIOS_H_
#define FCP_CLIENT_PRIVATELOGGER_TESTS_PRIVATE_LOGGER_CONNECTOR_TEST_SCENARIOS_H_

#include <concepts>
#include <string>

#include "fcp/client/privatelogger/private_logger_connector.h"

namespace fcp::client::privatelogger {

// Names of the scenarios to be used in parametrization.
inline constexpr char kScenarioConnectorSuccess[] = "connector_success";
inline constexpr char kScenarioConnectorMultipleTasks[] =
    "connector_multiple_tasks";
inline constexpr char kScenarioConnectorPartialFailure[] =
    "connector_partial_failure";
inline constexpr char kScenarioConnectorDataProviderError[] =
    "connector_data_provider_error";
inline constexpr char kScenarioConnectorTotalFailure[] =
    "connector_total_failure";
inline constexpr char kScenarioConnectorEmptyData[] = "connector_empty_data";

template <typename T>
concept PrivateLoggerConnectorImpl =
    std::derived_from<T, PrivateLoggerConnector>;

template <PrivateLoggerConnectorImpl T>
class PrivateLoggerConnectorTestSetup {
 public:
  virtual ~PrivateLoggerConnectorTestSetup() = default;
  virtual void setup(const std::string& scenario_name, T& connector) = 0;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_TESTS_PRIVATE_LOGGER_CONNECTOR_TEST_SCENARIOS_H_
