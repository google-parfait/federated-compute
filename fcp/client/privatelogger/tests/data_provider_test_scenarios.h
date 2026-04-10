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

#ifndef FCP_CLIENT_PRIVATELOGGER_TESTS_DATA_PROVIDER_TEST_SCENARIOS_H_
#define FCP_CLIENT_PRIVATELOGGER_TESTS_DATA_PROVIDER_TEST_SCENARIOS_H_

#include <concepts>
#include <string>

#include "fcp/client/privatelogger/data_provider.h"

namespace fcp::client::privatelogger {

// Names of the scenarios for DataProvider.
inline constexpr char kScenarioDataProviderSuccessEmpty[] =
    "data_provider_success_empty";
inline constexpr char kScenarioDataProviderSuccessSingleEntry[] =
    "data_provider_success_single_entry";
inline constexpr char kScenarioDataProviderSuccessMultipleEntries[] =
    "data_provider_success_multiple_entries";
inline constexpr char kScenarioDataProviderFailureInternal[] =
    "data_provider_failure_internal";
inline constexpr char kScenarioDataProviderTaskSpecific[] =
    "data_provider_task_specific";

template <typename T>
concept DataProviderImpl = std::derived_from<T, DataProvider>;

template <DataProviderImpl T>
class DataProviderTestSetup {
 public:
  virtual ~DataProviderTestSetup() = default;
  virtual void setup(const std::string& scenario_name, T& data_provider) = 0;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_TESTS_DATA_PROVIDER_TEST_SCENARIOS_H_
