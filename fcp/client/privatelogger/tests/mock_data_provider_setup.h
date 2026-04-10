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

#ifndef FCP_CLIENT_PRIVATELOGGER_TESTS_MOCK_DATA_PROVIDER_SETUP_H_
#define FCP_CLIENT_PRIVATELOGGER_TESTS_MOCK_DATA_PROVIDER_SETUP_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "fcp/client/privatelogger/tests/data_provider_mocks.h"
#include "fcp/client/privatelogger/tests/data_provider_test_scenarios.h"

namespace fcp::client::privatelogger {

class MockDataProviderSetup : public DataProviderTestSetup<MockDataProvider> {
 public:
  MockDataProviderSetup();
  void setup(const std::string& scenario_name,
             MockDataProvider& data_provider) override;

 private:
  void SetupSuccessEmpty(MockDataProvider& mock_data_provider);
  void SetupSuccessSingleEntry(MockDataProvider& mock_data_provider);
  void SetupSuccessMultipleEntries(MockDataProvider& mock_data_provider);
  void SetupFailureInternal(MockDataProvider& mock_data_provider);
  void SetupTaskSpecific(MockDataProvider& mock_data_provider);

  absl::flat_hash_map<std::string,
                      absl::AnyInvocable<void(MockDataProvider&) const>>
      scenario_setups_;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_TESTS_MOCK_DATA_PROVIDER_SETUP_H_
