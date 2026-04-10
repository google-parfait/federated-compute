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

#ifndef FCP_CLIENT_PRIVATELOGGER_TESTS_DATA_PROVIDER_MOCKS_H_
#define FCP_CLIENT_PRIVATELOGGER_TESTS_DATA_PROVIDER_MOCKS_H_

#include <vector>

#include "gmock/gmock.h"
#include "absl/status/statusor.h"
#include "fcp/client/privatelogger/data_provider.h"
#include "fcp/client/privatelogger/log_entry.h"
#include "fcp/client/selector_context.pb.h"

namespace fcp::client::privatelogger {

class MockDataProvider : public DataProvider {
 public:
  MOCK_METHOD(absl::StatusOr<std::vector<LogEntry>>, GetData,
              (const fcp::client::SelectorContext& selector_context),
              (override));
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_TESTS_DATA_PROVIDER_MOCKS_H_
