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

#ifndef FCP_CLIENT_PRIVATELOGGER_TESTS_TEST_SCENARIOS_H_
#define FCP_CLIENT_PRIVATELOGGER_TESTS_TEST_SCENARIOS_H_

#include <concepts>
#include <string>

#include "fcp/client/privatelogger/private_logger.h"

namespace fcp::client::privatelogger {

// Names of the scenarios to be used in parametrization.
inline constexpr char kScenarioSuccessSingleTask[] = "success_single_task";
inline constexpr char kScenarioSuccessMultipleTasks[] =
    "success_multiple_tasks";
inline constexpr char kScenarioUploadFailure[] = "upload_failure";
inline constexpr char kScenarioSequentialUploads[] = "sequential_uploads";

template <typename T>
concept PrivateLoggerImpl = std::derived_from<T, PrivateLogger>;

template <PrivateLoggerImpl T>
class PrivateLoggerTestSetup {
 public:
  virtual ~PrivateLoggerTestSetup() = default;
  virtual void setup(const std::string& scenario_name, T& logger) = 0;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_TESTS_TEST_SCENARIOS_H_
