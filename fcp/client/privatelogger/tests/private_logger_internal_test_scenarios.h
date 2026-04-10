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

#ifndef FCP_CLIENT_PRIVATELOGGER_TESTS_PRIVATE_LOGGER_INTERNAL_TEST_SCENARIOS_H_
#define FCP_CLIENT_PRIVATELOGGER_TESTS_PRIVATE_LOGGER_INTERNAL_TEST_SCENARIOS_H_

#include <concepts>
#include <string>

#include "fcp/client/privatelogger/private_logger_internal.h"

namespace fcp::client::privatelogger {

inline constexpr char kScenarioInternalSuccess[] = "InternalSuccess";
inline constexpr char kScenarioInternalFailure[] = "InternalFailure";

template <typename T>
concept PrivateLoggerInternalImpl = std::derived_from<T, PrivateLoggerInternal>;

template <PrivateLoggerInternalImpl T>
class PrivateLoggerInternalTestSetup {
 public:
  virtual ~PrivateLoggerInternalTestSetup() = default;
  virtual void setup(const std::string& scenario_name, T& logger) = 0;
};

}  // namespace fcp::client::privatelogger

#endif  // FCP_CLIENT_PRIVATELOGGER_TESTS_PRIVATE_LOGGER_INTERNAL_TEST_SCENARIOS_H_
