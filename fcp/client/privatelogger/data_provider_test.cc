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

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "fcp/client/privatelogger/tests/data_provider_mocks.h"
#include "fcp/client/privatelogger/tests/data_provider_test_scenarios.h"
#include "fcp/client/privatelogger/tests/mock_data_provider_setup.h"
#include "fcp/client/selector_context.pb.h"

namespace fcp::client::privatelogger {

using ::absl_testing::IsOk;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;

template <typename Provider, typename Setup>
struct TestConfig {
  using ProviderType = Provider;
  using SetupType = Setup;
};

template <typename Config>
class ParametrizedDataProviderTest : public ::testing::Test {
 protected:
  using ProviderType = typename Config::ProviderType;
  using SetupType = typename Config::SetupType;

  std::unique_ptr<ProviderType> data_provider_ =
      std::make_unique<ProviderType>();
  SetupType setup_;
};

using Implementations =
    ::testing::Types<TestConfig<MockDataProvider, MockDataProviderSetup>>;
TYPED_TEST_SUITE(ParametrizedDataProviderTest, Implementations);

TYPED_TEST(ParametrizedDataProviderTest, SuccessEmpty) {
  this->setup_.setup(kScenarioDataProviderSuccessEmpty, *this->data_provider_);

  fcp::client::SelectorContext context;
  auto result = this->data_provider_->GetData(context);

  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(*result, IsEmpty());
}

TYPED_TEST(ParametrizedDataProviderTest, SuccessSingleEntry) {
  this->setup_.setup(kScenarioDataProviderSuccessSingleEntry,
                     *this->data_provider_);

  fcp::client::SelectorContext context;
  auto result = this->data_provider_->GetData(context);

  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(*result, SizeIs(1));
  EXPECT_EQ((*result)[0].value, "data_1");
  EXPECT_EQ((*result)[0].timestamp, "ts_1");
}

TYPED_TEST(ParametrizedDataProviderTest, SuccessMultipleEntries) {
  this->setup_.setup(kScenarioDataProviderSuccessMultipleEntries,
                     *this->data_provider_);

  fcp::client::SelectorContext context;
  auto result = this->data_provider_->GetData(context);

  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(*result, SizeIs(2));
}

TYPED_TEST(ParametrizedDataProviderTest, FailureInternal) {
  this->setup_.setup(kScenarioDataProviderFailureInternal,
                     *this->data_provider_);

  fcp::client::SelectorContext context;
  auto result = this->data_provider_->GetData(context);

  EXPECT_THAT(result.status(), Not(IsOk()));
}

TYPED_TEST(ParametrizedDataProviderTest, TaskSpecific) {
  this->setup_.setup(kScenarioDataProviderTaskSpecific, *this->data_provider_);

  fcp::client::SelectorContext context_a;
  context_a.mutable_computation_properties()
      ->mutable_federated()
      ->set_task_name("task_A");

  auto result_a = this->data_provider_->GetData(context_a);
  ASSERT_THAT(result_a, IsOk());
  EXPECT_THAT(*result_a, SizeIs(1));
  EXPECT_EQ((*result_a)[0].value, "data_A");

  fcp::client::SelectorContext context_b;
  context_b.mutable_computation_properties()
      ->mutable_federated()
      ->set_task_name("task_B");

  auto result_b = this->data_provider_->GetData(context_b);
  ASSERT_THAT(result_b, IsOk());
  EXPECT_THAT(*result_b, SizeIs(1));
  EXPECT_EQ((*result_b)[0].value, "data_B");
}

}  // namespace fcp::client::privatelogger
