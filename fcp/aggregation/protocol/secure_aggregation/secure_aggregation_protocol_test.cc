/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fcp/aggregation/protocol/secure_aggregation/secure_aggregation_protocol.h"

#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/aggregation/protocol/testing/test_callback.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/testing/testing.h"

namespace fcp::aggregation {
namespace {

class SecureAggregationProtocolTest : public ::testing::Test {
 protected:
  MockAggregationProtocolCallback callback_;
  // Creates an instance of SecureAggregationProtocol
  absl::StatusOr<std::unique_ptr<SecureAggregationProtocol>> CreateProtocol();
};

absl::StatusOr<std::unique_ptr<SecureAggregationProtocol>>
SecureAggregationProtocolTest::CreateProtocol() {
  Configuration conf;
  auto worker_scheduler = CreateThreadPoolScheduler(8);
  auto callback_scheduler = CreateThreadPoolScheduler(1);

  worker_scheduler->WaitUntilIdle();
  callback_scheduler->WaitUntilIdle();

  auto protocol_or_status = SecureAggregationProtocol::Create(
      conf, &callback_, std::move(worker_scheduler),
      std::move(callback_scheduler));

  return protocol_or_status;
}

TEST_F(SecureAggregationProtocolTest, CreateProtocol_Succeeds) {
  // Verify that the protocol can be created successfully.
  auto protocol_or_status = CreateProtocol();
  EXPECT_THAT(protocol_or_status, IsOk());
}

TEST_F(SecureAggregationProtocolTest, StartProtocol_Succeeds) {
  auto protocol_or_status = CreateProtocol();
  EXPECT_THAT(protocol_or_status, IsOk());
  EXPECT_CALL(callback_,
              OnAcceptClients(/*start_client_id=*/0, /*int64_t num_clients=*/3,
                              /*message=*/testing::_));
  EXPECT_THAT(protocol_or_status.value()->Start(/*num_clients=*/3), IsOk());
}

}  // namespace
}  // namespace fcp::aggregation
