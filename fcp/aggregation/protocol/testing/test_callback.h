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

#ifndef FCP_AGGREGATION_PROTOCOL_TESTING_TEST_CALLBACK_H_
#define FCP_AGGREGATION_PROTOCOL_TESTING_TEST_CALLBACK_H_

#include "gmock/gmock.h"
#include "fcp/aggregation/protocol/aggregation_protocol.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"

namespace fcp::aggregation {

class MockAggregationProtocolCallback : public AggregationProtocol::Callback {
 public:
  MOCK_METHOD(void, OnAcceptClients,
              (int64_t start_client_id, int64_t num_clients,
               const AcceptanceMessage& message),
              (override));
  MOCK_METHOD(void, OnSendServerMessage,
              (int64_t client_id, const ServerMessage& message), (override));
  MOCK_METHOD(void, OnCloseClient,
              (int64_t client_id, absl::Status diagnostic_status), (override));
  MOCK_METHOD(void, OnComplete, (absl::Cord result), (override));
  MOCK_METHOD(void, OnAbort, (absl::Status diagnostic_status), (override));
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_TESTING_TEST_CALLBACK_H_
