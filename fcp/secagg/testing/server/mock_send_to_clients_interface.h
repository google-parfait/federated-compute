/*
 * Copyright 2018 Google LLC
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

#ifndef FCP_SECAGG_TESTING_SERVER_MOCK_SEND_TO_CLIENTS_INTERFACE_H_
#define FCP_SECAGG_TESTING_SERVER_MOCK_SEND_TO_CLIENTS_INTERFACE_H_

#include "gmock/gmock.h"
#include "fcp/secagg/server/send_to_clients_interface.h"
#include "fcp/secagg/shared/secagg_messages.pb.h"

namespace fcp {
namespace secagg {

// GMock Implementation of SendToClientsInterface.

class MockSendToClientsInterface : public SendToClientsInterface {
 public:
  MOCK_METHOD(void, SendBroadcast,
              (const ServerToClientWrapperMessage& message), (override));
  MOCK_METHOD(void, Send,
              (uint32_t recipient_id,
               const ServerToClientWrapperMessage& message),
              (override));
};

}  // namespace secagg
}  // namespace fcp

#endif  // FCP_SECAGG_TESTING_SERVER_MOCK_SEND_TO_CLIENTS_INTERFACE_H_
