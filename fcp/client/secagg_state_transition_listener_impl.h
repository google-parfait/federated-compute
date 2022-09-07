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
#ifndef FCP_CLIENT_SECAGG_STATE_TRANSITION_LISTENER_IMPL_H_
#define FCP_CLIENT_SECAGG_STATE_TRANSITION_LISTENER_IMPL_H_

#include "fcp/client/log_manager.h"
#include "fcp/client/secagg_event_publisher.h"
#include "fcp/secagg/client/send_to_server_interface.h"
#include "fcp/secagg/client/state_transition_listener_interface.h"

namespace fcp {
namespace client {

class SecAggSendToServerBase : public secagg::SendToServerInterface {
 public:
  size_t last_sent_message_size() const { return last_sent_message_size_; }
  size_t total_bytes_uploaded() const { return total_bytes_uploaded_; }

 protected:
  size_t total_bytes_uploaded_ = 0;
  size_t last_sent_message_size_ = 0;
};

class SecAggStateTransitionListenerImpl
    : public secagg::StateTransitionListenerInterface {
 public:
  SecAggStateTransitionListenerImpl(
      SecAggEventPublisher* secagg_event_publisher, LogManager* log_manager,
      const SecAggSendToServerBase& secagg_send_to_server_impl,
      const size_t& last_received_message_size);
  void Transition(secagg::ClientState new_state) override;

  void Started(secagg::ClientState state) override;

  void Stopped(secagg::ClientState state) override;

  void set_execution_session_id(int64_t execution_session_id) override;

 private:
  SecAggEventPublisher* const secagg_event_publisher_;
  LogManager* const log_manager_;
  const SecAggSendToServerBase& secagg_send_to_server_;
  const size_t& last_received_message_size_;
  secagg::ClientState state_ = secagg::ClientState::INITIAL;
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_SECAGG_STATE_TRANSITION_LISTENER_IMPL_H_
