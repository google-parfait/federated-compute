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

#include "fcp/client/secagg_state_transition_listener_impl.h"

#include "fcp/base/monitoring.h"
#include "fcp/client/secagg_event_publisher.h"

namespace fcp {
namespace client {

using ::fcp::secagg::ClientState;

SecAggStateTransitionListenerImpl::SecAggStateTransitionListenerImpl(
    SecAggEventPublisher* secagg_event_publisher, LogManager* log_manager,
    const SecAggSendToServerBase& secagg_send_to_server_impl,
    const size_t& last_received_message_size)
    : secagg_event_publisher_(secagg_event_publisher),
      log_manager_(log_manager),
      secagg_send_to_server_(secagg_send_to_server_impl),
      last_received_message_size_(last_received_message_size) {
  FCP_CHECK(secagg_event_publisher_)
      << "An implementation of "
      << "SecAggEventPublisher must be provided.";
}

void SecAggStateTransitionListenerImpl::Transition(ClientState new_state) {
  FCP_LOG(INFO) << "Transitioning from state: " << static_cast<int>(state_)
                << " to state: " << static_cast<int>(new_state);
  state_ = new_state;
  if (state_ == ClientState::ABORTED) {
    log_manager_->LogDiag(ProdDiagCode::SECAGG_CLIENT_NATIVE_ERROR_GENERIC);
  }
  secagg_event_publisher_->PublishStateTransition(
      new_state, secagg_send_to_server_.last_sent_message_size(),
      last_received_message_size_);
}

void SecAggStateTransitionListenerImpl::Started(ClientState state) {
  // TODO(team): Implement this.
}

void SecAggStateTransitionListenerImpl::Stopped(ClientState state) {
  // TODO(team): Implement this.
}

void SecAggStateTransitionListenerImpl::set_execution_session_id(
    int64_t execution_session_id) {
  secagg_event_publisher_->set_execution_session_id(execution_session_id);
}

}  // namespace client
}  // namespace fcp
