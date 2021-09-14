/*
 * Copyright 2020 Google LLC
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

#ifndef FCP_SECAGG_CLIENT_STATE_TRANSITION_LISTENER_INTERFACE_H_
#define FCP_SECAGG_CLIENT_STATE_TRANSITION_LISTENER_INTERFACE_H_

namespace fcp {
namespace secagg {

enum class ClientState : int {
  INITIAL = 0,
  R0_ADVERTISE_KEYS = 1,
  R1_SHARE_KEYS = 2,
  R2_MASKED_INPUT = 3,
  R3_UNMASKING = 4,
  COMPLETED = 5,
  ABORTED = 6,
};

// Listens for state transition messages.
class StateTransitionListenerInterface {
 public:
  virtual void Transition(ClientState new_state) = 0;
  virtual void set_execution_session_id(int64_t execution_session_id) = 0;

  virtual ~StateTransitionListenerInterface() = default;
};

}  // namespace secagg
}  // namespace fcp

#endif  // FCP_SECAGG_CLIENT_STATE_TRANSITION_LISTENER_INTERFACE_H_
