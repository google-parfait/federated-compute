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

#ifndef FCP_AGGREGATION_PROTOCOL_SECURE_AGGREGATION_SECURE_AGGREGATION_PROTOCOL_H_
#define FCP_AGGREGATION_PROTOCOL_SECURE_AGGREGATION_SECURE_AGGREGATION_PROTOCOL_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/aggregation/protocol/aggregation_protocol.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/base/scheduler.h"

namespace fcp::aggregation {

/**
 * Implementation of the secure aggregation protocol for the Federated Learning
 * server.
 *
 * In this version of the protocol client input is transmitted in encrypted
 * form via the Secure Aggregation protocol: The Secure Aggregation protocol
 * provides cryptographic guarantees that aggregates must be accumulated across
 * at minimum threshold of users before the server can successfully decrypt
 * them. The protocol also provides guarantees against the ability of attackers
 * to subvert privacy; for example, the protocol can ensure that users' inputs
 * remain private even against an adversary who can observe all communications
 * and internal state of the server and who can cause up to 1/3 of the clients
 * to deviate arbitrarily from the protocol.
 *
 * The SecAgg protocol proceeds in rounds, where a threshold number of clients
 * are required to complete each round before the protocol can proceed to the
 * next one and eventually produce an output in final round. It is therefore
 * important to have a round advancement policy that ensures that sufficient
 * clients make it to the final round in a reasonable time. This can be guided
 * by a per-round timeout or some other mechanism. The main purpose of
 * SecureAggregationProtocol is precisely to orchestate such round advancement,
 * and there for it does not include the cryptographic protocol implementation,
 * instead it
 * (a) re-routes al message to the underlying SecAggServer instance, and
 * (b) determines when to instruct SecAggServer to advance to the next round by
 *     following a round advancement policy.
 *
 */
class SecureAggregationProtocol final : public AggregationProtocol {
 public:
  // Factory method to create an instance of the Secure Aggregation Protocol.
  //
  // Does not take ownership of the callback, which must refer to a valid object
  // that outlives the SecureAggregationProtocol instance.
  // Similarly, does not take ownership of the schedulers, which must refer to a
  // valid object that outlives the SecureAggregationProtocol instance.
  static absl::StatusOr<std::unique_ptr<SecureAggregationProtocol>> Create(
      const Configuration& configuration,
      std::unique_ptr<fcp::Scheduler> worker_scheduler,
      std::unique_ptr<fcp::Scheduler> callback_scheduler);

  // Implementation of the overridden Aggregation Protocol methods.
  absl::Status Start(int64_t num_clients) override;
  absl::Status AddClients(int64_t num_clients) override;
  // Each client sends at most one (encrypted) input in round 2 of the protocol.
  absl::Status ReceiveClientInput(int64_t client_id,
                                  absl::Cord report) override;
  // Each client sends one message in each round of the protocol, except for
  // round 2, when it sends an encrypted input.
  absl::Status ReceiveClientMessage(int64_t client_id,
                                    const ClientMessage& message) override;
  absl::Status CloseClient(int64_t client_id,
                           absl::Status client_status) override;
  absl::Status Complete() override;
  absl::Status Abort() override;
  StatusMessage GetStatus() override;

  ~SecureAggregationProtocol() override = default;

  // SecureAggregationProtocol is neither copyable nor movable.
  SecureAggregationProtocol(const SecureAggregationProtocol&) = delete;
  SecureAggregationProtocol& operator=(const SecureAggregationProtocol&) =
      delete;

 private:
  // Private constructor
  SecureAggregationProtocol(std::unique_ptr<fcp::Scheduler>,
                            std::unique_ptr<fcp::Scheduler>);

  std::unique_ptr<fcp::Scheduler> worker_scheduler_;
  std::unique_ptr<fcp::Scheduler> callback_scheduler_;
};
}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_SECURE_AGGREGATION_SECURE_AGGREGATION_PROTOCOL_H_
