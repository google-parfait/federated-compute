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

#ifndef FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_SIMPLE_AGGREGATION_PROTOCOL_H_
#define FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_SIMPLE_AGGREGATION_PROTOCOL_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/aggregation/protocol/aggregation_protocol.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/aggregation/protocol/resource_resolver.h"

namespace fcp::aggregation {

// Implementation of the simple aggregation protocol.
//
// This version of the protocol receives updates in the clear from clients in a
// TF checkpoint and aggregates them in memory. The aggregated updates are
// released only if the number of participants exceed configured threshold.
class SimpleAggregationProtocol final : public AggregationProtocol {
 public:
  // Validates the Configuration that will subsequently be used to create an
  // instance of this protocol.
  // Returns INVALID_ARGUMENT if the configuration is invalid.
  static absl::Status ValidateConfig(const Configuration& configuration);

  // Factory method to create an instance of the Simple Aggregation Protocol.
  //
  // Does not take ownership of the callback, which must refer to a valid object
  // that outlives the SimpleAggregationProtocol instance.
  static absl::StatusOr<std::unique_ptr<SimpleAggregationProtocol>> Create(
      const Configuration& configuration,
      AggregationProtocol::Callback* callback,
      const CheckpointParserFactory* checkpoint_parser_factory,
      const CheckpointBuilderFactory* checkpoint_builder_factory,
      ResourceResolver* resource_resolver);

  // Implementation of the overridden Aggregation Protocol methods.
  absl::Status Start(int64_t num_clients) override;
  absl::Status AddClients(int64_t num_clients) override;
  absl::Status ReceiveClientMessage(int64_t client_id,
                                    const ClientMessage& message) override;
  absl::Status CloseClient(int64_t client_id,
                           absl::Status client_status) override;
  absl::Status Complete() override;
  absl::Status Abort() override;
  StatusMessage GetStatus() override;

  ~SimpleAggregationProtocol() override = default;

  // SimpleAggregationProtocol is neither copyable nor movable.
  SimpleAggregationProtocol(const SimpleAggregationProtocol&) = delete;
  SimpleAggregationProtocol& operator=(const SimpleAggregationProtocol&) =
      delete;

 private:
  // The structure representing a single aggregation intrinsic.
  // TODO(team): Implement mapping of multiple inputs and outputs to
  // individual TensorAggregator instances.
  struct Intrinsic {
    TensorSpec input;
    TensorSpec output;
    std::unique_ptr<TensorAggregator> aggregator
        ABSL_PT_GUARDED_BY(&SimpleAggregationProtocol::aggregation_mu_);
  };

  // Private constructor.
  SimpleAggregationProtocol(
      std::vector<Intrinsic> intrinsics,
      AggregationProtocol::Callback* callback,
      const CheckpointParserFactory* checkpoint_parser_factory,
      const CheckpointBuilderFactory* checkpoint_builder_factory,
      ResourceResolver* resource_resolver);

  // Creates an aggregation intrinsic based on the intrinsic configuration.
  static absl::StatusOr<Intrinsic> CreateIntrinsic(
      const Configuration::ServerAggregationConfig& aggregation_config);

  // Describes the overall protocol state.
  enum ProtocolState {
    // The initial state indicating that the protocol was created.
    PROTOCOL_CREATED,
    // The protocol `Start` method has been called.
    PROTOCOL_STARTED,
    // The protocol `Complete` method has finished successfully.
    PROTOCOL_COMPLETED,
    // The protocol `Abort` method has been called.
    PROTOCOL_ABORTED
  };

  // Describes state of each client participating in the protocol.
  enum ClientState : uint8_t {
    // No input received from the client yet.
    CLIENT_PENDING,
    // Client input received but the aggregation still pending, which may
    // be the case when there are multiple concurrent ReceiveClientMessage
    // calls.
    CLIENT_RECEIVED_INPUT_AND_PENDING,
    // Client input has been successfully aggregated.
    CLIENT_COMPLETED,
    // Client failed either by being closed with an error or by submitting a
    // malformed input.
    CLIENT_FAILED,
    // Client which has been aborted by the server before its input has been
    // received.
    CLIENT_ABORTED,
    // Client input has been received but discarded, for example due to the
    // protocol Abort method being called.
    CLIENT_DISCARDED
  };

  // Returns string representation of the protocol state.
  static absl::string_view ProtocolStateDebugString(ProtocolState state);

  // Returns string representation of the client state.
  static absl::string_view ClientStateDebugString(ClientState state);

  // Returns an error if the current protocol state isn't the expected one.
  absl::Status CheckProtocolState(ProtocolState state) const
      ABSL_SHARED_LOCKS_REQUIRED(state_mu_);

  // Changes the protocol state.
  void SetProtocolState(ProtocolState state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // Gets the client state for the given client ID.
  absl::StatusOr<ClientState> GetClientState(int64_t client_id) const
      ABSL_SHARED_LOCKS_REQUIRED(state_mu_);

  // Sets the client state for the given client ID.
  void SetClientState(int64_t client_id, ClientState state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  // Parses and validates the client report.
  // This function involves a potentially expensive I/O and parsing and should
  // run concurrently as much as possible. The ABSL_LOCKS_EXCLUDED attribution
  // below is used to emphasize that.
  using TensorMap = absl::flat_hash_map<std::string, Tensor>;
  absl::StatusOr<TensorMap> ParseCheckpoint(absl::Cord report) const
      ABSL_LOCKS_EXCLUDED(state_mu_, aggregation_mu_);

  // Aggregates the input via the underlying aggregators.
  absl::Status AggregateClientInput(TensorMap tensor_map)
      ABSL_LOCKS_EXCLUDED(state_mu_, aggregation_mu_);

  // Produces the report via the underlying aggregators.
  absl::StatusOr<absl::Cord> CreateReport()
      ABSL_LOCKS_EXCLUDED(aggregation_mu_);

  // Protects the mutable state.
  absl::Mutex state_mu_;
  // Protects calls into the aggregators.
  absl::Mutex aggregation_mu_;
  // This indicates that the aggregation has finished either by completing
  // the protocol or by aborting it. This can be triggered without locking on
  // the aggregation_mu_ mutex first to allow aborting the protocol promptly and
  // discarding all the pending aggregation calls.
  std::atomic_bool aggregation_finished_ = false;

  // The overall state of the protocol.
  ProtocolState protocol_state_ ABSL_GUARDED_BY(state_mu_);

  // Holds state of all clients. The length of the vector equals
  // to the number of clients accepted into the protocol.
  std::vector<ClientState> client_states_ ABSL_GUARDED_BY(state_mu_);

  // Counters for various client states other than pending.
  // Note that the number of pending clients can be found by subtracting the
  // sum of the below counters from `client_states_.size()`.
  uint64_t num_clients_received_and_pending_ ABSL_GUARDED_BY(state_mu_) = 0;
  uint64_t num_clients_aggregated_ ABSL_GUARDED_BY(state_mu_) = 0;
  uint64_t num_clients_failed_ ABSL_GUARDED_BY(state_mu_) = 0;
  uint64_t num_clients_aborted_ ABSL_GUARDED_BY(state_mu_) = 0;
  uint64_t num_clients_discarded_ ABSL_GUARDED_BY(state_mu_) = 0;

  // Intrinsics are immutable and shouldn't be guarded by the either of mutexes.
  // Please note that the access to the aggregators that intrinsics point to
  // still needs to be strictly sequential. That is guarded separatedly by
  // `aggregators_mu_`.
  std::vector<Intrinsic> const intrinsics_;

  AggregationProtocol::Callback* const callback_;
  const CheckpointParserFactory* const checkpoint_parser_factory_;
  const CheckpointBuilderFactory* const checkpoint_builder_factory_;
  ResourceResolver* const resource_resolver_;
};
}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_SIMPLE_AGGREGATION_PROTOCOL_H_
