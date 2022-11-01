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

#ifndef FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_PROTOCOL_H_
#define FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_PROTOCOL_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/aggregation/protocol/aggregation_protocol.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"
#include "fcp/aggregation/protocol/configuration.pb.h"

namespace fcp::aggregation {

// The structure representing a single aggregation intrinsic.
// TODO(team): Implement mapping of multiple inputs and outputs to
// individual TensorAggregator instances.
struct Intrinsic {
  TensorSpec input;
  TensorSpec output;
  std::unique_ptr<TensorAggregator> aggregator;
};

// Implementation of the simple aggregation protocol.
//
// This version of the protocol receives updates in the clear from clients in a
// TF checkpoint and aggregates them in memory. The aggregated updates are
// released only if the number of participants exceed configured threshold.
class SimpleAggregationProtocol final : public AggregationProtocol {
 public:
  // Factory method to create an instance of the Simple Aggregation Protocol.
  //
  // Does not take ownership of the callback, which must refer to a valid object
  // that outlives the SimpleAggregationProtocol instance.
  static absl::StatusOr<std::unique_ptr<SimpleAggregationProtocol>> Create(
      const Configuration& configuration,
      AggregationProtocol::Callback* callback,
      const CheckpointParserFactory* checkpoint_parser_factory,
      const CheckpointBuilderFactory* checkpoint_builder_factory);

  // Implementation of the overridden Aggregation Protocol methods.
  absl::Status Start(int64_t num_clients) override;
  absl::Status AddClients(int64_t num_clients) override;
  absl::Status ReceiveClientInput(int64_t client_id,
                                  absl::Cord report) override;
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
  SimpleAggregationProtocol(
      std::vector<Intrinsic> intrinsics,
      AggregationProtocol::Callback* callback,
      const CheckpointParserFactory* checkpoint_parser_factory,
      const CheckpointBuilderFactory* checkpoint_builder_factory);

  using TensorMap = absl::flat_hash_map<std::string, Tensor>;
  absl::StatusOr<TensorMap> ParseCheckpoint(absl::Cord report) const;
  // TODO(team): expect this function to be called under the locked
  // state.
  absl::Status AggregateClientInput(TensorMap tensor_map);

  std::vector<Intrinsic> const intrinsics_;
  AggregationProtocol::Callback* const callback_;
  const CheckpointParserFactory* const checkpoint_parser_factory_;
  const CheckpointBuilderFactory* const checkpoint_builder_factory_;
};
}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_PROTOCOL_H_
