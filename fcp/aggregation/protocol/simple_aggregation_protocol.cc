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

#include "fcp/aggregation/protocol/simple_aggregation_protocol.h"

#include <memory>

namespace fcp::aggregation {

absl::StatusOr<std::unique_ptr<SimpleAggregationProtocol>>
SimpleAggregationProtocol::Create(
    const Configuration& configuration,
    AggregationProtocol::Callback* const callback) {
  // TODO(team): Parse configuration and initialize TensorAggregators.
  return absl::WrapUnique(new SimpleAggregationProtocol());
}

// TODO(team): Implement Simple Aggregation Protocol methods.
absl::Status SimpleAggregationProtocol::Start(int64_t num_clients) {
  return absl::UnimplementedError("Start is not implemented");
}

absl::Status SimpleAggregationProtocol::AddClients(int64_t num_clients) {
  return absl::UnimplementedError("AddClients is not implemented");
}

absl::Status SimpleAggregationProtocol::ReceiveClientInput(int64_t client_id,
                                                           absl::Cord report) {
  return absl::UnimplementedError("ReceiveClientInput is not implemented");
}

absl::Status SimpleAggregationProtocol::ReceiveClientMessage(
    int64_t client_id, const ClientMessage& message) {
  return absl::UnimplementedError(
      "ReceiveClientMessage is not supported by SimpleAggregationProtocol");
}

absl::Status SimpleAggregationProtocol::CloseClient(
    int64_t client_id, absl::Status client_status) {
  return absl::UnimplementedError("CloseClient is not implemented");
}

absl::Status SimpleAggregationProtocol::Complete() {
  return absl::UnimplementedError("Complete is not implemented");
}

absl::Status SimpleAggregationProtocol::Abort() {
  return absl::UnimplementedError("Abort is not implemented");
}

StatusMessage SimpleAggregationProtocol::GetStatus() {
  StatusMessage status_message;
  // TODO(team): Populate status_message before returning.
  return status_message;
}

}  // namespace fcp::aggregation
