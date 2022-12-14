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

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/base/monitoring.h"

namespace fcp::aggregation {

absl::StatusOr<std::unique_ptr<SecureAggregationProtocol>>
SecureAggregationProtocol::Create(
    const Configuration& /*configuration*/,
    AggregationProtocol::Callback* callback,
    std::unique_ptr<fcp::Scheduler> worker_scheduler,
    std::unique_ptr<fcp::Scheduler> callback_scheduler) {
  // TODO(team): Create round advancement policy
  return absl::WrapUnique(new SecureAggregationProtocol(
      callback, std::move(worker_scheduler), std::move(callback_scheduler)));
}

// TODO(team): Implement Secure Aggregation Protocol methods.
absl::Status SecureAggregationProtocol::Start(int64_t num_clients) {
  // TODO(team): Populate SecureAggregation field in  AcceptanceMessage
  // with parameters required by SecAgg client.
  AcceptanceMessage acceptance_message;
  callback_->OnAcceptClients(0, num_clients, acceptance_message);
  return absl::OkStatus();
}

absl::Status SecureAggregationProtocol::AddClients(int64_t /*num_clients*/) {
  return absl::UnimplementedError(
      "AddClients is not relevant to SecureAggregationProtocol");
}

absl::Status SecureAggregationProtocol::ReceiveClientInput(int64_t client_id,
                                                           absl::Cord report) {
  return absl::UnimplementedError("ReceiveClientInput is not implemented");
}

absl::Status SecureAggregationProtocol::ReceiveClientMessage(
    int64_t client_id, const ClientMessage& message) {
  return absl::UnimplementedError("ReceiveClientMessage is not implemented");
}

absl::Status SecureAggregationProtocol::CloseClient(
    int64_t client_id, absl::Status client_status) {
  return absl::UnimplementedError("CloseClient is not implemented");
}

absl::Status SecureAggregationProtocol::Complete() {
  return absl::UnimplementedError("Complete is not implemented");
}

absl::Status SecureAggregationProtocol::Abort() {
  return absl::UnimplementedError("Abort is not implemented");
}

StatusMessage SecureAggregationProtocol::GetStatus() {
  StatusMessage status_message;
  // TODO(team): Populate status_message before returning.
  return status_message;
}

SecureAggregationProtocol::SecureAggregationProtocol(
    Callback* callback, std::unique_ptr<fcp::Scheduler> worker_scheduler,
    std::unique_ptr<fcp::Scheduler> callback_scheduler)
    : callback_(callback),
      worker_scheduler_(std::move(worker_scheduler)),
      callback_scheduler_(std::move(callback_scheduler)) {}

}  // namespace fcp::aggregation
