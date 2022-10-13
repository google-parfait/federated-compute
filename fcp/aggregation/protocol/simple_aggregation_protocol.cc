#include "fcp/aggregation/protocol/simple_aggregation_protocol.h"

#include <memory>

namespace fcp::aggregation {

absl::StatusOr<std::unique_ptr<SimpleAggregationProtocol>>
SimpleAggregationProtocol::Create(
    const Configuration& configuration,
    const AggregationProtocol::Callback* callback) {
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
