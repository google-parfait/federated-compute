#ifndef FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_PROTOCOL_H_
#define FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_PROTOCOL_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/aggregation/protocol/aggregation_protocol.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"

namespace fcp::aggregation {

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
      const AggregationProtocol::Callback* callback);

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
  SimpleAggregationProtocol() = default;
};
}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_PROTOCOL_H_
