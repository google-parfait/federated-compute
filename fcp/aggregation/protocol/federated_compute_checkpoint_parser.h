#ifndef FCP_AGGREGATION_PROTOCOL_FEDERATED_COMPUTE_CHECKPOINT_PARSER_H_
#define FCP_AGGREGATION_PROTOCOL_FEDERATED_COMPUTE_CHECKPOINT_PARSER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"

namespace fcp::aggregation {

// A CheckpointParserFactory implementation that creates federated compute wire
// format checkpoint parser.
class FederatedComputeCheckpointParserFactory : public CheckpointParserFactory {
 public:
  absl::StatusOr<std::unique_ptr<CheckpointParser>> Create(
      const absl::Cord& serialized_checkpoint) const override;
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_FEDERATED_COMPUTE_CHECKPOINT_PARSER_H_
