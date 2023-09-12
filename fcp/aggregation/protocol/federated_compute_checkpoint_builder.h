#ifndef FCP_AGGREGATION_PROTOCOL_FEDERATED_COMPUTE_CHECKPOINT_BUILDER_H_
#define FCP_AGGREGATION_PROTOCOL_FEDERATED_COMPUTE_CHECKPOINT_BUILDER_H_

#include <memory>

#include "fcp/aggregation/protocol/checkpoint_builder.h"

namespace fcp::aggregation {

// A CheckpointBuilderFactory implementation that builds checkpoint using new
// wire format for federated compute.
class FederatedComputeCheckpointBuilderFactory
    : public CheckpointBuilderFactory {
 public:
  std::unique_ptr<CheckpointBuilder> Create() const override;
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_FEDERATED_COMPUTE_CHECKPOINT_BUILDER_H_
