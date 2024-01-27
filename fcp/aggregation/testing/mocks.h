#ifndef FCP_AGGREGATION_TESTING_MOCKS_H_
#define FCP_AGGREGATION_TESTING_MOCKS_H_

#include <cstdint>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"
#include "fcp/aggregation/protocol/resource_resolver.h"

namespace fcp::aggregation {

class MockCheckpointParser : public CheckpointParser {
 public:
  MOCK_METHOD(absl::StatusOr<Tensor>, GetTensor, (const std::string& name),
              (override));
};

class MockCheckpointParserFactory : public CheckpointParserFactory {
 public:
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<CheckpointParser>>, Create,
              (const absl::Cord& serialized_checkpoint), (const override));
};

class MockCheckpointBuilder : public CheckpointBuilder {
 public:
  MOCK_METHOD(absl::Status, Add,
              (const std::string& name, const Tensor& tensor), (override));
  MOCK_METHOD(absl::StatusOr<absl::Cord>, Build, (), (override));
};

class MockCheckpointBuilderFactory : public CheckpointBuilderFactory {
 public:
  MOCK_METHOD(std::unique_ptr<CheckpointBuilder>, Create, (), (const override));
};

class MockResourceResolver : public ResourceResolver {
 public:
  MOCK_METHOD(absl::StatusOr<absl::Cord>, RetrieveResource,
              (int64_t client_id, const std::string& uri), (override));
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_TESTING_MOCKS_H_
