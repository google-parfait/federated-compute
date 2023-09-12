#include "fcp/aggregation/protocol/federated_compute_checkpoint_parser.h"

#include <cstdint>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/testing/testing.h"  // IWYU pragma: keep

namespace fcp::aggregation {
namespace {

TEST(FederatedComputeCheckpointParserTest, GetTensors) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DT_INT64, TensorShape({3}), CreateTestData<uint64_t>({1, 2, 3}));
  ASSERT_OK(t1.status());
  absl::StatusOr<Tensor> t2 =
      Tensor::Create(DT_STRING, TensorShape({2}),
                     CreateTestData<absl::string_view>({"value1", "value2"}));
  ASSERT_OK(t2.status());

  EXPECT_OK(builder->Add("t1", *t1));
  EXPECT_OK(builder->Add("t2", *t2));
  auto checkpoint = builder->Build();
  ASSERT_OK(checkpoint.status());

  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  ASSERT_OK(parser.status());
  auto tensor1 = (*parser)->GetTensor("t1");
  ASSERT_OK(tensor1.status());
  auto tensor2 = (*parser)->GetTensor("t2");
  ASSERT_OK(tensor2.status());
  EXPECT_THAT(*tensor1, IsTensor<int64_t>({3}, {1, 2, 3}));
  EXPECT_THAT(*tensor2, IsTensor<absl::string_view>({2}, {"value1", "value2"}));
}
}  // namespace
}  // namespace fcp::aggregation
