// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fcp/aggregation/checkpoint/federated_compute_checkpoint_parser.h"

#include <cstdint>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/aggregation/checkpoint/checkpoint_builder.h"
#include "fcp/aggregation/checkpoint/checkpoint_parser.h"
#include "fcp/aggregation/checkpoint/federated_compute_checkpoint_builder.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/testing/testing.h"

namespace fcp::aggregation {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(FederatedComputeCheckpointParserTest, GetTensors) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();

  Tensor t1({1l, 2l, 3l});
  Tensor t2({"value1", "value2"});
  Tensor t3({1, 2});

  ABSL_EXPECT_OK(builder->Add("t1", t1));
  ABSL_EXPECT_OK(builder->Add("t2", t2));
  ABSL_EXPECT_OK(builder->Add("t3", t3));
  auto checkpoint = builder->Build();
  ABSL_ASSERT_OK(checkpoint.status());

  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  ABSL_ASSERT_OK(parser.status());
  auto tensor1 = (*parser)->GetTensor("t1");
  ABSL_ASSERT_OK(tensor1.status());
  auto tensor2 = (*parser)->GetTensor("t2");
  ABSL_ASSERT_OK(tensor2.status());
  auto tensor3 = (*parser)->GetTensor("t3");
  ABSL_ASSERT_OK(tensor2.status());
  EXPECT_THAT(*tensor1, IsTensor<int64_t>({3}, {1, 2, 3}));
  EXPECT_THAT(*tensor2, IsTensor<absl::string_view>({2}, {"value1", "value2"}));
  EXPECT_THAT(*tensor3, IsTensor<int32_t>({2}, {1, 2}));
  EXPECT_EQ((*tensor1).name(), "t1");
  EXPECT_EQ((*tensor2).name(), "t2");
  EXPECT_EQ((*tensor3).name(), "t3");
}

TEST(FederatedComputeCheckpointParserTest, LoadAllTensors) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();

  Tensor t1({1l, 2l, 3l});
  Tensor t2({"value1", "value2"});
  Tensor t3({1, 2});

  ABSL_EXPECT_OK(builder->Add("t1", t1));
  ABSL_EXPECT_OK(builder->Add("t2", t2));
  ABSL_EXPECT_OK(builder->Add("t3", t3));
  auto checkpoint = builder->Build();
  ABSL_ASSERT_OK(checkpoint.status());

  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  ABSL_ASSERT_OK(parser.status());

  auto tensors = (*parser)->LoadAllTensors();
  ABSL_ASSERT_OK(tensors.status());
  EXPECT_THAT(
      *tensors,
      UnorderedElementsAre(
          Pair("t1", IsTensor<int64_t>({3}, {1, 2, 3})),
          Pair("t2", IsTensor<absl::string_view>({2}, {"value1", "value2"})),
          Pair("t3", IsTensor<int32_t>({2}, {1, 2}))));
}

}  // namespace
}  // namespace fcp::aggregation
