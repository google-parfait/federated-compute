/*
 * Copyright 2025 Google LLC
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
#include "fcp/aggregation/group_by/partitioner.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

absl::StatusOr<Partitioner> CreateDefaultPartitioner(int num_partitions) {
  std::vector<Tensor> key_tensors;
  key_tensors.push_back(Tensor({"a", "b", "c"}));
  key_tensors.push_back(Tensor(std::vector<int64_t>{1, 2, 3}));
  key_tensors.push_back(Tensor({2.0, 3.0, 4.0}));
  return Partitioner::Create(key_tensors, num_partitions);
}

TEST(PartitionerTest, CreatePartitioner_Succeeds) {
  std::vector<Tensor> key_tensors;
  key_tensors.push_back(Tensor({"a", "b", "c"}));
  key_tensors.push_back(Tensor(std::vector<int64_t>{1, 2, 3}));
  key_tensors.push_back(Tensor({1.0, 2.0, 3.0}));

  EXPECT_THAT(Partitioner::Create(key_tensors, /*num_partitions=*/2), IsOk());
}

TEST(PartitionerTest, CreatePartitioner_EmptyKeyTensors) {
  std::vector<Tensor> key_tensors;
  FCP_ASSERT_OK_AND_ASSIGN(
      Partitioner partitioner,
      Partitioner::Create(key_tensors, /*num_partitions=*/2));
  EXPECT_EQ(partitioner.GetNumPartitions(), 2);

  FCP_ASSERT_OK_AND_ASSIGN(auto slices,
                           partitioner.PartitionData<double>({42.0}));
  EXPECT_EQ(slices.size(), 2);
  EXPECT_THAT(slices[0], ::testing::ElementsAre(42.0));
  EXPECT_TRUE(slices[1].empty());
}

TEST(PartitionerTest, CreatePartitioner_KeyTensorNotOneDimensional) {
  std::vector<Tensor> key_tensors;
  key_tensors.push_back(Tensor::Create(DT_STRING, {1, 2},
                                       CreateTestData<string_view>({
                                           "a",
                                           "b",
                                       }))
                            .value());
  EXPECT_THAT(Partitioner::Create(key_tensors,
                                  /*num_partitions=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "Expected key tensor to be one-dimensional.")));
}

TEST(PartitionerTest, CreatePartitioner_KeyTensorsWithDifferentSizes) {
  std::vector<Tensor> key_tensors;
  key_tensors.push_back(Tensor({"a", "b", "c"}));
  key_tensors.push_back(Tensor(std::vector<int64_t>{1, 2}));
  EXPECT_THAT(Partitioner::Create(key_tensors,
                                  /*num_partitions=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("All key tensors must have the "
                                            "same one-dimensional size.")));
}

TEST(PartitionerTest, PartitionKeys_Succeeds) {
  std::vector<Tensor> key_tensors;
  key_tensors.push_back(Tensor({"a", "b", "c"}));
  key_tensors.push_back(Tensor(std::vector<int64_t>{1, 2, 3}));
  key_tensors.push_back(Tensor({2.0, 3.0, 4.0}));
  FCP_ASSERT_OK_AND_ASSIGN(
      Partitioner partitioner,
      Partitioner::Create(key_tensors, /*num_partitions=*/2));

  FCP_ASSERT_OK_AND_ASSIGN(std::vector<Tensor> partitioned_tensors,
                           partitioner.PartitionKeys(key_tensors[0]));

  // Verify the contents of the partitioned tensors.
  EXPECT_EQ(partitioned_tensors.size(), 2);
  EXPECT_THAT(partitioned_tensors[0],
              IsTensor<string_view>({"c"}));
  EXPECT_THAT(partitioned_tensors[1],
              IsTensor<string_view>({"a", "b"}));
}

TEST(PartitionerTest, PartitionAggregatorData_Succeeds) {
  FCP_ASSERT_OK_AND_ASSIGN(Partitioner partitioner,
                           CreateDefaultPartitioner(/*num_partitions=*/2));

  FCP_ASSERT_OK_AND_ASSIGN(auto slices,
                           partitioner.PartitionData<double>({1.0, 2.0, 3.0}));

  // Verify the contents of the partitioned tensors.
  EXPECT_EQ(slices.size(), 2);
  EXPECT_THAT(slices[0], ::testing::UnorderedElementsAre(3.0));
  EXPECT_THAT(slices[1], ::testing::UnorderedElementsAre(1.0, 2.0));
}

TEST(PartitionerTest, PartitionData_InvalidInputSize) {
  FCP_ASSERT_OK_AND_ASSIGN(Partitioner partitioner,
                           CreateDefaultPartitioner(/*num_partitions=*/2));

  EXPECT_THAT(partitioner.PartitionData<double>({1.0, 2.0, 3.0, 4.0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "The number of elements in the input data "
                           "should be equal to the number of hashes.")));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
