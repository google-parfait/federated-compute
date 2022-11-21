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

#include "fcp/aggregation/tensorflow/tensorflow_checkpoint_builder_factory.h"

#include <memory>
#include <string>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/platform.h"
#include "fcp/tensorflow/status.h"
#include "fcp/testing/testing.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/tensor.h"

namespace fcp::aggregation::tensorflow {
namespace {

using ::tensorflow::StatusFromTF_Status;
using ::tensorflow::TF_StatusPtr;
using ::tensorflow::checkpoint::CheckpointReader;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
SummarizeCheckpoint(const absl::Cord& checkpoint) {
  std::string filename = TemporaryTestFile(".ckpt");
  FCP_RETURN_IF_ERROR(WriteCordToFile(filename, checkpoint));

  TF_StatusPtr tf_status(TF_NewStatus());
  auto reader = std::make_unique<CheckpointReader>(filename, tf_status.get());
  FCP_RETURN_IF_ERROR(
      ConvertFromTensorFlowStatus(StatusFromTF_Status(tf_status.get())));

  absl::flat_hash_map<std::string, std::string> tensors;
  for (const auto& [name, shape] : reader->GetVariableToShapeMap()) {
    std::unique_ptr<::tensorflow::Tensor> tensor;
    reader->GetTensor(name, &tensor, tf_status.get());
    FCP_RETURN_IF_ERROR(
        ConvertFromTensorFlowStatus(StatusFromTF_Status(tf_status.get())));
    tensors[name] = tensor->SummarizeValue(/*max_entries=*/10);
  }
  return tensors;
}

TEST(TensorflowCheckpointBuilderFactoryTest, BuildCheckpoint) {
  TensorflowCheckpointBuilderFactory factory;
  std::unique_ptr<CheckpointBuilder> builder = factory.Create();

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DT_FLOAT, TensorShape({4}), CreateTestData<float>({1.0, 2.0, 3.0, 4.0}));
  ASSERT_OK(t1.status());
  absl::StatusOr<Tensor> t2 = Tensor::Create(DT_FLOAT, TensorShape({2}),
                                             CreateTestData<float>({5.0, 6.0}));
  ASSERT_OK(t2.status());

  EXPECT_OK(builder->Add("t1", *t1));
  EXPECT_OK(builder->Add("t2", *t2));
  absl::StatusOr<absl::Cord> checkpoint = builder->Build();
  ASSERT_OK(checkpoint.status());
  auto summary = SummarizeCheckpoint(*checkpoint);
  ASSERT_OK(summary.status());
  EXPECT_THAT(*summary,
              UnorderedElementsAre(Pair("t1", "1 2 3 4"), Pair("t2", "5 6")));
}

// Check that multiple checkpoints can be built simultanously.
TEST(TensorflowCheckpointBuilderFactoryTest, SimultaneousWrites) {
  TensorflowCheckpointBuilderFactory factory;

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DT_FLOAT, TensorShape({4}), CreateTestData<float>({1.0, 2.0, 3.0, 4.0}));
  ASSERT_OK(t1.status());
  absl::StatusOr<Tensor> t2 = Tensor::Create(DT_FLOAT, TensorShape({2}),
                                             CreateTestData<float>({5.0, 6.0}));
  ASSERT_OK(t2.status());

  std::unique_ptr<CheckpointBuilder> builder1 = factory.Create();
  std::unique_ptr<CheckpointBuilder> builder2 = factory.Create();
  EXPECT_OK(builder1->Add("t1", *t1));
  EXPECT_OK(builder2->Add("t2", *t2));
  absl::StatusOr<absl::Cord> checkpoint1 = builder1->Build();
  ASSERT_OK(checkpoint1.status());
  absl::StatusOr<absl::Cord> checkpoint2 = builder2->Build();
  ASSERT_OK(checkpoint2.status());
  auto summary1 = SummarizeCheckpoint(*checkpoint1);
  ASSERT_OK(summary1.status());
  EXPECT_THAT(*summary1, UnorderedElementsAre(Pair("t1", "1 2 3 4")));
  auto summary2 = SummarizeCheckpoint(*checkpoint2);
  ASSERT_OK(summary2.status());
  EXPECT_THAT(*summary2, UnorderedElementsAre(Pair("t2", "5 6")));
}

}  // namespace
}  // namespace fcp::aggregation::tensorflow
