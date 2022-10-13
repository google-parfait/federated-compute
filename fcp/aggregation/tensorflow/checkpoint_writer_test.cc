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

#include "fcp/aggregation/tensorflow/checkpoint_writer.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/tensorflow/checkpoint_reader.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/testing/testing.h"

namespace fcp::aggregation::tensorflow {
namespace {

using ::testing::Key;
using ::testing::UnorderedElementsAre;

TEST(CheckpointWriterTest, WriteTensors) {
  // Write the checkpoint using Aggregation Core checkpoint writer.
  auto temp_filename = TemporaryTestFile(".ckpt");

  auto t1 = Tensor::Create(DT_FLOAT, TensorShape({4}),
                           CreateTestData<float>({1.0, 2.0, 3.0, 4.0}))
                .value();
  auto t2 = Tensor::Create(DT_INT32, TensorShape({2, 3}),
                           CreateTestData<int32_t>({11, 12, 13, 14, 15, 16}))
                .value();
  auto t3 =
      Tensor::Create(DT_DOUBLE, TensorShape({}), CreateTestData<double>({3.14}))
          .value();

  CheckpointWriter checkpoint_writer(temp_filename);
  EXPECT_OK(checkpoint_writer.Add("a", t1));
  EXPECT_OK(checkpoint_writer.Add("b", t2));
  EXPECT_OK(checkpoint_writer.Add("c", t3));
  EXPECT_OK(checkpoint_writer.Finish());

  // Read the checkpoint using the Aggregation Core checkpoint reader.
  auto checkpoint_reader_or_status = CheckpointReader::Create(temp_filename);
  EXPECT_OK(checkpoint_reader_or_status.status());

  auto checkpoint_reader = std::move(checkpoint_reader_or_status).value();
  EXPECT_THAT(checkpoint_reader->GetDataTypeMap(),
              UnorderedElementsAre(Key("a"), Key("b"), Key("c")));
  EXPECT_THAT(checkpoint_reader->GetTensorShapeMap(),
              UnorderedElementsAre(Key("a"), Key("b"), Key("c")));

  // Read and verify the tensors.
  EXPECT_THAT(*checkpoint_reader->GetTensor("a"),
              IsTensor<float>({4}, {1.0, 2.0, 3.0, 4.0}));
  EXPECT_THAT(*checkpoint_reader->GetTensor("b"),
              IsTensor<int32_t>({2, 3}, {11, 12, 13, 14, 15, 16}));
  EXPECT_THAT(*checkpoint_reader->GetTensor("c"), IsTensor<double>({}, {3.14}));
}

TEST(CheckpointWriterTest, SparseTensorFailure) {
  CheckpointWriter checkpoint_writer(TemporaryTestFile(".ckpt"));
  // This creates a tensor with zero slices.
  Tensor t =
      Tensor::Create(DT_INT32, TensorShape({10}), CreateTestData<int32_t>(10))
          .value();
  EXPECT_DEATH(checkpoint_writer.Add("foo", t).IgnoreError(),
               "Only dense tensors with one slice are supported");
}

}  // namespace
}  // namespace fcp::aggregation::tensorflow
