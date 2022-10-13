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

#include "fcp/aggregation/tensorflow/checkpoint_reader.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/platform.h"
#include "fcp/testing/testing.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/public/session.h"

namespace fcp::aggregation::tensorflow {
namespace {

namespace tf = ::tensorflow;

using ::testing::Key;
using ::testing::UnorderedElementsAre;

template <typename T>
tf::Tensor CreateTfTensor(tf::DataType data_type,
                          std::initializer_list<int64_t> dim_sizes,
                          std::initializer_list<T> values) {
  tf::TensorShape shape;
  EXPECT_TRUE(tf::TensorShape::BuildTensorShape(dim_sizes, &shape).ok());
  tf::Tensor tensor(data_type, shape);
  T* tensor_data_ptr = reinterpret_cast<T*>(tensor.data());
  for (auto value : values) {
    *tensor_data_ptr++ = value;
  }
  return tensor;
}

// Wrapper around tf::ops::Save that sets up and runs the op.
tf::Status CreateTfCheckpoint(tf::Input filename, tf::Input tensor_names,
                              tf::InputList tensors) {
  tf::Scope scope = tf::Scope::NewRootScope();

  tf::ops::Save save(scope, std::move(filename), std::move(tensor_names),
                     std::move(tensors));

  tf::GraphDef graph;
  if (auto s = scope.ToGraphDef(&graph); !s.ok()) return s;

  auto session = absl::WrapUnique(tf::NewSession(tf::SessionOptions()));
  if (auto s = session->Create(graph); !s.ok()) return s;
  return session->Run({}, {}, {save.operation.node()->name()}, nullptr);
}

TEST(CheckpointReaderTest, ReadTensors) {
  // Write a test TF checkpoint with 3 tensors
  auto temp_filename = TemporaryTestFile(".ckpt");
  auto tensor_a =
      CreateTfTensor<float>(tf::DT_FLOAT, {4}, {1.0, 2.0, 3.0, 4.0});
  auto tensor_b =
      CreateTfTensor<int32_t>(tf::DT_INT32, {2, 3}, {11, 12, 13, 14, 15, 16});
  auto tensor_c = CreateTfTensor<double>(tf::DT_DOUBLE, {}, {3.14});
  EXPECT_TRUE(CreateTfCheckpoint(temp_filename, {"a", "b", "c"},
                                 {tensor_a, tensor_b, tensor_c})
                  .ok());

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

TEST(CheckpointReaderTest, InvalidFileName) {
  auto checkpoint_reader_or_status = CheckpointReader::Create("foo/bar");
  EXPECT_THAT(checkpoint_reader_or_status, IsCode(INTERNAL));
}

TEST(CheckpointReaderTest, MalformedFile) {
  auto temp_filename = TemporaryTestFile(".ckpt");
  WriteStringToFile(temp_filename, "foobar").IgnoreError();
  auto checkpoint_reader_or_status = CheckpointReader::Create(temp_filename);
  EXPECT_THAT(checkpoint_reader_or_status, IsCode(INTERNAL));
}

}  // namespace
}  // namespace fcp::aggregation::tensorflow
