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

#include "fcp/aggregation/testing/testing.h"

#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "fcp/base/platform.h"
#include "fcp/tensorflow/status.h"
#include "fcp/testing/testing.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace fcp::aggregation {

using ::tensorflow::StatusFromTF_Status;
using ::tensorflow::TF_StatusPtr;
using ::tensorflow::checkpoint::CheckpointReader;

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  DTYPE_CASES(tensor.dtype(), T,
              DescribeTensor<T>(&os, tensor.dtype(), tensor.shape(),
                                TensorValuesToVector<T>(tensor)));
  return os;
}

tf::Tensor CreateStringTfTensor(std::initializer_list<int64_t> dim_sizes,
                                std::initializer_list<string_view> values) {
  tf::TensorShape shape;
  EXPECT_TRUE(tf::TensorShape::BuildTensorShape(dim_sizes, &shape).ok());
  tf::Tensor tensor(tf::DT_STRING, shape);
  auto* tensor_data_ptr = reinterpret_cast<tf::tstring*>(tensor.data());
  for (auto value : values) {
    *tensor_data_ptr++ = value;
  }
  return tensor;
}

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
}  // namespace fcp::aggregation
