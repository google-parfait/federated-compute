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

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/base/monitoring.h"
#include "fcp/tensorflow/status.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"

namespace fcp::aggregation::tensorflow {

namespace tf = ::tensorflow;

tf::TensorShape ConvertShape(const TensorShape& shape) {
  tf::TensorShape tf_shape;
  for (auto dim : shape.dim_sizes()) {
    tf_shape.AddDim(dim);
  }
  FCP_CHECK(tf_shape.IsValid());
  return tf_shape;
}

template <typename T>
const T* GetTensorData(const Tensor& tensor) {
  FCP_CHECK(tensor.is_dense())
      << "Only dense tensors with one slice are supported";
  return static_cast<const T*>(tensor.data().get_slice(0).data);
}

CheckpointWriter::CheckpointWriter(const std::string& filename)
    : tensorflow_writer_(filename,
                         tf::checkpoint::CreateTableTensorSliceBuilder) {}

CheckpointWriter::CheckpointWriter(
    const std::string& filename,
    tf::checkpoint::TensorSliceWriter::CreateBuilderFunction create_builder_fn)
    : tensorflow_writer_(filename, create_builder_fn) {}

absl::Status CheckpointWriter::Add(const std::string& tensor_name,
                                   const Tensor& tensor) {
  tf::TensorShape tf_shape = ConvertShape(tensor.shape());
  tf::Status tf_status;
  tf::TensorSlice tf_slice(tf_shape.dims());
  CASES(tensor.dtype(),
        tf_status = tensorflow_writer_.Add(tensor_name, tf_shape, tf_slice,
                                           GetTensorData<T>(tensor)));
  return ConvertFromTensorFlowStatus(tf_status);
}

absl::Status CheckpointWriter::Finish() {
  return ConvertFromTensorFlowStatus(tensorflow_writer_.Finish());
}

}  // namespace fcp::aggregation::tensorflow
