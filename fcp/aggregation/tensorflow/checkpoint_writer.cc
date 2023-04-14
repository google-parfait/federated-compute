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
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/base/monitoring.h"
#include "fcp/tensorflow/status.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/platform/tstring.h"

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
tf::Status AddTensorSlice(tf::checkpoint::TensorSliceWriter* writer,
                          const std::string& name, const tf::TensorShape& shape,
                          const tf::TensorSlice& slice, const Tensor& tensor) {
  return writer->Add<T>(name, shape, slice,
                        static_cast<const T*>(tensor.data().data()));
}

template <>
tf::Status AddTensorSlice<string_view>(
    tf::checkpoint::TensorSliceWriter* writer, const std::string& name,
    const tf::TensorShape& shape, const tf::TensorSlice& slice,
    const Tensor& tensor) {
  std::vector<tf::tstring> values(tensor.shape().NumElements());
  const auto* string_views =
      static_cast<const string_view*>(tensor.data().data());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i].assign_as_view(string_views[i].data(), string_views[i].size());
  }
  return writer->Add(name, shape, slice, values.data());
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
  tf::TensorSlice tf_slice(tf_shape.dims());
  FCP_CHECK(tensor.is_dense())
      << "Only dense tensors with one slice are supported";
  tf::Status tf_status;
  DTYPE_CASES(tensor.dtype(), T,
              tf_status = AddTensorSlice<T>(&tensorflow_writer_, tensor_name,
                                            tf_shape, tf_slice, tensor));
  return ConvertFromTensorFlowStatus(tf_status);
}

absl::Status CheckpointWriter::Finish() {
  return ConvertFromTensorFlowStatus(tensorflow_writer_.Finish());
}

}  // namespace fcp::aggregation::tensorflow
