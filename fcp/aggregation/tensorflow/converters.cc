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

#include "fcp/aggregation/tensorflow/converters.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/base/monitoring.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace fcp::aggregation::tensorflow {

namespace tf = ::tensorflow;

StatusOr<DataType> ConvertDataType(tf::DataType dtype) {
  switch (dtype) {
    case tf::DT_FLOAT:
      return DT_FLOAT;
    case tf::DT_DOUBLE:
      return DT_DOUBLE;
    case tf::DT_INT32:
      return DT_INT32;
    case tf::DT_INT64:
      return DT_INT64;
    case tf::DT_STRING:
      return DT_STRING;
    default:
      return FCP_STATUS(INVALID_ARGUMENT)
             << "Unsupported tf::DataType: " << dtype;
  }
}

TensorShape ConvertShape(const tf::TensorShape& shape) {
  FCP_CHECK(shape.IsFullyDefined());
  std::vector<size_t> dim_sizes;
  for (auto dim_size : shape.dim_sizes()) {
    FCP_CHECK(dim_size >= 0);
    dim_sizes.push_back(dim_size);
  }
  return TensorShape(dim_sizes.begin(), dim_sizes.end());
}

StatusOr<TensorSpec> ConvertTensorSpec(
    const ::tensorflow::TensorSpecProto& spec) {
  FCP_ASSIGN_OR_RETURN(DataType dtype, ConvertDataType(spec.dtype()));
  tf::TensorShape tf_shape;
  if (!tf::TensorShape::BuildTensorShape(spec.shape(), &tf_shape).ok()) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "Unsupported tf::TensorShape: " << spec.shape().DebugString();
  }
  return TensorSpec(spec.name(), dtype, ConvertShape(tf_shape));
}

// A primitive TensorData implementation that wraps the original
// tf::Tensor data.
// NumericTensorDataAdapter gets the ownership of the wrapped tensor, which
// keeps the underlying data alive.
class NumericTensorDataAdapter : public TensorData {
 public:
  explicit NumericTensorDataAdapter(std::unique_ptr<tf::Tensor> tensor)
      : tensor_(std::move(tensor)) {}

  // The source tf::Tensor has the data as one continuous blob.
  size_t byte_size() const override { return tensor_->tensor_data().size(); }
  const void* data() const override { return tensor_->tensor_data().data(); }

 private:
  std::unique_ptr<tf::Tensor> tensor_;
};

// Similar to  NumericTensorDataAdapter but performs additional conversion
// of the original tensor tstring values to string_view while keeping the
// the tstring values owned by the original tensor.
class StringTensorDataAdapter : public TensorData {
 public:
  explicit StringTensorDataAdapter(std::unique_ptr<tf::Tensor> tensor)
      : tensor_(std::move(tensor)), string_views_(tensor_->NumElements()) {
    auto string_values = tensor_->flat<tf::tstring>();
    for (size_t i = 0; i < string_values.size(); ++i) {
      string_views_[i] = string_values(i);
    }
  }

  size_t byte_size() const override {
    return string_views_.size() * sizeof(string_view);
  }
  const void* data() const override { return string_views_.data(); }

 private:
  std::unique_ptr<tf::Tensor> tensor_;
  std::vector<string_view> string_views_;
};

// Conversion of tensor data for numeric data types, which can be
// done by simply wrapping the original tensorflow tensor data.
template <typename t>
std::unique_ptr<TensorData> ConvertTensorData(
    std::unique_ptr<tf::Tensor> tensor) {
  return std::make_unique<NumericTensorDataAdapter>(std::move(tensor));
}

// Specialization of ConvertTensorData for the DT_STRING data type.
template <>
std::unique_ptr<TensorData> ConvertTensorData<string_view>(
    std::unique_ptr<tf::Tensor> tensor) {
  return std::make_unique<StringTensorDataAdapter>(std::move(tensor));
}

StatusOr<Tensor> ConvertTensor(std::unique_ptr<tf::Tensor> tensor) {
  FCP_ASSIGN_OR_RETURN(DataType dtype, ConvertDataType(tensor->dtype()));
  TensorShape shape = ConvertShape(tensor->shape());
  std::unique_ptr<TensorData> data;
  DTYPE_CASES(dtype, T, data = ConvertTensorData<T>(std::move(tensor)));
  return Tensor::Create(dtype, std::move(shape), std::move(data));
}

}  // namespace fcp::aggregation::tensorflow
