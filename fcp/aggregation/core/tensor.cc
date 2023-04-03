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

#include "fcp/aggregation/core/tensor.h"

#include <memory>
#include <string>
#include <utility>

#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/base/monitoring.h"

#ifndef FCP_NANOLIBC
#include "fcp/aggregation/core/tensor.pb.h"
#endif

namespace fcp {
namespace aggregation {

Status Tensor::CheckValid() const {
  if (dtype_ == DT_INVALID) {
    return FCP_STATUS(FAILED_PRECONDITION) << "Invalid Tensor dtype.";
  }

  size_t value_size = 0;
  DTYPE_CASES(dtype_, T, value_size = sizeof(T));

  // Verify that the storage is consistent with the value size in terms of
  // size and alignment.
  FCP_RETURN_IF_ERROR(data_->CheckValid(value_size));

  // Verify that the total size of the data is consistent with the value type
  // and the shape.
  // TODO(team): Implement sparse tensors.
  if (data_->byte_size() != shape_.NumElements() * value_size) {
    return FCP_STATUS(FAILED_PRECONDITION)
           << "TensorData byte_size is inconsistent with the Tensor dtype and "
              "shape.";
  }

  return FCP_STATUS(OK);
}

StatusOr<Tensor> Tensor::Create(DataType dtype, TensorShape shape,
                                std::unique_ptr<TensorData> data) {
  Tensor tensor(dtype, std::move(shape), std::move(data));
  FCP_RETURN_IF_ERROR(tensor.CheckValid());
  return std::move(tensor);
}

#ifndef FCP_NANOLIBC

// SerializedContentNumericData implements TensorData by wrapping the serialized
// content string and using it directly as a backing storage. This relies on the
// fact that the serialized content uses the same layout as in memory
// representation if we assume that this code runs on a little-endian system.
// TODO(team): Ensure little-endianness.
class SerializedContentNumericData : public TensorData {
 public:
  explicit SerializedContentNumericData(std::string str)
      : str_(std::move(str)) {}
  ~SerializedContentNumericData() override = default;

  // Implementation of TensorData methods.
  size_t byte_size() const override { return str_.size(); }
  const void* data() const override { return str_.data(); }

 private:
  std::string str_;
};

// Converts the tensor data to a serialized blob saved as the content field
// in the TensorProto.  The `num` argument is needed in case the number of
// values can't be derived from the TensorData size.
template <typename T>
std::string EncodeContent(const TensorData* data, size_t num) {
  // Default encoding of tensor data, valid only for numeric data types.
  return std::string(reinterpret_cast<const char*>(data->data()),
                     data->byte_size());
}

// Converts the serialized TensorData content stored in TensorProto to an
// instance of TensorData.   The `num` argument is needed in case the number of
// values can't be derived from the content size.
template <typename T>
std::unique_ptr<TensorData> DecodeContent(std::string content, size_t num) {
  // Default decoding of tensor data, valid only for numeric data types.
  return std::make_unique<SerializedContentNumericData>(std::move(content));
}

StatusOr<Tensor> Tensor::FromProto(const TensorProto& tensor_proto) {
  FCP_ASSIGN_OR_RETURN(TensorShape shape,
                       TensorShape::FromProto(tensor_proto.shape()));
  // TODO(team): The num_values is valid only for dense tensors.
  size_t num_values = shape.NumElements();
  std::unique_ptr<TensorData> data;
  DTYPE_CASES(tensor_proto.dtype(), T,
              data = DecodeContent<T>(tensor_proto.content(), num_values));
  return Create(tensor_proto.dtype(), std::move(shape), std::move(data));
}

StatusOr<Tensor> Tensor::FromProto(TensorProto&& tensor_proto) {
  FCP_ASSIGN_OR_RETURN(TensorShape shape,
                       TensorShape::FromProto(tensor_proto.shape()));
  // TODO(team): The num_values is valid only for dense tensors.
  size_t num_values = shape.NumElements();
  std::string content = std::move(*tensor_proto.mutable_content());
  std::unique_ptr<TensorData> data;
  DTYPE_CASES(tensor_proto.dtype(), T,
              data = DecodeContent<T>(std::move(content), num_values));
  return Create(tensor_proto.dtype(), std::move(shape), std::move(data));
}

TensorProto Tensor::ToProto() const {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(dtype_);
  *(tensor_proto.mutable_shape()) = shape_.ToProto();
  // TODO(team): The num_values is valid only for dense tensors.
  size_t num_values = shape_.NumElements();
  std::string content;
  DTYPE_CASES(dtype_, T, content = EncodeContent<T>(data_.get(), num_values));
  *(tensor_proto.mutable_content()) = std::move(content);
  return tensor_proto;
}

#endif  // FCP_NANOLIBC

}  // namespace aggregation
}  // namespace fcp
