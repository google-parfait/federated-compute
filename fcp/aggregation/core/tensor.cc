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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/base/monitoring.h"

#ifndef FCP_NANOLIBC
#include "fcp/aggregation/core/tensor.pb.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
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
  explicit SerializedContentNumericData(std::string content)
      : content_(std::move(content)) {}
  ~SerializedContentNumericData() override = default;

  // Implementation of TensorData methods.
  size_t byte_size() const override { return content_.size(); }
  const void* data() const override { return content_.data(); }

 private:
  std::string content_;
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

// Specialization of EncodeContent for DT_STRING data type.
template <>
std::string EncodeContent<string_view>(const TensorData* data, size_t num) {
  std::string content;
  google::protobuf::io::StringOutputStream out(&content);
  google::protobuf::io::CodedOutputStream coded_out(&out);
  auto ptr = reinterpret_cast<const string_view*>(data->data());

  // Write all string sizes as Varint64.
  for (size_t i = 0; i < num; ++i) {
    coded_out.WriteVarint64(ptr[i].size());
  }

  // Write all string contents.
  for (size_t i = 0; i < num; ++i) {
    coded_out.WriteRaw(ptr[i].data(), static_cast<int>(ptr[i].size()));
  }

  return content;
}

// Converts the serialized TensorData content stored in TensorProto to an
// instance of TensorData.   The `num` argument is needed in case the number of
// values can't be derived from the content size.
template <typename T>
StatusOr<std::unique_ptr<TensorData>> DecodeContent(std::string content,
                                                    size_t num) {
  // Default decoding of tensor data, valid only for numeric data types.
  return std::make_unique<SerializedContentNumericData>(std::move(content));
}

// Wraps the serialized TensorData content stored and surfaces it as pointer
// string_view values pointing back into the wrapped content. This class is
// be created and initialized from within the DecodeContent<string_view>().
class SerializedContentStringData : public TensorData {
 public:
  SerializedContentStringData() = default;
  ~SerializedContentStringData() override = default;

  // Implementation of TensorData methods.
  size_t byte_size() const override {
    return string_views_.size() * sizeof(string_view);
  }
  const void* data() const override { return string_views_.data(); }

  // Initializes the string_view values to point to the strings embedded in the
  // content.
  Status Initialize(std::string content, size_t num) {
    content_ = std::move(content);
    google::protobuf::io::ArrayInputStream input(content_.data(),
                                       static_cast<int>(content_.size()));
    google::protobuf::io::CodedInputStream coded_input(&input);

    // The pointer to the first string in the content is unknown at this point
    // because there are multiple string sizes at the front, all encoded as
    // VarInts.  To avoid using the extra storage this code reuses the same
    // string_views_ vector in the two passes. First it initializes the data
    // pointers to start with the beginning of the content. Then in the second
    // pass it shifts all data pointers to where strings actually begin in the
    // content.
    string_views_.resize(num);
    size_t cumulative_size = 0;

    // The first pass reads the string sizes;
    for (size_t i = 0; i < num; ++i) {
      size_t size;
      if (!coded_input.ReadVarint64(&size)) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "Expected to read " << num
               << " string values but the input tensor content doesn't contain "
                  "a size for the "
               << i << "th string. The content size is " << content_.size()
               << " bytes.";
      }
      string_views_[i] = string_view(content_.data() + cumulative_size, size);
      cumulative_size += size;
    }

    // The current position in the input stream after reading all the string
    // sizes. The input stream must be at the beginning of the first string now.
    size_t offset = coded_input.CurrentPosition();

    // Verify that the content is large enough.
    if (content_.size() < offset + cumulative_size) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "Input tensor content has insufficient size to store " << num
             << " string values. The content size is " << content_.size()
             << " bytes, but " << offset + cumulative_size
             << " bytes are required.";
    }

    // The second pass offsets string_view pointers so that the first one points
    // to the first string embedded in the content, then all others are shifted
    // by the same offset to point to subsequent strings.
    for (size_t i = 0; i < num; ++i) {
      string_views_[i] = string_view(string_views_[i].data() + offset,
                                     string_views_[i].size());
    }

    return FCP_STATUS(OK);
  }

 private:
  std::string content_;
  std::vector<string_view> string_views_;
};

template <>
StatusOr<std::unique_ptr<TensorData>> DecodeContent<string_view>(
    std::string content, size_t num) {
  auto tensor_data = std::make_unique<SerializedContentStringData>();
  FCP_RETURN_IF_ERROR(tensor_data->Initialize(std::move(content), num));
  return tensor_data;
}

StatusOr<Tensor> Tensor::FromProto(const TensorProto& tensor_proto) {
  FCP_ASSIGN_OR_RETURN(TensorShape shape,
                       TensorShape::FromProto(tensor_proto.shape()));
  // TODO(team): The num_values is valid only for dense tensors.
  size_t num_values = shape.NumElements();
  StatusOr<std::unique_ptr<TensorData>> data;
  DTYPE_CASES(tensor_proto.dtype(), T,
              data = DecodeContent<T>(tensor_proto.content(), num_values));
  FCP_RETURN_IF_ERROR(data);
  return Create(tensor_proto.dtype(), std::move(shape),
                std::move(data).value());
}

StatusOr<Tensor> Tensor::FromProto(TensorProto&& tensor_proto) {
  FCP_ASSIGN_OR_RETURN(TensorShape shape,
                       TensorShape::FromProto(tensor_proto.shape()));
  // TODO(team): The num_values is valid only for dense tensors.
  size_t num_values = shape.NumElements();
  std::string content = std::move(*tensor_proto.mutable_content());
  StatusOr<std::unique_ptr<TensorData>> data;
  DTYPE_CASES(tensor_proto.dtype(), T,
              data = DecodeContent<T>(std::move(content), num_values));
  FCP_RETURN_IF_ERROR(data);
  return Create(tensor_proto.dtype(), std::move(shape),
                std::move(data).value());
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
