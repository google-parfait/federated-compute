// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fcp/client/converters.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace fcp::client {

using ::absl::StatusOr;
using ::absl::string_view;
using tensorflow_federated::aggregation::Tensor;
using tensorflow_federated::aggregation::TensorData;
using tensorflow_federated::aggregation::TensorShape;

// Similar to NumericTensorDataAdapter but performs additional conversion
// of the original value to string_view while keeping the reference to the
// original.
class StringTensorDataAdapter : public TensorData {
 public:
  explicit StringTensorDataAdapter(
      const ::google::protobuf::RepeatedPtrField<std::string>& value)
      : value_(value), string_views_(value_.begin(), value_.end()) {}

  size_t byte_size() const override {
    return string_views_.size() * sizeof(string_view);
  }
  const void* data() const override { return string_views_.data(); }

 private:
  const ::google::protobuf::RepeatedPtrField<std::string>& value_;
  std::vector<string_view> string_views_;
};

StatusOr<Tensor> ConvertStringTensor(
    TensorShape tensor_shape,
    const ::google::protobuf::RepeatedPtrField<std::string>& value) {
  return Tensor::Create(tensorflow_federated::aggregation::DT_STRING,
                        tensor_shape,
                        std::make_unique<StringTensorDataAdapter>(value));
}

}  // namespace fcp::client
