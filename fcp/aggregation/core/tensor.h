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

#ifndef FCP_AGGREGATION_CORE_TENSOR_H_
#define FCP_AGGREGATION_CORE_TENSOR_H_

#include <memory>
#include <utility>

#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor_data.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/base/monitoring.h"

#ifndef FCP_NANOLIBC
#include "fcp/aggregation/core/tensor.pb.h"
#endif

namespace fcp {
namespace aggregation {

// Tensor class is a container that packages the tensor data with the tensor
// metadata such as the value type and the shape.
//
// For the most part, the aggregation code won't be consuming tensors directly.
// Instead the aggregation code will be working with AggVector instances that
// represent the tensor data in a flattened way.
class Tensor final {
 public:
  // Tensor class isn't copyable.
  Tensor(const Tensor&) = delete;

  // Move constructor.
  Tensor(Tensor&& other)
      : dtype_(other.dtype_),
        shape_(std::move(other.shape_)),
        data_(std::move(other.data_)) {
    other.dtype_ = DT_INVALID;
  }

  // Move assignment.
  Tensor& operator=(Tensor&& other) {
    dtype_ = other.dtype_;
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
    other.dtype_ = DT_INVALID;
    return *this;
  }

  // Define a default constructor to allow for initalization of array
  // to enable creation of a vector of Tensors.
  // A tensor created with the default constructor is not valid and thus should
  // not actually be used.
  Tensor() : dtype_(DT_INVALID), shape_{}, data_(nullptr) {}

  // Validates parameters and creates a Tensor instance.
  static StatusOr<Tensor> Create(DataType dtype, TensorShape shape,
                                 std::unique_ptr<TensorData> data);

#ifndef FCP_NANOLIBC
  // Creates a Tensor instance from a TensorProto.
  static StatusOr<Tensor> FromProto(const TensorProto& tensor_proto);

  // Creates a Tensor instance from a TensorProto, consuming the proto.
  static StatusOr<Tensor> FromProto(TensorProto&& tensor_proto);

  // Converts Tensor to TensorProto
  TensorProto ToProto() const;
#endif  // FCP_NANOLIBC

  // Validates the tensor.
  Status CheckValid() const;

  // Gets the tensor value type.
  DataType dtype() const { return dtype_; }

  // Gets the tensor shape.
  const TensorShape& shape() const { return shape_; }

  // Readonly access to the tensor data.
  const TensorData& data() const { return *data_; }

  // Returns true is the current tensor data is dense.
  // TODO(team): Implement sparse tensors.
  bool is_dense() const { return true; }

  // Provides access to the tensor data via a strongly typed AggVector.
  template <typename T>
  AggVector<T> AsAggVector() const {
    FCP_CHECK(internal::TypeTraits<T>::kDataType == dtype_)
        << "Incompatible tensor dtype()";
    return AggVector<T>(data_.get());
  }

  // TODO(team): Add serialization functions.

 private:
  Tensor(DataType dtype, TensorShape shape, std::unique_ptr<TensorData> data)
      : dtype_(dtype), shape_(std::move(shape)), data_(std::move(data)) {}

  // Tensor data type.
  DataType dtype_;
  // Tensor shape.
  TensorShape shape_;
  // The underlying tensor data.
  std::unique_ptr<TensorData> data_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_TENSOR_H_
