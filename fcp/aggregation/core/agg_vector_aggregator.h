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

#ifndef FCP_AGGREGATION_CORE_AGG_VECTOR_AGGREGATOR_H_
#define FCP_AGGREGATION_CORE_AGG_VECTOR_AGGREGATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fcp/aggregation/core/agg_core.pb.h"
#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/input_tensor_list.h"
#include "fcp/aggregation/core/mutable_vector_data.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// AggVectorAggregator class is a specialization of TensorAggregator which
// operates on AggVector<T> instances rather than tensors.
template <typename T>
class AggVectorAggregator : public TensorAggregator {
 public:
  AggVectorAggregator(DataType dtype, TensorShape shape)
      : AggVectorAggregator(dtype, shape, CreateData(shape), 0) {}

  AggVectorAggregator(DataType dtype, TensorShape shape,
                      std::unique_ptr<MutableVectorData<T>> data,
                      int num_inputs)
      : dtype_(dtype),
        shape_(std::move(shape)),
        data_vector_(std::move(data)),
        num_inputs_(num_inputs) {
    FCP_CHECK(internal::TypeTraits<T>::kDataType == dtype)
        << "Incompatible dtype";
  }

  // Provides mutable access to the aggregator data as a vector<T>
  inline std::vector<T>& data() { return *data_vector_; }

  int GetNumInputs() const override { return num_inputs_; }

  Status MergeWith(TensorAggregator&& other) override {
    FCP_RETURN_IF_ERROR(CheckValid());
    FCP_ASSIGN_OR_RETURN(AggVectorAggregator<T> * other_ptr, CastOther(other));
    FCP_RETURN_IF_ERROR((*other_ptr).CheckValid());
    int64_t other_num_inputs = other.GetNumInputs();
    OutputTensorList output_tensors = std::move(*other_ptr).TakeOutputs();
    FCP_CHECK(output_tensors.size() == 1)
        << "AggVectorAggregator::MergeOutputTensors: AggVectorAggregator "
           "should produce a single output tensor";
    const Tensor& output = output_tensors[0];
    if (output.shape() != shape_) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "AggVectorAggregator::MergeOutputTensors: tensor shape "
                "mismatch";
    }
    // Delegate the actual aggregation to the specific aggregation
    // intrinsic implementation.
    AggregateVector(output.AsAggVector<T>());
    num_inputs_ += other_num_inputs;
    return FCP_STATUS(OK);
  }

  StatusOr<std::string> Serialize() && override {
    AggVectorAggregatorState aggregator_state;
    aggregator_state.set_num_inputs(num_inputs_);
    *(aggregator_state.mutable_vector_data()) = data_vector_->EncodeContent();
    return aggregator_state.SerializeAsString();
  }

 protected:
  // Implementation of the tensor aggregation.
  Status AggregateTensors(InputTensorList tensors) override {
    FCP_CHECK(tensors.size() == 1)
        << "AggVectorAggregator should operate on a single input tensor";

    const Tensor* tensor = tensors[0];
    if (tensor->dtype() != internal::TypeTraits<T>::kDataType) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "AggVectorAggregator::AggregateTensors: dtype mismatch";
    }
    if (tensor->shape() != shape_) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "AggVectorAggregator::AggregateTensors: tensor shape mismatch";
    }
    // Delegate the actual aggregation to the specific aggregation
    // intrinsic implementation.
    AggregateVector(tensor->AsAggVector<T>());
    num_inputs_++;
    return FCP_STATUS(OK);
  }

  Status CheckValid() const override {
    if (data_vector_ == nullptr) {
      return FCP_STATUS(FAILED_PRECONDITION)
             << "AggVectorAggregator::CheckValid: Output has already been "
             << "consumed.";
    }
    return FCP_STATUS(OK);
  }

  OutputTensorList TakeOutputs() && override {
    OutputTensorList outputs = std::vector<Tensor>();
    outputs.push_back(
        Tensor::Create(dtype_, shape_, std::move(data_vector_)).value());
    return outputs;
  }

  // Delegates AggVector aggregation to a derived class.
  virtual void AggregateVector(const AggVector<T>& agg_vector) = 0;

 private:
  static std::unique_ptr<MutableVectorData<T>> CreateData(
      const TensorShape& shape) {
    StatusOr<size_t> num_elements = shape.NumElements();
    FCP_CHECK(num_elements.ok()) << "AggVectorAggregator: All dimensions of "
                                    "tensor shape must be known in advance.";
    return std::make_unique<MutableVectorData<T>>(num_elements.value());
  }

  StatusOr<AggVectorAggregator<T>*> CastOther(TensorAggregator& other) {
    AggVectorAggregator<T>* other_ptr =
        dynamic_cast<AggVectorAggregator<T>*>(&other);
    if (other_ptr == nullptr) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "AggVectorAggregator::MergeOutputTensors: Can only merge with"
             << "another AggVectorAggregator operating on the same dtype "
             << internal::TypeTraits<T>::kDataType;
    }
    return other_ptr;
  }

  const DataType dtype_;
  const TensorShape shape_;
  std::unique_ptr<MutableVectorData<T>> data_vector_;
  int num_inputs_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_AGG_VECTOR_AGGREGATOR_H_
