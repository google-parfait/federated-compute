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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/input_tensor_list.h"
#include "fcp/aggregation/core/mutable_vector_data.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_data.h"
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
      : AggVectorAggregator(dtype, shape,
                            new MutableVectorData<T>(shape.NumElements())) {}

  // Provides mutable access to the aggregator data as a vector<T>
  inline std::vector<T>& data() { return data_vector_; }

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
    if (output.shape() != result_tensor_.shape()) {
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
    if (tensor->shape() != result_tensor_.shape()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "AggVectorAggregator::AggregateTensors: tensor shape mismatch";
    }
    // Delegate the actual aggregation to the specific aggregation
    // intrinsic implementation.
    AggregateVector(tensor->AsAggVector<T>());
    num_inputs_++;
    return FCP_STATUS(OK);
  }

  Status CheckValid() const override { return result_tensor_.CheckValid(); }

  OutputTensorList TakeOutputs() && override {
    OutputTensorList outputs = std::vector<Tensor>();
    outputs.push_back(std::move(result_tensor_));
    return outputs;
  }

  // Delegates AggVector aggregation to a derived class.
  virtual void AggregateVector(const AggVector<T>& agg_vector) = 0;

 private:
  AggVectorAggregator(DataType dtype, TensorShape shape,
                      MutableVectorData<T>* data)
      : result_tensor_(
            Tensor::Create(dtype, shape, std::unique_ptr<TensorData>(data))
                .value()),
        data_vector_(*data),
        num_inputs_(0) {
    FCP_CHECK(internal::TypeTraits<T>::kDataType == dtype)
        << "Incompatible dtype";
  }

  StatusOr<AggVectorAggregator<T>*> CastOther(TensorAggregator& other) {
#ifndef FCP_NANOLIBC
    AggVectorAggregator<T>* other_ptr =
        dynamic_cast<AggVectorAggregator<T>*>(&other);
    if (other_ptr == nullptr) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "AggVectorAggregator::MergeOutputTensors: Can only merge with"
             << "another AggVectorAggregator operating on the same dtype "
             << internal::TypeTraits<T>::kDataType;
    }
    return other_ptr;
#else /* FCP_NANOLIBC */
    // When compiling in nanolibc we do not have access to runtime type
    // information or std::type_traits. Thus we cannot use dynamic cast and use
    // static_cast instead.
    // This means we are relying on the caller to always call the MergeWith
    // method on two TensorAggregators of the same underlying type, or the
    // program will have undefined behavior due to a static_cast to the wrong
    // type.
    return static_cast<AggVectorAggregator<T>*>(&other);
#endif
  }

  Tensor result_tensor_;
  std::vector<T>& data_vector_;
  int num_inputs_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_AGG_VECTOR_AGGREGATOR_H_
