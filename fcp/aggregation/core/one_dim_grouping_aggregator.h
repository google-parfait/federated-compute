/*
 * Copyright 2023 Google LLC
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
#ifndef FCP_AGGREGATION_CORE_ONE_DIM_GROUPING_AGGREGATOR_H_
#define FCP_AGGREGATION_CORE_ONE_DIM_GROUPING_AGGREGATOR_H_

#include <climits>
#include <cstddef>
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
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// OneDimGroupingAggregator class is a specialization of TensorAggregator which
// takes in a tensor containing ordinals and a tensor containing values, and
// accumulates the values into the output positions indicated by the
// corresponding ordinals.
//
// Currently only 1D input tensors are supported.
//
// The specific means of accumulating values and producing default values are
// left to the subclass.
//
// The implementation operates on AggVector<T> instances rather than tensors.
//
// This class is not thread safe.
template <typename InputT, typename OutputT = InputT>
class OneDimGroupingAggregator : public TensorAggregator {
 public:
  // TODO(team): Support accumulating tensors of multiple dimensions. In
  // that case, the size of all dimensions but one (the dimension corresponding
  // to the ordinal tensor) should be known in advance and thus this constructor
  // should take in a shape with a single unknown dimension.
  OneDimGroupingAggregator()
      : data_vector_(std::make_unique<MutableVectorData<OutputT>>()),
        num_inputs_(0) {}

  Status MergeWith(TensorAggregator&& other) override {
    FCP_RETURN_IF_ERROR(CheckValid());
    OneDimGroupingAggregator<InputT, OutputT>* other_ptr =
        dynamic_cast<OneDimGroupingAggregator<InputT, OutputT>*>(&other);
    if (other_ptr == nullptr) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::MergeOutputTensors: Can only merge "
                "with another GroupingAggregator with the same input data type "
             << internal::TypeTraits<InputT>::kDataType
             << " producing the same dtype "
             << internal::TypeTraits<OutputT>::kDataType;
    }
    FCP_RETURN_IF_ERROR((*other_ptr).CheckValid());
    int other_num_inputs = other.GetNumInputs();
    OutputTensorList output_tensors = std::move(*other_ptr).TakeOutputs();

    if (output_tensors.size() == 1) {
      AggVector<OutputT> other_data_vector =
          output_tensors[0].AsAggVector<OutputT>();
      if (other_data_vector.size() > data_vector_->size()) {
        data_vector_->resize(other_data_vector.size(), GetDefaultValue());
      }
      AggregateVector(other_data_vector);
    } else {
      // An empty output is valid and merging it into the current
      // GroupingAggregator is a no-op.
      FCP_CHECK(output_tensors.empty())
          << "OneDimGroupingAggregator::MergeOutputTensors: GroupingAggregator "
             "should produce at most a single output tensor.";
    }

    num_inputs_ += other_num_inputs;
    return FCP_STATUS(OK);
  }

  int GetNumInputs() const override { return num_inputs_; }

  Status CheckValid() const override {
    if (data_vector_ == nullptr) {
      return FCP_STATUS(FAILED_PRECONDITION)
             << "OneDimGroupingAggregator::CheckValid: Output has already been "
                "consumed.";
    }
    return FCP_STATUS(OK);
  }

 protected:
  // Provides mutable access to the aggregator data as a vector<T>
  inline std::vector<OutputT>& data() { return *data_vector_; }

  // Implementation of the tensor aggregation.
  // Expects 2 tensors as input: a tensor containing ordinals and a tensor
  // containing values.
  //
  // Accumulates the values into the positions in the output tensor which are
  // indicated by the corresponding ordinals.
  Status AggregateTensors(InputTensorList tensors) override {
    FCP_CHECK(tensors.size() == 2)
        << "OneDimGroupingAggregator should operate on 2 input tensors";

    const Tensor* ordinals = tensors[0];
    if (ordinals->dtype() != DT_INT64) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::AggregateTensors: dtype mismatch "
                "for tensor 0. Expected DT_INT64.";
    }
    const Tensor* tensor = tensors[1];
    if (tensor->dtype() != internal::TypeTraits<InputT>::kDataType) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::AggregateTensors: dtype mismatch "
                "for tensor 1";
    }
    if (ordinals->shape() != tensor->shape()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::AggregateTensors: tensor shape "
                "mismatch. Shape of both tensors must be the same.";
    }
    int num_dimensions = tensor->shape().dim_sizes().size();
    if (num_dimensions > 1) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::AggregateTensors: Only 1 "
                "dimensional tensors supported. Input tensor has "
             << num_dimensions << " dimensions.";
    }
    if (!ordinals->is_dense() || !tensor->is_dense()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "OneDimGroupingAggregator::AggregateTensors: Only dense "
                "tensors are supported.";
    }
    num_inputs_++;
    AggVector<InputT> value_vector = tensor->AsAggVector<InputT>();
    AggVector<int64_t> ordinals_vector = ordinals->AsAggVector<int64_t>();
    size_t final_size = data_vector_->size();
    for (auto o : ordinals_vector) {
      if (o.value >= final_size) {
        final_size = o.value + 1;
      }
    }
    // Resize once outside the loop to avoid quadratic behavior.
    data_vector_->resize(final_size, GetDefaultValue());
    AggregateVectorByOrdinals(ordinals_vector, value_vector);
    return FCP_STATUS(OK);
  }

  OutputTensorList TakeOutputs() && override {
    OutputTensorList outputs = std::vector<Tensor>();
    size_t outputs_size = data_vector_->size();
    FCP_CHECK(outputs_size <= LONG_MAX)
        << "TensorShape: Dimension size too large to be represented as a "
           "signed long.";
    outputs.push_back(
        Tensor::Create(internal::TypeTraits<OutputT>::kDataType,
                       TensorShape{static_cast<int64_t>(outputs_size)},
                       std::move(data_vector_))
            .value());
    data_vector_ = nullptr;
    return outputs;
  }

  // Delegates AggVector aggregation by ordinal to a derived class.
  //
  // The size of the vector returned by data() must be greater than the largest
  // ordinal in this vector.
  //
  // To avoid making a virtual function call per value in the tensor, the whole
  // vector is passed to the subclass for aggregation, which provides better
  // performance but comes at the cost of duplicated code between subclasses for
  // iterating over the vectors.
  virtual void AggregateVectorByOrdinals(
      const AggVector<int64_t>& ordinals_vector,
      const AggVector<InputT>& value_vector) = 0;

  // Delegates AggVector aggregation to a derived class.
  //
  // This vector must be the same size as the vector returned by data().
  //
  // To avoid making a virtual function call per value in the tensor, the whole
  // vector is passed to the subclass for aggregation, which provides better
  // performance but comes at the cost of duplicated code between subclasses for
  // iterating over the vectors.
  virtual void AggregateVector(const AggVector<OutputT>& agg_vector) = 0;

  // Delegates initialization of previously unseen ordinals to a derived class.
  virtual OutputT GetDefaultValue() = 0;

 private:
  std::unique_ptr<MutableVectorData<OutputT>> data_vector_;
  int num_inputs_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_ONE_DIM_GROUPING_AGGREGATOR_H_
