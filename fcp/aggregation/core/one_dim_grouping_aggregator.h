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
#include "fcp/aggregation/core/tensor_data.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// GroupingAggregator class is a specialization of TensorAggregator which
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
template <typename T>
class OneDimGroupingAggregator : public TensorAggregator {
 public:
  // TODO(team): Support accumulating tensors of multiple dimensions. In
  // that case, the size of all dimensions but one (the dimension corresponding
  // to the ordinal tensor) should be known in advance and thus this constructor
  // should take in a shape with a single unknown dimension.
  explicit OneDimGroupingAggregator(DataType dtype)
      : data_vector_(std::make_unique<MutableVectorData<T>>()), num_inputs_(0) {
    FCP_CHECK(internal::TypeTraits<T>::kDataType == dtype)
        << "Incompatible dtype";
  }

  Status MergeWith(TensorAggregator&& other) override {
    FCP_RETURN_IF_ERROR(CheckValid());
    OneDimGroupingAggregator<T>* other_ptr =
        dynamic_cast<OneDimGroupingAggregator<T>*>(&other);
    if (other_ptr == nullptr) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingAggregator::MergeOutputTensors: Can only merge with "
                "another GroupingAggregator operating on the same dtype "
             << internal::TypeTraits<T>::kDataType;
    }
    FCP_RETURN_IF_ERROR((*other_ptr).CheckValid());
    int other_num_inputs = other.GetNumInputs();
    OutputTensorList output_tensors = std::move(*other_ptr).TakeOutputs();

    if (output_tensors.size() == 1) {
      AggVector<T> other_data_vector = output_tensors[0].AsAggVector<T>();
      if (other_data_vector.size() > data_vector_->size()) {
        data_vector_->resize(other_data_vector.size(), GetDefaultValue());
      }
      AggregateVector(other_data_vector);
    } else {
      // An empty output is valid and merging it into the current
      // GroupingAggregator is a no-op.
      FCP_CHECK(output_tensors.empty())
          << "GroupingAggregator::MergeOutputTensors: GroupingAggregator "
             "should produce at most a single output tensor.";
    }

    num_inputs_ += other_num_inputs;
    return FCP_STATUS(OK);
  }

  int GetNumInputs() const override { return num_inputs_; }

 protected:
  // Provides mutable access to the aggregator data as a vector<T>
  inline std::vector<T>& data() { return *data_vector_; }

  // Implementation of the tensor aggregation.
  // Expects 2 tensors as input: a tensor containing ordinals and a tensor
  // containing values.
  //
  // Accumulates the values into the positions in the output tensor which are
  // indicated by the corresponding ordinals.
  Status AggregateTensors(InputTensorList tensors) override {
    FCP_CHECK(tensors.size() == 2)
        << "GroupingAggregator should operate on 2 input tensors";

    const Tensor* ordinals = tensors[0];
    if (ordinals->dtype() != DT_INT64) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingAggregator::AggregateTensors: dtype mismatch for "
                "tensor 0. Expected DT_INT64.";
    }
    const Tensor* tensor = tensors[1];
    if (tensor->dtype() != internal::TypeTraits<T>::kDataType) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingAggregator::AggregateTensors: dtype mismatch for "
                "tensor 1";
    }
    if (ordinals->shape() != tensor->shape()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingAggregator::AggregateTensors: tensor shape mismatch. "
                "Shape of both tensors must be the same.";
    }
    int num_dimensions = tensor->shape().dim_sizes().size();
    if (num_dimensions > 1) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingAggregator::AggregateTensors: Only 1 dimensional "
                "tensors supported. Input tensor has "
             << num_dimensions << " dimensions.";
    }
    if (!ordinals->is_dense() || !tensor->is_dense()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupingAggregator::AggregateTensors: Only dense tensors are "
                "supported.";
    }
    num_inputs_++;
    AggVector<T> value_vector = tensor->AsAggVector<T>();
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

  Status CheckValid() const override {
    if (data_vector_ == nullptr) {
      return FCP_STATUS(FAILED_PRECONDITION)
             << "GroupingAggregator::CheckValid: Output has already been "
                "consumed.";
    }
    return FCP_STATUS(OK);
  }

  OutputTensorList TakeOutputs() && override {
    OutputTensorList outputs = std::vector<Tensor>();
    if (!data_vector_->empty()) {
      outputs.push_back(Tensor::Create(internal::TypeTraits<T>::kDataType,
                                       TensorShape{data_vector_->size()},
                                       std::move(data_vector_))
                            .value());
    }
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
      const AggVector<T>& value_vector) = 0;

  // Delegates AggVector aggregation to a derived class.
  //
  // This vector must be the same size as the vector returned by data().
  //
  // To avoid making a virtual function call per value in the tensor, the whole
  // vector is passed to the subclass for aggregation, which provides better
  // performance but comes at the cost of duplicated code between subclasses for
  // iterating over the vectors.
  virtual void AggregateVector(const AggVector<T>& agg_vector) = 0;

  // Delegates initialization of previously unseen ordinals to a derived class.
  virtual T GetDefaultValue() = 0;

 private:
  std::unique_ptr<MutableVectorData<T>> data_vector_;
  int num_inputs_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_ONE_DIM_GROUPING_AGGREGATOR_H_
