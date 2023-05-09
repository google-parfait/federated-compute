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

#include "fcp/aggregation/core/group_by_aggregator.h"

#include <optional>
#include <typeinfo>
#include <utility>
#include <vector>

#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/composite_key_combiner.h"
#include "fcp/aggregation/core/input_tensor_list.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/one_dim_grouping_aggregator.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

namespace {
// Ensures that the provided TensorAggregator is a subclass of
// OneDimGroupingAggregator templated by the expected type, and that
// CheckValid returns true on the OneDimGroupingAggregator.
template <typename T>
Status CheckValidOneDimGroupingAggregator(const TensorAggregator& aggregator) {
  // TODO(team): For the bare metal environment, we will need a version
  // of this class that does not rely on dynamic_cast.
  const OneDimGroupingAggregator<T>* grouping_aggregator =
      dynamic_cast<const OneDimGroupingAggregator<T>*>(&aggregator);
  if (grouping_aggregator == nullptr) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "Expected OneDimGroupingAggregator of type " << typeid(T).name();
  }
  return grouping_aggregator->CheckValid();
}
}  // namespace

GroupByAggregator::GroupByAggregator(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs,
    std::vector<Intrinsic>&& intrinsics)
    : num_inputs_(0),
      num_keys_per_input_(input_key_specs.size()),
      intrinsics_(std::move(intrinsics)),
      output_key_specs_(output_key_specs) {
  // All intrinsics held by a GroupByAggregator must hold some subclass
  // of OneDimGroupingAggregator templated by a type matching the output data
  // type specified in the intrinsic.
  int num_value_inputs = 0;
  for (const Intrinsic& intrinsic : intrinsics_) {
    Status intrinsic_status;
    DTYPE_CASES(intrinsic.outputs()[0].dtype(), T,
                intrinsic_status = CheckValidOneDimGroupingAggregator<T>(
                    intrinsic.const_aggregator()));
    FCP_CHECK(intrinsic_status.ok())
        << "GroupByAggregator: " << intrinsic_status.message();
    num_value_inputs += intrinsic.inputs().size();
  }
  num_tensors_per_input_ = num_keys_per_input_ + num_value_inputs;
  FCP_CHECK(num_tensors_per_input_ > 0)
      << "GroupByAggregator: Must operate on a nonzero number of tensors.";
  // TODO(team): If there are no input keys we should support a columnar
  // aggregation that aggregates all the values in each column and produces a
  // single output value per column. This would be equivalent to having
  // identical key values for all rows.
  FCP_CHECK(num_keys_per_input_ > 0)
      << "GroupByAggregator: Must group by a nonzero number of keys.";
  FCP_CHECK(num_keys_per_input_ == output_key_specs->size())
      << "GroupByAggregator: Size of input_key_specs must match size of "
         "output_key_specs.";
  std::vector<DataType> key_types;
  key_types.reserve(num_keys_per_input_);
  for (int i = 0; i < num_keys_per_input_; ++i) {
    auto& input_spec = input_key_specs[i];
    auto& output_spec = (*output_key_specs_)[i];
    FCP_CHECK(input_spec.dtype() == output_spec.dtype())
        << "GroupByAggregator: Input and output tensor specifications must "
           "have matching data types";
    // TODO(team): Support accumulating value tensors of multiple
    // dimensions. In that case, the size of all output dimensions but one
    // (the dimension corresponding to the number of unique composite keys)
    // should be known in advance and thus this constructor should take in a
    // shape with a single unknown dimension.
    FCP_CHECK(input_spec.shape() == TensorShape{-1} &&
              output_spec.shape() == TensorShape{-1})
        << "All input and output tensors must have one dimension of unknown "
           "size. "
           "TensorShape should be {-1}";
    key_types.push_back(input_spec.dtype());
  }
  key_combiner_.emplace(std::move(key_types));
}

Status GroupByAggregator::MergeWith(TensorAggregator&& other) {
  FCP_RETURN_IF_ERROR(CheckValid());
  GroupByAggregator* other_ptr = dynamic_cast<GroupByAggregator*>(&other);
  if (other_ptr == nullptr) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::MergeOutputTensors: Can only merge with "
              "another GroupByAggregator";
  }
  FCP_RETURN_IF_ERROR((*other_ptr).CheckValid());

  if (!other_ptr->CompatibleKeySpecs(key_combiner_->dtypes(),
                                     *output_key_specs_)) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::MergeWith: Expected other "
              "GroupByAggregator to have the same key input and output specs";
  }
  if (!other_ptr->CompatibleInnerIntrinsics(intrinsics_)) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::MergeWith: Expected other "
              "GroupByAggregator to use compatible inner intrinsics";
  }
  int other_num_inputs = other_ptr->GetNumInputs();
  OutputTensorList other_output_tensors =
      std::move(*other_ptr).TakeOutputsInternal();
  InputTensorList tensors(other_output_tensors.size());
  for (int i = 0; i < other_output_tensors.size(); ++i)
    tensors[i] = &other_output_tensors[i];
  if (tensors.size() != 0) {
    FCP_RETURN_IF_ERROR(AggregateTensorsInternal(std::move(tensors)));
  }
  num_inputs_ += other_num_inputs;
  return FCP_STATUS(OK);
}

Status GroupByAggregator::AggregateTensors(InputTensorList tensors) {
  FCP_RETURN_IF_ERROR(AggregateTensorsInternal(std::move(tensors)));
  num_inputs_++;
  return FCP_STATUS(OK);
}

Status GroupByAggregator::CheckValid() const {
  if (key_combiner_ == std::nullopt) {
    return FCP_STATUS(FAILED_PRECONDITION)
           << "GroupByAggregator::CheckValid: Output has already been "
              "consumed.";
  }
  return FCP_STATUS(OK);
}

OutputTensorList GroupByAggregator::TakeOutputs() && {
  size_t num_keys = num_keys_per_input_;
  // Make a copy of the pointer to the externally-owned OutputKeySpecs since we
  // need to move from *this to call TakeOutputsInternal, and thus cannot access
  // any more class level state.
  const std::vector<TensorSpec>* output_key_specs = output_key_specs_;
  OutputTensorList internal_outputs = std::move(*this).TakeOutputsInternal();
  if (internal_outputs.empty()) return internal_outputs;
  // Keys should only be included in the final outputs if their name is nonempty
  // in the output_key_specs.
  OutputTensorList outputs;
  for (int i = 0; i < num_keys; ++i) {
    if ((*output_key_specs)[i].name().empty()) continue;
    outputs.push_back(std::move(internal_outputs[i]));
  }
  // Include all outputs from sub-intrinsics.
  for (size_t j = num_keys; j < internal_outputs.size(); ++j) {
    outputs.push_back(std::move(internal_outputs[j]));
  }
  return outputs;
}

Status GroupByAggregator::AggregateTensorsInternal(InputTensorList tensors) {
  if (tensors.size() != num_tensors_per_input_) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator should operate on " << num_tensors_per_input_
           << " input tensors";
  }
  // Get the shape of the first key tensor in order to ensure that all the value
  // tensors have the same shape. CompositeKeyCombiner::Accumulate will ensure
  // that all keys have the same shape before making any changes to its own
  // internal state.
  TensorShape key_shape = tensors[0]->shape();
  if (key_shape.dim_sizes().size() > 1) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator: Only scalar or one-dimensional tensors are "
              "supported.";
  }

  // Check all required invariants on the input tensors, so this function can
  // fail before changing the state of this GroupByAggregator if there is an
  // invalid input tensor.
  size_t input_index = num_keys_per_input_;
  for (Intrinsic& intrinsic : intrinsics_) {
    for (const TensorSpec& tensor_spec : intrinsic.inputs()) {
      const Tensor* tensor = tensors[input_index];
      // Ensure the types of the value input tensors match the expected types.
      if (tensor->dtype() != tensor_spec.dtype()) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "Tensor at position " << input_index
               << " did not have expected dtype " << tensor_spec.dtype()
               << " and instead had dtype " << tensor->dtype();
      }
      if (tensor->shape() != key_shape) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "GroupByAggregator: Shape of value tensor at index "
               << input_index << " does not match expected shape.";
      }
      if (!tensor->is_dense()) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "GroupByAggregator: Only dense tensors are supported.";
      }
      ++input_index;
    }
  }

  InputTensorList keys(num_keys_per_input_);
  for (int i = 0; i < num_keys_per_input_; ++i) {
    keys[i] = tensors[i];
  }
  FCP_ASSIGN_OR_RETURN(Tensor ordinals,
                       key_combiner_->Accumulate(std::move(keys)));

  input_index = num_keys_per_input_;
  for (Intrinsic& intrinsic : intrinsics_) {
    InputTensorList intrinsic_inputs(intrinsic.inputs().size() + 1);
    intrinsic_inputs[0] = &ordinals;
    for (int i = 0; i < intrinsic.inputs().size(); ++i) {
      intrinsic_inputs[i + 1] = tensors[input_index++];
    }
    Status accumulate_status =
        intrinsic.aggregator().Accumulate(std::move(intrinsic_inputs));
    // If the accumulate operation fails on a sub-intrinsic, the key_combiner_
    // and any previous sub-intrinsics have already been modified. Thus, exit
    // the program with a CHECK failure rather than a failed status which might
    // leave the GroupByAggregator in an inconsistent state.
    FCP_CHECK(accumulate_status.ok())
        << "GroupByAggregator: " << accumulate_status.message();
  }
  return FCP_STATUS(OK);
}

OutputTensorList GroupByAggregator::TakeOutputsInternal() && {
  OutputTensorList outputs = key_combiner_->GetOutputKeys();
  key_combiner_ = std::nullopt;
  if (num_inputs_ == 0) return outputs;
  outputs.reserve(outputs.size() + intrinsics_.size());
  for (auto& intrinsic : intrinsics_) {
    StatusOr<OutputTensorList> value_output =
        std::move(intrinsic.aggregator()).Report();
    FCP_CHECK(value_output.ok()) << value_output.status().message();
    for (Tensor& output_tensor : value_output.value()) {
      outputs.push_back(std::move(output_tensor));
    }
  }
  return outputs;
}

bool GroupByAggregator::CompatibleInnerIntrinsics(
    const std::vector<Intrinsic>& intrinsics) const {
  if (intrinsics.size() != intrinsics_.size()) return false;
  for (int i = 0; i < intrinsics.size(); ++i) {
    if (intrinsics[i].inputs() != intrinsics_[i].inputs()) return false;
    if (intrinsics[i].outputs() != intrinsics_[i].outputs()) return false;
  }
  return true;
}

bool GroupByAggregator::CompatibleKeySpecs(
    const std::vector<DataType>& input_key_types,
    const std::vector<TensorSpec>& output_key_specs) const {
  return key_combiner_->dtypes() == input_key_types &&
         output_key_specs == *output_key_specs_;
}

}  // namespace aggregation
}  // namespace fcp
