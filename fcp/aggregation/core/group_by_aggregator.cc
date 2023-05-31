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

#include <memory>
#include <optional>
#include <string>
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
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

constexpr char kGroupByUri[] = "fedsql_group_by";

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
    const std::vector<Intrinsic>* intrinsics,
    std::vector<std::unique_ptr<TensorAggregator>> aggregators)
    : num_inputs_(0),
      num_keys_per_input_(input_key_specs.size()),
      intrinsics_(*intrinsics),
      output_key_specs_(*output_key_specs),
      aggregators_(std::move(aggregators)) {
  // All intrinsics held by a GroupByAggregator must hold some subclass
  // of OneDimGroupingAggregator templated by a type matching the output data
  // type specified in the intrinsic.
  int num_value_inputs = 0;
  FCP_CHECK(intrinsics_.size() == aggregators_.size())
      << "Intrinsics and aggregators vectors must be the same size.";
  for (int i = 0; i < intrinsics_.size(); ++i) {
    Status intrinsic_status;
    DTYPE_CASES(intrinsics_[i].outputs[0].dtype(), T,
                intrinsic_status =
                    CheckValidOneDimGroupingAggregator<T>(*aggregators_[i]));
    FCP_CHECK(intrinsic_status.ok())
        << "GroupByAggregator: " << intrinsic_status.message();
    num_value_inputs += intrinsics_[i].inputs.size();
  }
  num_tensors_per_input_ = num_keys_per_input_ + num_value_inputs;
  FCP_CHECK(num_tensors_per_input_ > 0)
      << "GroupByAggregator: Must operate on a nonzero number of tensors.";
  FCP_CHECK(num_keys_per_input_ == output_key_specs->size())
      << "GroupByAggregator: Size of input_key_specs must match size of "
         "output_key_specs.";
  // If there are no input keys, support a columnar aggregation that aggregates
  // all the values in each column and produces a single output value per
  // column. This would be equivalent to having identical key values for all
  // rows.
  if (num_keys_per_input_ == 0) {
    return;
  }
  std::vector<DataType> key_types;
  key_types.reserve(num_keys_per_input_);
  for (int i = 0; i < num_keys_per_input_; ++i) {
    const TensorSpec& input_spec = input_key_specs[i];
    const TensorSpec& output_spec = output_key_specs_[i];
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
  FCP_RETURN_IF_ERROR(other_ptr->IsCompatible(*this));
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
  if (output_consumed_) {
    return FCP_STATUS(FAILED_PRECONDITION)
           << "GroupByAggregator::CheckValid: Output has already been "
              "consumed.";
  }
  return FCP_STATUS(OK);
}

OutputTensorList GroupByAggregator::TakeOutputs() && {
  size_t num_keys = num_keys_per_input_;
  OutputTensorList internal_outputs = std::move(*this).TakeOutputsInternal();
  if (internal_outputs.empty()) return internal_outputs;
  // Keys should only be included in the final outputs if their name is nonempty
  // in the output_key_specs.
  OutputTensorList outputs;
  for (int i = 0; i < num_keys; ++i) {
    if (output_key_specs_[i].name().empty()) continue;
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
  for (const Intrinsic& intrinsic : intrinsics_) {
    for (const TensorSpec& tensor_spec : intrinsic.inputs) {
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

  FCP_ASSIGN_OR_RETURN(Tensor ordinals, CreateOrdinalsByGroupingKeys(tensors));

  input_index = num_keys_per_input_;
  for (int i = 0; i < intrinsics_.size(); ++i) {
    InputTensorList intrinsic_inputs(intrinsics_[i].inputs.size() + 1);
    intrinsic_inputs[0] = &ordinals;
    for (int j = 0; j < intrinsics_[i].inputs.size(); ++j) {
      intrinsic_inputs[j + 1] = tensors[input_index++];
    }
    Status accumulate_status =
        aggregators_[i]->Accumulate(std::move(intrinsic_inputs));
    // If the accumulate operation fails on a sub-intrinsic, the key_combiner_
    // and any previous sub-intrinsics have already been modified. Thus, exit
    // the program with a CHECK failure rather than a failed status which might
    // leave the GroupByAggregator in an inconsistent state.
    FCP_CHECK(accumulate_status.ok())
        << "GroupByAggregator: " << accumulate_status.message();
  }
  return FCP_STATUS(OK);
}

OutputTensorList GroupByAggregator::TakeOutputsInternal() {
  output_consumed_ = true;
  OutputTensorList outputs;
  if (num_inputs_ == 0) return outputs;
  if (key_combiner_.has_value()) {
    outputs = key_combiner_->GetOutputKeys();
  }
  outputs.reserve(outputs.size() + intrinsics_.size());
  for (int i = 0; i < intrinsics_.size(); ++i) {
    auto tensor_aggregator = std::move(aggregators_[i]);
    StatusOr<OutputTensorList> value_output =
        std::move(*tensor_aggregator).Report();
    FCP_CHECK(value_output.ok()) << value_output.status().message();
    for (Tensor& output_tensor : value_output.value()) {
      outputs.push_back(std::move(output_tensor));
    }
  }
  return outputs;
}

StatusOr<Tensor> GroupByAggregator::CreateOrdinalsByGroupingKeys(
    const InputTensorList& inputs) {
  if (key_combiner_.has_value()) {
    InputTensorList keys(num_keys_per_input_);
    for (int i = 0; i < num_keys_per_input_; ++i) {
      keys[i] = inputs[i];
    }
    return key_combiner_->Accumulate(std::move(keys));
  }
  // If there are no keys, we should aggregate all elements in a column into one
  // element, as if there were an imaginary key column with identical values for
  // all rows.
  auto ordinals =
      std::make_unique<MutableVectorData<int64_t>>(inputs[0]->num_elements());
  return Tensor::Create(internal::TypeTraits<int64_t>::kDataType,
                        inputs[0]->shape(), std::move(ordinals));
}

Status GroupByAggregator::IsCompatible(const GroupByAggregator& other) const {
  if (other.key_combiner_.has_value() != key_combiner_.has_value()) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::MergeWith: "
              "Expected other GroupByAggregator to have the same key input and "
              "output specs";
  }
  if (!key_combiner_.has_value()) {
    return FCP_STATUS(OK);
  }
  // The constructor validates that input key types match output key types, so
  // checking that the output key types of both aggregators match is sufficient
  // to verify key compatibility.
  if (other.output_key_specs_ != output_key_specs_) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::MergeWith: "
              "Expected other GroupByAggregator to have the same key input and "
              "output specs";
  }
  if (other.intrinsics_.size() != intrinsics_.size()) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "GroupByAggregator::MergeWith: Expected other "
              "GroupByAggregator to use the same number of inner intrinsics";
  }
  for (int i = 0; i < other.intrinsics_.size(); ++i) {
    const std::vector<Intrinsic>& other_intrinsics = other.intrinsics_;
    if (other_intrinsics[i].inputs != intrinsics_[i].inputs) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupByAggregator::MergeWith: Expected other "
                "GroupByAggregator to use inner intrinsics with the same "
                "inputs.";
    }
    if (other_intrinsics[i].outputs != intrinsics_[i].outputs) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupByAggregator::MergeWith: Expected other "
                "GroupByAggregator to use inner intrinsics with the same "
                "outputs.";
    }
  }
  return FCP_STATUS(OK);
}

StatusOr<std::unique_ptr<TensorAggregator>> GroupByFactory::Create(
    const Intrinsic& intrinsic) const {
  // Check that the configuration is valid for fedsql_group_by.
  if (intrinsic.uri != kGroupByUri) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "GroupByFactory: Expected intrinsic URI " << kGroupByUri
           << " but got uri " << intrinsic.uri;
  }
  if (intrinsic.inputs.size() != intrinsic.outputs.size()) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "GroupByFactory: Exactly the same number of input args and "
              "output tensors are "
              "expected but got "
           << intrinsic.inputs.size() << " inputs vs "
           << intrinsic.outputs.size() << " outputs.";
  }
  if (!intrinsic.parameters.empty()) {
    return FCP_STATUS(INVALID_ARGUMENT)
           << "GroupByFactory: No input parameters expected.";
  }

  for (int i = 0; i < intrinsic.inputs.size(); ++i) {
    const TensorSpec& input_spec = intrinsic.inputs[i];
    const TensorSpec& output_spec = intrinsic.outputs[i];
    if (input_spec.dtype() != output_spec.dtype() ||
        input_spec.shape() != output_spec.shape()) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "Input and output tensors have mismatched specs.";
    }

    if (input_spec.shape() != TensorShape{-1} ||
        output_spec.shape() != TensorShape{-1}) {
      return FCP_STATUS(INVALID_ARGUMENT)
             << "All input and output tensors must have one dimension of "
                "unknown "
                "size. "
                "TensorShape should be {-1}";
    }
  }

  std::vector<std::unique_ptr<TensorAggregator>> nested_aggregators;
  int num_value_inputs = 0;
  constexpr char nested_prefix[] = "GoogleSQL:";
  for (const Intrinsic& nested : intrinsic.nested_intrinsics) {
    // Disable lint checks recommending use of absl::StrContains() here, because
    // we don't want to take a dependency on absl libraries in the aggregation
    // core implementations.
    if (nested.uri.find(nested_prefix) == std::string::npos) {  // NOLINT
      return FCP_STATUS(INVALID_ARGUMENT)
             << "GroupByFactory: Nested intrinsic URIs must start with "
                "'GoogleSQL:'.";
    }
    // Resolve the intrinsic_uri to the registered TensorAggregatorFactory.
    FCP_ASSIGN_OR_RETURN(const TensorAggregatorFactory* factory,
                         GetAggregatorFactory(nested.uri));

    // Use the factory to create the TensorAggregator instance.
    FCP_ASSIGN_OR_RETURN(auto nested_aggregator, factory->Create(nested));
    nested_aggregators.push_back(std::move(nested_aggregator));
    num_value_inputs += nested.inputs.size();
  }
  if (num_value_inputs + intrinsic.inputs.size() == 0) {
    return FCP_STATUS(INVALID_ARGUMENT) << "GroupByFactory: Must operate on a "
                                           "nonzero number of input tensors.";
  }
  return std::make_unique<GroupByAggregator>(
      intrinsic.inputs, &intrinsic.outputs, &intrinsic.nested_intrinsics,
      std::move(nested_aggregators));
}

// TODO(team): Revise the registration mechanism below.
REGISTER_AGGREGATOR_FACTORY(kGroupByUri, GroupByFactory);

}  // namespace aggregation
}  // namespace fcp
