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

#ifndef FCP_AGGREGATION_CORE_GROUP_BY_AGGREGATOR_H_
#define FCP_AGGREGATION_CORE_GROUP_BY_AGGREGATOR_H_

#include <optional>
#include <vector>

#include "fcp/aggregation/core/composite_key_combiner.h"
#include "fcp/aggregation/core/input_tensor_list.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// GroupByAggregator class is a specialization of TensorAggregator which
// takes in a predefined number of tensors to be used as keys and and predefined
// number of tensors to be used as values. It computes the unique combined keys
// across all tensors and then accumulates the values into the output positions
// matching those of the corresponding keys.
//
// Currently only 1D input tensors are supported.
//
// The specific means of accumulating values are delegated to an inner
// OneDimGroupingAggregator intrinsic for each value tensor that should be
// grouped.
//
// When Report is called, this TensorAggregator outputs all key types for which
// the output_key_specs have a nonempty tensor name, as well as all value
// tensors output by the OneDimGroupingAggregator intrinsics.
//
// This class is not thread safe.
class GroupByAggregator : public TensorAggregator {
 public:
  // Constructs a GroupByAggregator from the following inputs:
  //
  // input_key_specs: A vector of TensorSpecs for the tensors that this
  // GroupByAggregator should treat as keys in the input. The first n tensors in
  // any InputTensorList provided to an Accumulate call are expected to match
  // the n TensorSpecs in this vector. For now the shape of each tensor should
  // be {-1} as only one-dimensional aggregations are supported and different
  // calls to Accumulate may have different numbers of examples in each tensor.
  //
  // output_key_specs: A vector of TensorSpecs providing this GroupByAggregator
  // with information on which key tensors should be included in the output.
  // An empty string for the tensor name indicates that the tensor should not
  // be included in the output.
  // Regardless of the output_key_specs, all key tensors listed in the
  // input_key_specs will be used for grouping.
  // output_key_specs must have the same number of TensorSpecs as
  // input_key_specs, and all TensorSpec attributes but the tensor names must
  // match those in input_key_specs. The lifetime of output_key_specs must
  // outlast this class.
  //
  // intrinsics: A vector of Intrinsic classes that should contain
  // subclasses of OneDimGroupingAggregator to which this class will delegate
  // grouping of values.
  // The number of tensors in each InputTensorList provided to Accumulate must
  // match the number of TensorSpecs in input_key_specs plus the number of
  // Intrinsics in this vector.
  // This class takes ownership of the intrinsics vector.
  explicit GroupByAggregator(const std::vector<TensorSpec>& input_key_specs,
                             const std::vector<TensorSpec>* output_key_specs,
                             std::vector<Intrinsic>&& intrinsics);

  // Merge this GroupByAggregator with another GroupByAggregator that operates
  // on compatible types using compatible inner intrinsics.
  Status MergeWith(TensorAggregator&& other) override;

  // Returns the number of inputs that have been accumulated or merged into this
  // GroupByAggregator.
  int GetNumInputs() const override { return num_inputs_; }

 protected:
  // Perform aggregation of the tensors in a single InputTensorList into the
  // state of this GroupByAggregator and increment the count of aggregated
  // tensors.
  //
  // The order in which the tensors must appear in the input is the following:
  // first, the key tensors in the order they appear in the input_tensor_specs,
  // and next, the value tensors in the order the inner intrinsics that
  // aggregate each value appear in the intrinsics input vector.
  Status AggregateTensors(InputTensorList tensors) override;

  // Ensures that the output has not yet been consumed for this
  // GroupByAggregator.
  Status CheckValid() const override;

  // Produce final outputs from this GroupByAggregator. Keys will only be output
  // for those tensors with nonempty tensor names in the output_key_specs_.
  // Values will be output from all inner intrinsics.
  //
  // The order in which the tensors will appear in the output is the following:
  // first, the keys with nonempty tensor names in the order they appear in the
  // output_tensor_specs, and next, the value tensors in the order the inner
  // intrinsics that produce each value tensor appear in the intrinsics input
  // vector.
  //
  // Once this function is called, CheckValid will return false.
  OutputTensorList TakeOutputs() && override;

 private:
  // Internal implementation of performing aggregation of the tensors in a
  // single InputTensorList into the state of this GroupByAggregator.
  Status AggregateTensorsInternal(InputTensorList tensors);

  // Internal implementation of TakeOutputs that returns all keys and values,
  // including keys that should not actually be returned in the final output.
  // Once this function is called, CheckValid will return false.
  OutputTensorList TakeOutputsInternal() &&;

  // If there are key tensors for this GroupByAggregator, then group key inputs
  // into unique composite keys, and produce an ordinal for each element of the
  // input corresponding to the index of the unique composite key in the output.
  // Otherwise, produce an ordinals vector of the same shape as the inputs, but
  // made up of all zeroes, so that all elements will be aggregated into a
  // single output element.
  StatusOr<Tensor> CreateOrdinalsByGroupingKeys(const InputTensorList& inputs);

  // Returns OK if the input and output tensor specs of the intrinsics
  // held by other match those of the sub-intrinsics held by this
  // GroupByAggregator, and the data types of input keys and the TensorSpecs of
  // output keys match those for this GroupByAggregator. Otherwise returns
  // INVALID_ARGUMENT.
  // TODO(team): Also validate that intrinsic URIs match.
  Status IsCompatible(const GroupByAggregator& other) const;

  bool output_consumed_ = false;
  int num_inputs_;
  const size_t num_keys_per_input_;
  size_t num_tensors_per_input_;
  std::optional<CompositeKeyCombiner> key_combiner_ = std::nullopt;
  std::vector<Intrinsic> intrinsics_;
  const std::vector<TensorSpec>* output_key_specs_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_GROUP_BY_AGGREGATOR_H_
