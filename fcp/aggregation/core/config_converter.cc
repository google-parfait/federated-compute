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
#include "fcp/aggregation/core/config_converter.h"

#include <utility>
#include <vector>

#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/aggregation/tensorflow/converters.h"

namespace fcp {
namespace aggregation {

Status ServerAggregationConfigArgumentError(
    const Configuration::ServerAggregationConfig& aggregation_config,
    string_view error_message) {
  return FCP_STATUS(INVALID_ARGUMENT)
         << "ServerAggregationConfig: " << error_message << ":\n"
         << aggregation_config.DebugString();
}

StatusOr<Intrinsic> ParseFromConfig(
    const Configuration::ServerAggregationConfig& aggregation_config) {
  // Convert the tensor specifications.
  std::vector<TensorSpec> input_tensor_specs;
  std::vector<Tensor> params;
  for (const auto& intrinsic_arg : aggregation_config.intrinsic_args()) {
    switch (intrinsic_arg.arg_case()) {
      case Configuration_ServerAggregationConfig_IntrinsicArg::kInputTensor: {
        FCP_ASSIGN_OR_RETURN(
            TensorSpec input_spec,
            tensorflow::ConvertTensorSpec(intrinsic_arg.input_tensor()));
        input_tensor_specs.push_back(std::move(input_spec));
        break;
      }
      case Configuration_ServerAggregationConfig_IntrinsicArg::kParameter: {
        FCP_ASSIGN_OR_RETURN(Tensor param, tensorflow::ConvertTensorProto(
                                               intrinsic_arg.parameter()));
        params.push_back(std::move(param));
        break;
      }
      case Configuration_ServerAggregationConfig_IntrinsicArg::ARG_NOT_SET: {
        return ServerAggregationConfigArgumentError(
            aggregation_config, "All intrinsic args must have an arg value.");
      }
    }
  }
  std::vector<TensorSpec> output_tensor_specs;
  for (const auto& output_tensor_spec_proto :
       aggregation_config.output_tensors()) {
    FCP_ASSIGN_OR_RETURN(TensorSpec output_spec, tensorflow::ConvertTensorSpec(
                                                     output_tensor_spec_proto));
    output_tensor_specs.push_back(std::move(output_spec));
  }
  std::vector<Intrinsic> intrinsics;
  for (const Configuration::ServerAggregationConfig& nested_config :
       aggregation_config.inner_aggregations()) {
    FCP_ASSIGN_OR_RETURN(Intrinsic intrinsic, ParseFromConfig(nested_config));
    intrinsics.push_back(std::move(intrinsic));
  }
  return Intrinsic{
      aggregation_config.intrinsic_uri(), std::move(input_tensor_specs),
      std::move(output_tensor_specs), std::move(params), std::move(intrinsics)};
}

}  // namespace aggregation
}  // namespace fcp
