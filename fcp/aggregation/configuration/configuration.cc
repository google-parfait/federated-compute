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

#include "fcp/aggregation/configuration/configuration.h"

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "fcp/aggregation/configuration/configuration.pb.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace fcp {
namespace aggregation {

namespace {

absl::Status IntrinsicConfigArgumentError(
    const Configuration::IntrinsicConfig& intrinsic_config,
    absl::string_view error_message) {
  return absl::InvalidArgumentError(
      absl::StrCat("IntrinsicConfig: ", error_message, ":\n",
                   intrinsic_config.DebugString()));
}

absl::Status ValidateConfiguration(const Configuration& configuration) {
  for (const Configuration::IntrinsicConfig& intrinsic_config :
       configuration.intrinsic_configs()) {
    if (!GetAggregatorFactory(intrinsic_config.intrinsic_uri()).ok()) {
      return IntrinsicConfigArgumentError(
          intrinsic_config, absl::StrCat(intrinsic_config.intrinsic_uri(),
                                         " is not a supported intrinsic_uri"));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Intrinsic> ParseIntrinsicConfig(
    const Configuration::IntrinsicConfig& intrinsic_config);

absl::StatusOr<std::vector<Intrinsic>> ParseIntrinsicConfigs(
    const google::protobuf::RepeatedPtrField<Configuration::IntrinsicConfig>&
        intrinsic_configs) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.reserve(intrinsic_configs.size());
  for (const Configuration::IntrinsicConfig& intrinsic_config :
       intrinsic_configs) {
    ABSL_ASSIGN_OR_RETURN(Intrinsic intrinsic,
                          ParseIntrinsicConfig(intrinsic_config));
    intrinsics.push_back(std::move(intrinsic));
  }
  return intrinsics;
}

absl::StatusOr<Intrinsic> ParseIntrinsicConfig(
    const Configuration::IntrinsicConfig& intrinsic_config) {
  std::vector<TensorSpec> input_tensor_specs;
  std::vector<Tensor> params;
  for (const auto& intrinsic_arg : intrinsic_config.intrinsic_args()) {
    switch (intrinsic_arg.arg_case()) {
      case Configuration::IntrinsicConfig::IntrinsicArg::kInputTensor: {
        ABSL_ASSIGN_OR_RETURN(
            TensorSpec input_spec,
            TensorSpec::FromProto(intrinsic_arg.input_tensor()));
        input_tensor_specs.push_back(std::move(input_spec));
        break;
      }
      case Configuration::IntrinsicConfig::IntrinsicArg::kParameter: {
        ABSL_ASSIGN_OR_RETURN(Tensor param,
                              Tensor::FromProto(intrinsic_arg.parameter()));
        params.push_back(std::move(param));
        break;
      }
      case Configuration::IntrinsicConfig::IntrinsicArg::ARG_NOT_SET: {
        return IntrinsicConfigArgumentError(
            intrinsic_config, "All intrinsic args must have an arg value");
      }
    }
  }

  std::vector<TensorSpec> output_tensor_specs;
  output_tensor_specs.reserve(intrinsic_config.output_tensors_size());
  for (const auto& output_tensor_spec_proto :
       intrinsic_config.output_tensors()) {
    ABSL_ASSIGN_OR_RETURN(TensorSpec output_spec,
                          TensorSpec::FromProto(output_tensor_spec_proto));
    output_tensor_specs.push_back(std::move(output_spec));
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<Intrinsic> nested_intrinsics,
      ParseIntrinsicConfigs(intrinsic_config.inner_intrinsics()));

  return Intrinsic{intrinsic_config.intrinsic_uri(),
                   std::move(input_tensor_specs),
                   std::move(output_tensor_specs), std::move(params),
                   std::move(nested_intrinsics)};
}

}  // namespace

absl::StatusOr<std::vector<Intrinsic>> ParseConfiguration(
    const Configuration& configuration) {
  ABSL_RETURN_IF_ERROR(ValidateConfiguration(configuration));
  return ParseIntrinsicConfigs(configuration.intrinsic_configs());
}

}  // namespace aggregation
}  // namespace fcp
