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

#include "fcp/aggregation/protocol/simple_aggregation_protocol.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"
#include "fcp/aggregation/tensorflow/converters.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/plan.pb.h"

namespace fcp::aggregation {

using ::google::internal::federated::plan::ServerAggregationConfig;

// Creates an INVALID_ARGUMENT error with the provided error message.
absl::Status ServerAggregationConfigArgumentError(
    const ServerAggregationConfig& aggregation_config,
    absl::string_view error_message) {
  return absl::InvalidArgumentError(
      absl::StrFormat("ServerAggregationConfig: %s\n:%s", error_message,
                      aggregation_config.DebugString()));
}

// Creates an aggregation intrinsic based on the intrinsic configuration.
absl::StatusOr<Intrinsic> CreateIntrinsic(
    const ServerAggregationConfig& aggregation_config) {
  // TODO(team): Support multiple intrinsic args.
  if (aggregation_config.intrinsic_args_size() != 1) {
    return ServerAggregationConfigArgumentError(
        aggregation_config, "Exactly one intrinsic argument is expected.");
  }

  if (aggregation_config.output_tensors_size() != 1) {
    return ServerAggregationConfigArgumentError(
        aggregation_config, "Exactly one output tensor is expected.");
  }

  if (!aggregation_config.intrinsic_args(0).has_input_tensor()) {
    return ServerAggregationConfigArgumentError(
        aggregation_config, "Intrinsic arguments must be input tensors.");
  }

  // Resolve the intrinsic_uri to the registered TensorAggregatorFactory.
  FCP_ASSIGN_OR_RETURN(
      const TensorAggregatorFactory* factory,
      GetAggregatorFactory(aggregation_config.intrinsic_uri()));

  // Convert the input tensor specification.
  FCP_ASSIGN_OR_RETURN(
      TensorSpec input_spec,
      tensorflow::ConvertTensorSpec(
          aggregation_config.intrinsic_args(0).input_tensor()));

  // Convert the output tensor specification.
  FCP_ASSIGN_OR_RETURN(
      TensorSpec output_spec,
      tensorflow::ConvertTensorSpec(aggregation_config.output_tensors(0)));

  // TODO(team): currently the input and output data type and shape are
  // expected to be the same.
  if (input_spec.dtype() != output_spec.dtype() ||
      input_spec.shape() != output_spec.shape()) {
    return ServerAggregationConfigArgumentError(
        aggregation_config, "Input and output tensors have mismatched specs.");
  }

  // Use the factory to create the TensorAggregator instance.
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<TensorAggregator> aggregator,
                       factory->Create(input_spec.dtype(), input_spec.shape()));

  return Intrinsic{std::move(input_spec), std::move(output_spec),
                   std::move(aggregator)};
}

absl::StatusOr<std::unique_ptr<SimpleAggregationProtocol>>
SimpleAggregationProtocol::Create(
    const Configuration& configuration, AggregationProtocol::Callback* callback,
    const CheckpointParserFactory* checkpoint_parser_factory,
    const CheckpointBuilderFactory* checkpoint_builder_factory) {
  FCP_CHECK(callback != nullptr);
  FCP_CHECK(checkpoint_parser_factory != nullptr);
  FCP_CHECK(checkpoint_builder_factory != nullptr);

  std::vector<Intrinsic> intrinsics;
  for (const ServerAggregationConfig& aggregation_config :
       configuration.aggregation_configs()) {
    FCP_ASSIGN_OR_RETURN(Intrinsic intrinsic,
                         CreateIntrinsic(aggregation_config));
    intrinsics.emplace_back(std::move(intrinsic));
  }

  return absl::WrapUnique(new SimpleAggregationProtocol(
      std::move(intrinsics), callback, checkpoint_parser_factory,
      checkpoint_builder_factory));
}

SimpleAggregationProtocol::SimpleAggregationProtocol(
    std::vector<Intrinsic> intrinsics, AggregationProtocol::Callback* callback,
    const CheckpointParserFactory* checkpoint_parser_factory,
    const CheckpointBuilderFactory* checkpoint_builder_factory)
    : intrinsics_(std::move(intrinsics)),
      callback_(callback),
      checkpoint_parser_factory_(checkpoint_parser_factory),
      checkpoint_builder_factory_(checkpoint_builder_factory) {}

// TODO(team): Implement Simple Aggregation Protocol methods.
absl::Status SimpleAggregationProtocol::Start(int64_t num_clients) {
  return absl::UnimplementedError("Start is not implemented");
}

absl::Status SimpleAggregationProtocol::AddClients(int64_t num_clients) {
  return absl::UnimplementedError("AddClients is not implemented");
}

absl::StatusOr<SimpleAggregationProtocol::TensorMap>
SimpleAggregationProtocol::ParseCheckpoint(absl::Cord report) const {
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<CheckpointParser> parser,
                       checkpoint_parser_factory_->Create(report));
  TensorMap tensor_map;
  for (const auto& intrinsic : intrinsics_) {
    // TODO(team): Support multiple input tensors.
    FCP_ASSIGN_OR_RETURN(Tensor tensor,
                         parser->GetTensor(intrinsic.input.name()));
    if (tensor.dtype() != intrinsic.input.dtype() ||
        tensor.shape() != intrinsic.input.shape()) {
      // TODO(team): Detailed diagnostics including the expected vs
      // actual data types and shapes.
      return absl::InvalidArgumentError("Input tensor spec mismatch.");
    }
    tensor_map.emplace(intrinsic.input.name(), std::move(tensor));
  }

  return tensor_map;
}

absl::Status SimpleAggregationProtocol::AggregateClientInput(
    SimpleAggregationProtocol::TensorMap tensor_map) {
  for (const auto& intrinsic : intrinsics_) {
    // TODO(team): Support multiple input tensors.
    const auto& it = tensor_map.find(intrinsic.input.name());
    FCP_CHECK(it != tensor_map.end());
    FCP_RETURN_IF_ERROR(intrinsic.aggregator->Accumulate(it->second));
  }
  return absl::OkStatus();
}

absl::Status SimpleAggregationProtocol::ReceiveClientInput(int64_t client_id,
                                                           absl::Cord report) {
  // Do the initial parsing of the report before the synchronized part of the
  // processing.
  absl::StatusOr<TensorMap> tensor_map_or_status =
      ParseCheckpoint(std::move(report));
  if (!tensor_map_or_status.ok()) {
    // TODO(team): Call CloseClient once it is implemented.
    FCP_LOG(WARNING)
        << "ReceiveClientInput failed to parse the input for client "
        << client_id << " : " << tensor_map_or_status.status().ToString();
    return absl::OkStatus();
  }

  // TODO(team): Add synchronization locking here.
  // TODO(team): Verify client specific state.
  absl::Status aggregation_status =
      AggregateClientInput(std::move(tensor_map_or_status).value());

  // TODO(team): Update the client specific state and close the client
  // connection.
  return aggregation_status;
}

absl::Status SimpleAggregationProtocol::ReceiveClientMessage(
    int64_t client_id, const ClientMessage& message) {
  return absl::UnimplementedError(
      "ReceiveClientMessage is not supported by SimpleAggregationProtocol");
}

absl::Status SimpleAggregationProtocol::CloseClient(
    int64_t client_id, absl::Status client_status) {
  return absl::UnimplementedError("CloseClient is not implemented");
}

absl::Status SimpleAggregationProtocol::Complete() {
  for (auto& intrinsic : intrinsics_) {
    if (!intrinsic.aggregator->CanReport()) {
      return absl::FailedPreconditionError(
          "The aggregation can't be completed due to failed preconditions.");
    }
  }

  // Build the resulting checkpoint.
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      checkpoint_builder_factory_->Create();
  for (auto& intrinsic : intrinsics_) {
    // TODO(team): Support multiple output tensors per intrinsic.
    FCP_ASSIGN_OR_RETURN(Tensor tensor,
                         std::move(*intrinsic.aggregator).Report());
    FCP_CHECK(tensor.dtype() == intrinsic.output.dtype());
    FCP_CHECK(tensor.shape() == intrinsic.output.shape());
    FCP_RETURN_IF_ERROR(
        checkpoint_builder->Add(intrinsic.output.name(), tensor));
  }

  FCP_ASSIGN_OR_RETURN(absl::Cord result, checkpoint_builder->Build());
  callback_->Complete(result);
  return absl::OkStatus();
}

absl::Status SimpleAggregationProtocol::Abort() {
  return absl::UnimplementedError("Abort is not implemented");
}

StatusMessage SimpleAggregationProtocol::GetStatus() {
  StatusMessage status_message;
  // TODO(team): Populate status_message before returning.
  return status_message;
}

}  // namespace fcp::aggregation
