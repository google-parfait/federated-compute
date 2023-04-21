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

#include "fcp/aggregation/protocol/simple_aggregation/simple_aggregation_protocol.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"
#include "fcp/aggregation/tensorflow/converters.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/plan.pb.h"

namespace fcp::aggregation {

// Creates an INVALID_ARGUMENT error with the provided error message.
absl::Status ServerAggregationConfigArgumentError(
    const Configuration::ServerAggregationConfig& aggregation_config,
    absl::string_view error_message) {
  return absl::InvalidArgumentError(
      absl::StrFormat("ServerAggregationConfig: %s\n:%s", error_message,
                      aggregation_config.DebugString()));
}

// Creates an aggregation intrinsic based on the intrinsic configuration.
absl::StatusOr<SimpleAggregationProtocol::Intrinsic>
SimpleAggregationProtocol::CreateIntrinsic(
    const Configuration::ServerAggregationConfig& aggregation_config) {
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

absl::Status SimpleAggregationProtocol::ValidateConfig(
    const Configuration& configuration) {
  for (const Configuration::ServerAggregationConfig& aggregation_config :
       configuration.aggregation_configs()) {
    // TODO(team): Add support for other intrinsics after MVP launch.
    if (!GetAggregatorFactory(aggregation_config.intrinsic_uri()).ok()) {
      return ServerAggregationConfigArgumentError(
          aggregation_config,
          absl::StrFormat("%s is not a supported intrinsic_uri.",
                          aggregation_config.intrinsic_uri()));
    }

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
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<SimpleAggregationProtocol>>
SimpleAggregationProtocol::Create(
    const Configuration& configuration, AggregationProtocol::Callback* callback,
    const CheckpointParserFactory* checkpoint_parser_factory,
    const CheckpointBuilderFactory* checkpoint_builder_factory,
    ResourceResolver* resource_resolver) {
  FCP_CHECK(callback != nullptr);
  FCP_CHECK(checkpoint_parser_factory != nullptr);
  FCP_CHECK(checkpoint_builder_factory != nullptr);
  FCP_CHECK(resource_resolver != nullptr);
  FCP_RETURN_IF_ERROR(ValidateConfig(configuration));

  std::vector<Intrinsic> intrinsics;
  for (const Configuration::ServerAggregationConfig& aggregation_config :
       configuration.aggregation_configs()) {
    FCP_ASSIGN_OR_RETURN(Intrinsic intrinsic,
                         CreateIntrinsic(aggregation_config));
    intrinsics.emplace_back(std::move(intrinsic));
  }

  return absl::WrapUnique(new SimpleAggregationProtocol(
      std::move(intrinsics), callback, checkpoint_parser_factory,
      checkpoint_builder_factory, resource_resolver));
}

SimpleAggregationProtocol::SimpleAggregationProtocol(
    std::vector<Intrinsic> intrinsics, AggregationProtocol::Callback* callback,
    const CheckpointParserFactory* checkpoint_parser_factory,
    const CheckpointBuilderFactory* checkpoint_builder_factory,
    ResourceResolver* resource_resolver)
    : protocol_state_(PROTOCOL_CREATED),
      intrinsics_(std::move(intrinsics)),
      callback_(callback),
      checkpoint_parser_factory_(checkpoint_parser_factory),
      checkpoint_builder_factory_(checkpoint_builder_factory),
      resource_resolver_(resource_resolver) {}

absl::string_view SimpleAggregationProtocol::ProtocolStateDebugString(
    ProtocolState state) {
  switch (state) {
    case PROTOCOL_CREATED:
      return "PROTOCOL_CREATED";
    case PROTOCOL_STARTED:
      return "PROTOCOL_STARTED";
    case PROTOCOL_COMPLETED:
      return "PROTOCOL_COMPLETED";
    case PROTOCOL_ABORTED:
      return "PROTOCOL_ABORTED";
  }
}

absl::string_view SimpleAggregationProtocol::ClientStateDebugString(
    ClientState state) {
  switch (state) {
    case CLIENT_PENDING:
      return "CLIENT_PENDING";
    case CLIENT_RECEIVED_INPUT_AND_PENDING:
      return "CLIENT_RECEIVED_INPUT_AND_PENDING";
    case CLIENT_COMPLETED:
      return "CLIENT_COMPLETED";
    case CLIENT_FAILED:
      return "CLIENT_FAILED";
    case CLIENT_ABORTED:
      return "CLIENT_ABORTED";
    case CLIENT_DISCARDED:
      return "CLIENT_DISCARDED";
  }
}

absl::Status SimpleAggregationProtocol::CheckProtocolState(
    ProtocolState state) const {
  if (protocol_state_ != state) {
    return absl::FailedPreconditionError(
        absl::StrFormat("The current protocol state is %s, expected %s.",
                        ProtocolStateDebugString(protocol_state_),
                        ProtocolStateDebugString(state)));
  }
  return absl::OkStatus();
}

void SimpleAggregationProtocol::SetProtocolState(ProtocolState state) {
  FCP_CHECK(
      (protocol_state_ == PROTOCOL_CREATED && state == PROTOCOL_STARTED) ||
      (protocol_state_ == PROTOCOL_STARTED &&
       (state == PROTOCOL_COMPLETED || state == PROTOCOL_ABORTED)))
      << "Invalid protocol state transition from "
      << ProtocolStateDebugString(protocol_state_) << " to "
      << ProtocolStateDebugString(state) << ".";
  protocol_state_ = state;
}

absl::StatusOr<SimpleAggregationProtocol::ClientState>
SimpleAggregationProtocol::GetClientState(int64_t client_id) const {
  if (client_id < 0 || client_id >= client_states_.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("client_id %ld is outside the valid range", client_id));
  }
  return client_states_[client_id];
}

void SimpleAggregationProtocol::SetClientState(int64_t client_id,
                                               ClientState to_state) {
  FCP_CHECK(client_id >= 0 && client_id < client_states_.size());
  ClientState from_state = client_states_[client_id];
  FCP_CHECK(from_state != to_state);
  if (from_state == CLIENT_RECEIVED_INPUT_AND_PENDING) {
    num_clients_received_and_pending_--;
  } else if (from_state == CLIENT_COMPLETED) {
    FCP_CHECK(to_state == CLIENT_DISCARDED)
        << "Client state can't be changed from CLIENT_COMPLETED to "
        << ClientStateDebugString(to_state);
    num_clients_aggregated_--;
  } else {
    FCP_CHECK(from_state == CLIENT_PENDING)
        << "Client state can't be changed from "
        << ClientStateDebugString(from_state);
  }
  client_states_[client_id] = to_state;
  switch (to_state) {
    case CLIENT_PENDING:
      FCP_LOG(FATAL) << "Client state can't be changed to CLIENT_PENDING";
      break;
    case CLIENT_RECEIVED_INPUT_AND_PENDING:
      num_clients_received_and_pending_++;
      break;
    case CLIENT_COMPLETED:
      num_clients_aggregated_++;
      break;
    case CLIENT_FAILED:
      num_clients_failed_++;
      break;
    case CLIENT_ABORTED:
      num_clients_aborted_++;
      break;
    case CLIENT_DISCARDED:
      num_clients_discarded_++;
      break;
  }
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
  absl::MutexLock lock(&aggregation_mu_);
  if (!aggregation_finished_) {
    for (const auto& intrinsic : intrinsics_) {
      // TODO(team): Support multiple input tensors.
      const auto& it = tensor_map.find(intrinsic.input.name());
      FCP_CHECK(it != tensor_map.end());
      FCP_CHECK(intrinsic.aggregator != nullptr)
          << "CreateReport() has already been called.";
      FCP_RETURN_IF_ERROR(intrinsic.aggregator->Accumulate(it->second));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::Cord> SimpleAggregationProtocol::CreateReport() {
  absl::MutexLock lock(&aggregation_mu_);
  for (auto& intrinsic : intrinsics_) {
    FCP_CHECK(intrinsic.aggregator != nullptr)
        << "CreateReport() has already been called.";
    if (!intrinsic.aggregator->CanReport()) {
      return absl::FailedPreconditionError(
          "The aggregation can't be completed due to failed preconditions.");
    }
  }

  // Build the resulting checkpoint.
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      checkpoint_builder_factory_->Create();
  for (auto& intrinsic : intrinsics_) {
    FCP_ASSIGN_OR_RETURN(OutputTensorList output_tensors,
                         std::move(*intrinsic.aggregator).Report());
    // TODO(team): Support multiple output tensors per intrinsic.
    FCP_CHECK(output_tensors.size() == 1);
    const Tensor& tensor = output_tensors[0];
    FCP_CHECK(tensor.dtype() == intrinsic.output.dtype());
    FCP_CHECK(tensor.shape() == intrinsic.output.shape());
    FCP_RETURN_IF_ERROR(
        checkpoint_builder->Add(intrinsic.output.name(), tensor));
  }
  aggregation_finished_ = true;
  return checkpoint_builder->Build();
}

absl::Status SimpleAggregationProtocol::Start(int64_t num_clients) {
  if (num_clients < 0) {
    return absl::InvalidArgumentError("Number of clients cannot be negative.");
  }
  {
    absl::MutexLock lock(&state_mu_);
    FCP_RETURN_IF_ERROR(CheckProtocolState(PROTOCOL_CREATED));
    SetProtocolState(PROTOCOL_STARTED);
    FCP_CHECK(client_states_.empty());
    client_states_.resize(num_clients, CLIENT_PENDING);
  }
  if (num_clients > 0) {
    AcceptanceMessage acceptance_message;
    callback_->OnAcceptClients(0, num_clients, acceptance_message);
  }
  return absl::OkStatus();
}

absl::Status SimpleAggregationProtocol::AddClients(int64_t num_clients) {
  int64_t start_index;
  {
    absl::MutexLock lock(&state_mu_);
    FCP_RETURN_IF_ERROR(CheckProtocolState(PROTOCOL_STARTED));
    if (num_clients <= 0) {
      return absl::InvalidArgumentError("Non-zero number of clients required");
    }
    start_index = client_states_.size();
    client_states_.resize(start_index + num_clients, CLIENT_PENDING);
  }
  AcceptanceMessage acceptance_message;
  callback_->OnAcceptClients(start_index, num_clients, acceptance_message);
  return absl::OkStatus();
}

absl::Status SimpleAggregationProtocol::ReceiveClientMessage(
    int64_t client_id, const ClientMessage& message) {
  if (!message.has_simple_aggregation() ||
      !message.simple_aggregation().has_input()) {
    return absl::InvalidArgumentError("Unexpected message");
  }

  if (!message.simple_aggregation().input().has_inline_bytes() &&
      !message.simple_aggregation().input().has_uri()) {
    return absl::InvalidArgumentError(
        "Only inline_bytes or uri type of input is supported");
  }

  // Verify the state.
  {
    absl::MutexLock lock(&state_mu_);
    if (protocol_state_ == PROTOCOL_CREATED) {
      return absl::FailedPreconditionError("The protocol hasn't been started");
    }
    FCP_ASSIGN_OR_RETURN(auto client_state, GetClientState(client_id));
    if (client_state != CLIENT_PENDING) {
      // TODO(team): Decide whether the logging level should be INFO or
      // WARNING, or perhaps it should depend on the client state (e.g. WARNING
      // for COMPLETED and INFO for other states).
      FCP_LOG(INFO) << "ReceiveClientMessage: client " << client_id
                    << " message ignored, the state is already "
                    << ClientStateDebugString(client_state);
      return absl::OkStatus();
    }
    SetClientState(client_id, CLIENT_RECEIVED_INPUT_AND_PENDING);
  }

  absl::Status client_completion_status = absl::OkStatus();
  ClientState client_completion_state = CLIENT_COMPLETED;

  absl::Cord report;
  if (message.simple_aggregation().input().has_inline_bytes()) {
    // Parse the client input concurrently with other protocol calls.
    report =
        absl::Cord(message.simple_aggregation().input().inline_bytes());
  } else {
    absl::StatusOr<absl::Cord> report_or_status =
        resource_resolver_->RetrieveResource(
            client_id, message.simple_aggregation().input().uri());
    if (!report_or_status.ok()) {
      client_completion_status = report_or_status.status();
      client_completion_state = CLIENT_FAILED;
      FCP_LOG(WARNING) << "Report with resource uri "
                       << message.simple_aggregation().input().uri()
                       << " for client " << client_id << "is missing. "
                       << client_completion_status.ToString();
    } else {
      report = std::move(report_or_status.value());
    }
  }

  if (client_completion_state != CLIENT_FAILED) {
    absl::StatusOr<TensorMap> tensor_map_or_status =
        ParseCheckpoint(std::move(report));
    if (!tensor_map_or_status.ok()) {
      client_completion_status = tensor_map_or_status.status();
      client_completion_state = CLIENT_FAILED;
      FCP_LOG(WARNING) << "Client " << client_id << " input can't be parsed: "
                       << client_completion_status.ToString();
    } else {
      // Aggregate the client input which would block on aggregation_mu_ if
      // there are any concurrent AggregateClientInput calls.
      client_completion_status =
          AggregateClientInput(std::move(tensor_map_or_status).value());
      if (!client_completion_status.ok()) {
        client_completion_state = CLIENT_DISCARDED;
        FCP_LOG(INFO) << "Client " << client_id << " input is discarded: "
                      << client_completion_status.ToString();
      }
    }
  }

  // Update the state post aggregation.
  {
    absl::MutexLock lock(&state_mu_);
    // Change the client state only if the current state is still
    // CLIENT_RECEIVED_INPUT_AND_PENDING, meaning that the client wasn't already
    // closed by a concurrent Complete or Abort call.
    if (client_states_[client_id] == CLIENT_RECEIVED_INPUT_AND_PENDING) {
      SetClientState(client_id, client_completion_state);
      callback_->OnCloseClient(client_id, client_completion_status);
    }
  }
  return absl::OkStatus();
}

absl::Status SimpleAggregationProtocol::CloseClient(
    int64_t client_id, absl::Status client_status) {
  {
    absl::MutexLock lock(&state_mu_);
    if (protocol_state_ == PROTOCOL_CREATED) {
      return absl::FailedPreconditionError("The protocol hasn't been started");
    }
    FCP_ASSIGN_OR_RETURN(auto client_state, GetClientState(client_id));
    // Close the client only if the client is currently pending.
    if (client_state == CLIENT_PENDING) {
      FCP_LOG(INFO) << "Closing client " << client_id << " with the status "
                    << client_status.ToString();
      SetClientState(client_id,
                     client_status.ok() ? CLIENT_DISCARDED : CLIENT_FAILED);
    }
  }

  return absl::OkStatus();
}

absl::Status SimpleAggregationProtocol::Complete() {
  absl::Cord result;
  std::vector<int64_t> client_ids_to_close;
  {
    absl::MutexLock lock(&state_mu_);
    FCP_RETURN_IF_ERROR(CheckProtocolState(PROTOCOL_STARTED));
    FCP_ASSIGN_OR_RETURN(result, CreateReport());
    SetProtocolState(PROTOCOL_COMPLETED);
    for (int64_t client_id = 0; client_id < client_states_.size();
         client_id++) {
      switch (client_states_[client_id]) {
        case CLIENT_PENDING:
          SetClientState(client_id, CLIENT_ABORTED);
          client_ids_to_close.push_back(client_id);
          break;
        case CLIENT_RECEIVED_INPUT_AND_PENDING:
          SetClientState(client_id, CLIENT_DISCARDED);
          client_ids_to_close.push_back(client_id);
          break;
        default:
          break;
      }
    }
  }
  for (int64_t client_id : client_ids_to_close) {
    callback_->OnCloseClient(
        client_id, absl::AbortedError("The protocol has completed before the "
                                      "client input has been aggregated."));
  }
  callback_->OnComplete(std::move(result));
  return absl::OkStatus();
}

absl::Status SimpleAggregationProtocol::Abort() {
  std::vector<int64_t> client_ids_to_close;
  {
    absl::MutexLock lock(&state_mu_);
    FCP_RETURN_IF_ERROR(CheckProtocolState(PROTOCOL_STARTED));
    aggregation_finished_ = true;
    SetProtocolState(PROTOCOL_ABORTED);
    for (int64_t client_id = 0; client_id < client_states_.size();
         client_id++) {
      switch (client_states_[client_id]) {
        case CLIENT_PENDING:
          SetClientState(client_id, CLIENT_ABORTED);
          client_ids_to_close.push_back(client_id);
          break;
        case CLIENT_RECEIVED_INPUT_AND_PENDING:
          SetClientState(client_id, CLIENT_DISCARDED);
          client_ids_to_close.push_back(client_id);
          break;
        case CLIENT_COMPLETED:
          SetClientState(client_id, CLIENT_DISCARDED);
          break;
        default:
          break;
      }
    }
  }

  for (int64_t client_id : client_ids_to_close) {
    callback_->OnCloseClient(
        client_id, absl::AbortedError("The protocol has aborted before the "
                                      "client input has been aggregated."));
  }
  return absl::OkStatus();
}

StatusMessage SimpleAggregationProtocol::GetStatus() {
  absl::MutexLock lock(&state_mu_);
  int64_t num_clients_completed = num_clients_received_and_pending_ +
                                  num_clients_aggregated_ +
                                  num_clients_discarded_;
  StatusMessage message;
  message.set_num_clients_completed(num_clients_completed);
  message.set_num_clients_failed(num_clients_failed_);
  message.set_num_clients_pending(client_states_.size() -
                                  num_clients_completed - num_clients_failed_ -
                                  num_clients_aborted_);
  message.set_num_inputs_aggregated_and_included(num_clients_aggregated_);
  message.set_num_inputs_aggregated_and_pending(
      num_clients_received_and_pending_);
  message.set_num_clients_aborted(num_clients_aborted_);
  message.set_num_inputs_discarded(num_clients_discarded_);
  return message;
}

}  // namespace fcp::aggregation
