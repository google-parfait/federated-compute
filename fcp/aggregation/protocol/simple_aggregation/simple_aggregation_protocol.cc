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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "fcp/aggregation/core/input_tensor_list.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"
#include "fcp/aggregation/protocol/config_converter.h"
#include "fcp/base/monitoring.h"

namespace fcp::aggregation {

// Creates an aggregation intrinsic based on the intrinsic configuration.
absl::StatusOr<std::unique_ptr<TensorAggregator>>
SimpleAggregationProtocol::CreateAggregator(const Intrinsic& intrinsic) {
  // Resolve the intrinsic_uri to the registered TensorAggregatorFactory.
  FCP_ASSIGN_OR_RETURN(const TensorAggregatorFactory* factory,
                       GetAggregatorFactory(intrinsic.uri));

  // Use the factory to create the TensorAggregator instance.
  return factory->Create(intrinsic);
}

absl::Status SimpleAggregationProtocol::ValidateConfig(
    const Configuration& configuration) {
  for (const Configuration::ServerAggregationConfig& aggregation_config :
       configuration.aggregation_configs()) {
    if (!GetAggregatorFactory(aggregation_config.intrinsic_uri()).ok()) {
      return ServerAggregationConfigArgumentError(
          aggregation_config,
          absl::StrFormat("%s is not a supported intrinsic_uri.",
                          aggregation_config.intrinsic_uri()));
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

  FCP_ASSIGN_OR_RETURN(std::vector<Intrinsic> intrinsics,
                       ParseFromConfig(configuration));
  std::vector<std::unique_ptr<TensorAggregator>> aggregators;
  for (const Intrinsic& intrinsic : intrinsics) {
    FCP_ASSIGN_OR_RETURN(std::unique_ptr<TensorAggregator> aggregator,
                         CreateAggregator(intrinsic));
    aggregators.emplace_back(std::move(aggregator));
  }

  return absl::WrapUnique(new SimpleAggregationProtocol(
      std::move(intrinsics), std::move(aggregators), callback,
      checkpoint_parser_factory, checkpoint_builder_factory,
      resource_resolver));
}

SimpleAggregationProtocol::SimpleAggregationProtocol(
    std::vector<Intrinsic> intrinsics,
    std::vector<std::unique_ptr<TensorAggregator>> aggregators,
    AggregationProtocol::Callback* callback,
    const CheckpointParserFactory* checkpoint_parser_factory,
    const CheckpointBuilderFactory* checkpoint_builder_factory,
    ResourceResolver* resource_resolver)
    : protocol_state_(PROTOCOL_CREATED),
      intrinsics_(std::move(intrinsics)),
      aggregators_(std::move(aggregators)),
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

namespace {

size_t CountInputs(const Intrinsic& intrinsic) {
  size_t count = intrinsic.inputs.size();
  for (const Intrinsic& nested_intrinsic : intrinsic.nested_intrinsics) {
    count += CountInputs(nested_intrinsic);
  }
  return count;
}

absl::Status AddInputsToMap(
    const Intrinsic& intrinsic, const CheckpointParser& parser,
    absl::flat_hash_map<std::string, Tensor>& tensor_map) {
  for (const TensorSpec& input_spec : intrinsic.inputs) {
    FCP_ASSIGN_OR_RETURN(Tensor tensor, parser.GetTensor(input_spec.name()));
    if (tensor.dtype() != input_spec.dtype() ||
        !input_spec.shape().MatchesKnownDimensions(tensor.shape())) {
      // TODO(team): Detailed diagnostics including the expected vs
      // actual data types and shapes.
      return absl::InvalidArgumentError("Input tensor spec mismatch.");
    }
    tensor_map.emplace(input_spec.name(), std::move(tensor));
  }
  for (const Intrinsic& nested_intrinsic : intrinsic.nested_intrinsics) {
    FCP_RETURN_IF_ERROR(AddInputsToMap(nested_intrinsic, parser, tensor_map));
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> PopulateInputs(
    const Intrinsic& intrinsic,
    const absl::flat_hash_map<std::string, Tensor>& tensor_map, size_t index,
    InputTensorList& inputs) {
  size_t num_inputs = intrinsic.inputs.size();
  for (const TensorSpec& input_spec : intrinsic.inputs) {
    const auto& it = tensor_map.find(input_spec.name());
    FCP_CHECK(it != tensor_map.end());
    inputs[index++] = &it->second;
  }
  for (const Intrinsic& nested_intrinsic : intrinsic.nested_intrinsics) {
    FCP_ASSIGN_OR_RETURN(
        size_t nested_num_inputs,
        PopulateInputs(nested_intrinsic, tensor_map, index, inputs));
    index += nested_num_inputs;
    num_inputs += nested_num_inputs;
  }
  return num_inputs;
}

absl::StatusOr<int> AddOutputsToCheckpoint(
    const Intrinsic& intrinsic, const OutputTensorList& outputs,
    int output_index, CheckpointBuilder& checkpoint_builder) {
  int num_outputs = 0;
  for (const TensorSpec& output_spec : intrinsic.outputs) {
    if (output_spec.name().empty()) {
      // TensorSpecs with empty names are not included in the output.
      continue;
    }
    num_outputs++;
    const Tensor& tensor = outputs[output_index++];
    FCP_CHECK(tensor.dtype() == output_spec.dtype());
    FCP_CHECK(output_spec.shape().MatchesKnownDimensions(tensor.shape()));
    FCP_RETURN_IF_ERROR(checkpoint_builder.Add(output_spec.name(), tensor));
  }
  for (const Intrinsic& nested_intrinsic : intrinsic.nested_intrinsics) {
    FCP_ASSIGN_OR_RETURN(
        int nested_num_outputs,
        AddOutputsToCheckpoint(nested_intrinsic, outputs, output_index,
                               checkpoint_builder));
    output_index += nested_num_outputs;
    num_outputs += nested_num_outputs;
  }
  return num_outputs;
}

}  // namespace

absl::StatusOr<SimpleAggregationProtocol::TensorMap>
SimpleAggregationProtocol::ParseCheckpoint(absl::Cord report) const {
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<CheckpointParser> parser,
                       checkpoint_parser_factory_->Create(report));
  TensorMap tensor_map;
  for (const auto& intrinsic : intrinsics_) {
    FCP_RETURN_IF_ERROR(AddInputsToMap(intrinsic, *parser, tensor_map));
  }

  return tensor_map;
}

absl::Status SimpleAggregationProtocol::AggregateClientInput(
    SimpleAggregationProtocol::TensorMap tensor_map) {
  absl::MutexLock lock(&aggregation_mu_);
  if (!aggregation_finished_) {
    for (int i = 0; i < intrinsics_.size(); ++i) {
      const Intrinsic& intrinsic = intrinsics_[i];
      InputTensorList inputs(CountInputs(intrinsic));
      FCP_RETURN_IF_ERROR(PopulateInputs(intrinsic, tensor_map, 0, inputs));
      FCP_CHECK(aggregators_[i] != nullptr)
          << "CreateReport() has already been called.";
      FCP_RETURN_IF_ERROR(aggregators_[i]->Accumulate(std::move(inputs)));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::Cord> SimpleAggregationProtocol::CreateReport() {
  absl::MutexLock lock(&aggregation_mu_);
  for (const auto& aggregator : aggregators_) {
    FCP_CHECK(aggregator != nullptr)
        << "CreateReport() has already been called.";
    if (!aggregator->CanReport()) {
      return absl::FailedPreconditionError(
          "The aggregation can't be completed due to failed preconditions.");
    }
  }

  // Build the resulting checkpoint.
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      checkpoint_builder_factory_->Create();
  for (int i = 0; i < intrinsics_.size(); ++i) {
    auto tensor_aggregator = std::move(aggregators_[i]);
    FCP_ASSIGN_OR_RETURN(OutputTensorList output_tensors,
                         std::move(*tensor_aggregator).Report());
    const Intrinsic& intrinsic = intrinsics_[i];
    FCP_ASSIGN_OR_RETURN(int num_outputs,
                         AddOutputsToCheckpoint(intrinsic, output_tensors, 0,
                                                *checkpoint_builder));
    FCP_CHECK(num_outputs == output_tensors.size())
        << "Number of tensors produced by TensorAggregator "
        << output_tensors.size()
        << " does not match number of output tensors with nonempty names "
        << num_outputs << ".";
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

absl::StatusOr<std::shared_ptr<ServerMessage>>
SimpleAggregationProtocol::PollServerMessage(int64_t client_id) {
  return std::shared_ptr<ServerMessage>(nullptr);
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

absl::StatusOr<absl::Cord> SimpleAggregationProtocol::GetResult() {
  return absl::UnimplementedError(
      "Results should be provided by Callback until this is implemented");
}

}  // namespace fcp::aggregation
