/*
 * Copyright 2020 Google LLC
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
#include "fcp/client/grpc_federated_protocol.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "google/protobuf/duration.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/time_util.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/federated_protocol_util.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/grpc_bidi_stream.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/in_memory_request_response.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/secagg_event_publisher.h"
#include "fcp/client/secagg_runner.h"
#include "fcp/client/stats.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/secagg/client/secagg_client.h"
#include "fcp/secagg/client/send_to_server_interface.h"
#include "fcp/secagg/client/state_transition_listener_interface.h"
#include "fcp/secagg/shared/aes_ctr_prng_factory.h"
#include "fcp/secagg/shared/crypto_rand_prng.h"
#include "fcp/secagg/shared/input_vector_specification.h"
#include "fcp/secagg/shared/math.h"
#include "fcp/secagg/shared/secagg_messages.pb.h"
#include "fcp/secagg/shared/secagg_vector.h"

namespace fcp {
namespace client {

using ::fcp::client::http::UriOrInlineData;
using ::fcp::secagg::ClientToServerWrapperMessage;
using ::google::internal::federatedml::v2::CheckinRequest;
using ::google::internal::federatedml::v2::CheckinRequestAck;
using ::google::internal::federatedml::v2::CheckinResponse;
using ::google::internal::federatedml::v2::ClientExecutionStats;
using ::google::internal::federatedml::v2::ClientStreamMessage;
using ::google::internal::federatedml::v2::EligibilityEvalCheckinRequest;
using ::google::internal::federatedml::v2::EligibilityEvalCheckinResponse;
using ::google::internal::federatedml::v2::EligibilityEvalPayload;
using ::google::internal::federatedml::v2::HttpCompressionFormat;
using ::google::internal::federatedml::v2::ProtocolOptionsRequest;
using ::google::internal::federatedml::v2::RetryWindow;
using ::google::internal::federatedml::v2::ServerStreamMessage;
using ::google::internal::federatedml::v2::SideChannelExecutionInfo;
using ::google::internal::federatedml::v2::TaskEligibilityInfo;

// A note on error handling:
//
// The implementation here makes a distinction between what we call 'transient'
// and 'permanent' errors. While the exact categorization of transient vs.
// permanent errors is defined by a flag, the intent is that transient errors
// are those types of errors that may occur in the regular course of business,
// e.g. due to an interrupted network connection, a load balancer temporarily
// rejecting our request etc. Generally, these are expected to be resolvable by
// merely retrying the request at a slightly later time. Permanent errors are
// intended to be those that are not expected to be resolvable as quickly or by
// merely retrying the request. E.g. if a client checks in to the server with a
// population name that doesn't exist, then the server may return NOT_FOUND, and
// until the server-side configuration is changed, it will continue returning
// such an error. Hence, such errors can warrant a longer retry period (to waste
// less of both the client's and server's resources).
//
// The errors also differ in how they interact with the server-specified retry
// windows that are returned via the CheckinRequestAck message.
// - If a permanent error occurs, then we will always return a retry window
//   based on the target 'permanent errors retry period' flag, regardless of
//   whether we received a CheckinRequestAck from the server at an earlier time.
// - If a transient error occurs, then we will only return a retry window
//   based on the target 'transient errors retry period' flag if the server
//   didn't already return a CheckinRequestAck. If it did return such an ack,
//   then one of the retry windows in that message will be used instead.
//
// Finally, note that for simplicity's sake we generally check whether a
// permanent error was received at the level of this class's public method,
// rather than deeper down in each of our helper methods that actually call
// directly into the gRPC stack. This keeps our state-managing code simpler, but
// does mean that if any of our helper methods like SendCheckinRequest produce a
// permanent error code locally (i.e. without it being sent by the server), it
// will be treated as if the server sent it and the permanent error retry period
// will be used. We consider this a reasonable tradeoff.

GrpcFederatedProtocol::GrpcFederatedProtocol(
    EventPublisher* event_publisher, LogManager* log_manager,
    std::unique_ptr<SecAggRunnerFactory> secagg_runner_factory,
    const Flags* flags, ::fcp::client::http::HttpClient* http_client,
    const std::string& federated_service_uri, const std::string& api_key,
    const std::string& test_cert_path, absl::string_view population_name,
    absl::string_view retry_token, absl::string_view client_version,
    absl::string_view attestation_measurement,
    std::function<bool()> should_abort,
    const InterruptibleRunner::TimingConfig& timing_config,
    const int64_t grpc_channel_deadline_seconds,
    cache::ResourceCache* resource_cache)
    : GrpcFederatedProtocol(
          event_publisher, log_manager, std::move(secagg_runner_factory), flags,
          http_client,
          std::make_unique<GrpcBidiStream>(
              federated_service_uri, api_key, std::string(population_name),
              grpc_channel_deadline_seconds, test_cert_path),
          population_name, retry_token, client_version, attestation_measurement,
          should_abort, absl::BitGen(), timing_config, resource_cache) {}

GrpcFederatedProtocol::GrpcFederatedProtocol(
    EventPublisher* event_publisher, LogManager* log_manager,
    std::unique_ptr<SecAggRunnerFactory> secagg_runner_factory,
    const Flags* flags, ::fcp::client::http::HttpClient* http_client,
    std::unique_ptr<GrpcBidiStreamInterface> grpc_bidi_stream,
    absl::string_view population_name, absl::string_view retry_token,
    absl::string_view client_version, absl::string_view attestation_measurement,
    std::function<bool()> should_abort, absl::BitGen bit_gen,
    const InterruptibleRunner::TimingConfig& timing_config,
    cache::ResourceCache* resource_cache)
    : object_state_(ObjectState::kInitialized),
      event_publisher_(event_publisher),
      log_manager_(log_manager),
      secagg_runner_factory_(std::move(secagg_runner_factory)),
      flags_(flags),
      http_client_(http_client),
      grpc_bidi_stream_(std::move(grpc_bidi_stream)),
      population_name_(population_name),
      retry_token_(retry_token),
      client_version_(client_version),
      attestation_measurement_(attestation_measurement),
      bit_gen_(std::move(bit_gen)),
      resource_cache_(resource_cache) {
  interruptible_runner_ = std::make_unique<InterruptibleRunner>(
      log_manager, should_abort, timing_config,
      InterruptibleRunner::DiagnosticsConfig{
          .interrupted = ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_GRPC,
          .interrupt_timeout =
              ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_GRPC_TIMED_OUT,
          .interrupted_extended = ProdDiagCode::
              BACKGROUND_TRAINING_INTERRUPT_GRPC_EXTENDED_COMPLETED,
          .interrupt_timeout_extended = ProdDiagCode::
              BACKGROUND_TRAINING_INTERRUPT_GRPC_EXTENDED_TIMED_OUT});
  // Note that we could cast the provided error codes to absl::StatusCode
  // values here. However, that means we'd have to handle the case when
  // invalid integers that don't map to a StatusCode enum are provided in the
  // flag here. Instead, we cast absl::StatusCodes to int32_t each time we
  // compare them with the flag-provided list of codes, which means we never
  // have to worry about invalid flag values (besides the fact that invalid
  // values will be silently ignored, which could make it harder to realize when
  // flag is misconfigured).
  const std::vector<int32_t>& error_codes =
      flags->federated_training_permanent_error_codes();
  federated_training_permanent_error_codes_ =
      absl::flat_hash_set<int32_t>(error_codes.begin(), error_codes.end());
}

GrpcFederatedProtocol::~GrpcFederatedProtocol() { grpc_bidi_stream_->Close(); }

absl::Status GrpcFederatedProtocol::Send(
    google::internal::federatedml::v2::ClientStreamMessage*
        client_stream_message) {
  // Note that this stopwatch measurement may not fully measure the time it
  // takes to send all of the data, as it may return before all data was written
  // to the network socket. It's the best estimate we can provide though.
  auto started_stopwatch = network_stopwatch_->Start();
  FCP_RETURN_IF_ERROR(interruptible_runner_->Run(
      [this, &client_stream_message]() {
        return this->grpc_bidi_stream_->Send(client_stream_message);
      },
      [this]() { this->grpc_bidi_stream_->Close(); }));
  return absl::OkStatus();
}

absl::Status GrpcFederatedProtocol::Receive(
    google::internal::federatedml::v2::ServerStreamMessage*
        server_stream_message) {
  // Note that this stopwatch measurement will generally include time spent
  // waiting for the server to return a response (i.e. idle time rather than the
  // true time it takes to send/receive data on the network). It's the best
  // estimate we can provide though.
  auto started_stopwatch = network_stopwatch_->Start();
  FCP_RETURN_IF_ERROR(interruptible_runner_->Run(
      [this, &server_stream_message]() {
        return grpc_bidi_stream_->Receive(server_stream_message);
      },
      [this]() { this->grpc_bidi_stream_->Close(); }));
  return absl::OkStatus();
}

ProtocolOptionsRequest GrpcFederatedProtocol::CreateProtocolOptionsRequest(
    bool should_ack_checkin) const {
  ProtocolOptionsRequest request;
  request.set_should_ack_checkin(should_ack_checkin);
  request.set_supports_http_download(http_client_ != nullptr);
  request.set_supports_eligibility_eval_http_download(
      http_client_ != nullptr &&
      flags_->enable_grpc_with_eligibility_eval_http_resource_support());

  // Note that we set this field for both eligibility eval checkin requests
  // and regular checkin requests. Even though eligibility eval tasks do not
  // have any aggregation phase, we still advertise the client's support for
  // Secure Aggregation during the eligibility eval checkin phase. We do
  // this because it doesn't hurt anything, and because letting the server
  // know whether client supports SecAgg sooner rather than later in the
  // protocol seems to provide maximum flexibility if the server ever were
  // to use that information at this stage of the protocol in the future.
  request.mutable_side_channels()
      ->mutable_secure_aggregation()
      ->add_client_variant(secagg::SECAGG_CLIENT_VARIANT_NATIVE_V1);
    request.mutable_supported_http_compression_formats()->Add(
        HttpCompressionFormat::HTTP_COMPRESSION_FORMAT_GZIP);
  return request;
}

absl::Status GrpcFederatedProtocol::SendEligibilityEvalCheckinRequest() {
  ClientStreamMessage client_stream_message;
  EligibilityEvalCheckinRequest* eligibility_checkin_request =
      client_stream_message.mutable_eligibility_eval_checkin_request();
  eligibility_checkin_request->set_population_name(population_name_);
  eligibility_checkin_request->set_retry_token(retry_token_);
  eligibility_checkin_request->set_client_version(client_version_);
  eligibility_checkin_request->set_attestation_measurement(
      attestation_measurement_);
  *eligibility_checkin_request->mutable_protocol_options_request() =
      CreateProtocolOptionsRequest(
          /* should_ack_checkin=*/true);

  return Send(&client_stream_message);
}

absl::Status GrpcFederatedProtocol::SendCheckinRequest(
    const std::optional<TaskEligibilityInfo>& task_eligibility_info) {
  ClientStreamMessage client_stream_message;
  CheckinRequest* checkin_request =
      client_stream_message.mutable_checkin_request();
  checkin_request->set_population_name(population_name_);
  checkin_request->set_retry_token(retry_token_);
  checkin_request->set_client_version(client_version_);
  checkin_request->set_attestation_measurement(attestation_measurement_);
  *checkin_request->mutable_protocol_options_request() =
      CreateProtocolOptionsRequest(/* should_ack_checkin=*/false);

  if (task_eligibility_info.has_value()) {
    *checkin_request->mutable_task_eligibility_info() = *task_eligibility_info;
  }

  return Send(&client_stream_message);
}

absl::Status GrpcFederatedProtocol::ReceiveCheckinRequestAck() {
  // Wait for a CheckinRequestAck.
  ServerStreamMessage server_stream_message;
  absl::Status receive_status = Receive(&server_stream_message);
  if (receive_status.code() == absl::StatusCode::kNotFound) {
    FCP_LOG(INFO) << "Server responded NOT_FOUND to checkin request, "
                     "population name '"
                  << population_name_ << "' is likely incorrect.";
  }
  FCP_RETURN_IF_ERROR(receive_status);

  if (!server_stream_message.has_checkin_request_ack()) {
    log_manager_->LogDiag(
        ProdDiagCode::
            BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_EXPECTED_BUT_NOT_RECVD);
    return absl::UnimplementedError(
        "Requested but did not receive CheckinRequestAck");
  }
  log_manager_->LogDiag(
      ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED);
  // Process the received CheckinRequestAck message.
  const CheckinRequestAck& checkin_request_ack =
      server_stream_message.checkin_request_ack();
  if (!checkin_request_ack.has_retry_window_if_accepted() ||
      !checkin_request_ack.has_retry_window_if_rejected()) {
    return absl::UnimplementedError(
        "Received CheckinRequestAck message with missing retry windows");
  }
  // Upon receiving the server's RetryWindows we immediately choose a concrete
  // target timestamp to retry at. This ensures that a) clients of this class
  // don't have to implement the logic to select a timestamp from a min/max
  // range themselves, b) we tell clients of this class to come back at exactly
  // a point in time the server intended us to come at (i.e. "now +
  // server_specified_retry_period", and not a point in time that is partly
  // determined by how long the remaining protocol interactions (e.g. training
  // and results upload) will take (i.e. "now +
  // duration_of_remaining_protocol_interactions +
  // server_specified_retry_period").
  checkin_request_ack_info_ = CheckinRequestAckInfo{
      .retry_info_if_rejected =
          RetryTimeAndToken{
              PickRetryTimeFromRange(
                  checkin_request_ack.retry_window_if_rejected().delay_min(),
                  checkin_request_ack.retry_window_if_rejected().delay_max(),
                  bit_gen_),
              checkin_request_ack.retry_window_if_rejected().retry_token()},
      .retry_info_if_accepted = RetryTimeAndToken{
          PickRetryTimeFromRange(
              checkin_request_ack.retry_window_if_accepted().delay_min(),
              checkin_request_ack.retry_window_if_accepted().delay_max(),
              bit_gen_),
          checkin_request_ack.retry_window_if_accepted().retry_token()}};
  return absl::OkStatus();
}

absl::StatusOr<FederatedProtocol::EligibilityEvalCheckinResult>
GrpcFederatedProtocol::ReceiveEligibilityEvalCheckinResponse(
    absl::Time start_time, std::function<void(const EligibilityEvalTask&)>
                               payload_uris_received_callback) {
  ServerStreamMessage server_stream_message;
  FCP_RETURN_IF_ERROR(Receive(&server_stream_message));

  if (!server_stream_message.has_eligibility_eval_checkin_response()) {
    return absl::UnimplementedError(
        absl::StrCat("Bad response to EligibilityEvalCheckinRequest; Expected "
                     "EligibilityEvalCheckinResponse but got ",
                     server_stream_message.kind_case(), "."));
  }

  const EligibilityEvalCheckinResponse& eligibility_checkin_response =
      server_stream_message.eligibility_eval_checkin_response();
  switch (eligibility_checkin_response.checkin_result_case()) {
    case EligibilityEvalCheckinResponse::kEligibilityEvalPayload: {
      const EligibilityEvalPayload& eligibility_eval_payload =
          eligibility_checkin_response.eligibility_eval_payload();
      object_state_ = ObjectState::kEligibilityEvalEnabled;
      EligibilityEvalTask result{.execution_id =
                                     eligibility_eval_payload.execution_id()};

      payload_uris_received_callback(result);

      PlanAndCheckpointPayloads payloads;
      if (http_client_ == nullptr ||
          !flags_->enable_grpc_with_eligibility_eval_http_resource_support()) {
        result.payloads = {
            .plan = eligibility_eval_payload.plan(),
            .checkpoint = eligibility_eval_payload.init_checkpoint()};
      } else {
        // Fetch the task resources, returning any errors that may be
        // encountered in the process.
        FCP_ASSIGN_OR_RETURN(
            result.payloads,
            FetchTaskResources(
                {.plan =
                     {
                         .has_uri =
                             eligibility_eval_payload.has_plan_resource(),
                         .uri = eligibility_eval_payload.plan_resource().uri(),
                         .data = eligibility_eval_payload.plan(),
                         .client_cache_id =
                             eligibility_eval_payload.plan_resource()
                                 .client_cache_id(),
                         .max_age = TimeUtil::ConvertProtoToAbslDuration(
                             eligibility_eval_payload.plan_resource()
                                 .max_age()),
                     },
                 .checkpoint = {
                     .has_uri = eligibility_eval_payload
                                    .has_init_checkpoint_resource(),
                     .uri = eligibility_eval_payload.init_checkpoint_resource()
                                .uri(),
                     .data = eligibility_eval_payload.init_checkpoint(),
                     .client_cache_id =
                         eligibility_eval_payload.init_checkpoint_resource()
                             .client_cache_id(),
                     .max_age = TimeUtil::ConvertProtoToAbslDuration(
                         eligibility_eval_payload.init_checkpoint_resource()
                             .max_age()),
                 }}));
      }
      return std::move(result);
    }
    case EligibilityEvalCheckinResponse::kNoEligibilityEvalConfigured: {
      // Nothing to do...
      object_state_ = ObjectState::kEligibilityEvalDisabled;
      return EligibilityEvalDisabled{};
    }
    case EligibilityEvalCheckinResponse::kRejectionInfo: {
      object_state_ = ObjectState::kEligibilityEvalCheckinRejected;
      return Rejection{};
    }
    default:
      return absl::UnimplementedError(
          "Unrecognized EligibilityEvalCheckinResponse");
  }
}

absl::StatusOr<FederatedProtocol::CheckinResult>
GrpcFederatedProtocol::ReceiveCheckinResponse(
    absl::Time start_time,
    std::function<void(const TaskAssignment&)> payload_uris_received_callback) {
  ServerStreamMessage server_stream_message;
  absl::Status receive_status = Receive(&server_stream_message);
  FCP_RETURN_IF_ERROR(receive_status);

  if (!server_stream_message.has_checkin_response()) {
    return absl::UnimplementedError(absl::StrCat(
        "Bad response to CheckinRequest; Expected CheckinResponse but got ",
        server_stream_message.kind_case(), "."));
  }

  const CheckinResponse& checkin_response =
      server_stream_message.checkin_response();

  execution_phase_id_ =
      checkin_response.has_acceptance_info()
          ? checkin_response.acceptance_info().execution_phase_id()
          : "";
  switch (checkin_response.checkin_result_case()) {
    case CheckinResponse::kAcceptanceInfo: {
      const auto& acceptance_info = checkin_response.acceptance_info();

      for (const auto& [k, v] : acceptance_info.side_channels())
        side_channels_[k] = v;
      side_channel_protocol_execution_info_ =
          acceptance_info.side_channel_protocol_execution_info();
      side_channel_protocol_options_response_ =
          checkin_response.protocol_options_response().side_channels();

      std::optional<SecAggInfo> sec_agg_info = std::nullopt;
      if (side_channel_protocol_execution_info_.has_secure_aggregation()) {
        sec_agg_info = SecAggInfo{
            .expected_number_of_clients =
                side_channel_protocol_execution_info_.secure_aggregation()
                    .expected_number_of_clients(),
            .minimum_clients_in_server_visible_aggregate =
                side_channel_protocol_execution_info_.secure_aggregation()
                    .minimum_clients_in_server_visible_aggregate()};
      }

      TaskAssignment result{
          .federated_select_uri_template =
              acceptance_info.federated_select_uri_info().uri_template(),
          .aggregation_session_id = acceptance_info.execution_phase_id(),
          .sec_agg_info = sec_agg_info};

      payload_uris_received_callback(result);

      PlanAndCheckpointPayloads payloads;
      if (http_client_ == nullptr) {
        result.payloads = {.plan = acceptance_info.plan(),
                           .checkpoint = acceptance_info.init_checkpoint()};
      } else {
        // Fetch the task resources, returning any errors that may be
        // encountered in the process.
        FCP_ASSIGN_OR_RETURN(
            result.payloads,
            FetchTaskResources(
                {.plan =
                     {
                         .has_uri = acceptance_info.has_plan_resource(),
                         .uri = acceptance_info.plan_resource().uri(),
                         .data = acceptance_info.plan(),
                         .client_cache_id =
                             acceptance_info.plan_resource().client_cache_id(),
                         .max_age = TimeUtil::ConvertProtoToAbslDuration(
                             acceptance_info.plan_resource().max_age()),
                     },
                 .checkpoint = {
                     .has_uri = acceptance_info.has_init_checkpoint_resource(),
                     .uri = acceptance_info.init_checkpoint_resource().uri(),
                     .data = acceptance_info.init_checkpoint(),
                     .client_cache_id =
                         acceptance_info.init_checkpoint_resource()
                             .client_cache_id(),
                     .max_age = TimeUtil::ConvertProtoToAbslDuration(
                         acceptance_info.init_checkpoint_resource().max_age()),
                 }}));
      }

      object_state_ = ObjectState::kCheckinAccepted;
      return result;
    }
    case CheckinResponse::kRejectionInfo: {
      object_state_ = ObjectState::kCheckinRejected;
      return Rejection{};
    }
    default:
      return absl::UnimplementedError("Unrecognized CheckinResponse");
  }
}

absl::StatusOr<FederatedProtocol::EligibilityEvalCheckinResult>
GrpcFederatedProtocol::EligibilityEvalCheckin(
    std::function<void(const EligibilityEvalTask&)>
        payload_uris_received_callback) {
  FCP_CHECK(object_state_ == ObjectState::kInitialized)
      << "Invalid call sequence";
  object_state_ = ObjectState::kEligibilityEvalCheckinFailed;

  absl::Time start_time = absl::Now();

  // Send an EligibilityEvalCheckinRequest.
  absl::Status request_status = SendEligibilityEvalCheckinRequest();
  // See note about how we handle 'permanent' errors at the top of this file.
  UpdateObjectStateIfPermanentError(
      request_status, ObjectState::kEligibilityEvalCheckinFailedPermanentError);
  FCP_RETURN_IF_ERROR(request_status);

  // Receive a CheckinRequestAck.
  absl::Status ack_status = ReceiveCheckinRequestAck();
  UpdateObjectStateIfPermanentError(
      ack_status, ObjectState::kEligibilityEvalCheckinFailedPermanentError);
  FCP_RETURN_IF_ERROR(ack_status);

  // Receive + handle an EligibilityEvalCheckinResponse message, and update the
  // object state based on the received response.
  auto response = ReceiveEligibilityEvalCheckinResponse(
      start_time, payload_uris_received_callback);
  UpdateObjectStateIfPermanentError(
      response.status(),
      ObjectState::kEligibilityEvalCheckinFailedPermanentError);
  return response;
}

// This is not supported in gRPC federated protocol, we'll do nothing.
void GrpcFederatedProtocol::ReportEligibilityEvalError(
    absl::Status error_status) {}

absl::StatusOr<FederatedProtocol::CheckinResult> GrpcFederatedProtocol::Checkin(
    const std::optional<TaskEligibilityInfo>& task_eligibility_info,
    std::function<void(const TaskAssignment&)> payload_uris_received_callback) {
  // Checkin(...) must follow an earlier call to EligibilityEvalCheckin() that
  // resulted in a CheckinResultPayload or an EligibilityEvalDisabled result.
  FCP_CHECK(object_state_ == ObjectState::kEligibilityEvalDisabled ||
            object_state_ == ObjectState::kEligibilityEvalEnabled)
      << "Checkin(...) called despite failed/rejected earlier "
         "EligibilityEvalCheckin";
  if (object_state_ == ObjectState::kEligibilityEvalEnabled) {
    FCP_CHECK(task_eligibility_info.has_value())
        << "Missing TaskEligibilityInfo despite receiving prior "
           "EligibilityEvalCheckin payload";
  } else {
    FCP_CHECK(!task_eligibility_info.has_value())
        << "Received TaskEligibilityInfo despite not receiving a prior "
           "EligibilityEvalCheckin payload";
  }

  object_state_ = ObjectState::kCheckinFailed;

  absl::Time start_time = absl::Now();
  // Send a CheckinRequest.
  absl::Status request_status = SendCheckinRequest(task_eligibility_info);
  // See note about how we handle 'permanent' errors at the top of this file.
  UpdateObjectStateIfPermanentError(request_status,
                                    ObjectState::kCheckinFailedPermanentError);
  FCP_RETURN_IF_ERROR(request_status);

  // Receive + handle a CheckinResponse message, and update the object state
  // based on the received response.
  auto response =
      ReceiveCheckinResponse(start_time, payload_uris_received_callback);
  UpdateObjectStateIfPermanentError(response.status(),
                                    ObjectState::kCheckinFailedPermanentError);
  return response;
}

absl::StatusOr<FederatedProtocol::MultipleTaskAssignments>
GrpcFederatedProtocol::PerformMultipleTaskAssignments(
    const std::vector<std::string>& task_names) {
  return absl::UnimplementedError(
      "PerformMultipleTaskAssignments is not supported by "
      "GrpcFederatedProtocol.");
}

absl::Status GrpcFederatedProtocol::ReportCompleted(
    ComputationResults results, absl::Duration plan_duration,
    std::optional<std::string> aggregation_session_id) {
  FCP_LOG(INFO) << "Reporting outcome: " << static_cast<int>(engine::COMPLETED);
  FCP_CHECK(object_state_ == ObjectState::kCheckinAccepted)
      << "Invalid call sequence";
  object_state_ = ObjectState::kReportCalled;
  auto response = Report(std::move(results), engine::COMPLETED, plan_duration);
  // See note about how we handle 'permanent' errors at the top of this file.
  UpdateObjectStateIfPermanentError(response,
                                    ObjectState::kReportFailedPermanentError);
  return response;
}

absl::Status GrpcFederatedProtocol::ReportNotCompleted(
    engine::PhaseOutcome phase_outcome, absl::Duration plan_duration,
    std::optional<std::string> aggregation_session_Id) {
  FCP_LOG(WARNING) << "Reporting outcome: " << static_cast<int>(phase_outcome);
  FCP_CHECK(object_state_ == ObjectState::kCheckinAccepted)
      << "Invalid call sequence";
  object_state_ = ObjectState::kReportCalled;
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", "");
  auto response = Report(std::move(results), phase_outcome, plan_duration);
  // See note about how we handle 'permanent' errors at the top of this file.
  UpdateObjectStateIfPermanentError(response,
                                    ObjectState::kReportFailedPermanentError);
  return response;
}

class GrpcSecAggSendToServerImpl : public SecAggSendToServerBase {
 public:
  GrpcSecAggSendToServerImpl(
      GrpcBidiStreamInterface* grpc_bidi_stream,
      const std::function<absl::Status(ClientToServerWrapperMessage*)>&
          report_func)
      : grpc_bidi_stream_(grpc_bidi_stream), report_func_(report_func) {}
  ~GrpcSecAggSendToServerImpl() override = default;

  void Send(ClientToServerWrapperMessage* message) override {
    // The commit message (MaskedInputRequest) must be piggy-backed onto the
    // ReportRequest message, the logic for which is encapsulated in
    // report_func_ so that it may be held in common between both accumulation
    // methods.
    if (message->message_content_case() ==
        ClientToServerWrapperMessage::MessageContentCase::
            kMaskedInputResponse) {
      auto status = report_func_(message);
      if (!status.ok())
        FCP_LOG(ERROR) << "Could not send ReportRequest: " << status;
      return;
    }
    ClientStreamMessage client_stream_message;
    client_stream_message.mutable_secure_aggregation_client_message()->Swap(
        message);
    auto bytes_to_upload = client_stream_message.ByteSizeLong();
    auto status = grpc_bidi_stream_->Send(&client_stream_message);
    if (status.ok()) {
      last_sent_message_size_ = bytes_to_upload;
    }
  }

 private:
  GrpcBidiStreamInterface* grpc_bidi_stream_;
  // SecAgg's output must be wrapped in a ReportRequest; because the report
  // logic is mostly generic, this lambda allows it to be shared between
  // aggregation types.
  const std::function<absl::Status(ClientToServerWrapperMessage*)>&
      report_func_;
};

class GrpcSecAggProtocolDelegate : public SecAggProtocolDelegate {
 public:
  GrpcSecAggProtocolDelegate(
      absl::flat_hash_map<std::string, SideChannelExecutionInfo> side_channels,
      GrpcBidiStreamInterface* grpc_bidi_stream)
      : side_channels_(std::move(side_channels)),
        grpc_bidi_stream_(grpc_bidi_stream) {}

  absl::StatusOr<uint64_t> GetModulus(const std::string& key) override {
    auto execution_info = side_channels_.find(key);
    if (execution_info == side_channels_.end())
      return absl::InternalError(
          absl::StrCat("Execution not found for aggregand: ", key));
    uint64_t modulus;
    auto secure_aggregand = execution_info->second.secure_aggregand();
    // TODO(team): Delete output_bitwidth support once
    // modulus is fully rolled out.
    if (secure_aggregand.modulus() > 0) {
      modulus = secure_aggregand.modulus();
    } else {
      // Note: we ignore vector.get_bitwidth() here, because (1)
      // it is only an upper bound on the *input* bitwidth,
      // based on the Tensorflow dtype, but (2) we have exact
      // *output* bitwidth information from the execution_info,
      // and that is what SecAgg needs.
      modulus = 1ULL << secure_aggregand.output_bitwidth();
    }
    return modulus;
  }

  absl::StatusOr<secagg::ServerToClientWrapperMessage> ReceiveServerMessage()
      override {
    ServerStreamMessage server_stream_message;
    absl::Status receive_status =
        grpc_bidi_stream_->Receive(&server_stream_message);
    if (!receive_status.ok()) {
      return absl::Status(receive_status.code(),
                          absl::StrCat("Error during SecAgg receive: ",
                                       receive_status.message()));
    }
    last_received_message_size_ = server_stream_message.ByteSizeLong();
    if (!server_stream_message.has_secure_aggregation_server_message()) {
      return absl::InternalError(
          absl::StrCat("Bad response to SecAgg protocol; Expected "
                       "ServerToClientWrapperMessage but got ",
                       server_stream_message.kind_case(), "."));
    }
    return server_stream_message.secure_aggregation_server_message();
  }

  void Abort() override { grpc_bidi_stream_->Close(); }
  size_t last_received_message_size() override {
    return last_received_message_size_;
  };

 private:
  absl::flat_hash_map<std::string, SideChannelExecutionInfo> side_channels_;
  GrpcBidiStreamInterface* grpc_bidi_stream_;
  size_t last_received_message_size_;
};

absl::Status GrpcFederatedProtocol::ReportInternal(
    std::string tf_checkpoint, engine::PhaseOutcome phase_outcome,
    absl::Duration plan_duration,
    ClientToServerWrapperMessage* secagg_commit_message) {
  ClientStreamMessage client_stream_message;
  auto report_request = client_stream_message.mutable_report_request();
  report_request->set_population_name(population_name_);
  report_request->set_execution_phase_id(execution_phase_id_);
  auto report = report_request->mutable_report();

  // 1. Include TF checkpoint and/or SecAgg commit message.
  report->set_update_checkpoint(std::move(tf_checkpoint));
  if (secagg_commit_message) {
    client_stream_message.mutable_secure_aggregation_client_message()->Swap(
        secagg_commit_message);
  }

  // 2. Include outcome of computation.
  report->set_status_code(phase_outcome == engine::COMPLETED
                              ? google::rpc::OK
                              : google::rpc::INTERNAL);

  // 3. Include client execution statistics, if any.
  ClientExecutionStats client_execution_stats;
  client_execution_stats.mutable_duration()->set_seconds(
      absl::IDivDuration(plan_duration, absl::Seconds(1), &plan_duration));
  client_execution_stats.mutable_duration()->set_nanos(static_cast<int32_t>(
      absl::IDivDuration(plan_duration, absl::Nanoseconds(1), &plan_duration)));
  report->add_serialized_train_event()->PackFrom(client_execution_stats);

  // 4. Send ReportRequest.

  // Note that we do not use the GrpcFederatedProtocol::Send(...) helper method
  // here, since we are already running within a call to
  // InterruptibleRunner::Run.
  const auto status = this->grpc_bidi_stream_->Send(&client_stream_message);
  if (!status.ok()) {
    return absl::Status(
        status.code(),
        absl::StrCat("Error sending ReportRequest: ", status.message()));
  }

  return absl::OkStatus();
}

absl::Status GrpcFederatedProtocol::Report(ComputationResults results,
                                           engine::PhaseOutcome phase_outcome,
                                           absl::Duration plan_duration) {
  std::string tf_checkpoint;
  bool has_checkpoint;
  for (auto& [k, v] : results) {
    if (std::holds_alternative<TFCheckpoint>(v)) {
      tf_checkpoint = std::get<TFCheckpoint>(std::move(v));
      has_checkpoint = true;
      break;
    }
  }

  // This lambda allows for convenient reporting from within SecAgg's
  // SendToServerInterface::Send().
  std::function<absl::Status(ClientToServerWrapperMessage*)> report_lambda =
      [&](ClientToServerWrapperMessage* secagg_commit_message) -> absl::Status {
    return ReportInternal(std::move(tf_checkpoint), phase_outcome,
                          plan_duration, secagg_commit_message);
  };

  // Run the Secure Aggregation protocol, if necessary.
  if (side_channel_protocol_execution_info_.has_secure_aggregation()) {
    auto secure_aggregation_protocol_execution_info =
        side_channel_protocol_execution_info_.secure_aggregation();
    auto expected_number_of_clients =
        secure_aggregation_protocol_execution_info.expected_number_of_clients();

    FCP_LOG(INFO) << "Reporting via Secure Aggregation";
    if (phase_outcome != engine::COMPLETED)
      return absl::InternalError(
          "Aborting the SecAgg protocol (no update was produced).");

    if (side_channel_protocol_options_response_.secure_aggregation()
            .client_variant() != secagg::SECAGG_CLIENT_VARIANT_NATIVE_V1) {
      log_manager_->LogDiag(
          ProdDiagCode::SECAGG_CLIENT_ERROR_UNSUPPORTED_VERSION);
      return absl::InternalError(absl::StrCat(
          "Unsupported SecAgg client variant: ",
          side_channel_protocol_options_response_.secure_aggregation()
              .client_variant()));
    }

    auto send_to_server_impl = std::make_unique<GrpcSecAggSendToServerImpl>(
        grpc_bidi_stream_.get(), report_lambda);
    auto secagg_event_publisher = event_publisher_->secagg_event_publisher();
    FCP_CHECK(secagg_event_publisher)
        << "An implementation of "
        << "SecAggEventPublisher must be provided.";
    auto delegate = std::make_unique<GrpcSecAggProtocolDelegate>(
        side_channels_, grpc_bidi_stream_.get());
    std::unique_ptr<SecAggRunner> secagg_runner =
        secagg_runner_factory_->CreateSecAggRunner(
            std::move(send_to_server_impl), std::move(delegate),
            secagg_event_publisher, log_manager_, interruptible_runner_.get(),
            expected_number_of_clients,
            secure_aggregation_protocol_execution_info
                .minimum_surviving_clients_for_reconstruction());

    FCP_RETURN_IF_ERROR(secagg_runner->Run(std::move(results)));
  } else {
    // Report without secure aggregation.
    FCP_LOG(INFO) << "Reporting via Simple Aggregation";
    if (results.size() != 1 || !has_checkpoint) {
      return absl::InternalError(
          "Simple Aggregation aggregands have unexpected format.");
    }
    FCP_RETURN_IF_ERROR(interruptible_runner_->Run(
        [&report_lambda]() { return report_lambda(nullptr); },
        [this]() {
          // What about event_publisher_ and log_manager_?
          this->grpc_bidi_stream_->Close();
        }));
  }

  FCP_LOG(INFO) << "Finished reporting.";

  // Receive ReportResponse.
  ServerStreamMessage server_stream_message;
  absl::Status receive_status = Receive(&server_stream_message);
  if (receive_status.code() == absl::StatusCode::kAborted) {
    FCP_LOG(INFO) << "Server responded ABORTED.";
  } else if (receive_status.code() == absl::StatusCode::kCancelled) {
    FCP_LOG(INFO) << "Upload was cancelled by the client.";
  }
  if (!receive_status.ok()) {
    return absl::Status(
        receive_status.code(),
        absl::StrCat("Error after ReportRequest: ", receive_status.message()));
  }
  if (!server_stream_message.has_report_response()) {
    return absl::UnimplementedError(absl::StrCat(
        "Bad response to ReportRequest; Expected REPORT_RESPONSE but got ",
        server_stream_message.kind_case(), "."));
  }
  return absl::OkStatus();
}

RetryWindow GrpcFederatedProtocol::GetLatestRetryWindow() {
  // We explicitly enumerate all possible states here rather than using
  // "default", to ensure that when new states are added later on, the author
  // is forced to update this method and consider which is the correct
  // RetryWindow to return.
  switch (object_state_) {
    case ObjectState::kCheckinAccepted:
    case ObjectState::kReportCalled:
      // If a client makes it past the 'checkin acceptance' stage, we use the
      // 'accepted' RetryWindow unconditionally (unless a permanent error is
      // encountered). This includes cases where the checkin is accepted, but
      // the report request results in a (transient) error.
      FCP_CHECK(checkin_request_ack_info_.has_value());
      return GenerateRetryWindowFromRetryTimeAndToken(
          checkin_request_ack_info_->retry_info_if_accepted);
    case ObjectState::kEligibilityEvalCheckinRejected:
    case ObjectState::kEligibilityEvalDisabled:
    case ObjectState::kEligibilityEvalEnabled:
    case ObjectState::kCheckinRejected:
      FCP_CHECK(checkin_request_ack_info_.has_value());
      return GenerateRetryWindowFromRetryTimeAndToken(
          checkin_request_ack_info_->retry_info_if_rejected);
    case ObjectState::kInitialized:
    case ObjectState::kEligibilityEvalCheckinFailed:
    case ObjectState::kCheckinFailed:
      // If the flag is true, then we use the previously chosen absolute retry
      // time instead (if available).
      if (checkin_request_ack_info_.has_value()) {
        // If we already received a server-provided retry window, then use it.
        return GenerateRetryWindowFromRetryTimeAndToken(
            checkin_request_ack_info_->retry_info_if_rejected);
      }
      // Otherwise, we generate a retry window using the flag-provided transient
      // error retry period.
      return GenerateRetryWindowFromTargetDelay(
          absl::Seconds(
              flags_->federated_training_transient_errors_retry_delay_secs()),
          // NOLINTBEGIN(whitespace/line_length)
          flags_
              ->federated_training_transient_errors_retry_delay_jitter_percent(),
          // NOLINTEND
          bit_gen_);
    case ObjectState::kEligibilityEvalCheckinFailedPermanentError:
    case ObjectState::kCheckinFailedPermanentError:
    case ObjectState::kReportFailedPermanentError:
      // If we encountered a permanent error during the eligibility eval or
      // regular checkins, then we use the Flags-configured 'permanent error'
      // retry period. Note that we do so regardless of whether the server had,
      // by the time the permanent error was received, already returned a
      // CheckinRequestAck containing a set of retry windows. See note on error
      // handling at the top of this file.
      return GenerateRetryWindowFromTargetDelay(
          absl::Seconds(
              flags_->federated_training_permanent_errors_retry_delay_secs()),
          // NOLINTBEGIN(whitespace/line_length)
          flags_
              ->federated_training_permanent_errors_retry_delay_jitter_percent(),
          // NOLINTEND
          bit_gen_);
    case ObjectState::kMultipleTaskAssignmentsAccepted:
    case ObjectState::kMultipleTaskAssignmentsFailed:
    case ObjectState::kMultipleTaskAssignmentsFailedPermanentError:
    case ObjectState::kMultipleTaskAssignmentsNoAvailableTask:
    case ObjectState::kReportMultipleTaskPartialError:
      FCP_LOG(FATAL) << "Multi-task assignments is not supported by gRPC.";
      RetryWindow retry_window;
      return retry_window;
  }
}

// Converts the given RetryTimeAndToken to a zero-width RetryWindow (where
// delay_min and delay_max are set to the same value), by converting the target
// retry time to a delay relative to the current timestamp.
RetryWindow GrpcFederatedProtocol::GenerateRetryWindowFromRetryTimeAndToken(
    const GrpcFederatedProtocol::RetryTimeAndToken& retry_info) {
  // Generate a RetryWindow with delay_min and delay_max both set to the same
  // value.
  RetryWindow retry_window =
      GenerateRetryWindowFromRetryTime(retry_info.retry_time);
  retry_window.set_retry_token(retry_info.retry_token);
  return retry_window;
}

void GrpcFederatedProtocol::UpdateObjectStateIfPermanentError(
    absl::Status status,
    GrpcFederatedProtocol::ObjectState permanent_error_object_state) {
  if (federated_training_permanent_error_codes_.contains(
          static_cast<int32_t>(status.code()))) {
    object_state_ = permanent_error_object_state;
  }
}

absl::StatusOr<FederatedProtocol::PlanAndCheckpointPayloads>
GrpcFederatedProtocol::FetchTaskResources(
    GrpcFederatedProtocol::TaskResources task_resources) {
  FCP_ASSIGN_OR_RETURN(UriOrInlineData plan_uri_or_data,
                       ConvertResourceToUriOrInlineData(task_resources.plan));
  FCP_ASSIGN_OR_RETURN(
      UriOrInlineData checkpoint_uri_or_data,
      ConvertResourceToUriOrInlineData(task_resources.checkpoint));

  // Log a diag code if either resource is about to be downloaded via HTTP.
  if (!plan_uri_or_data.uri().uri.empty() ||
      !checkpoint_uri_or_data.uri().uri.empty()) {
    log_manager_->LogDiag(
        ProdDiagCode::HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_USES_HTTP);
  }

  // Fetch the plan and init checkpoint resources if they need to be fetched
  // (using the inline data instead if available).
  absl::StatusOr<
      std::vector<absl::StatusOr<::fcp::client::http::InMemoryHttpResponse>>>
      resource_responses;
  {
    auto started_stopwatch = network_stopwatch_->Start();
    resource_responses = ::fcp::client::http::FetchResourcesInMemory(
        *http_client_, *interruptible_runner_,
        {plan_uri_or_data, checkpoint_uri_or_data}, &http_bytes_downloaded_,
        &http_bytes_uploaded_, resource_cache_);
  }
  if (!resource_responses.ok()) {
    log_manager_->LogDiag(
        ProdDiagCode::
            HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_HTTP_FETCH_FAILED);
    return resource_responses.status();
  }
  auto& plan_data_response = (*resource_responses)[0];
  auto& checkpoint_data_response = (*resource_responses)[1];

  if (!plan_data_response.ok() || !checkpoint_data_response.ok()) {
    log_manager_->LogDiag(
        ProdDiagCode::
            HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_HTTP_FETCH_FAILED);
  }
  // Note: we forward any error during the fetching of the plan/checkpoint
  // resources resources to the caller, which means that these error codes
  // will be checked against the set of 'permanent' error codes, just like the
  // errors in response to the protocol request are.
  if (!plan_data_response.ok()) {
    return absl::Status(plan_data_response.status().code(),
                        absl::StrCat("plan fetch failed: ",
                                     plan_data_response.status().ToString()));
  }
  if (!checkpoint_data_response.ok()) {
    return absl::Status(
        checkpoint_data_response.status().code(),
        absl::StrCat("checkpoint fetch failed: ",
                     checkpoint_data_response.status().ToString()));
  }
  if (!plan_uri_or_data.uri().uri.empty() ||
      !checkpoint_uri_or_data.uri().uri.empty()) {
    // We only want to log this diag code when we actually did fetch something
    // via HTTP.
    log_manager_->LogDiag(
        ProdDiagCode::
            HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_HTTP_FETCH_SUCCEEDED);
  }

  return PlanAndCheckpointPayloads{plan_data_response->body,
                                   checkpoint_data_response->body};
}

// Convert a Resource proto into a UriOrInlineData object. Returns an
// `INVALID_ARGUMENT` error if the given `Resource` has the `uri` field set to
// an empty value, or an `UNIMPLEMENTED` error if the `Resource` has an unknown
// field set.
absl::StatusOr<UriOrInlineData>
GrpcFederatedProtocol::ConvertResourceToUriOrInlineData(
    const GrpcFederatedProtocol::TaskResource& resource) {
  // We need to support 3 states:
  // - Inline data is available.
  // - No inline data nor is there a URI. This should be treated as there being
  //   an 'empty' inline data.
  // - No inline data is available but a URI is available.
  if (!resource.has_uri) {
    // If the URI field wasn't set, then we'll just use the inline data field
    // (which will either be set or be empty).
    //
    // Note: this copies the data into the new absl::Cord. However, this Cord is
    // then passed around all the way to fl_runner.cc without copying its data,
    // so this is ultimately approx. as efficient as the non-HTTP resource code
    // path where we also make a copy of the protobuf string into a new string
    // which is then returned.
    return UriOrInlineData::CreateInlineData(
        absl::Cord(resource.data),
        UriOrInlineData::InlineData::CompressionFormat::kUncompressed);
  }
  if (resource.uri.empty()) {
    return absl::InvalidArgumentError(
        "Resource uri must be non-empty when set");
  }
  return UriOrInlineData::CreateUri(resource.uri, resource.client_cache_id,
                                    resource.max_age);
}

NetworkStats GrpcFederatedProtocol::GetNetworkStats() {
  // Note: the `HttpClient` bandwidth stats are similar to the gRPC protocol's
  // "chunking layer" stats, in that they reflect as closely as possible the
  // amount of data sent on the wire.
  return {.bytes_downloaded = grpc_bidi_stream_->ChunkingLayerBytesReceived() +
                              http_bytes_downloaded_,
          .bytes_uploaded = grpc_bidi_stream_->ChunkingLayerBytesSent() +
                            http_bytes_uploaded_,
          .network_duration = network_stopwatch_->GetTotalDuration()};
}

}  // namespace client
}  // namespace fcp
