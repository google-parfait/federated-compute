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
#include "fcp/client/federated_protocol.h"

#include <algorithm>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/grpc_bidi_stream.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/secagg_event_publisher.h"
#include "fcp/client/task_environment.h"
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

using ::fcp::client::opstats::OperationalStats;
using ::fcp::client::opstats::OpStatsLogger;
using ::fcp::secagg::AesCtrPrngFactory;
using ::fcp::secagg::ClientState;
using ::fcp::secagg::ClientToServerWrapperMessage;
using ::fcp::secagg::CryptoRandPrng;
using ::fcp::secagg::InputVectorSpecification;
using ::fcp::secagg::SecAggClient;
using ::fcp::secagg::SecAggVector;
using ::fcp::secagg::SecAggVectorMap;
using ::fcp::secagg::SendToServerInterface;
using ::fcp::secagg::StateTransitionListenerInterface;
using ::google::internal::federated::plan::ClientOnlyPlan;
using ::google::internal::federatedml::v2::CheckinRequest;
using ::google::internal::federatedml::v2::CheckinRequestAck;
using ::google::internal::federatedml::v2::CheckinResponse;
using ::google::internal::federatedml::v2::ClientExecutionStats;
using ::google::internal::federatedml::v2::ClientStreamMessage;
using ::google::internal::federatedml::v2::EligibilityEvalCheckinRequest;
using ::google::internal::federatedml::v2::EligibilityEvalCheckinResponse;
using ::google::internal::federatedml::v2::EligibilityEvalPayload;
using ::google::internal::federatedml::v2::ProtocolOptionsRequest;
using ::google::internal::federatedml::v2::RetryWindow;
using ::google::internal::federatedml::v2::ServerStreamMessage;
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
// Finally, note that for simplicity sake we generally check whether a
// permanent error was received at the level of this class's public method,
// rather than deeper down in each of our helper methods that actually call
// directly into the gRPC stack. This keeps our state-managing code simpler, but
// does mean that if any of our helper methods like SendCheckinRequest produce a
// permanent error code locally (i.e. without it being sent by the server), it
// will be treated as if the server sent it and the permanent error retry period
// will be used. We consider this a reasonable tradeoff.

namespace {

// Takes the given minimum and maximum delays, and uniformly randomly
// chooses a delay in that range.
absl::Duration PickRetryDelayFromRange(absl::Duration min_delay,
                                       absl::Duration max_delay,
                                       absl::BitGen& bit_gen) {
  // Sanitize inputs (ensure min_delay is >= 0, and max_delay is >= min_delay).
  min_delay =
      min_delay >= absl::ZeroDuration() ? min_delay : absl::ZeroDuration();
  max_delay = max_delay >= min_delay ? max_delay : min_delay;

  // Pick a value.
  absl::Duration window_width = max_delay - min_delay;
  double random = absl::Uniform(bit_gen, 0, 1.0);
  return min_delay + (window_width * random);
}

// Converts an absl::Duration to a google::protobuf::Duration.
// Note that we assume the duration's we deal with here are representable by
// both formats.
google::protobuf::Duration ConvertAbslToProtoDuration(
    absl::Duration absl_duration) {
  google::protobuf::Duration proto_duration;
  proto_duration.set_seconds(int32_t(
      absl::IDivDuration(absl_duration, absl::Seconds(1), &absl_duration)));
  proto_duration.set_nanos(int32_t(
      absl::IDivDuration(absl_duration, absl::Nanoseconds(1), &absl_duration)));
  return proto_duration;
}

// Picks a retry delay and encodes it as a zero-width RetryWindow (where
// delay_min and delay_max are set to the same value), from a given target delay
// and a configured amount of jitter.
RetryWindow GenerateRetryWindowFromTargetDelay(absl::Duration target_delay,
                                               double jitter_percent,
                                               absl::BitGen& bit_gen) {
  // Sanitize the jitter_percent input, ensuring it's within [0.0 and 1.0]
  jitter_percent = std::min(1.0, std::max(0.0, jitter_percent));
  // Pick a retry delay from the target range.
  absl::Duration retry_delay =
      PickRetryDelayFromRange(target_delay * (1.0 - jitter_percent),
                              target_delay * (1.0 + jitter_percent), bit_gen);
  // Generate a RetryWindow with delay_min and delay_max both set to the same
  // value.
  RetryWindow result;
  *result.mutable_delay_min() = *result.mutable_delay_max() =
      ConvertAbslToProtoDuration(retry_delay);
  return result;
}

// Picks an absolute retry time by picking a retry delay from the range
// specified by the RetryWindow, and then adding it to the current timestamp.
absl::Time PickRetryTimeFromWindow(RetryWindow retry_window,
                                   absl::BitGen& bit_gen) {
  return absl::Now() +
         PickRetryDelayFromRange(
             absl::Seconds(retry_window.delay_min().seconds()) +
                 absl::Nanoseconds(retry_window.delay_min().nanos()),
             absl::Seconds(retry_window.delay_max().seconds()) +
                 absl::Nanoseconds(retry_window.delay_max().nanos()),
             bit_gen);
}
}  // anonymous namespace

FederatedProtocol::FederatedProtocol(
    EventPublisher* event_publisher, LogManager* log_manager,
    ::fcp::client::opstats::OpStatsLogger* opstats_logger, const Flags* flags,
    const std::string& federated_service_uri, const std::string& api_key,
    const std::string& test_cert_path, absl::string_view population_name,
    absl::string_view retry_token, absl::string_view client_version,
    absl::string_view attestation_measurement,
    std::function<bool()> should_abort,
    const InterruptibleRunner::TimingConfig& timing_config,
    const int64_t grpc_channel_deadline_seconds)
    : FederatedProtocol(
          event_publisher, log_manager, opstats_logger, flags,
          absl::make_unique<GrpcBidiStream>(
              federated_service_uri, api_key, std::string(population_name),
              grpc_channel_deadline_seconds, test_cert_path),
          nullptr, population_name, retry_token, client_version,
          attestation_measurement, should_abort, absl::BitGen(),
          timing_config) {}

FederatedProtocol::FederatedProtocol(
    EventPublisher* event_publisher, LogManager* log_manager,
    OpStatsLogger* opstats_logger, const Flags* flags,
    std::unique_ptr<GrpcBidiStreamInterface> grpc_bidi_stream,
    std::unique_ptr<SecAggClient> secagg_client,
    absl::string_view population_name, absl::string_view retry_token,
    absl::string_view client_version, absl::string_view attestation_measurement,
    std::function<bool()> should_abort, absl::BitGen bit_gen,
    const InterruptibleRunner::TimingConfig& timing_config)
    : object_state_(ObjectState::kInitialized),
      event_publisher_(event_publisher),
      log_manager_(log_manager),
      opstats_logger_(opstats_logger),
      flags_(flags),
      grpc_bidi_stream_(std::move(grpc_bidi_stream)),
      secagg_client_(std::move(secagg_client)),
      population_name_(population_name),
      retry_token_(retry_token),
      client_version_(client_version),
      attestation_measurement_(attestation_measurement),
      bit_gen_(std::move(bit_gen)),
      federated_training_use_new_retry_delay_behavior_(
          flags_->federated_training_use_new_retry_delay_behavior()) {
  interruptible_runner_ = absl::make_unique<InterruptibleRunner>(
      log_manager, should_abort, timing_config,
      InterruptibleRunner::DiagnosticsConfig{
          .interrupted = ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_GRPC,
          .interrupt_timeout =
              ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_GRPC_TIMED_OUT,
          .interrupted_extended = ProdDiagCode::
              BACKGROUND_TRAINING_INTERRUPT_GRPC_EXTENDED_COMPLETED,
          .interrupt_timeout_extended = ProdDiagCode::
              BACKGROUND_TRAINING_INTERRUPT_GRPC_EXTENDED_TIMED_OUT});
  if (flags->federated_training_use_new_retry_delay_behavior()) {
    // Note that if this flag is false then the
    // federated_training_permanent_error_codes_ set will be empty, which
    // effectively prevents us from ever entering the newly added permanent
    // error ObjectStates.
    //
    // Note that we could cast the provided error codes to absl::StatusCode
    // values here. However, that means we'd have to handle the case when
    // invalid integers that don't map to a StatusCode enum are provided in the
    // flag here. Instead, we cast absl::StatusCodes to int32_t each time we
    // compare them with the flag-provided list of codes, which means we never
    // have to worry about invalid flag values.
    const std::vector<int32_t>& error_codes =
        flags->federated_training_permanent_error_codes();
    federated_training_permanent_error_codes_ =
        absl::flat_hash_set<int32_t>(error_codes.begin(), error_codes.end());
  }
}

FederatedProtocol::~FederatedProtocol() { grpc_bidi_stream_->Close(); }

absl::Status FederatedProtocol::Send(
    google::internal::federatedml::v2::ClientStreamMessage*
        client_stream_message) {
  FCP_RETURN_IF_ERROR(interruptible_runner_->Run(
      [this, &client_stream_message]() {
        return this->grpc_bidi_stream_->Send(client_stream_message);
      },
      [this]() { this->grpc_bidi_stream_->Close(); }));
  bytes_uploaded_ += client_stream_message->ByteSizeLong();
  UpdateOpStatsNetworkStats();
  return absl::OkStatus();
}

absl::Status FederatedProtocol::Receive(
    google::internal::federatedml::v2::ServerStreamMessage*
        server_stream_message) {
  FCP_RETURN_IF_ERROR(interruptible_runner_->Run(
      [this, &server_stream_message]() {
        return grpc_bidi_stream_->Receive(server_stream_message);
      },
      [this]() { this->grpc_bidi_stream_->Close(); }));
  bytes_downloaded_ += server_stream_message->ByteSizeLong();
  UpdateOpStatsNetworkStats();
  return absl::OkStatus();
}

ProtocolOptionsRequest FederatedProtocol::CreateProtocolOptionsRequest(
    bool should_ack_checkin) const {
  ProtocolOptionsRequest request;
  request.set_should_ack_checkin(should_ack_checkin);

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
  return request;
}

absl::Status FederatedProtocol::SendEligibilityEvalCheckinRequest() {
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

  // Log that we are about to check in with the server.
  event_publisher_->PublishEligibilityEvalCheckin();
  opstats_logger_->AddEvent(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED);

  return Send(&client_stream_message);
}

absl::Status FederatedProtocol::SendCheckinRequest(
    const absl::optional<TaskEligibilityInfo>& task_eligibility_info,
    bool should_ack_checkin) {
  ClientStreamMessage client_stream_message;
  CheckinRequest* checkin_request =
      client_stream_message.mutable_checkin_request();
  checkin_request->set_population_name(population_name_);
  checkin_request->set_retry_token(retry_token_);
  checkin_request->set_client_version(client_version_);
  checkin_request->set_attestation_measurement(attestation_measurement_);
  *checkin_request->mutable_protocol_options_request() =
      CreateProtocolOptionsRequest(should_ack_checkin);

  if (task_eligibility_info.has_value()) {
    *checkin_request->mutable_task_eligibility_info() = *task_eligibility_info;
  }

  // Log that we are about to check in with the server.
  event_publisher_->PublishCheckin();
  opstats_logger_->AddEvent(
      OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED);

  return Send(&client_stream_message);
}

absl::Status FederatedProtocol::ReceiveCheckinRequestAck() {
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
  retry_window_if_rejected_ = checkin_request_ack.retry_window_if_rejected();
  retry_window_if_accepted_ = checkin_request_ack.retry_window_if_accepted();
  if (federated_training_use_new_retry_delay_behavior_) {
    // If the flag is on, then upon receiving the server's RetryWindows we
    // immediately choose a concrete target timestamp to retry at. This ensures
    // that a) clients of this class don't have to implement the logic to select
    // a timestamp from a min/max range themselves, b) we tell clients of this
    // class to come back at exactly a point in time the server intended us to
    // come at (i.e. "now + server_specified_retry_period", and not a point in
    // time that is partly determined by how long the remaining protocol
    // interactions (e.g. training and results upload) will take (i.e. "now +
    // duration_of_remaining_protocol_interactions +
    // server_specified_retry_period").
    checkin_request_ack_info_ = CheckinRequestAckInfo{
        .retry_info_if_rejected =
            RetryTimeAndToken{
                PickRetryTimeFromWindow(
                    checkin_request_ack.retry_window_if_rejected(), bit_gen_),
                checkin_request_ack.retry_window_if_rejected().retry_token()},
        .retry_info_if_accepted = RetryTimeAndToken{
            PickRetryTimeFromWindow(
                checkin_request_ack.retry_window_if_accepted(), bit_gen_),
            checkin_request_ack.retry_window_if_accepted().retry_token()}};
  }
  return absl::OkStatus();
}

absl::StatusOr<FederatedProtocol::EligibilityEvalCheckinResult>
FederatedProtocol::ReceiveEligibilityEvalCheckinResponse(
    absl::Time start_time) {
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

  absl::Duration download_duration = absl::Now() - start_time;

  switch (eligibility_checkin_response.checkin_result_case()) {
    case EligibilityEvalCheckinResponse::kEligibilityEvalPayload: {
      const EligibilityEvalPayload& eligibility_eval_payload =
          eligibility_checkin_response.eligibility_eval_payload();
      std::string model_identifier = eligibility_eval_payload.execution_id();
      // If the server didn't yet actually populate the execution_id field, then
      // we still set a placeholder value to ensure we can distinguish events
      // for the eligibility eval task from events for regular tasks.
      if (model_identifier.empty()) {
        model_identifier = absl::StrCat(population_name_,
                                        "/eligibility_eval_task_placeholder");
      }
      log_manager_->SetModelIdentifier(model_identifier);
      event_publisher_->SetModelIdentifier(model_identifier);
      event_publisher_->PublishEligibilityEvalPlanReceived(
          bytes_downloaded_, grpc_bidi_stream_->ChunkingLayerBytesReceived(),
          download_duration);

      ClientOnlyPlan plan;
      if (!plan.ParseFromString(eligibility_eval_payload.plan())) {
        log_manager_->LogDiag(
            ProdDiagCode::
                BACKGROUND_TRAINING_ELIGIBILITY_EVAL_FAILED_CANNOT_PARSE_PLAN);
        // We use InternalError here, rather than the perhaps more appropriate
        // InvalidArgumentError, because we want to distinguish this rare, and
        // likely transient/temporary issue (e.g. a memory corruption, or an
        // invalid payload that is temporarily being served by the server), from
        // other error cases where the server indicates an InvalidArgumentError
        // to the client (which would indicate a more serious, and likely more
        // permanent bug in the client-side protocol implementation).
        return absl::InternalError(
            "Could not parse received eligibility eval plan");
      }
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_ELIGIBILITY_ENABLED);
      object_state_ = ObjectState::kEligibilityEvalEnabled;
      return CheckinResultPayload{
          plan, eligibility_eval_payload.init_checkpoint(), model_identifier};
    }
    case EligibilityEvalCheckinResponse::kNoEligibilityEvalConfigured: {
      // Nothing to do...
      event_publisher_->PublishEligibilityEvalNotConfigured(
          bytes_downloaded_, grpc_bidi_stream_->ChunkingLayerBytesReceived(),
          download_duration);
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_ELIGIBILITY_DISABLED);
      object_state_ = ObjectState::kEligibilityEvalDisabled;
      return EligibilityEvalDisabled{};
    }
    case EligibilityEvalCheckinResponse::kRejectionInfo: {
      event_publisher_->PublishEligibilityEvalRejected(
          bytes_downloaded_, grpc_bidi_stream_->ChunkingLayerBytesReceived(),
          download_duration);
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED);
      object_state_ = ObjectState::kEligibilityEvalCheckinRejected;
      return Rejection{};
    }
    default:
      return absl::UnimplementedError(
          "Unrecognized EligibilityEvalCheckinResponse");
  }
}

absl::StatusOr<FederatedProtocol::CheckinResult>
FederatedProtocol::ReceiveCheckinResponse(absl::Time start_time) {
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

  absl::Duration download_duration = absl::Now() - start_time;
  execution_phase_id_ =
      checkin_response.has_acceptance_info()
          ? checkin_response.acceptance_info().execution_phase_id()
          : "";
  log_manager_->SetModelIdentifier(execution_phase_id_);
  event_publisher_->SetModelIdentifier(execution_phase_id_);
  event_publisher_->PublishCheckinFinished(
      bytes_downloaded_, grpc_bidi_stream_->ChunkingLayerBytesReceived(),
      download_duration);
  switch (checkin_response.checkin_result_case()) {
    case CheckinResponse::kAcceptanceInfo: {
      const auto& acceptance_info = checkin_response.acceptance_info();

      // Record the task name in the opstats db. The phase id has the format
      // "population_name/task_name#round_id.shard_id"
      const auto& phase_id = acceptance_info.execution_phase_id();
      auto population_start = phase_id.find(population_name_ + "/");
      auto task_end = phase_id.find('#');
      std::string task_name = phase_id;
      if (population_start != 0 || task_end == std::string::npos ||
          task_end <= population_name_.length() + 1) {
        log_manager_->LogDiag(
            ProdDiagCode::OPSTATS_TASK_NAME_EXTRACTION_FAILED);
      } else {
        task_name = phase_id.substr(population_name_.length() + 1,
                                    task_end - population_name_.length() - 1);
      }
      opstats_logger_->AddCheckinAcceptedEventWithTaskName(task_name);

      ClientOnlyPlan plan;
      if (!plan.ParseFromString(acceptance_info.plan())) {
        log_manager_->LogDiag(
            ProdDiagCode::BACKGROUND_TRAINING_FAILED_CANNOT_PARSE_PLAN);
        // See note about using InternalError instead of InvalidArgumentError
        // the ReceiveEligibilityEvalCheckinResponse(...) method above.
        return absl::InternalError("Could not parse received plan");
      }
      object_state_ = ObjectState::kCheckinAccepted;
      for (const auto& [k, v] : acceptance_info.side_channels())
        side_channels_[k] = v;
      side_channel_protocol_execution_info_ =
          acceptance_info.side_channel_protocol_execution_info();
      side_channel_protocol_options_response_ =
          checkin_response.protocol_options_response().side_channels();

      int32_t minimum_clients_in_server_visible_aggregate = 0;

      if (side_channel_protocol_execution_info_.has_secure_aggregation()) {
        auto secure_aggregation_protocol_execution_info =
            side_channel_protocol_execution_info_.secure_aggregation();
        auto expected_number_of_clients =
            secure_aggregation_protocol_execution_info
                .expected_number_of_clients();
        auto minimum_number_of_participants =
            plan.phase().minimum_number_of_participants();
        if (expected_number_of_clients < minimum_number_of_participants) {
          return absl::InternalError(
              "expectedNumberOfClients was less than Plan's "
              "minimumNumberOfParticipants.");
        }
        minimum_clients_in_server_visible_aggregate =
            secure_aggregation_protocol_execution_info
                .minimum_clients_in_server_visible_aggregate();
      }
      return CheckinResultPayload{plan, acceptance_info.init_checkpoint(),
                                  task_name,
                                  minimum_clients_in_server_visible_aggregate};
    }
    case CheckinResponse::kRejectionInfo: {
      event_publisher_->PublishRejected();
      opstats_logger_->AddEvent(
          OperationalStats::Event::EVENT_KIND_CHECKIN_REJECTED);
      object_state_ = ObjectState::kCheckinRejected;
      return Rejection{};
    }
    default:
      return absl::UnimplementedError("Unrecognized CheckinResponse");
  }
}

absl::StatusOr<FederatedProtocol::EligibilityEvalCheckinResult>
FederatedProtocol::EligibilityEvalCheckin() {
  FCP_CHECK(object_state_ == ObjectState::kInitialized)
      << "Invalid call sequence";
  object_state_ = ObjectState::kEligibilityEvalCheckinFailed;

  absl::Time start_time = absl::Now();

  // Send an EligibilityEvalCheckinRequest.
  absl::Status request_status = SendEligibilityEvalCheckinRequest();
  // See note about how we handle 'permanent' errors at the top of this file.
  UpdateObjectStateForPermanentError(
      request_status, ObjectState::kEligibilityEvalCheckinFailedPermanentError);
  FCP_RETURN_IF_ERROR(request_status);

  // Receive a CheckinRequestAck.
  absl::Status ack_status = ReceiveCheckinRequestAck();
  UpdateObjectStateForPermanentError(
      ack_status, ObjectState::kEligibilityEvalCheckinFailedPermanentError);
  FCP_RETURN_IF_ERROR(ack_status);

  // Receive + handle an EligibilityEvalCheckinResponse message, and update the
  // object state based on the received response.
  auto response = ReceiveEligibilityEvalCheckinResponse(start_time);
  UpdateObjectStateForPermanentError(
      response.status(),
      ObjectState::kEligibilityEvalCheckinFailedPermanentError);
  return response;
}

absl::StatusOr<FederatedProtocol::CheckinResult> FederatedProtocol::Checkin(
    const absl::optional<TaskEligibilityInfo>& task_eligibility_info) {
  // Checkin(...) must either be the very first method called on this object, or
  // it must follow an earlier call to EligibilityEvalCheckin() that resulted in
  // a non-Rejection response from the server.
  FCP_CHECK(object_state_ == ObjectState::kInitialized ||
            object_state_ == ObjectState::kEligibilityEvalDisabled ||
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
  // We should only request another CheckinRequestAck if we didn't issue an
  // eligibility checkin request yet (which would've received such an ack
  // already).
  bool should_ack_checkin = object_state_ == ObjectState::kInitialized;

  if (object_state_ != ObjectState::kInitialized) {
    // Clear any eligibility eval model identifier we may have set before this
    // Checkin(...) call, since that identifier does not apply to the upcoming
    // checkin request.
    log_manager_->SetModelIdentifier("");
    event_publisher_->SetModelIdentifier("");
  }

  object_state_ = ObjectState::kCheckinFailed;

  absl::Time start_time = absl::Now();
  // Send a CheckinRequest.
  absl::Status request_status =
      SendCheckinRequest(task_eligibility_info, should_ack_checkin);
  // See note about how we handle 'permanent' errors at the top of this file.
  UpdateObjectStateForPermanentError(request_status,
                                     ObjectState::kCheckinFailedPermanentError);
  FCP_RETURN_IF_ERROR(request_status);

  if (should_ack_checkin) {
    // Receive a CheckinRequestAck, if we requested one.
    absl::Status ack_status = ReceiveCheckinRequestAck();
    UpdateObjectStateForPermanentError(
        ack_status, ObjectState::kCheckinFailedPermanentError);
    FCP_RETURN_IF_ERROR(ack_status);
  }

  // Receive + handle a CheckinResponse message, and update the object state
  // based on the received response.
  auto response = ReceiveCheckinResponse(start_time);
  UpdateObjectStateForPermanentError(response.status(),
                                     ObjectState::kCheckinFailedPermanentError);
  return response;
}

absl::Status FederatedProtocol::ReportCompleted(
    ComputationResults results,
    const std::vector<std::pair<std::string, double>>& stats,
    absl::Duration plan_duration) {
  FCP_LOG(INFO) << "Reporting outcome: " << static_cast<int>(engine::COMPLETED);
  FCP_CHECK(object_state_ == ObjectState::kCheckinAccepted)
      << "Invalid call sequence";
  object_state_ = ObjectState::kReportCalled;
  auto response =
      Report(std::move(results), engine::COMPLETED, plan_duration, stats);
  // See note about how we handle 'permanent' errors at the top of this file.
  UpdateObjectStateForPermanentError(response,
                                     ObjectState::kReportFailedPermanentError);
  return response;
}

absl::Status FederatedProtocol::ReportNotCompleted(
    engine::PhaseOutcome phase_outcome, absl::Duration plan_duration) {
  FCP_LOG(WARNING) << "Reporting outcome: " << static_cast<int>(phase_outcome);
  FCP_CHECK(object_state_ == ObjectState::kCheckinAccepted)
      << "Invalid call sequence";
  object_state_ = ObjectState::kReportCalled;
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", "");
  auto response = Report(std::move(results), phase_outcome, plan_duration, {});
  // See note about how we handle 'permanent' errors at the top of this file.
  UpdateObjectStateForPermanentError(response,
                                     ObjectState::kReportFailedPermanentError);
  return response;
}

class SecAggSendToServerImpl : public SendToServerInterface {
 public:
  SecAggSendToServerImpl(
      GrpcBidiStreamInterface* grpc_bidi_stream,
      const std::function<absl::Status(ClientToServerWrapperMessage*)>&
          report_func)
      : grpc_bidi_stream_(grpc_bidi_stream), report_func_(report_func) {}
  ~SecAggSendToServerImpl() override = default;

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
      total_bytes_uploaded_ += last_sent_message_size_;
    }
  }
  int64_t last_sent_message_size() const { return last_sent_message_size_; }
  int64_t total_bytes_uploaded() const { return total_bytes_uploaded_; }

 private:
  GrpcBidiStreamInterface* grpc_bidi_stream_;
  // SecAgg's output must be wrapped in a ReportRequest; because the report
  // logic is mostly generic, this lambda allows it to be shared between
  // aggregation types.
  const std::function<absl::Status(ClientToServerWrapperMessage*)>&
      report_func_;
  size_t total_bytes_uploaded_ = 0;
  size_t last_sent_message_size_ = 0;
};

class SecAggStateTransitionListenerImpl
    : public StateTransitionListenerInterface {
 public:
  SecAggStateTransitionListenerImpl(
      SecAggEventPublisher* secagg_event_publisher, LogManager* log_manager,
      const SecAggSendToServerImpl& secagg_send_to_server_impl,
      const size_t& last_received_message_size)
      : secagg_event_publisher_(secagg_event_publisher),
        log_manager_(log_manager),
        secagg_send_to_server_impl_(secagg_send_to_server_impl),
        last_received_message_size_(last_received_message_size) {
    FCP_CHECK(secagg_event_publisher_)
        << "An implementation of "
        << "SecAggEventPublisher must be provided.";
  }
  void Transition(ClientState new_state) override {
    FCP_LOG(INFO) << "Transitioning from state: " << static_cast<int>(state_)
                  << " to state: " << static_cast<int>(new_state);
    state_ = new_state;
    if (state_ == ClientState::ABORTED)
      log_manager_->LogDiag(ProdDiagCode::SECAGG_CLIENT_NATIVE_ERROR_GENERIC);
    secagg_event_publisher_->PublishStateTransition(
        new_state, secagg_send_to_server_impl_.last_sent_message_size(),
        last_received_message_size_);
  }

  void Started(ClientState state) override {
    // TODO(team): Implement this.
  }

  void Stopped(ClientState state) override {
    // TODO(team): Implement this.
  }

  void set_execution_session_id(int64_t execution_session_id) override {
    secagg_event_publisher_->set_execution_session_id(execution_session_id);
  }

 private:
  SecAggEventPublisher* const secagg_event_publisher_;
  LogManager* const log_manager_;
  const SecAggSendToServerImpl& secagg_send_to_server_impl_;
  const size_t& last_received_message_size_;
  ClientState state_ = ClientState::INITIAL;
};

absl::Status FederatedProtocol::ReportInternal(
    std::string tf_checkpoint, engine::PhaseOutcome phase_outcome,
    absl::Duration plan_duration,
    const std::vector<std::pair<std::string, double>>& stats,
    int64_t* report_request_size,
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
  for (const auto& [name, value] : stats) {
    auto training_stat = client_execution_stats.add_training_stat();
    training_stat->set_stat_name(name);
    training_stat->set_stat_value(value);
  }
  client_execution_stats.mutable_duration()->set_seconds(
      absl::IDivDuration(plan_duration, absl::Seconds(1), &plan_duration));
  client_execution_stats.mutable_duration()->set_nanos(static_cast<int32_t>(
      absl::IDivDuration(plan_duration, absl::Nanoseconds(1), &plan_duration)));
  report->add_serialized_train_event()->PackFrom(client_execution_stats);

  // 4. Send ReportRequest.
  *report_request_size += client_stream_message.ByteSizeLong();
  opstats_logger_->AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED);
  if (flags_->commit_opstats_on_upload_started()) {
    // Commit the run data accumulated thus far to Opstats and fail if something
    // goes wrong.
    FCP_RETURN_IF_ERROR(opstats_logger_->CommitToStorage());
  }
  // Log the event after we know we've successfully committed the event to
  // Opstats.
  event_publisher_->PublishReportStarted(*report_request_size);

  // Note that we do not use the FederatedProtocol::Send(...) helper method
  // here, since we are already running within a call to
  // InterruptibleRunner::Run.
  const auto status = this->grpc_bidi_stream_->Send(&client_stream_message);
  if (!status.ok()) {
    return absl::Status(
        status.code(),
        absl::StrCat("Error sending ReportRequest: ", status.message()));
  }
  bytes_uploaded_ += *report_request_size;
  UpdateOpStatsNetworkStats();

  return absl::OkStatus();
}

absl::Status FederatedProtocol::Report(
    ComputationResults results, engine::PhaseOutcome phase_outcome,
    absl::Duration plan_duration,
    const std::vector<std::pair<std::string, double>>& stats) {
  std::string tf_checkpoint;
  int64_t report_request_size = 0;

  // This lambda allows for convenient reporting from within SecAgg's
  // SendToServerInterface::Send().
  std::function<absl::Status(ClientToServerWrapperMessage*)> report_lambda =
      [&](ClientToServerWrapperMessage* secagg_commit_message) -> absl::Status {
    return ReportInternal(std::move(tf_checkpoint), phase_outcome,
                          plan_duration, stats, &report_request_size,
                          secagg_commit_message);
  };

  absl::Time start_time = absl::Now();

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

    SecAggSendToServerImpl* send_to_server_impl_raw_ptr = nullptr;
    auto input_map = absl::make_unique<SecAggVectorMap>();
    auto send_to_server_impl_unique_ptr =
        absl::make_unique<SecAggSendToServerImpl>(grpc_bidi_stream_.get(),
                                                  report_lambda);
    send_to_server_impl_raw_ptr = send_to_server_impl_unique_ptr.get();
    size_t last_received_message_size = 0;
    auto secagg_event_publisher = event_publisher_->secagg_event_publisher();
    FCP_CHECK(secagg_event_publisher)
        << "An implementation of "
        << "SecAggEventPublisher must be provided.";
    auto secagg_state_transition_listener =
        absl::make_unique<SecAggStateTransitionListenerImpl>(
            secagg_event_publisher, log_manager_, *send_to_server_impl_raw_ptr,
            last_received_message_size);
    if (!secagg_client_) {  // non-test code.
      std::vector<InputVectorSpecification> input_vector_specification;
      for (auto& [k, v] : results) {
        if (absl::holds_alternative<TFCheckpoint>(v)) {
          tf_checkpoint = absl::get<TFCheckpoint>(std::move(v));
        } else if (absl::holds_alternative<QuantizedTensor>(v)) {
          auto execution_info = side_channels_.find(k);
          if (execution_info == side_channels_.end())
            return absl::InternalError(
                absl::StrCat("Execution not found for aggregand: ", k));
          // Note: std::move is used below to ensure that each QuantizedTensor
          // is consumed when converted to SecAggVector and that we don't
          // continue having both in memory for longer than needed.
          auto vector = absl::get<QuantizedTensor>(std::move(v));
          uint64_t modulus;
          auto secure_aggregand = execution_info->second.secure_aggregand();
          // TODO(team): Delete output_bitwidth support once modulus is
          // fully rolled out.
          if (secure_aggregand.modulus() > 0) {
            modulus = secure_aggregand.modulus();
          } else {
            // Note: we ignore vector.get_bitwidth() here, because (1) it is
            // only an upper bound on the *input* bitwidth, based on the
            // Tensorflow dtype, but (2) we have exact *output* bitwidth
            // information from the execution_info, and that is what SecAgg
            // needs.
            modulus = 1ULL << secure_aggregand.output_bitwidth();
          }
          if (modulus <= 1 || modulus > SecAggVector::kMaxModulus) {
            return absl::InternalError(absl::StrCat(
                "Invalid SecAgg modulus configuration: ", modulus));
          }
          if (vector.values.empty())
            return absl::InternalError(
                absl::StrCat("Zero sized vector found: ", k));
          int64_t flattened_length = 1;
          for (const auto& size : vector.dimensions) flattened_length *= size;
          auto data_length = vector.values.size();
          if (flattened_length != data_length)
            return absl::InternalError(
                absl::StrCat("Flattened length: ", flattened_length,
                             " does not match vector size: ", data_length));
          for (const auto& v : vector.values) {
            if (v >= modulus) {
              return absl::InternalError(absl::StrCat(
                  "The input SecAgg vector doesn't have the appropriate "
                  "modulus: element with value ",
                  v, " found, max value allowed ", (modulus - 1ULL)));
            }
          }
          input_vector_specification.emplace_back(k, flattened_length, modulus);
          input_map->try_emplace(
              k, absl::MakeConstSpan(vector.values.data(), data_length),
              modulus);
        }
      }
      secagg_client_ = std::make_unique<SecAggClient>(
          expected_number_of_clients,
          secure_aggregation_protocol_execution_info
              .minimum_surviving_clients_for_reconstruction(),
          std::move(input_vector_specification),
          absl::make_unique<CryptoRandPrng>(),
          std::move(send_to_server_impl_unique_ptr),
          std::move(secagg_state_transition_listener),
          absl::make_unique<AesCtrPrngFactory>());
    }
    FCP_RETURN_IF_ERROR(interruptible_runner_->Run(
        [this, &input_map, &last_received_message_size,
         &secagg_event_publisher]() -> absl::Status {
          FCP_RETURN_IF_ERROR(secagg_client_->Start());
          FCP_RETURN_IF_ERROR(secagg_client_->SetInput(std::move(input_map)));
          while (!secagg_client_->IsCompletedSuccessfully()) {
            ServerStreamMessage server_stream_message;
            // Note that we do not use the FederatedProtocol::Receive(...)
            // helper method here, since we are already running within a call to
            // InterruptibleRunner::Run.
            absl::Status receive_status =
                this->grpc_bidi_stream_->Receive(&server_stream_message);
            if (!receive_status.ok()) {
              return absl::Status(receive_status.code(),
                                  absl::StrCat("Error during SecAgg receive: ",
                                               receive_status.message()));
            }
            last_received_message_size = server_stream_message.ByteSizeLong();
            this->bytes_downloaded_ += last_received_message_size;
            UpdateOpStatsNetworkStats();
            if (!server_stream_message
                     .has_secure_aggregation_server_message()) {
              return absl::InternalError(
                  absl::StrCat("Bad response to SecAgg protocol; Expected "
                               "ServerToClientWrapperMessage but got ",
                               server_stream_message.kind_case(), "."));
            }
            auto result = secagg_client_->ReceiveMessage(
                server_stream_message.secure_aggregation_server_message());
            if (!result.ok()) {
              secagg_event_publisher->PublishError();
              return absl::Status(
                  result.status().code(),
                  absl::StrCat("Error receiving SecAgg message: ",
                               result.status().message()));
            }
            if (secagg_client_->IsAborted()) {
              std::string error_message = "error message not found.";
              if (secagg_client_->ErrorMessage().ok())
                error_message = secagg_client_->ErrorMessage().value();
              secagg_event_publisher->PublishAbort(false, error_message);
              return absl::CancelledError("SecAgg aborted: " + error_message);
            }
          }
          return absl::OkStatus();
        },
        [this, &secagg_event_publisher]() {
          log_manager_->LogDiag(
              ProdDiagCode::SECAGG_CLIENT_NATIVE_ERROR_GENERIC);
          auto abort_message = "Client-initiated abort.";
          auto result = secagg_client_->Abort(abort_message);
          if (!result.ok()) {
            FCP_LOG(ERROR) << "Could not initiate client abort, code: "
                           << result.code() << " message: " << result.message();
          }
          // Note: the implementation assumes that secagg_event_publisher
          // cannot hang indefinitely, i.e. does not need its own interruption
          // trigger.
          secagg_event_publisher->PublishAbort(true, abort_message);
          grpc_bidi_stream_->Close();
          // What about event_publisher_ and log_manager_?
        }));
    if (send_to_server_impl_raw_ptr) {
      bytes_uploaded_ += send_to_server_impl_raw_ptr->total_bytes_uploaded();
      UpdateOpStatsNetworkStats();
    }
  } else {
    // Report without secure aggregation.
    FCP_LOG(INFO) << "Reporting via Simple Aggregation";
    if (results.size() != 1 ||
        !absl::holds_alternative<TFCheckpoint>(results.begin()->second)) {
      return absl::InternalError(
          "Simple Aggregation aggregands have unexpected format.");
    }
    tf_checkpoint = absl::get<TFCheckpoint>(std::move(results.begin()->second));
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

  absl::Duration upload_time = absl::Now() - start_time;
  event_publisher_->PublishReportFinished(
      report_request_size, grpc_bidi_stream_->ChunkingLayerBytesSent(),
      upload_time);
  opstats_logger_->AddEvent(
      OperationalStats::Event::EVENT_KIND_UPLOAD_FINISHED);
  return absl::OkStatus();
}

RetryWindow FederatedProtocol::GetLatestRetryWindow() {
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
      if (!federated_training_use_new_retry_delay_behavior_) {
        return retry_window_if_accepted_;
      }
      FCP_CHECK(checkin_request_ack_info_.has_value());
      return GenerateRetryWindowFromRetryTimeAndToken(
          checkin_request_ack_info_->retry_info_if_accepted);
    case ObjectState::kEligibilityEvalCheckinRejected:
    case ObjectState::kEligibilityEvalDisabled:
    case ObjectState::kEligibilityEvalEnabled:
    case ObjectState::kCheckinRejected:
      if (!federated_training_use_new_retry_delay_behavior_) {
        return retry_window_if_rejected_;
      }
      FCP_CHECK(checkin_request_ack_info_.has_value());
      return GenerateRetryWindowFromRetryTimeAndToken(
          checkin_request_ack_info_->retry_info_if_rejected);
    case ObjectState::kInitialized:
    case ObjectState::kEligibilityEvalCheckinFailed:
    case ObjectState::kCheckinFailed:
      if (!federated_training_use_new_retry_delay_behavior_) {
        // Note: if state is kInitialized then we will not yet have received any
        // RetryWindows from the server. This can also be (but isn't always) the
        // case with kEligibilityEvalCheckinFailed and kCheckinFailed. However,
        // in these cases retry_window_if_rejected_ will still be initialized to
        // the default RetryWindow instance, which is the right value to return
        // in that case (as long as the
        // federated_training_use_new_retry_delay_behavior_ flag is false).
        return retry_window_if_rejected_;
      }
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
      if (!federated_training_use_new_retry_delay_behavior_) {
        // If the above flag is off, then we don't expect to ever enter these
        // 'permanent' error states. However, for extra safety while the flag
        // rolls out we do handle them just like the other 'rejected' states.
        // This is closest to the pre-flag behavior.
        return retry_window_if_rejected_;
      }
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
  }
}

// Converts the given RetryTimeAndToken to a zero-width RetryWindow (where
// delay_min and delay_max are set to the same value), by converting the target
// retry time to a delay relative to the current timestamp.
RetryWindow FederatedProtocol::GenerateRetryWindowFromRetryTimeAndToken(
    const FederatedProtocol::RetryTimeAndToken& retry_info) {
  // Convert the target retry time back to a duration, based on the current
  // time. I.e. if at 09:50AM the CheckinRequestAck was received and the chosen
  // target retry time was 11:00AM, and if it is now 09:55AM, then the
  // calculated duration will be 1 hour and 5 minutes.
  absl::Duration retry_delay = retry_info.retry_time - absl::Now();
  // If the target retry time has already passed, then use a zero-length
  // duration.
  retry_delay =
      retry_delay >= absl::ZeroDuration() ? retry_delay : absl::ZeroDuration();

  // Generate a RetryWindow with delay_min and delay_max both set to the same
  // value.
  RetryWindow retry_window;
  retry_window.set_retry_token(retry_info.retry_token);
  *retry_window.mutable_delay_min() = *retry_window.mutable_delay_max() =
      ConvertAbslToProtoDuration(retry_delay);
  return retry_window;
}

void FederatedProtocol::UpdateOpStatsNetworkStats() {
  opstats_logger_->SetNetworkStats(
      bytes_downloaded_, bytes_uploaded_,
      grpc_bidi_stream_->ChunkingLayerBytesReceived(),
      grpc_bidi_stream_->ChunkingLayerBytesSent());
}

void FederatedProtocol::UpdateObjectStateForPermanentError(
    absl::Status status,
    FederatedProtocol::ObjectState permanent_error_object_state) {
  if (federated_training_permanent_error_codes_.contains(
          static_cast<int32_t>(status.code()))) {
    object_state_ = permanent_error_object_state;
  }
}

}  // namespace client
}  // namespace fcp
