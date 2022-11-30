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
#include "fcp/client/http/http_federated_protocol.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "google/longrunning/operations.pb.h"
#include "google/protobuf/any.pb.h"
#include "google/rpc/code.pb.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "fcp/base/clock.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/time_util.h"
#include "fcp/base/wall_clock_stopwatch.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/federated_protocol_util.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/http_client_util.h"
#include "fcp/client/http/http_secagg_send_to_server_impl.h"
#include "fcp/client/http/in_memory_request_response.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/stats.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/federatedcompute/aggregations.pb.h"
#include "fcp/protos/federatedcompute/common.pb.h"
#include "fcp/protos/federatedcompute/eligibility_eval_tasks.pb.h"
#include "fcp/protos/federatedcompute/secure_aggregations.pb.h"
#include "fcp/protos/federatedcompute/task_assignments.pb.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace http {
namespace {

using ::fcp::client::GenerateRetryWindowFromRetryTime;
using ::fcp::client::GenerateRetryWindowFromTargetDelay;
using ::fcp::client::PickRetryTimeFromRange;
using ::google::internal::federatedcompute::v1::AbortAggregationRequest;
using ::google::internal::federatedcompute::v1::ClientStats;
using ::google::internal::federatedcompute::v1::EligibilityEvalTaskRequest;
using ::google::internal::federatedcompute::v1::EligibilityEvalTaskResponse;
using ::google::internal::federatedcompute::v1::
    ReportEligibilityEvalTaskResultRequest;
using ::google::internal::federatedcompute::v1::ReportTaskResultRequest;
using ::google::internal::federatedcompute::v1::Resource;
using ::google::internal::federatedcompute::v1::ResourceCompressionFormat;
using ::google::internal::federatedcompute::v1::
    SecureAggregationProtocolExecutionInfo;
using ::google::internal::federatedcompute::v1::
    StartAggregationDataUploadRequest;
using ::google::internal::federatedcompute::v1::
    StartAggregationDataUploadResponse;
using ::google::internal::federatedcompute::v1::StartSecureAggregationRequest;
using ::google::internal::federatedcompute::v1::StartSecureAggregationResponse;
using ::google::internal::federatedcompute::v1::StartTaskAssignmentRequest;
using ::google::internal::federatedcompute::v1::StartTaskAssignmentResponse;
using ::google::internal::federatedcompute::v1::SubmitAggregationResultRequest;
using ::google::internal::federatedml::v2::TaskEligibilityInfo;
using ::google::longrunning::Operation;

using CompressionFormat =
    ::fcp::client::http::UriOrInlineData::InlineData::CompressionFormat;

// Creates the URI suffix for a RequestEligibilityEvalTask protocol request.
absl::StatusOr<std::string> CreateRequestEligibilityEvalTaskUriSuffix(
    absl::string_view population_name) {
  constexpr absl::string_view kRequestEligibilityEvalTaskUriSuffix =
      "/v1/eligibilityevaltasks/$0:request";
  FCP_ASSIGN_OR_RETURN(std::string encoded_population_name,
                       EncodeUriSinglePathSegment(population_name));
  return absl::Substitute(kRequestEligibilityEvalTaskUriSuffix,
                          encoded_population_name);
}

// Creates the URI suffix for a ReportEligibilityEvalTaskResult protocol
// request.
absl::StatusOr<std::string> CreateReportEligibilityEvalTaskResultUriSuffix(
    absl::string_view population_name, absl::string_view session_id) {
  constexpr absl::string_view kReportEligibilityEvalTaskResultUriSuffix =
      "/v1/populations/$0/eligibilityevaltasks/$1:reportresult";
  FCP_ASSIGN_OR_RETURN(std::string encoded_population_name,
                       EncodeUriSinglePathSegment(population_name));
  FCP_ASSIGN_OR_RETURN(std::string encoded_session_id,
                       EncodeUriSinglePathSegment(session_id));
  return absl::Substitute(kReportEligibilityEvalTaskResultUriSuffix,
                          encoded_population_name, encoded_session_id);
}

// Creates the URI suffix for a StartTaskAssignment protocol request.
absl::StatusOr<std::string> CreateStartTaskAssignmentUriSuffix(
    absl::string_view population_name, absl::string_view session_id) {
  constexpr absl::string_view kStartTaskAssignmentUriSuffix =
      "/v1/populations/$0/taskassignments/$1:start";
  FCP_ASSIGN_OR_RETURN(std::string encoded_population_name,
                       EncodeUriSinglePathSegment(population_name));
  FCP_ASSIGN_OR_RETURN(std::string encoded_session_id,
                       EncodeUriSinglePathSegment(session_id));
  return absl::Substitute(kStartTaskAssignmentUriSuffix,
                          encoded_population_name, encoded_session_id);
}

// Creates he URI suffix for a ReportTaskResult protocol request.
absl::StatusOr<std::string> CreateReportTaskResultUriSuffix(
    absl::string_view population_name, absl::string_view session_id) {
  constexpr absl::string_view pattern =
      "/v1/populations/$0/taskassignments/$1:reportresult";
  FCP_ASSIGN_OR_RETURN(std::string encoded_population_name,
                       EncodeUriSinglePathSegment(population_name));
  FCP_ASSIGN_OR_RETURN(std::string encoded_session_id,
                       EncodeUriSinglePathSegment(session_id));
  // Construct the URI suffix.
  return absl::Substitute(pattern, encoded_population_name, encoded_session_id);
}

absl::StatusOr<std::string> CreateStartAggregationDataUploadUriSuffix(
    absl::string_view aggregation_id, absl::string_view client_token) {
  constexpr absl::string_view pattern =
      "/v1/aggregations/$0/clients/$1:startdataupload";
  FCP_ASSIGN_OR_RETURN(std::string encoded_aggregation_id,
                       EncodeUriSinglePathSegment(aggregation_id));
  FCP_ASSIGN_OR_RETURN(std::string encoded_client_token,
                       EncodeUriSinglePathSegment(client_token));
  // Construct the URI suffix.
  return absl::Substitute(pattern, encoded_aggregation_id,
                          encoded_client_token);
}

absl::StatusOr<std::string> CreateSubmitAggregationResultUriSuffix(
    absl::string_view aggregation_id, absl::string_view client_token) {
  constexpr absl::string_view pattern = "/v1/aggregations/$0/clients/$1:submit";
  FCP_ASSIGN_OR_RETURN(std::string encoded_aggregation_id,
                       EncodeUriSinglePathSegment(aggregation_id));
  FCP_ASSIGN_OR_RETURN(std::string encoded_client_token,
                       EncodeUriSinglePathSegment(client_token));
  // Construct the URI suffix.
  return absl::Substitute(pattern, encoded_aggregation_id,
                          encoded_client_token);
}

absl::StatusOr<std::string> CreateAbortAggregationUriSuffix(
    absl::string_view aggregation_id, absl::string_view client_token) {
  constexpr absl::string_view pattern = "/v1/aggregations/$0/clients/$1:abort";
  FCP_ASSIGN_OR_RETURN(std::string encoded_aggregation_id,
                       EncodeUriSinglePathSegment(aggregation_id));
  FCP_ASSIGN_OR_RETURN(std::string encoded_client_token,
                       EncodeUriSinglePathSegment(client_token));
  // Construct the URI suffix.
  return absl::Substitute(pattern, encoded_aggregation_id,
                          encoded_client_token);
}

absl::StatusOr<std::string> CreateStartSecureAggregationUriSuffix(
    absl::string_view aggregation_id, absl::string_view client_token) {
  constexpr absl::string_view pattern =
      "/v1/secureaggregations/$0/clients/$1:start";
  FCP_ASSIGN_OR_RETURN(std::string encoded_aggregation_id,
                       EncodeUriSinglePathSegment(aggregation_id));
  FCP_ASSIGN_OR_RETURN(std::string encoded_client_token,
                       EncodeUriSinglePathSegment(client_token));
  // Construct the URI suffix.
  return absl::Substitute(pattern, encoded_aggregation_id,
                          encoded_client_token);
}

// Convert a Resource proto into a UriOrInlineData object. Returns an
// `INVALID_ARGUMENT` error if the given `Resource` has the `uri` field set to
// an empty value, or an `UNIMPLEMENTED` error if the `Resource` has an unknown
// field set.
absl::StatusOr<UriOrInlineData> ConvertResourceToUriOrInlineData(
    const Resource& resource) {
  switch (resource.resource_case()) {
    case Resource::ResourceCase::kUri:
      if (resource.uri().empty()) {
        return absl::InvalidArgumentError(
            "Resource.uri must be non-empty when set");
      }
      return UriOrInlineData::CreateUri(
          resource.uri(), resource.client_cache_id(),
          TimeUtil::ConvertProtoToAbslDuration(resource.max_age()));
    case Resource::ResourceCase::kInlineResource: {
      CompressionFormat compression_format = CompressionFormat::kUncompressed;
      if (resource.inline_resource().has_compression_format()) {
        switch (resource.inline_resource().compression_format()) {
          case ResourceCompressionFormat::RESOURCE_COMPRESSION_FORMAT_GZIP:
            compression_format = CompressionFormat::kGzip;
            break;
          default:
            return absl::UnimplementedError(
                "Unknown ResourceCompressionFormat");
        }
      }
      return UriOrInlineData::CreateInlineData(
          absl::Cord(resource.inline_resource().data()), compression_format);
    }
    case Resource::ResourceCase::RESOURCE_NOT_SET:
      // If neither field is set at all, we'll just act as if we got an empty
      // inline data field.
      return UriOrInlineData::CreateInlineData(
          absl::Cord(), CompressionFormat::kUncompressed);
    default:
      return absl::UnimplementedError("Unknown Resource type");
  }
}

::google::rpc::Code ConvertPhaseOutcomeToRpcCode(
    engine::PhaseOutcome phase_outcome) {
  switch (phase_outcome) {
    case engine::PhaseOutcome::COMPLETED:
      return ::google::rpc::Code::OK;
    case engine::PhaseOutcome::ERROR:
      return ::google::rpc::Code::INTERNAL;
    case engine::PhaseOutcome::INTERRUPTED:
      return ::google::rpc::Code::CANCELLED;
    default:
      return ::google::rpc::Code::UNKNOWN;
  }
}

absl::StatusOr<ReportTaskResultRequest> CreateReportTaskResultRequest(
    engine::PhaseOutcome phase_outcome, absl::Duration plan_duration,
    absl::string_view aggregation_id) {
  ReportTaskResultRequest request;
  request.set_aggregation_id(std::string(aggregation_id));
  request.set_computation_status_code(
      ConvertPhaseOutcomeToRpcCode(phase_outcome));
  ClientStats* client_stats = request.mutable_client_stats();
  *client_stats->mutable_computation_execution_duration() =
      TimeUtil::ConvertAbslToProtoDuration(plan_duration);
  return request;
}

// Creates a special InterruptibleRunner which won't check the should_abort
// function until the timeout duration is passed.  This special
// InterruptibleRunner is used to issue Cancellation requests or Abort requests.
std::unique_ptr<InterruptibleRunner> CreateDelayedInterruptibleRunner(
    LogManager* log_manager, std::function<bool()> should_abort,
    const InterruptibleRunner::TimingConfig& timing_config,
    absl::Time deadline) {
  return std::make_unique<InterruptibleRunner>(
      log_manager,
      [deadline, should_abort]() {
        return absl::Now() > deadline && should_abort();
      },
      timing_config,
      InterruptibleRunner::DiagnosticsConfig{
          .interrupted = ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP,
          .interrupt_timeout =
              ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP_TIMED_OUT,
          .interrupted_extended = ProdDiagCode::
              BACKGROUND_TRAINING_INTERRUPT_HTTP_EXTENDED_COMPLETED,
          .interrupt_timeout_extended = ProdDiagCode::
              BACKGROUND_TRAINING_INTERRUPT_HTTP_EXTENDED_TIMED_OUT});
}
}  // namespace

HttpFederatedProtocol::HttpFederatedProtocol(
    Clock* clock, LogManager* log_manager, const Flags* flags,
    HttpClient* http_client,
    std::unique_ptr<SecAggRunnerFactory> secagg_runner_factory,
    SecAggEventPublisher* secagg_event_publisher,
    absl::string_view entry_point_uri, absl::string_view api_key,
    absl::string_view population_name, absl::string_view retry_token,
    absl::string_view client_version, absl::string_view attestation_measurement,
    std::function<bool()> should_abort, absl::BitGen bit_gen,
    const InterruptibleRunner::TimingConfig& timing_config,
    cache::ResourceCache* resource_cache)
    : object_state_(ObjectState::kInitialized),
      clock_(*clock),
      log_manager_(log_manager),
      flags_(flags),
      http_client_(http_client),
      secagg_runner_factory_(std::move(secagg_runner_factory)),
      secagg_event_publisher_(secagg_event_publisher),
      interruptible_runner_(std::make_unique<InterruptibleRunner>(
          log_manager, should_abort, timing_config,
          InterruptibleRunner::DiagnosticsConfig{
              .interrupted = ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP,
              .interrupt_timeout =
                  ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP_TIMED_OUT,
              .interrupted_extended = ProdDiagCode::
                  BACKGROUND_TRAINING_INTERRUPT_HTTP_EXTENDED_COMPLETED,
              .interrupt_timeout_extended = ProdDiagCode::
                  BACKGROUND_TRAINING_INTERRUPT_HTTP_EXTENDED_TIMED_OUT})),
      eligibility_eval_request_creator_(
          std::make_unique<ProtocolRequestCreator>(
              entry_point_uri, api_key, HeaderList{},
              !flags->disable_http_request_body_compression())),
      protocol_request_helper_(http_client, &bytes_downloaded_,
                               &bytes_uploaded_, network_stopwatch_.get(),
                               clock),
      api_key_(api_key),
      population_name_(population_name),
      retry_token_(retry_token),
      client_version_(client_version),
      attestation_measurement_(attestation_measurement),
      should_abort_(std::move(should_abort)),
      bit_gen_(std::move(bit_gen)),
      timing_config_(timing_config),
      waiting_period_for_cancellation_(
          absl::Seconds(flags->waiting_period_sec_for_cancellation())),
      resource_cache_(resource_cache) {
  // Note that we could cast the provided error codes to absl::StatusCode
  // values here. However, that means we'd have to handle the case when
  // invalid integers that don't map to a StatusCode enum are provided in the
  // flag here. Instead, we cast absl::StatusCodes to int32_t each time we
  // compare them with the flag-provided list of codes, which means we never
  // have to worry about invalid flag values (besides the fact that invalid
  // values will be silently ignored, which could make it harder to realize when
  // a flag is misconfigured).
  const std::vector<int32_t>& error_codes =
      flags->federated_training_permanent_error_codes();
  federated_training_permanent_error_codes_ =
      absl::flat_hash_set<int32_t>(error_codes.begin(), error_codes.end());
}

absl::StatusOr<FederatedProtocol::EligibilityEvalCheckinResult>
HttpFederatedProtocol::EligibilityEvalCheckin(
    std::function<void(const EligibilityEvalTask&)>
        payload_uris_received_callback) {
  FCP_CHECK(object_state_ == ObjectState::kInitialized)
      << "Invalid call sequence";
  object_state_ = ObjectState::kEligibilityEvalCheckinFailed;

  // Send the request and parse the response.
  auto response = HandleEligibilityEvalTaskResponse(
      PerformEligibilityEvalTaskRequest(), payload_uris_received_callback);
  // Update the object state to ensure we return the correct retry delay.
  UpdateObjectStateIfPermanentError(
      response.status(),
      ObjectState::kEligibilityEvalCheckinFailedPermanentError);
  if (response.ok() && std::holds_alternative<EligibilityEvalTask>(*response)) {
    eligibility_eval_enabled_ = true;
  }
  return response;
}

absl::StatusOr<InMemoryHttpResponse>
HttpFederatedProtocol::PerformEligibilityEvalTaskRequest() {
  // Create and serialize the request body. Note that the `population_name`
  // field is set in the URI instead of in this request proto message.
  EligibilityEvalTaskRequest request;
  request.mutable_client_version()->set_version_code(client_version_);
  request.mutable_attestation_measurement()->set_value(
      attestation_measurement_);

    request.mutable_resource_capabilities()->add_supported_compression_formats(
        ResourceCompressionFormat::RESOURCE_COMPRESSION_FORMAT_GZIP);

  FCP_ASSIGN_OR_RETURN(
      std::string uri_suffix,
      CreateRequestEligibilityEvalTaskUriSuffix(population_name_));
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<HttpRequest> http_request,
      eligibility_eval_request_creator_->CreateProtocolRequest(
          uri_suffix, {}, HttpRequest::Method::kPost,
          request.SerializeAsString(), /*is_protobuf_encoded=*/true));

  // Issue the request.
  return protocol_request_helper_.PerformProtocolRequest(
      std::move(http_request), *interruptible_runner_);
}

absl::StatusOr<FederatedProtocol::EligibilityEvalCheckinResult>
HttpFederatedProtocol::HandleEligibilityEvalTaskResponse(
    absl::StatusOr<InMemoryHttpResponse> http_response,
    std::function<void(const EligibilityEvalTask&)>
        payload_uris_received_callback) {
  if (!http_response.ok()) {
    // If the protocol request failed then forward the error, but add a prefix
    // to the error message to ensure we can easily distinguish an HTTP error
    // occurring in response to the protocol request from HTTP errors occurring
    // during checkpoint/plan resource fetch requests later on.
    return absl::Status(http_response.status().code(),
                        absl::StrCat("protocol request failed: ",
                                     http_response.status().ToString()));
  }

  EligibilityEvalTaskResponse response_proto;
    if (!response_proto.ParseFromString(std::string(http_response->body))) {
    return absl::InvalidArgumentError("Could not parse response_proto");
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
  retry_times_ = RetryTimes{
      .retry_time_if_rejected = PickRetryTimeFromRange(
          response_proto.retry_window_if_rejected().delay_min(),
          response_proto.retry_window_if_rejected().delay_max(), bit_gen_),
      .retry_time_if_accepted = PickRetryTimeFromRange(
          response_proto.retry_window_if_accepted().delay_min(),
          response_proto.retry_window_if_accepted().delay_max(), bit_gen_)};

  // If the request was rejected then the protocol session has ended and there's
  // no more work for us to do.
  if (response_proto.has_rejection_info()) {
    object_state_ = ObjectState::kEligibilityEvalCheckinRejected;
    return Rejection{};
  }

  session_id_ = response_proto.session_id();

  FCP_ASSIGN_OR_RETURN(
      task_assignment_request_creator_,
      ProtocolRequestCreator::Create(
          api_key_, response_proto.task_assignment_forwarding_info(),
          !flags_->disable_http_request_body_compression()));

  switch (response_proto.result_case()) {
    case EligibilityEvalTaskResponse::kEligibilityEvalTask: {
      const auto& task = response_proto.eligibility_eval_task();

      EligibilityEvalTask result{.execution_id = task.execution_id()};
      payload_uris_received_callback(result);

      // Fetch the task resources, returning any errors that may be encountered
      // in the process.
      FCP_ASSIGN_OR_RETURN(
          result.payloads,
          FetchTaskResources(
              {.plan = task.plan(), .checkpoint = task.init_checkpoint()}));

      object_state_ = ObjectState::kEligibilityEvalEnabled;
      return std::move(result);
    }
    case EligibilityEvalTaskResponse::kNoEligibilityEvalConfigured: {
      // Nothing to do...
      object_state_ = ObjectState::kEligibilityEvalDisabled;
      return EligibilityEvalDisabled{};
    }
    default:
      return absl::UnimplementedError(
          "Unrecognized EligibilityEvalCheckinResponse");
  }
}

absl::StatusOr<std::unique_ptr<HttpRequest>>
HttpFederatedProtocol::CreateReportEligibilityEvalTaskResultRequest(
    absl::Status status) {
  ReportEligibilityEvalTaskResultRequest request;
  request.set_status_code(static_cast<google::rpc::Code>(status.code()));
  FCP_ASSIGN_OR_RETURN(std::string uri_suffix,
                       CreateReportEligibilityEvalTaskResultUriSuffix(
                           population_name_, session_id_));
  return eligibility_eval_request_creator_->CreateProtocolRequest(
      uri_suffix, QueryParams(), HttpRequest::Method::kPost,
      request.SerializeAsString(),
      /*is_protobuf_encoded=*/true);
}

void HttpFederatedProtocol::ReportEligibilityEvalError(
    absl::Status error_status) {
  if (!ReportEligibilityEvalErrorInternal(error_status).ok()) {
    log_manager_->LogDiag(
        ProdDiagCode::HTTP_REPORT_ELIGIBILITY_EVAL_RESULT_REQUEST_FAILED);
  }
}

absl::Status HttpFederatedProtocol::ReportEligibilityEvalErrorInternal(
    absl::Status error_status) {
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<HttpRequest> request,
      CreateReportEligibilityEvalTaskResultRequest(error_status));
  return protocol_request_helper_
      .PerformProtocolRequest(std::move(request), *interruptible_runner_)
      .status();
}

absl::StatusOr<FederatedProtocol::CheckinResult> HttpFederatedProtocol::Checkin(
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

  // Send the request and parse the response.
  auto response = HandleTaskAssignmentOperationResponse(
      PerformTaskAssignmentAndReportEligibilityEvalResultRequests(
          task_eligibility_info),
      payload_uris_received_callback);

  // Update the object state to ensure we return the correct retry delay.
  UpdateObjectStateIfPermanentError(
      response.status(),
      ObjectState::kEligibilityEvalCheckinFailedPermanentError);
  return response;
}

absl::StatusOr<InMemoryHttpResponse> HttpFederatedProtocol::
    PerformTaskAssignmentAndReportEligibilityEvalResultRequests(
        const std::optional<TaskEligibilityInfo>& task_eligibility_info) {
  // Create and serialize the request body. Note that the `population_name`
  // and `session_id` fields are set in the URI instead of in this request
  // proto message.
  StartTaskAssignmentRequest request;
  request.mutable_client_version()->set_version_code(client_version_);

  if (task_eligibility_info.has_value()) {
    *request.mutable_task_eligibility_info() = *task_eligibility_info;
  }

    request.mutable_resource_capabilities()->add_supported_compression_formats(
        ResourceCompressionFormat::RESOURCE_COMPRESSION_FORMAT_GZIP);

  std::vector<std::unique_ptr<HttpRequest>> requests;

  // Construct the URI suffix.
  FCP_ASSIGN_OR_RETURN(
      std::string task_assignment_uri_suffix,
      CreateStartTaskAssignmentUriSuffix(population_name_, session_id_));
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<HttpRequest> task_assignment_http_request,
      task_assignment_request_creator_->CreateProtocolRequest(
          task_assignment_uri_suffix, {}, HttpRequest::Method::kPost,
          request.SerializeAsString(), /*is_protobuf_encoded=*/true));
  requests.push_back(std::move(task_assignment_http_request));

  if (eligibility_eval_enabled_) {
    FCP_ASSIGN_OR_RETURN(
        std::unique_ptr<HttpRequest>
            report_eligibility_eval_result_http_request,
        CreateReportEligibilityEvalTaskResultRequest(absl::OkStatus()));
    requests.push_back(std::move(report_eligibility_eval_result_http_request));
  }

  // Issue the request.
  FCP_ASSIGN_OR_RETURN(
      std::vector<absl::StatusOr<InMemoryHttpResponse>> responses,
      protocol_request_helper_.PerformMultipleProtocolRequests(
          std::move(requests), *interruptible_runner_));
  // The responses are returned in order. The first one is for the task
  // assignment request. The second one (optional) is for the report eligibility
  // eval task result request.  We only care about the first one.
  if (eligibility_eval_enabled_ && !responses[1].ok()) {
    log_manager_->LogDiag(
        ProdDiagCode::HTTP_REPORT_ELIGIBILITY_EVAL_RESULT_REQUEST_FAILED);
  }
  return responses[0];
}

absl::StatusOr<FederatedProtocol::CheckinResult>
HttpFederatedProtocol::HandleTaskAssignmentOperationResponse(
    absl::StatusOr<InMemoryHttpResponse> http_response,
    std::function<void(const TaskAssignment&)> payload_uris_received_callback) {
  // If the initial response was not successful, then return immediately, even
  // if the result was CANCELLED, since we won't have received an operation name
  // to issue a CancelOperationRequest with anyway.
  absl::StatusOr<Operation> initial_operation =
      ParseOperationProtoFromHttpResponse(http_response);
  if (!initial_operation.ok()) {
    return absl::Status(initial_operation.status().code(),
                        absl::StrCat("protocol request failed: ",
                                     initial_operation.status().ToString()));
  }
  absl::StatusOr<Operation> response_operation_proto =
      protocol_request_helper_.PollOperationResponseUntilDone(
          *initial_operation, *task_assignment_request_creator_,
          *interruptible_runner_);
  if (!response_operation_proto.ok()) {
    // If the protocol request failed then issue a cancellation request to let
    // the server know the operation will be abandoned, and forward the error,
    // but add a prefix to the error message to ensure we can easily distinguish
    // an HTTP error occurring in response to the protocol request from HTTP
    // errors occurring during checkpoint/plan resource fetch requests later on.
    FCP_ASSIGN_OR_RETURN(std::string operation_name,
                         ExtractOperationName(*initial_operation));
    // Client interruption
    std::unique_ptr<InterruptibleRunner> cancellation_runner =
        CreateDelayedInterruptibleRunner(
            log_manager_, should_abort_, timing_config_,
            absl::Now() + waiting_period_for_cancellation_);
    if (!protocol_request_helper_
             .CancelOperation(operation_name, *task_assignment_request_creator_,
                              *cancellation_runner)
             .ok()) {
      log_manager_->LogDiag(
          ProdDiagCode::HTTP_CANCELLATION_OR_ABORT_REQUEST_FAILED);
    }
    return absl::Status(
        response_operation_proto.status().code(),
        absl::StrCat("protocol request failed: ",
                     response_operation_proto.status().ToString()));
  }

  // The Operation has finished. Check if it resulted in an error, and if so
  // forward it after converting it to an absl::Status error.
  if (response_operation_proto->has_error()) {
    auto rpc_error =
        ConvertRpcStatusToAbslStatus(response_operation_proto->error());
    return absl::Status(
        rpc_error.code(),
        absl::StrCat("Operation ", response_operation_proto->name(),
                     " contained error: ", rpc_error.ToString()));
  }

  // Otherwise, handle the StartTaskAssignmentResponse that should have been
  // returned by the Operation response proto.
  return HandleTaskAssignmentInnerResponse(response_operation_proto->response(),
                                           payload_uris_received_callback);
}

absl::StatusOr<FederatedProtocol::CheckinResult>
HttpFederatedProtocol::HandleTaskAssignmentInnerResponse(
    const ::google::protobuf::Any& operation_response,
    std::function<void(const TaskAssignment&)> payload_uris_received_callback) {
  StartTaskAssignmentResponse response_proto;
  if (!operation_response.UnpackTo(&response_proto)) {
    return absl::InvalidArgumentError(
        "could not parse StartTaskAssignmentResponse proto");
  }
  if (response_proto.has_rejection_info()) {
    object_state_ = ObjectState::kCheckinRejected;
    return Rejection{};
  }
  if (!response_proto.has_task_assignment()) {
    return absl::UnimplementedError("Unrecognized StartTaskAssignmentResponse");
  }
  const auto& task_assignment = response_proto.task_assignment();

  FCP_ASSIGN_OR_RETURN(
      aggregation_request_creator_,
      ProtocolRequestCreator::Create(
          api_key_, task_assignment.aggregation_data_forwarding_info(),
          !flags_->disable_http_request_body_compression()));

  TaskAssignment result = {
      .federated_select_uri_template =
          task_assignment.federated_select_uri_info().uri_template(),
      .aggregation_session_id = task_assignment.aggregation_id(),
      .sec_agg_info = std::nullopt};
  if (task_assignment.has_secure_aggregation_info()) {
    result.sec_agg_info =
        SecAggInfo{.minimum_clients_in_server_visible_aggregate =
                       task_assignment.secure_aggregation_info()
                           .minimum_clients_in_server_visible_aggregate()};
  }

  payload_uris_received_callback(result);

  // Fetch the task resources, returning any errors that may be encountered in
  // the process.
  FCP_ASSIGN_OR_RETURN(
      result.payloads,
      FetchTaskResources({.plan = task_assignment.plan(),
                          .checkpoint = task_assignment.init_checkpoint()}));

  object_state_ = ObjectState::kCheckinAccepted;
  session_id_ = task_assignment.session_id();
  aggregation_session_id_ = task_assignment.aggregation_id();
  // TODO(team): Consider renaming aggregation_client_token_ to
  // aggregation_authorization_token_.
  aggregation_client_token_ = task_assignment.authorization_token();

  return std::move(result);
}

absl::Status HttpFederatedProtocol::ReportCompleted(
    ComputationResults results,
    absl::Duration plan_duration) {
  FCP_LOG(INFO) << "Reporting outcome: " << static_cast<int>(engine::COMPLETED);
  FCP_CHECK(object_state_ == ObjectState::kCheckinAccepted)
      << "Invalid call sequence";
  object_state_ = ObjectState::kReportCalled;
  auto find_secagg_tensor_lambda = [](const auto& item) {
    return std::holds_alternative<QuantizedTensor>(item.second);
  };
  if (std::find_if(results.begin(), results.end(), find_secagg_tensor_lambda) ==
      results.end()) {
    return ReportViaSimpleAggregation(std::move(results), plan_duration);
  } else {
    return ReportViaSecureAggregation(std::move(results), plan_duration);
  }
}

absl::Status HttpFederatedProtocol::ReportViaSimpleAggregation(
    ComputationResults results, absl::Duration plan_duration) {
  if (results.size() != 1 ||
      !std::holds_alternative<TFCheckpoint>(results.begin()->second)) {
    return absl::InternalError(
        "Simple Aggregation aggregands have unexpected format.");
  }
  auto start_upload_status = HandleStartDataAggregationUploadOperationResponse(
      PerformStartDataUploadRequestAndReportTaskResult(plan_duration));
  if (!start_upload_status.ok()) {
    object_state_ = ObjectState::kReportFailedPermanentError;
    // We only issue AbortAggregation when the error is not kAborted, which
    // means the server did not need our data anymore and won't expect any more
    // requests from this client. In all other cases we assume that the server
    // should still be informed that this client is about to terminate its
    // protocol and won't send any further requests.
    if (start_upload_status.code() != absl::StatusCode::kAborted &&
        !AbortAggregation(start_upload_status,
                          "StartDataAggregationUpload failed.")
             .ok()) {
      log_manager_->LogDiag(
          ProdDiagCode::HTTP_CANCELLATION_OR_ABORT_REQUEST_FAILED);
    }
    return start_upload_status;
  }
  auto upload_status = UploadDataViaSimpleAgg(
      std::get<TFCheckpoint>(std::move(results.begin()->second)));
  if (!upload_status.ok()) {
    object_state_ = ObjectState::kReportFailedPermanentError;
    if (upload_status.code() != absl::StatusCode::kAborted &&
        !AbortAggregation(upload_status,
                          "Upload data via simple aggregation failed.")
             .ok()) {
      log_manager_->LogDiag(
          ProdDiagCode::HTTP_CANCELLATION_OR_ABORT_REQUEST_FAILED);
    }
    return upload_status;
  }
  return SubmitAggregationResult();
}

absl::StatusOr<InMemoryHttpResponse>
HttpFederatedProtocol::PerformStartDataUploadRequestAndReportTaskResult(
    absl::Duration plan_duration) {
  FCP_ASSIGN_OR_RETURN(
      ReportTaskResultRequest report_task_result_request,
      CreateReportTaskResultRequest(engine::PhaseOutcome::COMPLETED,
                                    plan_duration, aggregation_session_id_));
  FCP_ASSIGN_OR_RETURN(
      std::string report_task_result_uri_suffix,
      CreateReportTaskResultUriSuffix(population_name_, session_id_));
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<HttpRequest> http_report_task_result_request,
      task_assignment_request_creator_->CreateProtocolRequest(
          report_task_result_uri_suffix, {}, HttpRequest::Method::kPost,
          report_task_result_request.SerializeAsString(),
          /*is_protobuf_encoded=*/true));

  StartAggregationDataUploadRequest start_upload_request;
  FCP_ASSIGN_OR_RETURN(std::string start_aggregation_data_upload_uri_suffix,
                       CreateStartAggregationDataUploadUriSuffix(
                           aggregation_session_id_, aggregation_client_token_));
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<HttpRequest> http_start_aggregation_data_upload_request,
      aggregation_request_creator_->CreateProtocolRequest(
          start_aggregation_data_upload_uri_suffix, {},
          HttpRequest::Method::kPost, start_upload_request.SerializeAsString(),
          /*is_protobuf_encoded=*/true));
  std::vector<std::unique_ptr<HttpRequest>> requests;
  requests.push_back(std::move(http_start_aggregation_data_upload_request));
  requests.push_back(std::move(http_report_task_result_request));
  FCP_ASSIGN_OR_RETURN(
      std::vector<absl::StatusOr<InMemoryHttpResponse>> responses,
      protocol_request_helper_.PerformMultipleProtocolRequests(
          std::move(requests), *interruptible_runner_));
  // We should have two responses, otherwise we have made a developer error.
  FCP_CHECK(responses.size() == 2);
  // The responses are returned in order so the first response will be the one
  // for StartAggregationDataUpload request.  We only care about this response,
  // the ReportTaskResult request is just a best effort to report client metrics
  // to the server, and we don't want to abort the aggregation even if it
  // failed.
  if (!responses[1].ok()) {
    log_manager_->LogDiag(ProdDiagCode::HTTP_REPORT_TASK_RESULT_REQUEST_FAILED);
  }
  return responses[0];
}

absl::Status
HttpFederatedProtocol::HandleStartDataAggregationUploadOperationResponse(
    absl::StatusOr<InMemoryHttpResponse> http_response) {
  absl::StatusOr<Operation> operation =
      ParseOperationProtoFromHttpResponse(http_response);
  if (!operation.ok()) {
    // If the protocol request failed then forward the error, but add a prefix
    // to the error message to ensure we can easily distinguish an HTTP error
    // occurring in response to the protocol request from HTTP errors
    // occurring during upload requests later on.
    return absl::Status(
        operation.status().code(),
        absl::StrCat(
            "StartAggregationDataUpload request failed during polling: ",
            operation.status().ToString()));
  }
  absl::StatusOr<Operation> response_operation_proto =
      protocol_request_helper_.PollOperationResponseUntilDone(
          *operation, *aggregation_request_creator_, *interruptible_runner_);
  if (!response_operation_proto.ok()) {
    return absl::Status(
        response_operation_proto.status().code(),
        absl::StrCat("StartAggregationDataUpload request failed: ",
                     response_operation_proto.status().ToString()));
  }

  // The Operation has finished. Check if it resulted in an error, and if so
  // forward it after converting it to an absl::Status error.
  if (response_operation_proto->has_error()) {
    auto rpc_error =
        ConvertRpcStatusToAbslStatus(response_operation_proto->error());
    return absl::Status(
        rpc_error.code(),
        absl::StrCat("Operation ", response_operation_proto->name(),
                     " contained error: ", rpc_error.ToString()));
  }

  // Otherwise, handle the StartDataAggregationUploadResponse that should have
  // been returned by the Operation response proto.

  StartAggregationDataUploadResponse response_proto;
  if (!response_operation_proto->response().UnpackTo(&response_proto)) {
    return absl::InvalidArgumentError(
        "could not parse StartTaskAssignmentResponse proto");
  }

  // Note that we reassign `aggregation_request_creator_` because from this
  // point onwards, subsequent aggregation protocol requests should go to the
  // endpoint identified in the aggregation_protocol_forwarding_info.
  FCP_ASSIGN_OR_RETURN(
      aggregation_request_creator_,
      ProtocolRequestCreator::Create(
          api_key_, response_proto.aggregation_protocol_forwarding_info(),
          !flags_->disable_http_request_body_compression()));
  auto upload_resource = response_proto.resource();
  aggregation_resource_name_ = upload_resource.resource_name();
  FCP_ASSIGN_OR_RETURN(
      data_upload_request_creator_,
      ProtocolRequestCreator::Create(
          api_key_, upload_resource.data_upload_forwarding_info(),
          !flags_->disable_http_request_body_compression()));
  return absl::OkStatus();
}

absl::Status HttpFederatedProtocol::UploadDataViaSimpleAgg(
    std::string tf_checkpoint) {
  FCP_LOG(INFO) << "Uploading checkpoint with simple aggregation.";
  FCP_ASSIGN_OR_RETURN(std::string uri_suffix, CreateByteStreamUploadUriSuffix(
                                                   aggregation_resource_name_));
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<HttpRequest> http_request,
      data_upload_request_creator_->CreateProtocolRequest(
          uri_suffix, {{"upload_protocol", "raw"}}, HttpRequest::Method::kPost,
          std::move(tf_checkpoint), /*is_protobuf_encoded=*/false));
  auto http_response = protocol_request_helper_.PerformProtocolRequest(
      std::move(http_request), *interruptible_runner_);
  if (!http_response.ok()) {
    // If the request failed, we'll forward the error status.
    return absl::Status(http_response.status().code(),
                        absl::StrCat("Data upload failed: ",
                                     http_response.status().ToString()));
  }
  return absl::OkStatus();
}

absl::Status HttpFederatedProtocol::SubmitAggregationResult() {
  FCP_LOG(INFO) << "Notifying the server that data upload is complete.";
  FCP_ASSIGN_OR_RETURN(std::string uri_suffix,
                       CreateSubmitAggregationResultUriSuffix(
                           aggregation_session_id_, aggregation_client_token_));
  SubmitAggregationResultRequest request;
  request.set_resource_name(aggregation_resource_name_);
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<HttpRequest> http_request,
      aggregation_request_creator_->CreateProtocolRequest(
          uri_suffix, {}, HttpRequest::Method::kPost,
          request.SerializeAsString(), /*is_protobuf_encoded=*/true));
  auto http_response = protocol_request_helper_.PerformProtocolRequest(
      std::move(http_request), *interruptible_runner_);
  if (!http_response.ok()) {
    // If the request failed, we'll forward the error status.
    return absl::Status(http_response.status().code(),
                        absl::StrCat("SubmitAggregationResult failed: ",
                                     http_response.status().ToString()));
  }
  return absl::OkStatus();
}

absl::Status HttpFederatedProtocol::AbortAggregation(
    absl::Status original_error_status,
    absl::string_view error_message_for_server) {
  FCP_LOG(INFO) << "Aborting aggregation: " << original_error_status;
  FCP_CHECK(object_state_ == ObjectState::kReportFailedPermanentError)
      << "Invalid call sequence";
  FCP_ASSIGN_OR_RETURN(std::string uri_suffix,
                       CreateAbortAggregationUriSuffix(
                           aggregation_session_id_, aggregation_client_token_));
  // We only provide the server with a simplified error message.
  absl::Status error_status(original_error_status.code(),
                            error_message_for_server);
  AbortAggregationRequest request;
  *request.mutable_status() = ConvertAbslStatusToRpcStatus(error_status);
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<HttpRequest> http_request,
      aggregation_request_creator_->CreateProtocolRequest(
          uri_suffix, {}, HttpRequest::Method::kPost,
          request.SerializeAsString(), /*is_protobuf_encoded=*/true));
  std::unique_ptr<InterruptibleRunner> cancellation_runner =
      CreateDelayedInterruptibleRunner(
          log_manager_, should_abort_, timing_config_,
          absl::Now() + waiting_period_for_cancellation_);
  return protocol_request_helper_
      .PerformProtocolRequest(std::move(http_request), *cancellation_runner)
      .status();
}

absl::Status HttpFederatedProtocol::ReportViaSecureAggregation(
    ComputationResults results, absl::Duration plan_duration) {
  FCP_ASSIGN_OR_RETURN(
      StartSecureAggregationResponse response_proto,
      StartSecureAggregationAndReportTaskResult(plan_duration));
  SecureAggregationProtocolExecutionInfo protocol_execution_info =
      response_proto.protocol_execution_info();

  // Move checkpoint out of ComputationResults, and put it into a std::optional.
  std::optional<TFCheckpoint> tf_checkpoint;
  for (auto& [k, v] : results) {
    if (std::holds_alternative<TFCheckpoint>(v)) {
      tf_checkpoint = std::get<TFCheckpoint>(std::move(v));
      results.erase(k);
      break;
    }
  }
  absl::StatusOr<secagg::ServerToClientWrapperMessage> server_response_holder;
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<SecAggSendToServerBase> send_to_server_impl,
      HttpSecAggSendToServerImpl::Create(
          api_key_, &clock_, &protocol_request_helper_,
          interruptible_runner_.get(),
          [this](absl::Time deadline) {
            return CreateDelayedInterruptibleRunner(
                this->log_manager_, this->should_abort_, this->timing_config_,
                deadline);
          },
          &server_response_holder, aggregation_session_id_,
          aggregation_client_token_,
          response_proto.secagg_protocol_forwarding_info(),
          response_proto.masked_result_resource(),
          response_proto.nonmasked_result_resource(), std::move(tf_checkpoint),
          flags_->disable_http_request_body_compression(),
          waiting_period_for_cancellation_));
  auto protocol_delegate = std::make_unique<HttpSecAggProtocolDelegate>(
      response_proto.secure_aggregands(), &server_response_holder);
  auto secagg_interruptible_runner = std::make_unique<InterruptibleRunner>(
      log_manager_, should_abort_, timing_config_,
      InterruptibleRunner::DiagnosticsConfig{
          .interrupted = ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP,
          .interrupt_timeout =
              ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP_TIMED_OUT,
          .interrupted_extended = ProdDiagCode::
              BACKGROUND_TRAINING_INTERRUPT_HTTP_EXTENDED_COMPLETED,
          .interrupt_timeout_extended = ProdDiagCode::
              BACKGROUND_TRAINING_INTERRUPT_HTTP_EXTENDED_TIMED_OUT});
  std::unique_ptr<SecAggRunner> secagg_runner =
      secagg_runner_factory_->CreateSecAggRunner(
          std::move(send_to_server_impl), std::move(protocol_delegate),
          secagg_event_publisher_, log_manager_,
          secagg_interruptible_runner.get(),
          protocol_execution_info.expected_number_of_clients(),
          protocol_execution_info
              .minimum_surviving_clients_for_reconstruction());
  FCP_RETURN_IF_ERROR(secagg_runner->Run(std::move(results)));
  return absl::OkStatus();
}

absl::StatusOr<StartSecureAggregationResponse>
HttpFederatedProtocol::StartSecureAggregationAndReportTaskResult(
    absl::Duration plan_duration) {
  FCP_ASSIGN_OR_RETURN(std::string start_secure_aggregation_uri_suffix,
                       CreateStartSecureAggregationUriSuffix(
                           aggregation_session_id_, aggregation_client_token_));
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<HttpRequest> start_secure_aggregation_http_request,
      aggregation_request_creator_->CreateProtocolRequest(
          start_secure_aggregation_uri_suffix, QueryParams(),
          HttpRequest::Method::kPost,
          StartSecureAggregationRequest::default_instance().SerializeAsString(),
          /*is_protobuf_encoded=*/true));

  FCP_ASSIGN_OR_RETURN(
      std::string report_task_result_uri_suffix,
      CreateReportTaskResultUriSuffix(population_name_, session_id_));
  FCP_ASSIGN_OR_RETURN(
      ReportTaskResultRequest report_task_result_request,
      CreateReportTaskResultRequest(engine::PhaseOutcome::COMPLETED,
                                    plan_duration, aggregation_session_id_));
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<HttpRequest> report_task_result_http_request,
      task_assignment_request_creator_->CreateProtocolRequest(
          report_task_result_uri_suffix, QueryParams(),
          HttpRequest::Method::kPost,
          report_task_result_request.SerializeAsString(),
          /*is_protobuf_encoded=*/true));

  std::vector<std::unique_ptr<HttpRequest>> requests;
  requests.push_back(std::move(start_secure_aggregation_http_request));
  requests.push_back(std::move(report_task_result_http_request));

  FCP_ASSIGN_OR_RETURN(
      std::vector<absl::StatusOr<InMemoryHttpResponse>> responses,
      protocol_request_helper_.PerformMultipleProtocolRequests(
          std::move(requests), *interruptible_runner_));
  // We will handle the response for StartSecureAggregation RPC.
  // The ReportTaskResult RPC is for best efforts only, we will ignore the
  // response, only log a diagcode if it fails.
  FCP_CHECK(responses.size() == 2);
  if (!responses[1].ok()) {
    log_manager_->LogDiag(ProdDiagCode::HTTP_REPORT_TASK_RESULT_REQUEST_FAILED);
  }
  FCP_ASSIGN_OR_RETURN(Operation initial_operation,
                       ParseOperationProtoFromHttpResponse(responses[0]));
  FCP_ASSIGN_OR_RETURN(Operation completed_operation,
                       protocol_request_helper_.PollOperationResponseUntilDone(
                           initial_operation, *aggregation_request_creator_,
                           *interruptible_runner_));
  // The Operation has finished. Check if it resulted in an error, and if so
  // forward it after converting it to an absl::Status error.
  if (completed_operation.has_error()) {
    auto rpc_error = ConvertRpcStatusToAbslStatus(completed_operation.error());
    return absl::Status(
        rpc_error.code(),
        absl::StrCat("Operation ", completed_operation.name(),
                     " contained error: ", rpc_error.ToString()));
  }
  StartSecureAggregationResponse response_proto;
  if (!completed_operation.response().UnpackTo(&response_proto)) {
    return absl::InvalidArgumentError(
        "could not parse StartSecureAggregationResponse proto");
  }
  return response_proto;
}

absl::Status HttpFederatedProtocol::ReportNotCompleted(
    engine::PhaseOutcome phase_outcome, absl::Duration plan_duration) {
  FCP_LOG(WARNING) << "Reporting outcome: " << static_cast<int>(phase_outcome);
  FCP_CHECK(object_state_ == ObjectState::kCheckinAccepted)
      << "Invalid call sequence";
  object_state_ = ObjectState::kReportCalled;
  FCP_ASSIGN_OR_RETURN(
      ReportTaskResultRequest request,
      CreateReportTaskResultRequest(phase_outcome, plan_duration,
                                    aggregation_session_id_));
  // Construct the URI suffix.
  FCP_ASSIGN_OR_RETURN(
      std::string uri_suffix,
      CreateReportTaskResultUriSuffix(population_name_, session_id_));
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<HttpRequest> http_request,
      task_assignment_request_creator_->CreateProtocolRequest(
          uri_suffix, {}, HttpRequest::Method::kPost,
          request.SerializeAsString(), /*is_protobuf_encoded=*/true));

  // Issue the request.
  absl::StatusOr<InMemoryHttpResponse> http_response =
      protocol_request_helper_.PerformProtocolRequest(std::move(http_request),
                                                      *interruptible_runner_);
  if (!http_response.ok()) {
    // If the request failed, we'll forward the error status.
    return absl::Status(http_response.status().code(),
                        absl::StrCat("ReportTaskResult request failed: ",
                                     http_response.status().ToString()));
  }
  return absl::OkStatus();
}

::google::internal::federatedml::v2::RetryWindow
HttpFederatedProtocol::GetLatestRetryWindow() {
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
      FCP_CHECK(retry_times_.has_value());
      return GenerateRetryWindowFromRetryTime(
          retry_times_->retry_time_if_accepted);
    case ObjectState::kEligibilityEvalCheckinRejected:
    case ObjectState::kEligibilityEvalDisabled:
    case ObjectState::kEligibilityEvalEnabled:
    case ObjectState::kCheckinRejected:
      FCP_CHECK(retry_times_.has_value());
      return GenerateRetryWindowFromRetryTime(
          retry_times_->retry_time_if_rejected);
    case ObjectState::kInitialized:
    case ObjectState::kEligibilityEvalCheckinFailed:
    case ObjectState::kCheckinFailed:
      if (retry_times_.has_value()) {
        // If we already received a server-provided retry window, then use it.
        return GenerateRetryWindowFromRetryTime(
            retry_times_->retry_time_if_rejected);
      }
      // Otherwise, we generate a retry window using the flag-provided transient
      // error retry period.
      return GenerateRetryWindowFromTargetDelay(
          absl::Seconds(
              flags_->federated_training_transient_errors_retry_delay_secs()),
          // NOLINTBEGIN(whitespace/line_length)
          flags_
              ->federated_training_transient_errors_retry_delay_jitter_percent(),
          // NOLINTEND(whitespace/line_length)
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
          // NOLINTEND(whitespace/line_length)
          bit_gen_);
  }
}

absl::StatusOr<FederatedProtocol::PlanAndCheckpointPayloads>
HttpFederatedProtocol::FetchTaskResources(
    HttpFederatedProtocol::TaskResources task_resources) {
  FCP_ASSIGN_OR_RETURN(UriOrInlineData plan_uri_or_data,
                       ConvertResourceToUriOrInlineData(task_resources.plan));
  FCP_ASSIGN_OR_RETURN(
      UriOrInlineData checkpoint_uri_or_data,
      ConvertResourceToUriOrInlineData(task_resources.checkpoint));

  // Fetch the plan and init checkpoint resources if they need to be fetched
  // (using the inline data instead if available).
  absl::StatusOr<std::vector<absl::StatusOr<InMemoryHttpResponse>>>
      resource_responses;
  {
    auto started_stopwatch = network_stopwatch_->Start();
    resource_responses = FetchResourcesInMemory(
        *http_client_, *interruptible_runner_,
        {plan_uri_or_data, checkpoint_uri_or_data}, &bytes_downloaded_,
        &bytes_uploaded_, resource_cache_);
  }
  FCP_RETURN_IF_ERROR(resource_responses);
  auto& plan_data_response = (*resource_responses)[0];
  auto& checkpoint_data_response = (*resource_responses)[1];

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

  return PlanAndCheckpointPayloads{plan_data_response->body,
                                   checkpoint_data_response->body};
}

void HttpFederatedProtocol::UpdateObjectStateIfPermanentError(
    absl::Status status,
    HttpFederatedProtocol::ObjectState permanent_error_object_state) {
  if (federated_training_permanent_error_codes_.contains(
          static_cast<int32_t>(status.code()))) {
    object_state_ = permanent_error_object_state;
  }
}

NetworkStats HttpFederatedProtocol::GetNetworkStats() {
  return {.bytes_downloaded = bytes_downloaded_,
          .bytes_uploaded = bytes_uploaded_,
          .network_duration = network_stopwatch_->GetTotalDuration()};
}

}  // namespace http
}  // namespace client
}  // namespace fcp
