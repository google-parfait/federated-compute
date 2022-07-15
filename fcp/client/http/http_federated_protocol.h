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
#ifndef FCP_CLIENT_HTTP_HTTP_FEDERATED_PROTOCOL_H_
#define FCP_CLIENT_HTTP_HTTP_FEDERATED_PROTOCOL_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/in_memory_request_response.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/selector_context.pb.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/federatedcompute/common.pb.h"
#include "fcp/protos/federatedcompute/eligibility_eval_tasks.pb.h"
#include "fcp/protos/federatedcompute/task_assignments.pb.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace http {

// Note the uri query parameters should be percent encoded.
using QueryParams = absl::flat_hash_map<std::string, std::string>;

// A helper for creating HTTP request with base uri, request headers and
// compression setting.
class ProtocolRequestCreator {
 public:
  ProtocolRequestCreator(absl::string_view request_base_uri,
                         HeaderList request_headers, bool use_compression);

  // Creates a `ProtocolRequestCreator` based on the forwarding info.
  // Validates and extracts the base URI and headers to use for the subsequent
  // request(s).
  static absl::StatusOr<std::unique_ptr<ProtocolRequestCreator>> Create(
      const ::google::internal::federatedcompute::v1::ForwardingInfo&
          forwarding_info,
      bool use_compression);

  // Creates an `HttpRequest` with base uri, request headers and compression
  // setting. The `uri_path_suffix` argument must always either be empty or
  // start with a leading '/'. The method will return `InvalidArgumentError` if
  // this isn't the case.  The `uri_path_suffix` should not contain any query
  // parameters, instead, query parameters should be specified in `params`.
  //
  // The URI to which the protocol request will be sent will be constructed by
  // joining `next_request_base_uri_` with `uri_path_suffix` (see
  // `JoinBaseUriWithSuffix` for details), and any query parameters if `params`
  // is not empty.
  //
  // When `is_protobuf_encoded` is true, `%24alt=proto` will be added to the uri
  // as a query parameter to indicate that the proto encoded payload is
  // expected. When the `request_body` is not empty, a `Content-Type` header
  // will also be added to the request
  absl::StatusOr<std::unique_ptr<HttpRequest>> CreateProtocolRequest(
      absl::string_view uri_path_suffix, QueryParams params,
      HttpRequest::Method method, std::string request_body,
      bool is_protobuf_encoded) const;

  // Creates an `HttpRequest` for getting the result of a
  // `google.longrunning.operation`. Note that the request body is empty,
  // because its only field (`name`) is included in the URI instead. Also note
  // that the `next_request_headers_` will be attached to this request.
  absl::StatusOr<std::unique_ptr<HttpRequest>> CreateGetOperationRequest(
      absl::string_view operation_name) const;

 private:
  absl::StatusOr<std::unique_ptr<HttpRequest>> CreateHttpRequest(
      absl::string_view uri_path_suffix, QueryParams params,
      HttpRequest::Method method, std::string request_body,
      bool is_protobuf_encoded, bool use_compression) const;
  // The URI to use for the next protocol request. See `ForwardingInfo`.
  std::string next_request_base_uri_;
  // The set of headers to attach to the next protocol request. See
  // `ForwardingInfo`.
  HeaderList next_request_headers_;
  const bool use_compression_;
};

// A helper for issuing protocol requests.
class ProtocolRequestHelper {
 public:
  ProtocolRequestHelper(HttpClient* http_client, int64_t* bytes_downloaded,
                        int64_t* bytes_uploaded,
                        bool client_decoded_http_resources);

  // Performs the given request (handling any interruptions that may occur) and
  // updates the network stats.
  absl::StatusOr<InMemoryHttpResponse> PerformProtocolRequest(
      std::unique_ptr<HttpRequest> request, InterruptibleRunner& runner);

  // Performs the vector of requests (handling any interruptions that may occur)
  // concurrently and updates the network stats.
  // The returned vector of responses has the same order of the issued requests.
  absl::StatusOr<std::vector<absl::StatusOr<InMemoryHttpResponse>>>
  PerformMultipleProtocolRequests(
      std::vector<std::unique_ptr<HttpRequest>> requests,
      InterruptibleRunner& runner);

  // Helper function for handling an HTTP response that contains an `Operation`
  // proto.
  //
  // Takes an HTTP response (which must have been produced by a call to
  // `PerformRequestInMemory`), parses the proto, and returns it if its
  // `Operation.done` field is true. If the field is false then this method
  // keeps polling the Operation via performing requests created by
  // `CreateGetOperationRequest` until it a response is received where the field
  // is true, at which point that most recent response is returned. If at any
  // point an HTTP or response parsing error is encountered, then that error is
  // returned instead.
  absl::StatusOr<::google::longrunning::Operation>
  PollOperationResponseUntilDone(
      absl::StatusOr<InMemoryHttpResponse> http_response,
      const ProtocolRequestCreator& request_creator,
      InterruptibleRunner& runner);

 private:
  HttpClient& http_client_;
  int64_t& bytes_downloaded_;
  int64_t& bytes_uploaded_;
  const bool client_decoded_http_resources_;
};

// Implements a single session of the HTTP-based Federated Compute protocol.
class HttpFederatedProtocol : public fcp::client::FederatedProtocol {
 public:
  HttpFederatedProtocol(
      LogManager* log_manager, const Flags* flags, HttpClient* http_client,
      absl::string_view entry_point_uri, absl::string_view api_key,
      absl::string_view population_name, absl::string_view retry_token,
      absl::string_view client_version,
      absl::string_view attestation_measurement,
      std::function<bool()> should_abort, absl::BitGen bit_gen,
      const InterruptibleRunner::TimingConfig& timing_config);

  ~HttpFederatedProtocol() override = default;

  absl::StatusOr<fcp::client::FederatedProtocol::EligibilityEvalCheckinResult>
  EligibilityEvalCheckin() override;

  absl::StatusOr<fcp::client::FederatedProtocol::CheckinResult> Checkin(
      const std::optional<
          google::internal::federatedml::v2::TaskEligibilityInfo>&
          task_eligibility_info) override;

  absl::Status ReportCompleted(
      ComputationResults results,
      absl::Duration plan_duration) override;

  absl::Status ReportNotCompleted(engine::PhaseOutcome phase_outcome,
                                  absl::Duration plan_duration) override;

  google::internal::federatedml::v2::RetryWindow GetLatestRetryWindow()
      override;

  int64_t bytes_downloaded() override;

  int64_t bytes_uploaded() override;

  int64_t chunking_layer_bytes_received() override;

  int64_t chunking_layer_bytes_sent() override;

  int64_t report_request_size_bytes() override;

 private:
  // Helper function to perform an eligibility eval task request and get its
  // response.
  absl::StatusOr<InMemoryHttpResponse> PerformEligibilityEvalTaskRequest();

  // Helper function for handling an eligibility eval task response (incl.
  // fetching any resources, if necessary).
  absl::StatusOr<fcp::client::FederatedProtocol::EligibilityEvalCheckinResult>
  HandleEligibilityEvalTaskResponse(
      absl::StatusOr<InMemoryHttpResponse> http_response);

  // Helper function to perform a task assignment request and get its response.
  absl::StatusOr<InMemoryHttpResponse> PerformTaskAssignmentRequest(
      const std::optional<
          ::google::internal::federatedml::v2::TaskEligibilityInfo>&
          task_eligibility_info);

  // Helper function for handling the 'outer' task assignment response, which
  // consists of an `Operation` which may or may not need to be polled before a
  // final 'inner' response is available.
  absl::StatusOr<::fcp::client::FederatedProtocol::CheckinResult>
  HandleTaskAssignmentOperationResponse(
      absl::StatusOr<InMemoryHttpResponse> http_response);

  // Helper function for handling an 'inner' task assignment response (i.e.
  // after the outer `Operation` has concluded). This includes fetching any
  // resources, if necessary.
  absl::StatusOr<::fcp::client::FederatedProtocol::CheckinResult>
  HandleTaskAssignmentInnerResponse(
      const ::google::protobuf::Any& operation_response);

  // Helper function to perform a StartDataUploadRequest and a ReportTaskResult
  // request concurrently.
  // This method will only return the response from the StartDataUploadRequest.
  absl::StatusOr<InMemoryHttpResponse>
  PerformStartDataUploadRequestAndReportTaskResult(
      absl::Duration plan_duration);

  // Helper function for handling a longrunning operation returned by a
  // StartDataAggregationUpload request.
  absl::Status HandleStartDataAggregationUploadOperationResponse(
      absl::StatusOr<InMemoryHttpResponse> http_response);

  // Helper function to perform data upload via simple aggregation.
  absl::Status UploadDataViaSimpleAgg(std::string tf_checkpoint);

  // Helper function to perform a SubmitAggregationResult request.
  absl::Status SubmitAggregationResult();

  // Helper function to perform an AbortAggregation request.
  // We only provide the server with a simplified error message.
  absl::Status AbortAggregation(absl::Status original_error_status,
                                absl::string_view error_message_for_server);

  struct TaskResources {
    const ::google::internal::federatedcompute::v1::Resource& plan;
    const ::google::internal::federatedcompute::v1::Resource& checkpoint;
  };

  // Helper function for fetching the checkpoint/plan resources for an
  // eligibility eval task or regular task.
  absl::StatusOr<PlanAndCheckpointPayloads> FetchTaskResources(
      TaskResources task_resources);

  // Helper that moves to the given object state if the given status represents
  // a permanent error.
  void UpdateObjectStateIfPermanentError(
      absl::Status status, ObjectState permanent_error_object_state);

  ObjectState object_state_;
  LogManager* log_manager_;
  const Flags* const flags_;
  HttpClient* const http_client_;
  std::unique_ptr<InterruptibleRunner> interruptible_runner_;
  std::unique_ptr<ProtocolRequestCreator> eligibility_eval_request_creator_;
  std::unique_ptr<ProtocolRequestCreator> task_assignment_request_creator_;
  std::unique_ptr<ProtocolRequestCreator> aggregation_request_creator_;
  std::unique_ptr<ProtocolRequestCreator> data_upload_request_creator_;
  ProtocolRequestHelper protocol_request_helper_;
  const std::string api_key_;
  const std::string population_name_;
  const std::string retry_token_;
  const std::string client_version_;
  const std::string attestation_measurement_;
  std::function<bool()> should_abort_;
  absl::BitGen bit_gen_;
  const InterruptibleRunner::TimingConfig& timing_config_;
  // The graceful waiting period for cancellation requests before checking
  // whether the client should be interrupted.
  const absl::Duration waiting_period_for_cancellation_;
  // The set of canonical error codes that should be treated as 'permanent'
  // errors.
  absl::flat_hash_set<int32_t> federated_training_permanent_error_codes_;
  int64_t bytes_downloaded_ = 0;
  int64_t bytes_uploaded_ = 0;
  int64_t report_request_size_bytes_ = 0;
  // Represents 2 absolute retry timestamps to use when the device is rejected
  // or accepted. The retry timestamps will have been generated based on the
  // retry windows specified in the server's EligibilityEvalTaskResponse message
  // and the time at which that message was received.
  struct RetryTimes {
    absl::Time retry_time_if_rejected;
    absl::Time retry_time_if_accepted;
  };
  // Represents the information received via the EligibilityEvalTaskResponse
  // message. This field will have an absent value until that message has been
  // received.
  std::optional<RetryTimes> retry_times_;
  std::string session_id_;
  // The identifier of the aggregation session we are participating in (or empty
  // if that phase of the protocol hasn't been reached yet).
  std::string aggregation_session_id_;
  // Unique identifier for the client's participation in an aggregation session.
  std::string aggregation_client_token_;
  // Resource name for the checkpoint in simple aggregation.
  std::string aggregation_resource_name_;
};

}  // namespace http
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_HTTP_HTTP_FEDERATED_PROTOCOL_H_
