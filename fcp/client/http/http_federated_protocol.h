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

// A helper for managing a chain of protocol requests, with each request being
// pointed at an endpoint specified by a previous request's
// `ForwardingInfo.target_uri_prefix`.
class ProtocolRequestHelper {
 public:
  ProtocolRequestHelper(HttpClient* http_client,
                        InterruptibleRunner* interruptible_runner,
                        int64_t* bytes_downloaded, int64_t* bytes_uploaded,
                        absl::string_view entry_point_uri);

  // Performs the given request (handling any interruptions that may occur) and
  // updates the network stats.
  //
  // The `uri_suffix` argument must always either be empty or start with a
  // leading '/'. The method will CHECK-fail if this isn't the case.
  //
  // The URI to which the protocol request will be sent will be constructed by
  // joining `next_request_base_uri_` with `uri_suffix` (see
  // `JoinBaseUriWithSuffix` for details).
  absl::StatusOr<InMemoryHttpResponse> PerformProtocolRequest(
      absl::string_view uri_suffix, HttpRequest::Method method,
      std::string request_body);

  // Helper function for handling an HTTP response that contains an `Operation`
  // proto.
  //
  // Takes an HTTP response (which must have been produced by a call to
  // `PerformRequestInMemory`), parses the proto, and returns it if its
  // `Operation.done` field is true. If the field is false then this method
  // keeps polling the Operation via `PerformGetOperationRequest` until it a
  // response is received where the field is true, at which point that most
  // recent response is returned. If at any point an HTTP or response parsing
  // error is encountered, then that error is returned instead.
  absl::StatusOr<::google::longrunning::Operation>
  PollOperationResponseUntilDone(
      absl::StatusOr<InMemoryHttpResponse> http_response);

  // Validates and extracts the base URI and headers to use for the subsequent
  // request(s). This should be called by the user of this class after every
  // successful `PerformProtocolRequest(...)` call.
  absl::Status ProcessForwardingInfo(
      const ::google::internal::federatedcompute::v1::ForwardingInfo&
          forwarding_info);

 private:
  absl::StatusOr<InMemoryHttpResponse> PerformProtocolRequest(
      absl::string_view uri_suffix, HttpRequest::Method method,
      std::string request_body, InterruptibleRunner& interruptible_runner);

  // Helper function for issuing a `GetOperationRequest` with which to poll the
  // given operation.
  absl::StatusOr<InMemoryHttpResponse> PerformGetOperationRequest(
      std::string operation_name);

  HttpClient& http_client_;
  InterruptibleRunner& interruptible_runner_;
  int64_t& bytes_downloaded_;
  int64_t& bytes_uploaded_;
  // The URI to use for the next protocol request. See `ForwardingInfo`.
  std::string next_request_base_uri_;
  // The set of headers to attach to the next protocol request. See
  // `ForwardingInfo`.
  HeaderList next_request_headers_;
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
  const Flags* const flags_;
  HttpClient* const http_client_;
  std::unique_ptr<InterruptibleRunner> interruptible_runner_;
  ProtocolRequestHelper protocol_request_helper_;
  const std::string api_key_;
  const std::string population_name_;
  const std::string retry_token_;
  const std::string client_version_;
  const std::string attestation_measurement_;
  std::function<absl::StatusOr<bool>()> should_abort_;
  absl::BitGen bit_gen_;
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
};

}  // namespace http
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_HTTP_HTTP_FEDERATED_PROTOCOL_H_
