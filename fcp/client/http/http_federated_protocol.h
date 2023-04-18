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
#include "fcp/base/clock.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/wall_clock_stopwatch.h"
#include "fcp/client/cache/resource_cache.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/in_memory_request_response.h"
#include "fcp/client/http/protocol_request_helper.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/secagg_runner.h"
#include "fcp/client/selector_context.pb.h"
#include "fcp/client/stats.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/federatedcompute/common.pb.h"
#include "fcp/protos/federatedcompute/eligibility_eval_tasks.pb.h"
#include "fcp/protos/federatedcompute/secure_aggregations.pb.h"
#include "fcp/protos/federatedcompute/task_assignments.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/secagg/client/secagg_client.h"

namespace fcp {
namespace client {
namespace http {

// Implements a single session of the HTTP-based Federated Compute protocol.
class HttpFederatedProtocol : public fcp::client::FederatedProtocol {
 public:
  HttpFederatedProtocol(
      Clock* clock, LogManager* log_manager, const Flags* flags,
      HttpClient* http_client,
      std::unique_ptr<SecAggRunnerFactory> secagg_runner_factory,
      SecAggEventPublisher* secagg_event_publisher,
      absl::string_view entry_point_uri, absl::string_view api_key,
      absl::string_view population_name, absl::string_view retry_token,
      absl::string_view client_version,
      absl::string_view attestation_measurement,
      std::function<bool()> should_abort, absl::BitGen bit_gen,
      const InterruptibleRunner::TimingConfig& timing_config,
      cache::ResourceCache* resource_cache);

  ~HttpFederatedProtocol() override = default;

  absl::StatusOr<fcp::client::FederatedProtocol::EligibilityEvalCheckinResult>
  EligibilityEvalCheckin(std::function<void(const EligibilityEvalTask&)>
                             payload_uris_received_callback) override;

  void ReportEligibilityEvalError(absl::Status error_status) override;

  absl::StatusOr<fcp::client::FederatedProtocol::CheckinResult> Checkin(
      const std::optional<
          google::internal::federatedml::v2::TaskEligibilityInfo>&
          task_eligibility_info,
      std::function<void(const TaskAssignment&)> payload_uris_received_callback)
      override;

  absl::Status ReportCompleted(
      ComputationResults results,
      absl::Duration plan_duration) override;

  absl::Status ReportNotCompleted(engine::PhaseOutcome phase_outcome,
                                  absl::Duration plan_duration) override;

  google::internal::federatedml::v2::RetryWindow GetLatestRetryWindow()
      override;

  NetworkStats GetNetworkStats() override;

 private:
  // Helper function to perform an eligibility eval task request and get its
  // response.
  absl::StatusOr<InMemoryHttpResponse> PerformEligibilityEvalTaskRequest();

  // Helper function for handling an eligibility eval task response (incl.
  // fetching any resources, if necessary).
  absl::StatusOr<fcp::client::FederatedProtocol::EligibilityEvalCheckinResult>
  HandleEligibilityEvalTaskResponse(
      absl::StatusOr<InMemoryHttpResponse> http_response,
      std::function<void(const EligibilityEvalTask&)>
          payload_uris_received_callback);

  absl::StatusOr<std::unique_ptr<HttpRequest>>
  CreateReportEligibilityEvalTaskResultRequest(absl::Status status);

  // Helper function to perform an ReportEligibilityEvalResult request.
  absl::Status ReportEligibilityEvalErrorInternal(absl::Status error_status);

  // Helper function to perform a task assignment request and get its response.
  absl::StatusOr<InMemoryHttpResponse>
  PerformTaskAssignmentAndReportEligibilityEvalResultRequests(
      const std::optional<
          ::google::internal::federatedml::v2::TaskEligibilityInfo>&
          task_eligibility_info);

  // Helper function for handling the 'outer' task assignment response, which
  // consists of an `Operation` which may or may not need to be polled before a
  // final 'inner' response is available.
  absl::StatusOr<::fcp::client::FederatedProtocol::CheckinResult>
  HandleTaskAssignmentOperationResponse(
      absl::StatusOr<InMemoryHttpResponse> http_response,
      std::function<void(const TaskAssignment&)>
          payload_uris_received_callback);

  // Helper function for handling an 'inner' task assignment response (i.e.
  // after the outer `Operation` has concluded). This includes fetching any
  // resources, if necessary.
  absl::StatusOr<::fcp::client::FederatedProtocol::CheckinResult>
  HandleTaskAssignmentInnerResponse(
      const ::google::protobuf::Any& operation_response,
      std::function<void(const TaskAssignment&)>
          payload_uris_received_callback);

  // Helper function for reporting result via simple aggregation.
  absl::Status ReportViaSimpleAggregation(ComputationResults results,
                                          absl::Duration plan_duration);
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

  // Helper function for reporting via secure aggregation.
  absl::Status ReportViaSecureAggregation(ComputationResults results,
                                          absl::Duration plan_duration);

  // Helper function to perform a StartSecureAggregationRequest and a
  // ReportTaskResultRequest.
  absl::StatusOr<
      google::internal::federatedcompute::v1::StartSecureAggregationResponse>
  StartSecureAggregationAndReportTaskResult(absl::Duration plan_duration);

  struct TaskResources {
    const ::google::internal::federatedcompute::v1::Resource& plan;
    const ::google::internal::federatedcompute::v1::Resource& checkpoint;
  };

  // Helper function for fetching the checkpoint/plan resources for an
  // eligibility eval task or regular task.
  absl::StatusOr<PlanAndCheckpointPayloads> FetchTaskResources(
      TaskResources task_resources);

  // Helper function for fetching the PopulationEligibilitySpec.
  absl::StatusOr<
      google::internal::federatedcompute::v1::PopulationEligibilitySpec>
  FetchPopulationEligibilitySpec(
      const ::google::internal::federatedcompute::v1::Resource&
          population_eligibility_spec_resource);

  // Helper that moves to the given object state if the given status represents
  // a permanent error.
  void UpdateObjectStateIfPermanentError(
      absl::Status status, ObjectState permanent_error_object_state);

  ObjectState object_state_;
  Clock& clock_;
  LogManager* log_manager_;
  const Flags* const flags_;
  HttpClient* const http_client_;
  std::unique_ptr<SecAggRunnerFactory> secagg_runner_factory_;
  SecAggEventPublisher* secagg_event_publisher_;
  std::unique_ptr<InterruptibleRunner> interruptible_runner_;
  std::unique_ptr<ProtocolRequestCreator> eligibility_eval_request_creator_;
  std::unique_ptr<ProtocolRequestCreator> task_assignment_request_creator_;
  std::unique_ptr<ProtocolRequestCreator> aggregation_request_creator_;
  std::unique_ptr<ProtocolRequestCreator> data_upload_request_creator_;
  std::unique_ptr<WallClockStopwatch> network_stopwatch_ =
      WallClockStopwatch::Create();
  ProtocolRequestHelper protocol_request_helper_;
  const std::string api_key_;
  const std::string population_name_;
  const std::string retry_token_;
  const std::string client_version_;
  const std::string attestation_measurement_;
  std::function<bool()> should_abort_;
  absl::BitGen bit_gen_;
  const InterruptibleRunner::TimingConfig timing_config_;
  // The graceful waiting period for cancellation requests before checking
  // whether the client should be interrupted.
  const absl::Duration waiting_period_for_cancellation_;
  // The set of canonical error codes that should be treated as 'permanent'
  // errors.
  absl::flat_hash_set<int32_t> federated_training_permanent_error_codes_;
  int64_t bytes_downloaded_ = 0;
  int64_t bytes_uploaded_ = 0;
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
  // The token authorizing the client to participate in an aggregation session.
  std::string aggregation_authorization_token_;
  // The name identifying the task that was assigned.
  std::string task_name_;
  // Unique identifier for the client's participation in an aggregation session.
  std::string aggregation_client_token_;
  // Resource name for the checkpoint in simple aggregation.
  std::string aggregation_resource_name_;
  // Set this field to true if an eligibility eval task was received from the
  // server in the EligibilityEvalTaskResponse.
  bool eligibility_eval_enabled_ = false;
  // `nullptr` if the feature is disabled.
  cache::ResourceCache* resource_cache_;
};

}  // namespace http
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_HTTP_HTTP_FEDERATED_PROTOCOL_H_
