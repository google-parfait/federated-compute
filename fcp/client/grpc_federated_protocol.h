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
#ifndef FCP_CLIENT_GRPC_FEDERATED_PROTOCOL_H_
#define FCP_CLIENT_GRPC_FEDERATED_PROTOCOL_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/wall_clock_stopwatch.h"
#include "fcp/client/cache/resource_cache.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/grpc_bidi_stream.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/in_memory_request_response.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/secagg_runner.h"
#include "fcp/client/selector_context.pb.h"
#include "fcp/client/stats.h"
#include "fcp/protocol/grpc_chunked_bidi_stream.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/secagg/client/secagg_client.h"

namespace fcp {
namespace client {

// Implements a single session of the gRPC-based Federated Learning protocol.
class GrpcFederatedProtocol : public ::fcp::client::FederatedProtocol {
 public:
  GrpcFederatedProtocol(
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
      cache::ResourceCache* resource_cache);

  // Test c'tor.
  GrpcFederatedProtocol(
      EventPublisher* event_publisher, LogManager* log_manager,
      std::unique_ptr<SecAggRunnerFactory> secagg_runner_factory,
      const Flags* flags, ::fcp::client::http::HttpClient* http_client,
      std::unique_ptr<GrpcBidiStreamInterface> grpc_bidi_stream,
      absl::string_view population_name, absl::string_view retry_token,
      absl::string_view client_version,
      absl::string_view attestation_measurement,
      std::function<bool()> should_abort, absl::BitGen bit_gen,
      const InterruptibleRunner::TimingConfig& timing_config,
      cache::ResourceCache* resource_cache);

  ~GrpcFederatedProtocol() override;

  absl::StatusOr<::fcp::client::FederatedProtocol::EligibilityEvalCheckinResult>
  EligibilityEvalCheckin(std::function<void(const EligibilityEvalTask&)>
                             payload_uris_received_callback) override;

  void ReportEligibilityEvalError(absl::Status error_status) override;

  absl::StatusOr<::fcp::client::FederatedProtocol::CheckinResult> Checkin(
      const std::optional<
          google::internal::federatedml::v2::TaskEligibilityInfo>&
          task_eligibility_info,
      std::function<void(const TaskAssignment&)> payload_uris_received_callback)
      override;

  absl::StatusOr<::fcp::client::FederatedProtocol::MultipleTaskAssignments>
  PerformMultipleTaskAssignments(
      const std::vector<std::string>& task_names) override;

  absl::Status ReportCompleted(
      ComputationResults results, absl::Duration plan_duration,
      std::optional<std::string> aggregation_session_id) override;

  absl::Status ReportNotCompleted(
      engine::PhaseOutcome phase_outcome, absl::Duration plan_duration,
      std::optional<std::string> aggregation_session_id) override;

  google::internal::federatedml::v2::RetryWindow GetLatestRetryWindow()
      override;

  NetworkStats GetNetworkStats() override;

 private:
  // Internal implementation of reporting for use by ReportCompleted() and
  // ReportNotCompleted().
  absl::Status Report(ComputationResults results,
                      engine::PhaseOutcome phase_outcome,
                      absl::Duration plan_duration);
  absl::Status ReportInternal(
      std::string tf_checkpoint, engine::PhaseOutcome phase_outcome,
      absl::Duration plan_duration,
      fcp::secagg::ClientToServerWrapperMessage* secagg_commit_message);

  // Helper function to send a ClientStreamMessage. If sending did not succeed,
  // closes the underlying grpc stream. If sending does succeed then it updates
  // `bytes_uploaded_`.
  absl::Status Send(google::internal::federatedml::v2::ClientStreamMessage*
                        client_stream_message);

  // Helper function to receive a ServerStreamMessage. If receiving did not
  // succeed, closes the underlying grpc stream. If receiving does succeed then
  // it updates `bytes_downloaded_`.
  absl::Status Receive(google::internal::federatedml::v2::ServerStreamMessage*
                           server_stream_message);

  // Helper function to compose a ProtocolOptionsRequest for eligibility eval or
  // regular checkin requests.
  google::internal::federatedml::v2::ProtocolOptionsRequest
  CreateProtocolOptionsRequest(bool should_ack_checkin) const;

  // Helper function to compose and send an EligibilityEvalCheckinRequest to the
  // server.
  absl::Status SendEligibilityEvalCheckinRequest();

  // Helper function to compose and send a CheckinRequest to the server.
  absl::Status SendCheckinRequest(
      const std::optional<
          google::internal::federatedml::v2::TaskEligibilityInfo>&
          task_eligibility_info);

  // Helper to receive + process a CheckinRequestAck message.
  absl::Status ReceiveCheckinRequestAck();

  // Helper to receive + process an EligibilityEvalCheckinResponse message.
  absl::StatusOr<EligibilityEvalCheckinResult>
  ReceiveEligibilityEvalCheckinResponse(
      absl::Time start_time, std::function<void(const EligibilityEvalTask&)>
                                 payload_uris_received_callback);

  // Helper to receive + process a CheckinResponse message.
  absl::StatusOr<CheckinResult> ReceiveCheckinResponse(
      absl::Time start_time, std::function<void(const TaskAssignment&)>
                                 payload_uris_received_callback);

  // Utility class for holding an absolute retry time and a corresponding retry
  // token.
  struct RetryTimeAndToken {
    absl::Time retry_time;
    std::string retry_token;
  };
  // Helper to generate a RetryWindow from a given time and token.
  google::internal::federatedml::v2::RetryWindow
  GenerateRetryWindowFromRetryTimeAndToken(const RetryTimeAndToken& retry_info);

  // Helper that moves to the given object state if the given status represents
  // a permanent error.
  void UpdateObjectStateIfPermanentError(
      absl::Status status, ObjectState permanent_error_object_state);

  // Utility struct to represent resource data coming from the gRPC protocol.
  // A resource is either represented by a URI from which the data should be
  // fetched (in which case `has_uri` is true and `uri` should not be empty), or
  // is available as inline data (in which case `has_uri` is false and `data`
  // may or may not be empty).
  struct TaskResource {
    bool has_uri;
    const std::string& uri;
    const std::string& data;
    // The following fields will be set if the client should attempt to cache
    // the resource.
    const std::string& client_cache_id;
    const absl::Duration max_age;
  };
  // Represents the common set of resources a task may have.
  struct TaskResources {
    TaskResource plan;
    TaskResource checkpoint;
  };

  // Helper function for fetching the checkpoint/plan resources for an
  // eligibility eval task or regular task. This function will return an error
  // if either `TaskResource` represents an invalid state (e.g. if `has_uri &&
  // uri.empty()`).
  absl::StatusOr<PlanAndCheckpointPayloads> FetchTaskResources(
      TaskResources task_resources);
  // Validates the given `TaskResource` and converts it to a `UriOrInlineData`
  // object for use with the `FetchResourcesInMemory` utility method.
  absl::StatusOr<::fcp::client::http::UriOrInlineData>
  ConvertResourceToUriOrInlineData(const TaskResource& resource);

  ObjectState object_state_;
  EventPublisher* const event_publisher_;
  LogManager* const log_manager_;
  std::unique_ptr<SecAggRunnerFactory> secagg_runner_factory_;
  const Flags* const flags_;
  ::fcp::client::http::HttpClient* const http_client_;
  std::unique_ptr<GrpcBidiStreamInterface> grpc_bidi_stream_;
  std::unique_ptr<InterruptibleRunner> interruptible_runner_;
  const std::string population_name_;
  const std::string retry_token_;
  const std::string client_version_;
  const std::string attestation_measurement_;
  std::function<absl::StatusOr<bool>()> should_abort_;
  absl::BitGen bit_gen_;
  // The set of canonical error codes that should be treated as 'permanent'
  // errors.
  absl::flat_hash_set<int32_t> federated_training_permanent_error_codes_;
  int64_t http_bytes_downloaded_ = 0;
  int64_t http_bytes_uploaded_ = 0;
  std::unique_ptr<WallClockStopwatch> network_stopwatch_ =
      WallClockStopwatch::Create();
  // Represents 2 absolute retry timestamps and their corresponding retry
  // tokens, to use when the device is rejected or accepted. The retry
  // timestamps will have been generated based on the retry windows specified in
  // the server's CheckinRequestAck message and the time at which that message
  // was received.
  struct CheckinRequestAckInfo {
    RetryTimeAndToken retry_info_if_rejected;
    RetryTimeAndToken retry_info_if_accepted;
  };
  // Represents the information received via the CheckinRequestAck message.
  // This field will have an absent value until that message has been received.
  std::optional<CheckinRequestAckInfo> checkin_request_ack_info_;
  // The identifier of the task that was received in a CheckinResponse. Note
  // that this does not refer to the identifier of the eligbility eval task that
  // may have been received in an EligibilityEvalCheckinResponse.
  std::string execution_phase_id_;
  absl::flat_hash_map<
      std::string, google::internal::federatedml::v2::SideChannelExecutionInfo>
      side_channels_;
  google::internal::federatedml::v2::SideChannelProtocolExecutionInfo
      side_channel_protocol_execution_info_;
  google::internal::federatedml::v2::SideChannelProtocolOptionsResponse
      side_channel_protocol_options_response_;
  // `nullptr` if the feature is disabled.
  cache::ResourceCache* resource_cache_;
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_GRPC_FEDERATED_PROTOCOL_H_
