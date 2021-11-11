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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/grpc_bidi_stream.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/selector_context.pb.h"
#include "fcp/client/task_environment.h"
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
      ::fcp::client::opstats::OpStatsLogger* opstats_logger, const Flags* flags,
      const std::string& federated_service_uri, const std::string& api_key,
      const std::string& test_cert_path, absl::string_view population_name,
      absl::string_view retry_token, absl::string_view client_version,
      absl::string_view attestation_measurement,
      std::function<bool()> should_abort,
      const InterruptibleRunner::TimingConfig& timing_config,
      const int64_t grpc_channel_deadline_seconds);

  // Test c'tor.
  GrpcFederatedProtocol(
      EventPublisher* event_publisher, LogManager* log_manager,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger, const Flags* flags,
      std::unique_ptr<GrpcBidiStreamInterface> grpc_bidi_stream,
      std::unique_ptr<secagg::SecAggClient> secagg_client,
      absl::string_view population_name, absl::string_view retry_token,
      absl::string_view client_version,
      absl::string_view attestation_measurement,
      std::function<bool()> should_abort, absl::BitGen bit_gen,
      const InterruptibleRunner::TimingConfig& timing_config);

  ~GrpcFederatedProtocol() override;

  absl::StatusOr<::fcp::client::FederatedProtocol::EligibilityEvalCheckinResult>
  EligibilityEvalCheckin() override;

  absl::StatusOr<::fcp::client::FederatedProtocol::CheckinResult> Checkin(
      const absl::optional<
          google::internal::federatedml::v2::TaskEligibilityInfo>&
          task_eligibility_info) override;

  absl::Status ReportCompleted(
      ComputationResults results,
      const std::vector<std::pair<std::string, double>>& stats,
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
  // Internal implementation of reporting for use by ReportCompleted() and
  // ReportNotCompleted().
  absl::Status Report(ComputationResults results,
                      engine::PhaseOutcome phase_outcome,
                      absl::Duration plan_duration,
                      const std::vector<std::pair<std::string, double>>& stats);
  absl::Status ReportInternal(
      std::string tf_checkpoint, engine::PhaseOutcome phase_outcome,
      absl::Duration plan_duration,
      const std::vector<std::pair<std::string, double>>& stats,
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
      const absl::optional<
          google::internal::federatedml::v2::TaskEligibilityInfo>&
          task_eligibility_info);

  // Helper to receive + process a CheckinRequestAck message.
  absl::Status ReceiveCheckinRequestAck();

  // Helper to receive + process an EligibilityEvalCheckinResponse message.
  absl::StatusOr<EligibilityEvalCheckinResult>
  ReceiveEligibilityEvalCheckinResponse(absl::Time start_time);

  // Helper to receive + process a CheckinResponse message.
  absl::StatusOr<CheckinResult> ReceiveCheckinResponse(absl::Time start_time);

  // Helper to update opstats network stats.
  void UpdateOpStatsNetworkStats();

  // Utility class for holding an absolute retry time and a corresponding retry
  // token.
  struct RetryTimeAndToken {
    absl::Time retry_time;
    std::string retry_token;
  };
  // Helper to generate a RetryWindow from a given time and token.
  google::internal::federatedml::v2::RetryWindow
  GenerateRetryWindowFromRetryTimeAndToken(const RetryTimeAndToken& retry_info);

  enum class ObjectState {
    // The initial object state.
    kInitialized,
    // EligibilityEvalCheckin() was called but it failed with a 'transient'
    // error (e.g. an UNAVAILABLE network error, although the set of transient
    // errors is flag-defined).
    kEligibilityEvalCheckinFailed,
    // EligibilityEvalCheckin() was called but it failed with a 'permanent'
    // error (e.g. a NOT_FOUND network error, although the set of permanent
    // errors is flag-defined).
    kEligibilityEvalCheckinFailedPermanentError,
    // EligibilityEvalCheckin() was called, and the server rejected the client.
    kEligibilityEvalCheckinRejected,
    // EligibilityEvalCheckin() was called, and the server did not return an
    // eligibility eval payload.
    kEligibilityEvalDisabled,
    // EligibilityEvalCheckin() was called, and the server did return an
    // eligibility eval payload, which must then be run to produce a
    // TaskEligibilityInfo value.
    kEligibilityEvalEnabled,
    // Checkin(...) was called but it failed with a 'transient' error.
    kCheckinFailed,
    // Checkin(...) was called but it failed with a 'permanent' error.
    kCheckinFailedPermanentError,
    // Checkin(...) was called, and the server rejected the client.
    kCheckinRejected,
    // Checkin(...) was called, and the server accepted the client and returned
    // a payload, which must then be run to produce a report.
    kCheckinAccepted,
    // Report(...) was called.
    kReportCalled,
    // Report(...) was called and it resulted in a 'permanent' error.
    //
    // Note: there is no kReportFailed (corresponding to 'transient' errors,
    // like the other phases have), because by the time the report phase is
    // reached, a set of RetryWindows is guaranteed to have been received from
    // the server.
    kReportFailedPermanentError
  };

  // Helper that moves to the given object state if the given status represents
  // a permanent error.
  void UpdateObjectStateForPermanentError(
      absl::Status status, ObjectState permanent_error_object_state);

  ObjectState object_state_;
  EventPublisher* const event_publisher_;
  LogManager* const log_manager_;
  ::fcp::client::opstats::OpStatsLogger* const opstats_logger_;
  const Flags* const flags_;
  std::unique_ptr<GrpcBidiStreamInterface> grpc_bidi_stream_;
  std::unique_ptr<secagg::SecAggClient> secagg_client_;
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
  int64_t bytes_downloaded_ = 0;
  int64_t bytes_uploaded_ = 0;
  int64_t report_request_size_bytes_ = 0;
  // TODO(team): Delete these fields after rollout is complete.
  google::internal::federatedml::v2::RetryWindow retry_window_if_accepted_;
  google::internal::federatedml::v2::RetryWindow retry_window_if_rejected_;
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
  absl::optional<CheckinRequestAckInfo> checkin_request_ack_info_;
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
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_GRPC_FEDERATED_PROTOCOL_H_
