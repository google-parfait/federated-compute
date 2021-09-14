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
#ifndef FCP_CLIENT_FEDERATED_PROTOCOL_H_
#define FCP_CLIENT_FEDERATED_PROTOCOL_H_

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

// Data type used to encode results of a computation - a TensorFlow
// checkpoint, or SecAgg quantized tensors.
// For non-SecAgg use (simple federated aggregation, or local computation),
// this map should only contain one entry - a TFCheckpoint - and the std::string
// should be ignored by downstream code.
// For SecAgg use, there should be
// * at most one TFCheckpoint - again, the key should be ignored - and
// * N QuantizedTensors, whose std::string keys must map to the tensor names
//   provided in the server's CheckinResponse's SideChannelExecutionInfo.
using TFCheckpoint = std::string;
struct QuantizedTensor {
  std::vector<uint64_t> values;
  int32_t bitwidth;
  std::vector<int64_t> dimensions;

  QuantizedTensor() = default;
  // Disallow copy and assign.
  QuantizedTensor(const QuantizedTensor&) = delete;
  QuantizedTensor& operator=(const QuantizedTensor&) = delete;
  // Enable move semantics.
  QuantizedTensor(QuantizedTensor&&) = default;
  QuantizedTensor& operator=(QuantizedTensor&&) = default;
};
// This is equivalent to using ComputationResults =
//    std::map<std::string, absl::variant<TFCheckpoint, QuantizedTensor>>;
// except copy construction and assignment are explicitly prohibited and move
// semantics is enforced.
class ComputationResults
    : public absl::node_hash_map<std::string,
                                 absl::variant<TFCheckpoint, QuantizedTensor>> {
 public:
  using Base =
      absl::node_hash_map<std::string,
                          absl::variant<TFCheckpoint, QuantizedTensor>>;
  using Base::Base;
  using Base::operator=;
  ComputationResults(const ComputationResults&) = delete;
  ComputationResults& operator=(const ComputationResults&) = delete;
  ComputationResults(ComputationResults&&) = default;
  ComputationResults& operator=(ComputationResults&&) = default;
};

// Implements a single session of the Federated Learning protocol.
//
// An instance of this class represents a single protocol session, backed by a
// single connection to the server. The instance is stateful, and instances
// therefore cannot be reused.
//
// The protocol consists of either 3 phases (one of which is optional), which
// must occur in the following order:
// 1. A call to `EligibilityEvalCheckin()`, which is optional.
// 2. A call to `Checkin(...)`, only if the client wasn't rejected by the server
//    in the previous phase.
// 3. A call to `ReportCompleted(...)` or `ReportNotCompleted(...)`, only if the
//    client wasn't rejected in the previous phase.
class FederatedProtocol {
 public:
  FederatedProtocol(
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
  FederatedProtocol(EventPublisher* event_publisher, LogManager* log_manager,
                    ::fcp::client::opstats::OpStatsLogger* opstats_logger,
                    const Flags* flags,
                    std::unique_ptr<GrpcBidiStreamInterface> grpc_bidi_stream,
                    std::unique_ptr<secagg::SecAggClient> secagg_client,
                    absl::string_view population_name,
                    absl::string_view retry_token,
                    absl::string_view client_version,
                    absl::string_view attestation_measurement,
                    std::function<bool()> should_abort, absl::BitGen bit_gen,
                    const InterruptibleRunner::TimingConfig& timing_config);

  virtual ~FederatedProtocol();

  // A computation payload consisting of a plan, checkpoint, and other relevant
  // metadata.
  struct CheckinResultPayload {
    google::internal::federated::plan::ClientOnlyPlan client_only_plan;
    std::string checkpoint;
    std::string task_name;
    // Secagg metadata, see SecureAggregationProtocolExecutionInfo in
    // federated_api.proto.
    int32_t minimum_clients_in_server_visible_aggregate;
  };
  // A rejection of a client by the server.
  struct Rejection {};
  // Indicates that the server does not have an eligibility eval task configured
  // for the population.
  struct EligibilityEvalDisabled {};
  // EligibilityEvalCheckin() returns either
  // 1. a CheckinResultPayload struct representing an eligibility eval task, if
  // the
  //    population is configured with such a task. In this case the caller
  //    should execute the task and pass the resulting `TaskEligibilityInfo`
  //    value to the `Checkin(...)` method.
  // 2. an EligibilityEvalDisabled struct if the population doesn't have an
  //    eligibility eval task configured. In this case the caller should
  //    continue the protocol by calling the `Checkin(...)` method without
  //    providing a `TaskEligibilityInfo` value.
  // 3. a Rejection if the server rejected this device. In this case the caller
  //    should end its protocol interaction.
  using EligibilityEvalCheckinResult =
      absl::variant<CheckinResultPayload, EligibilityEvalDisabled, Rejection>;

  // Checks in with a federated server to receive the population's eligibility
  // eval task. This method is optional and may be called 0 or 1 times. If it is
  // called, then it must be called before any call to `Checkin(...)`.
  //
  // Returns:
  // - On success, an EligibilityEvalCheckinResult.
  // - On error:
  //   - ABORTED when one of the I/O operations got aborted by the server.
  //   - CANCELLED when one of the I/O operations was interrupted by the client
  //     (possibly due to a positive result from the should_abort callback).
  //   - UNAVAILABLE when server cannot be reached or URI is invalid.
  //   - NOT_FOUND if the server responds with NOT_FOUND, e.g. because the
  //     specified population name is incorrect.
  //   - UNIMPLEMENTED if an unexpected ServerStreamMessage is received.
  //   - INTERNAL if the server-provided ClientOnlyPlan cannot be parsed. (See
  //     note in federated_protocol.cc for the reasoning for this.)
  //   - INTERNAL for other unexpected client-side errors.
  //   - any server-provided error code.
  virtual absl::StatusOr<EligibilityEvalCheckinResult> EligibilityEvalCheckin();

  // Checkin() returns either
  // 1. a CheckinResultPayload struct, or
  // 2. a Rejection struct if the server rejected this device.
  using CheckinResult = absl::variant<CheckinResultPayload, Rejection>;

  // Checks in with a federated server. Must only be called once. If the
  // `EligibilityEvalCheckin()` method was previously called, then this method
  // must only be called if the result of that call was not a `Rejection`.
  //
  // If the caller previously called `EligibilityEvalCheckin()` and:
  // - received a payload, then the `TaskEligibilityInfo` value computed by that
  //   payload must be provided via the `task_eligibility_info` parameter.
  // - received an `EligibilityEvalDisabled` result, then the
  //   `task_eligibility_info` parameter should be left empty.
  //
  // If the caller did not previously call `EligibilityEvalCheckin()`, then the
  // `task_eligibility_info` parameter should be left empty.
  //
  // Returns:
  // - On success, a CheckinResult that either contains
  //   - a ClientOnlyPlan & initial checkpoint for a federated computation,
  //     if the device is accepted for participation.
  //   - a Rejection, if the device is rejected.
  // - On error:
  //   - ABORTED when one of the I/O operations got aborted by the server.
  //   - CANCELLED when one of the I/O operations was interrupted by the client
  //     (possibly due to a positive result from the should_abort callback).
  //   - UNAVAILABLE when server cannot be reached or URI is invalid.
  //   - NOT_FOUND if the server responds with NOT_FOUND, e.g. because the
  //     specified population name is incorrect.
  //   - UNIMPLEMENTED if an unexpected ServerStreamMessage is received.
  //   - INTERNAL if the server-provided ClientOnlyPlan cannot be parsed. (See
  //     note in federated_protocol.cc for the reasoning for this.)
  //   - INTERNAL for other unexpected client-side errors.
  //   - any server-provided error code.
  virtual absl::StatusOr<CheckinResult> Checkin(
      const absl::optional<
          google::internal::federatedml::v2::TaskEligibilityInfo>&
          task_eligibility_info);

  // Reports the result of a federated computation to the server. Must only be
  // called once and after a successful call to Checkin().
  // @param checkpoint A checkpoint proto.
  // @param stats all stats reported during the computation.
  // @param plan_duration the duration for executing the plan in the plan
  //        engine. Does not include time spent on downloading the plan.
  // Returns:
  // - On success, OK.
  // - On error (e.g. an interruption, network error, or other unexpected
  //   error):
  //   - ABORTED when one of the I/O operations got aborted by the server.
  //   - CANCELLED when one of the I/O operations was interrupted by the client
  //     (possibly due to a positive result from the should_abort callback).
  //   - UNIMPLEMENTED if the server responded with an unexpected response
  //     message.
  //   - INTERNAL for other unexpected client-side errors.
  //   - any server-provided error code.
  virtual absl::Status ReportCompleted(
      ComputationResults results,
      const std::vector<std::pair<std::string, double>>& stats,
      absl::Duration plan_duration);

  // Reports the unsuccessful result of a federated computation to the server.
  // Must only be called once and after a successful call to Checkin().
  // @param phase_outcome the outcome of the federated computation.
  // @param plan_duration the duration for executing the plan in the plan
  //        engine. Does not include time spent on downloading the plan.
  // Returns:
  // - On success, OK.
  // - On error:
  //   - ABORTED when one of the I/O operations got aborted by the server.
  //   - CANCELLED when one of the I/O operations was interrupted by the client
  //     (possibly due to a positive result from the should_abort callback).
  //   - UNIMPLEMENTED if the server responded with an unexpected response
  //     message, or if the results to report require SecAgg support.
  //   - INTERNAL for other unexpected client-side errors.
  //   - any server-provided error code.
  virtual absl::Status ReportNotCompleted(engine::PhaseOutcome phase_outcome,
                                          absl::Duration plan_duration);

  // Returns the RetryWindow the caller should use when rescheduling, based on
  // the current protocol phase. The value returned by this method may change
  // after every interaction with the protocol, so callers should call this
  // right before ending their interactions with the FederatedProtocol object to
  // ensure they use the most recent value.
  virtual google::internal::federatedml::v2::RetryWindow GetLatestRetryWindow();

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
      int64_t* report_request_size,
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
          task_eligibility_info,
      bool should_ack_checkin);

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
  // TODO(team): Delete these fields after rollout is complete.
  google::internal::federatedml::v2::RetryWindow retry_window_if_accepted_;
  google::internal::federatedml::v2::RetryWindow retry_window_if_rejected_;
  // We store this flag value as a class member field, to ensure its value
  // cannot change across the life of this instance.
  const bool federated_training_use_new_retry_delay_behavior_;
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
  // This field will have an absent value until that message has been received,
  // or when the federated_training_use_new_retry_delay_behavior_ flag is false.
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

#endif  // FCP_CLIENT_FEDERATED_PROTOCOL_H_
