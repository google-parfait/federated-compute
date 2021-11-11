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

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/plan.pb.h"

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

// An interface that represents a single Federated Compute protocol session.
//
// An instance of this class represents a single session of client-server
// interaction. Instances are generally stateful, and therefore cannot be
// reused (each session should use a dedicated instance).
//
// The protocol consists of 3 phases, which must occur in the following order:
// 1. A call to `EligibilityEvalCheckin()`.
// 2. A call to `Checkin(...)`, only if the client wasn't rejected by the server
//    in the previous phase.
// 3. A call to `ReportCompleted(...)` or `ReportNotCompleted(...)`, only if the
//    client wasn't rejected in the previous phase.
class FederatedProtocol {
 public:
  virtual ~FederatedProtocol() = default;

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
  //   - UNIMPLEMENTED if an unexpected server response is received.
  //   - INTERNAL if the server-provided ClientOnlyPlan cannot be parsed. (See
  //     note in federated_protocol.cc for the reasoning for this.)
  //   - INTERNAL for other unexpected client-side errors.
  //   - any server-provided error code.
  virtual absl::StatusOr<EligibilityEvalCheckinResult>
  EligibilityEvalCheckin() = 0;

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
  //   - UNIMPLEMENTED if an unexpected server response is received.
  //   - INTERNAL if the server-provided ClientOnlyPlan cannot be parsed. (See
  //     note in federated_protocol.cc for the reasoning for this.)
  //   - INTERNAL for other unexpected client-side errors.
  //   - any server-provided error code.
  // TODO(team): Replace this reference to protocol-specific
  // TaskEligibilityInfo proto with a protocol-agnostic struct.
  virtual absl::StatusOr<CheckinResult> Checkin(
      const absl::optional<
          google::internal::federatedml::v2::TaskEligibilityInfo>&
          task_eligibility_info) = 0;

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
      absl::Duration plan_duration) = 0;

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
                                          absl::Duration plan_duration) = 0;

  // Returns the RetryWindow the caller should use when rescheduling, based on
  // the current protocol phase. The value returned by this method may change
  // after every interaction with the protocol, so callers should call this
  // right before ending their interactions with the FederatedProtocol object to
  // ensure they use the most recent value.
  // TODO(team): Replace this reference to protocol-specific
  // RetryWindow proto with a protocol-agnostic struct (or just a single
  // absl::Duration).
  virtual google::internal::federatedml::v2::RetryWindow
  GetLatestRetryWindow() = 0;

  // Returns the number of bytes downloaded.
  virtual int64_t bytes_downloaded() = 0;

  // Returns the number of bytes uploaded.
  virtual int64_t bytes_uploaded() = 0;

  // Returns the number of bytes received in the chunking layer.
  virtual int64_t chunking_layer_bytes_received() = 0;

  // Returns the number of bytes sent in the chunking layer.
  virtual int64_t chunking_layer_bytes_sent() = 0;

  // Returns the size of the report request in bytes.
  virtual int64_t report_request_size_bytes() = 0;
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_FEDERATED_PROTOCOL_H_
