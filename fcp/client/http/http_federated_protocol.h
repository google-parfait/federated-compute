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
#include "fcp/client/event_publisher.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/fl_runner.pb.h"
#include "fcp/client/flags.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/in_memory_request_response.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/selector_context.pb.h"
#include "fcp/client/task_environment.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/federatedcompute/common.pb.h"
#include "fcp/protos/federatedcompute/eligibility_eval_tasks.pb.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace http {

// Implements a single session of the HTTP-based Federated Compute protocol.
class HttpFederatedProtocol : public fcp::client::FederatedProtocol {
 public:
  HttpFederatedProtocol(
      EventPublisher* event_publisher, LogManager* log_manager,
      const Flags* flags, HttpClient* http_client,
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
      absl::string_view uri_suffix, std::string request_body);

  // Helper function to perform an eligibility eval task request and get its
  // response.
  absl::StatusOr<InMemoryHttpResponse> PerformEligibilityEvalTaskRequest();

  // Helper function for handling an eligibility eval task response (incl.
  // fetching any resources, if necessary).
  absl::StatusOr<fcp::client::FederatedProtocol::EligibilityEvalCheckinResult>
  HandleEligibilityEvalTaskResponse(
      absl::StatusOr<InMemoryHttpResponse> http_response);

  // Helper function for fetching the checkpoint/plan resources for an
  // eligibility eval task.
  absl::StatusOr<fcp::client::FederatedProtocol::CheckinResultPayload>
  FetchEligibilityEvalTaskResources(
      const ::google::internal::federatedcompute::v1::EligibilityEvalTask&
          task);

  // Helper that moves to the given object state if the given status represents
  // a permanent error.
  void UpdateObjectStateIfPermanentError(
      absl::Status status, ObjectState permanent_error_object_state);

  ObjectState object_state_;
  EventPublisher* const event_publisher_;
  LogManager* const log_manager_;
  const Flags* const flags_;
  HttpClient* const http_client_;
  std::unique_ptr<InterruptibleRunner> interruptible_runner_;
  // The URI to use for the next protocol request. See `ForwardingInfo`.
  std::string next_request_base_uri_;
  // The set of headers to attach to the next protocol request. See
  // `ForwardingInfo`.
  HeaderList next_request_headers_;
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
};

}  // namespace http
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_HTTP_HTTP_FEDERATED_PROTOCOL_H_
