/*
 * Copyright 2026 Google LLC
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

#include "fcp/client/http/http_federated_protocol_test_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/longrunning/operations.pb.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/duration.pb.h"
#include "google/rpc/code.pb.h"
#include "google/type/datetime.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/base/time_util.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/event_time_range.pb.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/http_federated_protocol.h"
#include "fcp/client/http/testing/test_helpers.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/stats.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/payload_metadata.pb.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/federatedcompute/aggregations.pb.h"
#include "fcp/protos/federatedcompute/common.pb.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"
#include "fcp/protos/federatedcompute/eligibility_eval_tasks.pb.h"
#include "fcp/protos/federatedcompute/secure_aggregations.pb.h"
#include "fcp/protos/federatedcompute/task_assignments.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/protos/population_eligibility_spec.pb.h"
#include "fcp/secagg/shared/secagg_messages.pb.h"
#include "proto/attestation/endorsement.pb.h"

namespace fcp::client::http::internal {

void ExpectTransientErrorRetryWindow(
    const ::google::internal::federatedml::v2::RetryWindow& retry_window) {
  // The calculated retry delay must lie within the expected transient errors
  // retry delay range.
  EXPECT_THAT(retry_window.delay_min().seconds() +
                  retry_window.delay_min().nanos() / 1000000000,
              AllOf(Ge(kExpectedTransientErrorsRetryPeriodSecsMin),
                    Lt(kExpectedTransientErrorsRetryPeriodSecsMax)));
  EXPECT_THAT(retry_window.delay_max(), EqualsProto(retry_window.delay_min()));
}

void ExpectPermanentErrorRetryWindow(
    const ::google::internal::federatedml::v2::RetryWindow& retry_window) {
  // The calculated retry delay must lie within the expected permanent errors
  // retry delay range.
  EXPECT_THAT(retry_window.delay_min().seconds() +
                  retry_window.delay_min().nanos() / 1000000000,
              AllOf(Ge(kExpectedPermanentErrorsRetryPeriodSecsMin),
                    Lt(kExpectedPermanentErrorsRetryPeriodSecsMax)));
  EXPECT_THAT(retry_window.delay_max(), EqualsProto(retry_window.delay_min()));
}

RetryWindow GetAcceptedRetryWindow() {
  // Must not overlap with kTransientErrorsRetryPeriodSecs or
  // kPermanentErrorsRetryPeriodSecs.
  RetryWindow retry_window;
  retry_window.mutable_delay_min()->set_seconds(200L);
  retry_window.mutable_delay_max()->set_seconds(299L);
  return retry_window;
}

void ExpectAcceptedRetryWindow(
    const ::google::internal::federatedml::v2::RetryWindow& retry_window) {
  // The calculated retry delay must lie within the expected 'rejected' retry
  // delay range.
  EXPECT_THAT(retry_window.delay_min().seconds() +
                  retry_window.delay_min().nanos() / 1000000000,
              AllOf(Ge(200L), Lt(299L)));
  EXPECT_THAT(retry_window.delay_max(), EqualsProto(retry_window.delay_min()));
}

RetryWindow GetRejectedRetryWindow() {
  // Must not overlap with kTransientErrorsRetryPeriodSecs or
  // kPermanentErrorsRetryPeriodSecs.
  RetryWindow retry_window;
  retry_window.mutable_delay_min()->set_seconds(300L);
  retry_window.mutable_delay_max()->set_seconds(399L);
  return retry_window;
}

void ExpectRejectedRetryWindow(
    const ::google::internal::federatedml::v2::RetryWindow& retry_window) {
  // The calculated retry delay must lie within the expected 'rejected' retry
  // delay range.
  EXPECT_THAT(retry_window.delay_min().seconds() +
                  retry_window.delay_min().nanos() / 1000000000,
              AllOf(Ge(300L), Lt(399L)));
  EXPECT_THAT(retry_window.delay_max(), EqualsProto(retry_window.delay_min()));
}

EligibilityEvalTaskRequest GetExpectedEligibilityEvalTaskRequest(
    bool enable_confidential_aggregation,
    bool enable_willow_secure_aggregation) {
  EligibilityEvalTaskRequest request;
  // Note: we don't expect population_name to be set, since it should be set in
  // the URI instead.
  request.mutable_client_version()->set_version_code(kClientVersion);
  request.mutable_attestation_measurement()->set_value(kAttestationMeasurement);
  request.mutable_resource_capabilities()
      ->mutable_supported_compression_formats()
      ->Add(ResourceCompressionFormat::RESOURCE_COMPRESSION_FORMAT_GZIP);
  request.mutable_resource_capabilities()
      ->set_supports_confidential_aggregation(enable_confidential_aggregation);
  request.mutable_resource_capabilities()
      ->set_supports_willow_secure_aggregation(
          enable_willow_secure_aggregation);
  request.mutable_eligibility_eval_task_capabilities()
      ->set_supports_multiple_task_assignment(true);
  request.mutable_eligibility_eval_task_capabilities()
      ->set_supports_native_eets(true);

  return request;
}

EligibilityEvalTaskResponse GetFakeEnabledEligibilityEvalTaskResponse(
    const Resource& plan, const Resource& checkpoint,
    const std::string& execution_id, const std::string& target_uri_prefix,
    std::optional<Resource> population_eligibility_spec,
    const RetryWindow& accepted_retry_window,
    const RetryWindow& rejected_retry_window) {
  EligibilityEvalTaskResponse response;
  response.set_session_id(kEligibilityEvalSessionId);
  EligibilityEvalTask* eval_task = response.mutable_eligibility_eval_task();
  *eval_task->mutable_plan() = plan;
  *eval_task->mutable_init_checkpoint() = checkpoint;
  if (population_eligibility_spec.has_value()) {
    *eval_task->mutable_population_eligibility_spec() =
        population_eligibility_spec.value();
  }
  eval_task->set_execution_id(execution_id);
  ForwardingInfo* forwarding_info =
      response.mutable_task_assignment_forwarding_info();
  forwarding_info->set_target_uri_prefix(target_uri_prefix);
  *response.mutable_retry_window_if_accepted() = accepted_retry_window;
  *response.mutable_retry_window_if_rejected() = rejected_retry_window;
  return response;
}

EligibilityEvalTaskResponse GetFakeDisabledEligibilityEvalTaskResponse() {
  EligibilityEvalTaskResponse response;
  response.set_session_id(kEligibilityEvalSessionId);
  response.mutable_no_eligibility_eval_configured();
  ForwardingInfo* forwarding_info =
      response.mutable_task_assignment_forwarding_info();
  forwarding_info->set_target_uri_prefix(kTaskAssignmentTargetUri);
  *response.mutable_retry_window_if_accepted() = GetAcceptedRetryWindow();
  *response.mutable_retry_window_if_rejected() = GetRejectedRetryWindow();
  return response;
}

EligibilityEvalTaskResponse GetFakeRejectedEligibilityEvalTaskResponse() {
  EligibilityEvalTaskResponse response;
  response.mutable_rejection_info();
  *response.mutable_retry_window_if_accepted() = GetAcceptedRetryWindow();
  *response.mutable_retry_window_if_rejected() = GetRejectedRetryWindow();
  return response;
}

TaskEligibilityInfo GetFakeTaskEligibilityInfo() {
  TaskEligibilityInfo eligibility_info;
  TaskWeight* task_weight = eligibility_info.mutable_task_weights()->Add();
  task_weight->set_task_name("foo");
  task_weight->set_weight(567.8);
  return eligibility_info;
}

StartTaskAssignmentRequest GetExpectedStartTaskAssignmentRequest(
    const std::optional<TaskEligibilityInfo>& task_eligibility_info,
    bool enable_confidential_aggregation,
    bool enable_willow_secure_aggregation) {
  // Note: we don't expect population_name or session_id to be set, since they
  // should be set in the URI instead.
  StartTaskAssignmentRequest request;
  request.mutable_client_version()->set_version_code(kClientVersion);
  if (task_eligibility_info.has_value()) {
    *request.mutable_task_eligibility_info() = *task_eligibility_info;
  }
  request.mutable_resource_capabilities()
      ->mutable_supported_compression_formats()
      ->Add(ResourceCompressionFormat::RESOURCE_COMPRESSION_FORMAT_GZIP);
  request.mutable_resource_capabilities()
      ->set_supports_confidential_aggregation(enable_confidential_aggregation);
  request.mutable_resource_capabilities()
      ->set_supports_willow_secure_aggregation(
          enable_willow_secure_aggregation);
  return request;
}

StartTaskAssignmentResponse GetFakeRejectedTaskAssignmentResponse() {
  StartTaskAssignmentResponse response;
  response.mutable_rejection_info();
  return response;
}

TaskAssignment CreateTaskAssignment(
    const Resource& plan, const Resource& checkpoint,
    const std::string& federated_select_uri_template,
    const std::string& client_session_id,
    const std::string& aggregation_session_id, const std::string& task_name,
    const std::string& target_uri_prefix,
    int32_t minimum_clients_in_server_visible_aggregate,
    std::optional<Resource> confidential_data_access_policy,
    std::optional<Resource> signed_endorsements,
    std::optional<FederatedProtocol::WillowAggInfo> willow_agg_info) {
  TaskAssignment task_assignment;
  ForwardingInfo* forwarding_info =
      task_assignment.mutable_aggregation_data_forwarding_info();
  forwarding_info->set_target_uri_prefix(target_uri_prefix);
  task_assignment.set_session_id(client_session_id);
  task_assignment.set_aggregation_id(aggregation_session_id);
  task_assignment.set_authorization_token(kAuthorizationToken);
  task_assignment.set_task_name(task_name);
  *task_assignment.mutable_plan() = plan;
  *task_assignment.mutable_init_checkpoint() = checkpoint;
  task_assignment.mutable_federated_select_uri_info()->set_uri_template(
      federated_select_uri_template);
  if (minimum_clients_in_server_visible_aggregate > 0) {
    task_assignment.mutable_secure_aggregation_info()
        ->set_minimum_clients_in_server_visible_aggregate(
            minimum_clients_in_server_visible_aggregate);
  } else if (confidential_data_access_policy.has_value()) {
    *task_assignment.mutable_confidential_aggregation_info()
         ->mutable_data_access_policy() = *confidential_data_access_policy;
    if (signed_endorsements.has_value()) {
      *task_assignment.mutable_confidential_aggregation_info()
           ->mutable_signed_endorsements() = *signed_endorsements;
    }
  } else if (willow_agg_info.has_value()) {
    task_assignment.mutable_willow_aggregation_info()
        ->set_max_number_of_clients(willow_agg_info->max_number_of_clients);
    task_assignment.mutable_willow_aggregation_info()
        ->set_max_flattened_domain_size(
            willow_agg_info->max_flattened_domain_size);
    *task_assignment.mutable_willow_aggregation_info()
         ->mutable_input_spec()
         ->mutable_inline_resource()
         ->mutable_data() = std::string(willow_agg_info->input_spec);
  } else {
    task_assignment.mutable_aggregation_info();
  }
  return task_assignment;
}

StartTaskAssignmentResponse GetFakeTaskAssignmentResponse(
    const Resource& plan, const Resource& checkpoint,
    const std::string& federated_select_uri_template,
    const std::string& aggregation_session_id,
    int32_t minimum_clients_in_server_visible_aggregate,
    const std::string& target_uri_prefix,
    std::optional<Resource> confidential_data_access_policy,
    std::optional<Resource> signed_endorsements,
    std::optional<FederatedProtocol::WillowAggInfo> willow_agg_info) {
  StartTaskAssignmentResponse response;
  *response.mutable_task_assignment() = CreateTaskAssignment(
      plan, checkpoint, federated_select_uri_template, kClientSessionId,
      aggregation_session_id, kTaskName, target_uri_prefix,
      minimum_clients_in_server_visible_aggregate,
      confidential_data_access_policy, signed_endorsements, willow_agg_info);
  return response;
}

ReportTaskResultRequest GetExpectedReportTaskResultRequest(
    absl::string_view aggregation_id, absl::string_view task_name,
    ::google::rpc::Code code, absl::Duration train_duration) {
  ReportTaskResultRequest request;
  request.set_aggregation_id(std::string(aggregation_id));
  request.set_task_name(std::string(task_name));
  request.set_computation_status_code(code);
  ClientStats client_stats;
  *client_stats.mutable_computation_execution_duration() =
      TimeUtil::ConvertAbslToProtoDuration(train_duration);
  *request.mutable_client_stats() = client_stats;
  return request;
}

StartAggregationDataUploadResponse GetFakeStartAggregationDataUploadResponse(
    absl::string_view aggregation_resource_name,
    absl::string_view byte_stream_uri_prefix,
    absl::string_view second_stage_aggregation_uri_prefix) {
  StartAggregationDataUploadResponse response;
  ByteStreamResource* resource = response.mutable_resource();
  *resource->mutable_resource_name() = aggregation_resource_name;
  ForwardingInfo* data_upload_forwarding_info =
      resource->mutable_data_upload_forwarding_info();
  *data_upload_forwarding_info->mutable_target_uri_prefix() =
      byte_stream_uri_prefix;
  ForwardingInfo* aggregation_protocol_forwarding_info =
      response.mutable_aggregation_protocol_forwarding_info();
  *aggregation_protocol_forwarding_info->mutable_target_uri_prefix() =
      second_stage_aggregation_uri_prefix;
  response.set_client_token(kClientToken);
  return response;
}

StartConfidentialAggregationDataUploadResponse
GetFakeStartConfidentialAggregationDataUploadResponse(
    absl::string_view aggregation_resource_name,
    absl::string_view byte_stream_uri_prefix,
    absl::string_view second_stage_aggregation_uri_prefix,
    const ConfidentialEncryptionConfig& confidential_encryption_config) {
  StartConfidentialAggregationDataUploadResponse response;
  ByteStreamResource* resource = response.mutable_resource();
  *resource->mutable_resource_name() = aggregation_resource_name;
  ForwardingInfo* data_upload_forwarding_info =
      resource->mutable_data_upload_forwarding_info();
  *data_upload_forwarding_info->mutable_target_uri_prefix() =
      byte_stream_uri_prefix;
  ForwardingInfo* aggregation_protocol_forwarding_info =
      response.mutable_aggregation_protocol_forwarding_info();
  *aggregation_protocol_forwarding_info->mutable_target_uri_prefix() =
      second_stage_aggregation_uri_prefix;
  response.set_client_token(kClientToken);
  response.mutable_encryption_config()->mutable_inline_resource()->set_data(
      confidential_encryption_config.SerializeAsString());
  return response;
}

FakeHttpResponse CreateEmptySuccessHttpResponse() {
  return FakeHttpResponse(200, HeaderList(), "");
}

confidentialcompute::SignedEndorsements GetFakeSignedEndorsements() {
  confidentialcompute::SignedEndorsements signed_endorsements;
  auto signed_endorsement = signed_endorsements.add_signed_endorsement();
  signed_endorsement->mutable_endorsement()->set_serialized(
      "{\"payload\":{\"subject\":{\"name\":\"stefans signed endorsement\"}}}");
  signed_endorsement->mutable_signature()->set_key_id(1);
  signed_endorsement->mutable_signature()->set_raw("stefans public key");
  return signed_endorsements;
}

ComputationResults CreateFCCheckpointsResults() {
  // Create two fake checkpoints.
  std::string checkpoint1_str = "ckpt1";
  std::string checkpoint2_str = "ckpt2";
  confidentialcompute::PayloadMetadata metadata1;
  metadata1.mutable_event_time_range()->mutable_start_event_time()->set_year(
      2025);
  confidentialcompute::PayloadMetadata metadata2;
  metadata2.mutable_event_time_range()->mutable_start_event_time()->set_year(
      2026);
  FCCheckpoints checkpoints;
  checkpoints.push_back(
      {.payload = absl::Cord(checkpoint1_str), .metadata = metadata1});
  checkpoints.push_back(
      {.payload = absl::Cord(checkpoint2_str), .metadata = metadata2});
  ComputationResults results;
  results.emplace("fc_checkpoints", std::move(checkpoints));
  return results;
}

void HttpFederatedProtocolTest::SetUp() {
  SetUp(std::make_unique<TestingFakeWillowPayloadEncryptor>());
}

void HttpFederatedProtocolTest::SetUp(
    std::unique_ptr<WillowPayloadEncryptor> willow_payload_encryptor) {
  EXPECT_CALL(mock_flags_, federated_training_transient_errors_retry_delay_secs)
      .WillRepeatedly(Return(kTransientErrorsRetryPeriodSecs));
  EXPECT_CALL(mock_flags_,
              federated_training_transient_errors_retry_delay_jitter_percent)
      .WillRepeatedly(Return(kTransientErrorsRetryDelayJitterPercent));
  EXPECT_CALL(mock_flags_, federated_training_permanent_errors_retry_delay_secs)
      .WillRepeatedly(Return(kPermanentErrorsRetryPeriodSecs));
  EXPECT_CALL(mock_flags_,
              federated_training_permanent_errors_retry_delay_jitter_percent)
      .WillRepeatedly(Return(kPermanentErrorsRetryDelayJitterPercent));
  EXPECT_CALL(mock_flags_, federated_training_permanent_error_codes)
      .WillRepeatedly(Return(std::vector<int32_t>{
          static_cast<int32_t>(absl::StatusCode::kNotFound),
          static_cast<int32_t>(absl::StatusCode::kInvalidArgument),
          static_cast<int32_t>(absl::StatusCode::kUnimplemented)}));
  // Note that we disable compression in test to make it easier to verify the
  // request body. The compression logic is tested in
  // in_memory_request_response_test.cc.
  EXPECT_CALL(mock_flags_, disable_http_request_body_compression)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(mock_flags_, waiting_period_sec_for_cancellation)
      .WillRepeatedly(Return(kCancellationWaitingPeriodSec));
  EXPECT_CALL(mock_flags_, enable_confidential_aggregation)
      .WillRepeatedly(Return(false));
  EXPECT_CALL(mock_flags_, enable_relative_uri_prefix)
      .WillRepeatedly(Return(true));
  // Disable http retries to simplify transient error tests that return
  // retriable http errors.
  EXPECT_CALL(mock_flags_, http_retry_max_attempts).WillRepeatedly(Return(0));

  // We only initialize federated_protocol_ in this SetUp method, rather than
  // in the test's constructor, to ensure that we can set mock flag values
  // before the HttpFederatedProtocol constructor is called. Using
  // std::unique_ptr conveniently allows us to assign the field a new value
  // after construction (which we could not do if the field's type was
  // HttpFederatedProtocol, since it doesn't have copy or move constructors).
  federated_protocol_ = std::make_unique<HttpFederatedProtocol>(
      clock_, &mock_log_manager_, &mock_flags_, &mock_http_client_,
      absl::WrapUnique(mock_secagg_runner_factory_),
      &mock_secagg_event_publisher_, &mock_resource_cache_,
      absl::WrapUnique(mock_attestation_verifier_),
      std::move(willow_payload_encryptor), kEntryPointUri, kApiKey,
      kPopulationName, kRetryToken, kClientVersion, kAttestationMeasurement,
      mock_should_abort_.AsStdFunction(), absl::BitGen(),
      InterruptibleRunner::TimingConfig{
          .polling_period = absl::ZeroDuration(),
          .graceful_shutdown_period = absl::InfiniteDuration(),
          .extended_shutdown_period = absl::InfiniteDuration()});
}

void HttpFederatedProtocolTest::TearDown() {
  // Regardless of the outcome of the test (or the protocol interaction being
  // tested), network usage must always be reflected in the network stats
  // methods.
  HttpRequestHandle::SentReceivedBytes sent_received_bytes =
      mock_http_client_.TotalSentReceivedBytes();

  NetworkStats network_stats = federated_protocol_->GetNetworkStats();
  EXPECT_EQ(network_stats.bytes_downloaded, sent_received_bytes.received_bytes);
  EXPECT_EQ(network_stats.bytes_uploaded, sent_received_bytes.sent_bytes);
  // If any network traffic occurred, we expect to see some time reflected in
  // the duration.
  if (network_stats.bytes_uploaded > 0) {
    EXPECT_THAT(network_stats.network_duration, Gt(absl::ZeroDuration()));
  }
}

absl::Status HttpFederatedProtocolTest::RunSuccessfulEligibilityEvalCheckin(
    bool eligibility_eval_enabled, bool enable_confidential_aggregation,
    bool enable_willow_secure_aggregation, bool set_relative_uri) {
  EligibilityEvalTaskResponse eval_task_response;
  if (eligibility_eval_enabled) {
    // We return a fake response which returns the plan/initial checkpoint
    // data inline, to keep things simple.
    std::string expected_plan = kPlan;
    Resource plan_resource;
    plan_resource.mutable_inline_resource()->set_data(kPlan);
    std::string expected_checkpoint = kInitCheckpoint;
    Resource checkpoint_resource;
    checkpoint_resource.mutable_inline_resource()->set_data(
        expected_checkpoint);
    if (set_relative_uri) {
      eval_task_response = GetFakeEnabledEligibilityEvalTaskResponse(
          plan_resource, checkpoint_resource, kEligibilityEvalExecutionId, "/");
    } else {
      eval_task_response = GetFakeEnabledEligibilityEvalTaskResponse(
          plan_resource, checkpoint_resource, kEligibilityEvalExecutionId);
    }
  } else {
    eval_task_response = GetFakeDisabledEligibilityEvalTaskResponse();
  }
  std::string request_uri =
      "https://initial.uri/v1/eligibilityevaltasks/"
      "TEST%2FPOPULATION:request?%24alt=proto";
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  request_uri, HttpRequest::Method::kPost, _,
                  EligibilityEvalTaskRequestMatcher(
                      EqualsProto(GetExpectedEligibilityEvalTaskRequest(
                          enable_confidential_aggregation,
                          enable_willow_secure_aggregation))))))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), eval_task_response.SerializeAsString())));

  // The 'EET received' callback should be called, even if the task resource
  // data was available inline.
  if (eligibility_eval_enabled) {
    EXPECT_CALL(mock_eet_received_callback_,
                Call(FieldsAre(FieldsAre("", ""), kEligibilityEvalExecutionId,
                               Eq(std::nullopt), _)));
  }

  return federated_protocol_
      ->EligibilityEvalCheckin(mock_eet_received_callback_.AsStdFunction())
      .status();
}

absl::StatusOr<FederatedProtocol::CheckinResult>
HttpFederatedProtocolTest::RunSuccessfulCheckin(
    bool report_eligibility_eval_result,
    std::optional<std::string> confidential_data_access_policy,
    std::optional<FederatedProtocol::WillowAggInfo> willow_agg_info,
    bool set_relative_uri, std::optional<std::string> signed_endorsements) {
  // We return a fake response which returns the plan/initial checkpoint
  // data inline, to keep things simple.
  std::string expected_plan = kPlan;
  std::string plan_uri = "https://fake.uri/plan";
  Resource plan_resource;
  plan_resource.set_uri(plan_uri);
  std::string expected_checkpoint = kInitCheckpoint;
  Resource checkpoint_resource;
  checkpoint_resource.mutable_inline_resource()->set_data(expected_checkpoint);
  std::string expected_aggregation_session_id = kAggregationSessionId;
  std::optional<Resource> confidential_agg_resource;
  std::optional<Resource> signed_endorsements_resource;
  if (confidential_data_access_policy.has_value()) {
    confidential_agg_resource = Resource();
    confidential_agg_resource->mutable_inline_resource()->set_data(
        *confidential_data_access_policy);
    if (signed_endorsements.has_value()) {
      signed_endorsements_resource = Resource();
      signed_endorsements_resource->mutable_inline_resource()->set_data(
          *signed_endorsements);
    }
  }

  StartTaskAssignmentResponse task_assignment_response =
      GetFakeTaskAssignmentResponse(
          plan_resource, checkpoint_resource, kFederatedSelectUriTemplate,
          expected_aggregation_session_id, 0,
          set_relative_uri ? "/" : kAggregationTargetUri,
          confidential_agg_resource, signed_endorsements_resource,
          willow_agg_info);

  std::string request_uri;
  if (set_relative_uri) {
    request_uri =
        "https://initial.uri/v1/populations/TEST%2FPOPULATION/"
        "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto";
  } else {
    request_uri =
        "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
        "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto";
  }
  TaskEligibilityInfo expected_eligibility_info = GetFakeTaskEligibilityInfo();
  bool enable_confidential_aggregation = false;
  bool enable_willow_secure_aggregation = false;
  if (confidential_data_access_policy.has_value()) {
    enable_confidential_aggregation = true;
  }
  if (willow_agg_info.has_value()) {
    // Enable confidential aggregation even when there is no confidential data
    // access policy
    enable_confidential_aggregation = true;
    enable_willow_secure_aggregation = true;
  }

  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          request_uri, HttpRequest::Method::kPost, _,
          StartTaskAssignmentRequestMatcher(
              EqualsProto(GetExpectedStartTaskAssignmentRequest(
                  expected_eligibility_info, enable_confidential_aggregation,
                  enable_willow_secure_aggregation))))))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          CreateDoneOperation(kOperationName, task_assignment_response)
              .SerializeAsString())));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), expected_plan)));

  if (report_eligibility_eval_result) {
    std::string report_eet_request_uri =
        "https://initial.uri/v1/populations/TEST%2FPOPULATION/"
        "eligibilityevaltasks/"
        "ELIGIBILITY%2FSESSION%23ID:reportresult?%24alt=proto";
    ExpectSuccessfulReportEligibilityEvalTaskResultRequest(
        report_eet_request_uri, absl::OkStatus());
  }

  return federated_protocol_->Checkin(
      expected_eligibility_info, mock_task_received_callback_.AsStdFunction(),
      std::nullopt);
}

absl::StatusOr<FederatedProtocol::MultipleTaskAssignments>
HttpFederatedProtocolTest::RunSuccessfulMultipleTaskAssignments(
    bool eligibility_eval_enabled, bool enable_confidential_aggregation,
    bool enable_attestation_transparency_verifier,
    std::optional<Resource> confidential_data_access_policy,
    std::optional<Resource> signed_endorsements) {
  if (eligibility_eval_enabled) {
    std::string report_eet_request_uri =
        "https://initial.uri/v1/populations/TEST%2FPOPULATION/"
        "eligibilityevaltasks/"
        "ELIGIBILITY%2FSESSION%23ID:reportresult?%24alt=proto";
    ExpectSuccessfulReportEligibilityEvalTaskResultRequest(
        report_eet_request_uri, absl::OkStatus());
  }

  std::vector<std::string> task_names{kMultiTaskId_1, kMultiTaskId_2};

  PerformMultipleTaskAssignmentsRequest request;
  request.mutable_client_version()->set_version_code(kClientVersion);
  request.mutable_resource_capabilities()->add_supported_compression_formats(
      ResourceCompressionFormat::RESOURCE_COMPRESSION_FORMAT_GZIP);
  if (enable_confidential_aggregation) {
    request.mutable_resource_capabilities()
        ->set_supports_confidential_aggregation(true);
  }
  if (enable_attestation_transparency_verifier) {
    request.mutable_resource_capabilities()
        ->set_supports_attestation_transparency_verifier(true);
  }
  for (const auto& task_name : task_names) {
    request.add_task_names(task_name);
  }

  PerformMultipleTaskAssignmentsResponse response;
  Resource plan_1;
  std::string expected_plan_1 = "plan1";
  *plan_1.mutable_inline_resource()->mutable_data() = expected_plan_1;
  Resource checkpoint_1;
  std::string expected_checkpoint_1 = "checkpoint1";
  *checkpoint_1.mutable_inline_resource()->mutable_data() =
      expected_checkpoint_1;
  *response.add_task_assignments() = CreateTaskAssignment(
      plan_1, checkpoint_1, kFederatedSelectUriTemplate,
      kMultiTaskClientSessionId_1, kMultiTaskAggregationSessionId_1,
      kMultiTaskId_1, kAggregationTargetUri,
      enable_confidential_aggregation ? 0
                                      : kMinimumClientsInServerVisibleAggregate,
      confidential_data_access_policy, signed_endorsements);
  Resource plan_2;
  std::string plan_uri = "https://fake.uri/plan";
  plan_2.set_uri(plan_uri);
  std::string checkpoint_uri = "https://fake.uri/checkpoint";
  Resource checkpoint_2;
  checkpoint_2.set_uri(checkpoint_uri);
  *response.add_task_assignments() = CreateTaskAssignment(
      plan_2, checkpoint_2, kFederatedSelectUriTemplate,
      kMultiTaskClientSessionId_2, kMultiTaskAggregationSessionId_2,
      kMultiTaskId_2, kAggregationTargetUri,
      enable_confidential_aggregation ? 0
                                      : kMinimumClientsInServerVisibleAggregate,
      confidential_data_access_policy);
  std::string expected_plan_2 = "expected_plan_2";
  std::string expected_checkpoint_2 = "expected_checkpoint_2";

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://taskassignment.uri/v1/populations/"
                  "TEST%2FPOPULATION/"
                  "taskassignments/"
                  "ELIGIBILITY%2FSESSION%23ID:performmultiple?%24alt=proto",
                  HttpRequest::Method::kPost, _, request.SerializeAsString())))
      .WillOnce(Return(
          FakeHttpResponse(200, HeaderList(), response.SerializeAsString())));
  EXPECT_CALL(mock_multiple_tasks_received_callback_, Call(2));
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  checkpoint_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(
          Return(FakeHttpResponse(200, HeaderList(), expected_checkpoint_2)));
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), expected_plan_2)));

  return federated_protocol_->PerformMultipleTaskAssignments(
      task_names, mock_multiple_tasks_received_callback_.AsStdFunction(),
      std::nullopt);
}

ReportResult HttpFederatedProtocolTest::RunSuccessfulUploadViaSimpleAgg(
    absl::string_view client_session_id,
    std::optional<std::string> task_identifier,
    absl::string_view aggregation_session_id, absl::string_view task_name,
    absl::Duration plan_duration, absl::string_view checkpoint_str,
    bool use_per_task_upload) {
  std::string report_task_result_request_url = absl::StrCat(
      "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
      "taskassignments/",
      client_session_id, ":reportresult?%24alt=proto");
  ExpectSuccessfulReportTaskResultRequest(report_task_result_request_url,
                                          aggregation_session_id, task_name,
                                          plan_duration);
  std::string start_aggregation_data_upload_request_url = absl::StrCat(
      "https://aggregation.uri/v1/aggregations/", aggregation_session_id,
      "/clients/AUTHORIZATION_TOKEN:startdataupload?%24alt=proto");
  ExpectSuccessfulStartAggregationDataUploadRequest(
      start_aggregation_data_upload_request_url, kResourceName,
      kByteStreamTargetUri, kSecondStageAggregationTargetUri);
  ExpectSuccessfulByteStreamUploadRequest(
      "https://bytestream.uri/upload/v1/media/"
      "CHECKPOINT_RESOURCE?upload_protocol=raw",
      checkpoint_str);
  std::string submit_aggregation_result_request_url = absl::StrCat(
      "https://aggregation.second.uri/v1/aggregations/", aggregation_session_id,
      "/clients/CLIENT_TOKEN:submit?%24alt=proto");
  ExpectSuccessfulSubmitAggregationResultRequest(
      submit_aggregation_result_request_url);
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", std::string(checkpoint_str));
  if (use_per_task_upload) {
    return federated_protocol_->ReportCompleted(std::move(results),
                                                plan_duration, task_identifier);
  } else {
    return federated_protocol_->ReportCompleted(std::move(results),
                                                plan_duration, std::nullopt);
  }
}

void HttpFederatedProtocolTest::
    ExpectSuccessfulReportEligibilityEvalTaskResultRequest(
        absl::string_view expected_request_uri, absl::Status eet_status) {
  ReportEligibilityEvalTaskResultRequest report_eet_request;
  report_eet_request.set_status_code(
      static_cast<google::rpc::Code>(eet_status.code()));
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          std::string(expected_request_uri), HttpRequest::Method::kPost, _,
          ReportEligibilityEvalTaskResultRequestMatcher(
              EqualsProto(report_eet_request)))))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "")));
}

void HttpFederatedProtocolTest::ExpectSuccessfulReportTaskResultRequest(
    absl::string_view expected_report_result_uri,
    absl::string_view aggregation_session_id, absl::string_view task_name,
    absl::Duration plan_duration) {
  ReportTaskResultResponse report_task_result_response;
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  std::string(expected_report_result_uri),
                  HttpRequest::Method::kPost, _,
                  ReportTaskResultRequestMatcher(
                      EqualsProto(GetExpectedReportTaskResultRequest(
                          aggregation_session_id, task_name,
                          google::rpc::Code::OK, plan_duration))))))
      .WillOnce(Return(CreateEmptySuccessHttpResponse()));
}

void HttpFederatedProtocolTest::
    ExpectSuccessfulStartAggregationDataUploadRequest(
        absl::string_view expected_start_data_upload_uri,
        absl::string_view aggregation_resource_name,
        absl::string_view byte_stream_uri_prefix,
        absl::string_view second_stage_aggregation_uri_prefix,
        bool set_relative_uri_prefix,
        std::optional<ConfidentialEncryptionConfig>
            confidential_encryption_config) {
  Operation pending_operation_response =
      CreatePendingOperation("operations/foo#bar");
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          std::string(expected_start_data_upload_uri),
          HttpRequest::Method::kPost, _,
          confidential_encryption_config.has_value()
              ? StartConfidentialAggregationDataUploadRequest()
                    .SerializeAsString()
              : StartAggregationDataUploadRequest().SerializeAsString())))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), pending_operation_response.SerializeAsString())));
  std::string expected_operation_uri;
  if (set_relative_uri_prefix) {
    expected_operation_uri =
        "https://initial.uri/v1/operations/foo%23bar?%24alt=proto";
  } else {
    expected_operation_uri =
        "https://aggregation.uri/v1/operations/foo%23bar?%24alt=proto";
  }

  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          // Note that the '#' character is encoded as "%23".
          expected_operation_uri, HttpRequest::Method::kGet, _,
          GetOperationRequestMatcher(EqualsProto(GetOperationRequest())))))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          confidential_encryption_config.has_value()
              ? CreateDoneOperation(
                    kOperationName,
                    GetFakeStartConfidentialAggregationDataUploadResponse(
                        aggregation_resource_name, byte_stream_uri_prefix,
                        second_stage_aggregation_uri_prefix,
                        *confidential_encryption_config))
                    .SerializeAsString()
              : CreateDoneOperation(
                    kOperationName,
                    GetFakeStartAggregationDataUploadResponse(
                        aggregation_resource_name, byte_stream_uri_prefix,
                        second_stage_aggregation_uri_prefix))
                    .SerializeAsString())));
}

void HttpFederatedProtocolTest::ExpectSuccessfulByteStreamUploadRequest(
    absl::string_view byte_stream_upload_uri,
    absl::string_view checkpoint_str) {
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  std::string(byte_stream_upload_uri),
                  HttpRequest::Method::kPost, _, std::string(checkpoint_str))))
      .WillOnce(Return(CreateEmptySuccessHttpResponse()));
}

void HttpFederatedProtocolTest::ExpectSuccessfulSubmitAggregationResultRequest(
    absl::string_view expected_submit_aggregation_result_uri,
    bool confidential_aggregation, std::string resource_name) {
  std::string expected_request_proto;
  if (confidential_aggregation) {
    SubmitConfidentialAggregationResultRequest
        submit_aggregation_result_request;
    submit_aggregation_result_request.set_resource_name(resource_name);
    expected_request_proto =
        submit_aggregation_result_request.SerializeAsString();
  } else {
    SubmitAggregationResultRequest submit_aggregation_result_request;
    submit_aggregation_result_request.set_resource_name(resource_name);
    expected_request_proto =
        submit_aggregation_result_request.SerializeAsString();
  }
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  std::string(expected_submit_aggregation_result_uri),
                  HttpRequest::Method::kPost, _, expected_request_proto)))
      .WillOnce(Return(CreateEmptySuccessHttpResponse()));
}

void HttpFederatedProtocolTest::ExpectSuccessfulAbortAggregationRequest(
    absl::string_view base_uri, bool confidential_aggregation) {
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  absl::StrCat(base_uri,
                               confidential_aggregation
                                   ? "/v1/confidentialaggregations/"
                                   : "/v1/aggregations/",
                               "AGGREGATION_SESSION_ID/clients/"
                               "CLIENT_TOKEN:abort?%24alt=proto"),
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(CreateEmptySuccessHttpResponse()));
}

void HttpFederatedProtocolTest::
    ExpectReportCompletedMatchesArgumentsForFakeWillowDecryptor(
        ComputationResults results, absl::Duration plan_duration,
        FederatedProtocol::WillowAggInfo willow_agg_info,
        absl::string_view public_key, absl::string_view checkpoint_string) {
  // Capture the raw uploaded data for later
  std::string uploaded_data;
  EXPECT_CALL(mock_http_client_, PerformSingleRequest(SimpleHttpRequestMatcher(
                                     "https://bytestream.uri/upload/v1/media/"
                                     "CHECKPOINT_RESOURCE?upload_protocol=raw",
                                     HttpRequest::Method::kPost, _, _)))
      .WillOnce([&uploaded_data](MockHttpClient::SimpleHttpRequest request) {
        uploaded_data = request.body;
        return CreateEmptySuccessHttpResponse();
      });
  ExpectSuccessfulSubmitAggregationResultRequest(
      "https://aggregation.second.uri/v1/confidentialaggregations/"
      "AGGREGATION_SESSION_ID/clients/CLIENT_TOKEN:submit?%24alt=proto",
      true);
  auto result = federated_protocol_->ReportCompleted(
      std::move(results), plan_duration, std::nullopt);
  EXPECT_THAT(result, IsOkReportResult());
  // Check the uploaded data to verify that the encryptor received the correct
  // arguments
  std::string expected_uploaded_data = absl::StrFormat(
      "%v%v%v%v%v", willow_agg_info.input_spec,
      willow_agg_info.max_flattened_domain_size,
      willow_agg_info.max_number_of_clients, public_key, checkpoint_string);
  EXPECT_EQ(uploaded_data, expected_uploaded_data);
}

}  // namespace fcp::client::http::internal
