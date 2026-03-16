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

#ifndef FCP_CLIENT_HTTP_HTTP_FEDERATED_PROTOCOL_TEST_UTILS_H_
#define FCP_CLIENT_HTTP_HTTP_FEDERATED_PROTOCOL_TEST_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "google/longrunning/operations.pb.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/duration.pb.h"
#include "google/rpc/code.pb.h"
#include "google/type/datetime.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/base/clock.h"
#include "fcp/client/cache/test_helpers.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/event_time_range.pb.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/http/http_federated_protocol.h"
#include "fcp/client/http/testing/test_helpers.h"
#include "fcp/client/test_helpers.h"
#include "fcp/client/willow/willow_payload_encryptor.h"
#include "fcp/confidentialcompute/cose.h"
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
#include "fcp/testing/testing.h"
#include "proto/attestation/endorsement.pb.h"

namespace fcp::client::http::internal {

using ::fcp::EqualsProto;
using ::fcp::IsCode;
using ::fcp::client::ReportOutcome;
using ::fcp::client::ReportResult;
using ::fcp::client::http::FakeHttpResponse;
using ::fcp::client::http::MockableHttpClient;
using ::fcp::client::http::MockHttpClient;
using ::fcp::client::http::SimpleHttpRequestMatcher;
using ::fcp::client::willow::TestingFakeWillowPayloadEncryptor;
using ::fcp::client::willow::WillowPayloadEncryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::google::internal::federated::plan::PopulationEligibilitySpec;
using ::google::internal::federatedcompute::v1::ByteStreamResource;
using ::google::internal::federatedcompute::v1::ClientStats;
using ::google::internal::federatedcompute::v1::ConfidentialEncryptionConfig;
using ::google::internal::federatedcompute::v1::EligibilityEvalTask;
using ::google::internal::federatedcompute::v1::EligibilityEvalTaskRequest;
using ::google::internal::federatedcompute::v1::EligibilityEvalTaskResponse;
using ::google::internal::federatedcompute::v1::ForwardingInfo;
using ::google::internal::federatedcompute::v1::
    PerformMultipleTaskAssignmentsRequest;
using ::google::internal::federatedcompute::v1::
    PerformMultipleTaskAssignmentsResponse;
using ::google::internal::federatedcompute::v1::
    ReportEligibilityEvalTaskResultRequest;
using ::google::internal::federatedcompute::v1::ReportTaskResultRequest;
using ::google::internal::federatedcompute::v1::ReportTaskResultResponse;
using ::google::internal::federatedcompute::v1::Resource;
using ::google::internal::federatedcompute::v1::ResourceCompressionFormat;
using ::google::internal::federatedcompute::v1::RetryWindow;
using ::google::internal::federatedcompute::v1::SecureAggregandExecutionInfo;
using ::google::internal::federatedcompute::v1::
    StartAggregationDataUploadRequest;
using ::google::internal::federatedcompute::v1::
    StartAggregationDataUploadResponse;
using ::google::internal::federatedcompute::v1::
    StartConfidentialAggregationDataUploadRequest;
using ::google::internal::federatedcompute::v1::
    StartConfidentialAggregationDataUploadResponse;
using ::google::internal::federatedcompute::v1::StartSecureAggregationRequest;
using ::google::internal::federatedcompute::v1::StartSecureAggregationResponse;
using ::google::internal::federatedcompute::v1::StartTaskAssignmentRequest;
using ::google::internal::federatedcompute::v1::StartTaskAssignmentResponse;
using ::google::internal::federatedcompute::v1::SubmitAggregationResultRequest;
using ::google::internal::federatedcompute::v1::
    SubmitConfidentialAggregationResultRequest;
using ::google::internal::federatedcompute::v1::TaskAssignment;
using ::google::internal::federatedml::v2::TaskEligibilityInfo;
using ::google::internal::federatedml::v2::TaskWeight;
using ::google::longrunning::GetOperationRequest;
using ::google::longrunning::Operation;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ByMove;
using ::testing::DescribeMatcher;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Eq;
using ::testing::ExplainMatchResult;
using ::testing::Field;
using ::testing::FieldsAre;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::InSequence;
using ::testing::IsEmpty;
using ::testing::Lt;
using ::testing::MockFunction;
using ::testing::NiceMock;
using ::testing::Not;
using ::testing::Optional;
using ::testing::Pair;
using ::testing::Return;
using ::testing::StrEq;
using ::testing::StrictMock;
using ::testing::UnorderedElementsAre;
using ::testing::VariantWith;
using ::testing::WithArg;

MATCHER(IsOkReportResult, "") {
  return arg.outcome == ReportOutcome::kSuccess && arg.status.ok();
}

MATCHER_P2(IsErrorReportResult, code, message_matcher, "") {
  return (arg.outcome == ReportOutcome::kFailure) &&
         arg.status.code() == code &&
         ExplainMatchResult(message_matcher, arg.status.message(),
                            result_listener);
}

constexpr char kEntryPointUri[] = "https://initial.uri/";
constexpr char kTaskAssignmentTargetUri[] = "https://taskassignment.uri/";
constexpr char kAggregationTargetUri[] = "https://aggregation.uri/";
constexpr char kSecondStageAggregationTargetUri[] =
    "https://aggregation.second.uri/";
constexpr char kByteStreamTargetUri[] = "https://bytestream.uri/";
constexpr char kApiKey[] = "TEST_APIKEY";
// Note that we include a '/' character in the population name, which allows us
// to verify that it is correctly URL-encoded into "%2F".
constexpr char kPopulationName[] = "TEST/POPULATION";
constexpr char kEligibilityEvalExecutionId[] = "ELIGIBILITY_EXECUTION_ID";
// Note that we include a '/' and '#' characters in the population name, which
// allows us to verify that it is correctly URL-encoded into "%2F" and "%23".
constexpr char kEligibilityEvalSessionId[] = "ELIGIBILITY/SESSION#ID";
constexpr char kPlan[] = "CLIENT_ONLY_PLAN";
constexpr char kInitCheckpoint[] = "INIT_CHECKPOINT";
constexpr char kRetryToken[] = "OLD_RETRY_TOKEN";
constexpr char kClientVersion[] = "CLIENT_VERSION";
constexpr char kAttestationMeasurement[] = "ATTESTATION_MEASUREMENT";
constexpr char kClientSessionId[] = "CLIENT_SESSION_ID";
constexpr char kAggregationSessionId[] = "AGGREGATION_SESSION_ID";
constexpr char kAuthorizationToken[] = "AUTHORIZATION_TOKEN";
constexpr char kTaskName[] = "TASK_NAME";
constexpr char kClientToken[] = "CLIENT_TOKEN";
constexpr char kResourceName[] = "CHECKPOINT_RESOURCE";
constexpr char kFederatedSelectUriTemplate[] = "https://federated.select";
constexpr char kOperationName[] = "my_operation";
constexpr char kMultiTaskId_1[] = "TASK_1";
constexpr char kMultiTaskClientSessionId_1[] = "CLIENT_SESSION_ID_1";
constexpr char kMultiTaskAggregationSessionId_1[] = "AGGREGATION_SESSION_ID_1";
constexpr char kMultiTaskId_2[] = "TASK_2";
constexpr char kMultiTaskClientSessionId_2[] = "CLIENT_SESSION_ID_2";
constexpr char kMultiTaskAggregationSessionId_2[] = "AGGREGATION_SESSION_ID_2";

const int32_t kCancellationWaitingPeriodSec = 1;
const int32_t kMinimumClientsInServerVisibleAggregate = 2;

MATCHER_P(EligibilityEvalTaskRequestMatcher, matcher,
          absl::StrCat(negation ? "doesn't parse" : "parses",
                       " as an EligibilityEvalTaskRequest, and that ",
                       DescribeMatcher<EligibilityEvalTaskRequest>(matcher,
                                                                   negation))) {
  EligibilityEvalTaskRequest request;
  if (!request.ParseFromString(arg)) {
    return false;
  }
  return ExplainMatchResult(matcher, request, result_listener);
}

MATCHER_P(
    ReportEligibilityEvalTaskResultRequestMatcher, matcher,
    absl::StrCat(negation ? "doesn't parse" : "parses",
                 " as a ReportEligibilityEvalTaskResultRequest, and that ",
                 DescribeMatcher<ReportEligibilityEvalTaskResultRequest>(
                     matcher, negation))) {
  ReportEligibilityEvalTaskResultRequest request;
  if (!request.ParseFromString(arg)) {
    return false;
  }
  return ExplainMatchResult(matcher, request, result_listener);
}

MATCHER_P(StartTaskAssignmentRequestMatcher, matcher,
          absl::StrCat(negation ? "doesn't parse" : "parses",
                       " as a StartTaskAssignmentRequest, and that ",
                       DescribeMatcher<StartTaskAssignmentRequest>(matcher,
                                                                   negation))) {
  StartTaskAssignmentRequest request;
  if (!request.ParseFromString(arg)) {
    return false;
  }
  return ExplainMatchResult(matcher, request, result_listener);
}

MATCHER_P(GetOperationRequestMatcher, matcher,
          absl::StrCat(negation ? "doesn't parse" : "parses",
                       " as a GetOperationRequest, and that ",
                       DescribeMatcher<GetOperationRequest>(matcher,
                                                            negation))) {
  GetOperationRequest request;
  if (!request.ParseFromString(arg)) {
    return false;
  }
  return ExplainMatchResult(matcher, request, result_listener);
}

MATCHER_P(ReportTaskResultRequestMatcher, matcher,
          absl::StrCat(negation ? "doesn't parse" : "parses",
                       " as a ReportTaskResultRequest, and that ",
                       DescribeMatcher<ReportTaskResultRequest>(matcher,
                                                                negation))) {
  ReportTaskResultRequest request;
  if (!request.ParseFromString(arg)) {
    return false;
  }
  return ExplainMatchResult(matcher, request, result_listener);
}

MATCHER(IsOk, "") { return arg.ok(); }

MATCHER_P(IsOkAndHolds, m, "") {
  return testing::ExplainMatchResult(IsOk(), arg, result_listener) &&
         testing::ExplainMatchResult(m, arg.value(), result_listener);
}
MATCHER_P(EqTaskIdentifier, task_identifier, "") {
  return testing::ExplainMatchResult(task_identifier, arg.task_identifier,
                                     result_listener);
}

constexpr int kTransientErrorsRetryPeriodSecs = 10;
constexpr double kTransientErrorsRetryDelayJitterPercent = 0.1;
constexpr double kExpectedTransientErrorsRetryPeriodSecsMin = 9.0;
constexpr double kExpectedTransientErrorsRetryPeriodSecsMax = 11.0;
constexpr int kPermanentErrorsRetryPeriodSecs = 100;
constexpr double kPermanentErrorsRetryDelayJitterPercent = 0.2;
constexpr double kExpectedPermanentErrorsRetryPeriodSecsMin = 80.0;
constexpr double kExpectedPermanentErrorsRetryPeriodSecsMax = 120.0;

void ExpectTransientErrorRetryWindow(
    const ::google::internal::federatedml::v2::RetryWindow& retry_window);

void ExpectPermanentErrorRetryWindow(
    const ::google::internal::federatedml::v2::RetryWindow& retry_window);

RetryWindow GetAcceptedRetryWindow();

void ExpectAcceptedRetryWindow(
    const ::google::internal::federatedml::v2::RetryWindow& retry_window);

RetryWindow GetRejectedRetryWindow();

void ExpectRejectedRetryWindow(
    const ::google::internal::federatedml::v2::RetryWindow& retry_window);
EligibilityEvalTaskRequest GetExpectedEligibilityEvalTaskRequest(
    bool enable_confidential_aggregation = false,
    bool enable_willow_secure_aggregation = false);

EligibilityEvalTaskResponse GetFakeEnabledEligibilityEvalTaskResponse(
    const Resource& plan, const Resource& checkpoint,
    const std::string& execution_id,
    const std::string& target_uri_prefix = kTaskAssignmentTargetUri,
    std::optional<Resource> population_eligibility_spec = std::nullopt,
    const RetryWindow& accepted_retry_window = GetAcceptedRetryWindow(),
    const RetryWindow& rejected_retry_window = GetRejectedRetryWindow());

EligibilityEvalTaskResponse GetFakeDisabledEligibilityEvalTaskResponse();

EligibilityEvalTaskResponse GetFakeRejectedEligibilityEvalTaskResponse();

TaskEligibilityInfo GetFakeTaskEligibilityInfo();

StartTaskAssignmentRequest GetExpectedStartTaskAssignmentRequest(
    const std::optional<TaskEligibilityInfo>& task_eligibility_info,
    bool enable_confidential_aggregation = false,
    bool enable_willow_secure_aggregation = false);

StartTaskAssignmentResponse GetFakeRejectedTaskAssignmentResponse();

TaskAssignment CreateTaskAssignment(
    const Resource& plan, const Resource& checkpoint,
    const std::string& federated_select_uri_template,
    const std::string& client_session_id,
    const std::string& aggregation_session_id, const std::string& task_name,
    const std::string& target_uri_prefix,
    int32_t minimum_clients_in_server_visible_aggregate,
    std::optional<Resource> confidential_data_access_policy = std::nullopt,
    std::optional<Resource> signed_endorsements = std::nullopt,
    std::optional<FederatedProtocol::WillowAggInfo> willow_agg_info =
        std::nullopt);

StartTaskAssignmentResponse GetFakeTaskAssignmentResponse(
    const Resource& plan, const Resource& checkpoint,
    const std::string& federated_select_uri_template,
    const std::string& aggregation_session_id,
    int32_t minimum_clients_in_server_visible_aggregate,
    const std::string& target_uri_prefix = kAggregationTargetUri,
    std::optional<Resource> confidential_data_access_policy = std::nullopt,
    std::optional<Resource> signed_endorsements = std::nullopt,
    std::optional<FederatedProtocol::WillowAggInfo> willow_agg_info =
        std::nullopt);

ReportTaskResultRequest GetExpectedReportTaskResultRequest(
    absl::string_view aggregation_id, absl::string_view task_name,
    ::google::rpc::Code code, absl::Duration train_duration);

StartAggregationDataUploadResponse GetFakeStartAggregationDataUploadResponse(
    absl::string_view aggregation_resource_name,
    absl::string_view byte_stream_uri_prefix,
    absl::string_view second_stage_aggregation_uri_prefix);

StartConfidentialAggregationDataUploadResponse
GetFakeStartConfidentialAggregationDataUploadResponse(
    absl::string_view aggregation_resource_name,
    absl::string_view byte_stream_uri_prefix,
    absl::string_view second_stage_aggregation_uri_prefix,
    const ConfidentialEncryptionConfig& confidential_encryption_config);

FakeHttpResponse CreateEmptySuccessHttpResponse();

confidentialcompute::SignedEndorsements GetFakeSignedEndorsements();

ComputationResults CreateFCCheckpointsResults();

class HttpFederatedProtocolTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void SetUp(std::unique_ptr<WillowPayloadEncryptor> willow_payload_encryptor);
  void TearDown() override;

  // This function runs a successful EligibilityEvalCheckin() that results in an
  // eligibility eval payload being returned by the server (if
  // `eligibility_eval_enabled` is true), or results in a 'no eligibility eval
  // configured' response (if `eligibility_eval_enabled` is false). This is a
  // utility function used by Checkin*() tests that depend on a prior,
  // successful execution of EligibilityEvalCheckin(). It returns a
  // absl::Status, which the caller should verify is OK using ASSERT_OK.
  absl::Status RunSuccessfulEligibilityEvalCheckin(
      bool eligibility_eval_enabled = true,
      bool enable_confidential_aggregation = false,
      bool enable_willow_secure_aggregation = false,
      bool set_relative_uri = false);

  // This function runs a successful Checkin() that results in a
  // task assignment payload being returned by the server. This is a
  // utility function used by Report*() tests that depend on a prior,
  // successful execution of Checkin(). It returns a
  // absl::StatusOr<CheckinResult>, which the caller should verify is OK using
  // ASSERT_OK.
  // For Willow tasks, willow_agg_info holds the input_spec and the
  // max_number_of_clients, which are both required for encrypting the payload.
  absl::StatusOr<FederatedProtocol::CheckinResult> RunSuccessfulCheckin(
      bool report_eligibility_eval_result = true,
      std::optional<std::string> confidential_data_access_policy = std::nullopt,
      std::optional<FederatedProtocol::WillowAggInfo> willow_agg_info =
          std::nullopt,
      bool set_relative_uri = false,
      std::optional<std::string> signed_endorsements = std::nullopt);

  absl::StatusOr<FederatedProtocol::MultipleTaskAssignments>
  RunSuccessfulMultipleTaskAssignments(
      bool eligibility_eval_enabled = true,
      bool enable_confidential_aggregation = false,
      bool enable_attestation_transparency_verifier = false,
      std::optional<Resource> confidential_data_access_policy = std::nullopt,
      std::optional<Resource> signed_endorsements = std::nullopt);

  ReportResult RunSuccessfulUploadViaSimpleAgg(
      absl::string_view client_session_id,
      std::optional<std::string> task_identifier,
      absl::string_view aggregation_session_id, absl::string_view task_name,
      absl::Duration plan_duration, absl::string_view checkpoint_str,
      bool use_per_task_upload = true);

  void ExpectSuccessfulReportEligibilityEvalTaskResultRequest(
      absl::string_view expected_request_uri, absl::Status eet_status);

  void ExpectSuccessfulReportTaskResultRequest(
      absl::string_view expected_report_result_uri,
      absl::string_view aggregation_session_id, absl::string_view task_name,
      absl::Duration plan_duration);

  void ExpectSuccessfulStartAggregationDataUploadRequest(
      absl::string_view expected_start_data_upload_uri,
      absl::string_view aggregation_resource_name,
      absl::string_view byte_stream_uri_prefix,
      absl::string_view second_stage_aggregation_uri_prefix,
      bool set_relative_uri_prefix = false,
      std::optional<ConfidentialEncryptionConfig>
          confidential_encryption_config = std::nullopt);

  void ExpectSuccessfulSubmitAggregationResultRequest(
      absl::string_view expected_submit_aggregation_result_uri,
      bool confidential_aggregation = false,
      std::string resource_name = kResourceName);

  void ExpectSuccessfulAbortAggregationRequest(
      absl::string_view base_uri, bool confidential_aggregation = false);

  void ExpectSuccessfulByteStreamUploadRequest(
      absl::string_view byte_stream_upload_uri,
      absl::string_view checkpoint_str);

  void ExpectReportCompletedMatchesArgumentsForFakeWillowDecryptor(
      ComputationResults results, absl::Duration plan_duration,
      FederatedProtocol::WillowAggInfo willow_agg_info,
      absl::string_view public_key, absl::string_view checkpoint_string);

  StrictMock<MockHttpClient> mock_http_client_;
  StrictMock<MockSecAggRunnerFactory>* mock_secagg_runner_factory_ =
      new StrictMock<MockSecAggRunnerFactory>();
  StrictMock<MockSecAggEventPublisher> mock_secagg_event_publisher_;
  StrictMock<MockLogManager> mock_log_manager_;
  NiceMock<MockFlags> mock_flags_;
  NiceMock<MockFunction<bool()>> mock_should_abort_;
  StrictMock<cache::MockResourceCache> mock_resource_cache_;
  StrictMock<MockAttestationVerifier>* mock_attestation_verifier_ =
      new StrictMock<MockAttestationVerifier>();
  Clock* clock_ = Clock::RealClock();
  NiceMock<MockFunction<void(
      const ::fcp::client::FederatedProtocol::EligibilityEvalTask&)>>
      mock_eet_received_callback_;
  NiceMock<MockFunction<void(
      const ::fcp::client::FederatedProtocol::TaskAssignment&)>>
      mock_task_received_callback_;
  NiceMock<MockFunction<void(size_t)>> mock_multiple_tasks_received_callback_;

  // The class under test.
  std::unique_ptr<HttpFederatedProtocol> federated_protocol_;
};

}  // namespace fcp::client::http::internal

#endif  // FCP_CLIENT_HTTP_HTTP_FEDERATED_PROTOCOL_TEST_UTILS_H_
