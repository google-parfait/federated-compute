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
#include "fcp/client/http/http_federated_protocol.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/longrunning/operations.pb.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/duration.pb.h"
#include "google/rpc/code.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/platform.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/federated_protocol_util.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/http_client_util.h"
#include "fcp/client/http/test_helpers.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/federatedcompute/aggregations.pb.h"
#include "fcp/protos/federatedcompute/common.pb.h"
#include "fcp/protos/federatedcompute/eligibility_eval_tasks.pb.h"
#include "fcp/protos/federatedcompute/task_assignments.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/testing/testing.h"

namespace fcp::client::http {
namespace {

using ::fcp::EqualsProto;
using ::fcp::IsCode;
using ::google::internal::federatedcompute::v1::ByteStreamResource;
using ::google::internal::federatedcompute::v1::ClientStats;
using ::google::internal::federatedcompute::v1::EligibilityEvalTask;
using ::google::internal::federatedcompute::v1::EligibilityEvalTaskRequest;
using ::google::internal::federatedcompute::v1::EligibilityEvalTaskResponse;
using ::google::internal::federatedcompute::v1::ForwardingInfo;
using ::google::internal::federatedcompute::v1::ReportTaskResultRequest;
using ::google::internal::federatedcompute::v1::ReportTaskResultResponse;
using ::google::internal::federatedcompute::v1::Resource;
using ::google::internal::federatedcompute::v1::RetryWindow;
using ::google::internal::federatedcompute::v1::
    StartAggregationDataUploadRequest;
using ::google::internal::federatedcompute::v1::
    StartAggregationDataUploadResponse;
using ::google::internal::federatedcompute::v1::StartTaskAssignmentRequest;
using ::google::internal::federatedcompute::v1::StartTaskAssignmentResponse;
using ::google::internal::federatedcompute::v1::SubmitAggregationResultRequest;
using ::google::internal::federatedcompute::v1::TaskAssignment;
using ::google::internal::federatedml::v2::TaskEligibilityInfo;
using ::google::internal::federatedml::v2::TaskWeight;
using ::google::longrunning::GetOperationRequest;
using ::google::longrunning::Operation;
using ::google::protobuf::Any;
using ::google::protobuf::Message;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ContainerEq;
using ::testing::DescribeMatcher;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Eq;
using ::testing::ExplainMatchResult;
using ::testing::Field;
using ::testing::FieldsAre;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Lt;
using ::testing::MockFunction;
using ::testing::NiceMock;
using ::testing::Not;
using ::testing::Return;
using ::testing::StrEq;
using ::testing::StrictMock;
using ::testing::UnorderedElementsAre;
using ::testing::VariantWith;

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
constexpr char kClientToken[] = "CLIENT_TOKEN";
constexpr char kResourceName[] = "CHECKPOINT_RESOURCE";

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

void VerifyInMemoryHttpResponse(const InMemoryHttpResponse& response, int code,
                                absl::string_view content_encoding,
                                absl::string_view body) {
  EXPECT_EQ(response.code, code);
  EXPECT_EQ(response.content_encoding, content_encoding);
  EXPECT_EQ(response.body, body);
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

EligibilityEvalTaskRequest GetExpectedEligibilityEvalTaskRequest() {
  EligibilityEvalTaskRequest request;
  // Note: we don't expect population_name to be set, since it should be set in
  // the URI instead.
  request.mutable_client_version()->set_version_code(kClientVersion);

  return request;
}

EligibilityEvalTaskResponse GetFakeEnabledEligibilityEvalTaskResponse(
    const Resource& plan, const Resource& checkpoint,
    const std::string& execution_id,
    const RetryWindow& accepted_retry_window = GetAcceptedRetryWindow(),
    const RetryWindow& rejected_retry_window = GetRejectedRetryWindow()) {
  EligibilityEvalTaskResponse response;
  response.set_session_id(kEligibilityEvalSessionId);
  EligibilityEvalTask* eval_task = response.mutable_eligibility_eval_task();
  *eval_task->mutable_plan() = plan;
  *eval_task->mutable_init_checkpoint() = checkpoint;
  eval_task->set_execution_id(execution_id);
  ForwardingInfo* forwarding_info =
      response.mutable_task_assignment_forwarding_info();
  forwarding_info->set_target_uri_prefix(kTaskAssignmentTargetUri);
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
    const std::optional<TaskEligibilityInfo>& task_eligibility_info) {
  // Note: we don't expect population_name or session_id to be set, since they
  // should be set in the URI instead.
  StartTaskAssignmentRequest request;
  request.mutable_client_version()->set_version_code(kClientVersion);
  if (task_eligibility_info.has_value()) {
    *request.mutable_task_eligibility_info() = *task_eligibility_info;
  }
  return request;
}

Operation CreatePendingOperation(const std::string operation_name) {
  Operation operation;
  operation.set_done(false);
  operation.set_name(operation_name);
  return operation;
}

// Creates a 'done' `Operation`, with the given already-packed-into-`Any`
// result.
Operation CreateDoneOperation(const Any& packed_inner_result) {
  Operation operation;
  operation.set_done(true);
  *operation.mutable_response() = packed_inner_result;
  return operation;
}

// Creates a 'done' `Operation`, packing the given message into an `Any`.
Operation CreateDoneOperation(const Message& inner_result) {
  Operation operation;
  operation.set_done(true);
  operation.mutable_response()->PackFrom(inner_result);
  return operation;
}

Operation CreateErrorOperation(const absl::StatusCode error_code,
                               const std::string error_message) {
  Operation operation;
  operation.set_done(true);
  operation.mutable_error()->set_code(static_cast<int>(error_code));
  operation.mutable_error()->set_message(error_message);
  return operation;
}

StartTaskAssignmentResponse GetFakeRejectedTaskAssignmentResponse() {
  StartTaskAssignmentResponse response;
  response.mutable_rejection_info();
  return response;
}

StartTaskAssignmentResponse GetFakeTaskAssignmentResponse(
    const Resource& plan, const Resource& checkpoint,
    const std::string& aggregation_session_id) {
  StartTaskAssignmentResponse response;
  TaskAssignment* task_assignment = response.mutable_task_assignment();
  ForwardingInfo* forwarding_info =
      task_assignment->mutable_aggregation_data_forwarding_info();
  forwarding_info->set_target_uri_prefix(kAggregationTargetUri);
  task_assignment->set_session_id(kClientSessionId);
  task_assignment->set_aggregation_id(aggregation_session_id);
  task_assignment->set_client_token(kClientToken);
  *task_assignment->mutable_plan() = plan;
  *task_assignment->mutable_init_checkpoint() = checkpoint;
  return response;
}

ReportTaskResultRequest GetExpectedReportTaskResultRequest(
    absl::string_view aggregation_id, ::google::rpc::Code code,
    absl::Duration train_duration) {
  ReportTaskResultRequest request;
  request.set_aggregation_id(std::string(aggregation_id));
  request.set_computation_status_code(code);
  ClientStats client_stats;
  *client_stats.mutable_computation_execution_duration() =
      ConvertAbslToProtoDuration(train_duration);
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
  return response;
}

FakeHttpResponse CreateEmptySuccessHttpResponse() {
  return FakeHttpResponse(200, HeaderList(), "");
}

class HttpFederatedProtocolTest : public testing::Test {
 protected:
  void SetUp() override {
    EXPECT_CALL(mock_flags_,
                federated_training_transient_errors_retry_delay_secs)
        .WillRepeatedly(Return(kTransientErrorsRetryPeriodSecs));
    EXPECT_CALL(mock_flags_,
                federated_training_transient_errors_retry_delay_jitter_percent)
        .WillRepeatedly(Return(kTransientErrorsRetryDelayJitterPercent));
    EXPECT_CALL(mock_flags_,
                federated_training_permanent_errors_retry_delay_secs)
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
        .WillRepeatedly(Return(10));

    // We only initialize federated_protocol_ in this SetUp method, rather than
    // in the test's constructor, to ensure that we can set mock flag values
    // before the HttpFederatedProtocol constructor is called. Using
    // std::unique_ptr conveniently allows us to assign the field a new value
    // after construction (which we could not do if the field's type was
    // HttpFederatedProtocol, since it doesn't have copy or move constructors).
    federated_protocol_ = std::make_unique<HttpFederatedProtocol>(
        &mock_log_manager_, &mock_flags_, &mock_http_client_, kEntryPointUri,
        kApiKey, kPopulationName, kRetryToken, kClientVersion,
        kAttestationMeasurement, mock_should_abort_.AsStdFunction(),
        absl::BitGen(),
        InterruptibleRunner::TimingConfig{
            .polling_period = absl::ZeroDuration(),
            .graceful_shutdown_period = absl::InfiniteDuration(),
            .extended_shutdown_period = absl::InfiniteDuration()});
  }

  void TearDown() override {
    // Regardless of the outcome of the test (or the protocol interaction being
    // tested), network usage must always be reflected in the network stats
    // methods. We only check the chunking_layer_bytes_downloaded/upload
    // methods, since the legacy bytes_downloaded/uploaded methods will be
    // removed in the future.
    EXPECT_THAT(federated_protocol_->chunking_layer_bytes_received(),
                mock_http_client_.TotalReceivedBytes());
    EXPECT_THAT(federated_protocol_->chunking_layer_bytes_sent(),
                mock_http_client_.TotalSentBytes());
  }

  // This function runs a successful EligibilityEvalCheckin() that results in an
  // eligibility eval payload being returned by the server (if
  // `eligibility_eval_enabled` is true), or results in a 'no eligibility eval
  // configured' response (if `eligibility_eval_enabled` is false). This is a
  // utility function used by Checkin*() tests that depend on a prior,
  // successful execution of EligibilityEvalCheckin(). It returns a
  // absl::Status, which the caller should verify is OK using ASSERT_OK.
  absl::Status RunSuccessfulEligibilityEvalCheckin(
      bool eligibility_eval_enabled = true) {
    EligibilityEvalTaskResponse eval_task_response;
    if (eligibility_eval_enabled) {
      // We return a fake response which returns the plan/initial checkpoint
      // data inline, to keep things simple.
      std::string expected_plan = kPlan;
      Resource plan_resource;
      plan_resource.set_data(kPlan);
      std::string expected_checkpoint = kInitCheckpoint;
      Resource checkpoint_resource;
      checkpoint_resource.set_data(expected_checkpoint);
      eval_task_response = GetFakeEnabledEligibilityEvalTaskResponse(
          plan_resource, checkpoint_resource, kEligibilityEvalExecutionId);
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
                        EqualsProto(GetExpectedEligibilityEvalTaskRequest())))))
        .WillOnce(Return(FakeHttpResponse(
            200, HeaderList(), eval_task_response.SerializeAsString())));
    return federated_protocol_->EligibilityEvalCheckin().status();
  }

  // This function runs a successful Checkin() that results in a
  // task assignment payload being returned by the server. This is a
  // utility function used by Report*() tests that depend on a prior,
  // successful execution of Checkin(). It returns a
  // absl::Status, which the caller should verify is OK using ASSERT_OK.
  absl::Status RunSuccessfulCheckin() {
    // We return a fake response which returns the plan/initial checkpoint
    // data inline, to keep things simple.
    std::string expected_plan = kPlan;
    std::string plan_uri = "https://fake.uri/plan";
    Resource plan_resource;
    plan_resource.set_uri(plan_uri);
    std::string expected_checkpoint = kInitCheckpoint;
    Resource checkpoint_resource;
    checkpoint_resource.set_data(expected_checkpoint);
    std::string expected_aggregation_session_id = kAggregationSessionId;
    StartTaskAssignmentResponse task_assignment_response =
        GetFakeTaskAssignmentResponse(plan_resource, checkpoint_resource,
                                      expected_aggregation_session_id);

    std::string request_uri =
        "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
        "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto";
    TaskEligibilityInfo expected_eligibility_info =
        GetFakeTaskEligibilityInfo();
    EXPECT_CALL(mock_http_client_,
                PerformSingleRequest(SimpleHttpRequestMatcher(
                    request_uri, HttpRequest::Method::kPost, _,
                    StartTaskAssignmentRequestMatcher(
                        EqualsProto(GetExpectedStartTaskAssignmentRequest(
                            expected_eligibility_info))))))
        .WillOnce(Return(
            FakeHttpResponse(200, HeaderList(),
                             CreateDoneOperation(task_assignment_response)
                                 .SerializeAsString())));

    EXPECT_CALL(mock_http_client_,
                PerformSingleRequest(SimpleHttpRequestMatcher(
                    plan_uri, HttpRequest::Method::kGet, _, "")))
        .WillOnce(Return(FakeHttpResponse(200, HeaderList(), expected_plan)));

    return federated_protocol_->Checkin(expected_eligibility_info).status();
  }

  void ExpectSuccessfulReportTaskResultRequest(
      absl::string_view expected_report_result_uri,
      absl::string_view aggregation_session_id, absl::Duration plan_duration) {
    ReportTaskResultResponse report_task_result_response;
    EXPECT_CALL(mock_http_client_,
                PerformSingleRequest(SimpleHttpRequestMatcher(
                    std::string(expected_report_result_uri),
                    HttpRequest::Method::kPost, _,
                    ReportTaskResultRequestMatcher(
                        EqualsProto(GetExpectedReportTaskResultRequest(
                            aggregation_session_id, google::rpc::Code::OK,
                            plan_duration))))))
        .WillOnce(Return(CreateEmptySuccessHttpResponse()));
  }

  void ExpectSuccessfulStartAggregationDataUploadRequest(
      absl::string_view expected_start_data_upload_uri,
      absl::string_view aggregation_resource_name,
      absl::string_view byte_stream_uri_prefix,
      absl::string_view second_stage_aggregation_uri_prefix) {
    Operation pending_operation_response =
        CreatePendingOperation("operations/foo#bar");
    EXPECT_CALL(mock_http_client_,
                PerformSingleRequest(SimpleHttpRequestMatcher(
                    std::string(expected_start_data_upload_uri),
                    HttpRequest::Method::kPost, _,
                    StartAggregationDataUploadRequest().SerializeAsString())))
        .WillOnce(Return(
            FakeHttpResponse(200, HeaderList(),
                             pending_operation_response.SerializeAsString())));
    EXPECT_CALL(
        mock_http_client_,
        PerformSingleRequest(SimpleHttpRequestMatcher(
            // Note that the '#' character is encoded as "%23".
            "https://aggregation.uri/v1/operations/foo%23bar",
            HttpRequest::Method::kGet, _,
            GetOperationRequestMatcher(EqualsProto(GetOperationRequest())))))
        .WillOnce(Return(FakeHttpResponse(
            200, HeaderList(),
            CreateDoneOperation(GetFakeStartAggregationDataUploadResponse(
                                    aggregation_resource_name,
                                    byte_stream_uri_prefix,
                                    second_stage_aggregation_uri_prefix))
                .SerializeAsString())));
  }

  void ExpectSuccessfulByteStreamUploadRequest(
      absl::string_view byte_stream_upload_uri,
      absl::string_view checkpoint_str) {
    EXPECT_CALL(
        mock_http_client_,
        PerformSingleRequest(SimpleHttpRequestMatcher(
            std::string(byte_stream_upload_uri), HttpRequest::Method::kPost, _,
            std::string(checkpoint_str))))
        .WillOnce(Return(CreateEmptySuccessHttpResponse()));
  }

  void ExpectSuccessfulSubmitAggregationResultRequest(
      absl::string_view expected_submit_aggregation_result_uri) {
    SubmitAggregationResultRequest submit_aggregation_result_request;
    submit_aggregation_result_request.set_resource_name(kResourceName);
    EXPECT_CALL(mock_http_client_,
                PerformSingleRequest(SimpleHttpRequestMatcher(
                    std::string(expected_submit_aggregation_result_uri),
                    HttpRequest::Method::kPost, _,
                    submit_aggregation_result_request.SerializeAsString())))
        .WillOnce(Return(CreateEmptySuccessHttpResponse()));
  }

  void ExpectSuccessfulAbortAggregationRequest(absl::string_view base_uri) {
    EXPECT_CALL(mock_http_client_,
                PerformSingleRequest(SimpleHttpRequestMatcher(
                    absl::StrCat(base_uri, "/v1/aggregations/",
                                 "AGGREGATION_SESSION_ID/clients/"
                                 "CLIENT_TOKEN:abort?%24alt=proto"),
                    HttpRequest::Method::kPost, _, _)))
        .WillOnce(Return(CreateEmptySuccessHttpResponse()));
  }

  StrictMock<MockHttpClient> mock_http_client_;

  NiceMock<MockLogManager> mock_log_manager_;
  NiceMock<MockFlags> mock_flags_;
  NiceMock<MockFunction<bool()>> mock_should_abort_;

  // The class under test.
  std::unique_ptr<HttpFederatedProtocol> federated_protocol_;
};

using HttpFederatedProtocolDeathTest = HttpFederatedProtocolTest;

TEST_F(HttpFederatedProtocolTest,
       TestTransientErrorRetryWindowDifferentAcrossDifferentInstances) {
  const ::google::internal::federatedml::v2::RetryWindow& retry_window1 =
      federated_protocol_->GetLatestRetryWindow();
  ExpectTransientErrorRetryWindow(retry_window1);
  federated_protocol_.reset(nullptr);

  // Create a new HttpFederatedProtocol instance. It should not produce the same
  // retry window value as the one we just got. This is a simple correctness
  // check to ensure that the value is at least randomly generated (and that we
  // don't accidentally use the random number generator incorrectly).
  federated_protocol_ = std::make_unique<HttpFederatedProtocol>(
      &mock_log_manager_, &mock_flags_, &mock_http_client_, kEntryPointUri,
      kApiKey, kPopulationName, kRetryToken, kClientVersion,
      kAttestationMeasurement, mock_should_abort_.AsStdFunction(),
      absl::BitGen(),
      InterruptibleRunner::TimingConfig{
          .polling_period = absl::ZeroDuration(),
          .graceful_shutdown_period = absl::InfiniteDuration(),
          .extended_shutdown_period = absl::InfiniteDuration()});

  const ::google::internal::federatedml::v2::RetryWindow& retry_window2 =
      federated_protocol_->GetLatestRetryWindow();
  ExpectTransientErrorRetryWindow(retry_window2);

  EXPECT_THAT(retry_window1, Not(EqualsProto(retry_window2)));
}

TEST_F(HttpFederatedProtocolTest,
       TestEligibilityEvalCheckinRequestFailsTransientError) {
  // Make the HTTP client return a 503 Service Unavailable error when the
  // EligibilityEvalCheckin(...) code issues the control protocol's HTTP
  // request. This should result in the error being returned as the result.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(503, HeaderList(), "")));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(UNAVAILABLE));
  EXPECT_THAT(eligibility_checkin_result.status().message(),
              HasSubstr("protocol request failed"));
  // The original 503 HTTP response code should be included in the message as
  // well.
  EXPECT_THAT(eligibility_checkin_result.status().message(), HasSubstr("503"));
  // No RetryWindows were received from the server, so we expect to get a
  // RetryWindow generated based on the transient errors retry delay flag.
  ExpectTransientErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_F(HttpFederatedProtocolTest,
       TestEligibilityEvalCheckinRequestFailsPermanentError) {
  // Make the HTTP client return a 404 Not Found error when the
  // EligibilityEvalCheckin(...) code issues the control protocol's HTTP
  // request. This should result in the error being returned as the result.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(404, HeaderList(), "")));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(eligibility_checkin_result.status().message(),
              HasSubstr("protocol request failed"));
  // The original 404 HTTP response code should be included in the message as
  // well.
  EXPECT_THAT(eligibility_checkin_result.status().message(), HasSubstr("404"));
  // No RetryWindows were received from the server, so we expect to get a
  // RetryWindow generated based on the *permanent* errors retry delay flag,
  // since NOT_FOUND is marked as a permanent error in the flags.
  ExpectPermanentErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Tests the case where we get interrupted while waiting for a response to the
// protocol request in EligibilityEvalCheckin.
TEST_F(HttpFederatedProtocolTest,
       TestEligibilityEvalCheckinRequestInterrupted) {
  absl::Notification request_issued;
  absl::Notification request_cancelled;

  // Make HttpClient::PerformRequests() block until the counter is decremented.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce([&request_issued, &request_cancelled](
                    MockableHttpClient::SimpleHttpRequest ignored) {
        request_issued.Notify();
        request_cancelled.WaitForNotification();
        return FakeHttpResponse(503, HeaderList(), "");
      });

  // Make should_abort return false until we know that the request was issued
  // (i.e. once InterruptibleRunner has actually started running the code it
  // was given), and then make it return true, triggering an abort sequence and
  // unblocking the PerformRequests()() call we caused to block above.
  EXPECT_CALL(mock_should_abort_, Call()).WillRepeatedly([&request_issued] {
    return request_issued.HasBeenNotified();
  });

  // When the HttpClient receives a HttpRequestHandle::Cancel call, we let the
  // request complete.
  mock_http_client_.SetCancellationListener(
      [&request_cancelled]() { request_cancelled.Notify(); });

  EXPECT_CALL(mock_log_manager_,
              LogDiag(ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(CANCELLED));
  // No RetryWindows were received from the server, so we expect to get a
  // RetryWindow generated based on the transient errors retry delay flag.
  ExpectTransientErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_F(HttpFederatedProtocolTest, TestEligibilityEvalCheckinRejection) {
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _,
                  EligibilityEvalTaskRequestMatcher(
                      EqualsProto(GetExpectedEligibilityEvalTaskRequest())))))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          GetFakeRejectedEligibilityEvalTaskResponse().SerializeAsString())));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(*eligibility_checkin_result,
              VariantWith<FederatedProtocol::Rejection>(_));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_F(HttpFederatedProtocolTest, TestEligibilityEvalCheckinDisabled) {
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _,
                  EligibilityEvalTaskRequestMatcher(
                      EqualsProto(GetExpectedEligibilityEvalTaskRequest())))))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          GetFakeDisabledEligibilityEvalTaskResponse().SerializeAsString())));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(*eligibility_checkin_result,
              VariantWith<FederatedProtocol::EligibilityEvalDisabled>(_));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_F(HttpFederatedProtocolTest, TestEligibilityEvalCheckinEnabled) {
  // We return a fake response which requires fetching the plan via HTTP, but
  // which has the checkpoint data available inline.
  std::string expected_plan = kPlan;
  std::string plan_uri = "https://fake.uri/plan";
  Resource plan_resource;
  plan_resource.set_uri(plan_uri);
  std::string expected_checkpoint = kInitCheckpoint;
  Resource checkpoint_resource;
  checkpoint_resource.set_data(expected_checkpoint);
  std::string expected_execution_id = kEligibilityEvalExecutionId;
  EligibilityEvalTaskResponse eval_task_response =
      GetFakeEnabledEligibilityEvalTaskResponse(
          plan_resource, checkpoint_resource, expected_execution_id);
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _,
                  EligibilityEvalTaskRequestMatcher(
                      EqualsProto(GetExpectedEligibilityEvalTaskRequest())))))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), eval_task_response.SerializeAsString())));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), expected_plan)));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(
      *eligibility_checkin_result,
      VariantWith<FederatedProtocol::EligibilityEvalTask>(FieldsAre(
          AllOf(Field(&FederatedProtocol::PlanAndCheckpointPayloads::plan,
                      absl::Cord(expected_plan)),
                Field(&FederatedProtocol::PlanAndCheckpointPayloads::checkpoint,
                      absl::Cord(expected_checkpoint))),
          expected_execution_id)));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Ensures that if the plan resource fails to be downloaded, the error is
// correctly returned from the EligibilityEvalCheckin(...) method.
TEST_F(HttpFederatedProtocolTest,
       TestEligibilityEvalCheckinEnabledPlanDataFetchFailed) {
  std::string plan_uri = "https://fake.uri/plan";
  Resource plan_resource;
  plan_resource.set_uri(plan_uri);
  std::string checkpoint_uri = "https://fake.uri/checkpoint";
  Resource checkpoint_resource;
  checkpoint_resource.set_uri(checkpoint_uri);
  std::string expected_execution_id = kEligibilityEvalExecutionId;
  EligibilityEvalTaskResponse eval_task_response =
      GetFakeEnabledEligibilityEvalTaskResponse(
          plan_resource, checkpoint_resource, expected_execution_id);
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), eval_task_response.SerializeAsString())));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  checkpoint_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "")));

  // Mock a failed plan fetch.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(404, HeaderList(), "")));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  // The 404 error for the resource request should be reflected in the return
  // value.
  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(eligibility_checkin_result.status().message(),
              HasSubstr("plan fetch failed"));
  // The original 404 HTTP response code should be included in the message as
  // well.
  EXPECT_THAT(eligibility_checkin_result.status().message(), HasSubstr("404"));
  // Since the error type is considered a permanent error, we should get a
  // permanent error retry window.
  ExpectPermanentErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Ensures that if the checkpoint resource fails to be downloaded, the error is
// correctly returned from the EligibilityEvalCheckin(...) method.
TEST_F(HttpFederatedProtocolTest,
       TestEligibilityEvalCheckinEnabledCheckpointDataFetchFailed) {
  std::string plan_uri = "https://fake.uri/plan";
  Resource plan_resource;
  plan_resource.set_uri(plan_uri);
  std::string checkpoint_uri = "https://fake.uri/checkpoint";
  Resource checkpoint_resource;
  checkpoint_resource.set_uri(checkpoint_uri);
  std::string expected_execution_id = kEligibilityEvalExecutionId;
  EligibilityEvalTaskResponse eval_task_response =
      GetFakeEnabledEligibilityEvalTaskResponse(
          plan_resource, checkpoint_resource, expected_execution_id);
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), eval_task_response.SerializeAsString())));

  // Mock a failed checkpoint fetch.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  checkpoint_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(503, HeaderList(), "")));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "")));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  // The 503 error for the resource request should be reflected in the return
  // value.
  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(UNAVAILABLE));
  EXPECT_THAT(eligibility_checkin_result.status().message(),
              HasSubstr("checkpoint fetch failed"));
  // The original 503 HTTP response code should be included in the message as
  // well.
  EXPECT_THAT(eligibility_checkin_result.status().message(), HasSubstr("503"));
  // RetryWindows were received from the server before the error was received,
  // and the error is considered 'transient', so we expect to get a rejected
  // RetryWindow.
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Tests that the protocol correctly sanitizes any invalid values it may have
// received from the server.
TEST_F(HttpFederatedProtocolTest,
       TestNegativeMinMaxRetryDelayValueSanitization) {
  RetryWindow retry_window;
  retry_window.mutable_delay_min()->set_seconds(-1);
  retry_window.mutable_delay_max()->set_seconds(-2);

  // The above retry window's negative min/max values should be clamped to 0.
  RetryWindow expected_retry_window;
  expected_retry_window.mutable_delay_min()->set_seconds(0);
  expected_retry_window.mutable_delay_max()->set_seconds(0);

  EligibilityEvalTaskResponse eval_task_response =
      GetFakeEnabledEligibilityEvalTaskResponse(Resource(), Resource(),
                                                kEligibilityEvalExecutionId,
                                                retry_window, retry_window);
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), eval_task_response.SerializeAsString())));

  ASSERT_OK(federated_protocol_->EligibilityEvalCheckin());

  const google::internal::federatedml::v2::RetryWindow& actual_retry_window =
      federated_protocol_->GetLatestRetryWindow();
  // The above retry window's invalid max value should be clamped to the min
  // value (minus some errors introduced by the inaccuracy of double
  // multiplication).
  EXPECT_THAT(actual_retry_window.delay_min().seconds() +
                  actual_retry_window.delay_min().nanos() / 1000000000.0,
              DoubleEq(0));
  EXPECT_THAT(actual_retry_window.delay_max().seconds() +
                  actual_retry_window.delay_max().nanos() / 1000000000.0,
              DoubleEq(0));
}

// Tests that the protocol correctly sanitizes any invalid values it may have
// received from the server.
TEST_F(HttpFederatedProtocolTest, TestInvalidMaxRetryDelayValueSanitization) {
  RetryWindow retry_window;
  retry_window.mutable_delay_min()->set_seconds(1234);
  retry_window.mutable_delay_max()->set_seconds(1233);  // less than delay_min

  EligibilityEvalTaskResponse eval_task_response =
      GetFakeEnabledEligibilityEvalTaskResponse(Resource(), Resource(),
                                                kEligibilityEvalExecutionId,
                                                retry_window, retry_window);
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), eval_task_response.SerializeAsString())));

  ASSERT_OK(federated_protocol_->EligibilityEvalCheckin());

  const google::internal::federatedml::v2::RetryWindow& actual_retry_window =
      federated_protocol_->GetLatestRetryWindow();
  // The above retry window's invalid max value should be clamped to the min
  // value (minus some errors introduced by the inaccuracy of double
  // multiplication). Note that DoubleEq enforces too precise of bounds, so we
  // use DoubleNear instead.
  EXPECT_THAT(actual_retry_window.delay_min().seconds() +
                  actual_retry_window.delay_min().nanos() / 1000000000.0,
              DoubleNear(1234.0, 0.01));
  EXPECT_THAT(actual_retry_window.delay_max().seconds() +
                  actual_retry_window.delay_max().nanos() / 1000000000.0,
              DoubleNear(1234.0, 0.01));
}

TEST_F(HttpFederatedProtocolDeathTest,
       TestCheckinAfterFailedEligibilityEvalCheckin) {
  // Make the HTTP client return a 503 Service Unavailable error when the
  // EligibilityEvalCheckin(...) code issues the protocol HTTP request.
  // This should result in the error being returned as the result.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(503, HeaderList(), "")));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(UNAVAILABLE));

  // A Checkin(...) request should now fail, because Checkin(...) should only
  // be a called after a successful EligibilityEvalCheckin(...) request.
  ASSERT_DEATH({ auto unused = federated_protocol_->Checkin(std::nullopt); },
               _);
}

TEST_F(HttpFederatedProtocolDeathTest,
       TestCheckinAfterEligibilityEvalCheckinRejection) {
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _,
                  EligibilityEvalTaskRequestMatcher(
                      EqualsProto(GetExpectedEligibilityEvalTaskRequest())))))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          GetFakeRejectedEligibilityEvalTaskResponse().SerializeAsString())));

  ASSERT_OK(federated_protocol_->EligibilityEvalCheckin());

  // A Checkin(...) request should now fail, because Checkin(...) should only
  // be a called after a successful EligibilityEvalCheckin(...) request, with a
  // non-rejection response.
  ASSERT_DEATH({ auto unused = federated_protocol_->Checkin(std::nullopt); },
               _);
}

TEST_F(HttpFederatedProtocolDeathTest,
       TestCheckinWithEligibilityInfoAfterEligibilityEvalCheckinDisabled) {
  ASSERT_OK(
      RunSuccessfulEligibilityEvalCheckin(/*eligibility_eval_enabled=*/false));

  // A Checkin(...) request with a TaskEligibilityInfo argument should now fail,
  // because such info should only be passed a successful
  // EligibilityEvalCheckin(...) request with an eligibility eval task in the
  // response.
  ASSERT_DEATH(
      { auto unused = federated_protocol_->Checkin(TaskEligibilityInfo()); },
      _);
}

TEST_F(HttpFederatedProtocolDeathTest, TestCheckinWithMissingEligibilityInfo) {
  ASSERT_OK(
      RunSuccessfulEligibilityEvalCheckin(/*eligibility_eval_enabled=*/true));

  // A Checkin(...) request with a missing TaskEligibilityInfo should now fail,
  // as the protocol requires us to provide one based on the plan includes in
  // the eligibility eval checkin response payload..
  ASSERT_DEATH({ auto unused = federated_protocol_->Checkin(std::nullopt); },
               _);
}

TEST_F(HttpFederatedProtocolDeathTest,
       TestCheckinAfterEligibilityEvalResourceDataFetchFailed) {
  Resource plan_resource;
  plan_resource.set_uri("https://fake.uri/plan");
  Resource checkpoint_resource;
  checkpoint_resource.set_uri("https://fake.uri/checkpoint");
  EligibilityEvalTaskResponse eval_task_response =
      GetFakeEnabledEligibilityEvalTaskResponse(
          plan_resource, checkpoint_resource, kEligibilityEvalExecutionId);
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/v1/eligibilityevaltasks/"
                  "TEST%2FPOPULATION:request?%24alt=proto",
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), eval_task_response.SerializeAsString())));

  // Mock a failed plan/resource fetch.
  EXPECT_CALL(mock_http_client_, PerformSingleRequest(SimpleHttpRequestMatcher(
                                     _, HttpRequest::Method::kGet, _, "")))
      .WillRepeatedly(Return(FakeHttpResponse(503, HeaderList(), "")));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  // A Checkin(...) request should now fail, because Checkin(...) should only
  // be a called after a successful EligibilityEvalCheckin(...) request, with a
  // non-rejection response.
  ASSERT_DEATH(
      { auto unused = federated_protocol_->Checkin(TaskEligibilityInfo()); },
      _);
}

// Ensures that if the HTTP layer returns an error code that maps to a transient
// error, it is handled correctly
TEST_F(HttpFederatedProtocolTest, TestCheckinFailsTransientError) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // Make the HTTP request return an 503 Service Unavailable error when the
  // Checkin(...) code tries to send its first request. This should result in
  // the error being returned as the result.
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
          "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(503, HeaderList(), "")));

  auto checkin_result =
      federated_protocol_->Checkin(GetFakeTaskEligibilityInfo());

  EXPECT_THAT(checkin_result.status(), IsCode(UNAVAILABLE));
  EXPECT_THAT(checkin_result.status().message(),
              HasSubstr("protocol request failed"));
  // The original 503 HTTP response code should be included in the message as
  // well.
  EXPECT_THAT(checkin_result.status().message(), HasSubstr("503"));
  // RetryWindows were already received from the server during the eligibility
  // eval checkin, so we expect to get a 'rejected' retry window.
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Ensures that if the HTTP layer returns an error code that maps to a permanent
// error, it is handled correctly.
TEST_F(HttpFederatedProtocolTest, TestCheckinFailsPermanentErrorFromHttp) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // Make the HTTP request return an 404 Not Found error when the Checkin(...)
  // code tries to send its first request. This should result in the error being
  // returned as the result.
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
          "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(404, HeaderList(), "")));

  auto checkin_result =
      federated_protocol_->Checkin(GetFakeTaskEligibilityInfo());

  EXPECT_THAT(checkin_result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(checkin_result.status().message(),
              HasSubstr("protocol request failed"));
  // The original 503 HTTP response code should be included in the message as
  // well.
  EXPECT_THAT(checkin_result.status().message(), HasSubstr("404"));
  // Even though RetryWindows were already received from the server during the
  // eligibility eval checkin, we expect a RetryWindow generated based on the
  // *permanent* errors retry delay flag, since NOT_FOUND is marked as a
  // permanent error in the flags, and permanent errors should always result in
  // permanent error windows (regardless of whether retry windows were already
  // received).
  ExpectPermanentErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Ensures that if the HTTP layer returns a successful response, but it contains
// an Operation proto with a permanent error, that it is handled correctly.
TEST_F(HttpFederatedProtocolTest, TestCheckinFailsPermanentErrorFromOperation) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // Make the HTTP request return successfully, but make it contain an Operation
  // proto that itself contains a permanent error. This should result in the
  // error being returned as the result.
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
          "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          CreateErrorOperation(absl::StatusCode::kNotFound, "foo")
              .SerializeAsString())));

  auto checkin_result =
      federated_protocol_->Checkin(GetFakeTaskEligibilityInfo());

  EXPECT_THAT(checkin_result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(checkin_result.status().message(),
              HasSubstr("Operation contained error"));
  // The original error message should be included in the message as well.
  EXPECT_THAT(checkin_result.status().message(), HasSubstr("foo"));
  // Even though RetryWindows were already received from the server during the
  // eligibility eval checkin, we expect a RetryWindow generated based on the
  // *permanent* errors retry delay flag, since NOT_FOUND is marked as a
  // permanent error in the flags, and permanent errors should always result in
  // permanent error windows (regardless of whether retry windows were already
  // received).
  ExpectPermanentErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Tests the case where we get interrupted while waiting for a response to the
// protocol request in Checkin.
TEST_F(HttpFederatedProtocolTest, TestCheckinInterrupted) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  absl::Notification request_issued;
  absl::Notification request_cancelled;

  // Make HttpClient::PerformRequests() block until the counter is decremented.
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
          "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce([&request_issued, &request_cancelled](
                    MockableHttpClient::SimpleHttpRequest ignored) {
        request_issued.Notify();
        request_cancelled.WaitForNotification();
        return FakeHttpResponse(503, HeaderList(), "");
      });

  // Make should_abort return false until we know that the request was issued
  // (i.e. once InterruptibleRunner has actually started running the code it
  // was given), and then make it return true, triggering an abort sequence and
  // unblocking the PerformRequests()() call we caused to block above.
  EXPECT_CALL(mock_should_abort_, Call()).WillRepeatedly([&request_issued] {
    return request_issued.HasBeenNotified();
  });

  // When the HttpClient receives a HttpRequestHandle::Cancel call, we let the
  // request complete.
  mock_http_client_.SetCancellationListener(
      [&request_cancelled]() { request_cancelled.Notify(); });

  EXPECT_CALL(mock_log_manager_,
              LogDiag(ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP));

  auto checkin_result =
      federated_protocol_->Checkin(GetFakeTaskEligibilityInfo());
  EXPECT_THAT(checkin_result.status(), IsCode(CANCELLED));
  // RetryWindows were already received from the server during the eligibility
  // eval checkin, so we expect to get a 'rejected' retry window.
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Tests whether 'rejection' responses to the main Checkin(...) request are
// handled correctly.
TEST_F(HttpFederatedProtocolTest, TestCheckinRejectionWithTaskEligibilityInfo) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  TaskEligibilityInfo expected_eligibility_info = GetFakeTaskEligibilityInfo();
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
          "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto",
          HttpRequest::Method::kPost, _,
          StartTaskAssignmentRequestMatcher(
              EqualsProto(GetExpectedStartTaskAssignmentRequest(
                  expected_eligibility_info))))))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          CreateDoneOperation(GetFakeRejectedTaskAssignmentResponse())
              .SerializeAsString())));

  // Issue the regular checkin.
  auto checkin_result = federated_protocol_->Checkin(expected_eligibility_info);

  ASSERT_OK(checkin_result.status());
  EXPECT_THAT(*checkin_result, VariantWith<FederatedProtocol::Rejection>(_));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Tests whether we can issue a Checkin() request correctly without passing a
// TaskEligibilityInfo, in the case that the eligibility eval checkin didn't
// return any eligibility eval task to run.
TEST_F(HttpFederatedProtocolTest,
       TestCheckinRejectionWithoutTaskEligibilityInfo) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(
      RunSuccessfulEligibilityEvalCheckin(/*eligibility_eval_enabled=*/false));

  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
          "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto",
          HttpRequest::Method::kPost, _,
          StartTaskAssignmentRequestMatcher(EqualsProto(
              GetExpectedStartTaskAssignmentRequest(std::nullopt))))))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          CreateDoneOperation(GetFakeRejectedTaskAssignmentResponse())
              .SerializeAsString())));

  // Issue the regular checkin, without a TaskEligibilityInfo (since we didn't
  // receive an eligibility eval task to run during eligibility eval checkin).
  auto checkin_result = federated_protocol_->Checkin(std::nullopt);

  ASSERT_OK(checkin_result.status());
  EXPECT_THAT(*checkin_result, VariantWith<FederatedProtocol::Rejection>(_));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Tests whether a successful task assignment response is handled correctly.
TEST_F(HttpFederatedProtocolTest, TestCheckinTaskAssigned) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  TaskEligibilityInfo expected_eligibility_info = GetFakeTaskEligibilityInfo();
  // We return a fake response which requires fetching the plan via HTTP, but
  // which has the checkpoint data available inline.
  std::string expected_plan = kPlan;
  std::string plan_uri = "https://fake.uri/plan";
  Resource plan_resource;
  plan_resource.set_uri(plan_uri);
  std::string expected_checkpoint = kInitCheckpoint;
  Resource checkpoint_resource;
  checkpoint_resource.set_data(expected_checkpoint);
  std::string expected_aggregation_session_id = kAggregationSessionId;
  // Note that in this particular test we check that the CheckinRequest is as
  // expected (in all prior tests we just use the '_' matcher, because the
  // request isn't really relevant to the test).
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
          "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto",
          HttpRequest::Method::kPost, _,
          StartTaskAssignmentRequestMatcher(
              EqualsProto(GetExpectedStartTaskAssignmentRequest(
                  expected_eligibility_info))))))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          CreateDoneOperation(
              GetFakeTaskAssignmentResponse(plan_resource, checkpoint_resource,
                                            expected_aggregation_session_id))
              .SerializeAsString())));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), expected_plan)));

  // Issue the regular checkin.
  auto checkin_result = federated_protocol_->Checkin(expected_eligibility_info);

  ASSERT_OK(checkin_result.status());
  EXPECT_THAT(
      *checkin_result,
      VariantWith<FederatedProtocol::TaskAssignment>(FieldsAre(
          FieldsAre(absl::Cord(expected_plan), absl::Cord(expected_checkpoint)),
          expected_aggregation_session_id, Eq(std::nullopt))));
  // The Checkin call is expected to return the accepted retry window from the
  // response to the first eligibility eval request.
  ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Ensures that polling the Operation returned by a StartTaskAssignmentRequest
// works as expected. This serves mostly as a high-level check. Further
// polling-specific behavior is tested in more detail in
// ProtocolRequestHelperTest.
TEST_F(HttpFederatedProtocolTest,
       TestCheckinTaskAssignedAfterOperationPolling) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // Make the initial StartTaskAssignmentRequest return a pending Operation
  // result. Note that we use a '#' character in the operation name to allow us
  // to verify that it is properly URL-encoded.
  Operation pending_operation_response =
      CreatePendingOperation("operations/foo#bar");
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
          "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), pending_operation_response.SerializeAsString())));

  // Then, after letting the operation get polled twice more, eventually return
  // a fake response.
  std::string expected_plan = kPlan;
  Resource plan_resource;
  plan_resource.set_data(expected_plan);
  std::string expected_checkpoint = kInitCheckpoint;
  Resource checkpoint_resource;
  checkpoint_resource.set_data(expected_checkpoint);
  std::string expected_aggregation_session_id = kAggregationSessionId;
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          // Note that the '#' character is encoded as "%23".
          "https://taskassignment.uri/v1/operations/foo%23bar",
          HttpRequest::Method::kGet, _,
          GetOperationRequestMatcher(EqualsProto(GetOperationRequest())))))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), pending_operation_response.SerializeAsString())))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), pending_operation_response.SerializeAsString())))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          CreateDoneOperation(
              GetFakeTaskAssignmentResponse(plan_resource, checkpoint_resource,
                                            expected_aggregation_session_id))
              .SerializeAsString())));

  // Issue the regular checkin.
  auto checkin_result =
      federated_protocol_->Checkin(GetFakeTaskEligibilityInfo());

  ASSERT_OK(checkin_result.status());
  EXPECT_THAT(
      *checkin_result,
      VariantWith<FederatedProtocol::TaskAssignment>(FieldsAre(
          FieldsAre(absl::Cord(expected_plan), absl::Cord(expected_checkpoint)),
          expected_aggregation_session_id, Eq(std::nullopt))));
  // The Checkin call is expected to return the accepted retry window from the
  // response to the first eligibility eval request.
  ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Ensures that if the plan resource fails to be downloaded, the error is
// correctly returned from the Checkin(...) method.
TEST_F(HttpFederatedProtocolTest, TestCheckinTaskAssignedPlanDataFetchFailed) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  std::string plan_uri = "https://fake.uri/plan";
  Resource plan_resource;
  plan_resource.set_uri(plan_uri);
  std::string checkpoint_uri = "https://fake.uri/checkpoint";
  Resource checkpoint_resource;
  checkpoint_resource.set_uri(checkpoint_uri);
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
          "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          CreateDoneOperation(
              GetFakeTaskAssignmentResponse(plan_resource, checkpoint_resource,
                                            kAggregationSessionId))
              .SerializeAsString())));

  // Mock a failed plan fetch.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(404, HeaderList(), "")));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  checkpoint_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "")));

  // Issue the regular checkin.
  auto checkin_result =
      federated_protocol_->Checkin(GetFakeTaskEligibilityInfo());

  // The 404 error for the resource request should be reflected in the return
  // value.
  EXPECT_THAT(checkin_result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(checkin_result.status().message(),
              HasSubstr("plan fetch failed"));
  EXPECT_THAT(checkin_result.status().message(), HasSubstr("404"));
  // The Checkin call is expected to return the permanent error retry window,
  // since 404 maps to a permanent error.
  ExpectPermanentErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Ensures that if the checkpoint resource fails to be downloaded, the error is
// correctly returned from the Checkin(...) method.
TEST_F(HttpFederatedProtocolTest,
       TestCheckinTaskAssignedCheckpointDataFetchFailed) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  std::string plan_uri = "https://fake.uri/plan";
  Resource plan_resource;
  plan_resource.set_uri(plan_uri);
  std::string checkpoint_uri = "https://fake.uri/checkpoint";
  Resource checkpoint_resource;
  checkpoint_resource.set_uri(checkpoint_uri);
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
          "taskassignments/ELIGIBILITY%2FSESSION%23ID:start?%24alt=proto",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(),
          CreateDoneOperation(
              GetFakeTaskAssignmentResponse(plan_resource, checkpoint_resource,
                                            kAggregationSessionId))
              .SerializeAsString())));

  // Mock a failed checkpoint fetch.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  checkpoint_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(503, HeaderList(), "")));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "")));

  // Issue the regular checkin.
  auto checkin_result =
      federated_protocol_->Checkin(GetFakeTaskEligibilityInfo());

  // The 503 error for the resource request should be reflected in the return
  // value.
  EXPECT_THAT(checkin_result.status(), IsCode(UNAVAILABLE));
  EXPECT_THAT(checkin_result.status().message(),
              HasSubstr("checkpoint fetch failed"));
  EXPECT_THAT(checkin_result.status().message(), HasSubstr("503"));
  // The Checkin call is expected to return the rejected retry window from the
  // response to the first eligibility eval request.
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_F(HttpFederatedProtocolTest, TestReportCompletedSuccess) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  // Issue a regular checkin
  ASSERT_OK(RunSuccessfulCheckin());

  // Create a fake checkpoint with 32 'X'.
  std::string checkpoint_str(32, 'X');
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", checkpoint_str);
  absl::Duration plan_duration = absl::Minutes(5);

  ExpectSuccessfulReportTaskResultRequest(
      "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
      "taskassignments/CLIENT_SESSION_ID:reportresult?%24alt=proto",
      kAggregationSessionId, plan_duration);
  ExpectSuccessfulStartAggregationDataUploadRequest(
      "https://aggregation.uri/v1/aggregations/AGGREGATION_SESSION_ID/"
      "clients/CLIENT_TOKEN:startdataupload?%24alt=proto",
      kResourceName, kByteStreamTargetUri, kSecondStageAggregationTargetUri);
  ExpectSuccessfulByteStreamUploadRequest(
      "https://bytestream.uri/upload/v1/media/"
      "CHECKPOINT_RESOURCE?upload_protocol=raw",
      checkpoint_str);
  ExpectSuccessfulSubmitAggregationResultRequest(
      "https://aggregation.second.uri/v1/aggregations/"
      "AGGREGATION_SESSION_ID/clients/CLIENT_TOKEN:submit?%24alt=proto");

  EXPECT_OK(
      federated_protocol_->ReportCompleted(std::move(results), plan_duration));
}

TEST_F(HttpFederatedProtocolTest, TestReportCompletedReportTaskResultFailed) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  // Issue a regular checkin
  ASSERT_OK(RunSuccessfulCheckin());

  // Create a fake checkpoint with 32 'X'.
  std::string checkpoint_str(32, 'X');
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", checkpoint_str);
  absl::Duration plan_duration = absl::Minutes(5);

  // Mock a failed ReportTaskResult request.
  ReportTaskResultResponse report_task_result_response;
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
                  "taskassignments/CLIENT_SESSION_ID:reportresult?%24alt=proto",
                  HttpRequest::Method::kPost, _,
                  ReportTaskResultRequestMatcher(
                      EqualsProto(GetExpectedReportTaskResultRequest(
                          kAggregationSessionId, google::rpc::Code::OK,
                          plan_duration))))))
      .WillOnce(Return(FakeHttpResponse(503, HeaderList())));

  ExpectSuccessfulStartAggregationDataUploadRequest(
      "https://aggregation.uri/v1/aggregations/AGGREGATION_SESSION_ID/"
      "clients/CLIENT_TOKEN:startdataupload?%24alt=proto",
      kResourceName, kByteStreamTargetUri, kSecondStageAggregationTargetUri);
  ExpectSuccessfulByteStreamUploadRequest(
      "https://bytestream.uri/upload/v1/media/"
      "CHECKPOINT_RESOURCE?upload_protocol=raw",
      checkpoint_str);
  ExpectSuccessfulSubmitAggregationResultRequest(
      "https://aggregation.second.uri/v1/aggregations/"
      "AGGREGATION_SESSION_ID/clients/CLIENT_TOKEN:submit?%24alt=proto");

  // Despite the ReportTaskResult request failed, we still consider the overall
  // ReportCompleted succeeded because the rest of the steps succeeds, and the
  // ReportTaskResult is a just a metric reporting on a best effort basis.
  EXPECT_OK(
      federated_protocol_->ReportCompleted(std::move(results), plan_duration));
}

TEST_F(HttpFederatedProtocolTest,
       TestReportCompletedStartAggregationFailedImmediately) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  // Issue a regular checkin
  ASSERT_OK(RunSuccessfulCheckin());

  std::string checkpoint_str;
  const size_t kTFCheckpointSize = 32;
  checkpoint_str.resize(kTFCheckpointSize, 'X');
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", checkpoint_str);
  absl::Duration plan_duration = absl::Minutes(5);

  ExpectSuccessfulReportTaskResultRequest(
      "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
      "taskassignments/CLIENT_SESSION_ID:reportresult?%24alt=proto",
      kAggregationSessionId, plan_duration);
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://aggregation.uri/v1/aggregations/AGGREGATION_SESSION_ID/"
          "clients/CLIENT_TOKEN:startdataupload?%24alt=proto",
          HttpRequest::Method::kPost, _,
          StartAggregationDataUploadRequest().SerializeAsString())))
      .WillOnce(Return(FakeHttpResponse(503, HeaderList())));
  ExpectSuccessfulAbortAggregationRequest("https://aggregation.uri");
  absl::Status report_result =
      federated_protocol_->ReportCompleted(std::move(results), plan_duration);
  ASSERT_THAT(report_result, IsCode(absl::StatusCode::kUnavailable));
  EXPECT_THAT(report_result.message(),
              HasSubstr("StartAggregationDataUpload request failed"));
  EXPECT_THAT(report_result.message(), HasSubstr("503"));
}

TEST_F(HttpFederatedProtocolTest,
       TestReportCompletedStartAggregationFailedDuringPolling) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  // Issue a regular checkin
  ASSERT_OK(RunSuccessfulCheckin());

  std::string checkpoint_str;
  const size_t kTFCheckpointSize = 32;
  checkpoint_str.resize(kTFCheckpointSize, 'X');
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", checkpoint_str);
  absl::Duration plan_duration = absl::Minutes(5);

  ExpectSuccessfulReportTaskResultRequest(
      "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
      "taskassignments/CLIENT_SESSION_ID:reportresult?%24alt=proto",
      kAggregationSessionId, plan_duration);
  Operation pending_operation_response =
      CreatePendingOperation("operations/foo#bar");
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://aggregation.uri/v1/aggregations/AGGREGATION_SESSION_ID/"
          "clients/CLIENT_TOKEN:startdataupload?%24alt=proto",
          HttpRequest::Method::kPost, _,
          StartAggregationDataUploadRequest().SerializeAsString())))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), pending_operation_response.SerializeAsString())));
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          // Note that the '#' character is encoded as "%23".
          "https://aggregation.uri/v1/operations/foo%23bar",
          HttpRequest::Method::kGet, _,
          GetOperationRequestMatcher(EqualsProto(GetOperationRequest())))))
      .WillOnce(Return(FakeHttpResponse(401, HeaderList())));
  ExpectSuccessfulAbortAggregationRequest("https://aggregation.uri");
  absl::Status report_result =
      federated_protocol_->ReportCompleted(std::move(results), plan_duration);
  ASSERT_THAT(report_result, IsCode(absl::StatusCode::kUnauthenticated));
  EXPECT_THAT(report_result.message(),
              HasSubstr("StartAggregationDataUpload request failed"));
  EXPECT_THAT(report_result.message(), HasSubstr("401"));
}

TEST_F(HttpFederatedProtocolTest, TestReportCompletedUploadFailed) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  // Issue a regular checkin
  ASSERT_OK(RunSuccessfulCheckin());

  std::string checkpoint_str;
  const size_t kTFCheckpointSize = 32;
  checkpoint_str.resize(kTFCheckpointSize, 'X');
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", checkpoint_str);
  absl::Duration plan_duration = absl::Minutes(5);

  ExpectSuccessfulReportTaskResultRequest(
      "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
      "taskassignments/CLIENT_SESSION_ID:reportresult?%24alt=proto",
      kAggregationSessionId, plan_duration);
  ExpectSuccessfulStartAggregationDataUploadRequest(
      "https://aggregation.uri/v1/aggregations/AGGREGATION_SESSION_ID/"
      "clients/CLIENT_TOKEN:startdataupload?%24alt=proto",
      kResourceName, kByteStreamTargetUri, kSecondStageAggregationTargetUri);
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  StrEq("https://bytestream.uri/upload/v1/media/"
                        "CHECKPOINT_RESOURCE?upload_protocol=raw"),
                  HttpRequest::Method::kPost, _, std::string(checkpoint_str))))
      .WillOnce(Return(FakeHttpResponse(501, HeaderList())));
  ExpectSuccessfulAbortAggregationRequest("https://aggregation.second.uri");
  absl::Status report_result =
      federated_protocol_->ReportCompleted(std::move(results), plan_duration);
  ASSERT_THAT(report_result, IsCode(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(report_result.message(), HasSubstr("Data upload failed"));
  EXPECT_THAT(report_result.message(), HasSubstr("501"));
}

TEST_F(HttpFederatedProtocolTest, TestReportCompletedUploadInterrupted) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  // Issue a regular checkin
  ASSERT_OK(RunSuccessfulCheckin());

  std::string checkpoint_str;
  const size_t kTFCheckpointSize = 32;
  checkpoint_str.resize(kTFCheckpointSize, 'X');
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", checkpoint_str);
  absl::Duration plan_duration = absl::Minutes(5);

  ExpectSuccessfulReportTaskResultRequest(
      "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
      "taskassignments/CLIENT_SESSION_ID:reportresult?%24alt=proto",
      kAggregationSessionId, plan_duration);
  ExpectSuccessfulStartAggregationDataUploadRequest(
      "https://aggregation.uri/v1/aggregations/AGGREGATION_SESSION_ID/"
      "clients/CLIENT_TOKEN:startdataupload?%24alt=proto",
      kResourceName, kByteStreamTargetUri, kSecondStageAggregationTargetUri);
  absl::Notification request_issued;
  absl::Notification request_cancelled;

  // Make HttpClient::PerformRequests() block until the counter is decremented.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  StrEq("https://bytestream.uri/upload/v1/media/"
                        "CHECKPOINT_RESOURCE?upload_protocol=raw"),
                  HttpRequest::Method::kPost, _, std::string(checkpoint_str))))
      .WillOnce([&request_issued, &request_cancelled](
                    MockableHttpClient::SimpleHttpRequest ignored) {
        request_issued.Notify();
        request_cancelled.WaitForNotification();
        return FakeHttpResponse(503, HeaderList(), "");
      });
  // Make should_abort return false until we know that the request was issued
  // (i.e. once InterruptibleRunner has actually started running the code it
  // was given), and then make it return true, triggering an abort sequence and
  // unblocking the PerformRequests()() call we caused to block above.
  EXPECT_CALL(mock_should_abort_, Call()).WillRepeatedly([&request_issued] {
    return request_issued.HasBeenNotified();
  });

  // When the HttpClient receives a HttpRequestHandle::Cancel call, we let the
  // request complete.
  mock_http_client_.SetCancellationListener(
      [&request_cancelled]() { request_cancelled.Notify(); });

  EXPECT_CALL(mock_log_manager_,
              LogDiag(ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP));
  ExpectSuccessfulAbortAggregationRequest("https://aggregation.second.uri");
  absl::Status report_result =
      federated_protocol_->ReportCompleted(std::move(results), plan_duration);
  ASSERT_THAT(report_result, IsCode(absl::StatusCode::kCancelled));
  EXPECT_THAT(report_result.message(), HasSubstr("Data upload failed"));
}

TEST_F(HttpFederatedProtocolTest,
       TestReportCompletedSubmitAggregationResultFailed) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  // Issue a regular checkin
  ASSERT_OK(RunSuccessfulCheckin());

  std::string checkpoint_str;
  const size_t kTFCheckpointSize = 32;
  checkpoint_str.resize(kTFCheckpointSize, 'X');
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", checkpoint_str);
  absl::Duration plan_duration = absl::Minutes(5);

  ExpectSuccessfulReportTaskResultRequest(
      "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
      "taskassignments/CLIENT_SESSION_ID:reportresult?%24alt=proto",
      kAggregationSessionId, plan_duration);
  ExpectSuccessfulStartAggregationDataUploadRequest(
      "https://aggregation.uri/v1/aggregations/AGGREGATION_SESSION_ID/"
      "clients/CLIENT_TOKEN:startdataupload?%24alt=proto",
      kResourceName, kByteStreamTargetUri, kSecondStageAggregationTargetUri);
  ExpectSuccessfulByteStreamUploadRequest(
      "https://bytestream.uri/upload/v1/media/"
      "CHECKPOINT_RESOURCE?upload_protocol=raw",
      checkpoint_str);

  SubmitAggregationResultRequest submit_aggregation_result_request;
  submit_aggregation_result_request.set_resource_name(kResourceName);
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://aggregation.second.uri/v1/aggregations/"
          "AGGREGATION_SESSION_ID/clients/CLIENT_TOKEN:submit?%24alt=proto",
          HttpRequest::Method::kPost, _,
          submit_aggregation_result_request.SerializeAsString())))
      .WillOnce(Return(FakeHttpResponse(409, HeaderList())));
  absl::Status report_result =
      federated_protocol_->ReportCompleted(std::move(results), plan_duration);

  ASSERT_THAT(report_result, IsCode(absl::StatusCode::kAborted));
  EXPECT_THAT(report_result.message(),
              HasSubstr("SubmitAggregationResult failed"));
  EXPECT_THAT(report_result.message(), HasSubstr("409"));
}

TEST_F(HttpFederatedProtocolTest, TestReportNotCompletedSuccess) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  // Issue a regular checkin
  ASSERT_OK(RunSuccessfulCheckin());
  absl::Duration plan_duration = absl::Minutes(5);
  ReportTaskResultResponse response;
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
                  "taskassignments/CLIENT_SESSION_ID:reportresult?%24alt=proto",
                  HttpRequest::Method::kPost, _,
                  ReportTaskResultRequestMatcher(
                      EqualsProto(GetExpectedReportTaskResultRequest(
                          kAggregationSessionId, ::google::rpc::Code::INTERNAL,
                          plan_duration))))))
      .WillOnce(Return(
          FakeHttpResponse(200, HeaderList(), response.SerializeAsString())));

  ASSERT_OK(federated_protocol_->ReportNotCompleted(engine::PhaseOutcome::ERROR,
                                                    plan_duration));

  ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_F(HttpFederatedProtocolTest, TestReportNotCompletedError) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  // Issue a regular checkin
  ASSERT_OK(RunSuccessfulCheckin());
  ReportTaskResultResponse response;
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
                  "taskassignments/CLIENT_SESSION_ID:reportresult?%24alt=proto",
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(503, HeaderList())));

  absl::Status status = federated_protocol_->ReportNotCompleted(
      engine::PhaseOutcome::ERROR, absl::Minutes(5));
  EXPECT_THAT(status, IsCode(UNAVAILABLE));
  EXPECT_THAT(
      status.message(),
      AllOf(HasSubstr("ReportTaskResult request failed:"), HasSubstr("503")));
  ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_F(HttpFederatedProtocolTest, TestReportNotCompletedPermanentError) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  // Issue a regular checkin
  ASSERT_OK(RunSuccessfulCheckin());
  ReportTaskResultResponse response;
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://taskassignment.uri/v1/populations/TEST%2FPOPULATION/"
                  "taskassignments/CLIENT_SESSION_ID:reportresult?%24alt=proto",
                  HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(404, HeaderList())));

  absl::Status status = federated_protocol_->ReportNotCompleted(
      engine::PhaseOutcome::ERROR, absl::Minutes(5));
  EXPECT_THAT(status, IsCode(NOT_FOUND));
  EXPECT_THAT(
      status.message(),
      AllOf(HasSubstr("ReportTaskResult request failed:"), HasSubstr("404")));
  ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST(ProtocolRequestCreatorTest, TestInvalidForwardingInfo) {
  // If a ForwardingInfo does not have a target_uri_prefix field set then the
  // ProcessForwardingInfo call should fail.
  ForwardingInfo forwarding_info;
  EXPECT_THAT(ProtocolRequestCreator::Create(forwarding_info,
                                             /*use_compression=*/false),
              IsCode(INVALID_ARGUMENT));

  (*forwarding_info.mutable_extra_request_headers())["x-header1"] =
      "header-value1";
  EXPECT_THAT(ProtocolRequestCreator::Create(forwarding_info,
                                             /*use_compression=*/false),
              IsCode(INVALID_ARGUMENT));
}

TEST(ProtocolRequestCreatorTest, CreateProtocolRequestInvalidSuffix) {
  ProtocolRequestCreator creator("https://initial.uri", HeaderList(),
                                 /*use_compression=*/false);
  std::string uri_suffix = "v1/request";
  ASSERT_THAT(
      creator.CreateProtocolRequest(uri_suffix, QueryParams(),
                                    HttpRequest::Method::kPost, "request_body",
                                    /*is_protobuf_encoded=*/false),
      IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(ProtocolRequestCreatorTest, CreateProtocolRequest) {
  ProtocolRequestCreator creator("https://initial.uri", HeaderList(),
                                 /*use_compression=*/false);
  std::string expected_body = "expected_body";
  auto request = creator.CreateProtocolRequest(
      "/v1/request", QueryParams(), HttpRequest::Method::kPost, expected_body,
      /*is_protobuf_encoded=*/false);

  ASSERT_OK(request);
  EXPECT_EQ((*request)->uri(), "https://initial.uri/v1/request");
  EXPECT_EQ((*request)->method(), HttpRequest::Method::kPost);
  EXPECT_THAT((*request)->extra_headers(),
              UnorderedElementsAre(Header{
                  "Content-Length", std::to_string(expected_body.size())}));
  EXPECT_TRUE((*request)->HasBody());
  std::string actual_body;
  actual_body.resize(expected_body.size());
  ASSERT_OK((*request)->ReadBody(actual_body.data(), expected_body.size()));
  EXPECT_EQ(actual_body, expected_body);
}

TEST(ProtocolRequestCreatorTest, CreateProtobufEncodedProtocolRequest) {
  ProtocolRequestCreator creator("https://initial.uri", HeaderList(),
                                 /*use_compression=*/false);
  std::string expected_body = "expected_body";
  auto request = creator.CreateProtocolRequest(
      "/v1/request", QueryParams(), HttpRequest::Method::kPost, expected_body,
      /*is_protobuf_encoded=*/true);

  ASSERT_OK(request);
  EXPECT_EQ((*request)->uri(), "https://initial.uri/v1/request?%24alt=proto");
  EXPECT_EQ((*request)->method(), HttpRequest::Method::kPost);
  EXPECT_THAT((*request)->extra_headers(),
              UnorderedElementsAre(
                  Header{"Content-Length", absl::StrCat(expected_body.size())},
                  Header{"Content-Type", "application/x-protobuf"}));
  EXPECT_TRUE((*request)->HasBody());
  std::string actual_body;
  actual_body.resize(expected_body.size());
  ASSERT_OK((*request)->ReadBody(actual_body.data(), actual_body.size()));
  EXPECT_EQ(actual_body, expected_body);
}

TEST(ProtocolRequestCreatorTest, CreateGetOperationRequest) {
  ProtocolRequestCreator creator("https://initial.uri", HeaderList(),
                                 /*use_compression=*/false);
  std::string operation_name = "my_operation";
  auto request = creator.CreateGetOperationRequest(operation_name);
  ASSERT_OK(request);
  EXPECT_EQ((*request)->uri(), "https://initial.uri/v1/my_operation");
  EXPECT_EQ((*request)->method(), HttpRequest::Method::kGet);
  EXPECT_THAT((*request)->extra_headers(), IsEmpty());
  EXPECT_FALSE((*request)->HasBody());
}

class ProtocolRequestHelperTest : public testing::Test {
 public:
  ProtocolRequestHelperTest()
      : interruptible_runner_(
            &mock_log_manager_, mock_should_abort_.AsStdFunction(),
            InterruptibleRunner::TimingConfig{
                .polling_period = absl::ZeroDuration(),
                .graceful_shutdown_period = absl::InfiniteDuration(),
                .extended_shutdown_period = absl::InfiniteDuration()},
            InterruptibleRunner::DiagnosticsConfig{
                .interrupted = ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP,
                .interrupt_timeout =
                    ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP_TIMED_OUT,
                .interrupted_extended = ProdDiagCode::
                    BACKGROUND_TRAINING_INTERRUPT_HTTP_EXTENDED_COMPLETED,
                .interrupt_timeout_extended = ProdDiagCode::
                    BACKGROUND_TRAINING_INTERRUPT_HTTP_EXTENDED_TIMED_OUT}),
        initial_request_creator_("https://initial.uri", HeaderList(),
                                 /*use_compression=*/false),
        protocol_request_helper_(&mock_http_client_, &bytes_downloaded_,
                                 &bytes_uploaded_) {}

 protected:
  void TearDown() override {
    // Regardless of the outcome of the test (or the protocol interaction being
    // tested), network usage must always be reflected in the network stats.
    EXPECT_THAT(bytes_downloaded_, mock_http_client_.TotalReceivedBytes());
    EXPECT_THAT(bytes_uploaded_, mock_http_client_.TotalSentBytes());
  }

  StrictMock<MockHttpClient> mock_http_client_;

  NiceMock<MockLogManager> mock_log_manager_;
  NiceMock<MockFunction<bool()>> mock_should_abort_;

  int64_t bytes_downloaded_ = 0;
  int64_t bytes_uploaded_ = 0;

  InterruptibleRunner interruptible_runner_;
  ProtocolRequestCreator initial_request_creator_;
  // The class under test.
  ProtocolRequestHelper protocol_request_helper_;
};

Any GetFakeAnyProto() {
  Any fake_any;
  fake_any.set_type_url("the_type_url");
  *fake_any.mutable_value() = "the_value";
  return fake_any;
}

TEST_F(ProtocolRequestHelperTest, TestForwardingInfoIsPassedAlongCorrectly) {
  // The initial request should use the initial entry point URI and an empty set
  // of headers.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://initial.uri/suffix1", HttpRequest::Method::kPost,
                  // This request has a response body, so the HttpClient will
                  // add this header automatically.
                  ContainerEq(HeaderList{{"Content-Length", "5"}}), "body1")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "response1")));
  auto request_creator = std::make_unique<ProtocolRequestCreator>(
      "https://initial.uri", HeaderList(),
      /*use_compression=*/false);
  auto http_request = request_creator->CreateProtocolRequest(
      "/suffix1", QueryParams(), HttpRequest::Method::kPost, "body1",
      /*is_protobuf_encoded=*/false);
  ASSERT_OK(http_request);
  auto result = protocol_request_helper_.PerformProtocolRequest(
      *std::move(http_request), interruptible_runner_);
  ASSERT_OK(result);
  VerifyInMemoryHttpResponse(*result, 200, "", "response1");

  {
    // Process some fake ForwardingInfo.
    ForwardingInfo forwarding_info1;
    forwarding_info1.set_target_uri_prefix("https://second.uri/");
    (*forwarding_info1.mutable_extra_request_headers())["x-header1"] =
        "header-value1";
    (*forwarding_info1.mutable_extra_request_headers())["x-header2"] =
        "header-value2";
    auto new_request_creator = ProtocolRequestCreator::Create(
        forwarding_info1, /*use_compression=*/false);
    ASSERT_OK(new_request_creator);
    request_creator = std::move(*new_request_creator);
  }

  // The next series of requests should now use the ForwardingInfo (incl. use
  // the "https://second.uri/" prefix, and include the headers). Note that we
  // must use UnorderedElementsAre since the iteration order of the headers in
  // the `ForwardingInfo` is undefined.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://second.uri/suffix2", HttpRequest::Method::kGet,
                  UnorderedElementsAre(Header{"x-header1", "header-value1"},
                                       Header{"x-header2", "header-value2"}),
                  "")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "response2")));
  http_request = request_creator->CreateProtocolRequest(
      "/suffix2", QueryParams(), HttpRequest::Method::kGet, "",
      /*is_protobuf_encoded=*/false);
  ASSERT_OK(http_request);
  result = protocol_request_helper_.PerformProtocolRequest(
      *std::move(http_request), interruptible_runner_);
  ASSERT_OK(result);
  EXPECT_EQ(result->body, "response2");

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://second.uri/suffix3", HttpRequest::Method::kPut,
                  UnorderedElementsAre(Header{"x-header1", "header-value1"},
                                       Header{"x-header2", "header-value2"},
                                       // This request has a response body, so
                                       // the HttpClient will add this header
                                       // automatically.
                                       Header{"Content-Length", "5"}),
                  "body3")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "response3")));
  http_request = request_creator->CreateProtocolRequest(
      "/suffix3", QueryParams(), HttpRequest::Method::kPut, "body3",
      /*is_protobuf_encoded=*/false);
  ASSERT_OK(http_request);
  result = protocol_request_helper_.PerformProtocolRequest(
      *std::move(http_request), interruptible_runner_);
  ASSERT_OK(result);
  EXPECT_EQ(result->body, "response3");

  {
    // Process some more fake ForwardingInfo (without any headers this time).
    ForwardingInfo forwarding_info2;
    forwarding_info2.set_target_uri_prefix("https://third.uri");
    auto new_request_creator = ProtocolRequestCreator::Create(
        forwarding_info2, /*use_compression=*/false);
    ASSERT_OK(new_request_creator);
    request_creator = std::move(*new_request_creator);
  }

  // The next request should now use the latest ForwardingInfo again (i.e. use
  // the "https://third.uri/" prefix, and not specify any headers).
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  "https://third.uri/suffix4", HttpRequest::Method::kPost,
                  // This request has a response body, so the HttpClient will
                  // add this header automatically.
                  ContainerEq(HeaderList{{"Content-Length", "5"}}), "body4")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "response4")));
  http_request = request_creator->CreateProtocolRequest(
      "/suffix4", QueryParams(), HttpRequest::Method::kPost, "body4",
      /*is_protobuf_encoded=*/false);
  ASSERT_OK(http_request);
  result = protocol_request_helper_.PerformProtocolRequest(
      *std::move(http_request), interruptible_runner_);
  ASSERT_OK(result);
  EXPECT_EQ(result->body, "response4");
}

TEST_F(ProtocolRequestHelperTest, TestPollOperationInvalidResponse) {
  absl::StatusOr<Operation> result =
      protocol_request_helper_.PollOperationResponseUntilDone(
          InMemoryHttpResponse{.code = 200,
                               .content_encoding = "",
                               .body = absl::Cord("im_not_an_operation_proto")},
          initial_request_creator_, interruptible_runner_);
  EXPECT_THAT(result.status(), IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(result.status().message(),
              HasSubstr("could not parse Operation"));
}

TEST_F(ProtocolRequestHelperTest, TestPollOperationInvalidOperationName) {
  absl::StatusOr<Operation> result =
      protocol_request_helper_.PollOperationResponseUntilDone(
          InMemoryHttpResponse{.code = 200,
                               .content_encoding = "",
                               .body = absl::Cord(CreatePendingOperation(
                                                      "invalid_operation_name")
                                                      .SerializeAsString())},
          initial_request_creator_, interruptible_runner_);
  EXPECT_THAT(result.status(), IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(result.status().message(), HasSubstr("invalid name"));
}

TEST_F(ProtocolRequestHelperTest, TestPollOperationImmediateHttpError) {
  absl::StatusOr<Operation> result =
      protocol_request_helper_.PollOperationResponseUntilDone(
          absl::Status(absl::StatusCode::kNotFound, "foo"),
          initial_request_creator_, interruptible_runner_);
  EXPECT_THAT(result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(result.status().message(), HasSubstr("foo"));
}

// Ensures that if the very first HTTP response we receive indicates that the
// request was interrupted, we don't try to issue a cancellation request (since
// we haven't even received an Operation response with a name yet at that
// point).
TEST_F(ProtocolRequestHelperTest, TestPollOperationImmediateInterruptedError) {
  absl::StatusOr<Operation> result =
      protocol_request_helper_.PollOperationResponseUntilDone(
          absl::Status(absl::StatusCode::kCancelled, "foo"),
          initial_request_creator_, interruptible_runner_);
  EXPECT_THAT(result.status(), IsCode(CANCELLED));
  EXPECT_THAT(result.status().message(), HasSubstr("foo"));
}

TEST_F(ProtocolRequestHelperTest, TestPollOperationResponseImmediateSuccess) {
  Operation expected_response = CreateDoneOperation(GetFakeAnyProto());
  absl::StatusOr<Operation> result =
      protocol_request_helper_.PollOperationResponseUntilDone(
          InMemoryHttpResponse{
              .code = 200,
              .content_encoding = "",
              .body = absl::Cord(expected_response.SerializeAsString())},
          initial_request_creator_, interruptible_runner_);
  ASSERT_OK(result);
  EXPECT_THAT(*result, EqualsProto(expected_response));
}

TEST_F(ProtocolRequestHelperTest,
       TestPollOperationResponseImmediateOperationError) {
  Operation expected_response =
      CreateErrorOperation(ALREADY_EXISTS, "some error");
  absl::StatusOr<Operation> result =
      protocol_request_helper_.PollOperationResponseUntilDone(
          InMemoryHttpResponse{
              .code = 200,
              .content_encoding = "",
              .body = absl::Cord(expected_response.SerializeAsString())},
          initial_request_creator_, interruptible_runner_);
  ASSERT_OK(result);
  EXPECT_THAT(*result, EqualsProto(expected_response));
}

TEST_F(ProtocolRequestHelperTest,
       TestPollOperationResponseSuccessAfterPolling) {
  // Make the initial request return a pending Operation result. Note that we
  // use a '#' character in the operation name to allow us to verify that it
  // is properly URL-encoded.
  Operation pending_operation_response =
      CreatePendingOperation("operations/foo#bar");

  // Then, after letting the operation get polled twice more, eventually
  // return a fake response.
  Operation expected_response = CreateDoneOperation(GetFakeAnyProto());
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  // Note that the '#' character is encoded as "%23".
                  "https://initial.uri/v1/operations/foo%23bar",
                  HttpRequest::Method::kGet, _, IsEmpty())))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), pending_operation_response.SerializeAsString())))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), pending_operation_response.SerializeAsString())))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), expected_response.SerializeAsString())));

  absl::StatusOr<Operation> result =
      protocol_request_helper_.PollOperationResponseUntilDone(
          InMemoryHttpResponse{
              .code = 200,
              .content_encoding = "",
              .body =
                  absl::Cord(pending_operation_response.SerializeAsString())},
          initial_request_creator_, interruptible_runner_);
  ASSERT_OK(result);
  EXPECT_THAT(*result, EqualsProto(expected_response));
}

TEST_F(ProtocolRequestHelperTest, TestPollOperationResponseErrorAfterPolling) {
  // Make the initial request return a pending Operation result.
  Operation pending_operation_response =
      CreatePendingOperation("operations/foo#bar");

  // Then, after letting the operation get polled twice more, eventually
  // return a fake error.
  Operation expected_response =
      CreateErrorOperation(ALREADY_EXISTS, "some error");
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  // Note that the '#' character is encoded as "%23".
                  "https://initial.uri/v1/operations/foo%23bar",
                  HttpRequest::Method::kGet, _, IsEmpty())))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), pending_operation_response.SerializeAsString())))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), pending_operation_response.SerializeAsString())))
      .WillOnce(Return(FakeHttpResponse(
          200, HeaderList(), expected_response.SerializeAsString())));

  absl::StatusOr<Operation> result =
      protocol_request_helper_.PollOperationResponseUntilDone(
          InMemoryHttpResponse{
              .code = 200,
              .content_encoding = "",
              .body =
                  absl::Cord(pending_operation_response.SerializeAsString())},
          initial_request_creator_, interruptible_runner_);
  ASSERT_OK(result);
  EXPECT_THAT(*result, EqualsProto(expected_response));
}

TEST_F(ProtocolRequestHelperTest, PerformMultipleRequestsSuccess) {
  auto request_a = initial_request_creator_.CreateProtocolRequest(
      "/v1/request_a", QueryParams(), HttpRequest::Method::kPost, "body1",
      /*is_protobuf_encoded=*/false);
  ASSERT_OK(request_a);
  auto request_b = initial_request_creator_.CreateProtocolRequest(
      "/v1/request_b", QueryParams(), HttpRequest::Method::kPost, "body2",
      /*is_protobuf_encoded=*/false);
  ASSERT_OK(request_b);
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/request_a", HttpRequest::Method::kPost,
          // This request has a response body, so the HttpClient will
          // add this header automatically.
          ContainerEq(HeaderList{{"Content-Length", "5"}}), "body1")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "response1")));
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/request_b", HttpRequest::Method::kPost,
          // This request has a response body, so the HttpClient will
          // add this header automatically.
          ContainerEq(HeaderList{{"Content-Length", "5"}}), "body2")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "response2")));

  std::vector<std::unique_ptr<HttpRequest>> requests;
  requests.push_back(std::move(*request_a));
  requests.push_back(std::move(*request_b));

  auto result = protocol_request_helper_.PerformMultipleProtocolRequests(
      std::move(requests), interruptible_runner_);
  ASSERT_OK(result);
  auto response_1 = (*result)[0];
  ASSERT_OK(response_1);
  VerifyInMemoryHttpResponse(*response_1, 200, "", "response1");
  auto response_2 = (*result)[1];
  ASSERT_OK(response_2);
  VerifyInMemoryHttpResponse(*response_2, 200, "", "response2");
}

TEST_F(ProtocolRequestHelperTest, PerformMultipleRequestsPartialFail) {
  std::string uri_suffix_a = "/v1/request_a";
  auto request_a = initial_request_creator_.CreateProtocolRequest(
      uri_suffix_a, QueryParams(), HttpRequest::Method::kPost, "body1",
      /*is_protobuf_encoded=*/false);
  ASSERT_OK(request_a);
  std::string uri_suffix_b = "/v1/request_b";
  auto request_b = initial_request_creator_.CreateProtocolRequest(
      uri_suffix_b, QueryParams(), HttpRequest::Method::kPost, "body2",
      /*is_protobuf_encoded=*/false);
  ASSERT_OK(request_b);
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/request_a", HttpRequest::Method::kPost,
          // This request has a response body, so the HttpClient will
          // add this header automatically.
          ContainerEq(HeaderList{{"Content-Length", "5"}}), "body1")))
      .WillOnce(Return(FakeHttpResponse(200, HeaderList(), "response1")));
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/request_b", HttpRequest::Method::kPost,
          // This request has a response body, so the HttpClient will
          // add this header automatically.
          ContainerEq(HeaderList{{"Content-Length", "5"}}), "body2")))
      .WillOnce(
          Return(FakeHttpResponse(404, HeaderList(), "failure_response")));

  std::vector<std::unique_ptr<HttpRequest>> requests;
  requests.push_back(std::move(*request_a));
  requests.push_back(std::move(*request_b));

  auto result = protocol_request_helper_.PerformMultipleProtocolRequests(
      std::move(requests), interruptible_runner_);

  ASSERT_OK(result);
  auto response_1 = (*result)[0];
  ASSERT_OK(response_1);
  VerifyInMemoryHttpResponse(*response_1, 200, "", "response1");
  auto response_2 = (*result)[1];
  ASSERT_THAT(response_2, IsCode(absl::StatusCode::kNotFound));
}

}  // anonymous namespace
}  // namespace fcp::client::http
