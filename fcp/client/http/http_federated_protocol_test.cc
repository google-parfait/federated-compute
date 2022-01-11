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
#include <vector>

#include "google/protobuf/duration.pb.h"
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
#include "fcp/client/event_publisher.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/test_helpers.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/task_environment.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/federatedcompute/common.pb.h"
#include "fcp/protos/federatedcompute/eligibility_eval_tasks.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/testing/testing.h"

namespace fcp::client::http {
namespace {

using ::fcp::EqualsProto;
using ::fcp::IsCode;
using ::google::internal::federated::plan::ClientOnlyPlan;
using ::google::internal::federatedcompute::v1::EligibilityEvalTask;
using ::google::internal::federatedcompute::v1::EligibilityEvalTaskRequest;
using ::google::internal::federatedcompute::v1::EligibilityEvalTaskResponse;
using ::google::internal::federatedcompute::v1::ForwardingInfo;
using ::google::internal::federatedcompute::v1::Resource;
using ::google::internal::federatedcompute::v1::RetryWindow;
using ::google::internal::federatedml::v2::TaskEligibilityInfo;
using ::testing::_;
using ::testing::AllOf;
using ::testing::DescribeMatcher;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::ExplainMatchResult;
using ::testing::FieldsAre;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::Lt;
using ::testing::MockFunction;
using ::testing::NiceMock;
using ::testing::Not;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::VariantWith;

constexpr char kEntryPointUri[] = "https://initial.uri/";
constexpr char kApiKey[] = "TEST_APIKEY";
constexpr char kPopulationName[] = "TESTPOPULATION";
constexpr char kEligibilityEvalExecutionId[] =
    "TESTPOPULATION/ELIGIBILITY_EXECUTION_ID";
constexpr char kInitCheckpoint[] = "INIT_CHECKPOINT";
constexpr char kRetryToken[] = "OLD_RETRY_TOKEN";
constexpr char kClientVersion[] = "CLIENT_VERSION";
constexpr char kAttestationMeasurement[] = "ATTESTATION_MEASUREMENT";

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
  // The calculated retry delay must lie within the expected permanent errors
  // retry delay range.
  EXPECT_THAT(retry_window.delay_min().seconds() +
                  retry_window.delay_min().nanos() / 1000000000,
              AllOf(Ge(300L), Lt(399L)));
  EXPECT_THAT(retry_window.delay_max(), EqualsProto(retry_window.delay_min()));
}

ClientOnlyPlan GetFakePlan() {
  ClientOnlyPlan plan;
  plan.set_graph("im_a_tf_graph");
  return plan;
}

EligibilityEvalTaskResponse GetFakeEnabledEligibilityEvalTaskResponse(
    const Resource& plan, const Resource& checkpoint,
    const std::string& execution_id,
    const RetryWindow& accepted_retry_window = GetAcceptedRetryWindow(),
    const RetryWindow& rejected_retry_window = GetRejectedRetryWindow()) {
  EligibilityEvalTaskResponse response;
  EligibilityEvalTask* eval_task = response.mutable_eligibility_eval_task();
  *eval_task->mutable_plan() = plan;
  *eval_task->mutable_init_checkpoint() = checkpoint;
  eval_task->set_execution_id(execution_id);
  response.set_session_id("todo_session");
  ForwardingInfo* forwarding_info = response.mutable_forwarding_info();
  forwarding_info->set_target_uri_prefix("todo_target_uri");
  *response.mutable_retry_window_if_accepted() = accepted_retry_window;
  *response.mutable_retry_window_if_rejected() = rejected_retry_window;
  return response;
}

EligibilityEvalTaskResponse GetFakeDisabledEligibilityEvalTaskResponse() {
  EligibilityEvalTaskResponse response;
  response.mutable_no_eligibility_eval_configured();
  ForwardingInfo* forwarding_info = response.mutable_forwarding_info();
  forwarding_info->set_target_uri_prefix("todo_target_uri");
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

EligibilityEvalTaskRequest GetExpectedEligibilityEvalTaskRequest() {
  EligibilityEvalTaskRequest request;
  // Note: we don't expect population_name to be set, since it should be set in
  // the URI instead.
  request.mutable_client_version()->set_version_code(kClientVersion);

  return request;
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

    // We only initialize federated_protocol_ in this SetUp method, rather than
    // in the test's constructor, to ensure that we can set mock flag values
    // before the HttpFederatedProtocol constructor is called. Using
    // std::unique_ptr conveniently allows us to assign the field a new value
    // after construction (which we could not do if the field's type was
    // HttpFederatedProtocol, since it doesn't have copy or move constructors).
    federated_protocol_ = std::make_unique<HttpFederatedProtocol>(
        &mock_event_publisher_, &mock_log_manager_, &mock_flags_,
        &mock_http_client_, kEntryPointUri, kApiKey, kPopulationName,
        kRetryToken, kClientVersion, kAttestationMeasurement,
        mock_should_abort_.AsStdFunction(), absl::BitGen(),
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

  StrictMock<MockHttpClient> mock_http_client_;

  StrictMock<MockEventPublisher> mock_event_publisher_;
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
      &mock_event_publisher_, &mock_log_manager_, &mock_flags_,
      &mock_http_client_, kEntryPointUri, kApiKey, kPopulationName, kRetryToken,
      kClientVersion, kAttestationMeasurement,
      mock_should_abort_.AsStdFunction(), absl::BitGen(),
      InterruptibleRunner::TimingConfig{
          .polling_period = absl::ZeroDuration(),
          .graceful_shutdown_period = absl::InfiniteDuration(),
          .extended_shutdown_period = absl::InfiniteDuration()});

  const ::google::internal::federatedml::v2::RetryWindow& retry_window2 =
      federated_protocol_->GetLatestRetryWindow();
  ExpectTransientErrorRetryWindow(retry_window2);

  EXPECT_THAT(retry_window1, Not(EqualsProto(retry_window2)));
}

// Temporary test to ensure that non-alphanumeric population names are
// correctly rejected, for the time being, since we don't have any support for
// URL escaping yet.
//
// TODO(team): Replace this test with one that verifies that the
// population name is properly URL escaped, once we have support for URL
// escaping.
TEST_F(HttpFederatedProtocolTest,
       TestEligibilityEvalCheckinRequestUnsupportedPopulationName) {
  federated_protocol_ = std::make_unique<HttpFederatedProtocol>(
      &mock_event_publisher_, &mock_log_manager_, &mock_flags_,
      &mock_http_client_, kEntryPointUri, kApiKey, "UNSUPPORTED_POPULATION",
      kRetryToken, kClientVersion, kAttestationMeasurement,
      mock_should_abort_.AsStdFunction(), absl::BitGen(),
      InterruptibleRunner::TimingConfig{
          .polling_period = absl::ZeroDuration(),
          .graceful_shutdown_period = absl::InfiniteDuration(),
          .extended_shutdown_period = absl::InfiniteDuration()});

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(UNIMPLEMENTED));
}

TEST_F(HttpFederatedProtocolTest,
       TestEligibilityEvalCheckinRequestFailsTransientError) {
  // Make the HTTP client return a 503 Service Unavailable error when the
  // EligibilityEvalCheckin(...) code issues the control protocol's HTTP
  // request. This should result in the error being returned as the result.
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(503, {}, "")));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(UNAVAILABLE));
  // No RetryWindows were received from the server, so we expect to get a
  // RetryWindow generated based on the transient errors retry delay flag.
  ExpectTransientErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_F(HttpFederatedProtocolTest,
       TestEligibilityEvalCheckinRequestFailsPermanentError) {
  // Make the HTTP client return a 404 Not Found error when the
  // EligibilityEvalCheckin(...) code issues the control protocol's HTTP
  // request. This should result in the error being returned as the result.
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(404, {}, "")));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(NOT_FOUND));
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
  absl::BlockingCounter counter_should_abort(1);
  // When the HttpClient receives a HttpRequestHandle::Cancel call, we decrement
  // the counter.
  mock_http_client_.SetCancellationListener(
      [&counter_should_abort]() { counter_should_abort.DecrementCount(); });

  // Make HttpClient::PerformRequests() block until the counter is decremented.
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce([&request_issued, &counter_should_abort](
                    MockableHttpClient::SimpleHttpRequest ignored) {
        request_issued.Notify();
        counter_should_abort.Wait();
        return FakeHttpResponse(503, {}, "");
      });
  // Make should_abort return false until we know that the request was issued
  // (i.e. once InterruptibleRunner has actually started running the code it
  // was given), and then make it return true, triggering an abort sequence and
  // unblocking the PerformRequests()() call we caused to block above.
  EXPECT_CALL(mock_should_abort_, Call()).WillRepeatedly([&request_issued] {
    return request_issued.HasBeenNotified();
  });

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
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _,
          EligibilityEvalTaskRequestMatcher(
              EqualsProto(GetExpectedEligibilityEvalTaskRequest())))))
      .WillOnce(Return(FakeHttpResponse(
          200, {},
          GetFakeRejectedEligibilityEvalTaskResponse().SerializeAsString())));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(*eligibility_checkin_result,
              VariantWith<FederatedProtocol::Rejection>(_));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_F(HttpFederatedProtocolTest, TestEligibilityEvalCheckinDisabled) {
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _,
          EligibilityEvalTaskRequestMatcher(
              EqualsProto(GetExpectedEligibilityEvalTaskRequest())))))
      .WillOnce(Return(FakeHttpResponse(
          200, {},
          GetFakeDisabledEligibilityEvalTaskResponse().SerializeAsString())));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(*eligibility_checkin_result,
              VariantWith<FederatedProtocol::EligibilityEvalDisabled>(_));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_F(HttpFederatedProtocolTest,
       TestEligibilityEvalCheckinEnabledWithInvalidPlan) {
  // We return a fake response with invalid plan data inline.
  Resource plan_resource;
  plan_resource.set_data("does_not_parse");
  std::string expected_execution_id = kEligibilityEvalExecutionId;
  EligibilityEvalTaskResponse eval_task_response =
      GetFakeEnabledEligibilityEvalTaskResponse(plan_resource, Resource(),
                                                expected_execution_id);
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(
          FakeHttpResponse(200, {}, eval_task_response.SerializeAsString())));

  EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(expected_execution_id));

  // A diag code should be logged to indicate the invalid plan was encountered.
  EXPECT_CALL(
      mock_log_manager_,
      LogDiag(
          ProdDiagCode::
              BACKGROUND_TRAINING_ELIGIBILITY_EVAL_FAILED_CANNOT_PARSE_PLAN));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(INTERNAL));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_F(HttpFederatedProtocolTest, TestEligibilityEvalCheckinEnabled) {
  // We return a fake response which requires fetching the plan via HTTP, but
  // which has the checkpoint data available inline.
  ClientOnlyPlan expected_plan = GetFakePlan();
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
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _,
          EligibilityEvalTaskRequestMatcher(
              EqualsProto(GetExpectedEligibilityEvalTaskRequest())))))
      .WillOnce(Return(
          FakeHttpResponse(200, {}, eval_task_response.SerializeAsString())));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(
          Return(FakeHttpResponse(200, {}, expected_plan.SerializeAsString())));

  EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(expected_execution_id));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(*eligibility_checkin_result,
              VariantWith<FederatedProtocol::CheckinResultPayload>(FieldsAre(
                  EqualsProto(expected_plan), expected_checkpoint, _, _)));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

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
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(
          FakeHttpResponse(200, {}, eval_task_response.SerializeAsString())));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  checkpoint_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(200, {}, "")));

  // Mock a failed plan fetch.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(404, {}, "")));

  EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(expected_execution_id));

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
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(
          FakeHttpResponse(200, {}, eval_task_response.SerializeAsString())));

  // Mock a failed checkpoint fetch.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  checkpoint_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(503, {}, "")));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(200, {}, "")));

  EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(expected_execution_id));

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
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(
          FakeHttpResponse(200, {}, eval_task_response.SerializeAsString())));

  EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(_));

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
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(
          FakeHttpResponse(200, {}, eval_task_response.SerializeAsString())));

  EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(_));

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
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(FakeHttpResponse(503, {}, "")));

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
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _,
          EligibilityEvalTaskRequestMatcher(
              EqualsProto(GetExpectedEligibilityEvalTaskRequest())))))
      .WillOnce(Return(FakeHttpResponse(
          200, {},
          GetFakeRejectedEligibilityEvalTaskResponse().SerializeAsString())));

  ASSERT_OK(federated_protocol_->EligibilityEvalCheckin());

  // A Checkin(...) request should now fail, because Checkin(...) should only
  // be a called after a successful EligibilityEvalCheckin(...) request, with a
  // non-rejection response.
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
  EXPECT_CALL(
      mock_http_client_,
      PerformSingleRequest(SimpleHttpRequestMatcher(
          "https://initial.uri/v1/eligibilityevaltasks/TESTPOPULATION:request",
          HttpRequest::Method::kPost, _, _)))
      .WillOnce(Return(
          FakeHttpResponse(200, {}, eval_task_response.SerializeAsString())));

  // Mock a failed plan/resource fetch.
  EXPECT_CALL(mock_http_client_, PerformSingleRequest(SimpleHttpRequestMatcher(
                                     _, HttpRequest::Method::kGet, _, "")))
      .WillRepeatedly(Return(FakeHttpResponse(503, {}, "")));

  EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(_));

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  // A Checkin(...) request should now fail, because Checkin(...) should only
  // be a called after a successful EligibilityEvalCheckin(...) request, with a
  // non-rejection response.
  ASSERT_DEATH(
      { auto unused = federated_protocol_->Checkin(TaskEligibilityInfo()); },
      _);
}

}  // anonymous namespace
}  // namespace fcp::client::http
