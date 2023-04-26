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
#include "fcp/client/grpc_federated_protocol.h"

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "google/protobuf/text_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/cache/test_helpers.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/grpc_bidi_stream.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/testing/test_helpers.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/stats.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/secagg/client/secagg_client.h"
#include "fcp/secagg/shared/aes_ctr_prng_factory.h"
#include "fcp/secagg/testing/fake_prng.h"
#include "fcp/secagg/testing/mock_send_to_server_interface.h"
#include "fcp/secagg/testing/mock_state_transition_listener.h"
#include "fcp/testing/testing.h"

namespace fcp::client {
namespace {

using ::fcp::EqualsProto;
using ::fcp::IsCode;
using ::fcp::client::http::FakeHttpResponse;
using ::fcp::client::http::HttpRequest;
using ::fcp::client::http::MockHttpClient;
using ::fcp::client::http::SimpleHttpRequestMatcher;
using ::google::internal::federatedml::v2::AcceptanceInfo;
using ::google::internal::federatedml::v2::CheckinRequest;
using ::google::internal::federatedml::v2::ClientStreamMessage;
using ::google::internal::federatedml::v2::EligibilityEvalCheckinRequest;
using ::google::internal::federatedml::v2::EligibilityEvalPayload;
using ::google::internal::federatedml::v2::HttpCompressionFormat;
using ::google::internal::federatedml::v2::ReportResponse;
using ::google::internal::federatedml::v2::RetryWindow;
using ::google::internal::federatedml::v2::ServerStreamMessage;
using ::google::internal::federatedml::v2::TaskEligibilityInfo;
using ::google::internal::federatedml::v2::TaskWeight;
using ::testing::_;
using ::testing::AllOf;
using ::testing::DoAll;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Eq;
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
using ::testing::Pointee;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;
using ::testing::UnorderedElementsAre;
using ::testing::VariantWith;

constexpr char kPopulationName[] = "TEST/POPULATION";
constexpr char kFederatedSelectUriTemplate[] = "https://federated.select";
constexpr char kExecutionPhaseId[] = "TEST/POPULATION/TEST_TASK#1234.ab35";
constexpr char kPlan[] = "CLIENT_ONLY_PLAN";
constexpr char kInitCheckpoint[] = "INIT_CHECKPOINT";
constexpr char kRetryToken[] = "OLD_RETRY_TOKEN";
constexpr char kClientVersion[] = "CLIENT_VERSION";
constexpr char kAttestationMeasurement[] = "ATTESTATION_MEASUREMENT";
constexpr int kSecAggExpectedNumberOfClients = 10;
constexpr int kSecAggMinSurvivingClientsForReconstruction = 8;
constexpr int kSecAggMinClientsInServerVisibleAggregate = 4;

class MockGrpcBidiStream : public GrpcBidiStreamInterface {
 public:
  MOCK_METHOD(absl::Status, Send, (ClientStreamMessage*), (override));
  MOCK_METHOD(absl::Status, Receive, (ServerStreamMessage*), (override));
  MOCK_METHOD(void, Close, (), (override));
  MOCK_METHOD(int64_t, ChunkingLayerBytesSent, (), (override));
  MOCK_METHOD(int64_t, ChunkingLayerBytesReceived, (), (override));
};

constexpr int kTransientErrorsRetryPeriodSecs = 10;
constexpr double kTransientErrorsRetryDelayJitterPercent = 0.1;
constexpr double kExpectedTransientErrorsRetryPeriodSecsMin = 9.0;
constexpr double kExpectedTransientErrorsRetryPeriodSecsMax = 11.0;
constexpr int kPermanentErrorsRetryPeriodSecs = 100;
constexpr double kPermanentErrorsRetryDelayJitterPercent = 0.2;
constexpr double kExpectedPermanentErrorsRetryPeriodSecsMin = 80.0;
constexpr double kExpectedPermanentErrorsRetryPeriodSecsMax = 120.0;

void ExpectTransientErrorRetryWindow(const RetryWindow& retry_window) {
  // The calculated retry delay must lie within the expected transient errors
  // retry delay range.
  EXPECT_THAT(retry_window.delay_min().seconds() +
                  retry_window.delay_min().nanos() / 1000000000,
              AllOf(Ge(kExpectedTransientErrorsRetryPeriodSecsMin),
                    Lt(kExpectedTransientErrorsRetryPeriodSecsMax)));
  EXPECT_THAT(retry_window.delay_max(), EqualsProto(retry_window.delay_min()));
}

void ExpectPermanentErrorRetryWindow(const RetryWindow& retry_window) {
  // The calculated retry delay must lie within the expected permanent errors
  // retry delay range.
  EXPECT_THAT(retry_window.delay_min().seconds() +
                  retry_window.delay_min().nanos() / 1000000000,
              AllOf(Ge(kExpectedPermanentErrorsRetryPeriodSecsMin),
                    Lt(kExpectedPermanentErrorsRetryPeriodSecsMax)));
  EXPECT_THAT(retry_window.delay_max(), EqualsProto(retry_window.delay_min()));
}

google::internal::federatedml::v2::RetryWindow GetAcceptedRetryWindow() {
  google::internal::federatedml::v2::RetryWindow retry_window;
  // Must not overlap with kTransientErrorsRetryPeriodSecs or
  // kPermanentErrorsRetryPeriodSecs.
  retry_window.mutable_delay_min()->set_seconds(200L);
  retry_window.mutable_delay_max()->set_seconds(299L);
  *retry_window.mutable_retry_token() = "RETRY_TOKEN_ACCEPTED";
  return retry_window;
}

google::internal::federatedml::v2::RetryWindow GetRejectedRetryWindow() {
  google::internal::federatedml::v2::RetryWindow retry_window;
  // Must not overlap with kTransientErrorsRetryPeriodSecs or
  // kPermanentErrorsRetryPeriodSecs.
  retry_window.mutable_delay_min()->set_seconds(300);
  retry_window.mutable_delay_max()->set_seconds(399L);
  *retry_window.mutable_retry_token() = "RETRY_TOKEN_REJECTED";
  return retry_window;
}

void ExpectAcceptedRetryWindow(const RetryWindow& retry_window) {
  // The calculated retry delay must lie within the expected permanent errors
  // retry delay range.
  EXPECT_THAT(retry_window.delay_min().seconds() +
                  retry_window.delay_min().nanos() / 1000000000,
              AllOf(Ge(200), Lt(299L)));
  EXPECT_THAT(retry_window.delay_max(), EqualsProto(retry_window.delay_min()));
}

void ExpectRejectedRetryWindow(const RetryWindow& retry_window) {
  // The calculated retry delay must lie within the expected permanent errors
  // retry delay range.
  EXPECT_THAT(retry_window.delay_min().seconds() +
                  retry_window.delay_min().nanos() / 1000000000,
              AllOf(Ge(300), Lt(399)));
  EXPECT_THAT(retry_window.delay_max(), EqualsProto(retry_window.delay_min()));
}

ServerStreamMessage GetFakeCheckinRequestAck(
    const RetryWindow& accepted_retry_window = GetAcceptedRetryWindow(),
    const RetryWindow& rejected_retry_window = GetRejectedRetryWindow()) {
  ServerStreamMessage checkin_request_ack_message;
  *checkin_request_ack_message.mutable_checkin_request_ack()
       ->mutable_retry_window_if_accepted() = accepted_retry_window;
  *checkin_request_ack_message.mutable_checkin_request_ack()
       ->mutable_retry_window_if_rejected() = rejected_retry_window;
  return checkin_request_ack_message;
}

ServerStreamMessage GetFakeEnabledEligibilityCheckinResponse(
    const std::string& plan, const std::string& init_checkpoint,
    const std::string& execution_id) {
  ServerStreamMessage checkin_response_message;
  EligibilityEvalPayload* eval_payload =
      checkin_response_message.mutable_eligibility_eval_checkin_response()
          ->mutable_eligibility_eval_payload();
  eval_payload->set_plan(plan);
  eval_payload->set_init_checkpoint(init_checkpoint);
  eval_payload->set_execution_id(execution_id);
  return checkin_response_message;
}

ServerStreamMessage GetFakeDisabledEligibilityCheckinResponse() {
  ServerStreamMessage checkin_response_message;
  checkin_response_message.mutable_eligibility_eval_checkin_response()
      ->mutable_no_eligibility_eval_configured();
  return checkin_response_message;
}

ServerStreamMessage GetFakeRejectedEligibilityCheckinResponse() {
  ServerStreamMessage rejection_response_message;
  rejection_response_message.mutable_eligibility_eval_checkin_response()
      ->mutable_rejection_info();
  return rejection_response_message;
}

TaskEligibilityInfo GetFakeTaskEligibilityInfo() {
  TaskEligibilityInfo eligibility_info;
  TaskWeight* task_weight = eligibility_info.mutable_task_weights()->Add();
  task_weight->set_task_name("foo");
  task_weight->set_weight(567.8);
  return eligibility_info;
}

ServerStreamMessage GetFakeRejectedCheckinResponse() {
  ServerStreamMessage rejection_response_message;
  rejection_response_message.mutable_checkin_response()
      ->mutable_rejection_info();
  return rejection_response_message;
}

ServerStreamMessage GetFakeAcceptedCheckinResponse(
    const std::string& plan, const std::string& init_checkpoint,
    const std::string& federated_select_uri_template,
    const std::string& phase_id, bool use_secure_aggregation) {
  ServerStreamMessage checkin_response_message;
  AcceptanceInfo* acceptance_info =
      checkin_response_message.mutable_checkin_response()
          ->mutable_acceptance_info();
  acceptance_info->set_plan(plan);
  acceptance_info->set_execution_phase_id(phase_id);
  acceptance_info->set_init_checkpoint(init_checkpoint);
  acceptance_info->mutable_federated_select_uri_info()->set_uri_template(
      federated_select_uri_template);
  if (use_secure_aggregation) {
    auto sec_agg =
        acceptance_info->mutable_side_channel_protocol_execution_info()
            ->mutable_secure_aggregation();
    sec_agg->set_expected_number_of_clients(kSecAggExpectedNumberOfClients);
    sec_agg->set_minimum_surviving_clients_for_reconstruction(
        kSecAggMinSurvivingClientsForReconstruction);
    sec_agg->set_minimum_clients_in_server_visible_aggregate(
        kSecAggMinClientsInServerVisibleAggregate);
    checkin_response_message.mutable_checkin_response()
        ->mutable_protocol_options_response()
        ->mutable_side_channels()
        ->mutable_secure_aggregation()
        ->set_client_variant(secagg::SECAGG_CLIENT_VARIANT_NATIVE_V1);
  }
  return checkin_response_message;
}

ServerStreamMessage GetFakeReportResponse() {
  ServerStreamMessage report_response_message;
  *report_response_message.mutable_report_response() = ReportResponse();
  return report_response_message;
}

ClientStreamMessage GetExpectedEligibilityEvalCheckinRequest(
    bool enable_http_resource_support = false) {
  ClientStreamMessage expected_message;
  EligibilityEvalCheckinRequest* checkin_request =
      expected_message.mutable_eligibility_eval_checkin_request();
  checkin_request->set_population_name(kPopulationName);
  checkin_request->set_client_version(kClientVersion);
  checkin_request->set_retry_token(kRetryToken);
  checkin_request->set_attestation_measurement(kAttestationMeasurement);
  checkin_request->mutable_protocol_options_request()
      ->mutable_side_channels()
      ->mutable_secure_aggregation()
      ->add_client_variant(secagg::SECAGG_CLIENT_VARIANT_NATIVE_V1);
  checkin_request->mutable_protocol_options_request()->set_should_ack_checkin(
      true);
  checkin_request->mutable_protocol_options_request()
      ->add_supported_http_compression_formats(
          HttpCompressionFormat::HTTP_COMPRESSION_FORMAT_GZIP);

  if (enable_http_resource_support) {
    checkin_request->mutable_protocol_options_request()
        ->set_supports_http_download(true);
    checkin_request->mutable_protocol_options_request()
        ->set_supports_eligibility_eval_http_download(true);
  }

  return expected_message;
}

// This returns the CheckinRequest gRPC proto we expect each Checkin(...) call
// to result in.
ClientStreamMessage GetExpectedCheckinRequest(
    const std::optional<TaskEligibilityInfo>& task_eligibility_info =
        std::nullopt,
    bool enable_http_resource_support = false) {
  ClientStreamMessage expected_message;
  CheckinRequest* checkin_request = expected_message.mutable_checkin_request();
  checkin_request->set_population_name(kPopulationName);
  checkin_request->set_client_version(kClientVersion);
  checkin_request->set_retry_token(kRetryToken);
  checkin_request->set_attestation_measurement(kAttestationMeasurement);
  checkin_request->mutable_protocol_options_request()
      ->mutable_side_channels()
      ->mutable_secure_aggregation()
      ->add_client_variant(secagg::SECAGG_CLIENT_VARIANT_NATIVE_V1);
  checkin_request->mutable_protocol_options_request()->set_should_ack_checkin(
      false);
  checkin_request->mutable_protocol_options_request()
      ->add_supported_http_compression_formats(
          HttpCompressionFormat::HTTP_COMPRESSION_FORMAT_GZIP);

  if (enable_http_resource_support) {
    checkin_request->mutable_protocol_options_request()
        ->set_supports_http_download(true);
    checkin_request->mutable_protocol_options_request()
        ->set_supports_eligibility_eval_http_download(true);
  }

  if (task_eligibility_info.has_value()) {
    *checkin_request->mutable_task_eligibility_info() = *task_eligibility_info;
  }
  return expected_message;
}

class GrpcFederatedProtocolTest
    // The first parameter indicates whether support for HTTP task resources
    // should be enabled.
    : public testing::TestWithParam<bool> {
 public:
  GrpcFederatedProtocolTest() {
    // The gRPC stream should always be closed at the end of all tests.
    EXPECT_CALL(*mock_grpc_bidi_stream_, Close());
  }

 protected:
  void SetUp() override {
    enable_http_resource_support_ = GetParam();
    EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesReceived())
        .WillRepeatedly(Return(0));
    EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesSent())
        .WillRepeatedly(Return(0));
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
    EXPECT_CALL(mock_flags_,
                enable_grpc_with_eligibility_eval_http_resource_support)
        .WillRepeatedly(Return(enable_http_resource_support_));

    // We only initialize federated_protocol_ in this SetUp method, rather than
    // in the test's constructor, to ensure that we can set mock flag values
    // before the GrpcFederatedProtocol constructor is called. Using
    // std::unique_ptr conveniently allows us to assign the field a new value
    // after construction (which we could not do if the field's type was
    // GrpcFederatedProtocol, since it doesn't have copy or move constructors).
    federated_protocol_ = std::make_unique<GrpcFederatedProtocol>(
        &mock_event_publisher_, &mock_log_manager_,
        absl::WrapUnique(mock_secagg_runner_factory_), &mock_flags_,
        /*http_client=*/
        enable_http_resource_support_ ? &mock_http_client_ : nullptr,
        // We want to inject mocks stored in unique_ptrs to the
        // class-under-test, hence we transfer ownership via WrapUnique. To
        // write expectations for the mock, we retain the raw pointer to it,
        // which will be valid until GrpcFederatedProtocol's d'tor is called.
        absl::WrapUnique(mock_grpc_bidi_stream_), kPopulationName, kRetryToken,
        kClientVersion, kAttestationMeasurement,
        mock_should_abort_.AsStdFunction(), absl::BitGen(),
        InterruptibleRunner::TimingConfig{
            .polling_period = absl::ZeroDuration(),
            .graceful_shutdown_period = absl::InfiniteDuration(),
            .extended_shutdown_period = absl::InfiniteDuration()},
        &mock_resource_cache_);
  }

  void TearDown() override {
    fcp::client::http::HttpRequestHandle::SentReceivedBytes
        sent_received_bytes = mock_http_client_.TotalSentReceivedBytes();

    NetworkStats network_stats = federated_protocol_->GetNetworkStats();
    EXPECT_THAT(network_stats.bytes_downloaded,
                Ge(mock_grpc_bidi_stream_->ChunkingLayerBytesReceived() +
                   sent_received_bytes.received_bytes));
    EXPECT_THAT(network_stats.bytes_uploaded,
                Ge(mock_grpc_bidi_stream_->ChunkingLayerBytesSent() +
                   sent_received_bytes.sent_bytes));
    // If any network traffic occurred, we expect to see some time reflected in
    // the duration (if the flag is on).
    if (network_stats.bytes_uploaded > 0) {
      EXPECT_THAT(network_stats.network_duration, Gt(absl::ZeroDuration()));
    }
  }

  // This function runs a successful
  // EligibilityEvalCheckin(mock_eet_received_callback_.AsStdFunction()) that
  // results in an eligibility eval payload being returned by the server. This
  // is a utility function used by Checkin*() tests that depend on a prior,
  // successful execution of
  // EligibilityEvalCheckin(mock_eet_received_callback_.AsStdFunction()). It
  // returns a absl::Status, which the caller should verify is OK using
  // ASSERT_OK.
  absl::Status RunSuccessfulEligibilityEvalCheckin(
      bool eligibility_eval_enabled = true,
      const RetryWindow& accepted_retry_window = GetAcceptedRetryWindow(),
      const RetryWindow& rejected_retry_window = GetRejectedRetryWindow()) {
    EXPECT_CALL(
        *mock_grpc_bidi_stream_,
        Send(Pointee(EqualsProto(GetExpectedEligibilityEvalCheckinRequest(
            enable_http_resource_support_)))))
        .WillOnce(Return(absl::OkStatus()));

    const std::string expected_execution_id = "ELIGIBILITY_EVAL_EXECUTION_ID";
    EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
        .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                            accepted_retry_window, rejected_retry_window)),
                        Return(absl::OkStatus())))
        .WillOnce(
            DoAll(SetArgPointee<0>(
                      eligibility_eval_enabled
                          ? GetFakeEnabledEligibilityCheckinResponse(
                                kPlan, kInitCheckpoint, expected_execution_id)
                          : GetFakeDisabledEligibilityCheckinResponse()),
                  Return(absl::OkStatus())));

    return federated_protocol_
        ->EligibilityEvalCheckin(mock_eet_received_callback_.AsStdFunction())
        .status();
  }

  // This function runs a successful Checkin() that results in acceptance by the
  // server. This is a utility function used by Report*() tests that depend on a
  // prior, successful execution of Checkin().
  // It returns a absl::Status, which the caller should verify is OK using
  // ASSERT_OK.
  absl::StatusOr<FederatedProtocol::CheckinResult> RunSuccessfulCheckin(
      bool use_secure_aggregation,
      const std::optional<TaskEligibilityInfo>& task_eligibility_info =
          GetFakeTaskEligibilityInfo()) {
    EXPECT_CALL(*mock_grpc_bidi_stream_,
                Send(Pointee(EqualsProto(GetExpectedCheckinRequest(
                    task_eligibility_info, enable_http_resource_support_)))))
        .WillOnce(Return(absl::OkStatus()));

    {
      InSequence seq;
      EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
          .WillOnce(
              DoAll(SetArgPointee<0>(GetFakeAcceptedCheckinResponse(
                        kPlan, kInitCheckpoint, kFederatedSelectUriTemplate,
                        kExecutionPhaseId, use_secure_aggregation)),
                    Return(absl::OkStatus())))
          .RetiresOnSaturation();
    }

    return federated_protocol_->Checkin(
        task_eligibility_info, mock_task_received_callback_.AsStdFunction());
  }

  // See note in the constructor for why these are pointers.
  StrictMock<MockGrpcBidiStream>* mock_grpc_bidi_stream_ =
      new StrictMock<MockGrpcBidiStream>();

  StrictMock<MockEventPublisher> mock_event_publisher_;
  NiceMock<MockLogManager> mock_log_manager_;
  StrictMock<MockSecAggRunnerFactory>* mock_secagg_runner_factory_ =
      new StrictMock<MockSecAggRunnerFactory>();
  StrictMock<MockSecAggRunner>* mock_secagg_runner_;
  NiceMock<MockFlags> mock_flags_;
  StrictMock<MockHttpClient> mock_http_client_;
  NiceMock<MockFunction<bool()>> mock_should_abort_;
  StrictMock<cache::MockResourceCache> mock_resource_cache_;
  NiceMock<MockFunction<void(
      const ::fcp::client::FederatedProtocol::EligibilityEvalTask&)>>
      mock_eet_received_callback_;
  NiceMock<MockFunction<void(
      const ::fcp::client::FederatedProtocol::TaskAssignment&)>>
      mock_task_received_callback_;

  // The class under test.
  std::unique_ptr<GrpcFederatedProtocol> federated_protocol_;
  bool enable_http_resource_support_;
};

std::string GenerateTestName(
    const testing::TestParamInfo<GrpcFederatedProtocolTest::ParamType>& info) {
  std::string name = info.param ? "Http_resource_support_enabled"
                                : "Http_resource_support_disabled";
  return name;
}

INSTANTIATE_TEST_SUITE_P(NewVsOldBehavior, GrpcFederatedProtocolTest,
                         testing::Bool(), GenerateTestName);

using GrpcFederatedProtocolDeathTest = GrpcFederatedProtocolTest;
INSTANTIATE_TEST_SUITE_P(NewVsOldBehavior, GrpcFederatedProtocolDeathTest,
                         testing::Bool(), GenerateTestName);

TEST_P(GrpcFederatedProtocolTest,
       TestTransientErrorRetryWindowDifferentAcrossDifferentInstances) {
  const RetryWindow& retry_window1 =
      federated_protocol_->GetLatestRetryWindow();
  ExpectTransientErrorRetryWindow(retry_window1);
  federated_protocol_.reset(nullptr);

  mock_grpc_bidi_stream_ = new StrictMock<MockGrpcBidiStream>();
  EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesReceived())
      .WillRepeatedly(Return(0));
  EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesSent())
      .WillRepeatedly(Return(0));
  EXPECT_CALL(*mock_grpc_bidi_stream_, Close());
  mock_secagg_runner_factory_ = new StrictMock<MockSecAggRunnerFactory>();
  // Create a new GrpcFederatedProtocol instance. It should not produce the same
  // retry window value as the one we just got. This is a simple correctness
  // check to ensure that the value is at least randomly generated (and that we
  // don't accidentally use the random number generator incorrectly).
  federated_protocol_ = std::make_unique<GrpcFederatedProtocol>(
      &mock_event_publisher_, &mock_log_manager_,
      absl::WrapUnique(mock_secagg_runner_factory_), &mock_flags_,
      /*http_client=*/nullptr, absl::WrapUnique(mock_grpc_bidi_stream_),
      kPopulationName, kRetryToken, kClientVersion, kAttestationMeasurement,
      mock_should_abort_.AsStdFunction(), absl::BitGen(),
      InterruptibleRunner::TimingConfig{
          .polling_period = absl::ZeroDuration(),
          .graceful_shutdown_period = absl::InfiniteDuration(),
          .extended_shutdown_period = absl::InfiniteDuration()},
      &mock_resource_cache_);

  const RetryWindow& retry_window2 =
      federated_protocol_->GetLatestRetryWindow();
  ExpectTransientErrorRetryWindow(retry_window2);

  EXPECT_THAT(retry_window1, Not(EqualsProto(retry_window2)));
}

TEST_P(GrpcFederatedProtocolTest,
       TestEligibilityEvalCheckinSendFailsTransientError) {
  // Make the gRPC stream return an UNAVAILABLE error when the
  // EligibilityEvalCheckin(...) code tries to send its first message. This
  // should result in the error being returned as the result.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::UnavailableError("foo")));

  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(UNAVAILABLE));
  EXPECT_THAT(eligibility_checkin_result.status().message(), "foo");
  // No RetryWindows were received from the server, so we expect to get a
  // RetryWindow generated based on the transient errors retry delay flag.
  ExpectTransientErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest,
       TestEligibilityEvalCheckinSendFailsPermanentError) {
  // Make the gRPC stream return an NOT_FOUND error when the
  // EligibilityEvalCheckin(...) code tries to send its first message. This
  // should result in the error being returned as the result.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::NotFoundError("foo")));

  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(eligibility_checkin_result.status().message(), "foo");
  // No RetryWindows were received from the server, so we expect to get a
  // RetryWindow generated based on the *permanent* errors retry delay flag,
  // since NOT_FOUND is marked as a permanent error in the flags.
  ExpectPermanentErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Tests the case where the blocking Send() call in EligibilityEvalCheckin is
// interrupted.
TEST_P(GrpcFederatedProtocolTest, TestEligibilityEvalCheckinSendInterrupted) {
  absl::BlockingCounter counter_should_abort(1);

  // Make Send() block until the counter is decremented.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce([&counter_should_abort](ClientStreamMessage* ignored) {
        counter_should_abort.Wait();
        return absl::OkStatus();
      });
  // Make should_abort return false for the first two calls, and then make it
  // decrement the counter and return true, triggering an abort sequence and
  // unblocking the Send() call we caused to block above.
  EXPECT_CALL(mock_should_abort_, Call())
      .WillOnce(Return(false))
      .WillOnce(Return(false))
      .WillRepeatedly([&counter_should_abort] {
        counter_should_abort.DecrementCount();
        return true;
      });
  // In addition to the Close() call we expect in the test fixture above, expect
  // an additional one (the one that induced the abort).
  EXPECT_CALL(*mock_grpc_bidi_stream_, Close()).Times(1).RetiresOnSaturation();
  EXPECT_CALL(mock_log_manager_,
              LogDiag(ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_GRPC));

  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(CANCELLED));
  // No RetryWindows were received from the server, so we expect to get a
  // RetryWindow generated based on the transient errors retry delay flag.
  ExpectTransientErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// If a CheckinRequestAck is requested in the ProtocolOptionsRequest but not
// received, UNIMPLEMENTED should be returned.
TEST_P(GrpcFederatedProtocolTest,
       TestEligibilityEvalCheckinMissingCheckinRequestAck) {
  // We immediately return an EligibilityEvalCheckinResponse, rather than
  // returning a CheckinRequestAck first.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeRejectedEligibilityCheckinResponse()),
                Return(absl::OkStatus())));
  EXPECT_CALL(
      mock_log_manager_,
      LogDiag(
          ProdDiagCode::
              BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_EXPECTED_BUT_NOT_RECVD));  // NOLINT

  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(UNIMPLEMENTED));
  // No RetryWindows were received from the server, so we expect to get a
  // RetryWindow generated based on the *permanent* errors retry delay flag,
  // since UNIMPLEMENTED is marked as a permanent error in the flags.
  ExpectPermanentErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest,
       TestEligibilityEvalCheckinWaitForCheckinRequestAckFails) {
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::OkStatus()));

  // Make the very first Receive() call fail (i.e. the one expecting the
  // CheckinRequestAck).
  std::string expected_message = "foo";
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(Return(absl::AbortedError(expected_message)));

  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(ABORTED));
  EXPECT_THAT(eligibility_checkin_result.status().message(), expected_message);
  // No RetryWindows were received from the server, so we expect to get a
  // RetryWindow generated based on the transient errors retry delay flag.
  ExpectTransientErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest,
       TestEligibilityEvalCheckinWaitForCheckinResponseFails) {
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::OkStatus()));

  // Failed checkins that have received an ack already should return the
  // rejected retry window.
  std::string expected_message = "foo";
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck()),
                      Return(absl::OkStatus())))
      // Make the second Receive() call fail (i.e. the one expecting the
      // EligibilityEvalCheckinResponse).
      .WillOnce(Return(absl::AbortedError(expected_message)));

  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(ABORTED));
  EXPECT_THAT(eligibility_checkin_result.status().message(), expected_message);
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest, TestEligibilityEvalCheckinRejection) {
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::OkStatus()));

  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck()),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeRejectedEligibilityCheckinResponse()),
                Return(absl::OkStatus())));

  // The 'eet received' callback should not be invoked since no EET was given to
  // the client.
  EXPECT_CALL(mock_eet_received_callback_, Call(_)).Times(0);

  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(*eligibility_checkin_result,
              VariantWith<FederatedProtocol::Rejection>(_));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest, TestEligibilityEvalCheckinDisabled) {
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::OkStatus()));

  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck()),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeDisabledEligibilityCheckinResponse()),
                Return(absl::OkStatus())));

  // The 'eet received' callback should not be invoked since no EET was given to
  // the client.
  EXPECT_CALL(mock_eet_received_callback_, Call(_)).Times(0);

  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(*eligibility_checkin_result,
              VariantWith<FederatedProtocol::EligibilityEvalDisabled>(_));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest, TestEligibilityEvalCheckinEnabled) {
  // Note that in this particular test we check that the eligibility eval
  // checkin request is as expected (in all prior tests we just use the '_'
  // matcher, because the request isn't really relevant to the test).
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(GetExpectedEligibilityEvalCheckinRequest(
                  enable_http_resource_support_)))))
      .WillOnce(Return(absl::OkStatus()));

  // The EligibilityEvalCheckin(...) method should return the rejected
  // RetryWindow, since after merely completing an eligibility eval checkin the
  // client hasn't actually been accepted to a specific task yet.
  std::string expected_plan = kPlan;
  std::string expected_checkpoint = kInitCheckpoint;
  std::string expected_execution_id = "ELIGIBILITY_EVAL_EXECUTION_ID";
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck()),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeEnabledEligibilityCheckinResponse(
                    expected_plan, expected_checkpoint, expected_execution_id)),
                Return(absl::OkStatus())));
  EXPECT_CALL(
      mock_log_manager_,
      LogDiag(ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));

  // The 'EET received' callback should be called, even if the task resource
  // data was available inline.
  EXPECT_CALL(mock_eet_received_callback_,
              Call(FieldsAre(FieldsAre("", ""), expected_execution_id,
                             Eq(std::nullopt))));

  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  ASSERT_OK(eligibility_checkin_result);
  // If HTTP support is enabled then the checkpoint data gets returned in the
  // shape of an absl::Cord (rather than an std::string), regardless of
  // whether it was actually downloaded via HTTP.
  if (enable_http_resource_support_) {
    EXPECT_THAT(*eligibility_checkin_result,
                VariantWith<FederatedProtocol::EligibilityEvalTask>(
                    FieldsAre(FieldsAre(absl::Cord(expected_plan),
                                        absl::Cord(expected_checkpoint)),
                              expected_execution_id, Eq(std::nullopt))));
  } else {
    EXPECT_THAT(*eligibility_checkin_result,
                VariantWith<FederatedProtocol::EligibilityEvalTask>(
                    FieldsAre(FieldsAre(expected_plan, expected_checkpoint),
                              expected_execution_id, Eq(std::nullopt))));
  }
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest,
       TestEligiblityEvalCheckinEnabledWithHttpResourcesDownloaded) {
  if (!enable_http_resource_support_) {
    GTEST_SKIP() << "This test only applies if the HTTP task resources feature "
                    "is enabled";
    return;
  }

  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::OkStatus()));

  std::string expected_plan = kPlan;
  std::string plan_uri = "https://fake.uri/plan";
  std::string expected_checkpoint = kInitCheckpoint;
  std::string checkpoint_uri = "https://fake.uri/checkpoint";
  std::string expected_execution_id = "ELIGIBILITY_EVAL_EXECUTION_ID";
  ServerStreamMessage fake_response = GetFakeEnabledEligibilityCheckinResponse(
      /*plan=*/"", /*init_checkpoint=*/"", expected_execution_id);
  EligibilityEvalPayload* eligibility_eval_payload =
      fake_response.mutable_eligibility_eval_checkin_response()
          ->mutable_eligibility_eval_payload();
  eligibility_eval_payload->mutable_plan_resource()->set_uri(plan_uri);
  eligibility_eval_payload->mutable_init_checkpoint_resource()->set_uri(
      checkpoint_uri);

  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck()),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(fake_response), Return(absl::OkStatus())));
  EXPECT_CALL(
      mock_log_manager_,
      LogDiag(ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));

  {
    InSequence seq;
    // The 'EET received' callback should be called *before* the actual task
    // resources are fetched.
    EXPECT_CALL(mock_eet_received_callback_,
                Call(FieldsAre(FieldsAre("", ""), expected_execution_id,
                               Eq(std::nullopt))));

    EXPECT_CALL(mock_http_client_,
                PerformSingleRequest(SimpleHttpRequestMatcher(
                    plan_uri, HttpRequest::Method::kGet, _, "")))
        .WillOnce(Return(FakeHttpResponse(200, {}, expected_plan)));

    EXPECT_CALL(mock_http_client_,
                PerformSingleRequest(SimpleHttpRequestMatcher(
                    checkpoint_uri, HttpRequest::Method::kGet, _, "")))
        .WillOnce(Return(FakeHttpResponse(200, {}, expected_checkpoint)));
  }

  {
    InSequence seq;
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_USES_HTTP));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::
                HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_HTTP_FETCH_SUCCEEDED));
  }

  // Issue the Eligibility Eval checkin.
  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(
      *eligibility_checkin_result,
      VariantWith<FederatedProtocol::EligibilityEvalTask>(FieldsAre(
          FieldsAre(absl::Cord(expected_plan), absl::Cord(expected_checkpoint)),
          expected_execution_id, Eq(std::nullopt))));
}

TEST_P(GrpcFederatedProtocolTest,
       TestEligiblityEvalCheckinEnabledWithHttpResourcesPlanDataFetchFailed) {
  if (!enable_http_resource_support_) {
    GTEST_SKIP() << "This test only applies if the HTTP task resources feature "
                    "is enabled";
    return;
  }

  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::OkStatus()));

  std::string expected_plan = kPlan;
  std::string plan_uri = "https://fake.uri/plan";
  std::string expected_checkpoint = kInitCheckpoint;
  std::string expected_execution_id = "ELIGIBILITY_EVAL_EXECUTION_ID";
  ServerStreamMessage fake_response = GetFakeEnabledEligibilityCheckinResponse(
      /*plan=*/"", expected_checkpoint, expected_execution_id);
  fake_response.mutable_eligibility_eval_checkin_response()
      ->mutable_eligibility_eval_payload()
      ->mutable_plan_resource()
      ->set_uri(plan_uri);
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck()),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(fake_response), Return(absl::OkStatus())));
  EXPECT_CALL(
      mock_log_manager_,
      LogDiag(ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(404, {}, "")));

  {
    InSequence seq;
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_USES_HTTP));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::
                HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_HTTP_FETCH_FAILED));
  }

  // Issue the eligibility eval checkin.
  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(eligibility_checkin_result.status().message(),
              HasSubstr("plan fetch failed"));
  EXPECT_THAT(eligibility_checkin_result.status().message(), HasSubstr("404"));
  // The EligibilityEvalCheckin call is expected to return the permanent error
  // retry window, since 404 maps to a permanent error.
  ExpectPermanentErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest,
       TestEligiblityEvalCheckinEnabledWithHttpResourcesCheckpointFetchFailed) {
  if (!enable_http_resource_support_) {
    GTEST_SKIP() << "This test only applies if the HTTP task resources feature "
                    "is enabled";
    return;
  }

  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::OkStatus()));

  std::string expected_plan = kPlan;
  std::string expected_checkpoint = kInitCheckpoint;
  std::string checkpoint_uri = "https://fake.uri/checkpoint";
  std::string expected_execution_id = "ELIGIBILITY_EVAL_EXECUTION_ID";
  ServerStreamMessage fake_response = GetFakeEnabledEligibilityCheckinResponse(
      expected_plan, /*init_checkpoint=*/"", expected_execution_id);
  fake_response.mutable_eligibility_eval_checkin_response()
      ->mutable_eligibility_eval_payload()
      ->mutable_init_checkpoint_resource()
      ->set_uri(checkpoint_uri);
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck()),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(fake_response), Return(absl::OkStatus())));
  EXPECT_CALL(
      mock_log_manager_,
      LogDiag(ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  checkpoint_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(503, {}, "")));

  {
    InSequence seq;
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_USES_HTTP));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::
                HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_HTTP_FETCH_FAILED));
  }

  // Issue the eligibility eval checkin.
  auto eligibility_checkin_result = federated_protocol_->EligibilityEvalCheckin(
      mock_eet_received_callback_.AsStdFunction());

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(UNAVAILABLE));
  EXPECT_THAT(eligibility_checkin_result.status().message(),
              HasSubstr("checkpoint fetch failed"));
  EXPECT_THAT(eligibility_checkin_result.status().message(), HasSubstr("503"));
  // The EligibilityEvalCheckin call is expected to return the rejected error
  // retry window, since 503 maps to a transient error.
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Tests that the protocol correctly sanitizes any invalid values it may have
// received from the server.
TEST_P(GrpcFederatedProtocolTest,
       TestNegativeMinMaxRetryDelayValueSanitization) {
  google::internal::federatedml::v2::RetryWindow retry_window;
  retry_window.mutable_delay_min()->set_seconds(-1);
  retry_window.mutable_delay_max()->set_seconds(-2);

  // The above retry window's negative min/max values should be clamped to 0.
  google::internal::federatedml::v2::RetryWindow expected_retry_window;
  expected_retry_window.mutable_delay_min()->set_seconds(0);
  expected_retry_window.mutable_delay_max()->set_seconds(0);

  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin(
      /* eligibility_eval_enabled=*/true, retry_window, retry_window));
  const RetryWindow& actual_retry_window =
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
TEST_P(GrpcFederatedProtocolTest, TestInvalidMaxRetryDelayValueSanitization) {
  google::internal::federatedml::v2::RetryWindow retry_window;
  retry_window.mutable_delay_min()->set_seconds(1234);
  retry_window.mutable_delay_max()->set_seconds(1233);  // less than delay_min

  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin(
      /* eligibility_eval_enabled=*/true, retry_window, retry_window));
  const RetryWindow& actual_retry_window =
      federated_protocol_->GetLatestRetryWindow();
  // The above retry window's invalid max value should be clamped to the min
  // value (minus some errors introduced by the inaccuracy of double
  // multiplication). Note that DoubleEq enforces too precise of bounds, so we
  // use DoubleNear instead.
  EXPECT_THAT(actual_retry_window.delay_min().seconds() +
                  actual_retry_window.delay_min().nanos() / 1000000000.0,
              DoubleNear(1234.0, 0.02));
  EXPECT_THAT(actual_retry_window.delay_max().seconds() +
                  actual_retry_window.delay_max().nanos() / 1000000000.0,
              DoubleNear(1234.0, 0.02));
}

TEST_P(GrpcFederatedProtocolDeathTest, TestCheckinMissingTaskEligibilityInfo) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // A Checkin(...) request with a missing TaskEligibilityInfo should now fail,
  // as the protocol requires us to provide one based on the plan includes in
  // the eligibility eval checkin response payload.
  ASSERT_DEATH(
      {
        auto unused = federated_protocol_->Checkin(
            std::nullopt, mock_task_received_callback_.AsStdFunction());
      },
      _);
}

TEST_P(GrpcFederatedProtocolTest, TestCheckinSendFailsTransientError) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // Make the gRPC stream return an UNAVAILABLE error when the Checkin(...) code
  // tries to send its first message. This should result in the error being
  // returned as the result.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::UnavailableError("foo")));

  auto checkin_result = federated_protocol_->Checkin(
      GetFakeTaskEligibilityInfo(),
      mock_task_received_callback_.AsStdFunction());
  EXPECT_THAT(checkin_result.status(), IsCode(UNAVAILABLE));
  EXPECT_THAT(checkin_result.status().message(), "foo");
  // RetryWindows were already received from the server during the eligibility
  // eval checkin, so we expect to get a 'rejected' retry window.
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest, TestCheckinSendFailsPermanentError) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // Make the gRPC stream return an NOT_FOUND error when the Checkin(...) code
  // tries to send its first message. This should result in the error being
  // returned as the result.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::NotFoundError("foo")));

  auto checkin_result = federated_protocol_->Checkin(
      GetFakeTaskEligibilityInfo(),
      mock_task_received_callback_.AsStdFunction());
  EXPECT_THAT(checkin_result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(checkin_result.status().message(), "foo");
  // Even though RetryWindows were already received from the server during the
  // eligibility eval checkin, we expect a RetryWindow generated based on the
  // *permanent* errors retry delay flag, since NOT_FOUND is marked as a
  // permanent error in the flags, and permanent errors should always result in
  // permanent error windows (regardless of whether a CheckinRequestAck was
  // already received).
  ExpectPermanentErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Tests the case where the blocking Send() call in Checkin is interrupted.
TEST_P(GrpcFederatedProtocolTest, TestCheckinSendInterrupted) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  absl::BlockingCounter counter_should_abort(1);

  // Make Send() block until the counter is decremented.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce([&counter_should_abort](ClientStreamMessage* ignored) {
        counter_should_abort.Wait();
        return absl::OkStatus();
      });
  // Make should_abort return false for the first two calls, and then make it
  // decrement the counter and return true, triggering an abort sequence and
  // unblocking the Send() call we caused to block above.
  EXPECT_CALL(mock_should_abort_, Call())
      .WillOnce(Return(false))
      .WillOnce(Return(false))
      .WillRepeatedly([&counter_should_abort] {
        counter_should_abort.DecrementCount();
        return true;
      });
  // In addition to the Close() call we expect in the test fixture above, expect
  // an additional one (the one that induced the abort).
  EXPECT_CALL(*mock_grpc_bidi_stream_, Close()).Times(1).RetiresOnSaturation();
  EXPECT_CALL(mock_log_manager_,
              LogDiag(ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_GRPC));

  auto checkin_result = federated_protocol_->Checkin(
      GetFakeTaskEligibilityInfo(),
      mock_task_received_callback_.AsStdFunction());
  EXPECT_THAT(checkin_result.status(), IsCode(CANCELLED));
  // RetryWindows were already received from the server during the eligibility
  // eval checkin, so we expect to get a 'rejected' retry window.
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest, TestCheckinRejectionWithTaskEligibilityInfo) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // Expect a checkin request for the next call to Checkin(...).
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeRejectedCheckinResponse()),
                      Return(absl::OkStatus())));

  // The 'task received' callback should not be invoked since no task was given
  // to the client.
  EXPECT_CALL(mock_task_received_callback_, Call(_)).Times(0);

  // Issue the regular checkin.
  auto checkin_result = federated_protocol_->Checkin(
      GetFakeTaskEligibilityInfo(),
      mock_task_received_callback_.AsStdFunction());

  ASSERT_OK(checkin_result.status());
  EXPECT_THAT(*checkin_result, VariantWith<FederatedProtocol::Rejection>(_));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// Tests whether we can issue a Checkin() request correctly without passing a
// TaskEligibilityInfo, in the case that the eligibility eval checkin didn't
// return any eligibility eval task to run.
TEST_P(GrpcFederatedProtocolTest,
       TestCheckinRejectionWithoutTaskEligibilityInfo) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(
      RunSuccessfulEligibilityEvalCheckin(/*eligibility_eval_enabled=*/false));

  // Expect a checkin request for the next call to Checkin(...).
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(GetExpectedCheckinRequest(
                  /*task_eligibility_info=*/std::nullopt,
                  enable_http_resource_support_)))))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeRejectedCheckinResponse()),
                      Return(absl::OkStatus())));

  // The 'task received' callback should not be invoked since no task was given
  // to the client.
  EXPECT_CALL(mock_task_received_callback_, Call(_)).Times(0);

  // Issue the regular checkin, without a TaskEligibilityInfo (since we didn't
  // receive an eligibility eval task to run during eligibility eval checkin).
  auto checkin_result = federated_protocol_->Checkin(
      std::nullopt, mock_task_received_callback_.AsStdFunction());

  ASSERT_OK(checkin_result.status());
  EXPECT_THAT(*checkin_result, VariantWith<FederatedProtocol::Rejection>(_));
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest, TestCheckinAccept) {
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // Once the eligibility eval checkin has succeeded, let's fake some network
  // stats data so that we can verify that it is logged correctly.
  int64_t chunking_layer_bytes_downloaded = 555;
  int64_t chunking_layer_bytes_uploaded = 666;
  EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesReceived())
      .WillRepeatedly(Return(chunking_layer_bytes_downloaded));
  EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesSent())
      .WillRepeatedly(Return(chunking_layer_bytes_uploaded));

  // Note that in this particular test we check that the CheckinRequest is as
  // expected (in all prior tests we just use the '_' matcher, because the
  // request isn't really relevant to the test).
  TaskEligibilityInfo expected_eligibility_info = GetFakeTaskEligibilityInfo();
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(GetExpectedCheckinRequest(
                  expected_eligibility_info, enable_http_resource_support_)))))
      .WillOnce(Return(absl::OkStatus()));

  std::string expected_plan = kPlan;
  std::string expected_checkpoint = kInitCheckpoint;
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeAcceptedCheckinResponse(
                          expected_plan, expected_checkpoint,
                          kFederatedSelectUriTemplate, kExecutionPhaseId,
                          /* use_secure_aggregation=*/true)),
                      Return(absl::OkStatus())));

  // The 'task received' callback should be called even when the resources were
  // available inline.
  EXPECT_CALL(
      mock_task_received_callback_,
      Call(FieldsAre(
          FieldsAre("", ""), kFederatedSelectUriTemplate, kExecutionPhaseId,
          Optional(AllOf(
              Field(&FederatedProtocol::SecAggInfo::expected_number_of_clients,
                    kSecAggExpectedNumberOfClients),
              Field(&FederatedProtocol::SecAggInfo::
                        minimum_clients_in_server_visible_aggregate,
                    kSecAggMinClientsInServerVisibleAggregate))))));

  // Issue the regular checkin.
  auto checkin_result = federated_protocol_->Checkin(
      expected_eligibility_info, mock_task_received_callback_.AsStdFunction());

  ASSERT_OK(checkin_result.status());
  // If HTTP support is enabled then the checkpoint data gets returned in the
  // shape of an absl::Cord (rather than an std::string), regardless of whether
  // it was actually downloaded via HTTP.
  if (enable_http_resource_support_) {
    EXPECT_THAT(
        *checkin_result,
        VariantWith<FederatedProtocol::TaskAssignment>(FieldsAre(
            FieldsAre(absl::Cord(expected_plan),
                      absl::Cord(expected_checkpoint)),
            kFederatedSelectUriTemplate, kExecutionPhaseId,
            Optional(AllOf(
                Field(
                    &FederatedProtocol::SecAggInfo::expected_number_of_clients,
                    kSecAggExpectedNumberOfClients),
                Field(&FederatedProtocol::SecAggInfo::
                          minimum_clients_in_server_visible_aggregate,
                      kSecAggMinClientsInServerVisibleAggregate))))));
  } else {
    EXPECT_THAT(
        *checkin_result,
        VariantWith<FederatedProtocol::TaskAssignment>(FieldsAre(
            FieldsAre(expected_plan, expected_checkpoint),
            kFederatedSelectUriTemplate, kExecutionPhaseId,
            Optional(AllOf(
                Field(
                    &FederatedProtocol::SecAggInfo::expected_number_of_clients,
                    kSecAggExpectedNumberOfClients),
                Field(&FederatedProtocol::SecAggInfo::
                          minimum_clients_in_server_visible_aggregate,
                      kSecAggMinClientsInServerVisibleAggregate))))));
  }
  // The Checkin call is expected to return the accepted retry window from the
  // CheckinRequestAck response to the first eligibility eval request.
  ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest,
       TestCheckinAcceptWithHttpResourcesDownloaded) {
  if (!enable_http_resource_support_) {
    GTEST_SKIP() << "This test only applies the HTTP task resources feature "
                    "is enabled";
    return;
  }
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // Once the eligibility eval checkin has succeeded, let's fake some network
  // stats data so that we can verify that it is logged correctly.
  int64_t chunking_layer_bytes_downloaded = 555;
  int64_t chunking_layer_bytes_uploaded = 666;
  EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesReceived())
      .WillRepeatedly(Return(chunking_layer_bytes_downloaded));
  EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesSent())
      .WillRepeatedly(Return(chunking_layer_bytes_uploaded));

  // Note that in this particular test we check that the CheckinRequest is as
  // expected (in all prior tests we just use the '_' matcher, because the
  // request isn't really relevant to the test).
  TaskEligibilityInfo expected_eligibility_info = GetFakeTaskEligibilityInfo();
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedCheckinRequest(
          expected_eligibility_info, /*enable_http_resource_support=*/true)))))
      .WillOnce(Return(absl::OkStatus()));

  std::string expected_plan = kPlan;
  std::string plan_uri = "https://fake.uri/plan";
  std::string expected_checkpoint = kInitCheckpoint;
  std::string checkpoint_uri = "https://fake.uri/checkpoint";
  ServerStreamMessage fake_checkin_response = GetFakeAcceptedCheckinResponse(
      /*plan=*/"", /*init_checkpoint=*/"", kFederatedSelectUriTemplate,
      kExecutionPhaseId,
      /* use_secure_aggregation=*/true);
  AcceptanceInfo* acceptance_info =
      fake_checkin_response.mutable_checkin_response()
          ->mutable_acceptance_info();
  acceptance_info->mutable_plan_resource()->set_uri(plan_uri);
  acceptance_info->mutable_init_checkpoint_resource()->set_uri(checkpoint_uri);
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(fake_checkin_response),
                      Return(absl::OkStatus())));

  {
    InSequence seq;
    // The 'task received' callback should be called *before* the actual task
    // resources are fetched.
    EXPECT_CALL(
        mock_task_received_callback_,
        Call(FieldsAre(
            FieldsAre("", ""), kFederatedSelectUriTemplate, kExecutionPhaseId,
            Optional(AllOf(
                Field(
                    &FederatedProtocol::SecAggInfo::expected_number_of_clients,
                    kSecAggExpectedNumberOfClients),
                Field(&FederatedProtocol::SecAggInfo::
                          minimum_clients_in_server_visible_aggregate,
                      kSecAggMinClientsInServerVisibleAggregate))))));

    EXPECT_CALL(mock_http_client_,
                PerformSingleRequest(SimpleHttpRequestMatcher(
                    plan_uri, HttpRequest::Method::kGet, _, "")))
        .WillOnce(Return(FakeHttpResponse(200, {}, expected_plan)));

    EXPECT_CALL(mock_http_client_,
                PerformSingleRequest(SimpleHttpRequestMatcher(
                    checkpoint_uri, HttpRequest::Method::kGet, _, "")))
        .WillOnce(Return(FakeHttpResponse(200, {}, expected_checkpoint)));
  }

  {
    InSequence seq;
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_USES_HTTP));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::
                HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_HTTP_FETCH_SUCCEEDED));
  }

  // Issue the regular checkin.
  auto checkin_result = federated_protocol_->Checkin(
      expected_eligibility_info, mock_task_received_callback_.AsStdFunction());

  ASSERT_OK(checkin_result.status());
  EXPECT_THAT(
      *checkin_result,
      VariantWith<FederatedProtocol::TaskAssignment>(FieldsAre(
          FieldsAre(absl::Cord(expected_plan), absl::Cord(expected_checkpoint)),
          kFederatedSelectUriTemplate, kExecutionPhaseId,
          Optional(AllOf(
              Field(&FederatedProtocol::SecAggInfo::expected_number_of_clients,
                    kSecAggExpectedNumberOfClients),
              Field(&FederatedProtocol::SecAggInfo::
                        minimum_clients_in_server_visible_aggregate,
                    kSecAggMinClientsInServerVisibleAggregate))))));
  // The Checkin call is expected to return the accepted retry window from the
  // CheckinRequestAck response to the first eligibility eval request.
  ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest,
       TestCheckinAcceptWithHttpResourcePlanDataFetchFailed) {
  if (!enable_http_resource_support_) {
    GTEST_SKIP() << "This test only applies the HTTP task resources feature "
                    "is enabled";
    return;
  }
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // Note that in this particular test we check that the CheckinRequest is as
  // expected (in all prior tests we just use the '_' matcher, because the
  // request isn't really relevant to the test).
  TaskEligibilityInfo expected_eligibility_info = GetFakeTaskEligibilityInfo();
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedCheckinRequest(
          expected_eligibility_info, /*enable_http_resource_support=*/true)))))
      .WillOnce(Return(absl::OkStatus()));

  std::string expected_plan = kPlan;
  std::string plan_uri = "https://fake.uri/plan";
  std::string expected_checkpoint = kInitCheckpoint;
  ServerStreamMessage fake_checkin_response = GetFakeAcceptedCheckinResponse(
      /*plan=*/"", expected_checkpoint, kFederatedSelectUriTemplate,
      kExecutionPhaseId,
      /* use_secure_aggregation=*/true);
  AcceptanceInfo* acceptance_info =
      fake_checkin_response.mutable_checkin_response()
          ->mutable_acceptance_info();
  acceptance_info->mutable_plan_resource()->set_uri(plan_uri);
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(fake_checkin_response),
                      Return(absl::OkStatus())));

  // Mock a failed plan fetch.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  plan_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(404, {}, "")));

  {
    InSequence seq;
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_USES_HTTP));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::
                HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_HTTP_FETCH_FAILED));
  }

  // Issue the regular checkin.
  auto checkin_result = federated_protocol_->Checkin(
      expected_eligibility_info, mock_task_received_callback_.AsStdFunction());

  EXPECT_THAT(checkin_result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(checkin_result.status().message(),
              HasSubstr("plan fetch failed"));
  EXPECT_THAT(checkin_result.status().message(), HasSubstr("404"));
  // The Checkin call is expected to return the permanent error retry window,
  // since 404 maps to a permanent error.
  ExpectPermanentErrorRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest,
       TestCheckinAcceptWithHttpResourceCheckpointDataFetchFailed) {
  if (!enable_http_resource_support_) {
    GTEST_SKIP() << "This test only applies the HTTP task resources feature "
                    "is enabled";
    return;
  }
  // Issue an eligibility eval checkin first.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());

  // Note that in this particular test we check that the CheckinRequest is as
  // expected (in all prior tests we just use the '_' matcher, because the
  // request isn't really relevant to the test).
  TaskEligibilityInfo expected_eligibility_info = GetFakeTaskEligibilityInfo();
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedCheckinRequest(
          expected_eligibility_info, /*enable_http_resource_support=*/true)))))
      .WillOnce(Return(absl::OkStatus()));

  std::string expected_plan = kPlan;
  std::string expected_checkpoint = kInitCheckpoint;
  std::string checkpoint_uri = "https://fake.uri/checkpoint";
  ServerStreamMessage fake_checkin_response = GetFakeAcceptedCheckinResponse(
      expected_plan, /*init_checkpoint=*/"", kFederatedSelectUriTemplate,
      kExecutionPhaseId,
      /* use_secure_aggregation=*/true);
  AcceptanceInfo* acceptance_info =
      fake_checkin_response.mutable_checkin_response()
          ->mutable_acceptance_info();
  acceptance_info->mutable_init_checkpoint_resource()->set_uri(checkpoint_uri);
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(fake_checkin_response),
                      Return(absl::OkStatus())));

  // Mock a failed checkpoint fetch.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(SimpleHttpRequestMatcher(
                  checkpoint_uri, HttpRequest::Method::kGet, _, "")))
      .WillOnce(Return(FakeHttpResponse(503, {}, "")));

  {
    InSequence seq;
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_USES_HTTP));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::
                HTTP_GRPC_PROTOCOL_REGULAR_TASK_RESOURCE_HTTP_FETCH_FAILED));
  }

  // Issue the regular checkin.
  auto checkin_result = federated_protocol_->Checkin(
      expected_eligibility_info, mock_task_received_callback_.AsStdFunction());

  EXPECT_THAT(checkin_result.status(), IsCode(UNAVAILABLE));
  EXPECT_THAT(checkin_result.status().message(),
              HasSubstr("checkpoint fetch failed"));
  EXPECT_THAT(checkin_result.status().message(), HasSubstr("503"));
  // The Checkin call is expected to return the rejected retry window from the
  // response to the first eligibility eval request.
  ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

TEST_P(GrpcFederatedProtocolTest, TestCheckinAcceptNonSecAgg) {
  // Issue an eligibility eval checkin first, followed by a successful checkin
  // returning a task that doesn't use SecAgg.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  auto checkin_result = RunSuccessfulCheckin(/*use_secure_aggregation=*/false);
  ASSERT_OK(checkin_result.status());
  // If HTTP support is enabled then the checkpoint data gets returned in the
  // shape of an absl::Cord (rather than an std::string), regardless of whether
  // it was actually downloaded via HTTP.
  if (enable_http_resource_support_) {
    EXPECT_THAT(*checkin_result,
                VariantWith<FederatedProtocol::TaskAssignment>(FieldsAre(
                    FieldsAre(absl::Cord(kPlan), absl::Cord(kInitCheckpoint)),
                    kFederatedSelectUriTemplate, kExecutionPhaseId,
                    // There should be no SecAggInfo in the result.
                    Eq(std::nullopt))));
  } else {
    EXPECT_THAT(*checkin_result,
                VariantWith<FederatedProtocol::TaskAssignment>(
                    FieldsAre(FieldsAre(kPlan, kInitCheckpoint),
                              kFederatedSelectUriTemplate, kExecutionPhaseId,
                              // There should be no SecAggInfo in the result.
                              Eq(std::nullopt))));
  }
}

TEST_P(GrpcFederatedProtocolTest, TestReportWithSecAgg) {
  // Issue an eligibility eval checkin first, followed by a successful checkin.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  ASSERT_OK(RunSuccessfulCheckin(/*use_secure_aggregation=*/true));
  // Create a SecAgg like Checkpoint - a combination of a TF checkpoint, and
  // one or more SecAgg quantized aggregands.
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", "");
  results.emplace("some_tensor", QuantizedTensor());

  mock_secagg_runner_ = new StrictMock<MockSecAggRunner>();
  EXPECT_CALL(*mock_secagg_runner_factory_,
              CreateSecAggRunner(_, _, _, _, _, kSecAggExpectedNumberOfClients,
                                 kSecAggMinSurvivingClientsForReconstruction))
      .WillOnce(Return(ByMove(absl::WrapUnique(mock_secagg_runner_))));
  EXPECT_CALL(
      *mock_secagg_runner_,
      Run(UnorderedElementsAre(
          Pair("tensorflow_checkpoint", VariantWith<TFCheckpoint>(IsEmpty())),
          Pair("some_tensor", VariantWith<QuantizedTensor>(
                                  FieldsAre(IsEmpty(), 0, IsEmpty()))))))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
          .WillOnce(
              DoAll(SetArgPointee<0>(GetFakeReportResponse()),
                    Return(absl::OkStatus())));
  EXPECT_OK(federated_protocol_->ReportCompleted(
      std::move(results), absl::ZeroDuration(), std::nullopt));
}

TEST_P(GrpcFederatedProtocolTest, TestReportWithSecAggWithoutTFCheckpoint) {
  // Issue an eligibility eval checkin first, followed by a successful checkin.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  ASSERT_OK(RunSuccessfulCheckin(/*use_secure_aggregation=*/true));

  ComputationResults results;
  results.emplace("some_tensor", QuantizedTensor());

  mock_secagg_runner_ = new StrictMock<MockSecAggRunner>();
  EXPECT_CALL(*mock_secagg_runner_factory_,
              CreateSecAggRunner(_, _, _, _, _, kSecAggExpectedNumberOfClients,
                                 kSecAggMinSurvivingClientsForReconstruction))
      .WillOnce(Return(ByMove(absl::WrapUnique(mock_secagg_runner_))));
  EXPECT_CALL(*mock_secagg_runner_,
              Run(UnorderedElementsAre(
                  Pair("some_tensor", VariantWith<QuantizedTensor>(FieldsAre(
                                          IsEmpty(), 0, IsEmpty()))))))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
          .WillOnce(
              DoAll(SetArgPointee<0>(GetFakeReportResponse()),
                    Return(absl::OkStatus())));
  EXPECT_OK(federated_protocol_->ReportCompleted(
      std::move(results), absl::ZeroDuration(), std::nullopt));
}

// This function tests the Report(...) method's Send code path, ensuring the
// right events are logged / and the right data is transmitted to the server.
TEST_P(GrpcFederatedProtocolTest, TestReportSendFails) {
  // Issue an eligibility eval checkin first, followed by a successful checkin.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  ASSERT_OK(RunSuccessfulCheckin(/*use_secure_aggregation=*/false));

  // 1. Create input for the Report function.
  std::string checkpoint_str;
  const size_t kTFCheckpointSize = 32;
  checkpoint_str.resize(kTFCheckpointSize, 'X');
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", checkpoint_str);

  absl::Duration plan_duration = absl::Milliseconds(1337);

  // 2. The expected message sent to the server by the ReportCompleted()
  // function, as text proto.
  ClientStreamMessage expected_client_stream_message;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      absl::StrCat(
          "report_request {", "  population_name: \"", kPopulationName, "\"",
          "  execution_phase_id: \"", kExecutionPhaseId, "\"", "  report {",
          "    update_checkpoint: \"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\"",
          "    serialized_train_event {", "[type.googleapis.com/",
          "google.internal.federatedml.v2.ClientExecutionStats] {",
          "        duration { seconds: 1 nanos: 337000000 }", "      }",
          "    }", "  }", "}"),
      &expected_client_stream_message));

  // 3. Set up mocks.
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(expected_client_stream_message))))
      .WillOnce(Return(absl::AbortedError("foo")));

  // 4. Test that ReportCompleted() sends the expected message.
  auto report_result = federated_protocol_->ReportCompleted(
      std::move(results), plan_duration, std::nullopt);
  EXPECT_THAT(report_result, IsCode(ABORTED));
  EXPECT_THAT(report_result.message(), HasSubstr("foo"));

  // If we made it to the Report protocol phase, then the client must've been
  // accepted during the Checkin phase first, and so we should receive the
  // "accepted" RetryWindow.
  ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// This function tests the happy path of ReportCompleted() - results get
// reported, server replies with a RetryWindow.
TEST_P(GrpcFederatedProtocolTest, TestPublishReportSuccess) {
  // Issue an eligibility eval checkin first, followed by a successful checkin.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  ASSERT_OK(RunSuccessfulCheckin(/*use_secure_aggregation=*/false));

  // 1. Create input for the Report function.
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", "");

  // 2. Set up mocks.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::OkStatus()));
  ServerStreamMessage response_message;
  response_message.mutable_report_response();
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(
          DoAll(SetArgPointee<0>(response_message), Return(absl::OkStatus())));

  // 3. Test that ReportCompleted() sends the expected message.
  auto report_result = federated_protocol_->ReportCompleted(
      std::move(results), absl::ZeroDuration(), std::nullopt);
  EXPECT_OK(report_result);

  // If we made it to the Report protocol phase, then the client must've been
  // accepted during the Checkin phase first, and so we should receive the
  // "accepted" RetryWindow.
  ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// This function tests the Send code path when PhaseOutcome indicates an
// error. / In that case, no checkpoint, and only the duration stat, should be
// uploaded.
TEST_P(GrpcFederatedProtocolTest, TestPublishReportNotCompleteSendFails) {
  // Issue an eligibility eval checkin first, followed by a successful checkin.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  ASSERT_OK(RunSuccessfulCheckin(/*use_secure_aggregation=*/false));

  // 1. Create input for the Report function.
  absl::Duration plan_duration = absl::Milliseconds(1337);

  // 2. The expected message sent to the server by the ReportNotCompleted()
  // function, as text proto.
  ClientStreamMessage expected_client_stream_message;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      absl::StrCat("report_request {", "  population_name: \"", kPopulationName,
                   "\"", "  execution_phase_id: \"", kExecutionPhaseId, "\"",
                   "  report {", "    serialized_train_event {",
                   "[type.googleapis.com/",
                   "google.internal.federatedml.v2.ClientExecutionStats] {",
                   "        duration { seconds: 1 nanos: 337000000 }",
                   "      }", "    }", "    status_code: INTERNAL", "  }", "}"),
      &expected_client_stream_message));

  // 3. Set up mocks.
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(expected_client_stream_message))))
      .WillOnce(Return(absl::AbortedError("foo")));

  // 4. Test that ReportNotCompleted() sends the expected message.
  auto report_result = federated_protocol_->ReportNotCompleted(
      engine::PhaseOutcome::ERROR, plan_duration, std::nullopt);
  EXPECT_THAT(report_result, IsCode(ABORTED));
  EXPECT_THAT(report_result.message(), HasSubstr("foo"));

  // If we made it to the Report protocol phase, then the client must've been
  // accepted during the Checkin phase first, and so we should receive the
  // "accepted" RetryWindow.
  ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

// This function tests the happy path of ReportCompleted() - results get
// reported, server replies with a RetryWindow.
TEST_P(GrpcFederatedProtocolTest, TestPublishReportSuccessCommitsToOpstats) {
  // Issue an eligibility eval checkin first, followed by a successful checkin.
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin());
  ASSERT_OK(RunSuccessfulCheckin(/*use_secure_aggregation=*/false));

  // 1. Create input for the Report function.
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", "");

  // 2. Set up mocks.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::OkStatus()));
  ServerStreamMessage response_message;
  response_message.mutable_report_response();
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(
          DoAll(SetArgPointee<0>(response_message), Return(absl::OkStatus())));

  // 3. Test that ReportCompleted() sends the expected message.
  auto report_result = federated_protocol_->ReportCompleted(
      std::move(results), absl::ZeroDuration(), std::nullopt);
  EXPECT_OK(report_result);

  // If we made it to the Report protocol phase, then the client must've been
  // accepted during the Checkin phase first, and so we should receive the
  // "accepted" RetryWindow.
  ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
}

}  // anonymous namespace
}  // namespace fcp::client
