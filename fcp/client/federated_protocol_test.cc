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
#include "fcp/client/federated_protocol.h"

#include <memory>

#include "google/protobuf/text_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/platform.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/grpc_bidi_stream.h"
#include "fcp/client/task_environment.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/plan.pb.h"
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
using ::fcp::client::opstats::OperationalStats;
using ::fcp::secagg::AesCtrPrngFactory;
using ::fcp::secagg::FakePrng;
using ::fcp::secagg::InputVectorSpecification;
using ::fcp::secagg::MockSendToServerInterface;
using ::fcp::secagg::MockStateTransitionListener;
using ::fcp::secagg::SecAggClient;
using ::google::internal::federated::plan::ClientOnlyPlan;
using ::google::internal::federatedml::v2::AcceptanceInfo;
using ::google::internal::federatedml::v2::CheckinRequest;
using ::google::internal::federatedml::v2::ClientStreamMessage;
using ::google::internal::federatedml::v2::EligibilityEvalCheckinRequest;
using ::google::internal::federatedml::v2::EligibilityEvalPayload;
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
using ::testing::FieldsAre;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::InSequence;
using ::testing::Lt;
using ::testing::MockFunction;
using ::testing::NiceMock;
using ::testing::Not;
using ::testing::Pointee;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;
using ::testing::VariantWith;

constexpr char kPopulationName[] = "TEST/POPULATION";
constexpr char kTaskName[] = "TEST_TASK";
constexpr char kExecutionPhaseId[] = "TEST/POPULATION/TEST_TASK#1234.ab35";
constexpr char kInitCheckpoint[] = "INIT_CHECKPOINT";
constexpr char kRetryToken[] = "OLD_RETRY_TOKEN";
constexpr char kClientVersion[] = "CLIENT_VERSION";
constexpr char kAttestationMeasurement[] = "ATTESTATION_MEASUREMENT";

class MockGrpcBidiStream : public GrpcBidiStreamInterface {
 public:
  MOCK_METHOD(absl::Status, Send, (ClientStreamMessage*), (override));
  MOCK_METHOD(absl::Status, Receive, (ServerStreamMessage*), (override));
  MOCK_METHOD(void, Close, (), (override));
  MOCK_METHOD(int64_t, ChunkingLayerBytesSent, (), (override));
  MOCK_METHOD(int64_t, ChunkingLayerBytesReceived, (), (override));
};

class MockSecAggClient : public SecAggClient {
 public:
  MockSecAggClient()
      : SecAggClient(2,  // max_clients_expected
                     2,  // minimum_surviving_clients_for_reconstruction
                     {InputVectorSpecification("placeholder", 4, 32)},
                     absl::make_unique<FakePrng>(),
                     absl::make_unique<MockSendToServerInterface>(),
                     absl::make_unique<NiceMock<MockStateTransitionListener>>(),
                     absl::make_unique<AesCtrPrngFactory>()) {}
  MOCK_METHOD(absl::Status, Start, (), (override));
};

ClientOnlyPlan GetFakePlan() {
  ClientOnlyPlan plan;
  plan.set_graph("im_a_tf_graph");
  return plan;
}

ServerStreamMessage GetFakeCheckinRequestAck(
    const RetryWindow& accepted_window, const RetryWindow& rejected_window) {
  ServerStreamMessage checkin_request_ack_message;
  *checkin_request_ack_message.mutable_checkin_request_ack()
       ->mutable_retry_window_if_accepted() = accepted_window;
  *checkin_request_ack_message.mutable_checkin_request_ack()
       ->mutable_retry_window_if_rejected() = rejected_window;
  return checkin_request_ack_message;
}

ServerStreamMessage GetFakeEnabledEligibilityCheckinResponse(
    const ClientOnlyPlan& plan, const std::string& init_checkpoint,
    const std::string& execution_id) {
  ServerStreamMessage checkin_response_message;
  EligibilityEvalPayload* eval_payload =
      checkin_response_message.mutable_eligibility_eval_checkin_response()
          ->mutable_eligibility_eval_payload();
  eval_payload->set_plan(plan.SerializeAsString());
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
    const ClientOnlyPlan& plan, const std::string& init_checkpoint,
    const std::string& phase_id, bool use_secure_aggregation = true) {
  ServerStreamMessage checkin_response_message;
  AcceptanceInfo* acceptance_info =
      checkin_response_message.mutable_checkin_response()
          ->mutable_acceptance_info();
  acceptance_info->set_plan(plan.SerializeAsString());
  acceptance_info->set_execution_phase_id(phase_id);
  acceptance_info->set_init_checkpoint(init_checkpoint);
  if (use_secure_aggregation) {
    acceptance_info->mutable_side_channel_protocol_execution_info()
        ->mutable_secure_aggregation()
        ->set_expected_number_of_clients(1);
    checkin_response_message.mutable_checkin_response()
        ->mutable_protocol_options_response()
        ->mutable_side_channels()
        ->mutable_secure_aggregation()
        ->set_client_variant(secagg::SECAGG_CLIENT_VARIANT_NATIVE_V1);
  }
  return checkin_response_message;
}

ClientStreamMessage GetExpectedEligibilityEvalCheckinRequest() {
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

  return expected_message;
}

// This returns the CheckinRequest gRPC proto we expect each Checkin(...) call
// to result in.
ClientStreamMessage GetExpectedCheckinRequest(
    bool expect_checkin_request_ack = true,
    const absl::optional<TaskEligibilityInfo>& task_eligibility_info =
        absl::nullopt) {
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
      expect_checkin_request_ack);

  if (task_eligibility_info.has_value()) {
    *checkin_request->mutable_task_eligibility_info() = *task_eligibility_info;
  }
  return expected_message;
}

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
  retry_window.mutable_delay_min()->set_seconds(111L);
  retry_window.mutable_delay_max()->set_seconds(222L);
  *retry_window.mutable_retry_token() = "RETRY_TOKEN_ACCEPTED";
  return retry_window;
}

google::internal::federatedml::v2::RetryWindow GetRejectedRetryWindow() {
  google::internal::federatedml::v2::RetryWindow retry_window;
  retry_window.mutable_delay_min()->set_seconds(333L);
  retry_window.mutable_delay_max()->set_seconds(444L);
  *retry_window.mutable_retry_token() = "RETRY_TOKEN_REJECTED";
  return retry_window;
}

void ExpectAcceptedRetryWindow(const RetryWindow& retry_window) {
  // The calculated retry delay must lie within the expected permanent errors
  // retry delay range.
  EXPECT_THAT(retry_window.delay_min().seconds() +
                  retry_window.delay_min().nanos() / 1000000000,
              AllOf(Ge(111L), Lt(222L)));
  EXPECT_THAT(retry_window.delay_max(), EqualsProto(retry_window.delay_min()));
}

void ExpectRejectedRetryWindow(const RetryWindow& retry_window) {
  // The calculated retry delay must lie within the expected permanent errors
  // retry delay range.
  EXPECT_THAT(retry_window.delay_min().seconds() +
                  retry_window.delay_min().nanos() / 1000000000,
              AllOf(Ge(333L), Lt(444L)));
  EXPECT_THAT(retry_window.delay_max(), EqualsProto(retry_window.delay_min()));
}

class FederatedProtocolTest
    // The parameter indicates whether the new behavior w.r.t. retry delays
    // should be enabled.
    : public testing::TestWithParam<bool> {
 public:
  FederatedProtocolTest() {
    // The gRPC stream should always be closed at the end of all tests.
    EXPECT_CALL(*mock_grpc_bidi_stream_, Close());
  }

 protected:
  void SetUp() override {
    EXPECT_CALL(mock_flags_, federated_training_use_new_retry_delay_behavior)
        .WillRepeatedly(Return(GetParam()));
    EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesReceived())
        .WillRepeatedly(Return(0));
    EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesSent())
        .WillRepeatedly(Return(0));
    if (GetParam()) {
      EXPECT_CALL(mock_flags_,
                  federated_training_transient_errors_retry_delay_secs)
          .WillRepeatedly(Return(kTransientErrorsRetryPeriodSecs));
      EXPECT_CALL(
          mock_flags_,
          federated_training_transient_errors_retry_delay_jitter_percent)
          .WillRepeatedly(Return(kTransientErrorsRetryDelayJitterPercent));
      EXPECT_CALL(mock_flags_,
                  federated_training_permanent_errors_retry_delay_secs)
          .WillRepeatedly(Return(kPermanentErrorsRetryPeriodSecs));
      EXPECT_CALL(
          mock_flags_,
          federated_training_permanent_errors_retry_delay_jitter_percent)
          .WillRepeatedly(Return(kPermanentErrorsRetryDelayJitterPercent));
      EXPECT_CALL(mock_flags_, federated_training_permanent_error_codes)
          .WillRepeatedly(Return(std::vector<int32_t>{
              static_cast<int32_t>(absl::StatusCode::kNotFound),
              static_cast<int32_t>(absl::StatusCode::kInvalidArgument),
              static_cast<int32_t>(absl::StatusCode::kUnimplemented)}));
    }

    // We only initialize federated_protocol_ in this SetUp method, rather than
    // in the test's constructor, to ensure that we can set mock flag values
    // before the FederatedProtocol constructor is called. Using std::unique_ptr
    // conveniently allows us to assign the field a new value after construction
    // (which we could not do if the field's type was FederatedProtocol, since
    // it doesn't have copy or move constructors).
    federated_protocol_ = absl::make_unique<FederatedProtocol>(
        &mock_event_publisher_, &mock_log_manager_, &mock_opstats_logger_,
        &mock_flags_,
        // We want to inject mocks stored in unique_ptrs to the
        // class-under-test, hence we transfer ownership via WrapUnique. To
        // write expectations for the mock, we retain the raw pointer to it,
        // which will be valid until FederatedProtocol's d'tor is called.
        absl::WrapUnique(mock_grpc_bidi_stream_),
        absl::WrapUnique(mock_secagg_client_), kPopulationName, kRetryToken,
        kClientVersion, kAttestationMeasurement,
        mock_should_abort_.AsStdFunction(), absl::BitGen(),
        InterruptibleRunner::TimingConfig{
            .polling_period = absl::ZeroDuration(),
            .graceful_shutdown_period = absl::InfiniteDuration(),
            .extended_shutdown_period = absl::InfiniteDuration()});
  }

  // This function runs a successful EligibilityEvalCheckin() that results in an
  // eligibility eval payload being returned by the server. This is a utility
  // function used by Checkin*() tests that depend on a prior, successful
  // execution of EligibilityEvalCheckin().
  // It returns a absl::Status, which the caller should verify is OK using
  // ASSERT_OK.
  absl::Status RunSuccessfulEligibilityEvalCheckin(
      const RetryWindow& accepted_retry_window,
      const RetryWindow& rejected_retry_window) {
    EXPECT_CALL(
        *mock_grpc_bidi_stream_,
        Send(Pointee(EqualsProto(GetExpectedEligibilityEvalCheckinRequest()))))
        .WillOnce(Return(absl::OkStatus()));

    const std::string expected_execution_id = "ELIGIBILITY_EVAL_EXECUTION_ID";
    EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
        .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                            accepted_retry_window, rejected_retry_window)),
                        Return(absl::OkStatus())))
        .WillOnce(
            DoAll(SetArgPointee<0>(GetFakeEnabledEligibilityCheckinResponse(
                      GetFakePlan(), kInitCheckpoint, expected_execution_id)),
                  Return(absl::OkStatus())));

    {
      InSequence seq;
      EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
      EXPECT_CALL(
          mock_opstats_logger_,
          AddEvent(
              OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
      // Network stats should be updated after the one send and two receive
      // operations.
      EXPECT_CALL(
          mock_opstats_logger_,
          SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                          /*chunking_layer_bytes_downloaded=*/0,
                          /*chunking_layer_bytes_uploaded=*/0));
      EXPECT_CALL(
          mock_opstats_logger_,
          SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                          /*chunking_layer_bytes_downloaded=*/0,
                          /*chunking_layer_bytes_uploaded=*/0))
          .Times(2);
      EXPECT_CALL(mock_event_publisher_,
                  SetModelIdentifier(expected_execution_id));
      EXPECT_CALL(mock_event_publisher_,
                  PublishEligibilityEvalPlanReceived(_, _, _));
      EXPECT_CALL(
          mock_opstats_logger_,
          AddEvent(OperationalStats::Event::EVENT_KIND_ELIGIBILITY_ENABLED));
    }

    return federated_protocol_->EligibilityEvalCheckin().status();
  }

  // This function runs a successful Checkin() that results in acceptance by the
  // server. This is a utility function used by Report*() tests that depend on a
  // prior, successful execution of Checkin().
  // It returns a absl::Status, which the caller should verify is OK using
  // ASSERT_OK.
  absl::Status RunSuccessfulCheckin(bool use_secure_aggregation,
                                    bool expect_checkin_request_ack = true,
                                    const absl::optional<TaskEligibilityInfo>&
                                        task_eligibility_info = absl::nullopt) {
    EXPECT_CALL(*mock_grpc_bidi_stream_,
                Send(Pointee(EqualsProto(GetExpectedCheckinRequest(
                    expect_checkin_request_ack, task_eligibility_info)))))
        .WillOnce(Return(absl::OkStatus()));

    {
      InSequence seq;
      if (expect_checkin_request_ack) {
        EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
            .WillOnce(
                DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          GetAcceptedRetryWindow(), GetRejectedRetryWindow())),
                      Return(absl::OkStatus())))
            .RetiresOnSaturation();
      }
      EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
          .WillOnce(DoAll(SetArgPointee<0>(GetFakeAcceptedCheckinResponse(
                              GetFakePlan(), kInitCheckpoint, kExecutionPhaseId,
                              use_secure_aggregation)),
                          Return(absl::OkStatus())))
          .RetiresOnSaturation();
    }

    {
      InSequence seq;
      EXPECT_CALL(mock_event_publisher_, PublishCheckin());
      EXPECT_CALL(
          mock_opstats_logger_,
          AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
      // Network stats should be updated after the one send and two receive
      // operations.
      EXPECT_CALL(
          mock_opstats_logger_,
          SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                          /*chunking_layer_bytes_downloaded=*/0,
                          /*chunking_layer_bytes_uploaded=*/0));
      EXPECT_CALL(
          mock_opstats_logger_,
          SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                          /*chunking_layer_bytes_downloaded=*/0,
                          /*chunking_layer_bytes_uploaded=*/0))
          .Times(2);
      EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(kExecutionPhaseId));
      EXPECT_CALL(mock_event_publisher_, PublishCheckinFinished(_, _, _));
      EXPECT_CALL(mock_opstats_logger_,
                  AddCheckinAcceptedEventWithTaskName(kTaskName));
    }

    return federated_protocol_->Checkin(task_eligibility_info).status();
  }

  // See note in the constructor for why these are pointers.
  StrictMock<MockGrpcBidiStream>* mock_grpc_bidi_stream_ =
      new StrictMock<MockGrpcBidiStream>();
  StrictMock<MockSecAggClient>* mock_secagg_client_ =
      new StrictMock<MockSecAggClient>();

  StrictMock<MockEventPublisher> mock_event_publisher_;
  NiceMock<MockLogManager> mock_log_manager_;
  StrictMock<MockOpStatsLogger> mock_opstats_logger_;
  NiceMock<MockFlags> mock_flags_;
  NiceMock<MockFunction<bool()>> mock_should_abort_;

  // The class under test.
  std::unique_ptr<FederatedProtocol> federated_protocol_;
};

// We create a number of instances of the test suite: one set of instances with
// the new retry delay behavior flag disabled, and once with it enabled. Most
// tests should pass in both configurations, while a handful use GTEST_SKIP() to
// only run conditionally for the specific configuration they're testing.

INSTANTIATE_TEST_SUITE_P(NewVsOldRetryBehavior, FederatedProtocolTest,
                         testing::Values(false, true));

using FederatedProtocolDeathTest = FederatedProtocolTest;
INSTANTIATE_TEST_SUITE_P(NewVsOldRetryBehavior, FederatedProtocolDeathTest,
                         testing::Values(false, true));

TEST_P(FederatedProtocolTest,
       TestTransientErrorRetryWindowDifferentAcrossDifferentInstances) {
  if (!GetParam()) {  // new retry delay behavior
    GTEST_SKIP() << "This test does not apply if the new retry behavior is not "
                    "turned on";
  }
  const RetryWindow& retry_window1 =
      federated_protocol_->GetLatestRetryWindow();
  ExpectTransientErrorRetryWindow(retry_window1);
  federated_protocol_.reset(nullptr);

  mock_grpc_bidi_stream_ = new StrictMock<MockGrpcBidiStream>();
  EXPECT_CALL(*mock_grpc_bidi_stream_, Close());
  mock_secagg_client_ = new StrictMock<MockSecAggClient>();

  // Create a new FederatedProtocol instance. It should not produce the same
  // retry window value as the one we just got. This is a simple correctness
  // check to ensure that the value is at least randomly generated (and that we
  // don't accidentally use the random number generator incorrectly).
  federated_protocol_ = absl::make_unique<FederatedProtocol>(
      &mock_event_publisher_, &mock_log_manager_, &mock_opstats_logger_,
      &mock_flags_, absl::WrapUnique(mock_grpc_bidi_stream_),
      absl::WrapUnique(mock_secagg_client_), kPopulationName, kRetryToken,
      kClientVersion, kAttestationMeasurement,
      mock_should_abort_.AsStdFunction(), absl::BitGen(),
      InterruptibleRunner::TimingConfig{
          .polling_period = absl::ZeroDuration(),
          .graceful_shutdown_period = absl::InfiniteDuration(),
          .extended_shutdown_period = absl::InfiniteDuration()});

  const RetryWindow& retry_window2 =
      federated_protocol_->GetLatestRetryWindow();
  ExpectTransientErrorRetryWindow(retry_window2);

  EXPECT_THAT(retry_window1, Not(EqualsProto(retry_window2)));
}

TEST_P(FederatedProtocolTest,
       TestEligibilityEvalCheckinSendFailsTransientError) {
  // Make the gRPC stream return an UNAVAILABLE error when the
  // EligibilityEvalCheckin(...) code tries to send its first message. This
  // should result in the error being returned as the result.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::UnavailableError("foo")));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(
            OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
  }

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(UNAVAILABLE));
  EXPECT_THAT(eligibility_checkin_result.status().message(), "foo");
  if (GetParam()) {  // new retry delay behavior
    // No RetryWindows were received from the server, so we expect to get a
    // RetryWindow generated based on the transient errors retry delay flag.
    ExpectTransientErrorRetryWindow(
        federated_protocol_->GetLatestRetryWindow());
  } else {
    // No RetryWindows were received from the server, so we expect the latest
    // one to be the default instance.
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(RetryWindow()));
  }
}

TEST_P(FederatedProtocolTest,
       TestEligibilityEvalCheckinSendFailsPermanentError) {
  // Make the gRPC stream return an NOT_FOUND error when the
  // EligibilityEvalCheckin(...) code tries to send its first message. This
  // should result in the error being returned as the result.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::NotFoundError("foo")));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(
            OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
  }

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(eligibility_checkin_result.status().message(), "foo");
  if (GetParam()) {  // new retry delay behavior
    // No RetryWindows were received from the server, so we expect to get a
    // RetryWindow generated based on the *permanent* errors retry delay flag,
    // since NOT_FOUND is marked as a permanent error in the flags.
    ExpectPermanentErrorRetryWindow(
        federated_protocol_->GetLatestRetryWindow());
  } else {
    // No RetryWindows were received from the server, so we expect the latest
    // one to be the default instance.
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(RetryWindow()));
  }
}

// Tests the case where the blocking Send() call in EligibilityEvalCheckin is
// interrupted.
TEST_P(FederatedProtocolTest, TestEligibilityEvalCheckinSendInterrupted) {
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

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(
            OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
    EXPECT_CALL(mock_log_manager_,
                LogDiag(ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_GRPC));
  }

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(CANCELLED));
  if (GetParam()) {  // new retry delay behavior
    // No RetryWindows were received from the server, so we expect to get a
    // RetryWindow generated based on the transient errors retry delay flag.
    ExpectTransientErrorRetryWindow(
        federated_protocol_->GetLatestRetryWindow());
  } else {
    // No RetryWindows were received from the server, so we expect the latest
    // one to be the default instance.
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(RetryWindow()));
  }
}

// If a CheckinRequestAck is requested in the ProtocolOptionsRequest but not
// received, UNIMPLEMENTED should be returned.
TEST_P(FederatedProtocolTest,
       TestEligibilityEvalCheckinMissingCheckinRequestAck) {
  // We immediately return an EligibilityEvalCheckinResponse, rather than
  // returning a CheckinRequestAck first.
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedEligibilityEvalCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeRejectedEligibilityCheckinResponse()),
                Return(absl::OkStatus())));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(
            OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
    // Network stats should be updated after the one send and one receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::
                BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_EXPECTED_BUT_NOT_RECVD));  // NOLINT
  }

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(UNIMPLEMENTED));
  if (GetParam()) {  // new retry delay behavior
    // No RetryWindows were received from the server, so we expect to get a
    // RetryWindow generated based on the *permanent* errors retry delay flag,
    // since UNIMPLEMENTED is marked as a permanent error in the flags.
    ExpectPermanentErrorRetryWindow(
        federated_protocol_->GetLatestRetryWindow());
  } else {
    // No RetryWindows were received from the server, so we expect the latest
    // one to be the default instance.
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(RetryWindow()));
  }
}

TEST_P(FederatedProtocolTest,
       TestEligibilityEvalCheckinWaitForCheckinRequestAckFails) {
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedEligibilityEvalCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));

  // Make the very first Receive() call fail (i.e. the one expecting the
  // CheckinRequestAck).
  std::string expected_message = "foo";
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(Return(absl::AbortedError(expected_message)));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(
            OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
    // Network stats should be updated after the one send operation but not
    // after the unsucessful read operation.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
  }

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(ABORTED));
  EXPECT_THAT(eligibility_checkin_result.status().message(), expected_message);
  if (GetParam()) {  // new retry delay behavior
    // No RetryWindows were received from the server, so we expect to get a
    // RetryWindow generated based on the transient errors retry delay flag.
    ExpectTransientErrorRetryWindow(
        federated_protocol_->GetLatestRetryWindow());
  } else {
    // No RetryWindows were received from the server, so we expect the latest
    // one to be the default instance.
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(RetryWindow()));
  }
}

TEST_P(FederatedProtocolTest,
       TestEligibilityEvalCheckinWaitForCheckinResponseFails) {
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedEligibilityEvalCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));

  // Failed checkins that have received an ack already should return the
  // rejected retry window.
  const RetryWindow expected_retry_window = GetRejectedRetryWindow();
  std::string expected_message = "foo";
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          GetAcceptedRetryWindow(), expected_retry_window)),
                      Return(absl::OkStatus())))
      // Make the second Receive() call fail (i.e. the one expecting the
      // EligibilityEvalCheckinResponse).
      .WillOnce(Return(absl::AbortedError(expected_message)));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(
            OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
    // Network stats should be updated after the one send and one successful
    // receive operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
  }

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(ABORTED));
  EXPECT_THAT(eligibility_checkin_result.status().message(), expected_message);
  if (GetParam()) {  // new retry delay behavior
    ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest, TestEligibilityEvalCheckinRejection) {
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedEligibilityEvalCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));

  const RetryWindow expected_retry_window = GetRejectedRetryWindow();
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          GetAcceptedRetryWindow(), expected_retry_window)),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeRejectedEligibilityCheckinResponse()),
                Return(absl::OkStatus())));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(
            OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
    // Network stats should be updated after the one send and two receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0))
        .Times(2);
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalRejected(_, _, _));
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED));
  }

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(*eligibility_checkin_result,
              VariantWith<FederatedProtocol::Rejection>(_));
  if (GetParam()) {  // new retry delay behavior
    ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest, TestEligibilityEvalCheckinDisabled) {
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedEligibilityEvalCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));

  const RetryWindow expected_retry_window = GetRejectedRetryWindow();
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          GetAcceptedRetryWindow(), expected_retry_window)),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeDisabledEligibilityCheckinResponse()),
                Return(absl::OkStatus())));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(
            OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
    // Network stats should be updated after the one send and two receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0))
        .Times(2);
    EXPECT_CALL(mock_event_publisher_,
                PublishEligibilityEvalNotConfigured(_, _, _));
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(OperationalStats::Event::EVENT_KIND_ELIGIBILITY_DISABLED));
  }

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(*eligibility_checkin_result,
              VariantWith<FederatedProtocol::EligibilityEvalDisabled>(_));
  if (GetParam()) {  // new retry delay behavior
    ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest,
       TestEligibilityEvalCheckinEnabledWithInvalidPlan) {
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedEligibilityEvalCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));

  const RetryWindow expected_retry_window = GetRejectedRetryWindow();

  ServerStreamMessage response_message;
  EligibilityEvalPayload* eligibility_eval_payload =
      response_message.mutable_eligibility_eval_checkin_response()
          ->mutable_eligibility_eval_payload();
  eligibility_eval_payload->set_plan("does_not_parse");
  eligibility_eval_payload->set_init_checkpoint("");
  std::string expected_execution_id = "ELIGIBILITY_EVAL_EXECUTION_ID";
  eligibility_eval_payload->set_execution_id(expected_execution_id);
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          GetAcceptedRetryWindow(), expected_retry_window)),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(response_message), Return(absl::OkStatus())));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(
            OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
    // Network stats should be updated after the one send and two receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(mock_event_publisher_,
                SetModelIdentifier(expected_execution_id));
    EXPECT_CALL(mock_event_publisher_,
                PublishEligibilityEvalPlanReceived(_, _, _));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::
                BACKGROUND_TRAINING_ELIGIBILITY_EVAL_FAILED_CANNOT_PARSE_PLAN));
  }

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  EXPECT_THAT(eligibility_checkin_result.status(), IsCode(INTERNAL));
  if (GetParam()) {  // new retry delay behavior
    ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest, TestEligibilityEvalCheckinEnabled) {
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedEligibilityEvalCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));

  // The EligibilityEvalCheckin(...) method should return the rejected
  // RetryWindow, since after merely completing an eligibility eval checkin the
  // client hasn't actually been accepted to a specific task yet.
  const RetryWindow expected_retry_window = GetRejectedRetryWindow();
  ClientOnlyPlan expected_plan = ClientOnlyPlan::default_instance();
  std::string expected_checkpoint = kInitCheckpoint;
  std::string expected_execution_id = "ELIGIBILITY_EVAL_EXECUTION_ID";
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          GetAcceptedRetryWindow(), expected_retry_window)),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeEnabledEligibilityCheckinResponse(
                    expected_plan, expected_checkpoint, expected_execution_id)),
                Return(absl::OkStatus())));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishEligibilityEvalCheckin());
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(
            OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED));
    // Network stats should be updated after the one send and two receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(mock_event_publisher_,
                SetModelIdentifier(expected_execution_id));
    EXPECT_CALL(mock_event_publisher_,
                PublishEligibilityEvalPlanReceived(_, _, _));
    EXPECT_CALL(
        mock_opstats_logger_,
        AddEvent(OperationalStats::Event::EVENT_KIND_ELIGIBILITY_ENABLED));
  }

  auto eligibility_checkin_result =
      federated_protocol_->EligibilityEvalCheckin();

  ASSERT_OK(eligibility_checkin_result);
  EXPECT_THAT(*eligibility_checkin_result,
              VariantWith<FederatedProtocol::CheckinResultPayload>(FieldsAre(
                  EqualsProto(expected_plan), expected_checkpoint, _, _)));
  if (GetParam()) {  // new retry delay behavior
    ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

// Tests that the protocol correctly sanitizes any invalid values it may have
// received from the server.
TEST_P(FederatedProtocolTest, TestNegativeMinMaxRetryDelayValueSanitization) {
  if (!GetParam()) {  // new retry delay behavior
    GTEST_SKIP() << "This test does not apply if the new retry behavior is not "
                    "turned on";
  }

  google::internal::federatedml::v2::RetryWindow retry_window;
  retry_window.mutable_delay_min()->set_seconds(-1);
  retry_window.mutable_delay_max()->set_seconds(-2);

  // The above retry window's negative min/max values should be clamped to 0.
  google::internal::federatedml::v2::RetryWindow expected_retry_window;
  expected_retry_window.mutable_delay_min()->set_seconds(0);
  expected_retry_window.mutable_delay_max()->set_seconds(0);

  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin(retry_window, retry_window));
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
TEST_P(FederatedProtocolTest, TestInvalidMaxRetryDelayValueSanitization) {
  if (!GetParam()) {  // new retry delay behavior
    GTEST_SKIP() << "This test does not apply if the new retry behavior is not "
                    "turned on";
  }

  google::internal::federatedml::v2::RetryWindow retry_window;
  retry_window.mutable_delay_min()->set_seconds(1234);
  retry_window.mutable_delay_max()->set_seconds(1233);  // less than delay_min

  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin(retry_window, retry_window));
  const RetryWindow& actual_retry_window =
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

TEST_P(FederatedProtocolDeathTest, TestCheckinMissingTaskEligibilityInfo) {
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin(GetAcceptedRetryWindow(),
                                                GetRejectedRetryWindow()));

  // A Checkin(...) request with a missing TaskEligibilityInfo should now fail,
  // as the protocol requires us to provide one based on the plan includes in
  // the eligibility eval checkin response payload.
  ASSERT_DEATH({ auto unused = federated_protocol_->Checkin(absl::nullopt); },
               _);
}

TEST_P(FederatedProtocolTest, TestCheckinSendFailsTransientError) {
  // Make the gRPC stream return an UNAVAILABLE error when the Checkin(...) code
  // tries to send its first message. This should result in the error being
  // returned as the result.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::UnavailableError("foo")));

  {
    InSequence seq;
    // We expect a PublishCheckin() but no PublishCheckinFinished() event, since
    // the checkin fails.
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
  }

  auto checkin_result = federated_protocol_->Checkin(absl::nullopt);
  EXPECT_THAT(checkin_result.status(), IsCode(UNAVAILABLE));
  EXPECT_THAT(checkin_result.status().message(), "foo");
  if (GetParam()) {  // new retry delay behavior
    // No RetryWindows were received from the server, so we expect to get a
    // RetryWindow generated based on the transient errors retry delay flag.
    ExpectTransientErrorRetryWindow(
        federated_protocol_->GetLatestRetryWindow());
  } else {
    // No RetryWindows were received from the server, so we expect the latest
    // one to be the default instance.
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(RetryWindow()));
  }
}

TEST_P(FederatedProtocolTest, TestCheckinSendFailsPermanentError) {
  // Make the gRPC stream return an NOT_FOUND error when the Checkin(...) code
  // tries to send its first message. This should result in the error being
  // returned as the result.
  EXPECT_CALL(*mock_grpc_bidi_stream_, Send(_))
      .WillOnce(Return(absl::NotFoundError("foo")));

  {
    InSequence seq;
    // We expect a PublishCheckin() but no PublishCheckinFinished() event, since
    // the checkin fails.
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
  }

  auto checkin_result = federated_protocol_->Checkin(absl::nullopt);
  EXPECT_THAT(checkin_result.status(), IsCode(NOT_FOUND));
  EXPECT_THAT(checkin_result.status().message(), "foo");
  if (GetParam()) {  // new retry delay behavior
    // No RetryWindows were received from the server, so we expect to get a
    // RetryWindow generated based on the *permanent* errors retry delay flag,
    // since NOT_FOUND is marked as a permanent error in the flags.
    ExpectPermanentErrorRetryWindow(
        federated_protocol_->GetLatestRetryWindow());
  } else {
    // No RetryWindows were received from the server, so we expect the latest
    // one to be the default instance.
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(RetryWindow()));
  }
}

// Tests the case where the blocking Send() call in Checkin is interrupted.
TEST_P(FederatedProtocolTest, TestCheckinSendInterrupted) {
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

  {
    InSequence seq;
    // We expect a PublishCheckin() but no PublishCheckinFinished() event, since
    // the checkin is aborted.
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    EXPECT_CALL(mock_log_manager_,
                LogDiag(ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_GRPC));
  }

  auto checkin_result = federated_protocol_->Checkin(absl::nullopt);
  EXPECT_THAT(checkin_result.status(), IsCode(CANCELLED));
  if (GetParam()) {  // new retry delay behavior
    // No RetryWindows were received from the server, so we expect to get a
    // RetryWindow generated based on the transient errors retry delay flag.
    ExpectTransientErrorRetryWindow(
        federated_protocol_->GetLatestRetryWindow());
  } else {
    // No RetryWindows were received from the server, so we expect the latest
    // one to be the default instance.
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(RetryWindow()));
  }
}

// If a CheckinRequestAck is requested in the ProtocolOptionsRequest but not
// received, UNIMPLEMENTED should be returned.
TEST_P(FederatedProtocolTest, TestCheckinMissingCheckinRequestAck) {
  // We immediately return a CheckinResponse, rather than returning a
  // CheckinRequestAck first.
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(GetExpectedCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeRejectedCheckinResponse()),
                      Return(absl::OkStatus())));

  {
    InSequence seq;
    // We expect a PublishCheckin() but no PublishCheckinFinished() event, since
    // the checkin never actually finishes.
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    // Network stats should be updated after the one send and one receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::
                BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_EXPECTED_BUT_NOT_RECVD));  // NOLINT
  }

  auto checkin_result = federated_protocol_->Checkin(absl::nullopt);

  EXPECT_THAT(checkin_result.status(), IsCode(UNIMPLEMENTED));
  if (GetParam()) {  // new retry delay behavior
    // No RetryWindows were received from the server, so we expect to get a
    // RetryWindow generated based on the *permanent* errors retry delay flag,
    // since UNIMPLEMENTED is marked a permanent error in the flags.
    ExpectPermanentErrorRetryWindow(
        federated_protocol_->GetLatestRetryWindow());
  } else {
    // No RetryWindows were received from the server, so we expect the latest
    // one to be the default instance.
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(RetryWindow()));
  }
}

TEST_P(FederatedProtocolTest, TestCheckinWaitForCheckinRequestAckFails) {
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(GetExpectedCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));

  // Make the very first Receive() call fail (i.e. the one expecting the
  // CheckinRequestAck).
  const std::string expected_message = "foo";
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(Return(absl::AbortedError(expected_message)));

  {
    InSequence seq;
    // We expect a PublishCheckin() but no PublishCheckinFinished() event, since
    // the checkin is aborted.
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    // Network stats should be updated after the one send operation.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
  }

  auto checkin_result = federated_protocol_->Checkin(absl::nullopt);

  EXPECT_THAT(checkin_result.status(), IsCode(ABORTED));
  EXPECT_THAT(checkin_result.status().message(), expected_message);
  if (GetParam()) {  // new retry delay behavior
    // No RetryWindows were received from the server, so we expect to get a
    // RetryWindow generated based on the transient errors retry delay flag.
    ExpectTransientErrorRetryWindow(
        federated_protocol_->GetLatestRetryWindow());
  } else {
    // No RetryWindows were received from the server, so we expect the latest
    // one to be the default instance.
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(RetryWindow()));
  }
}

// Tests the case where the server sends an immediate CheckinRequestAck, but
// the subsequent receiving of the CheckinResponse fails (this happens in
// practice when the server cannot assemble enough devices to start a round,
// or if the device is interrupted while waiting). In that case the protocol
// should return the rejected retry window provided in the CheckinRequestAck,
// and the method itself should return the protocol error. The caller is
// expected to log the protocol error and use the RetryWindow in this case.
TEST_P(FederatedProtocolTest, TestCheckinWaitForCheckinResponseFails) {
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(GetExpectedCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));

  // Failed checkins that have received an ack already should return the
  // rejected retry window.
  const RetryWindow expected_retry_window = GetRejectedRetryWindow();
  const std::string expected_message = "foo";
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          GetAcceptedRetryWindow(), expected_retry_window)),
                      Return(absl::OkStatus())))
      // Make the second Receive() call fail (i.e. the one expecting the
      // CheckinResponse).
      .WillOnce(Return(absl::AbortedError(expected_message)));

  {
    InSequence seq;
    // We expect a PublishCheckin() but no PublishCheckinFinished() event, since
    // the checkin never actually finishes.
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    // Network stats should be updated after the one send and one successful
    // receive operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));
  }

  auto checkin_result = federated_protocol_->Checkin(absl::nullopt);

  EXPECT_THAT(checkin_result.status(), IsCode(ABORTED));
  EXPECT_THAT(checkin_result.status().message(), expected_message);
}

// Tests the case where the client issues a successful eligibility eval
// checkin, and then initiates a regular checkin, but the subsequent receiving
// of the CheckinResponse fails (this happens in practice when the server
// cannot assemble enough devices to start a round, or if the device is
// interrupted while waiting). In that case, the protocol should return the
// rejected retry window provided in the CheckinRequestAck that was returned
// to the initial eligibility eval checkin and the method itself should
// return the protocol error. The caller is expected to log the protocol error
// and use the RetryWindow in this case.
TEST_P(FederatedProtocolTest,
       TestCheckinWaitForCheckinResponseFailsWithEligibilityEval) {
  // Issue an eligibility eval checkin first.
  // Failed checkins that have received an ack already should return the
  // rejected retry window.
  const RetryWindow expected_retry_window = GetRejectedRetryWindow();
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin(GetAcceptedRetryWindow(),
                                                expected_retry_window));

  TaskEligibilityInfo expected_eligibility_info = GetFakeTaskEligibilityInfo();
  // Expect a checkin request for the next call to Checkin(...).
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedCheckinRequest(
          // We don't expect another CheckinRequestAck to be requested, since
          // the eligibility eval checkin already did that.
          /*expect_checkin_request_ack=*/false, expected_eligibility_info)))))
      .WillOnce(Return(absl::OkStatus()));

  // Make the next Receive() call fail (i.e. the one expecting the
  // CheckinResponse).
  const std::string expected_message = "foo";
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(Return(absl::AbortedError(expected_message)));

  {
    InSequence seq;
    // We expect the SetModelIdentifier method to be called before checkin,
    // since the prior eligibility eval task identifier must be cleared.
    EXPECT_CALL(mock_log_manager_, SetModelIdentifier(""));
    EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(""));
    // We expect a PublishCheckin() but no PublishCheckinFinished() event, since
    // the checkin never actually finishes.
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    // Network stats should be updated after the one send operation. Note that
    // the eligibility eval checkin already ran.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
  }

  auto checkin_result = federated_protocol_->Checkin(expected_eligibility_info);

  EXPECT_THAT(checkin_result.status(), IsCode(ABORTED));
  EXPECT_THAT(checkin_result.status().message(), expected_message);
  if (GetParam()) {  // new retry delay behavior
    ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest, TestCheckinRejection) {
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(GetExpectedCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));

  const RetryWindow expected_retry_window = GetRejectedRetryWindow();
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          GetAcceptedRetryWindow(), expected_retry_window)),
                      Return(absl::OkStatus())))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeRejectedCheckinResponse()),
                      Return(absl::OkStatus())));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    // Network stats should be updated after the one send and two receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(mock_log_manager_, SetModelIdentifier(""));
    EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(""));
    EXPECT_CALL(mock_event_publisher_, PublishCheckinFinished(_, _, _));
    EXPECT_CALL(mock_event_publisher_, PublishRejected());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_REJECTED));
  }

  auto checkin_result = federated_protocol_->Checkin(absl::nullopt);

  ASSERT_OK(checkin_result.status());
  EXPECT_THAT(*checkin_result, VariantWith<FederatedProtocol::Rejection>(_));
  if (GetParam()) {  // new retry delay behavior
    ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest,
       TestCheckinRejectionWithPriorEligibilityEvalCheckin) {
  // Issue an eligibility eval checkin first.
  // The final Checkin(...) method should return the rejected RetryWindow, since
  // it results in a rejection.
  const RetryWindow expected_retry_window = GetRejectedRetryWindow();
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin(GetAcceptedRetryWindow(),
                                                expected_retry_window));

  TaskEligibilityInfo expected_eligibility_info = GetFakeTaskEligibilityInfo();
  // Expect a checkin request for the next call to Checkin(...).
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedCheckinRequest(
          // We don't expect another CheckinRequestAck to be requested, since
          // the eligibility eval checkin already did that.
          /*expect_checkin_request_ack=*/false, expected_eligibility_info)))))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeRejectedCheckinResponse()),
                      Return(absl::OkStatus())));

  {
    InSequence seq;
    // We expect the SetModelIdentifier method to be called before checkin,
    // since the prior eligibility eval task identifier must be cleared.
    EXPECT_CALL(mock_log_manager_, SetModelIdentifier(""));
    EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(""));
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    // Network stats should be updated after the one send and one receive
    // operations. Note that the eligibility eval checkin already ran.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0))
        .Times(2);
    EXPECT_CALL(mock_log_manager_, SetModelIdentifier(""));
    EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(""));
    EXPECT_CALL(mock_event_publisher_, PublishCheckinFinished(_, _, _));
    EXPECT_CALL(mock_event_publisher_, PublishRejected());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_REJECTED));
  }

  // Issue the regular checkin.
  auto checkin_result = federated_protocol_->Checkin(expected_eligibility_info);

  ASSERT_OK(checkin_result.status());
  EXPECT_THAT(*checkin_result, VariantWith<FederatedProtocol::Rejection>(_));
  if (GetParam()) {  // new retry delay behavior
    ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest, TestCheckinAcceptWithInvalidPlan) {
  ServerStreamMessage response_message;
  auto acceptance_info =
      response_message.mutable_checkin_response()->mutable_acceptance_info();
  acceptance_info->set_plan("does_not_parse");
  acceptance_info->set_execution_phase_id(kExecutionPhaseId);
  acceptance_info->set_init_checkpoint("");

  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(GetExpectedCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));

  // If the we can't parse the plan then we can't move on to the next protocol
  // phase, so we treat this situation as a rejection, even though the server
  // technically "accepted" the client.
  const RetryWindow expected_retry_window = GetRejectedRetryWindow();
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          GetAcceptedRetryWindow(), expected_retry_window)),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(response_message), Return(absl::OkStatus())));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    // Network stats should be updated after the one send and two receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(mock_log_manager_, SetModelIdentifier(kExecutionPhaseId));
    EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(kExecutionPhaseId));
    EXPECT_CALL(mock_event_publisher_, PublishCheckinFinished(_, _, _));
    EXPECT_CALL(mock_opstats_logger_,
                AddCheckinAcceptedEventWithTaskName(kTaskName));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(ProdDiagCode::BACKGROUND_TRAINING_FAILED_CANNOT_PARSE_PLAN));
  }

  auto checkin_result = federated_protocol_->Checkin(absl::nullopt);

  EXPECT_THAT(checkin_result.status(), IsCode(INTERNAL));
  if (GetParam()) {  // new retry delay behavior
    ExpectRejectedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest, TestCheckinAccept) {
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(GetExpectedCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));
  const RetryWindow expected_retry_window = GetAcceptedRetryWindow();
  const ClientOnlyPlan expected_plan = GetFakePlan();
  const std::string expected_checkpoint = kInitCheckpoint;
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          expected_retry_window, GetRejectedRetryWindow())),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeAcceptedCheckinResponse(
                    expected_plan, expected_checkpoint, kExecutionPhaseId)),
                Return(absl::OkStatus())));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    // Network stats should be updated after the one send and two receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(mock_log_manager_, SetModelIdentifier(kExecutionPhaseId));
    EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(kExecutionPhaseId));
    EXPECT_CALL(mock_event_publisher_, PublishCheckinFinished(_, _, _));
    EXPECT_CALL(mock_opstats_logger_,
                AddCheckinAcceptedEventWithTaskName(kTaskName));
  }

  auto checkin_result = federated_protocol_->Checkin(absl::nullopt);

  ASSERT_OK(checkin_result);
  EXPECT_THAT(
      *checkin_result,
      VariantWith<FederatedProtocol::CheckinResultPayload>(FieldsAre(
          EqualsProto(expected_plan), expected_checkpoint, kTaskName, _)));
  if (GetParam()) {  // new retry delay behavior
    ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest, TestCheckinAcceptLogChunkingLayerBandwidth) {
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(GetExpectedCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));
  int64_t chunking_layer_bytes_downloaded = 555;
  int64_t chunking_layer_bytes_uploaded = 666;
  EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesReceived())
      .WillRepeatedly(Return(chunking_layer_bytes_downloaded));
  EXPECT_CALL(*mock_grpc_bidi_stream_, ChunkingLayerBytesSent())
      .WillRepeatedly(Return(chunking_layer_bytes_uploaded));

  const RetryWindow expected_retry_window = GetAcceptedRetryWindow();
  const ClientOnlyPlan expected_plan = GetFakePlan();
  const std::string expected_checkpoint = kInitCheckpoint;
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          expected_retry_window, GetRejectedRetryWindow())),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeAcceptedCheckinResponse(
                    expected_plan, expected_checkpoint, kExecutionPhaseId)),
                Return(absl::OkStatus())));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    // Network stats should be updated after the one send and two receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        Eq(chunking_layer_bytes_downloaded),
                        Eq(chunking_layer_bytes_uploaded)));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/
                        Eq(chunking_layer_bytes_downloaded),
                        /*chunking_layer_bytes_uploaded=*/
                        Eq(chunking_layer_bytes_uploaded)));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/
                        Eq(chunking_layer_bytes_downloaded),
                        /*chunking_layer_bytes_uploaded=*/
                        Eq(chunking_layer_bytes_uploaded)));
    EXPECT_CALL(mock_log_manager_, SetModelIdentifier(kExecutionPhaseId));
    EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(kExecutionPhaseId));
    EXPECT_CALL(
        mock_event_publisher_,
        PublishCheckinFinished(_, Eq(chunking_layer_bytes_downloaded), _));
    EXPECT_CALL(mock_opstats_logger_,
                AddCheckinAcceptedEventWithTaskName(kTaskName));
  }

  auto checkin_result = federated_protocol_->Checkin(absl::nullopt);

  ASSERT_OK(checkin_result);
  EXPECT_THAT(
      *checkin_result,
      VariantWith<FederatedProtocol::CheckinResultPayload>(FieldsAre(
          EqualsProto(expected_plan), expected_checkpoint, kTaskName, _)));
  if (GetParam()) {  // new retry delay behavior
    ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest,
       TestCheckinAcceptWithPriorEligibilityEvalCheckin) {
  // Issue an eligibility eval checkin first.
  // The eventual Checkin call is expected to return the accepted retry window
  // return in the CheckinRequestAck response to this eligibility eval request.
  const RetryWindow expected_retry_window = GetAcceptedRetryWindow();
  ASSERT_OK(RunSuccessfulEligibilityEvalCheckin(expected_retry_window,
                                                GetRejectedRetryWindow()));

  TaskEligibilityInfo expected_eligibility_info = GetFakeTaskEligibilityInfo();
  EXPECT_CALL(
      *mock_grpc_bidi_stream_,
      Send(Pointee(EqualsProto(GetExpectedCheckinRequest(
          // We don't expect another CheckinRequestAck to be requested, since
          // the eligibility eval checkin already did that.
          /*expect_checkin_request_ack=*/false, expected_eligibility_info)))))
      .WillOnce(Return(absl::OkStatus()));

  ClientOnlyPlan expected_plan = GetFakePlan();
  std::string expected_checkpoint = kInitCheckpoint;
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeAcceptedCheckinResponse(
                    expected_plan, expected_checkpoint, kExecutionPhaseId)),
                Return(absl::OkStatus())));

  {
    InSequence seq;
    // We expect the SetModelIdentifier method to be called before checkin,
    // since the prior eligibility eval task identifier must be cleared.
    EXPECT_CALL(mock_log_manager_, SetModelIdentifier(""));
    EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(""));
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    // Network stats should be updated after the one send and one receive
    // operations. Note that the eligibility eval checkin already ran.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0))
        .Times(2);
    EXPECT_CALL(mock_log_manager_, SetModelIdentifier(kExecutionPhaseId));
    EXPECT_CALL(mock_event_publisher_, SetModelIdentifier(kExecutionPhaseId));
    EXPECT_CALL(mock_event_publisher_, PublishCheckinFinished(_, _, _));
    EXPECT_CALL(mock_opstats_logger_,
                AddCheckinAcceptedEventWithTaskName(kTaskName));
  }

  // Issue the regular checkin.
  auto checkin_result = federated_protocol_->Checkin(expected_eligibility_info);

  ASSERT_OK(checkin_result.status());
  EXPECT_THAT(
      *checkin_result,
      VariantWith<FederatedProtocol::CheckinResultPayload>(FieldsAre(
          EqualsProto(expected_plan), expected_checkpoint, kTaskName, _)));
  if (GetParam()) {  // new retry delay behavior
    ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest, TestCheckinAcceptUnparseableExecutionPhaseId) {
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(GetExpectedCheckinRequest()))))
      .WillOnce(Return(absl::OkStatus()));
  const RetryWindow expected_retry_window = GetAcceptedRetryWindow();
  const ClientOnlyPlan expected_plan = GetFakePlan();
  const std::string expected_checkpoint = kInitCheckpoint;
  const std::string unparseable_phase_id = "unparseable_phase_id";
  EXPECT_CALL(*mock_grpc_bidi_stream_, Receive(_))
      .WillOnce(DoAll(SetArgPointee<0>(GetFakeCheckinRequestAck(
                          expected_retry_window, GetRejectedRetryWindow())),
                      Return(absl::OkStatus())))
      .WillOnce(
          DoAll(SetArgPointee<0>(GetFakeAcceptedCheckinResponse(
                    expected_plan, expected_checkpoint, unparseable_phase_id)),
                Return(absl::OkStatus())));

  {
    InSequence seq;
    EXPECT_CALL(mock_event_publisher_, PublishCheckin());
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED));
    // Network stats should be updated after the one send and two receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/0, /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(
        mock_log_manager_,
        LogDiag(
            ProdDiagCode::BACKGROUND_TRAINING_CHECKIN_REQUEST_ACK_RECEIVED));
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0));
    EXPECT_CALL(mock_log_manager_, SetModelIdentifier(unparseable_phase_id));
    EXPECT_CALL(mock_event_publisher_,
                SetModelIdentifier(unparseable_phase_id));
    EXPECT_CALL(mock_event_publisher_, PublishCheckinFinished(_, _, _));
    EXPECT_CALL(mock_log_manager_,
                LogDiag(ProdDiagCode::OPSTATS_TASK_NAME_EXTRACTION_FAILED));
    EXPECT_CALL(mock_opstats_logger_,
                AddCheckinAcceptedEventWithTaskName(unparseable_phase_id));
  }

  auto checkin_result = federated_protocol_->Checkin(absl::nullopt);

  ASSERT_OK(checkin_result);
  EXPECT_THAT(*checkin_result,
              VariantWith<FederatedProtocol::CheckinResultPayload>(FieldsAre(
                  EqualsProto(expected_plan), expected_checkpoint, _, _)));
  if (GetParam()) {  // new retry delay behavior
    ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(expected_retry_window));
  }
}

TEST_P(FederatedProtocolTest, TestReportWithSecAggReachesSecAggClientStart) {
  ASSERT_OK(RunSuccessfulCheckin(/*use_secure_aggregation=*/true));

  // Create a SecAgg like Checkpoint - a combination of a TF checkpoint, and
  // one or more SecAgg quantized aggregands.
  ComputationResults results;
  results.emplace("tensorflow_checkpoint", "");
  results.emplace("some_tensor", QuantizedTensor{{}, 0, {}});

  // Because the SecAgg client library is hard to fake or mock, we just test
  // whether the Report(...) call ends up initiating the SecAgg protocol, at
  // which point we stop the test by having the SecAgg client return an error.
  // This way we at least cover the first part of the SecAgg-related code path
  // in Report(...).
  EXPECT_CALL(*mock_secagg_client_, Start())
      .WillOnce(Return(absl::UnimplementedError("foo")));
  auto report_result = federated_protocol_->ReportCompleted(
      std::move(results), {}, absl::ZeroDuration());
  EXPECT_THAT(report_result, IsCode(UNIMPLEMENTED));
  EXPECT_THAT(report_result.message(), "foo");
}

TEST_P(FederatedProtocolTest,
       TestReportWithSecAggWithoutTFCheckpointReachesSecAggClientStart) {
  ASSERT_OK(RunSuccessfulCheckin(/*use_secure_aggregation=*/true));

  // Similar to the above one, this tests whether the Report(...) call ends up
  // initiating the SecAgg protocol and then stops the test. This particular
  // test checks whether this happens correctly, even if the ComputationResults
  // are empty.
  ComputationResults results;
  EXPECT_CALL(*mock_secagg_client_, Start())
      .WillOnce(Return(absl::UnimplementedError("foo")));
  auto report_result = federated_protocol_->ReportCompleted(
      std::move(results), {}, absl::ZeroDuration());
  EXPECT_THAT(report_result, IsCode(UNIMPLEMENTED));
  EXPECT_THAT(report_result.message(), "foo");
}

// This function tests the Report(...) method's Send code path, ensuring the
// right events are logged / and the right data is transmitted to the server.
TEST_P(FederatedProtocolTest, TestReportSendFails) {
  ASSERT_OK(RunSuccessfulCheckin(/*use_secure_aggregation=*/false));

  // 1. Create input for the Report function.
  std::vector<std::pair<std::string, double>> stats{{"a", 1.0}, {"b", -2.1}};
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
          "        training_stat { stat_name: \"a\" stat_value: 1 }",
          "        training_stat { stat_name: \"b\" stat_value: -2.1 }",
          "        duration { seconds: 1 nanos: 337000000 }", "      }",
          "    }", "  }", "}"),
      &expected_client_stream_message));

  // 3. Set up mocks.
  EXPECT_CALL(*mock_grpc_bidi_stream_,
              Send(Pointee(EqualsProto(expected_client_stream_message))))
      .WillOnce(Return(absl::AbortedError("foo")));
  {
    InSequence seq;
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED));
    EXPECT_CALL(
        mock_event_publisher_,
        PublishReportStarted(expected_client_stream_message.ByteSizeLong()));
  }

  // 4. Test that ReportCompleted() sends the expected message.
  auto report_result = federated_protocol_->ReportCompleted(
      std::move(results), stats, plan_duration);
  EXPECT_THAT(report_result, IsCode(ABORTED));
  EXPECT_THAT(report_result.message(), HasSubstr("foo"));

  // If we made it to the Report protocol phase, then the client must've been
  // accepted during the Checkin phase first, and so we should receive the
  // "accepted" RetryWindow.
  if (GetParam()) {  // new retry delay behavior
    ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(GetAcceptedRetryWindow()));
  }
}

// This function tests the happy path of ReportCompleted() - results get
// reported, server replies with a RetryWindow.
TEST_P(FederatedProtocolTest, TestPublishReportSuccess) {
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
  {
    InSequence seq;
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED));
    EXPECT_CALL(mock_event_publisher_, PublishReportStarted(_));
    // Network stats should be updated after the one send and one receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0))
        .Times(2);
    EXPECT_CALL(mock_event_publisher_, PublishReportFinished(_, _, _));
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_FINISHED));
  }

  // 3. Test that ReportCompleted() sends the expected message.
  auto report_result = federated_protocol_->ReportCompleted(
      std::move(results), {}, absl::ZeroDuration());
  EXPECT_OK(report_result);

  // If we made it to the Report protocol phase, then the client must've been
  // accepted during the Checkin phase first, and so we should receive the
  // "accepted" RetryWindow.
  if (GetParam()) {  // new retry delay behavior
    ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(GetAcceptedRetryWindow()));
  }
}

// This function tests the Send code path when PhaseOutcome indicates an
// error. / In that case, no checkpoint, and only the duration stat, should be
// uploaded.
TEST_P(FederatedProtocolTest, TestPublishReportNotCompleteSendFails) {
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
  {
    InSequence seq;
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED));
    EXPECT_CALL(
        mock_event_publisher_,
        PublishReportStarted(expected_client_stream_message.ByteSizeLong()));
  }

  // 4. Test that ReportNotCompleted() sends the expected message.
  auto report_result = federated_protocol_->ReportNotCompleted(
      engine::PhaseOutcome::ERROR, plan_duration);
  EXPECT_THAT(report_result, IsCode(ABORTED));
  EXPECT_THAT(report_result.message(), HasSubstr("foo"));

  // If we made it to the Report protocol phase, then the client must've been
  // accepted during the Checkin phase first, and so we should receive the
  // "accepted" RetryWindow.
  if (GetParam()) {  // new retry delay behavior
    ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(GetAcceptedRetryWindow()));
  }
}

// This function tests the happy path of ReportCompleted() - results get
// reported, server replies with a RetryWindow.
TEST_P(FederatedProtocolTest, TestPublishReportSuccessCommitsToOpstats) {
  EXPECT_CALL(mock_flags_, commit_opstats_on_upload_started)
      .WillRepeatedly(Return(true));
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
  {
    InSequence seq;
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED));
    EXPECT_CALL(mock_opstats_logger_, CommitToStorage())
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(mock_event_publisher_, PublishReportStarted(_));
    // Network stats should be updated after the one send and one receive
    // operations.
    EXPECT_CALL(
        mock_opstats_logger_,
        SetNetworkStats(/*bytes_downloaded=*/Gt(0), /*bytes_uploaded=*/Gt(0),
                        /*chunking_layer_bytes_downloaded=*/0,
                        /*chunking_layer_bytes_uploaded=*/0))
        .Times(2);
    EXPECT_CALL(mock_event_publisher_, PublishReportFinished(_, _, _));
    EXPECT_CALL(mock_opstats_logger_,
                AddEvent(OperationalStats::Event::EVENT_KIND_UPLOAD_FINISHED));
  }

  // 3. Test that ReportCompleted() sends the expected message.
  auto report_result = federated_protocol_->ReportCompleted(
      std::move(results), {}, absl::ZeroDuration());
  EXPECT_OK(report_result);

  // If we made it to the Report protocol phase, then the client must've been
  // accepted during the Checkin phase first, and so we should receive the
  // "accepted" RetryWindow.
  if (GetParam()) {  // new retry delay behavior
    ExpectAcceptedRetryWindow(federated_protocol_->GetLatestRetryWindow());
  } else {
    EXPECT_THAT(federated_protocol_->GetLatestRetryWindow(),
                EqualsProto(GetAcceptedRetryWindow()));
  }
}

}  // anonymous namespace
}  // namespace fcp::client
