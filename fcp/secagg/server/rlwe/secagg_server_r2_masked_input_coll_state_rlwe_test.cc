/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/str_cat.h"
#include "fcp/base/monitoring.h"
#include "fcp/secagg/server/rlwe/rlwe_secagg_server_protocol_impl.h"
#include "fcp/secagg/server/secagg_server_r2_masked_input_coll_state.h"
#include "fcp/secagg/server/secagg_server_r3_unmasking_state.h"
#include "fcp/secagg/server/secagg_server_state.h"
#include "fcp/secagg/server/secret_sharing_graph_factory.h"
#include "fcp/secagg/server/send_to_clients_interface.h"
#include "fcp/secagg/shared/aes_ctr_prng_factory.h"
#include "fcp/secagg/shared/compute_session_id.h"
#include "fcp/secagg/shared/ecdh_key_agreement.h"
#include "fcp/secagg/shared/ecdh_keys.h"
#include "fcp/secagg/shared/rlwe/input_vector_rlwe_specification.h"
#include "fcp/secagg/shared/secagg_messages.pb.h"
#include "fcp/secagg/shared/shamir_secret_sharing.h"
#include "fcp/secagg/testing/ecdh_pregenerated_test_keys.h"
#include "fcp/secagg/testing/fake_prng.h"
#include "fcp/secagg/testing/server/mock_secagg_server_metrics_listener.h"
#include "fcp/secagg/testing/server/mock_send_to_clients_interface.h"
#include "fcp/testing/testing.h"
#include "fcp/tracing/test_tracing_recorder.h"

namespace fcp {
namespace secagg {
namespace {

using ::testing::_;
using ::testing::Eq;
using ::testing::IsFalse;
using ::testing::IsTrue;

// Default test session_id.
SessionId session_id = {"session id number, 32 bytes long"};
const int kRlweDegree = rlwe::kDegreeBound29;
const unsigned char kRlweTestSeed[] = "0123456789abcdef0123456789abcdef";

std::unique_ptr<RlweSecAggServerProtocolImpl>
CreateRlweSecAggServerProtocolImpl(
    int minimum_number_of_clients_to_proceed, int total_number_of_clients,
    MockSendToClientsInterface* sender,
    MockSecAggServerMetricsListener* metrics_listener = nullptr) {
  auto input_vector_specs = std::vector<InputVectorRlweSpecification>();
  input_vector_specs.push_back(InputVectorRlweSpecification(
      "foobar", kRlweDegree, kRlweDegree, rlwe::kModulus29,
      rlwe::kLogDegreeBound29, 32));
  SecretSharingGraphFactory factory;
  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus29);
  FCP_CHECK(modulus_params_status.ok());
  auto modulus_params = std::move(modulus_params_status.value());
  AesKey prng_seed(kRlweTestSeed, 32);
  auto impl = std::make_unique<RlweSecAggServerProtocolImpl>(
      factory.CreateCompleteGraph(total_number_of_clients,
                                  minimum_number_of_clients_to_proceed),
      minimum_number_of_clients_to_proceed, input_vector_specs,
      std::unique_ptr<MockSecAggServerMetricsListener>(metrics_listener),
      std::make_unique<AesCtrPrngFactory>(), sender,
      nullptr,  // prng_runner
      std::move(modulus_params),
      std::vector<ClientStatus>(total_number_of_clients,
                                ClientStatus::SHARE_KEYS_RECEIVED),
      prng_seed, kRlweDegree);
  impl->set_session_id(std::make_unique<SessionId>(session_id));
  EcdhPregeneratedTestKeys ecdh_keys;
  for (int i = 0; i < total_number_of_clients; ++i) {
    impl->SetPairwisePublicKeys(i, ecdh_keys.GetPublicKey(i));
  }

  return impl;
}

TEST(SecaggServerR2MaskedInputCollStateRlweTest, IsAbortedReturnsFalse) {
  auto sender = std::make_shared<MockSendToClientsInterface>();

  SecAggServerR2MaskedInputCollState state(
      CreateRlweSecAggServerProtocolImpl(3, 4, sender.get()),
      0,  // number_of_clients_failed_after_sending_masked_input
      0,  // number_of_clients_failed_before_sending_masked_input
      0   // number_of_clients_terminated_without_unmasking
  );

  EXPECT_THAT(state.IsAborted(), IsFalse());
}

TEST(SecaggServerR2MaskedInputCollStateRlweTest,
     IsCompletedSuccessfullyReturnsFalse) {
  auto sender = std::make_shared<MockSendToClientsInterface>();

  SecAggServerR2MaskedInputCollState state(
      CreateRlweSecAggServerProtocolImpl(3, 4, sender.get()),
      0,  // number_of_clients_failed_after_sending_masked_input
      0,  // number_of_clients_failed_before_sending_masked_input
      0   // number_of_clients_terminated_without_unmasking
  );

  EXPECT_THAT(state.IsCompletedSuccessfully(), IsFalse());
}

TEST(SecaggServerR2MaskedInputCollStateRlweTest,
     ErrorMessageRaisesErrorStatus) {
  auto sender = std::make_shared<MockSendToClientsInterface>();

  SecAggServerR2MaskedInputCollState state(
      CreateRlweSecAggServerProtocolImpl(3, 4, sender.get()),
      0,  // number_of_clients_failed_after_sending_masked_input
      0,  // number_of_clients_failed_before_sending_masked_input
      0   // number_of_clients_terminated_without_unmasking
  );

  EXPECT_THAT(state.ErrorMessage().ok(), IsFalse());
}

TEST(SecaggServerR2MaskedInputCollStateRlweTest, ResultRaisesErrorStatus) {
  auto sender = std::make_shared<MockSendToClientsInterface>();

  SecAggServerR2MaskedInputCollState state(
      CreateRlweSecAggServerProtocolImpl(3, 4, sender.get()),
      0,  // number_of_clients_failed_after_sending_masked_input
      0,  // number_of_clients_failed_before_sending_masked_input
      0   // number_of_clients_terminated_without_unmasking
  );

  EXPECT_THAT(state.Result().ok(), IsFalse());
}

TEST(SecaggServerR2MaskedInputCollStateRlweTest,
     AbortReturnsValidStateAndNotifiesClients) {
  TestTracingRecorder tracing_recorder;
  MockSecAggServerMetricsListener* metrics =
      new MockSecAggServerMetricsListener();
  auto sender = std::make_shared<MockSendToClientsInterface>();

  SecAggServerR2MaskedInputCollState state(
      CreateRlweSecAggServerProtocolImpl(3, 4, sender.get(), metrics),
      0,  // number_of_clients_failed_after_sending_masked_input
      0,  // number_of_clients_failed_before_sending_masked_input
      0   // number_of_clients_terminated_without_unmasking
  );

  ServerToClientWrapperMessage abort_message;
  abort_message.mutable_abort()->set_early_success(false);
  abort_message.mutable_abort()->set_diagnostic_info("test abort reason");

  EXPECT_CALL(*metrics,
              ProtocolOutcomes(Eq(SecAggServerOutcome::UNHANDLED_ERROR)));
  EXPECT_CALL(*sender, Send(_, _)).Times(0);
  for (int i = 0; i < 4; ++i) {
    EXPECT_CALL(*sender, Send(i, EqualsProto(abort_message))).Times(1);
  }
  auto next_state =
      state.Abort("test abort reason", SecAggServerOutcome::UNHANDLED_ERROR);

  ASSERT_THAT(next_state->State(), Eq(SecAggServerStateKind::ABORTED));
  ASSERT_THAT(next_state->ErrorMessage(), absl_testing::IsOk());
  EXPECT_THAT(next_state->ErrorMessage().value(), Eq("test abort reason"));
  EXPECT_THAT(tracing_recorder.FindAllEvents<BroadcastMessageSent>(),
              ElementsAre(IsEvent<BroadcastMessageSent>(
                  Eq(ServerToClientMessageType_Abort),
                  Eq(abort_message.ByteSizeLong()))));
}

TEST(SecaggServerR2MaskedInputCollStateRlweTest,
     RlweVersionWorksWithoutAborts) {
  // In this test, all clients send in their valid masked inputs, and then the
  // server proceeds to the next state.
  TestTracingRecorder tracing_recorder;
  auto sender = std::make_shared<MockSendToClientsInterface>();

  auto impl = CreateRlweSecAggServerProtocolImpl(3, 4, sender.get());
  auto impl_ptr = impl.get();

  SecAggServerR2MaskedInputCollState state(
      std::move(impl),
      0,  // number_of_clients_failed_after_sending_masked_input
      0,  // number_of_clients_failed_before_sending_masked_input
      0   // number_of_clients_terminated_without_unmasking
  );

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status, absl_testing::IsOk());
  auto modulus_params = std::move(modulus_params_status.value());

  for (int i = 0; i < 5; ++i) {
    EXPECT_THAT(state.NeedsToAbort(), IsFalse());
    EXPECT_THAT(state.NumberOfAliveClients(), Eq(4));
    EXPECT_THAT(state.NumberOfClientsReadyForNextRound(), Eq(i));
    EXPECT_THAT(state.NumberOfMessagesReceivedInThisRound(), Eq(i));
    EXPECT_THAT(state.NumberOfPendingClients(), Eq(4 - i));
    if (i < 3) {
      EXPECT_THAT(state.MinimumMessagesNeededForNextRound(), Eq(3 - i));
      EXPECT_THAT(state.ReadyForNextRound(), IsFalse());
    } else {
      EXPECT_THAT(state.MinimumMessagesNeededForNextRound(), Eq(0));
      EXPECT_THAT(state.ReadyForNextRound(), IsTrue());
    }
    if (i < 4) {
      // Have client send a vector of the correct size to the server
      ClientToServerWrapperMessage client_message;
      MaskedInputVector encoded_vector;

      std::vector<internal::rlwe_mont_int> raw_uintm_vector;
      for (int j = 0; j < kRlweDegree; j++) {
        auto value =
            internal::rlwe_mont_int::ImportInt(i + 1, modulus_params.get());
        ASSERT_THAT(value, absl_testing::IsOk());
        raw_uintm_vector.emplace_back(std::move(value.value()));
      }
      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>> raw_vector = {
          rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector)};
      SecAggRlweVector encrypted_vector(raw_vector, modulus_params.get(), 5,
                                        rlwe::kLogDegreeBound29);

      for (const auto& serialized : encrypted_vector.GetAsSerializedVector()) {
        encoded_vector.add_extra_data()->PackFrom(serialized);
      }
      (*client_message.mutable_masked_input_response()
            ->mutable_vectors())["foobar"] = encoded_vector;
      ASSERT_THAT(
          state.HandleMessage(i, std::make_unique<ClientToServerWrapperMessage>(
                                     client_message)),
          absl_testing::IsOk());
      EXPECT_THAT(state.ReadyForNextRound(), Eq(i >= 2));
    }
  }

  std::vector<internal::rlwe_mont_int> raw_uintm_vector;
  for (int j = 0; j < kRlweDegree; j++) {
    auto value = internal::rlwe_mont_int::ImportInt(10, modulus_params.get());
    ASSERT_THAT(value, absl_testing::IsOk());
    raw_uintm_vector.emplace_back(std::move(value.value()));
  }
  std::vector<rlwe::Polynomial<internal::rlwe_mont_int>> expected_sum = {
      rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector)};

  ServerToClientWrapperMessage server_message;
  server_message.mutable_unmasking_request()
      ->mutable_dead_3_client_ids()
      ->Clear();  // Just to set it to an empty vector
  EXPECT_CALL(*sender, Send(_, _)).Times(0);
  for (int i = 0; i < 4; ++i) {
    EXPECT_CALL(*sender, Send(i, EqualsProto(server_message))).Times(1);
  }

  auto next_state = state.ProceedToNextRound();
  ASSERT_THAT(next_state, absl_testing::IsOk());
  EXPECT_THAT(next_state.value()->State(),
              Eq(SecAggServerStateKind::R3_UNMASKING));
  EXPECT_THAT(
      next_state.value()->NumberOfClientsFailedAfterSendingMaskedInput(),
      Eq(0));
  EXPECT_THAT(
      next_state.value()->NumberOfClientsFailedBeforeSendingMaskedInput(),
      Eq(0));
  EXPECT_THAT(next_state.value()->NumberOfClientsTerminatedWithoutUnmasking(),
              Eq(0));
  EXPECT_THAT(tracing_recorder.FindAllEvents<IndividualMessageSent>(),
              ElementsAre(IsEvent<IndividualMessageSent>(
                              0, Eq(ServerToClientMessageType_UnmaskingRequest),
                              Eq(server_message.ByteSizeLong())),
                          IsEvent<IndividualMessageSent>(
                              1, Eq(ServerToClientMessageType_UnmaskingRequest),
                              Eq(server_message.ByteSizeLong())),
                          IsEvent<IndividualMessageSent>(
                              2, Eq(ServerToClientMessageType_UnmaskingRequest),
                              Eq(server_message.ByteSizeLong())),
                          IsEvent<IndividualMessageSent>(
                              3, Eq(ServerToClientMessageType_UnmaskingRequest),
                              Eq(server_message.ByteSizeLong()))));
  EXPECT_THAT(impl_ptr->rlwe_encrypted_input_vector()
                  ->at("foobar")
                  .GetAsPolynomialVector()
                  .value(),
              Eq(expected_sum));
}

TEST(SecaggServerR2MaskedInputCollStateTest,
     StateProceedsCorrectlyWithOneClientSendingInvalidRlwePolynomial) {
  // In this test, client 0 sends an input with a polynomial of the wrong
  // degree, so it is aborted. The rest of the round goes normally.
  auto sender = std::make_shared<MockSendToClientsInterface>();

  SecAggServerR2MaskedInputCollState state(
      CreateRlweSecAggServerProtocolImpl(3, 4, sender.get()),
      0,  // number_of_clients_failed_after_sending_masked_input
      0,  // number_of_clients_failed_before_sending_masked_input
      0   // number_of_clients_terminated_without_unmasking
  );

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status, absl_testing::IsOk());
  auto modulus_params = std::move(modulus_params_status.value());

  ServerToClientWrapperMessage server_message;
  // TODO(team): 1 -> 0 below, once backwards compatibility not needed.
  server_message.mutable_unmasking_request()->add_dead_3_client_ids(1);
  ServerToClientWrapperMessage abort_message;
  abort_message.mutable_abort()->set_early_success(false);
  abort_message.mutable_abort()->set_diagnostic_info(
      "Masked input does not match input vector specification - polynomial has "
      "wrong degree.");

  EXPECT_CALL(*sender, Send(_, _)).Times(0);
  for (int i = 1; i < 4; ++i) {
    EXPECT_CALL(*sender, Send(i, EqualsProto(server_message))).Times(1);
  }
  EXPECT_CALL(*sender, Send(0, EqualsProto(abort_message))).Times(1);

  // Have client 0 send an invalid message.
  ClientToServerWrapperMessage invalid_message;
  MaskedInputVector encoded_vector;
  std::vector<internal::rlwe_mont_int> raw_uintm_vector;
  // Invalid because the polynomial degree is wrong!
  for (int j = 0; j < kRlweDegree - 1; j++) {
    auto value = internal::rlwe_mont_int::ImportInt(0xFF, modulus_params.get());
    ASSERT_THAT(value, absl_testing::IsOk());
    raw_uintm_vector.emplace_back(std::move(value.value()));
  }
  std::vector<rlwe::Polynomial<internal::rlwe_mont_int>> raw_vector = {
      rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector)};
  SecAggRlweVector encrypted_vector(raw_vector, modulus_params.get(), 5,
                                    rlwe::kLogDegreeBound29);

  for (const auto& serialized : encrypted_vector.GetAsSerializedVector()) {
    encoded_vector.add_extra_data()->PackFrom(serialized);
  }

  (*invalid_message.mutable_masked_input_response()
        ->mutable_vectors())["foobar"] = encoded_vector;
  ASSERT_THAT(
      state.HandleMessage(
          0, std::make_unique<ClientToServerWrapperMessage>(invalid_message)),
      absl_testing::IsOk());
  EXPECT_THAT(state.ReadyForNextRound(), IsFalse());

  for (int i = 1; i < 5; ++i) {
    EXPECT_THAT(state.NeedsToAbort(), IsFalse());
    EXPECT_THAT(state.NumberOfAliveClients(), Eq(3));
    EXPECT_THAT(state.NumberOfClientsReadyForNextRound(), Eq(i - 1));
    EXPECT_THAT(state.NumberOfMessagesReceivedInThisRound(), Eq(i - 1));
    EXPECT_THAT(state.NumberOfPendingClients(), Eq(4 - i));
    if (i < 4) {
      EXPECT_THAT(state.MinimumMessagesNeededForNextRound(), Eq(4 - i));
      EXPECT_THAT(state.ReadyForNextRound(), IsFalse());
    } else {
      EXPECT_THAT(state.MinimumMessagesNeededForNextRound(), Eq(0));
      EXPECT_THAT(state.ReadyForNextRound(), IsTrue());
    }
    if (i < 4) {
      // Have client send a vector of the correct size to the server
      MaskedInputVector encoded_vector;
      std::vector<internal::rlwe_mont_int> raw_uintm_vector;
      for (int j = 0; j < kRlweDegree; j++) {
        auto value =
            internal::rlwe_mont_int::ImportInt(i + 1, modulus_params.get());
        ASSERT_THAT(value, absl_testing::IsOk());
        raw_uintm_vector.emplace_back(std::move(value.value()));
      }
      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>> raw_vector = {
          rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector)};
      SecAggRlweVector encrypted_vector(raw_vector, modulus_params.get(), 5,
                                        rlwe::kLogDegreeBound29);

      for (const auto& serialized : encrypted_vector.GetAsSerializedVector()) {
        encoded_vector.add_extra_data()->PackFrom(serialized);
      }

      ClientToServerWrapperMessage client_message;
      (*client_message.mutable_masked_input_response()
            ->mutable_vectors())["foobar"] = encoded_vector;
      ASSERT_THAT(
          state.HandleMessage(i, std::make_unique<ClientToServerWrapperMessage>(
                                     client_message)),
          absl_testing::IsOk());
      EXPECT_THAT(state.ReadyForNextRound(), Eq(i >= 3));
    }
  }

  auto next_state = state.ProceedToNextRound();
  ASSERT_THAT(next_state, absl_testing::IsOk());
  EXPECT_THAT(next_state.value()->State(),
              Eq(SecAggServerStateKind::R3_UNMASKING));
  EXPECT_THAT(
      next_state.value()->NumberOfClientsFailedAfterSendingMaskedInput(),
      Eq(0));
  EXPECT_THAT(
      next_state.value()->NumberOfClientsFailedBeforeSendingMaskedInput(),
      Eq(1));
  EXPECT_THAT(next_state.value()->NumberOfClientsTerminatedWithoutUnmasking(),
              Eq(0));
}

TEST(SecaggServerR2MaskedInputCollStateTest,
     StateProceedsCorrectlyWithOneClientSendingInvalidLength) {
  // In this test, client 0 sends an input the wrong number of polynomial so it
  // is aborted. The rest of the round goes normally.
  auto sender = std::make_shared<MockSendToClientsInterface>();

  SecAggServerR2MaskedInputCollState state(
      CreateRlweSecAggServerProtocolImpl(3, 4, sender.get()),
      0,  // number_of_clients_failed_after_sending_masked_input
      0,  // number_of_clients_failed_before_sending_masked_input
      0   // number_of_clients_terminated_without_unmasking
  );

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status, absl_testing::IsOk());
  auto modulus_params = std::move(modulus_params_status.value());

  ServerToClientWrapperMessage server_message;
  // TODO(team): 1 -> 0 below, once backwards compatibility not needed.
  server_message.mutable_unmasking_request()->add_dead_3_client_ids(1);
  ServerToClientWrapperMessage abort_message;
  abort_message.mutable_abort()->set_early_success(false);
  abort_message.mutable_abort()->set_diagnostic_info(
      "Masked input does not match input vector specification - vector is "
      "wrong size.");

  EXPECT_CALL(*sender, Send(_, _)).Times(0);
  for (int i = 1; i < 4; ++i) {
    EXPECT_CALL(*sender, Send(i, EqualsProto(server_message))).Times(1);
  }
  EXPECT_CALL(*sender, Send(0, EqualsProto(abort_message))).Times(1);

  // Have client 0 send an invalid message.
  ClientToServerWrapperMessage invalid_message;
  MaskedInputVector encoded_vector;
  std::vector<internal::rlwe_mont_int> raw_uintm_vector;
  for (int j = 0; j < kRlweDegree; j++) {
    auto value =
        internal::rlwe_mont_int::ImportInt(0xFFFF, modulus_params.get());
    ASSERT_THAT(value, absl_testing::IsOk());
    raw_uintm_vector.emplace_back(std::move(value.value()));
  }
  // Too many polynomials
  std::vector<rlwe::Polynomial<internal::rlwe_mont_int>> raw_vector = {
      rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector),
      rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector)};
  SecAggRlweVector encrypted_vector(raw_vector, modulus_params.get(), 5,
                                    rlwe::kLogDegreeBound29);

  for (const auto& serialized : encrypted_vector.GetAsSerializedVector()) {
    encoded_vector.add_extra_data()->PackFrom(serialized);
  }

  (*invalid_message.mutable_masked_input_response()
        ->mutable_vectors())["foobar"] = encoded_vector;
  ASSERT_THAT(
      state.HandleMessage(
          0, std::make_unique<ClientToServerWrapperMessage>(invalid_message)),
      absl_testing::IsOk());
  EXPECT_THAT(state.ReadyForNextRound(), IsFalse());
  for (int i = 1; i < 5; ++i) {
    EXPECT_THAT(state.NeedsToAbort(), IsFalse());
    EXPECT_THAT(state.NumberOfAliveClients(), Eq(3));
    EXPECT_THAT(state.NumberOfClientsReadyForNextRound(), Eq(i - 1));
    EXPECT_THAT(state.NumberOfMessagesReceivedInThisRound(), Eq(i - 1));
    EXPECT_THAT(state.NumberOfPendingClients(), Eq(4 - i));
    if (i < 4) {
      EXPECT_THAT(state.MinimumMessagesNeededForNextRound(), Eq(4 - i));
      EXPECT_THAT(state.ReadyForNextRound(), IsFalse());
    } else {
      EXPECT_THAT(state.MinimumMessagesNeededForNextRound(), Eq(0));
      EXPECT_THAT(state.ReadyForNextRound(), IsTrue());
    }
    if (i < 4) {
      // Have client send a vector of the correct size to the server
      MaskedInputVector encoded_vector;
      std::vector<internal::rlwe_mont_int> raw_uintm_vector;
      for (int j = 0; j < kRlweDegree; j++) {
        auto value =
            internal::rlwe_mont_int::ImportInt(i + 1, modulus_params.get());
        ASSERT_THAT(value, absl_testing::IsOk());
        raw_uintm_vector.emplace_back(std::move(value.value()));
      }
      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>> raw_vector = {
          rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector)};
      SecAggRlweVector encrypted_vector(raw_vector, modulus_params.get(), 5,
                                        rlwe::kLogDegreeBound29);

      for (const auto& serialized : encrypted_vector.GetAsSerializedVector()) {
        encoded_vector.add_extra_data()->PackFrom(serialized);
      }

      ClientToServerWrapperMessage client_message;
      (*client_message.mutable_masked_input_response()
            ->mutable_vectors())["foobar"] = encoded_vector;
      ASSERT_THAT(
          state.HandleMessage(i, std::make_unique<ClientToServerWrapperMessage>(
                                     client_message)),
          absl_testing::IsOk());
      EXPECT_THAT(state.ReadyForNextRound(), Eq(i >= 3));
    }
  }

  auto next_state = state.ProceedToNextRound();
  ASSERT_THAT(next_state, absl_testing::IsOk());
  EXPECT_THAT(next_state.value()->State(),
              Eq(SecAggServerStateKind::R3_UNMASKING));
  EXPECT_THAT(
      next_state.value()->NumberOfClientsFailedAfterSendingMaskedInput(),
      Eq(0));
  EXPECT_THAT(
      next_state.value()->NumberOfClientsFailedBeforeSendingMaskedInput(),
      Eq(1));
  EXPECT_THAT(next_state.value()->NumberOfClientsTerminatedWithoutUnmasking(),
              Eq(0));
}

}  // namespace
}  // namespace secagg
}  // namespace fcp
