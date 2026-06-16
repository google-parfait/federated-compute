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
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/secagg/server/rlwe/rlwe_secagg_server_protocol_impl.h"
#include "fcp/secagg/server/secagg_server_enums.pb.h"
#include "fcp/secagg/server/secagg_server_r0_advertise_keys_state.h"
#include "fcp/secagg/server/secagg_server_state.h"
#include "fcp/secagg/server/secret_sharing_graph_factory.h"
#include "fcp/secagg/shared/aes_ctr_prng_factory.h"
#include "fcp/secagg/shared/compute_session_id.h"
#include "fcp/secagg/shared/ecdh_keys.h"
#include "fcp/secagg/shared/input_vector_specification.h"
#include "fcp/secagg/shared/rlwe/rlwe_prng_adapter.h"
#include "fcp/secagg/shared/secagg_messages.pb.h"
#include "fcp/secagg/testing/ecdh_pregenerated_test_keys.h"
#include "fcp/secagg/testing/server/mock_secagg_server_metrics_listener.h"
#include "fcp/secagg/testing/server/mock_send_to_clients_interface.h"
#include "fcp/testing/testing.h"
#include "fcp/tracing/test_tracing_recorder.h"
#include "third_party/rlwe/constants.h"
#include "third_party/rlwe/montgomery.h"
#include "third_party/rlwe/polynomial.h"

namespace fcp {
namespace secagg {
namespace {

using ::testing::_;
using ::testing::Eq;

const unsigned char kRlweTestSeed[] = "0123456789abcdef0123456789abcdef";

std::unique_ptr<RlweSecAggServerProtocolImpl>
CreateRlweSecAggServerProtocolImpl(
    MockSendToClientsInterface* sender,
    MockSecAggServerMetricsListener* metrics_listener = nullptr) {
  auto input_vector_specs = std::vector<InputVectorRlweSpecification>();
  input_vector_specs.push_back(
      InputVectorRlweSpecification("foobar", 32, 4, 4, 2, 64));
  SecretSharingGraphFactory factory;

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus29);
  FCP_CHECK(modulus_params_status.ok());
  auto modulus_params = std::move(modulus_params_status.value());
  AesKey prng_seed(kRlweTestSeed, 32);
  return std::make_unique<RlweSecAggServerProtocolImpl>(
      factory.CreateCompleteGraph(4, 3),  // total number of clients is 4
      3,  // minimum_number_of_clients_to_proceed
      input_vector_specs,
      std::unique_ptr<MockSecAggServerMetricsListener>(metrics_listener),
      std::make_unique<AesCtrPrngFactory>(), sender,
      nullptr,  // prng_runner
      std::move(modulus_params),
      std::vector<ClientStatus>(4, ClientStatus::READY_TO_START), prng_seed,
      rlwe::kDegreeBound29);
}

TEST(SecaggServerR0AdvertiseKeysStateTest, StateSetsRandomPolynomialForRlwe) {
  // In this test, all clients send two valid ECDH public keys apiece, and then
  // the server proceeds to the next state.
  TestTracingRecorder tracing_recorder;
  auto sender = std::make_shared<MockSendToClientsInterface>();

  SecAggServerR0AdvertiseKeysState state(
      CreateRlweSecAggServerProtocolImpl(sender.get()));

  EcdhPregeneratedTestKeys ecdh_keys;
  auto pairwise_public_keys = std::make_unique<std::vector<EcdhPublicKey>>();
  std::vector<ClientToServerWrapperMessage> client_messages(4);
  ServerToClientWrapperMessage expected_server_message;
  for (int i = 0; i < 4; ++i) {
    PairOfPublicKeys* public_keys =
        expected_server_message.mutable_share_keys_request()
            ->add_pairs_of_public_keys();
    client_messages[i]
        .mutable_advertise_keys()
        ->mutable_pair_of_public_keys()
        ->set_enc_pk(ecdh_keys.GetPublicKeyString(i));
    client_messages[i]
        .mutable_advertise_keys()
        ->mutable_pair_of_public_keys()
        ->set_noise_pk(ecdh_keys.GetPublicKeyString(i + 4));
    public_keys->set_enc_pk(ecdh_keys.GetPublicKeyString(i));
    public_keys->set_noise_pk(ecdh_keys.GetPublicKeyString(i + 4));
  }

  expected_server_message.mutable_share_keys_request()->set_session_id(
      ComputeSessionId(expected_server_message.share_keys_request()).data);

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  std::vector<rlwe::Polynomial<internal::uint_m>> random_polynomials;
  AesCtrPrngFactory prng_factory;
  auto prng = prng_factory.MakePrng(AesKey(kRlweTestSeed, 32));
  RlwePrngAdapter prng_adapter(prng.get());
  auto random_poly =
      rlwe::SamplePolynomialFromPrng<internal::uint_m, RlwePrngAdapter>(
          rlwe::kDegreeBound29, &prng_adapter, modulus_params.get());

  // The input vector specification is very short, so only one random polynomial
  // (one ciphertext from the clients) is expected.
  random_polynomials.push_back(random_poly.value());

  for (const auto& polynomial : random_polynomials) {
    auto serialized_status = polynomial.Serialize(modulus_params.get());
    ASSERT_THAT(serialized_status.ok(), Eq(true));
    expected_server_message.mutable_share_keys_request()
        ->add_extra_data()
        ->PackFrom(serialized_status.value());
  }

  EXPECT_CALL(*sender, Send(_, _)).Times(0);
  for (int i = 0; i < 4; ++i) {
    EXPECT_CALL(*sender, Send(i, EqualsProto(expected_server_message)))
        .Times(1);
  }

  for (int i = 0; i < 5; ++i) {
    EXPECT_THAT(state.NeedsToAbort(), Eq(false));
    EXPECT_THAT(state.NumberOfAliveClients(), Eq(4));
    EXPECT_THAT(state.NumberOfClientsReadyForNextRound(), Eq(i));
    EXPECT_THAT(state.NumberOfMessagesReceivedInThisRound(), Eq(i));
    EXPECT_THAT(state.NumberOfPendingClients(), Eq(4 - i));
    if (i < 3) {
      EXPECT_THAT(state.MinimumMessagesNeededForNextRound(), Eq(3 - i));
      EXPECT_THAT(state.ReadyForNextRound(), Eq(false));
    } else {
      EXPECT_THAT(state.MinimumMessagesNeededForNextRound(), Eq(0));
      EXPECT_THAT(state.ReadyForNextRound(), Eq(true));
    }
    if (i < 4) {
      ASSERT_THAT(state.HandleMessage(i, client_messages[i]),
                  absl_testing::IsOk());
      EXPECT_THAT(state.ReadyForNextRound(), Eq(i >= 2));
    }
  }

  auto next_state = state.ProceedToNextRound();
  ASSERT_THAT(next_state.ok(), Eq(true));
  EXPECT_THAT(next_state.value()->State(),
              Eq(SecAggServerStateKind::R1_SHARE_KEYS));
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
                              0, Eq(ServerToClientMessageType_ShareKeysRequest),
                              Eq(expected_server_message.ByteSizeLong())),
                          IsEvent<IndividualMessageSent>(
                              1, Eq(ServerToClientMessageType_ShareKeysRequest),
                              Eq(expected_server_message.ByteSizeLong())),
                          IsEvent<IndividualMessageSent>(
                              2, Eq(ServerToClientMessageType_ShareKeysRequest),
                              Eq(expected_server_message.ByteSizeLong())),
                          IsEvent<IndividualMessageSent>(
                              3, Eq(ServerToClientMessageType_ShareKeysRequest),
                              Eq(expected_server_message.ByteSizeLong()))));
}

}  // namespace
}  // namespace secagg
}  // namespace fcp
