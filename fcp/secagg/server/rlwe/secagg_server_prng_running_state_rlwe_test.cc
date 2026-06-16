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

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/str_cat.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/scheduler.h"
#include "fcp/secagg/server/rlwe/rlwe_secagg_server_protocol_impl.h"
#include "fcp/secagg/server/secagg_scheduler.h"
#include "fcp/secagg/server/secagg_server_prng_running_state.h"
#include "fcp/secagg/server/secret_sharing_graph_factory.h"
#include "fcp/secagg/shared/aes_ctr_prng_factory.h"
#include "fcp/secagg/shared/aes_key.h"
#include "fcp/secagg/shared/ecdh_key_agreement.h"
#include "fcp/secagg/shared/ecdh_keys.h"
#include "fcp/secagg/shared/rlwe/input_vector_rlwe_specification.h"
#include "fcp/secagg/shared/rlwe/map_of_rlwe_masks.h"
#include "fcp/secagg/shared/secagg_messages.pb.h"
#include "fcp/secagg/shared/secagg_vector.h"
#include "fcp/secagg/shared/shamir_secret_sharing.h"
#include "fcp/secagg/testing/ecdh_pregenerated_test_keys.h"
#include "fcp/secagg/testing/fake_prng.h"
#include "fcp/secagg/testing/server/mock_secagg_server_metrics_listener.h"
#include "fcp/secagg/testing/server/mock_send_to_clients_interface.h"
#include "fcp/secagg/testing/server/test_async_runner.h"
#include "fcp/secagg/testing/test_matchers.h"
#include "fcp/testing/testing.h"
#include "fcp/tracing/test_tracing_recorder.h"

namespace fcp {
namespace secagg {
namespace {

using ::testing::_;
using ::testing::Eq;
using ::testing::NiceMock;

// For testing purposes, make an AesKey out of a string.
AesKey MakeAesKey(const std::string& key) {
  EXPECT_THAT(key.size(), Eq(AesKey::kSize));
  return AesKey(reinterpret_cast<const uint8_t*>(key.c_str()));
}

// Mock class containing a callback that would be called when the PRNG is done.
class MockPrngDone {
 public:
  MOCK_METHOD(void, Callback, ());
};

class MockScheduler : public Scheduler {
 public:
  MOCK_METHOD(void, Schedule, (std::function<void()>), (override));
  MOCK_METHOD(void, WaitUntilIdle, ());
};

constexpr auto call_fn = [](const std::function<void()>& f) { f(); };

// Default test session_id.
SessionId session_id = {"session id number, 32 bytes long"};
const unsigned char kRlweTestSeed[] = "0123456789abcdef0123456789abcdef";
const uint64_t kRlweDegree = rlwe::kDegreeBound59;

std::unique_ptr<RlweSecAggServerProtocolImpl>
CreateRlweSecAggServerProtocolImpl(
    std::vector<InputVectorRlweSpecification> input_vector_specs,
    const absl::flat_hash_map<
        std::string, std::vector<rlwe::Polynomial<internal::rlwe_mont_int>>>&
        common_polynomials,
    MockSendToClientsInterface* sender,
    MockSecAggServerMetricsListener* metrics_listener = nullptr) {
  SecretSharingGraphFactory factory;
  auto parallel_scheduler = std::make_unique<NiceMock<MockScheduler>>();
  auto sequential_scheduler = std::make_unique<NiceMock<MockScheduler>>();
  EXPECT_CALL(*parallel_scheduler, Schedule(_)).WillRepeatedly(call_fn);
  EXPECT_CALL(*sequential_scheduler, Schedule(_)).WillRepeatedly(call_fn);
  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus59);
  FCP_CHECK(modulus_params_status.ok());
  auto modulus_params = std::move(modulus_params_status.value());
  AesKey prng_seed(kRlweTestSeed, 32);
  auto impl = std::make_unique<RlweSecAggServerProtocolImpl>(
      factory.CreateCompleteGraph(4, 3), 3, input_vector_specs,
      std::unique_ptr<MockSecAggServerMetricsListener>(metrics_listener),
      std::make_unique<AesCtrPrngFactory>(), sender,
      std::make_unique<TestAsyncRunner>(std::move(parallel_scheduler),
                                        std::move(sequential_scheduler)),
      std::move(modulus_params),
      std::vector<ClientStatus>(4, ClientStatus::UNMASKING_RESPONSE_RECEIVED),
      prng_seed, kRlweDegree);
  impl->set_common_polynomials(common_polynomials);
  impl->set_session_id(std::make_unique<SessionId>(session_id));
  EcdhPregeneratedTestKeys ecdh_keys;
  for (int i = 0; i < 4; ++i) {
    impl->SetPairwisePublicKeys(i, ecdh_keys.GetPublicKey(i));
  }

  return impl;
}

TEST(SecaggServerPrngRunningStateTest,
     RlweVersionPrngGetsRightMasksWhenAllClientsSurvive) {
  // First, set up necessary data for the SecAggServerPrngRunningState
  auto input_vector_specs = std::vector<InputVectorRlweSpecification>();

  input_vector_specs.push_back(InputVectorRlweSpecification(
      "foobar", kRlweDegree, kRlweDegree, rlwe::kModulus59,
      rlwe::kLogDegreeBound59, 32));

  auto sender = std::make_unique<MockSendToClientsInterface>();
  FakePrng prng;
  ShamirSecretSharing sharer;
  auto self_shamir_share_table = std::make_unique<
      absl::flat_hash_map<uint32_t, std::vector<ShamirShare>>>();
  for (int i = 0; i < 4; ++i) {
    self_shamir_share_table->insert(std::make_pair(
        i, sharer.Share(3, 4,
                        MakeAesKey(absl::StrCat(
                            "test 32 byte AES key for user #", i)))));
  }

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus59);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  auto zero_map = std::make_unique<SecAggRlweVectorMap>();
  zero_map->emplace(
      "foobar",
      SecAggRlweVector({internal::rlwe_polynomial(rlwe::kDegreeBound59,
                                                  modulus_params.get())},
                       modulus_params.get(), 32, rlwe::kLogDegreeBound59));

  std::vector<AesKey> prng_keys_to_add;
  std::vector<AesKey> prng_keys_to_subtract;
  for (int i = 0; i < 4; ++i) {
    prng_keys_to_add.push_back(
        MakeAesKey(absl::StrCat("test 32 byte AES key for user #", i)));
  }

  std::vector<internal::rlwe_mont_int> raw_vector;
  raw_vector.reserve(rlwe::kDegreeBound59);
  for (int i = 0; i < rlwe::kDegreeBound59; i++) {
    auto value_status =
        internal::rlwe_mont_int::ImportInt(1, modulus_params.get());
    ASSERT_THAT(value_status.ok(), Eq(true));
    auto value = value_status.value();
    raw_vector.emplace_back(value);
  }
  auto ntt_params_status =
      rlwe::InitializeNttParameters<internal::rlwe_mont_int>(
          rlwe::kLogDegreeBound59, modulus_params.get());
  ASSERT_THAT(ntt_params_status.ok(), Eq(true));
  internal::NttParams ntt_params(std::move(ntt_params_status.value()));

  absl::flat_hash_map<std::string,
                      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>>>
      common_polynomials = {
          {"foobar",
           {rlwe::Polynomial<internal::rlwe_mont_int>::ConvertToNtt(
               raw_vector, &ntt_params, modulus_params.get())}}};

  auto expected_map_of_masks = std::make_unique<SecAggVectorMap>();
  expected_map_of_masks->emplace(
      "foobar",
      SecAggVector(std::vector<uint64_t>(rlwe::kDegreeBound59, 0), 32));

  auto initial_sum_status = MapOfRlweMasks(
      prng_keys_to_add, prng_keys_to_subtract, input_vector_specs, session_id,
      AesCtrPrngFactory(), modulus_params.get(), common_polynomials);
  ASSERT_THAT(initial_sum_status.ok(), Eq(true));

  auto impl = CreateRlweSecAggServerProtocolImpl(
      input_vector_specs, common_polynomials, sender.get());
  impl->set_rlwe_encrypted_input_vector(std::move(initial_sum_status).value());
  impl->set_pairwise_shamir_share_table(
      std::make_unique<
          absl::flat_hash_map<uint32_t, std::vector<ShamirShare>>>());
  impl->set_self_shamir_share_table(std::move(self_shamir_share_table));

  SecAggServerPrngRunningState state(
      std::move(impl),
      0,   // number_of_clients_failed_after_sending_masked_input
      0,   // number_of_clients_failed_before_sending_masked_input
      0);  // number_of_clients_terminated_without_unmasking

  MockPrngDone prng_done;
  EXPECT_CALL(prng_done, Callback());

  state.EnterState();
  state.SetAsyncCallback([&]() { prng_done.Callback(); });

  EXPECT_THAT(state.ReadyForNextRound(), Eq(true));

  auto next_state = state.ProceedToNextRound();
  ASSERT_THAT(next_state.ok(), Eq(true));
  ASSERT_THAT(next_state.value()->State(),
              Eq(SecAggServerStateKind::COMPLETED));
  auto result = next_state.value()->Result();
  ASSERT_THAT(result.ok(), Eq(true));
  EXPECT_THAT(*result.value(),
              testing::MatchesSecAggVectorMap(*expected_map_of_masks));
}

TEST(SecaggServerPrngRunningStateTest,
     RlwePrngGetsRightMasksWithOneDeadClientAfterSendingInput) {
  // In this test, client 1 died after sending its masked input. Its input will
  // still be included.
  //
  // First, set up necessary data for the SecAggServerPrngRunningState.
  auto input_vector_specs = std::vector<InputVectorRlweSpecification>();

  input_vector_specs.push_back(InputVectorRlweSpecification(
      "foobar", kRlweDegree, kRlweDegree, rlwe::kModulus59,
      rlwe::kLogDegreeBound59, 32));

  auto sender = std::make_unique<MockSendToClientsInterface>();
  FakePrng prng;
  ShamirSecretSharing sharer;
  auto pairwise_shamir_share_table = std::make_unique<
      absl::flat_hash_map<uint32_t, std::vector<ShamirShare>>>();
  auto self_shamir_share_table = std::make_unique<
      absl::flat_hash_map<uint32_t, std::vector<ShamirShare>>>();
  EcdhPregeneratedTestKeys ecdh_keys;
  auto aborted_client_ids = std::make_unique<absl::flat_hash_set<uint32_t>>();
  aborted_client_ids->insert(1);

  for (int i = 0; i < 4; ++i) {
    self_shamir_share_table->insert(std::make_pair(
        i, sharer.Share(3, 4,
                        MakeAesKey(absl::StrCat(
                            "test 32 byte AES key for user #", i)))));
    // Blank out the share in position 1 because it would not have been sent.
    (*self_shamir_share_table)[i][1] = {""};
  }

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus59);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  std::vector<internal::rlwe_mont_int> raw_vector;
  raw_vector.reserve(rlwe::kDegreeBound59);
  for (int i = 0; i < rlwe::kDegreeBound59; i++) {
    auto value_status =
        internal::rlwe_mont_int::ImportInt(1, modulus_params.get());
    ASSERT_THAT(value_status.ok(), Eq(true));
    auto value = value_status.value();
    raw_vector.emplace_back(value);
  }
  auto ntt_params_status =
      rlwe::InitializeNttParameters<internal::rlwe_mont_int>(
          rlwe::kLogDegreeBound59, modulus_params.get());
  ASSERT_THAT(ntt_params_status.ok(), Eq(true));
  internal::NttParams ntt_params(std::move(ntt_params_status.value()));

  absl::flat_hash_map<std::string,
                      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>>>
      common_polynomials = {
          {"foobar",
           {rlwe::Polynomial<internal::rlwe_mont_int>::ConvertToNtt(
               raw_vector, &ntt_params, modulus_params.get())}}};

  // Generate the expected (negative) sum of masking vectors using MapofMasks.
  // We should subtract the self masks of clients 0, 2, and 3. We should
  // subtract the pairwise mask 2 and 3 added for 1, and add the pairwise mask
  // that 0 subtracted for 1.
  auto aborted_client_key_agreement =
      EcdhKeyAgreement::CreateFromPrivateKey(ecdh_keys.GetPrivateKey(1));
  std::vector<AesKey> prng_keys_to_add;
  std::vector<AesKey> prng_keys_to_subtract;
  for (int i = 0; i < 4; ++i) {
    prng_keys_to_subtract.push_back(
        MakeAesKey(absl::StrCat("test 32 byte AES key for user #", i)));
  }
  // We reverse the 'add' and 'subtract' arguments to simulate some of the keys
  // not being cancelled.
  auto initial_sum_status = MapOfRlweMasks(
      prng_keys_to_subtract, prng_keys_to_add, input_vector_specs, session_id,
      AesCtrPrngFactory(), modulus_params.get(), common_polynomials);
  ASSERT_THAT(initial_sum_status.ok(), Eq(true));

  // We expect the sum to be zero, because the keys should have all cancelled
  // following the PRNG state.
  auto expected_map_of_masks = std::make_unique<SecAggVectorMap>();
  expected_map_of_masks->emplace(
      "foobar",
      SecAggVector(std::vector<uint64_t>(rlwe::kDegreeBound59, 0), 32));

  auto impl = CreateRlweSecAggServerProtocolImpl(
      input_vector_specs, common_polynomials, sender.get());
  impl->set_client_status(
      1, ClientStatus::DEAD_AFTER_MASKED_INPUT_RESPONSE_RECEIVED);
  impl->set_rlwe_encrypted_input_vector(std::move(initial_sum_status).value());
  impl->set_pairwise_shamir_share_table(std::move(pairwise_shamir_share_table));
  impl->set_self_shamir_share_table(std::move(self_shamir_share_table));

  SecAggServerPrngRunningState state(
      std::move(impl),
      1,   // number_of_clients_failed_after_sending_masked_input
      0,   // number_of_clients_failed_before_sending_masked_input
      1);  // number_of_clients_terminated_without_unmasking

  MockPrngDone prng_done;
  EXPECT_CALL(prng_done, Callback());

  state.EnterState();
  state.SetAsyncCallback([&]() { prng_done.Callback(); });

  EXPECT_THAT(state.ReadyForNextRound(), Eq(true));

  auto next_state = state.ProceedToNextRound();
  ASSERT_THAT(next_state.ok(), Eq(true));
  if (next_state.value()->ErrorMessage().ok()) {
    ASSERT_THAT(next_state.value()->ErrorMessage().value(), Eq(""));
  }
  EXPECT_THAT(next_state.value()->State(),
              Eq(SecAggServerStateKind::COMPLETED));
  auto result = next_state.value()->Result();
  ASSERT_THAT(result.ok(), Eq(true));
  EXPECT_THAT(*result.value(),
              testing::MatchesSecAggVectorMap(*expected_map_of_masks));
}

TEST(SecaggServerPrngRunningStateTest,
     RlwePrngGetsRightMasksWithOneDeadClientBeforeSendingInput) {
  // In this test, client 1 died after sending its masked input. Its input will
  // still be included.
  //
  // First, set up necessary data for the SecAggServerPrngRunningState.
  auto input_vector_specs = std::vector<InputVectorRlweSpecification>();

  input_vector_specs.push_back(InputVectorRlweSpecification(
      "foobar", kRlweDegree, kRlweDegree, rlwe::kModulus59,
      rlwe::kLogDegreeBound59, 32));

  auto sender = std::make_unique<MockSendToClientsInterface>();
  FakePrng prng;
  ShamirSecretSharing sharer;
  auto pairwise_shamir_share_table = std::make_unique<
      absl::flat_hash_map<uint32_t, std::vector<ShamirShare>>>();
  auto self_shamir_share_table = std::make_unique<
      absl::flat_hash_map<uint32_t, std::vector<ShamirShare>>>();

  auto aborted_client_ids = std::make_unique<absl::flat_hash_set<uint32_t>>();
  aborted_client_ids->insert(1);

  EcdhPregeneratedTestKeys ecdh_keys;
  for (int i = 0; i < 4; ++i) {
    if (i == 1) {
      // Client 1 died in the previous step, so the other clients will have sent
      // shares of its pairwise key instead.
      pairwise_shamir_share_table->insert(
          std::make_pair(i, sharer.Share(3, 4, ecdh_keys.GetPrivateKey(i))));
      // Blank out the share in position 1 because it would not have been sent.
      (*pairwise_shamir_share_table)[i][1] = {""};
    } else {
      self_shamir_share_table->insert(std::make_pair(
          i, sharer.Share(3, 4,
                          MakeAesKey(absl::StrCat(
                              "test 32 byte AES key for user #", i)))));
      // Blank out the share in position 1 because it would not have been sent.
      (*self_shamir_share_table)[i][1] = {""};
    }
  }

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus59);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  std::vector<internal::rlwe_mont_int> raw_vector;
  raw_vector.reserve(rlwe::kDegreeBound59);
  for (int i = 0; i < rlwe::kDegreeBound59; i++) {
    auto value_status =
        internal::rlwe_mont_int::ImportInt(1, modulus_params.get());
    ASSERT_THAT(value_status.ok(), Eq(true));
    auto value = value_status.value();
    raw_vector.emplace_back(value);
  }
  auto ntt_params_status =
      rlwe::InitializeNttParameters<internal::rlwe_mont_int>(
          rlwe::kLogDegreeBound59, modulus_params.get());
  ASSERT_THAT(ntt_params_status.ok(), Eq(true));
  internal::NttParams ntt_params(std::move(ntt_params_status.value()));

  absl::flat_hash_map<std::string,
                      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>>>
      common_polynomials = {
          {"foobar",
           {rlwe::Polynomial<internal::rlwe_mont_int>::ConvertToNtt(
               raw_vector, &ntt_params, modulus_params.get())}}};

  // Generate the expected (negative) sum of masking vectors using MapofMasks.
  // We should subtract the self masks of clients 0, 2, and 3. We should
  // subtract the pairwise mask 2 and 3 added for 1, and add the pairwise mask
  // that 0 subtracted for 1.
  auto aborted_client_key_agreement =
      EcdhKeyAgreement::CreateFromPrivateKey(ecdh_keys.GetPrivateKey(1));
  std::vector<AesKey> prng_keys_to_add;
  std::vector<AesKey> prng_keys_to_subtract;
  for (int i = 0; i < 4; ++i) {
    if (i == 1) {
      continue;
    }
    prng_keys_to_subtract.push_back(
        MakeAesKey(absl::StrCat("test 32 byte AES key for user #", i)));
    AesKey pairwise_key = aborted_client_key_agreement.value()
                              ->ComputeSharedSecret(ecdh_keys.GetPublicKey(i))
                              .value();
    if (i == 0) {
      prng_keys_to_add.push_back(pairwise_key);
    } else {
      prng_keys_to_subtract.push_back(pairwise_key);
    }
  }
  // We reverse the 'add' and 'subtract' arguments to simulate some of the keys
  // not being cancelled.
  auto initial_sum_status = MapOfRlweMasks(
      prng_keys_to_subtract, prng_keys_to_add, input_vector_specs, session_id,
      AesCtrPrngFactory(), modulus_params.get(), common_polynomials);
  ASSERT_THAT(initial_sum_status.ok(), Eq(true));

  // We expect the sum to be zero, because the keys should have all cancelled
  // following the PRNG state.
  auto expected_map_of_masks = std::make_unique<SecAggVectorMap>();
  expected_map_of_masks->emplace(
      "foobar",
      SecAggVector(std::vector<uint64_t>(rlwe::kDegreeBound59, 0), 32));
  auto impl = CreateRlweSecAggServerProtocolImpl(
      input_vector_specs, common_polynomials, sender.get());
  impl->set_client_status(1, ClientStatus::DEAD_AFTER_SHARE_KEYS_RECEIVED);
  impl->set_rlwe_encrypted_input_vector(std::move(initial_sum_status).value());
  impl->set_pairwise_shamir_share_table(std::move(pairwise_shamir_share_table));
  impl->set_self_shamir_share_table(std::move(self_shamir_share_table));

  SecAggServerPrngRunningState state(
      std::move(impl),
      0,   // number_of_clients_failed_after_sending_masked_input
      1,   // number_of_clients_failed_before_sending_masked_input
      0);  // number_of_clients_terminated_without_unmasking

  MockPrngDone prng_done;
  EXPECT_CALL(prng_done, Callback());

  state.EnterState();
  state.SetAsyncCallback([&]() { prng_done.Callback(); });

  EXPECT_THAT(state.ReadyForNextRound(), Eq(true));

  auto next_state = state.ProceedToNextRound();
  ASSERT_THAT(next_state.ok(), Eq(true));
  if (next_state.value()->ErrorMessage().ok()) {
    ASSERT_THAT(next_state.value()->ErrorMessage().value(), Eq(""));
  }
  EXPECT_THAT(next_state.value()->State(),
              Eq(SecAggServerStateKind::COMPLETED));
  auto result = next_state.value()->Result();
  ASSERT_THAT(result.ok(), Eq(true));
  EXPECT_THAT(*result.value(),
              testing::MatchesSecAggVectorMap(*expected_map_of_masks));
}

}  // namespace
}  // namespace secagg
}  // namespace fcp
