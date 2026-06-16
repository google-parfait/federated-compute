// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fcp/secagg/shared/rlwe/map_of_rlwe_masks.h"

#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_cat.h"
#include "fcp/secagg/shared/aes_ctr_prng_factory.h"
#include "fcp/secagg/shared/math.h"
#include "fcp/secagg/shared/rlwe/input_vector_rlwe_specification.h"
#include "third_party/rlwe/symmetric_encryption.h"

namespace fcp {
namespace secagg {
namespace {

using ::testing::Eq;
using ::testing::Ne;

const int kBitWidth = 20;

TEST(MapOfMasksTest, RlweAddMapsWorks) {
  SecAggRlweVectorMap map1, map2;

  auto modulus_params_status =
      internal::rlwe_mont_int::Params::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  auto four = internal::rlwe_mont_int::ImportInt(4, modulus_params.get());
  ASSERT_THAT(four.ok(), Eq(true));

  auto five = internal::rlwe_mont_int::ImportInt(5, modulus_params.get());
  ASSERT_THAT(five.ok(), Eq(true));
  std::vector<internal::rlwe_mont_int> raw_uintm_vector = {four.value(),
                                                           five.value()};
  std::vector<rlwe::Polynomial<internal::rlwe_mont_int>> raw_vector = {
      rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector)};
  SecAggRlweVector vector1(raw_vector, modulus_params.get(), 5,
                           rlwe::kLogDegreeBound29);

  raw_uintm_vector = {five.value(), four.value()};
  raw_vector = {rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector)};
  SecAggRlweVector vector2(raw_vector, modulus_params.get(), 5,
                           rlwe::kLogDegreeBound29);

  map1.emplace("test", std::move(vector1));
  map2.emplace("test", std::move(vector2));
  absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>> sum_map_status =
      AddRlweMaps(map1, map2, modulus_params.get());
  ASSERT_THAT(sum_map_status.ok(), Eq(true));
  auto sum_map = std::move(sum_map_status.value());

  ASSERT_THAT(sum_map->find("test"), Ne(sum_map->end()));
  auto sum_vec = sum_map->at("test").GetAsPolynomialVector();
  ASSERT_THAT(sum_vec.ok(), Eq(true));
  auto polynomial = sum_vec.value()[0];

  auto nine = internal::rlwe_mont_int::ImportInt(9, modulus_params.get());
  ASSERT_THAT(nine.ok(), Eq(true));
  raw_uintm_vector = {nine.value(), nine.value()};
  internal::rlwe_polynomial expected_polynomial(raw_uintm_vector);

  EXPECT_THAT(polynomial, Eq(expected_polynomial));
}

TEST(MapOfMasksTest, RlweAddMapsFailsIfVectorSizeMismatch) {
  SecAggRlweVectorMap map1, map2;

  auto modulus_params_status =
      internal::rlwe_mont_int::Params::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  auto four = internal::rlwe_mont_int::ImportInt(4, modulus_params.get());
  ASSERT_THAT(four.ok(), Eq(true));

  auto five = internal::rlwe_mont_int::ImportInt(5, modulus_params.get());
  ASSERT_THAT(five.ok(), Eq(true));
  std::vector<internal::rlwe_mont_int> raw_uintm_vector = {four.value(),
                                                           five.value()};
  std::vector<rlwe::Polynomial<internal::rlwe_mont_int>> raw_vector = {
      rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector)};
  SecAggRlweVector vector1(raw_vector, modulus_params.get(), 5,
                           rlwe::kLogDegreeBound29);

  raw_uintm_vector = {five.value(), four.value()};

  // This vector will have one more item
  raw_vector = {rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector),
                rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector)};
  SecAggRlweVector vector2(raw_vector, modulus_params.get(), 5,
                           rlwe::kLogDegreeBound29);

  map1.emplace("test", std::move(vector1));
  map2.emplace("test", std::move(vector2));
  absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>> sum_map_status =
      AddRlweMaps(map1, map2, modulus_params.get());
  EXPECT_THAT(sum_map_status.ok(), Eq(false));
}

TEST(MapOfMasksTest, RlweVersionMapsWithOppositeMasksCancel) {
  std::vector<AesKey> prng_keys_to_add;
  uint8_t key[AesKey::kSize];
  memset(key, 'A', AesKey::kSize);
  prng_keys_to_add.push_back(AesKey(key));
  memset(key, 'B', AesKey::kSize);
  prng_keys_to_add.push_back(AesKey(key));
  std::vector<AesKey> prng_keys_to_subtract;
  SessionId session_id = {std::string(32, 'Z')};
  std::vector<InputVectorRlweSpecification> vector_specs;
  vector_specs.push_back(InputVectorRlweSpecification(
      "test", rlwe::kDegreeBound29, rlwe::kDegreeBound29, rlwe::kModulus29,
      rlwe::kLogDegreeBound29, 1ULL << kBitWidth));

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  std::vector<internal::rlwe_mont_int> raw_vector;
  raw_vector.reserve(rlwe::kDegreeBound29);
  for (int i = 0; i < rlwe::kDegreeBound29; i++) {
    auto value_status =
        internal::rlwe_mont_int::ImportInt(i, modulus_params.get());
    ASSERT_THAT(value_status.ok(), Eq(true));
    auto value = value_status.value();
    raw_vector.emplace_back(value);
  }
  absl::flat_hash_map<std::string,
                      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>>>
      common_polynomial = {
          {"test", {rlwe::Polynomial<internal::rlwe_mont_int>(raw_vector)}}};

  auto masks1_status = MapOfRlweMasks(
      prng_keys_to_add, prng_keys_to_subtract, vector_specs, session_id,
      AesCtrPrngFactory(), modulus_params.get(), common_polynomial);

  auto masks2_status = MapOfRlweMasks(
      prng_keys_to_subtract, prng_keys_to_add, vector_specs, session_id,
      AesCtrPrngFactory(), modulus_params.get(), common_polynomial);
  ASSERT_THAT(masks1_status.ok(), Eq(true));
  ASSERT_THAT(masks2_status.ok(), Eq(true));
  auto masks1(std::move(masks1_status.value()));
  auto masks2(std::move(masks2_status.value()));

  EXPECT_THAT(masks1->size(), Eq(1));
  EXPECT_THAT(masks2->size(), Eq(1));
  absl::StatusOr<std::vector<internal::rlwe_polynomial>> vector1_status =
      masks1->at("test").GetAsPolynomialVector();
  ASSERT_THAT(vector1_status.ok(), Eq(true));
  absl::StatusOr<std::vector<internal::rlwe_polynomial>> vector2_status =
      masks2->at("test").GetAsPolynomialVector();
  ASSERT_THAT(vector2_status.ok(), Eq(true));
  auto vector1 = vector1_status.value();
  auto vector2 = vector2_status.value();

  ASSERT_THAT(vector1.size(), Eq(vector2.size()));

  auto ntt_params = ::rlwe::InitializeNttParameters<internal::rlwe_mont_int>(
      rlwe::kLogDegreeBound29, modulus_params.get());
  ASSERT_THAT(ntt_params.ok(), Eq(true));

  for (int i = 0; i < vector1.size(); i++) {
    auto poly1 = vector1[i];
    auto poly2 = vector2[i];
    auto sum_poly = poly1.Add(poly2, modulus_params.get());
    ASSERT_THAT(sum_poly.ok(), Eq(true));
    auto coeffs = rlwe::RemoveError(
        sum_poly.value().InverseNtt(&ntt_params.value(), modulus_params.get()),
        modulus_params->modulus, (1 << kBitWidth) + 1, modulus_params.get());
    for (const auto& coeff : coeffs) {
      EXPECT_THAT(coeff, Eq(0));
    }
  }
}

TEST(MapOfMasksTest, RlweVersionMapsWithMixedOppositeMasksCancel) {
  std::vector<AesKey> prng_keys_to_add;
  uint8_t key[AesKey::kSize];
  memset(key, 'A', AesKey::kSize);
  prng_keys_to_add.push_back(AesKey(key));
  memset(key, 'B', AesKey::kSize);
  std::vector<AesKey> prng_keys_to_subtract;
  prng_keys_to_subtract.push_back(AesKey(key));
  SessionId session_id = {std::string(32, 'Z')};
  std::vector<InputVectorRlweSpecification> vector_specs;

  vector_specs.push_back(InputVectorRlweSpecification(
      "test", rlwe::kDegreeBound29, rlwe::kDegreeBound29, rlwe::kModulus29,
      rlwe::kLogDegreeBound29, 1ULL << kBitWidth));

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  std::vector<internal::rlwe_mont_int> raw_vector;
  raw_vector.reserve(rlwe::kDegreeBound29);
  for (int i = 0; i < rlwe::kDegreeBound29; i++) {
    auto value_status =
        internal::rlwe_mont_int::ImportInt(i, modulus_params.get());
    ASSERT_THAT(value_status.ok(), Eq(true));
    auto value = value_status.value();
    raw_vector.emplace_back(value);
  }
  absl::flat_hash_map<std::string,
                      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>>>
      common_polynomial = {
          {"test", {rlwe::Polynomial<internal::rlwe_mont_int>(raw_vector)}}};

  auto masks1_status = MapOfRlweMasks(
      prng_keys_to_add, prng_keys_to_subtract, vector_specs, session_id,
      AesCtrPrngFactory(), modulus_params.get(), common_polynomial);

  auto masks2_status = MapOfRlweMasks(
      prng_keys_to_subtract, prng_keys_to_add, vector_specs, session_id,
      AesCtrPrngFactory(), modulus_params.get(), common_polynomial);
  ASSERT_THAT(masks1_status.ok(), Eq(true));
  ASSERT_THAT(masks2_status.ok(), Eq(true));
  auto masks1(std::move(masks1_status.value()));
  auto masks2(std::move(masks2_status.value()));

  absl::StatusOr<std::vector<internal::rlwe_polynomial>> vector1_status =
      masks1->at("test").GetAsPolynomialVector();
  ASSERT_THAT(vector1_status.ok(), Eq(true));
  absl::StatusOr<std::vector<internal::rlwe_polynomial>> vector2_status =
      masks2->at("test").GetAsPolynomialVector();
  ASSERT_THAT(vector2_status.ok(), Eq(true));
  auto vector1 = vector1_status.value();
  auto vector2 = vector2_status.value();

  ASSERT_THAT(vector1.size(), Eq(vector2.size()));

  auto ntt_params = ::rlwe::InitializeNttParameters<internal::rlwe_mont_int>(
      rlwe::kLogDegreeBound29, modulus_params.get());
  ASSERT_THAT(ntt_params.ok(), Eq(true));

  for (int i = 0; i < vector1.size(); i++) {
    auto poly1 = vector1[i];
    auto poly2 = vector2[i];
    auto sum_poly = poly1.Add(poly2, modulus_params.get());
    ASSERT_THAT(sum_poly.ok(), Eq(true));
    auto coeffs = rlwe::RemoveError(
        sum_poly.value().InverseNtt(&ntt_params.value(), modulus_params.get()),
        modulus_params->modulus, (1 << kBitWidth) + 1, modulus_params.get());
    for (const auto& coeff : coeffs) {
      EXPECT_THAT(coeff, Eq(0));
    }
  }
}

TEST(MapOfMasksTest, RlweVersionReturnsZeroIfNoKeysSpecified) {
  std::vector<AesKey> prng_keys_to_add;
  std::vector<AesKey> prng_keys_to_subtract;
  SessionId session_id = {std::string(32, 'Z')};
  std::vector<InputVectorRlweSpecification> vector_specs;
  vector_specs.push_back(InputVectorRlweSpecification(
      "test", rlwe::kDegreeBound29, rlwe::kDegreeBound29, rlwe::kModulus29,
      rlwe::kLogDegreeBound29, 1ULL << kBitWidth));

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  std::vector<internal::rlwe_mont_int> raw_vector;
  raw_vector.reserve(rlwe::kDegreeBound29);
  for (int i = 0; i < rlwe::kDegreeBound29; i++) {
    auto value_status =
        internal::rlwe_mont_int::ImportInt(i, modulus_params.get());
    ASSERT_THAT(value_status.ok(), Eq(true));
    auto value = value_status.value();
    raw_vector.emplace_back(value);
  }

  absl::flat_hash_map<std::string,
                      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>>>
      common_polynomial = {
          {"test", {rlwe::Polynomial<internal::rlwe_mont_int>(raw_vector)}}};

  auto masks1_status = MapOfRlweMasks(
      prng_keys_to_add, prng_keys_to_subtract, vector_specs, session_id,
      AesCtrPrngFactory(), modulus_params.get(), common_polynomial);
  ASSERT_THAT(masks1_status.ok(), Eq(true));
  auto masks1(std::move(masks1_status.value()));
  EXPECT_THAT(masks1->size(), Eq(1));

  absl::StatusOr<std::vector<internal::rlwe_polynomial>> vector1_status =
      masks1->at("test").GetAsPolynomialVector();
  ASSERT_THAT(vector1_status.ok(), Eq(true));
  auto vector1 = vector1_status.value();

  auto ntt_params = ::rlwe::InitializeNttParameters<internal::rlwe_mont_int>(
      rlwe::kLogDegreeBound29, modulus_params.get());
  ASSERT_THAT(ntt_params.ok(), Eq(true));

  for (int i = 0; i < vector1.size(); i++) {
    auto poly1 = vector1[i];
    auto coeffs = rlwe::RemoveError(
        poly1.InverseNtt(&ntt_params.value(), modulus_params.get()),
        modulus_params->modulus, (1 << kBitWidth) + 1, modulus_params.get());
    for (const auto& coeff : coeffs) {
      EXPECT_THAT(coeff, Eq(0));
    }
  }
}

TEST(MapOfMasksTest, RlweVersionReturnsNonZeroIfOneKeySpecified) {
  std::vector<AesKey> prng_keys_to_add;
  uint8_t key[AesKey::kSize];
  memset(key, 'A', AesKey::kSize);
  prng_keys_to_add.push_back(AesKey(key));
  std::vector<AesKey> prng_keys_to_subtract;
  SessionId session_id = {std::string(32, 'Z')};
  std::vector<InputVectorRlweSpecification> vector_specs;
  vector_specs.push_back(InputVectorRlweSpecification(
      "test", rlwe::kDegreeBound29, rlwe::kDegreeBound29, rlwe::kModulus29,
      rlwe::kLogDegreeBound29, 1ULL << kBitWidth));

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  std::vector<internal::rlwe_mont_int> raw_vector;
  raw_vector.reserve(rlwe::kDegreeBound29);
  for (int i = 0; i < rlwe::kDegreeBound29; i++) {
    auto value_status =
        internal::rlwe_mont_int::ImportInt(i, modulus_params.get());
    ASSERT_THAT(value_status.ok(), Eq(true));
    auto value = value_status.value();
    raw_vector.emplace_back(value);
  }

  absl::flat_hash_map<std::string,
                      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>>>
      common_polynomial = {
          {"test", {rlwe::Polynomial<internal::rlwe_mont_int>(raw_vector)}}};

  auto masks1_status = MapOfRlweMasks(
      prng_keys_to_add, prng_keys_to_subtract, vector_specs, session_id,
      AesCtrPrngFactory(), modulus_params.get(), common_polynomial);
  ASSERT_THAT(masks1_status.ok(), Eq(true));
  auto masks1(std::move(masks1_status.value()));
  EXPECT_THAT(masks1->size(), Eq(1));

  absl::StatusOr<std::vector<internal::rlwe_polynomial>> vector1_status =
      masks1->at("test").GetAsPolynomialVector();
  ASSERT_THAT(vector1_status.ok(), Eq(true));
  auto vector1 = vector1_status.value();

  auto ntt_params = ::rlwe::InitializeNttParameters<internal::rlwe_mont_int>(
      rlwe::kLogDegreeBound29, modulus_params.get());
  ASSERT_THAT(ntt_params.ok(), Eq(true));

  for (int i = 0; i < vector1.size(); i++) {
    auto poly1 = vector1[i];
    auto coeffs = rlwe::RemoveError(
        poly1.InverseNtt(&ntt_params.value(), modulus_params.get()),
        modulus_params->modulus, (1 << kBitWidth) + 1, modulus_params.get());
    for (const auto& coeff : coeffs) {
      EXPECT_THAT(coeff, Ne(0));
    }
  }
}

TEST(MapOfMasksTest, RlweVersionMapWithOneKeyDiffersFromMapWithTwoKeys) {
  std::vector<AesKey> prng_keys_to_add;
  uint8_t
      key[AesKey::kSize];  // This key is reusable because AesKey makes a copy
  memset(key, 'A', AesKey::kSize);
  prng_keys_to_add.push_back(AesKey(key));
  std::vector<AesKey> prng_keys_to_subtract;
  SessionId session_id = {std::string(32, 'Z')};
  std::vector<InputVectorRlweSpecification> vector_specs;

  vector_specs.push_back(InputVectorRlweSpecification(
      "test", rlwe::kDegreeBound29, rlwe::kDegreeBound29, rlwe::kModulus29,
      rlwe::kLogDegreeBound29, 1ULL << kBitWidth));

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus29);
  ASSERT_THAT(modulus_params_status.ok(), Eq(true));
  auto modulus_params = std::move(modulus_params_status.value());

  std::vector<internal::rlwe_mont_int> raw_vector;
  raw_vector.reserve(rlwe::kDegreeBound29);
  for (int i = 0; i < rlwe::kDegreeBound29; i++) {
    auto value_status =
        internal::rlwe_mont_int::ImportInt(i, modulus_params.get());
    ASSERT_THAT(value_status.ok(), Eq(true));
    auto value = value_status.value();
    raw_vector.emplace_back(value);
  }
  absl::flat_hash_map<std::string,
                      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>>>
      common_polynomial = {
          {"test", {rlwe::Polynomial<internal::rlwe_mont_int>(raw_vector)}}};

  auto masks1_status = MapOfRlweMasks(
      prng_keys_to_add, prng_keys_to_subtract, vector_specs, session_id,
      AesCtrPrngFactory(), modulus_params.get(), common_polynomial);

  memset(key, 'B', AesKey::kSize);
  prng_keys_to_add.push_back(AesKey(key));

  auto masks2_status = MapOfRlweMasks(
      prng_keys_to_subtract, prng_keys_to_add, vector_specs, session_id,
      AesCtrPrngFactory(), modulus_params.get(), common_polynomial);
  ASSERT_THAT(masks1_status.ok(), Eq(true));
  ASSERT_THAT(masks2_status.ok(), Eq(true));
  auto masks1(std::move(masks1_status.value()));
  auto masks2(std::move(masks2_status.value()));

  absl::StatusOr<std::vector<internal::rlwe_polynomial>> vector1_status =
      masks1->at("test").GetAsPolynomialVector();
  ASSERT_THAT(vector1_status.ok(), Eq(true));
  absl::StatusOr<std::vector<internal::rlwe_polynomial>> vector2_status =
      masks2->at("test").GetAsPolynomialVector();
  ASSERT_THAT(vector2_status.ok(), Eq(true));
  auto vector1 = vector1_status.value();
  auto vector2 = vector2_status.value();

  ASSERT_THAT(vector1.size(), Eq(vector2.size()));

  auto ntt_params = ::rlwe::InitializeNttParameters<internal::rlwe_mont_int>(
      rlwe::kLogDegreeBound29, modulus_params.get());
  ASSERT_THAT(ntt_params.ok(), Eq(true));

  for (int i = 0; i < vector1.size(); i++) {
    auto poly1 = vector1[i];
    auto poly2 = vector2[i];
    EXPECT_THAT(poly1, Ne(poly2));
  }
}

}  // namespace
}  // namespace secagg
}  // namespace fcp
