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

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/benchmark.h"
#include "absl/strings/str_cat.h"
#include "fcp/secagg/shared/aes_ctr_prng_factory.h"
#include "fcp/secagg/shared/rlwe/input_vector_rlwe_specification.h"
#include "fcp/secagg/shared/rlwe/map_of_rlwe_masks.h"
#include "fcp/secagg/shared/rlwe/secagg_rlwe_vector.h"
#include "fcp/secagg/shared/secagg_vector.h"
#include "third_party/rlwe/symmetric_encryption.h"

namespace fcp {
namespace secagg {
namespace {

constexpr auto kVectorSize = 1024 * 1024;
constexpr auto kNumKeys = 128;

void BM_MapOfRlweMasks(benchmark::State& state) {
  std::vector<AesKey> prng_keys_to_add;
  uint8_t key[AesKey::kSize];
  memset(key, 'A', AesKey::kSize);
  prng_keys_to_add.reserve(kNumKeys);
  for (int i = 0; i < kNumKeys; i++) {
    prng_keys_to_add.emplace_back(key);
  }
  std::vector<AesKey> prng_keys_to_subtract;
  SessionId session_id = {std::string(32, 'Z')};
  std::vector<InputVectorRlweSpecification> vector_specs;

  auto modulus_params_status = internal::RlweParams::Create(rlwe::kModulus59);
  FCP_CHECK(modulus_params_status.ok());
  auto modulus_params = std::move(modulus_params_status.value());

  std::vector<internal::rlwe_mont_int> raw_vector;
  raw_vector.reserve(rlwe::kDegreeBound59);
  for (int i = 0; i < rlwe::kDegreeBound59; i++) {
    auto value_status =
        internal::rlwe_mont_int::ImportInt(i, modulus_params.get());
    FCP_CHECK(value_status.ok());
    auto value = value_status.value();
    raw_vector.emplace_back(value);
  }
  std::vector<rlwe::Polynomial<internal::rlwe_mont_int>> common_vector;
  common_vector.reserve(kVectorSize / rlwe::kDegreeBound59);
  for (int i = 0; i < kVectorSize / rlwe::kDegreeBound59; i++) {
    common_vector.emplace_back(
        rlwe::Polynomial<internal::rlwe_mont_int>(raw_vector));
  }

  // Check a variety of moduli
  std::vector<uint64_t> moduli{1ULL << 9, 1ULL << 25, 1ULL << 41, 1ULL << 53,
                               SecAggVector::kMaxModulus};
  vector_specs.reserve(moduli.size());
  absl::flat_hash_map<std::string,
                      std::vector<rlwe::Polynomial<internal::rlwe_mont_int>>>
      common_polynomial;
  for (uint64_t i : moduli) {
    vector_specs.emplace_back(absl::StrCat("unused", i), kVectorSize,
                              rlwe::kDegreeBound59, rlwe::kModulus59,
                              rlwe::kLogDegreeBound59, i);
    common_polynomial[absl::StrCat("unused", i)] = common_vector;
  }

  for (auto s : state) {
    auto rvalue = MapOfRlweMasks(
        prng_keys_to_add, prng_keys_to_subtract, vector_specs, session_id,
        static_cast<const AesPrngFactory&>(AesCtrPrngFactory()),
        modulus_params.get(), common_polynomial);
    ::benchmark::DoNotOptimize(rvalue.ok());
    state.SetItemsProcessed(kVectorSize);
  }
}

BENCHMARK(BM_MapOfRlweMasks);

}  // namespace
}  // namespace secagg
}  // namespace fcp
