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

#include "fcp/secagg/shared/rlwe/map_of_rlwe_masks.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "fcp/base/monitoring.h"
#include "fcp/secagg/shared/math.h"
#include "fcp/secagg/shared/rlwe/rlwe_prng_adapter.h"
#include "openssl/evp.h"
#include "third_party/rlwe/prng/single_thread_chacha_prng.h"
#include "third_party/rlwe/symmetric_encryption.h"

namespace fcp {
namespace secagg {

namespace {

// TODO(team): Constant for compatibility only, remove when not needed.
uint8_t kPrngSeedConstant = 0x02;

static AesKey DigestKey(EVP_MD_CTX* mdctx, const std::string& prng_input,
                        int bit_width, const AesKey& prng_key) {
  uint32_t input_size = static_cast<uint32_t>(prng_input.size());
  std::string input_size_data = IntToByteString(input_size);
  std::string bit_width_data = IntToByteString(bit_width);
  FCP_CHECK(EVP_DigestInit_ex(mdctx, EVP_sha256(), nullptr));
  FCP_CHECK(EVP_DigestUpdate(mdctx, bit_width_data.c_str(), sizeof(int)));
  FCP_CHECK(EVP_DigestUpdate(mdctx, prng_key.data(), prng_key.size()));
  FCP_CHECK(EVP_DigestUpdate(mdctx, &kPrngSeedConstant, 1));
  FCP_CHECK(EVP_DigestUpdate(mdctx, input_size_data.c_str(), sizeof(int)));
  FCP_CHECK(EVP_DigestUpdate(mdctx, prng_input.c_str(), input_size));

  uint8_t digest[AesKey::kSize];
  uint32_t digest_length = 0;
  FCP_CHECK(EVP_DigestFinal_ex(mdctx, digest, &digest_length));
  FCP_CHECK(digest_length == AesKey::kSize);
  return AesKey(digest);
}

absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>> MapOfRlweMasksImpl(
    const std::vector<AesKey>& prng_keys_to_add,
    const std::vector<AesKey>& prng_keys_to_subtract,
    const std::vector<InputVectorRlweSpecification>& input_vector_specs,
    const SessionId& session_id, const AesPrngFactory& prng_factory,
    const internal::RlweParams* rlwe_params,
    const absl::flat_hash_map<std::string,
                              std::vector<internal::rlwe_polynomial>>&
        common_polynomials,
    AsyncAbort* async_abort, EVP_MD_CTX* mdctx) {
  auto map_of_masks = std::make_unique<SecAggRlweVectorMap>();
  for (const InputVectorRlweSpecification& vector_spec : input_vector_specs) {
    if (async_abort && async_abort->Signalled())
      return absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>(nullptr);
    int bit_width =
        static_cast<int>(absl::bit_width(vector_spec.modulus() - 1ULL));
    // TODO(team): Support non power-of-2 moduli
    FCP_CHECK(1ULL << bit_width == vector_spec.modulus())
        << "Current Random Mask Generation impl requires power-of-2 moduli";
    std::string prng_input =
        absl::StrCat(session_id.data, IntToByteString(bit_width),
                     IntToByteString(vector_spec.length()), vector_spec.name());
    ABSL_ASSIGN_OR_RETURN(
        internal::NttParams ntt_params,
        ::rlwe::InitializeNttParameters<internal::rlwe_mont_int>(
            vector_spec.log_degree(), rlwe_params));

    ABSL_ASSIGN_OR_RETURN(internal::RlweKey mask_key,
                          internal::RlweKey::NullKey(
                              vector_spec.log_degree(), internal::kRlweVariance,
                              bit_width, rlwe_params, &ntt_params));

    for (const auto& prng_key : prng_keys_to_add) {
      if (async_abort && async_abort->Signalled())
        return absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>(nullptr);
      AesKey digest_key = DigestKey(mdctx, prng_input, bit_width, prng_key);
      std::unique_ptr<SecurePrng> prng = prng_factory.MakePrng(digest_key);
      RlwePrngAdapter rlwe_prng(prng.get());
      ABSL_ASSIGN_OR_RETURN(
          internal::RlweKey add_key,
          internal::RlweKey::Sample(vector_spec.log_degree(),
                                    internal::kRlweVariance, bit_width,
                                    rlwe_params, &ntt_params, &rlwe_prng));
      ABSL_ASSIGN_OR_RETURN(mask_key, mask_key.Add(add_key));
    }

    for (const auto& prng_key : prng_keys_to_subtract) {
      if (async_abort && async_abort->Signalled())
        return absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>(nullptr);
      AesKey digest_key = DigestKey(mdctx, prng_input, bit_width, prng_key);
      std::unique_ptr<SecurePrng> prng = prng_factory.MakePrng(digest_key);
      RlwePrngAdapter rlwe_prng(prng.get());
      ABSL_ASSIGN_OR_RETURN(
          internal::RlweKey sub_key,
          internal::RlweKey::Sample(vector_spec.log_degree(),
                                    internal::kRlweVariance, bit_width,
                                    rlwe_params, &ntt_params, &rlwe_prng));
      ABSL_ASSIGN_OR_RETURN(mask_key, mask_key.Sub(sub_key));
    }

    std::vector<internal::rlwe_polynomial> mask_vector;
    mask_vector.reserve(vector_spec.length() /
                        vector_spec.rlwe_polynomial_degree());
    for (int i = 0;
         i < vector_spec.length() / vector_spec.rlwe_polynomial_degree(); i++) {
      ABSL_ASSIGN_OR_RETURN(std::string prng_encryption_seed,
                            rlwe::SingleThreadChaChaPrng::GenerateSeed());
      ABSL_ASSIGN_OR_RETURN(
          auto prng_encryption,
          rlwe::SingleThreadChaChaPrng::Create(prng_encryption_seed));
      internal::rlwe_polynomial zero(vector_spec.rlwe_polynomial_degree(),
                                     rlwe_params);

      auto common_polynomial = common_polynomials.find(vector_spec.name());
      FCP_CHECK(common_polynomial != common_polynomials.end());
      ABSL_ASSIGN_OR_RETURN(
          internal::rlwe_polynomial encrypted_zero,
          rlwe::internal::Encrypt(mask_key, zero, common_polynomial->second[i],
                                  prng_encryption.get()));
      mask_vector.emplace_back(encrypted_zero);
    }

    if (async_abort && async_abort->Signalled())
      return absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>(nullptr);
    map_of_masks->emplace(
        vector_spec.name(),
        SecAggRlweVector(mask_vector, rlwe_params, vector_spec.modulus(),
                         vector_spec.log_degree()));
  }
  return map_of_masks;
}

}  // namespace

absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>> MapOfRlweMasks(
    const std::vector<AesKey>& prng_keys_to_add,
    const std::vector<AesKey>& prng_keys_to_subtract,
    const std::vector<InputVectorRlweSpecification>& input_vector_specs,
    const SessionId& session_id, const AesPrngFactory& prng_factory,
    const internal::RlweParams* rlwe_params,
    const absl::flat_hash_map<std::string,
                              std::vector<internal::rlwe_polynomial>>&
        common_polynomials,
    AsyncAbort* async_abort) {
  EVP_MD_CTX* mdctx;
  FCP_CHECK(mdctx = EVP_MD_CTX_create());
  auto result = MapOfRlweMasksImpl(
      prng_keys_to_add, prng_keys_to_subtract, input_vector_specs, session_id,
      prng_factory, rlwe_params, common_polynomials, async_abort, mdctx);
  EVP_MD_CTX_destroy(mdctx);
  return result;
}

absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>> AddRlweMaps(
    const SecAggRlweVectorMap& a, const SecAggRlweVectorMap& b,
    const internal::RlweParams* rlwe_params) {
  auto result = std::make_unique<SecAggRlweVectorMap>();
  for (const auto& item : a) {
    ABSL_ASSIGN_OR_RETURN(std::vector<internal::rlwe_polynomial> poly_vec_a,
                          item.second.GetAsPolynomialVector());

    ABSL_ASSIGN_OR_RETURN(std::vector<internal::rlwe_polynomial> poly_vec_b,
                          b.at(item.first).GetAsPolynomialVector());

    if (poly_vec_a.size() != poly_vec_b.size()) {
      return absl::InvalidArgumentError(
          "Attempted to add SecAggRlweVectors with different sizes.");
    }

    std::vector<internal::rlwe_polynomial> sum_vec;
    sum_vec.reserve(poly_vec_a.size());
    for (size_t i = 0; i < poly_vec_a.size(); i++) {
      ABSL_ASSIGN_OR_RETURN(auto sum_poly,
                            poly_vec_a[i].Add(poly_vec_b[i], rlwe_params));

      sum_vec.push_back(sum_poly);
    }
    result->emplace(item.first, SecAggRlweVector(sum_vec, rlwe_params,
                                                 item.second.modulus(),
                                                 item.second.log_degree()));
  }
  return result;
}

}  // namespace secagg
}  // namespace fcp
