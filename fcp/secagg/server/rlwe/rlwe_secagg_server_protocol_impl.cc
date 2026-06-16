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

#include "fcp/secagg/server/rlwe/rlwe_secagg_server_protocol_impl.h"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>

#include "google/protobuf/any.pb.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "fcp/base/monitoring.h"
#include "fcp/secagg/shared/math.h"
#include "fcp/secagg/shared/rlwe/map_of_rlwe_masks.h"
#include "fcp/secagg/shared/rlwe/rlwe_prng_adapter.h"

namespace fcp {
namespace secagg {

// The number of keys included in a single PRNG job.
static constexpr int kPrngBatchSize = 32;

absl::Status RlweSecAggServerProtocolImpl::InitializeShareKeysRequest(
    ShareKeysRequest* request) const {
  auto prng = prng_factory()->MakePrng(rlwe_prng_seed_);
  RlwePrngAdapter prng_adapter(prng.get());
  for (const auto& input_vector_spec : input_vector_specs()) {
    uint32_t number_ciphertexts =
        DivideRoundUp(input_vector_spec.length(), rlwe_coeffs_);
    for (int i = 0; i < number_ciphertexts; i++) {
      ABSL_ASSIGN_OR_RETURN(
          auto random_poly,
          (rlwe::SamplePolynomialFromPrng<internal::uint_m, RlwePrngAdapter>(
              rlwe_coeffs_, &prng_adapter, rlwe_modulus_params())));
      ABSL_ASSIGN_OR_RETURN(auto serialized,
                            random_poly.Serialize(rlwe_modulus_params()));
      request->add_extra_data()->PackFrom(serialized);
    }
  }

  return ::absl::OkStatus();
}

AsyncToken RlweSecAggServerProtocolImpl::SetupMaskedInputCollection() {
  rlwe_encrypted_input_vector_ = std::make_unique<SecAggRlweVectorMap>();
  for (const InputVectorRlweSpecification& vector_spec : input_vector_specs()) {
    std::vector<rlwe::Polynomial<internal::rlwe_mont_int>> polynomial_vector;
    for (int i = 0;
         i < vector_spec.length() / vector_spec.rlwe_polynomial_degree(); i++) {
      // Initial all values in the accumulator to 0
      auto value = internal::rlwe_mont_int::ImportInt(0, rlwe_modulus_params());
      FCP_CHECK(value.ok());
      std::vector<internal::rlwe_mont_int> raw_uintm_vector(
          vector_spec.rlwe_polynomial_degree(), value.value());
      polynomial_vector.push_back(
          rlwe::Polynomial<internal::rlwe_mont_int>(raw_uintm_vector));
    }
    rlwe_encrypted_input_vector_->insert(std::make_pair(
        vector_spec.name(),
        SecAggRlweVector(polynomial_vector, rlwe_modulus_params(),
                         vector_spec.modulus(), vector_spec.log_degree())));
  }
  return nullptr;
}

absl::Status RlweSecAggServerProtocolImpl::HandleMaskedInputCollectionResponse(
    std::unique_ptr<MaskedInputCollectionResponse> masked_input_response) {
  FCP_CHECK(rlwe_encrypted_input_vector_);
  SecAggRlweVectorMap checked_masked_vectors;

  // Make sure the received vectors match the specification.
  if (masked_input_response->vectors().size() != input_vector_specs().size()) {
    return ::absl::InvalidArgumentError(
        "Masked input does not match input vector specification - "
        "wrong number of vectors.");
  }
  for (const InputVectorRlweSpecification& vector_spec : input_vector_specs()) {
    std::string vector_name = vector_spec.name();
    auto iter = masked_input_response->vectors().find(vector_name);
    if (iter == masked_input_response->vectors().end()) {
      return ::absl::InvalidArgumentError(
          "Masked input does not match input vector specification - wrong "
          "vector names.");
    }
    // In this case the extra_data field will contain SerializedNTTPolynomials.
    if (iter->second.extra_data().size() !=
        vector_spec.length() / vector_spec.rlwe_polynomial_degree()) {
      return ::absl::InvalidArgumentError(
          "Masked input does not match input vector specification - vector is "
          "wrong size.");
    }
    std::vector<::rlwe::SerializedNttPolynomial> rlwe_encrypted_vectors;
    rlwe_encrypted_vectors.reserve(iter->second.extra_data().size());
    for (const auto& encrypted_vector : iter->second.extra_data()) {
      if (!encrypted_vector.Is<::rlwe::SerializedNttPolynomial>()) {
        return ::absl::InvalidArgumentError(
            "Encrypted input vector is of the wrong type.");
      }
      ::rlwe::SerializedNttPolynomial serialized;
      encrypted_vector.UnpackTo(&serialized);
      if (serialized.num_coeffs() != vector_spec.rlwe_polynomial_degree()) {
        return ::absl::InvalidArgumentError(
            "Masked input does not match input vector specification - "
            "polynomial has wrong degree.");
      }

      rlwe_encrypted_vectors.emplace_back(serialized);
    }
    checked_masked_vectors.emplace(
        vector_name,
        SecAggRlweVector(
            std::vector<rlwe::SerializedNttPolynomial>(
                rlwe_encrypted_vectors.begin(), rlwe_encrypted_vectors.end()),
            rlwe_modulus_params(), vector_spec.modulus(),
            vector_spec.log_degree()));
  }
  ABSL_ASSIGN_OR_RETURN(
      rlwe_encrypted_input_vector_,
      AddRlweMaps(*rlwe_encrypted_input_vector_, checked_masked_vectors,
                  rlwe_modulus_params()));

  return ::absl::OkStatus();
}

void RlweSecAggServerProtocolImpl::FinalizeMaskedInputCollection() {}

AsyncToken RlweSecAggServerProtocolImpl::StartPrng(
    const PrngWorkItems& work_items,
    std::function<void(absl::Status)> done_callback) {
  FCP_CHECK(done_callback);
  FCP_CHECK(rlwe_encrypted_input_vector_);

  auto generators = std::vector<std::function<std::unique_ptr<
      absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>>()>>();

  // Break the keys to add or subtract into vectors of size kBatchSize (or less
  // for the last one) and schedule them as tasks.
  for (auto it = work_items.prng_keys_to_add.begin();
       it < work_items.prng_keys_to_add.end(); it += kPrngBatchSize) {
    std::vector<AesKey> batch_prng_keys_to_add;
    std::copy(it,
              std::min(it + kPrngBatchSize, work_items.prng_keys_to_add.end()),
              std::back_inserter(batch_prng_keys_to_add));
    generators.emplace_back([=]() {
      return std::make_unique<
          absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>>(
          MapOfRlweMasks(batch_prng_keys_to_add, std::vector<AesKey>(),
                         input_vector_specs(), session_id(), *prng_factory(),
                         rlwe_modulus_params(), common_polynomials_));
    });
  }

  for (auto it = work_items.prng_keys_to_subtract.begin();
       it < work_items.prng_keys_to_subtract.end(); it += kPrngBatchSize) {
    std::vector<AesKey> batch_prng_keys_to_subtract;
    std::copy(
        it,
        std::min(it + kPrngBatchSize, work_items.prng_keys_to_subtract.end()),
        std::back_inserter(batch_prng_keys_to_subtract));
    generators.emplace_back([=]() {
      return std::make_unique<
          absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>>(
          MapOfRlweMasks(std::vector<AesKey>(), batch_prng_keys_to_subtract,
                         input_vector_specs(), session_id(), *prng_factory(),
                         rlwe_modulus_params(), common_polynomials_));
    });
  }

  // TODO(team): extra std::unique wrapping SecAggRlweVectorMap shouldn't
  // be needed because SecAggRlweVectorMap is already move-only type.
  auto accumulator =
      scheduler()
          ->CreateAccumulator<
              absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>>(
              std::make_unique<
                  absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>>(
                  std::move(rlwe_encrypted_input_vector_)),
              [this](
                  const absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>& x,
                  const absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>&
                      y) {
                // TODO(team): Need test coverage for the cases when
                // either x or y have an error status.
                if (x.ok() && y.ok()) {
                  auto status_or_sum = AddRlweMaps(*x.value(), *y.value(),
                                                   this->rlwe_modulus_params());
                  if (status_or_sum.ok()) {
                    return std::make_unique<
                        absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>>(
                        std::move(status_or_sum).value());
                  } else {
                    return std::make_unique<
                        absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>>(
                        status_or_sum.status());
                  }
                } else if (x.ok()) {  // y is not OK
                  return std::make_unique<
                      absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>>(
                      y.status());
                } else {  // x is not OK
                  return std::make_unique<
                      absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>>(
                      x.status());
                }
              });
  for (const auto& generator : generators) {
    accumulator->Schedule(generator);
  }
  accumulator->SetAsyncObserver([=, accumulator = accumulator.get()]() {
    auto result = accumulator->GetResultAndCancel();
    done_callback(PrngRunnerFinished(std::move(result)));
  });

  return accumulator;
}

absl::Status RlweSecAggServerProtocolImpl::PrngRunnerFinished(
    std::unique_ptr<absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>>
        final_sum) {
  if (!final_sum->ok()) {
    return absl::InternalError(
        "An error occurred while computing the final sum.");
  }

  // Convert to SecAggVectorMap
  auto vector_map = std::make_unique<SecAggVectorMap>();
  for (const auto& item : *final_sum->value()) {
    ABSL_ASSIGN_OR_RETURN(
        std::vector<internal::rlwe_polynomial> polynomial_vector,
        item.second.GetAsPolynomialVector());

    auto status_or_ntt_params =
        rlwe::InitializeNttParameters<internal::rlwe_mont_int>(
            static_cast<int>(item.second.log_degree()), rlwe_modulus_params());
    if (!status_or_ntt_params.ok()) {
      return absl::InternalError(
          "An error occurred while setting up the NTT parameters");
    }
    auto ntt_params = std::move(status_or_ntt_params.value());

    std::vector<uint64_t> raw_sum_vector;
    raw_sum_vector.reserve(polynomial_vector.size() *
                           polynomial_vector[0].Coeffs().size());
    for (const auto& polynomial : polynomial_vector) {
      auto error_free_poly = rlwe::RemoveError(
          polynomial.InverseNtt(&ntt_params, rlwe_modulus_params()),
          rlwe_modulus_params()->modulus, (1 << item.second.bit_width()) + 1,
          rlwe_modulus_params());
      for (const auto& coeff : error_free_poly) {
        raw_sum_vector.push_back(coeff);
      }
    }
    vector_map->emplace(item.first,
                        SecAggVector(raw_sum_vector, item.second.modulus()));
  }
  SetResult(std::move(vector_map));
  return absl::OkStatus();
}

}  // namespace secagg
}  // namespace fcp
