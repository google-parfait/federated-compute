/*
 * Copyright 2020 Google LLC
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

#ifndef FCP_SECAGG_SERVER_RLWE_RLWE_SECAGG_SERVER_PROTOCOL_IMPL_H_
#define FCP_SECAGG_SERVER_RLWE_RLWE_SECAGG_SERVER_PROTOCOL_IMPL_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "fcp/secagg/server/secagg_scheduler.h"
#include "fcp/secagg/server/secagg_server_protocol_impl.h"
#include "fcp/secagg/shared/rlwe/input_vector_rlwe_specification.h"
#include "fcp/secagg/shared/rlwe/secagg_rlwe_params.h"
#include "fcp/secagg/shared/rlwe/secagg_rlwe_vector.h"
#include "third_party/rlwe/montgomery.h"
#include "third_party/rlwe/polynomial.h"

namespace fcp {
namespace secagg {
namespace internal {

typedef rlwe::MontgomeryInt<uint64_t> uint_m;
}

class RlweSecAggServerProtocolImpl : public SecAggServerProtocolImpl {
 public:
  RlweSecAggServerProtocolImpl(
      std::unique_ptr<SecretSharingGraph> graph,
      int minimum_number_of_clients_to_proceed,
      std::vector<InputVectorRlweSpecification> input_vector_rlwe_specs,
      std::unique_ptr<SecAggServerMetricsListener> metrics,
      std::unique_ptr<AesPrngFactory> prng_factory,
      SendToClientsInterface* sender,
      std::unique_ptr<SecAggScheduler> scheduler,
      std::unique_ptr<const internal::RlweParams> rlwe_modulus_params,
      std::vector<ClientStatus> client_statuses, AesKey rlwe_prng_seed,
      int rlwe_coeffs)
      : SecAggServerProtocolImpl(
            std::move(graph), minimum_number_of_clients_to_proceed,
            std::move(metrics), std::move(prng_factory), sender,
            std::move(scheduler), std::move(client_statuses)),
        rlwe_modulus_params_(std::move(rlwe_modulus_params)),
        rlwe_prng_seed_(rlwe_prng_seed),
        rlwe_coeffs_(rlwe_coeffs),
        input_vector_specs_(std::move(input_vector_rlwe_specs)) {}

  ServerVariant server_variant() const override {
    return ServerVariant::RLWE_HOMOMORPHIC_KEYS;
  }

  // Returns one InputVectorSpecification for each input vector which the
  // protocol will aggregate.
  const std::vector<InputVectorRlweSpecification>& input_vector_specs() const {
    return input_vector_specs_;
  }

  absl::Status InitializeShareKeysRequest(
      ShareKeysRequest* request) const override;

  const internal::RlweParams* rlwe_modulus_params() const {
    return rlwe_modulus_params_.get();
  }

  void set_common_polynomials(
      const absl::flat_hash_map<std::string,
                                std::vector<internal::rlwe_polynomial>>&
          common_polynomials) {
    common_polynomials_ = common_polynomials;
  }

  const SecAggRlweVectorMap* rlwe_encrypted_input_vector() const {
    return rlwe_encrypted_input_vector_.get();
  }

  // TODO(team): Remove this method. This field must be set from
  // inside the protocol implementation.
  void set_rlwe_encrypted_input_vector(
      std::unique_ptr<SecAggRlweVectorMap> rlwe_encrypted_input_vector) {
    rlwe_encrypted_input_vector_ = std::move(rlwe_encrypted_input_vector);
  }

  AsyncToken SetupMaskedInputCollection() override;

  absl::Status HandleMaskedInputCollectionResponse(
      std::unique_ptr<MaskedInputCollectionResponse> masked_input_response)
      override;

  void FinalizeMaskedInputCollection() override;

  AsyncToken StartPrng(
      const PrngWorkItems& work_items,
      std::function<void(absl::Status)> done_callback) override;

 private:
  absl::Status PrngRunnerFinished(
      std::unique_ptr<absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>>>
          final_sum);

  std::unique_ptr<SecAggRlweVectorMap> rlwe_encrypted_input_vector_;

  std::unique_ptr<const internal::RlweParams> rlwe_modulus_params_;
  absl::flat_hash_map<std::string, std::vector<internal::rlwe_polynomial>>
      common_polynomials_;

  // This is the seed used when compute the common polynomials.
  AesKey rlwe_prng_seed_;

  // The number of coefficients in each polynomial for the common polynomials.
  int rlwe_coeffs_ = -1;

  std::vector<InputVectorRlweSpecification> input_vector_specs_;
};

}  // namespace secagg
}  // namespace fcp

#endif  // FCP_SECAGG_SERVER_RLWE_RLWE_SECAGG_SERVER_PROTOCOL_IMPL_H_
