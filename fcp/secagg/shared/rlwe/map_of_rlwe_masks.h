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

#ifndef FCP_SECAGG_SHARED_RLWE_MAP_OF_RLWE_MASKS_H_
#define FCP_SECAGG_SHARED_RLWE_MAP_OF_RLWE_MASKS_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "fcp/secagg/shared/aes_key.h"
#include "fcp/secagg/shared/aes_prng_factory.h"
#include "fcp/secagg/shared/async_abort.h"
#include "fcp/secagg/shared/compute_session_id.h"
#include "fcp/secagg/shared/rlwe/input_vector_rlwe_specification.h"
#include "fcp/secagg/shared/rlwe/secagg_rlwe_vector.h"

namespace fcp {
namespace secagg {

// Generates and returns a map of RLWE polynomials, using the common polynomial
// and the keys needed to generate the RLWE secret.
absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>> MapOfRlweMasks(
    const std::vector<AesKey>& prng_keys_to_add,
    const std::vector<AesKey>& prng_keys_to_subtract,
    const std::vector<InputVectorRlweSpecification>& input_vector_specs,
    const SessionId& session_id, const AesPrngFactory& prng_factory,
    const internal::RlweParams* rlwe_params,
    const absl::flat_hash_map<std::string,
                              std::vector<internal::rlwe_polynomial>>&
        common_polynomials,
    AsyncAbort* async_abort = nullptr);

absl::StatusOr<std::unique_ptr<SecAggRlweVectorMap>> AddRlweMaps(
    const SecAggRlweVectorMap& a, const SecAggRlweVectorMap& b,
    const internal::RlweParams* rlwe_params);

}  // namespace secagg
}  // namespace fcp

#endif  // FCP_SECAGG_SHARED_RLWE_MAP_OF_RLWE_MASKS_H_
