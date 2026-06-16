/*
 * Copyright 2021 Google LLC
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

#ifndef FCP_SECAGG_SHARED_RLWE_SECAGG_RLWE_PARAMS_H_
#define FCP_SECAGG_SHARED_RLWE_SECAGG_RLWE_PARAMS_H_

#include "third_party/rlwe/error_params.h"
#include "third_party/rlwe/montgomery.h"
#include "third_party/rlwe/polynomial.h"
#include "third_party/rlwe/symmetric_encryption.h"

namespace fcp::secagg::internal {

typedef rlwe::MontgomeryInt<uint64_t> rlwe_mont_int;
typedef rlwe::Polynomial<internal::rlwe_mont_int> rlwe_polynomial;
typedef internal::rlwe_mont_int::Params RlweParams;
typedef rlwe::ErrorParams<rlwe_mont_int> ErrorParams;
typedef rlwe::NttParameters<rlwe_mont_int> NttParams;
typedef rlwe::SymmetricRlweKey<rlwe_mont_int> RlweKey;

// According to the New Hope paper, a variance of 8 is standard.
//
// [1] "Post-quantum key exchange -- a new hope", Erdem Alkim, Leo Ducas, Thomas
// Poppelmann, Peter Schwabe, USENIX Security Sumposium.
const int kRlweVariance = 8;

}  // namespace fcp::secagg::internal

#endif  // FCP_SECAGG_SHARED_RLWE_SECAGG_RLWE_PARAMS_H_
