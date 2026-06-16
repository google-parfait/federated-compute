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

#include "fcp/secagg/shared/rlwe/input_vector_rlwe_specification.h"

#include <string>

#include "fcp/base/monitoring.h"
#include "fcp/secagg/shared/secagg_vector.h"

namespace fcp {
namespace secagg {

InputVectorRlweSpecification::InputVectorRlweSpecification(
    const std::string& name, int length, int degree, uint64_t rlwe_modulus,
    int log_degree, uint64_t modulus)
    : InputVectorSpecification(name, length, modulus),
      rlwe_polynomial_degree_(degree),
      rlwe_modulus_(rlwe_modulus),
      log_degree_(log_degree) {
  FCP_CHECK(degree > 0) << "Polynomial degree must be greater than 0, was "
                        << degree;
  FCP_CHECK(length % degree == 0)
      << "Length must be a multiple of polynomial degree";
  FCP_CHECK(rlwe_modulus > 0)
      << "RlweModulus must be greater than 0, was " << rlwe_modulus;
  FCP_CHECK(log_degree > 0)
      << "Log degree must be greater than 0, was " << log_degree;
  FCP_CHECK((1 << log_degree) >= degree)
      << "Log degree must be the bit-width of degree but is too small";
  FCP_CHECK(modulus > 1 && modulus <= SecAggVector::kMaxModulus)
      << "The specified modulus is not valid: must be > 1 and <= "
      << SecAggVector::kMaxModulus << ", supplied value : " << this->modulus();
}

}  // namespace secagg
}  // namespace fcp
