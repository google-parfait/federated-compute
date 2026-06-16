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

#ifndef FCP_SECAGG_SHARED_RLWE_INPUT_VECTOR_RLWE_SPECIFICATION_H_
#define FCP_SECAGG_SHARED_RLWE_INPUT_VECTOR_RLWE_SPECIFICATION_H_

#include <cstdint>
#include <string>

#include "absl/base/attributes.h"
#include "fcp/secagg/shared/input_vector_specification.h"

namespace fcp {
namespace secagg {

class InputVectorRlweSpecification : public InputVectorSpecification {
 public:
  // 'length' must be a multiple of 'degree'.  'log_degree' is at least the
  // bit-width of 'degree' (the logarithm in base 2); typically degree = (1 <<
  // log_degree).
  InputVectorRlweSpecification(const std::string& name, int length, int degree,
                               uint64_t rlwe_modulus, int log_degree,
                               uint64_t modulus);

  ABSL_MUST_USE_RESULT inline int rlwe_polynomial_degree() const {
    return rlwe_polynomial_degree_;
  }

  ABSL_MUST_USE_RESULT inline int log_degree() const { return log_degree_; }

 private:
  // RLWE parameters
  //
  // In the RLWE version of the protocol, rlwe_modulus is equivalent to the
  // modulus of the original protocol.  The length parameter in this case
  // should be a multiple of the polynomial degree.
  const int rlwe_polynomial_degree_;
  const uint64_t rlwe_modulus_;
  const int log_degree_;
};

}  // namespace secagg
}  // namespace fcp

#endif  // FCP_SECAGG_SHARED_RLWE_INPUT_VECTOR_RLWE_SPECIFICATION_H_
