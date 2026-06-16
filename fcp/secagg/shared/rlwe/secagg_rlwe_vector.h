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

#ifndef FCP_SECAGG_SHARED_RLWE_SECAGG_RLWE_VECTOR_H_
#define FCP_SECAGG_SHARED_RLWE_SECAGG_RLWE_VECTOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "fcp/base/monitoring.h"
#include "fcp/secagg/shared/rlwe/secagg_rlwe_params.h"
#include "third_party/rlwe/polynomial.h"
#include "third_party/rlwe/serialization.pb.h"

namespace fcp {
namespace secagg {

// Represents an immutable vector of RLWE polynomials.  This is used in the
// SecAgg package both to provide input to SecAggClient and by SecAggServer to
// provide its output.

class SecAggRlweVector {
 public:
  // Constructs a new SecAggRlweVector from unserialized polynomials.
  // raw_vector must not be empty, or the constructor will abort.  Ownership of
  // params will not be taken, and params must remain valid for the lifetime of
  // the constructed SecAddRlweVector.
  SecAggRlweVector(const std::vector<internal::rlwe_polynomial>& raw_vector,
                   const internal::RlweParams* params, uint64_t modulus,
                   size_t log_degree);

  // Constructs a new SecAggRlweVector from a vector of serialized polynomials.
  // serialized_vector must not be empty, or the constructor will abort.
  // Ownership of params will not be taken, and params must remain valid for the
  // lifetime of the constructed SecAddRlweVector.
  SecAggRlweVector(
      const std::vector<rlwe::SerializedNttPolynomial>& serialized_vector,
      const internal::RlweParams* params, uint64_t modulus, size_t log_degree);

  SecAggRlweVector(const SecAggRlweVector&) = delete;
  SecAggRlweVector& operator=(const SecAggRlweVector&) = delete;

  SecAggRlweVector(SecAggRlweVector&& other) = default;
  SecAggRlweVector& operator=(SecAggRlweVector&& other) = default;

  // Produces and returns a representation of this SecAggRlweVector as a vector
  // of rlwe::Polynomial. The returned vector is obtained by unpacking the
  // stored packed representation of the vector.
  absl::StatusOr<std::vector<internal::rlwe_polynomial>> GetAsPolynomialVector()
      const;

  // Returns the stored, compressed representation of the SecAggRlweVector. The
  // The bytes are stored in little-endian order, using only bit_width bits to
  // represent each element of the vector.
  const std::vector<rlwe::SerializedNttPolynomial>& GetAsSerializedVector()
      const {
    return serialized_;
  }

  ABSL_MUST_USE_RESULT inline uint64_t modulus() const { return modulus_; }

  ABSL_MUST_USE_RESULT inline size_t bit_width() const { return bit_width_; }

  ABSL_MUST_USE_RESULT inline size_t log_degree() const { return log_degree_; }

 private:
  std::vector<rlwe::SerializedNttPolynomial> serialized_;
  const internal::RlweParams* params_;  // not owned
  uint64_t modulus_;
  size_t bit_width_;
  size_t log_degree_;
};

// This is equivalent to
// using SecAggVectorMap = absl::node_hash_map<std::string, SecAggVector>;
// except copy construction and assignment are explicitly prohibited.
class SecAggRlweVectorMap
    : public absl::node_hash_map<std::string, SecAggRlweVector> {
 public:
  using Base = absl::node_hash_map<std::string, SecAggRlweVector>;
  using Base::Base;
  using Base::operator=;
  SecAggRlweVectorMap(const SecAggRlweVectorMap&) = delete;
  SecAggRlweVectorMap& operator=(const SecAggRlweVectorMap&) = delete;
};

}  // namespace secagg
}  // namespace fcp

#endif  // FCP_SECAGG_SHARED_RLWE_SECAGG_RLWE_VECTOR_H_
