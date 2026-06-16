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

#include "fcp/secagg/shared/rlwe/secagg_rlwe_vector.h"

#include <utility>

#include "absl/numeric/bits.h"

namespace fcp {
namespace secagg {

SecAggRlweVector::SecAggRlweVector(
    const std::vector<internal::rlwe_polynomial>& raw_vector,
    const internal::RlweParams* params, uint64_t modulus, size_t log_degree)
    : params_(params),
      modulus_(modulus),
      bit_width_(absl::bit_width(modulus - 1ULL)),
      log_degree_(log_degree) {
  FCP_CHECK(!raw_vector.empty())
      << "Attempted to initialize empty SecAggRlweVector";
  serialized_.reserve(raw_vector.size());
  for (const auto& element : raw_vector) {
    serialized_.emplace_back(std::move(element.Serialize(params_)).value());
  }
}

SecAggRlweVector::SecAggRlweVector(
    const std::vector<rlwe::SerializedNttPolynomial>& serialized,
    const internal::RlweParams* params, uint64_t modulus, size_t log_degree)
    : serialized_(serialized),
      params_(params),
      modulus_(modulus),
      bit_width_(absl::bit_width(modulus - 1ULL)),
      log_degree_(log_degree) {
  FCP_CHECK(!serialized_.empty())
      << "Attempted to initialize empty SecAggRlweVector";
}

absl::StatusOr<std::vector<internal::rlwe_polynomial>>
SecAggRlweVector::GetAsPolynomialVector() const {
  std::vector<internal::rlwe_polynomial> long_vector;
  long_vector.reserve(serialized_.size());
  for (const auto& poly : serialized_) {
    auto deserialized = internal::rlwe_polynomial::Deserialize(poly, params_);
    if (!deserialized.ok()) {
      return deserialized.status();
    }
    long_vector.emplace_back(std::move(deserialized.value()));
  }
  return long_vector;
}

}  // namespace secagg
}  // namespace fcp
