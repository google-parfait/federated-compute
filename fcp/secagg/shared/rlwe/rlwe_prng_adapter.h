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

#ifndef FCP_SECAGG_SHARED_RLWE_RLWE_PRNG_ADAPTER_H_
#define FCP_SECAGG_SHARED_RLWE_RLWE_PRNG_ADAPTER_H_

#include "fcp/secagg/shared/prng.h"
#include "third_party/rlwe/prng/prng.h"
#include "third_party/rlwe/statusor.h"

namespace fcp {
namespace secagg {

// The PRNG base class for the RLWE library expects StatusOr return values, but
// the PRNG base class for the SecAgg library only returns numbers.  This class
// only forwards method calls and its methods should never return errors.
class RlwePrngAdapter : public rlwe::SecurePrng {
 public:
  explicit RlwePrngAdapter(fcp::secagg::SecurePrng* prng) : prng_(prng) {}

  rlwe::StatusOr<uint8_t> Rand8() override { return prng_->Rand8(); }
  rlwe::StatusOr<uint64_t> Rand64() override { return prng_->Rand64(); }

 private:
  fcp::secagg::SecurePrng* prng_;  // not owned
};

}  // namespace secagg
}  // namespace fcp

#endif  // FCP_SECAGG_SHARED_RLWE_RLWE_PRNG_ADAPTER_H_
