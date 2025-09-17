/*
 * Copyright 2025 Google LLC
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

// Utilities for verifying signatures using keys encoded as
// fcp::confidentialcompute::Key messages.

#ifndef FCP_CONFIDENTIALCOMPUTE_PAYLOAD_TRANSPARENCY_SIGNATURES_H_
#define FCP_CONFIDENTIALCOMPUTE_PAYLOAD_TRANSPARENCY_SIGNATURES_H_

#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/container/fixed_array.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "openssl/base.h"
#include "openssl/digest.h"

namespace fcp::confidential_compute::payload_transparency {

// The data that was signed, provided one of several ways.
using SignedData = std::variant<
    // A function that will emit the data as a sequence of absl::string_views.
    // This avoids needing to combine the payload into a single buffer just for
    // signature verification.
    absl::FunctionRef<void(absl::FunctionRef<void(absl::string_view)> emitter)>,
    // A pre-computed digest of the data to be verified. The caller is
    // responsible for ensuring that the digest used matches the key's settings.
    // Prefer a different variant in most cases.
    absl::Span<const uint8_t>>;

// Computes the digest of the data using the given hash function.
inline absl::FixedArray<uint8_t> ComputeDigest(const EVP_MD* type,
                                               SignedData signed_data);

// The format of a signature.
enum class SignatureFormat {
  // The ASN.1 format: a DER-encoded ECDSA-Sig-Value object.
  kAsn1,

  // The fixed-width format defined in IEEE P1363 (raw `r || s`).
  kP1363,
};

// Verifies the signature using a set of public keys, succeeding if any key
// works. To improve performance, callers can pre-filter the list of keys to
// those that could plausibly verify the signature (e.g. by checking key ids).
absl::Status VerifySignature(
    absl::string_view signature, SignatureFormat format,
    absl::Span<const confidentialcompute::Key* const> keys,
    SignedData signed_data);

//////////////////////// Implementation details follow. ////////////////////////

inline absl::FixedArray<uint8_t> ComputeDigest(const EVP_MD* type,
                                               SignedData signed_data) {
  // std::visit enables compile-time detection of unhandled SignedData types.
  return std::visit(
      [type](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<
                          T,
                          absl::FunctionRef<void(
                              absl::FunctionRef<void(absl::string_view)>)>>) {
          bssl::ScopedEVP_MD_CTX ctx;
          FCP_CHECK(EVP_DigestInit_ex(ctx.get(), type, /*engine=*/nullptr));
          absl::FixedArray<uint8_t> digest(EVP_MD_CTX_size(ctx.get()));
          arg([&ctx](absl::string_view s) {
            FCP_CHECK(EVP_DigestUpdate(ctx.get(), s.data(), s.size()));
          });
          FCP_CHECK(EVP_DigestFinal_ex(ctx.get(), &digest[0], nullptr));
          return digest;
        } else {
          static_assert(std::is_same_v<T, absl::Span<const uint8_t>>,
                        "Unsupported SignedData type");
          return absl::FixedArray<uint8_t>(arg.begin(), arg.end());
        }
      },
      std::move(signed_data));
}

}  // namespace fcp::confidential_compute::payload_transparency

#endif  // FCP_CONFIDENTIALCOMPUTE_PAYLOAD_TRANSPARENCY_SIGNATURES_H_
