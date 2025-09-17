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

#include "fcp/confidentialcompute/payload_transparency/signatures.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "openssl/base.h"
#include "openssl/digest.h"
#include "openssl/ec.h"
#include "openssl/ec_key.h"
#include "openssl/ecdsa.h"
#include "openssl/err.h"

namespace fcp::confidential_compute::payload_transparency {
namespace {

using ::fcp::confidentialcompute::Key;

// Attempts to verify a signature using a single key.
absl::Status VerifySignatureWithKey(absl::string_view signature,
                                    SignatureFormat format, const Key& key,
                                    SignedData signed_data) {
  if (key.purpose() != Key::VERIFY) {
    return absl::InvalidArgumentError("key is not a verifying key");
  }

  // Compute the digest of the message. std::optional works around
  // absl::FixedArray having a deleted assignment operator.
  std::optional<absl::FixedArray<uint8_t>> digest;
  switch (key.algorithm()) {
    case Key::ECDSA_P256:
      digest.emplace(ComputeDigest(EVP_sha256(), signed_data));
      break;

    default:
      return absl::InvalidArgumentError("unsupported key algorithm");
  }

  // Verify the signature.
  switch (key.algorithm()) {
    case Key::ECDSA_P256: {
      bssl::UniquePtr<EC_KEY> ec_key(EC_KEY_new());
      FCP_CHECK(ec_key);
      FCP_CHECK(EC_KEY_set_group(ec_key.get(), EC_group_p256()));
      if (!EC_KEY_oct2key(
              ec_key.get(),
              reinterpret_cast<const uint8_t*>(key.key_material().data()),
              key.key_material().size(), nullptr)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "invalid public key: ", ERR_reason_error_string(ERR_get_error())));
      }
      int result;
      switch (format) {
        case SignatureFormat::kAsn1:
          result =
              ECDSA_verify(0, digest->data(), digest->size(),
                           reinterpret_cast<const uint8_t*>(signature.data()),
                           signature.size(), ec_key.get());
          break;

        case SignatureFormat::kP1363:
          result = ECDSA_verify_p1363(
              digest->data(), digest->size(),
              reinterpret_cast<const uint8_t*>(signature.data()),
              signature.size(), ec_key.get());
          break;
      }
      if (result != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "invalid signature: ", ERR_reason_error_string(ERR_get_error())));
      }
      return absl::OkStatus();
    }

    default:
      return absl::InternalError("unsupported verifying key algorithm");
  }
}

}  // namespace

absl::Status VerifySignature(
    absl::string_view signature, SignatureFormat format,
    absl::Span<const confidentialcompute::Key* const> keys,
    SignedData signed_data) {
  if (keys.empty()) {
    return absl::InvalidArgumentError("no matching signing keys");
  }

  // Attempt to verify the signature using each key. If all fail, arbitrarily
  // return the first error.
  absl::Status status;
  for (const Key* key : keys) {
    if (auto s = VerifySignatureWithKey(signature, format, *key, signed_data);
        s.ok()) {
      return absl::OkStatus();
    } else {
      status.Update(std::move(s));
    }
  }
  return status;
}

}  // namespace fcp::confidential_compute::payload_transparency
