// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "fcp/confidentialcompute/crypto.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "openssl/aead.h"
#include "openssl/base.h"
#include "openssl/err.h"
#include "openssl/hpke.h"
#include "openssl/mem.h"
#include "openssl/rand.h"

namespace fcp {
namespace confidential_compute {

// A fixed random nonce, which is safe to reuse because the symmetric key is
// never reused and we also use a nonce-misuse resistant AEAD (AES-GCM-SIV).
constexpr absl::string_view kNonce =
    "\x74\xDF\x8F\xD4\xBE\x34\xAF\x64\x7F\x5E\x54\xF6";

// The HPKE info field.
constexpr absl::string_view kInfo;

MessageEncryptor::MessageEncryptor()
    : hpke_kem_(EVP_hpke_x25519_hkdf_sha256()),
      hpke_kdf_(EVP_hpke_hkdf_sha256()),
      hpke_aead_(EVP_hpke_aes_128_gcm()),
      aead_(EVP_aead_aes_128_gcm_siv()) {}

absl::StatusOr<EncryptMessageResult> MessageEncryptor::Encrypt(
    absl::string_view plaintext, absl::string_view recipient_public_key,
    absl::string_view associated_data) const {
  SymmetricKey symmetric_key{
      .algorithm = crypto_internal::kAeadAes128GcmSivFixedNonce,
      .k = std::string(EVP_AEAD_key_length(aead_), '\0'),
  };
  RAND_bytes(reinterpret_cast<uint8_t*>(symmetric_key.k.data()),
             symmetric_key.k.size());
  // Cleanse the memory containing the symmetric key upon exiting the scope so
  // the key cannot be accessed outside this function.
  absl::Cleanup key_cleanup = [&symmetric_key]() {
    OPENSSL_cleanse(symmetric_key.k.data(), symmetric_key.k.size());
  };
  FCP_ASSIGN_OR_RETURN(std::string serialized_symmetric_key,
                       symmetric_key.Encode());
  absl::Cleanup serialized_key_cleanup = [&serialized_symmetric_key]() {
    OPENSSL_cleanse(serialized_symmetric_key.data(),
                    serialized_symmetric_key.size());
  };

  FCP_ASSIGN_OR_RETURN(OkpCwt cwt, OkpCwt::Decode(recipient_public_key));
  if (!cwt.public_key ||
      cwt.public_key->algorithm !=
          crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm ||
      cwt.public_key->curve != crypto_internal::kX25519) {
    return absl::InvalidArgumentError("unsupported public key");
  }

  bssl::UniquePtr<EVP_AEAD_CTX> aead_ctx(EVP_AEAD_CTX_new(
      aead_, reinterpret_cast<const uint8_t*>(symmetric_key.k.data()),
      symmetric_key.k.size(), EVP_AEAD_DEFAULT_TAG_LENGTH));
  if (aead_ctx == nullptr) {
    return FCP_STATUS(fcp::INTERNAL)
           << "Failed to initialize EVP_AEAD_CTX: "
           << ERR_reason_error_string(ERR_get_error());
  }
  std::string ciphertext(plaintext.size() + EVP_AEAD_MAX_OVERHEAD, '\0');
  size_t ciphertext_len = 0;
  if (EVP_AEAD_CTX_seal(
          aead_ctx.get(), reinterpret_cast<uint8_t*>(ciphertext.data()),
          &ciphertext_len, ciphertext.size(),
          reinterpret_cast<const uint8_t*>(kNonce.data()), kNonce.size(),
          reinterpret_cast<const uint8_t*>(plaintext.data()), plaintext.size(),
          reinterpret_cast<const uint8_t*>(associated_data.data()),
          associated_data.size()) != 1) {
    return FCP_STATUS(fcp::INTERNAL)
           << "AEAD encryption failed: "
           << ERR_reason_error_string(ERR_get_error());
  }
  ciphertext.resize(ciphertext_len);
  FCP_ASSIGN_OR_RETURN(
      crypto_internal::WrapSymmetricKeyResult wrap_symmetric_key_result,
      crypto_internal::WrapSymmetricKey(hpke_kem_, hpke_kdf_, hpke_aead_,
                                        serialized_symmetric_key,
                                        cwt.public_key->x, associated_data));

  return EncryptMessageResult{
      .ciphertext = std::move(ciphertext),
      .encapped_key = std::move(wrap_symmetric_key_result.encapped_key),
      .encrypted_symmetric_key =
          std::move(wrap_symmetric_key_result.encrypted_symmetric_key),
  };
}

MessageDecryptor::MessageDecryptor()
    : hpke_kem_(EVP_hpke_x25519_hkdf_sha256()),
      hpke_kdf_(EVP_hpke_hkdf_sha256()),
      hpke_aead_(EVP_hpke_aes_128_gcm()),
      hpke_key_(),
      aead_(EVP_aead_aes_128_gcm_siv()) {
  FCP_CHECK(EVP_HPKE_KEY_generate(hpke_key_.get(), hpke_kem_) == 1)
      << "Failed to generate HPKE public/private keypair: "
      << ERR_reason_error_string(ERR_get_error());
}

absl::StatusOr<std::string> MessageDecryptor::GetPublicKey(
    absl::FunctionRef<absl::StatusOr<std::string>(absl::string_view)> signer) {
  OkpCwt cwt{
      .public_key =
          OkpKey{
              .algorithm = crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm,
              .curve = crypto_internal::kX25519,
              .x = std::string(EVP_HPKE_MAX_PUBLIC_KEY_LENGTH, '\0'),
          },
  };
  size_t public_key_len = 0;
  if (EVP_HPKE_KEY_public_key(
          hpke_key_.get(), reinterpret_cast<uint8_t*>(cwt.public_key->x.data()),
          &public_key_len, cwt.public_key->x.size()) != 1) {
    return FCP_STATUS(fcp::INTERNAL)
           << "Failed to obtain public key from HPKE public/private keypair: "
           << ERR_reason_error_string(ERR_get_error());
  }
  cwt.public_key->x.resize(public_key_len);

  FCP_ASSIGN_OR_RETURN(std::string sig_structure,
                       cwt.BuildSigStructure(/*aad=*/""));
  FCP_ASSIGN_OR_RETURN(cwt.signature, signer(sig_structure));
  return cwt.Encode();
}

absl::StatusOr<std::string> MessageDecryptor::Decrypt(
    absl::string_view ciphertext, absl::string_view ciphertext_associated_data,
    absl::string_view encrypted_symmetric_key,
    absl::string_view encrypted_symmetric_key_associated_data,
    absl::string_view encapped_key) const {
  FCP_ASSIGN_OR_RETURN(
      std::string symmetric_key,
      crypto_internal::UnwrapSymmetricKey(
          hpke_key_.get(), hpke_kdf_, hpke_aead_, encrypted_symmetric_key,
          encapped_key, encrypted_symmetric_key_associated_data));
  // Cleanse the memory containing the symmetric key upon exiting the scope so
  // the key cannot be accessed outside this function.
  absl::Cleanup symmetric_key_cleanup = [&symmetric_key]() {
    OPENSSL_cleanse(symmetric_key.data(), symmetric_key.size());
  };

  FCP_ASSIGN_OR_RETURN(SymmetricKey key, SymmetricKey::Decode(symmetric_key));
  absl::Cleanup key_cleanup = [&key]() {
    OPENSSL_cleanse(key.k.data(), key.k.size());
  };
  if (key.algorithm != crypto_internal::kAeadAes128GcmSivFixedNonce) {
    return absl::InvalidArgumentError("unsupported symmetric key algorithm ");
  }

  bssl::UniquePtr<EVP_AEAD_CTX> aead_ctx(
      EVP_AEAD_CTX_new(aead_, reinterpret_cast<const uint8_t*>(key.k.data()),
                       key.k.size(), EVP_AEAD_DEFAULT_TAG_LENGTH));
  if (aead_ctx == nullptr) {
    return FCP_STATUS(fcp::INTERNAL)
           << "Failed to initialize EVP_AEAD_CTX: "
           << ERR_reason_error_string(ERR_get_error());
  }
  std::string plaintext(ciphertext.size(), '\0');
  size_t plaintext_len = 0;
  if (EVP_AEAD_CTX_open(
          aead_ctx.get(), reinterpret_cast<uint8_t*>(plaintext.data()),
          &plaintext_len, ciphertext.size(),
          reinterpret_cast<const uint8_t*>(kNonce.data()), kNonce.size(),
          reinterpret_cast<const uint8_t*>(ciphertext.data()),
          ciphertext.size(),
          reinterpret_cast<const uint8_t*>(ciphertext_associated_data.data()),
          ciphertext_associated_data.size()) != 1) {
    // Clear the plaintext buffer in case partial data was written.
    OPENSSL_cleanse(plaintext.data(), ciphertext.size());
    return FCP_STATUS(fcp::INVALID_ARGUMENT)
           << "AEAD decryption failed: "
           << ERR_reason_error_string(ERR_get_error());
  }
  plaintext.resize(plaintext_len);
  return plaintext;
}

namespace crypto_internal {

absl::StatusOr<WrapSymmetricKeyResult> WrapSymmetricKey(
    const EVP_HPKE_KEM* hpke_kem, const EVP_HPKE_KDF* hpke_kdf,
    const EVP_HPKE_AEAD* hpke_aead, absl::string_view symmetric_key,
    absl::string_view recipient_public_key, absl::string_view associated_data) {
  // Set up the HPKE context.
  bssl::ScopedEVP_HPKE_CTX hpke_ctx;
  std::string encapped_key(EVP_HPKE_MAX_ENC_LENGTH, '\0');
  size_t encapped_key_len = 0;
  if (EVP_HPKE_CTX_setup_sender(
          hpke_ctx.get(), reinterpret_cast<uint8_t*>(encapped_key.data()),
          &encapped_key_len, encapped_key.size(), hpke_kem, hpke_kdf, hpke_aead,
          reinterpret_cast<const uint8_t*>(recipient_public_key.data()),
          recipient_public_key.size(),
          reinterpret_cast<const uint8_t*>(kInfo.data()), kInfo.size()) != 1) {
    return FCP_STATUS(fcp::INVALID_ARGUMENT)
           << "Failed to set up HPKE: "
           << ERR_reason_error_string(ERR_get_error());
  }
  encapped_key.resize(encapped_key_len);
  // Seal the secret key.
  std::string encrypted_symmetric_key(
      symmetric_key.size() + EVP_HPKE_CTX_max_overhead(hpke_ctx.get()), '\0');
  size_t encrypted_symmetric_key_len = 0;
  if (EVP_HPKE_CTX_seal(
          hpke_ctx.get(),
          reinterpret_cast<uint8_t*>(encrypted_symmetric_key.data()),
          &encrypted_symmetric_key_len, encrypted_symmetric_key.size(),
          reinterpret_cast<const uint8_t*>(symmetric_key.data()),
          symmetric_key.size(),
          reinterpret_cast<const uint8_t*>(associated_data.data()),
          associated_data.size()) != 1) {
    return FCP_STATUS(fcp::INTERNAL)
           << "Failed to seal secret key: "
           << ERR_reason_error_string(ERR_get_error());
  }
  encrypted_symmetric_key.resize(encrypted_symmetric_key_len);
  return WrapSymmetricKeyResult{
      .encapped_key = std::move(encapped_key),
      .encrypted_symmetric_key = std::move(encrypted_symmetric_key)};
}

absl::StatusOr<std::string> UnwrapSymmetricKey(
    const EVP_HPKE_KEY* hpke_key, const EVP_HPKE_KDF* hpke_kdf,
    const EVP_HPKE_AEAD* hpke_aead, absl::string_view encrypted_symmetric_key,
    absl::string_view encapped_key, absl::string_view associated_data) {
  // Set up the HPKE context.
  bssl::ScopedEVP_HPKE_CTX hpke_ctx;

  if (EVP_HPKE_CTX_setup_recipient(
          hpke_ctx.get(), hpke_key, hpke_kdf, hpke_aead,
          reinterpret_cast<const uint8_t*>(encapped_key.data()),
          encapped_key.size(), reinterpret_cast<const uint8_t*>(kInfo.data()),
          kInfo.size()) != 1) {
    return FCP_STATUS(fcp::INVALID_ARGUMENT)
           << " Failed to set up HPKE context: "
           << ERR_reason_error_string(ERR_get_error());
  }

  std::string symmetric_key(encrypted_symmetric_key.size(), '\0');
  size_t symmetric_key_len = 0;
  if (EVP_HPKE_CTX_open(
          hpke_ctx.get(), reinterpret_cast<uint8_t*>(symmetric_key.data()),
          &symmetric_key_len, encrypted_symmetric_key.size(),
          reinterpret_cast<const uint8_t*>(encrypted_symmetric_key.data()),
          encrypted_symmetric_key.size(),
          reinterpret_cast<const uint8_t*>(associated_data.data()),
          associated_data.size()) != 1) {
    // Clear the symmetric key buffer in case partial data was written.
    OPENSSL_cleanse(symmetric_key.data(), symmetric_key.size());
    return FCP_STATUS(fcp::INVALID_ARGUMENT)
           << "Failed to unwrap symmetric key: "
           << ERR_reason_error_string(ERR_get_error());
  }
  symmetric_key.resize(symmetric_key_len);
  return symmetric_key;
}
}  // namespace crypto_internal

}  // namespace confidential_compute
}  // namespace fcp
