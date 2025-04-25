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
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/struct.pb.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "fcp/base/digest.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "openssl/aead.h"
#include "openssl/base.h"
#include "openssl/bn.h"
#include "openssl/ec.h"
#include "openssl/ec_key.h"
#include "openssl/ecdsa.h"
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

// The expected length of COSE ECDSA256 signatures, when encoded as per RFC 8152
// section 8.1.
constexpr size_t kCoseP256SignatureLen = 64;

namespace {
// The size of the session-level nonce.
constexpr size_t kNonceSize = 16;

// Generates a blob-level nonce from a session-level nonce and a blob counter.
std::string GenerateBlobNonce(std::string session_nonce, uint32_t counter) {
  // We assume that the untrusted and trusted code are running on a machine with
  // the same endianness.
  std::string blob_nonce(session_nonce.length() + sizeof(uint32_t), '\0');
  std::memcpy(blob_nonce.data(), session_nonce.data(), session_nonce.length());
  std::memcpy(blob_nonce.data() + session_nonce.length(), &counter,
              sizeof(uint32_t));
  return blob_nonce;
}

// Parses serialized COSE_Key decryption keys and converts them to
// EVP_HPKE_KEYs, grouped by key ID.
absl::flat_hash_map<std::string, std::vector<bssl::ScopedEVP_HPKE_KEY>>
ProcessDecryptionKeys(const std::vector<absl::string_view>& decryption_keys) {
  absl::flat_hash_map<std::string, std::vector<bssl::ScopedEVP_HPKE_KEY>>
      processed_keys;
  for (absl::string_view decryption_key : decryption_keys) {
    absl::StatusOr<OkpKey> key = OkpKey::Decode(decryption_key);
    if (!key.ok()) {
      FCP_LOG(WARNING) << "Skipping invalid key: " << key.status();
      continue;
    }
    if (key->algorithm != crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm ||
        key->curve != crypto_internal::kX25519) {
      FCP_LOG(WARNING) << "Skipping key with unsupported algorithm or curve";
      continue;
    }

    bssl::ScopedEVP_HPKE_KEY hpke_key;
    if (EVP_HPKE_KEY_init(hpke_key.get(), EVP_hpke_x25519_hkdf_sha256(),
                          reinterpret_cast<const uint8_t*>(key->d.data()),
                          key->d.size()) != 1) {
      FCP_LOG(WARNING) << "Skipping key with invalid private key";
      continue;
    }

    processed_keys[key->key_id].push_back(std::move(hpke_key));
  }
  return processed_keys;
}

}  // namespace

NonceChecker::NonceChecker() {
  std::string nonce(kNonceSize, '\0');
  // BoringSSL documentation says that it always returns 1 so we don't check
  // the return value.
  (void)RAND_bytes(reinterpret_cast<unsigned char*>(nonce.data()),
                   nonce.size());
  session_nonce_ = std::move(nonce);
}

absl::StatusOr<NonceAndCounter> NonceGenerator::GetNextBlobNonce() {
  if (counter_ == UINT32_MAX) {
    return absl::InternalError("Counter has overflowed.");
  }
  std::string next_blob_nonce = GenerateBlobNonce(session_nonce_, counter_);

  NonceAndCounter result = {.blob_nonce = next_blob_nonce, .counter = counter_};
  counter_++;
  return result;
}

absl::Status NonceChecker::CheckBlobNonce(
    const confidentialcompute::BlobMetadata& metadata) {
  if (metadata.has_unencrypted()) {
    return absl::OkStatus();
  }

  if (metadata.hpke_plus_aead_data().counter() == UINT32_MAX) {
    return absl::InternalError("Counter has overflowed.");
  }
  if (metadata.hpke_plus_aead_data().counter() < counter_) {
    return absl::PermissionDeniedError(
        absl::StrFormat("Blob counter %d is less than the minimum expected "
                        "value %d; caller may be "
                        "attempting to reuse a previously seen nonce.",
                        metadata.hpke_plus_aead_data().counter(), counter_));
  }
  if (metadata.hpke_plus_aead_data()
          .rewrapped_symmetric_key_associated_data()
          .nonce() !=
      GenerateBlobNonce(session_nonce_,
                        metadata.hpke_plus_aead_data().counter())) {
    return absl::PermissionDeniedError(
        "RewrappedAssociatedData nonce does not match the expected value.");
  }
  counter_ = metadata.hpke_plus_aead_data().counter() + 1;
  return absl::OkStatus();
}

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

  // All (untagged) CWTs start with '\x84' (4 element array). COSE_Keys are a
  // map type, so they always have a different prefix.
  OkpKey okp_key;
  if (absl::StartsWith(recipient_public_key, "\x84")) {
    FCP_ASSIGN_OR_RETURN(OkpCwt cwt, OkpCwt::Decode(recipient_public_key));
    if (!cwt.public_key) {
      return absl::InvalidArgumentError("CWT has no public key");
    }
    okp_key = std::move(*cwt.public_key);
  } else {
    FCP_ASSIGN_OR_RETURN(okp_key, OkpKey::Decode(recipient_public_key));
  }
  if (okp_key.algorithm != crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm ||
      okp_key.curve != crypto_internal::kX25519) {
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
                                        serialized_symmetric_key, okp_key.x,
                                        associated_data));

  return EncryptMessageResult{
      .ciphertext = std::move(ciphertext),
      .encapped_key = std::move(wrap_symmetric_key_result.encapped_key),
      .encrypted_symmetric_key =
          std::move(wrap_symmetric_key_result.encrypted_symmetric_key),
  };
}

MessageDecryptor::MessageDecryptor(
    google::protobuf::Struct config_properties,
    const std::vector<absl::string_view>& decryption_keys)
    : config_properties_(std::move(config_properties)),
      decryption_keys_(ProcessDecryptionKeys(decryption_keys)),
      hpke_kem_(EVP_hpke_x25519_hkdf_sha256()),
      hpke_kdf_(EVP_hpke_hkdf_sha256()),
      hpke_aead_(EVP_hpke_aes_128_gcm()),
      hpke_key_(),
      aead_(EVP_aead_aes_128_gcm_siv()) {
  FCP_CHECK(EVP_HPKE_KEY_generate(hpke_key_.get(), hpke_kem_) == 1)
      << "Failed to generate HPKE public/private keypair: "
      << ERR_reason_error_string(ERR_get_error());
}

absl::StatusOr<std::string> MessageDecryptor::GetPublicKey(
    absl::FunctionRef<absl::StatusOr<std::string>(absl::string_view)> signer,
    int64_t signer_algorithm) const {
  OkpCwt cwt{
      .algorithm = signer_algorithm,
      .public_key =
          OkpKey{
              .algorithm = crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm,
              .curve = crypto_internal::kX25519,
              .x = std::string(EVP_HPKE_MAX_PUBLIC_KEY_LENGTH, '\0'),
          },
      .config_properties = config_properties_,
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
                       cwt.BuildSigStructureForSigning(/*aad=*/""));
  FCP_ASSIGN_OR_RETURN(cwt.signature, signer(sig_structure));
  return cwt.Encode();
}

absl::StatusOr<std::string> MessageDecryptor::Decrypt(
    absl::string_view ciphertext, absl::string_view ciphertext_associated_data,
    absl::string_view encrypted_symmetric_key,
    absl::string_view encrypted_symmetric_key_associated_data,
    absl::string_view encapped_key, absl::string_view key_id) const {
  std::optional<std::string> symmetric_key =
      UnwrapSymmetricKeyWithDecryptionKeys(
          encrypted_symmetric_key, encrypted_symmetric_key_associated_data,
          encapped_key, key_id);
  if (!symmetric_key.has_value()) {
    FCP_ASSIGN_OR_RETURN(
        symmetric_key,
        crypto_internal::UnwrapSymmetricKey(
            hpke_key_.get(), hpke_kdf_, hpke_aead_, encrypted_symmetric_key,
            encapped_key, encrypted_symmetric_key_associated_data));
  }
  // Cleanse the memory containing the symmetric key upon exiting the scope so
  // the key cannot be accessed outside this function.
  absl::Cleanup symmetric_key_cleanup = [&symmetric_key]() {
    OPENSSL_cleanse(symmetric_key->data(), symmetric_key->size());
  };
  return DecryptReleasedResult(ciphertext, ciphertext_associated_data,
                               *symmetric_key);
}

absl::StatusOr<std::string> MessageDecryptor::DecryptReleasedResult(
    absl::string_view ciphertext, absl::string_view associated_data,
    absl::string_view symmetric_key) const {
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
          reinterpret_cast<const uint8_t*>(associated_data.data()),
          associated_data.size()) != 1) {
    // Clear the plaintext buffer in case partial data was written.
    OPENSSL_cleanse(plaintext.data(), ciphertext.size());
    return FCP_STATUS(fcp::INVALID_ARGUMENT)
           << "AEAD decryption failed: "
           << ERR_reason_error_string(ERR_get_error());
  }
  plaintext.resize(plaintext_len);
  return plaintext;
}

std::optional<std::string>
MessageDecryptor::UnwrapSymmetricKeyWithDecryptionKeys(
    absl::string_view encrypted_symmetric_key,
    absl::string_view encrypted_symmetric_key_associated_data,
    absl::string_view encapped_key, absl::string_view key_id) const {
  // Fail immediately if no decryption keys were provided or if a key_id wasn't
  // provided.
  if (decryption_keys_.empty() || key_id.empty()) {
    return std::nullopt;
  }

  // Attempt to decrypt with each key with a matching key ID. If all matching
  // keys fail, return nullopt so that the caller can attempt to decrypt with
  // the internally generated key (just in case the MessageDecryptor is being
  // used to decrypt a mix of inputs using the Ledger and KMS).
  auto it = decryption_keys_.find(key_id);
  if (it == decryption_keys_.end()) {
    return std::nullopt;
  }
  for (const auto& hpke_key : it->second) {
    absl::StatusOr<std::string> symmetric_key =
        crypto_internal::UnwrapSymmetricKey(
            hpke_key.get(), hpke_kdf_, hpke_aead_, encrypted_symmetric_key,
            encapped_key, encrypted_symmetric_key_associated_data);
    if (symmetric_key.ok()) {
      return *std::move(symmetric_key);
    }
  }
  return std::nullopt;
}

EcdsaP256R1Signer EcdsaP256R1Signer::Create() {
  const EC_GROUP* p256_group = EC_group_p256();
  bssl::UniquePtr<EC_KEY> key(EC_KEY_new());
  FCP_CHECK(key) << "Failed to allocate EC_KEY: "
                 << ERR_reason_error_string(ERR_get_error());

  // Generate a new key pair with which we can generate a signature.
  FCP_CHECK(EC_KEY_set_group(key.get(), p256_group) == 1)
      << "Failed to set key group: "
      << ERR_reason_error_string(ERR_get_error());
  FCP_CHECK(EC_KEY_generate_key(key.get()) == 1)
      << "Failed to generate key: " << ERR_reason_error_string(ERR_get_error());

  // Extract the public key in octet encoded form for later use by the
  // verifier.
  bssl::UniquePtr<uint8_t> public_key_buf;
  size_t public_key_size;
  uint8_t* public_key_ptr = nullptr;
  public_key_size =
      EC_KEY_key2buf(key.get(), POINT_CONVERSION_UNCOMPRESSED, &public_key_ptr,
                     /*ctx=*/nullptr);
  FCP_CHECK(public_key_size > 0) << "Failed encode public key: "
                                 << ERR_reason_error_string(ERR_get_error());
  // Ensure the buffer is freed when it goes out of scope.
  public_key_buf.reset(public_key_ptr);
  return EcdsaP256R1Signer(
      std::move(key), std::string(reinterpret_cast<char*>(public_key_buf.get()),
                                  public_key_size));
}

EcdsaP256R1Signer::EcdsaP256R1Signer(bssl::UniquePtr<EC_KEY> key,
                                     std::string encoded_public_key)
    : key_(std::move(key)),
      encoded_public_key_(std::move(encoded_public_key)) {}

std::string EcdsaP256R1Signer::GetPublicKey() const {
  return encoded_public_key_;
}

std::string EcdsaP256R1Signer::Sign(absl::string_view data) const {
  // Compute a digest over the data.
  auto data_digest = ComputeSHA256(data);

  // Calculate the signature over the digest.
  bssl::UniquePtr<ECDSA_SIG> sig(
      ECDSA_do_sign(reinterpret_cast<uint8_t*>(data_digest.data()),
                    data_digest.size(), key_.get()));
  FCP_CHECK(sig) << "Failed to calculate signature: "
                 << ERR_reason_error_string(ERR_get_error());

  // Return the signature as by concatenating the R and S signature components
  // as 32 byte long big endian integers.
  std::string signature_buf(kCoseP256SignatureLen, '\0');
  FCP_CHECK(BN_bn2bin_padded(reinterpret_cast<uint8_t*>(signature_buf.data()),
                             kCoseP256SignatureLen / 2,
                             ECDSA_SIG_get0_r(sig.get())) == 1)
      << "Failed to serialize signature R value: "
      << ERR_reason_error_string(ERR_get_error());
  FCP_CHECK(BN_bn2bin_padded(reinterpret_cast<uint8_t*>(signature_buf.data()) +
                                 kCoseP256SignatureLen / 2,
                             kCoseP256SignatureLen / 2,
                             ECDSA_SIG_get0_s(sig.get())) == 1)
      << "Failed to serialize signature S value: "
      << ERR_reason_error_string(ERR_get_error());
  return signature_buf;
}

absl::StatusOr<EcdsaP256R1SignatureVerifier>
EcdsaP256R1SignatureVerifier::Create(absl::string_view public_key) {
  bssl::UniquePtr<EC_KEY> key(EC_KEY_new());
  FCP_CHECK(key) << "Failed to allocate EC_KEY: "
                 << ERR_reason_error_string(ERR_get_error());

  // Initialize the public key from the encoded `public_key` parameter.
  FCP_CHECK(EC_KEY_set_group(key.get(), EC_group_p256()) == 1)
      << "Failed to set key group: "
      << ERR_reason_error_string(ERR_get_error());
  if (EC_KEY_oct2key(key.get(),
                     reinterpret_cast<const uint8_t*>(public_key.data()),
                     public_key.size(),
                     /*ctx=*/nullptr) != 1) {
    return FCP_STATUS(fcp::INVALID_ARGUMENT)
           << "Failed to initialize public key: "
           << ERR_reason_error_string(ERR_get_error());
  }

  return EcdsaP256R1SignatureVerifier(std::move(key));
}

EcdsaP256R1SignatureVerifier::EcdsaP256R1SignatureVerifier(
    bssl::UniquePtr<EC_KEY> public_key)
    : public_key_(std::move(public_key)) {}

absl::Status EcdsaP256R1SignatureVerifier::Verify(
    absl::string_view data, absl::string_view signature) const {
  // Compute a digest over the data.
  auto digest = ComputeSHA256(data);

  // Parse the signature.
  if (signature.size() != kCoseP256SignatureLen) {
    return FCP_STATUS(fcp::INVALID_ARGUMENT)
           << "Invalid signature size: " << signature.size() << " - "
           << ERR_reason_error_string(ERR_get_error());
  }
  bssl::UniquePtr<BIGNUM> r(
      BN_bin2bn(reinterpret_cast<const uint8_t*>(signature.data()),
                kCoseP256SignatureLen / 2, nullptr));
  if (!r) {
    return FCP_STATUS(fcp::INVALID_ARGUMENT)
           << "Failed to convert signature R value to BIGNUM: "
           << ERR_reason_error_string(ERR_get_error());
  }
  bssl::UniquePtr<BIGNUM> s(
      BN_bin2bn(reinterpret_cast<const uint8_t*>(signature.data() +
                                                 kCoseP256SignatureLen / 2),
                kCoseP256SignatureLen / 2, nullptr));
  if (!s) {
    return FCP_STATUS(fcp::INVALID_ARGUMENT)
           << "Failed to convert signature S value to BIGNUM: "
           << ERR_reason_error_string(ERR_get_error());
  }

  // Initialize the signature based on the parsed R and S values.
  bssl::UniquePtr<ECDSA_SIG> sig(ECDSA_SIG_new());
  FCP_CHECK(sig) << "Failed to allocate ECDSA_SIG: "
                 << ERR_reason_error_string(ERR_get_error());
  // Note: ECDSA_SIG_set0 takes ownership of the `r` and `s` values, so we must
  // use `release()`.
  if (ECDSA_SIG_set0(sig.get(), r.release(), s.release()) != 1) {
    return FCP_STATUS(fcp::INVALID_ARGUMENT)
           << "Failed to set signature: "
           << ERR_reason_error_string(ERR_get_error());
  }

  // Verify the signature over the digest.
  if (ECDSA_do_verify(reinterpret_cast<const uint8_t*>(digest.data()),
                      digest.size(), sig.get(), public_key_.get()) != 1) {
    return FCP_STATUS(fcp::INVALID_ARGUMENT)
           << "Invalid signature: " << signature << " - "
           << ERR_reason_error_string(ERR_get_error());
  }
  return absl::OkStatus();
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
