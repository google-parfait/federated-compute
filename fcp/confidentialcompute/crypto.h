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

// This file contains utilities for encrypting and decrypting messages to be
// used by confidential compute machines executing federated computations.
//
// The messages are encrypted with AEAD using a per-message generated symmetric
// key and then the symmetric key is encrypted using HPKE with the public key of
// the intended message recipient.
//
// Separating the step of encrypting the symmetric key from encrypting the
// message allows the AEAD symmetric key to be passed off to another party to
// be used to decrypt, without sharing the private key between the two parties.

#ifndef FCP_CONFIDENTIALCOMPUTE_CRYPTO_H_
#define FCP_CONFIDENTIALCOMPUTE_CRYPTO_H_

#include <optional>
#include <string>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "openssl/base.h"
#include "openssl/hpke.h"

namespace fcp {
namespace confidential_compute {

struct EncryptMessageResult {
  std::string ciphertext;
  std::string encapped_key;
  std::string encrypted_symmetric_key;
};

// Encrypts messages for particular intended recipients.
//
// Returns an EncryptMessageResult containing the ciphertext, the encrypted
// symmetric key to be decrypted and used to decrypt the message, and the
// encapsulated key, which along with the recipient's private key, can be used
// for decrypting the encrypted symmetric key.
//
// IMPORTANT: This class DOES NOT validate the public key passed to Encrypt. If
// the public key is a CWT, it's the caller's responsibility to verify the
// signature and claims.
class MessageEncryptor {
 public:
  MessageEncryptor();

  absl::StatusOr<EncryptMessageResult> Encrypt(
      absl::string_view plaintext, absl::string_view recipient_public_key,
      absl::string_view associated_data) const;

 private:
  const EVP_HPKE_KEM* hpke_kem_;
  const EVP_HPKE_KDF* hpke_kdf_;
  const EVP_HPKE_AEAD* hpke_aead_;
  const EVP_AEAD* aead_;
};

// Decrypts messages intended for this recipient.
class MessageDecryptor {
 public:
  MessageDecryptor();

  // MessageDecryptor is not copyable or moveable due to the use of
  // bssl::ScopedEVP_HPKE_KEY.
  MessageDecryptor(const MessageDecryptor& other) = delete;
  MessageDecryptor& operator=(const MessageDecryptor& other) = delete;

  // Obtain a public key that can be used to encrypt messages that this class
  // can subsequently decrypt.
  //
  // Generates a new public key if the key has not yet been generated, otherwise
  // returns the key that was previously generated on the first call to this
  // method.
  //
  // If `signer` is provided, the result will be a signed CBOR Web Token
  // containing the public key. Otherwise, it'll be the raw public key.
  // TODO: b/313640181 - make signer required once all keys are structured.
  //
  // This function must be called before Decrypt.
  absl::StatusOr<std::string> GetPublicKey(
      std::optional<
          absl::FunctionRef<absl::StatusOr<std::string>(absl::string_view)>>
          signer = std::nullopt);

  // Decrypts `ciphertext` using a symmetric key produced by decrypting
  // `encrypted_symmetric_key` with the `encapped_key` and the private key
  // corresponding to the public key returned by `GetPublicKey`.
  //
  // The ciphertext to decrypt should have been produced by
  // `MessageEncryptor::Encrypt` or an equivalent implementation.
  //
  // `ciphertext_associated_data` and `encrypted_symmetric_key_associated_data`
  // may differ in the case that the symmetric key was rewrapped by an
  // intermediary for decryption by this recipient.
  //
  // Returns the decrypted plaintext, a FAILED_PRECONDITION status if
  // `GetPublicKey` has never been called for this class, or an INVALID_ARGUMENT
  // status if the ciphertext could not be decrypted with the provided
  // arguments.
  absl::StatusOr<std::string> Decrypt(
      absl::string_view ciphertext,
      absl::string_view ciphertext_associated_data,
      absl::string_view encrypted_symmetric_key,
      absl::string_view encrypted_symmetric_key_associated_data,
      absl::string_view encapped_key) const;

 private:
  const EVP_HPKE_KEM* hpke_kem_;
  const EVP_HPKE_KDF* hpke_kdf_;
  const EVP_HPKE_AEAD* hpke_aead_;
  bssl::ScopedEVP_HPKE_KEY hpke_key_;
  const EVP_AEAD* aead_;
};

// Helper functions exposed for testing purposes.
namespace crypto_internal {

// Supported COSE Algorithms; see ../protos/confidentialcompute/cbor_ids.md.
enum CoseAlgorithm {
  kHpkeBaseX25519Sha256Aes128Gcm = -65537,
  kAeadAes128GcmSivFixedNonce = -65538,
};

// Supported COSE Elliptic Curves; see
// https://www.iana.org/assignments/cose/cose.xhtml#elliptic-curves.
enum CoseEllipticCurve {
  kX25519 = 4,
};

struct WrapSymmetricKeyResult {
  std::string encapped_key;
  std::string encrypted_symmetric_key;
};

// Wraps a symmetric encryption key using HPKE.
absl::StatusOr<WrapSymmetricKeyResult> WrapSymmetricKey(
    const EVP_HPKE_KEM* hpke_kem, const EVP_HPKE_KDF* hpke_kdf,
    const EVP_HPKE_AEAD* hpke_aead, absl::string_view symmetric_key,
    absl::string_view recipient_public_key, absl::string_view associated_data);

// Unwraps a symmetric encryption key using HPKE.
absl::StatusOr<std::string> UnwrapSymmetricKey(
    const EVP_HPKE_KEY* hpke_key, const EVP_HPKE_KDF* hpke_kdf,
    const EVP_HPKE_AEAD* hpke_aead, absl::string_view encrypted_symmetric_key,
    absl::string_view encapped_key, absl::string_view associated_data);

}  // namespace crypto_internal

}  // namespace confidential_compute
}  // namespace fcp

#endif  // FCP_CONFIDENTIALCOMPUTE_CRYPTO_H_
