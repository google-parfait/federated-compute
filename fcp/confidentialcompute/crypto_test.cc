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
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "fcp/base/monitoring.h"
#include "fcp/testing/testing.h"
#include "openssl/base.h"
#include "openssl/hpke.h"

namespace fcp {
namespace confidential_compute {
namespace {

TEST(CryptoTest, EncryptAndDecrypt) {
  std::string message = "some plaintext message";
  std::string associated_data = "plaintext associated data";

  MessageEncryptor encryptor;
  MessageDecryptor decryptor;

  absl::StatusOr<std::string> recipient_public_key = decryptor.GetPublicKey();
  ASSERT_OK(recipient_public_key);

  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, *recipient_public_key, associated_data);
  ASSERT_OK(encrypt_result);

  absl::StatusOr<std::string> decrypt_result =
      decryptor.Decrypt(encrypt_result->ciphertext, associated_data,
                        encrypt_result->encrypted_symmetric_key,
                        associated_data, encrypt_result->encapped_key);
  ASSERT_OK(decrypt_result);
  EXPECT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, EncryptRewrapKeyAndDecrypt) {
  std::string message = "some plaintext message";
  std::string message_associated_data = "plaintext associated data";

  MessageEncryptor encryptor;
  MessageDecryptor decryptor;

  absl::StatusOr<std::string> recipient_public_key = decryptor.GetPublicKey();
  ASSERT_OK(recipient_public_key);

  // Encrypt the symmetric key with the public key of an intermediary.
  bssl::ScopedEVP_HPKE_KEY intermediary_key;
  const EVP_HPKE_KEM* kem = EVP_hpke_x25519_hkdf_sha256();
  const EVP_HPKE_KDF* kdf = EVP_hpke_hkdf_sha256();
  const EVP_HPKE_AEAD* aead = EVP_hpke_aes_128_gcm();
  ASSERT_EQ(EVP_HPKE_KEY_generate(intermediary_key.get(), kem), 1);
  size_t public_key_len;
  std::string intermediary_public_key(EVP_HPKE_MAX_PUBLIC_KEY_LENGTH, '\0');
  ASSERT_EQ(EVP_HPKE_KEY_public_key(
                intermediary_key.get(),
                reinterpret_cast<uint8_t*>(intermediary_public_key.data()),
                &public_key_len, intermediary_public_key.size()),
            1);
  intermediary_public_key.resize(public_key_len);
  absl::StatusOr<EncryptMessageResult> encrypt_result = encryptor.Encrypt(
      message, intermediary_public_key, message_associated_data);
  ASSERT_OK(encrypt_result);

  // Have the intermediary rewrap the symmetric key with the public key of the
  // final recipient.
  absl::StatusOr<std::string> symmetric_key =
      crypto_internal::UnwrapSymmetricKey(
          intermediary_key.get(), kdf, aead,
          encrypt_result->encrypted_symmetric_key, encrypt_result->encapped_key,
          message_associated_data);
  ASSERT_OK(symmetric_key);

  std::string symmetric_key_associated_data =
      "rewrap symmetric key associated data";
  absl::StatusOr<crypto_internal::WrapSymmetricKeyResult>
      rewrapped_symmetric_key_result = crypto_internal::WrapSymmetricKey(
          kem, kdf, aead, *symmetric_key, *recipient_public_key,
          symmetric_key_associated_data);
  ASSERT_OK(rewrapped_symmetric_key_result);

  // The final recipient should be able to decrypt the message using the
  // rewrapped key.
  absl::StatusOr<std::string> decrypt_result =
      decryptor.Decrypt(encrypt_result->ciphertext, message_associated_data,
                        rewrapped_symmetric_key_result->encrypted_symmetric_key,
                        symmetric_key_associated_data,
                        rewrapped_symmetric_key_result->encapped_key);
  ASSERT_OK(decrypt_result);
  EXPECT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, EncryptWithInvalidPublicKeyFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "plaintext associated data";

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, "invalid public key", associated_data);
  ASSERT_THAT(encrypt_result, IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, EncryptTwiceWithSameKeyUsesDifferentSymmetricKey) {
  std::string message = "some plaintext message";
  std::string message_associated_data = "plaintext associated data";

  MessageEncryptor encryptor;

  // Encrypt the symmetric key with the public key of an intermediary.
  bssl::ScopedEVP_HPKE_KEY intermediary_key;
  const EVP_HPKE_KEM* kem = EVP_hpke_x25519_hkdf_sha256();
  const EVP_HPKE_KDF* kdf = EVP_hpke_hkdf_sha256();
  const EVP_HPKE_AEAD* aead = EVP_hpke_aes_128_gcm();
  ASSERT_EQ(EVP_HPKE_KEY_generate(intermediary_key.get(), kem), 1);
  size_t public_key_len;
  std::string intermediary_public_key(EVP_HPKE_MAX_PUBLIC_KEY_LENGTH, '\0');
  ASSERT_EQ(EVP_HPKE_KEY_public_key(
                intermediary_key.get(),
                reinterpret_cast<uint8_t*>(intermediary_public_key.data()),
                &public_key_len, intermediary_public_key.size()),
            1);
  intermediary_public_key.resize(public_key_len);

  absl::StatusOr<EncryptMessageResult> encrypt_result_1 = encryptor.Encrypt(
      message, intermediary_public_key, message_associated_data);
  ASSERT_OK(encrypt_result_1);

  // Encrypt the same plaintext a second time. This should produce a different
  // ciphertext as a new symmetric key will be generated.
  absl::StatusOr<EncryptMessageResult> encrypt_result_2 = encryptor.Encrypt(
      message, intermediary_public_key, message_associated_data);
  ASSERT_OK(encrypt_result_2);
  ASSERT_NE(encrypt_result_1->ciphertext, encrypt_result_2->ciphertext);

  // Unwrap the symmetric key for each EncryptMessageResult.
  absl::StatusOr<std::string> symmetric_key_1 =
      crypto_internal::UnwrapSymmetricKey(
          intermediary_key.get(), kdf, aead,
          encrypt_result_1->encrypted_symmetric_key,
          encrypt_result_1->encapped_key, message_associated_data);
  ASSERT_OK(symmetric_key_1);

  absl::StatusOr<std::string> symmetric_key_2 =
      crypto_internal::UnwrapSymmetricKey(
          intermediary_key.get(), kdf, aead,
          encrypt_result_2->encrypted_symmetric_key,
          encrypt_result_2->encapped_key, message_associated_data);
  ASSERT_OK(symmetric_key_2);

  // The symmetric keys should be different.
  ASSERT_NE(*symmetric_key_1, *symmetric_key_2);
}

TEST(CryptoTest, DecryptWithWrongKeyFails) {
  std::string message = "some plaintext message";
  std::string message_associated_data = "plaintext associated data";

  MessageEncryptor encryptor;
  MessageDecryptor decryptor;

  absl::StatusOr<std::string> recipient_public_key = decryptor.GetPublicKey();
  ASSERT_OK(recipient_public_key);

  // Encrypt the symmetric key with the public key of an intermediary.
  bssl::ScopedEVP_HPKE_KEY intermediary_key;
  const EVP_HPKE_KEM* kem = EVP_hpke_x25519_hkdf_sha256();
  ASSERT_EQ(EVP_HPKE_KEY_generate(intermediary_key.get(), kem), 1);
  size_t public_key_len;
  std::string intermediary_public_key(EVP_HPKE_MAX_PUBLIC_KEY_LENGTH, '\0');
  ASSERT_EQ(EVP_HPKE_KEY_public_key(
                intermediary_key.get(),
                reinterpret_cast<uint8_t*>(intermediary_public_key.data()),
                &public_key_len, intermediary_public_key.size()),
            1);
  intermediary_public_key.resize(public_key_len);
  absl::StatusOr<EncryptMessageResult> encrypt_result = encryptor.Encrypt(
      message, intermediary_public_key, message_associated_data);
  ASSERT_OK(encrypt_result);

  // Attempting to decrypt without the symmetric key being rewrapped with the
  // public key of the recipient should fail.
  absl::StatusOr<std::string> decrypt_result =
      decryptor.Decrypt(encrypt_result->ciphertext, message_associated_data,
                        encrypt_result->encrypted_symmetric_key,
                        message_associated_data, encrypt_result->encapped_key);
  EXPECT_THAT(decrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, DecryptWithWrongCiphertextAssociatedDataFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "plaintext associated data";

  MessageEncryptor encryptor;
  MessageDecryptor decryptor;

  absl::StatusOr<std::string> recipient_public_key = decryptor.GetPublicKey();
  ASSERT_OK(recipient_public_key);

  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, *recipient_public_key, associated_data);
  ASSERT_OK(encrypt_result);

  absl::StatusOr<std::string> decrypt_result = decryptor.Decrypt(
      encrypt_result->ciphertext, "wrong ciphertext associated data",
      encrypt_result->encrypted_symmetric_key, associated_data,
      encrypt_result->encapped_key);
  EXPECT_THAT(decrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, DecryptWithWrongSymmetricKeyAssociatedDataFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "associated data";

  MessageEncryptor encryptor;
  MessageDecryptor decryptor;

  absl::StatusOr<std::string> recipient_public_key = decryptor.GetPublicKey();
  ASSERT_OK(recipient_public_key);

  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, *recipient_public_key, associated_data);
  ASSERT_OK(encrypt_result);

  absl::StatusOr<std::string> decrypt_result = decryptor.Decrypt(
      encrypt_result->ciphertext, associated_data,
      encrypt_result->encrypted_symmetric_key,
      "wrong symmetric key associated data", encrypt_result->encapped_key);
  EXPECT_THAT(decrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, DecryptWithWrongEncappedKeyFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "associated data";

  MessageEncryptor encryptor;
  MessageDecryptor decryptor;

  absl::StatusOr<std::string> recipient_public_key = decryptor.GetPublicKey();
  ASSERT_OK(recipient_public_key);

  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, *recipient_public_key, associated_data);
  ASSERT_OK(encrypt_result);

  absl::StatusOr<std::string> decrypt_result = decryptor.Decrypt(
      encrypt_result->ciphertext, associated_data,
      encrypt_result->encrypted_symmetric_key,
      "wrong symmetric key associated data", encrypt_result->encapped_key);
  EXPECT_THAT(decrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, DecryptWithInvalidCiphertextFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "associated data";

  MessageEncryptor encryptor;
  MessageDecryptor decryptor;

  absl::StatusOr<std::string> recipient_public_key = decryptor.GetPublicKey();
  ASSERT_OK(recipient_public_key);

  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, *recipient_public_key, associated_data);
  ASSERT_OK(encrypt_result);

  absl::StatusOr<std::string> decrypt_result =
      decryptor.Decrypt("invalid ciphertext", associated_data,
                        encrypt_result->encrypted_symmetric_key,
                        associated_data, encrypt_result->encapped_key);
  EXPECT_THAT(decrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, DecryptWithInvalidSymmetricKeyFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "associated data";

  MessageEncryptor encryptor;
  MessageDecryptor decryptor;

  absl::StatusOr<std::string> recipient_public_key = decryptor.GetPublicKey();
  ASSERT_OK(recipient_public_key);

  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, *recipient_public_key, associated_data);
  ASSERT_OK(encrypt_result);

  absl::StatusOr<std::string> decrypt_result = decryptor.Decrypt(
      encrypt_result->ciphertext, associated_data, "invalid symmetric key",
      associated_data, encrypt_result->encapped_key);
  EXPECT_THAT(decrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, DecryptWithInvalidEncappedKeyFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "associated data";

  MessageEncryptor encryptor;
  MessageDecryptor decryptor;

  absl::StatusOr<std::string> recipient_public_key = decryptor.GetPublicKey();
  ASSERT_OK(recipient_public_key);

  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, *recipient_public_key, associated_data);
  ASSERT_OK(encrypt_result);

  absl::StatusOr<std::string> decrypt_result =
      decryptor.Decrypt(encrypt_result->ciphertext, associated_data,
                        encrypt_result->encrypted_symmetric_key,
                        associated_data, "invalid encapped key");
  EXPECT_THAT(decrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

}  // namespace
}  // namespace confidential_compute
}  // namespace fcp
