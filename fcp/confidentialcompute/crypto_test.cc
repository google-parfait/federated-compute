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

#include "google/protobuf/struct.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/testing/testing.h"
#include "openssl/base.h"
#include "openssl/hpke.h"

namespace fcp {
namespace confidential_compute {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::SaveArg;

// Helper function to generate a new public and private key pair.
// This function uses output parameters because ScopedEVP_HPKE_KEY is not
// movable in older versions of BoringSSL.
void GenerateKeyPair(const EVP_HPKE_KEM& kem, std::string& public_key,
                     bssl::ScopedEVP_HPKE_KEY& private_key) {
  CHECK_EQ(EVP_HPKE_KEY_generate(private_key.get(), &kem), 1);
  size_t public_key_len;
  public_key.resize(EVP_HPKE_MAX_PUBLIC_KEY_LENGTH, '\0');
  CHECK_EQ(EVP_HPKE_KEY_public_key(
               private_key.get(), reinterpret_cast<uint8_t*>(public_key.data()),
               &public_key_len, public_key.size()),
           1);
  public_key.resize(public_key_len);
}

TEST(CryptoTest, GetPublicKey) {
  testing::MockFunction<absl::StatusOr<std::string>(absl::string_view)> signer;
  std::string sig_structure;
  EXPECT_CALL(signer, Call(_))
      .WillOnce(DoAll(SaveArg<0>(&sig_structure), Return("signature")));

  google::protobuf::Struct config_properties;
  (*config_properties.mutable_fields())["key"].set_string_value("value");

  MessageDecryptor decryptor(config_properties);
  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey(signer.AsStdFunction());
  ASSERT_OK(recipient_public_key);

  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(*recipient_public_key);
  ASSERT_OK(cwt);
  ASSERT_NE(cwt->public_key, std::nullopt);
  EXPECT_EQ(cwt->public_key->algorithm,
            crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm);
  EXPECT_EQ(cwt->public_key->curve, crypto_internal::kX25519);
  EXPECT_NE(cwt->public_key->x, "");
  EXPECT_THAT(cwt->config_properties, EqualsProto(config_properties));
  EXPECT_EQ(cwt->signature, "signature");

  // The signature structure is a COSE implementation detail, but it should at
  // least contain the public key.
  EXPECT_THAT(sig_structure, HasSubstr(cwt->public_key->x));
}

TEST(CryptoTest, GetPublicKeyCwtSigningError) {
  testing::MockFunction<absl::StatusOr<std::string>(absl::string_view)> signer;
  EXPECT_CALL(signer, Call(_))
      .WillOnce(Return(absl::FailedPreconditionError("")));

  MessageDecryptor decryptor;
  EXPECT_THAT(decryptor.GetPublicKey(signer.AsStdFunction()),
              IsCode(FAILED_PRECONDITION));
}

TEST(CryptoTest, EncryptAndDecrypt) {
  std::string message = "some plaintext message";
  std::string associated_data = "plaintext associated data";

  MessageEncryptor encryptor;
  MessageDecryptor decryptor;

  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; });
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

  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; });
  ASSERT_OK(recipient_public_key);
  absl::StatusOr<OkpCwt> recipient_cwt = OkpCwt::Decode(*recipient_public_key);
  ASSERT_OK(recipient_cwt);
  ASSERT_TRUE(recipient_cwt->public_key.has_value());

  // Encrypt the symmetric key with the public key of an intermediary.
  const EVP_HPKE_KEM* kem = EVP_hpke_x25519_hkdf_sha256();
  const EVP_HPKE_KDF* kdf = EVP_hpke_hkdf_sha256();
  const EVP_HPKE_AEAD* aead = EVP_hpke_aes_128_gcm();
  std::string intermediary_public_key;
  bssl::ScopedEVP_HPKE_KEY intermediary_key;
  GenerateKeyPair(*kem, intermediary_public_key, intermediary_key);
  absl::StatusOr<std::string> intermediary_cwt_bytes = OkpCwt{
      .public_key = OkpKey{
          .algorithm = crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm,
          .curve = crypto_internal::kX25519,
          .x = intermediary_public_key,
      }}.Encode();
  ASSERT_OK(intermediary_cwt_bytes);
  absl::StatusOr<EncryptMessageResult> encrypt_result = encryptor.Encrypt(
      message, *intermediary_cwt_bytes, message_associated_data);
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
          kem, kdf, aead, *symmetric_key, recipient_cwt->public_key->x,
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

  // The symmetric key should be an encoded CoseKey.
  absl::StatusOr<SymmetricKey> decoded_symmetric_key =
      SymmetricKey::Decode(*symmetric_key);
  ASSERT_OK(decoded_symmetric_key);
  EXPECT_EQ(decoded_symmetric_key->algorithm,
            crypto_internal::kAeadAes128GcmSivFixedNonce);
  EXPECT_NE(decoded_symmetric_key->k, "");
}

TEST(CryptoTest, EncryptWithInvalidPublicKeyFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "plaintext associated data";

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, "invalid public key", associated_data);
  ASSERT_THAT(encrypt_result, IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, EncryptWithInvalidCwtFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "associated data";

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, std::string(50, 'x'), associated_data);
  EXPECT_THAT(encrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, EncryptWithMissingCwtPublicKeyFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "associated data";

  absl::StatusOr<std::string> cwt_bytes = OkpCwt().Encode();
  ASSERT_OK(cwt_bytes);

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, *cwt_bytes, associated_data);
  EXPECT_THAT(encrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, EncryptWithInvalidCwtAlgorithmFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "associated data";

  absl::StatusOr<std::string> public_key =
      MessageDecryptor().GetPublicKey([](absl::string_view) { return ""; });
  ASSERT_OK(public_key);
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(*public_key);
  ASSERT_OK(cwt);
  cwt->public_key->algorithm = crypto_internal::kAeadAes128GcmSivFixedNonce;
  absl::StatusOr<std::string> cwt_bytes = cwt->Encode();
  ASSERT_OK(cwt_bytes);

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, *cwt_bytes, associated_data);
  EXPECT_THAT(encrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, EncryptWithInvalidCwtCurveFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "associated data";

  absl::StatusOr<std::string> public_key =
      MessageDecryptor().GetPublicKey([](absl::string_view) { return ""; });
  ASSERT_OK(public_key);
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(*public_key);
  ASSERT_OK(cwt);
  cwt->public_key->curve = 0;  // 0 is a reserved value.
  absl::StatusOr<std::string> cwt_bytes = cwt->Encode();
  ASSERT_OK(cwt_bytes);

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, *cwt_bytes, associated_data);
  EXPECT_THAT(encrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, EncryptWithInvalidCwtPublicKeyFails) {
  std::string message = "some plaintext message";
  std::string associated_data = "associated data";

  OkpCwt cwt{
      .public_key =
          OkpKey{
              .algorithm = crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm,
              .curve = crypto_internal::kX25519,
              .x = "invalid public key",
          },
  };
  absl::StatusOr<std::string> cwt_bytes = cwt.Encode();
  ASSERT_OK(cwt_bytes);

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, *cwt_bytes, associated_data);
  EXPECT_THAT(encrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

TEST(CryptoTest, EncryptTwiceWithSameKeyUsesDifferentSymmetricKey) {
  std::string message = "some plaintext message";
  std::string message_associated_data = "plaintext associated data";

  MessageEncryptor encryptor;

  // Encrypt the symmetric key with the public key of an intermediary.
  const EVP_HPKE_KEM* kem = EVP_hpke_x25519_hkdf_sha256();
  const EVP_HPKE_KDF* kdf = EVP_hpke_hkdf_sha256();
  const EVP_HPKE_AEAD* aead = EVP_hpke_aes_128_gcm();
  std::string intermediary_public_key;
  bssl::ScopedEVP_HPKE_KEY intermediary_key;
  GenerateKeyPair(*kem, intermediary_public_key, intermediary_key);
  absl::StatusOr<std::string> intermediary_cwt_bytes = OkpCwt{
      .public_key = OkpKey{
          .algorithm = crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm,
          .curve = crypto_internal::kX25519,
          .x = intermediary_public_key,
      }}.Encode();
  ASSERT_OK(intermediary_cwt_bytes);
  absl::StatusOr<EncryptMessageResult> encrypt_result_1 = encryptor.Encrypt(
      message, *intermediary_cwt_bytes, message_associated_data);
  ASSERT_OK(encrypt_result_1);

  // Encrypt the same plaintext a second time. This should produce a different
  // ciphertext as a new symmetric key will be generated.
  absl::StatusOr<EncryptMessageResult> encrypt_result_2 = encryptor.Encrypt(
      message, *intermediary_cwt_bytes, message_associated_data);
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

  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; });
  ASSERT_OK(recipient_public_key);

  // Encrypt the symmetric key with the public key of an intermediary.
  const EVP_HPKE_KEM* kem = EVP_hpke_x25519_hkdf_sha256();
  std::string intermediary_public_key;
  bssl::ScopedEVP_HPKE_KEY intermediary_key;
  GenerateKeyPair(*kem, intermediary_public_key, intermediary_key);
  absl::StatusOr<std::string> intermediary_cwt_bytes = OkpCwt{
      .public_key = OkpKey{
          .algorithm = crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm,
          .curve = crypto_internal::kX25519,
          .x = intermediary_public_key,
      }}.Encode();
  ASSERT_OK(intermediary_cwt_bytes);
  absl::StatusOr<EncryptMessageResult> encrypt_result = encryptor.Encrypt(
      message, *intermediary_cwt_bytes, message_associated_data);
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

  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; });
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

  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; });
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

  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; });
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

  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; });
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

  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; });
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

  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; });
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

TEST(CryptoTest, DecryptWithInvalidAlgorithmFails) {
  std::string message = "some plaintext message";
  std::string message_associated_data = "plaintext associated data";

  MessageEncryptor encryptor;
  MessageDecryptor decryptor;

  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; });
  ASSERT_OK(recipient_public_key);
  absl::StatusOr<OkpCwt> recipient_cwt = OkpCwt::Decode(*recipient_public_key);
  ASSERT_OK(recipient_cwt);
  ASSERT_TRUE(recipient_cwt->public_key.has_value());

  // Encrypt the symmetric key with the public key of an intermediary.
  const EVP_HPKE_KEM* kem = EVP_hpke_x25519_hkdf_sha256();
  const EVP_HPKE_KDF* kdf = EVP_hpke_hkdf_sha256();
  const EVP_HPKE_AEAD* aead = EVP_hpke_aes_128_gcm();
  std::string intermediary_public_key;
  bssl::ScopedEVP_HPKE_KEY intermediary_key;
  GenerateKeyPair(*kem, intermediary_public_key, intermediary_key);
  absl::StatusOr<std::string> intermediary_cwt_bytes = OkpCwt{
      .public_key = OkpKey{
          .algorithm = crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm,
          .curve = crypto_internal::kX25519,
          .x = intermediary_public_key,
      }}.Encode();
  ASSERT_OK(intermediary_cwt_bytes);
  absl::StatusOr<EncryptMessageResult> encrypt_result = encryptor.Encrypt(
      message, *intermediary_cwt_bytes, message_associated_data);
  ASSERT_OK(encrypt_result);

  // Have the intermediary rewrap the symmetric key with the public key of the
  // final recipient -- after changing the key's algorithm.
  absl::StatusOr<std::string> symmetric_key =
      crypto_internal::UnwrapSymmetricKey(
          intermediary_key.get(), kdf, aead,
          encrypt_result->encrypted_symmetric_key, encrypt_result->encapped_key,
          message_associated_data);
  ASSERT_OK(symmetric_key);

  absl::StatusOr<SymmetricKey> decoded_symmetric_key =
      SymmetricKey::Decode(*symmetric_key);
  ASSERT_OK(decoded_symmetric_key);
  decoded_symmetric_key->algorithm =
      crypto_internal::kHpkeBaseX25519Sha256Aes128Gcm;
  symmetric_key = decoded_symmetric_key->Encode();
  ASSERT_OK(symmetric_key);

  std::string symmetric_key_associated_data =
      "rewrap symmetric key associated data";
  absl::StatusOr<crypto_internal::WrapSymmetricKeyResult>
      rewrapped_symmetric_key_result = crypto_internal::WrapSymmetricKey(
          kem, kdf, aead, *symmetric_key, recipient_cwt->public_key->x,
          symmetric_key_associated_data);
  ASSERT_OK(rewrapped_symmetric_key_result);

  // Decryption should fail because the algorithm is unsupported.
  absl::StatusOr<std::string> decrypt_result =
      decryptor.Decrypt(encrypt_result->ciphertext, message_associated_data,
                        rewrapped_symmetric_key_result->encrypted_symmetric_key,
                        symmetric_key_associated_data,
                        rewrapped_symmetric_key_result->encapped_key);
  EXPECT_THAT(decrypt_result, fcp::IsCode(INVALID_ARGUMENT));
}

}  // namespace
}  // namespace confidential_compute
}  // namespace fcp
