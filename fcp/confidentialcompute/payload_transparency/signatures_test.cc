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

#include <cstddef>
#include <cstdint>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/fixed_array.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "fcp/base/digest.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "fcp/testing/testing.h"
#include "openssl/bn.h"
#include "openssl/digest.h"
#include "openssl/ecdsa.h"
#include "openssl/mem.h"

namespace fcp::confidential_compute::payload_transparency {
namespace {

using ::fcp::confidentialcompute::Key;
using ::testing::HasSubstr;

TEST(ComputeDigestTest, Emitter) {
  EXPECT_EQ(
      ComputeDigest(EVP_sha256(),
                    [](absl::FunctionRef<void(absl::string_view)> emitter) {
                      emitter("data");
                    }),
      ComputeDigest(EVP_sha256(),
                    [](absl::FunctionRef<void(absl::string_view)> emitter) {
                      emitter("d");
                      emitter("at");
                      emitter("a");
                    }));
}

TEST(ComputeDigestTest, PrecomputedDigest) {
  absl::FixedArray<uint8_t> digest = {0, 1, 2, 3, 4, 5};
  EXPECT_EQ(ComputeDigest(EVP_sha256(), digest), digest);
}

TEST(VerifySignatureTest, P1363Format) {
  auto signer = EcdsaP256R1Signer::Create();
  Key key;
  key.set_algorithm(Key::ECDSA_P256);
  key.set_purpose(Key::VERIFY);
  key.set_key_material(signer.GetPublicKey());

  EXPECT_OK(
      VerifySignature(signer.Sign("data"), SignatureFormat::kP1363, {&key},
                      [](absl::FunctionRef<void(absl::string_view)> emitter) {
                        emitter("data");
                      }));
}

TEST(VerifySignatureTest, Asn1Format) {
  auto signer = EcdsaP256R1Signer::Create();
  Key key;
  key.set_algorithm(Key::ECDSA_P256);
  key.set_purpose(Key::VERIFY);
  key.set_key_material(signer.GetPublicKey());

  std::string signature = signer.Sign("data");
  ASSERT_EQ(signature.size() % 2, 0) << "Signature does not have even length";

  // Convert the P1363 signature to ASN.1.
  bssl::UniquePtr<ECDSA_SIG> sig(ECDSA_SIG_new());
  ASSERT_TRUE(ECDSA_SIG_set0(
      sig.get(),
      BN_bin2bn(reinterpret_cast<const uint8_t*>(signature.data()),
                signature.size() / 2, nullptr),
      BN_bin2bn(reinterpret_cast<const uint8_t*>(signature.data() +
                                                 signature.size() / 2),
                signature.size() / 2, nullptr)));
  uint8_t* sig_bytes;
  size_t sig_bytes_len;
  ASSERT_TRUE(ECDSA_SIG_to_bytes(&sig_bytes, &sig_bytes_len, sig.get()));
  std::string asn1_signature(reinterpret_cast<const char*>(sig_bytes),
                             sig_bytes_len);
  OPENSSL_free(sig_bytes);

  EXPECT_OK(
      VerifySignature(asn1_signature, SignatureFormat::kAsn1, {&key},
                      [](absl::FunctionRef<void(absl::string_view)> emitter) {
                        emitter("data");
                      }));
}

TEST(VerifySignatureTest, NoKeys) {
  auto signer = EcdsaP256R1Signer::Create();
  absl::Status status =
      VerifySignature(signer.Sign("data"), SignatureFormat::kP1363, {},
                      [](absl::FunctionRef<void(absl::string_view)> emitter) {
                        emitter("data");
                      });
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("no matching signing keys"));
}

TEST(VerifySignatureTest, MultipleKeys) {
  auto signer = EcdsaP256R1Signer::Create();
  Key key1;
  key1.set_algorithm(Key::ECDSA_P256);
  key1.set_purpose(Key::VERIFY);
  key1.set_key_material(EcdsaP256R1Signer::Create().GetPublicKey());
  Key key2;
  key2.set_algorithm(Key::ECDSA_P256);
  key2.set_purpose(Key::VERIFY);
  key2.set_key_material(signer.GetPublicKey());
  Key key3;
  key3.set_algorithm(Key::ECDSA_P256);
  key3.set_purpose(Key::VERIFY);
  key3.set_key_material(EcdsaP256R1Signer::Create().GetPublicKey());

  EXPECT_OK(VerifySignature(
      signer.Sign("data"), SignatureFormat::kP1363, {&key1, &key2, &key3},
      [](absl::FunctionRef<void(absl::string_view)> emitter) {
        emitter("data");
      }));
}

TEST(VerifySignatureTest, MultipleEmitterCalls) {
  auto signer = EcdsaP256R1Signer::Create();
  Key key;
  key.set_algorithm(Key::ECDSA_P256);
  key.set_purpose(Key::VERIFY);
  key.set_key_material(signer.GetPublicKey());

  EXPECT_OK(
      VerifySignature(signer.Sign("data"), SignatureFormat::kP1363, {&key},
                      [](absl::FunctionRef<void(absl::string_view)> emitter) {
                        emitter("d");
                        emitter("at");
                        emitter("a");
                      }));
}

TEST(VerifySignatureTest, PrecomputedDigest) {
  auto signer = EcdsaP256R1Signer::Create();
  Key key;
  key.set_algorithm(Key::ECDSA_P256);
  key.set_purpose(Key::VERIFY);
  key.set_key_material(signer.GetPublicKey());

  std::string digest = ComputeSHA256("data");

  EXPECT_OK(VerifySignature(
      signer.Sign("data"), SignatureFormat::kP1363, {&key},
      absl::MakeSpan(reinterpret_cast<const uint8_t*>(digest.data()),
                     digest.size())));
}

TEST(VerifySignatureTest, InvalidSignature) {
  Key key;
  key.set_algorithm(Key::ECDSA_P256);
  key.set_purpose(Key::VERIFY);
  key.set_key_material(EcdsaP256R1Signer::Create().GetPublicKey());

  absl::Status status =
      VerifySignature("invalid", SignatureFormat::kP1363, {&key},
                      [](absl::FunctionRef<void(absl::string_view)> emitter) {
                        emitter("data");
                      });
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("invalid signature"));

  status =
      VerifySignature("invalid", SignatureFormat::kAsn1, {&key},
                      [](absl::FunctionRef<void(absl::string_view)> emitter) {
                        emitter("data");
                      });
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("invalid signature"));
}

TEST(VerifySignatureTest, InvalidAlgorithm) {
  auto signer = EcdsaP256R1Signer::Create();
  Key key;
  key.set_algorithm(Key::HPKE_X25519_SHA256_AES128_GCM);
  key.set_purpose(Key::VERIFY);
  key.set_key_material(signer.GetPublicKey());

  absl::Status status =
      VerifySignature(signer.Sign("data"), SignatureFormat::kP1363, {&key},
                      [](absl::FunctionRef<void(absl::string_view)> emitter) {
                        emitter("data");
                      });
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("unsupported key algorithm"));
}

TEST(VerifySignatureTest, InvalidPurpose) {
  auto signer = EcdsaP256R1Signer::Create();
  Key key;
  key.set_algorithm(Key::ECDSA_P256);
  key.set_purpose(Key::SIGN);
  key.set_key_material(signer.GetPublicKey());

  absl::Status status =
      VerifySignature(signer.Sign("data"), SignatureFormat::kP1363, {&key},
                      [](absl::FunctionRef<void(absl::string_view)> emitter) {
                        emitter("data");
                      });
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("key is not a verifying key"));
}

}  // namespace
}  // namespace fcp::confidential_compute::payload_transparency
