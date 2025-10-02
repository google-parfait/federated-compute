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

#include "fcp/confidentialcompute/payload_transparency/payload_transparency.h"

#include <string>
#include <tuple>
#include <utility>

#include "google/protobuf/timestamp.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "fcp/base/digest.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/access_policy_endorsement_options.pb.h"
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"
#include "fcp/testing/testing.h"

namespace fcp::confidential_compute::payload_transparency {
namespace {

using ::fcp::confidentialcompute::AccessPolicyEndorsementOptions;
using ::fcp::confidentialcompute::Key;
using ::fcp::confidentialcompute::SignedPayload;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

// Returns a new EcdsaP256R1Signer and its corresponding verifying Key.
std::tuple<EcdsaP256R1Signer, Key> CreateSignerAndVerifyingKey() {
  auto signer = EcdsaP256R1Signer::Create();
  Key verifying_key;
  verifying_key.set_algorithm(Key::ECDSA_P256);
  verifying_key.set_purpose(Key::VERIFY);
  verifying_key.set_key_material(signer.GetPublicKey());
  verifying_key.set_key_id(
      ComputeSHA256(verifying_key.key_material()).substr(0, 4));
  return {std::move(signer), std::move(verifying_key)};
}

// Adds a signature to a SignedPayload message.
SignedPayload::Signature& AddSignature(
    SignedPayload& signed_payload,
    const SignedPayload::Signature::Headers& headers,
    const EcdsaP256R1Signer& signer) {
  auto* signature = signed_payload.add_signatures();
  headers.SerializeToString(signature->mutable_headers());

  std::string to_sign;
  GetSignedPayloadSigStructureEmitter(signature->headers(),
                                      signed_payload.payload())(
      [&to_sign](absl::string_view part) { absl::StrAppend(&to_sign, part); });
  signature->set_raw_signature(signer.Sign(to_sign));
  return *signature;
}

// Adds a Rekor log entry to a SignedPayload message.
SignedPayload::Signature& AddRekorLogEntry(
    SignedPayload& signed_payload,
    const SignedPayload::Signature::Headers& headers,
    const EcdsaP256R1Signer& signer, const EcdsaP256R1Signer& rekor_signer,
    const Key& rekor_verifying_key) {
  auto* signature = signed_payload.add_signatures();
  headers.SerializeToString(signature->mutable_headers());

  std::string to_sign;
  GetSignedPayloadSigStructureEmitter(signature->headers(),
                                      signed_payload.payload())(
      [&to_sign](absl::string_view part) { absl::StrAppend(&to_sign, part); });
  absl::StatusOr<std::string> asn1_signature =
      ConvertP1363SignatureToAsn1(signer.Sign(to_sign));
  CHECK_OK(asn1_signature);

  auto* rekor = signature->mutable_log_entry()->mutable_rekor();
  rekor->set_body(absl::Substitute(
      R"({"apiVersion":"0.0.1","kind":"hashedrekord","spec":{)"
      R"("data":{"hash":{"algorithm":"sha256","value":"$0"}},)"
      R"("signature":{"content":"$1","publicKey":{"content":"unused"}}}})",
      absl::BytesToHexString(ComputeSHA256(to_sign)),
      absl::Base64Escape(*asn1_signature)));
  // Create an inclusion proof for a single-element tree, which is simpler to
  // construct.
  rekor->set_log_index(0);
  rekor->set_tree_size(1);
  rekor->set_checkpoint_origin("origin");
  std::string root_hash =
      ComputeSHA256(absl::StrCat(absl::string_view("\0", 1), rekor->body()));
  rekor->set_checkpoint_signature(rekor_signer.Sign(
      absl::Substitute("$0\n$1\n$2\n", rekor->checkpoint_origin(),
                       rekor->tree_size(), absl::Base64Escape(root_hash))));
  rekor->set_checkpoint_signature_key_id(rekor_verifying_key.key_id());
  return *signature;
}

TEST(VerifySignedPayloadTest, NoSignatures) {
  SignedPayload signed_payload;
  signed_payload.set_payload("payload");

  absl::StatusOr<VerifySignedPayloadResult> result =
      VerifySignedPayload(signed_payload, {}, {}, absl::Now());
  EXPECT_THAT(result, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("signature verification failed: []"));
}

TEST(VerifySignedPayloadTest, SingleSignature) {
  auto [signer, verifying_key] = CreateSignerAndVerifyingKey();

  SignedPayload signed_payload;
  signed_payload.set_payload("payload");
  SignedPayload::Signature::Headers headers;
  headers.set_algorithm(verifying_key.algorithm());
  AddSignature(signed_payload, headers, signer)
      .set_verifying_key_id(verifying_key.key_id());

  absl::StatusOr<VerifySignedPayloadResult> result =
      VerifySignedPayload(signed_payload, {&verifying_key}, {}, absl::Now());
  ASSERT_OK(result);
  EXPECT_THAT(result->headers, ElementsAre(EqualsProto(headers)));
}

TEST(VerifySignedPayloadTest, MultipleSignatures) {
  SignedPayload signed_payload;
  signed_payload.set_payload("payload");

  auto [signer1, verifying_key1] = CreateSignerAndVerifyingKey();
  SignedPayload::Signature::Headers headers1;
  headers1.set_algorithm(verifying_key1.algorithm());
  headers1.add_claims("key1");
  AddSignature(signed_payload, headers1, signer1)
      .set_verifying_key_id(verifying_key1.key_id());

  auto [signer2, verifying_key2] = CreateSignerAndVerifyingKey();
  SignedPayload::Signature::Headers headers2;
  headers2.set_algorithm(verifying_key2.algorithm());
  headers2.add_claims("key2");
  AddSignature(signed_payload, headers2, signer2)
      .set_verifying_key_id(verifying_key2.key_id());

  // The first signature should fail to verify since verifying_key1 is not in
  // the list of specified valid verifying keys. The second signature should
  // succeed, hence verification should succeed overall.
  absl::StatusOr<VerifySignedPayloadResult> result =
      VerifySignedPayload(signed_payload, {&verifying_key2}, {}, absl::Now());
  ASSERT_OK(result);
  EXPECT_THAT(result->headers, ElementsAre(EqualsProto(headers2)));
}

TEST(VerifySignedPayloadTest, SignatureChain) {
  SignedPayload signed_payload;
  signed_payload.set_payload("payload");

  auto [signer1, verifying_key1] = CreateSignerAndVerifyingKey();
  SignedPayload::Signature::Headers headers1;
  headers1.set_algorithm(verifying_key1.algorithm());
  headers1.add_claims("key1");
  SignedPayload* signed_verifying_key =
      AddSignature(signed_payload, headers1, signer1).mutable_verifying_key();

  auto [signer2, verifying_key2] = CreateSignerAndVerifyingKey();
  verifying_key1.SerializeToString(signed_verifying_key->mutable_payload());
  SignedPayload::Signature::Headers headers2;
  headers2.set_algorithm(verifying_key2.algorithm());
  headers2.add_claims("key2");
  AddSignature(*signed_verifying_key, headers2, signer2)
      .set_verifying_key_id(verifying_key2.key_id());

  absl::StatusOr<VerifySignedPayloadResult> result =
      VerifySignedPayload(signed_payload, {&verifying_key2}, {}, absl::Now());
  ASSERT_OK(result);
  EXPECT_THAT(result->headers,
              ElementsAre(EqualsProto(headers2), EqualsProto(headers1)));
}

TEST(VerifySignedPayloadTest, NoMatchingVerifyingKeys) {
  auto [signer1, verifying_key1] = CreateSignerAndVerifyingKey();

  SignedPayload signed_payload;
  signed_payload.set_payload("payload");
  SignedPayload::Signature::Headers headers;
  headers.set_algorithm(verifying_key1.algorithm());
  AddSignature(signed_payload, headers, signer1)
      .set_verifying_key_id(verifying_key1.key_id());

  // Use a different verifying key, which should fail.
  auto [signer2, verifying_key2] = CreateSignerAndVerifyingKey();
  absl::StatusOr<VerifySignedPayloadResult> result =
      VerifySignedPayload(signed_payload, {&verifying_key2}, {}, absl::Now());
  EXPECT_THAT(result, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("signature verification failed"));
}

TEST(VerifySignedPayloadTest, InvalidHeaders) {
  auto [signer, verifying_key] = CreateSignerAndVerifyingKey();

  SignedPayload signed_payload;
  signed_payload.set_payload("payload");
  SignedPayload::Signature::Headers headers;
  headers.set_algorithm(verifying_key.algorithm());
  AddSignature(signed_payload, headers, signer)
      .set_verifying_key_id(verifying_key.key_id());

  // Corrupt the headers.
  signed_payload.mutable_signatures(0)->set_headers("invalid");

  absl::StatusOr<VerifySignedPayloadResult> result =
      VerifySignedPayload(signed_payload, {&verifying_key}, {}, absl::Now());
  EXPECT_THAT(result, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("failed to parse signature headers"));
}

TEST(VerifySignedPayloadTest, InvalidVerifyingKey) {
  SignedPayload signed_payload;
  signed_payload.set_payload("payload");

  auto [signer1, verifying_key1] = CreateSignerAndVerifyingKey();
  SignedPayload::Signature::Headers headers1;
  headers1.set_algorithm(verifying_key1.algorithm());
  headers1.add_claims("key1");
  SignedPayload* signed_verifying_key =
      AddSignature(signed_payload, headers1, signer1).mutable_verifying_key();

  auto [signer2, verifying_key2] = CreateSignerAndVerifyingKey();
  signed_verifying_key->set_payload("invalid");
  SignedPayload::Signature::Headers headers2;
  headers2.set_algorithm(verifying_key2.algorithm());
  headers2.add_claims("key2");
  AddSignature(*signed_verifying_key, headers2, signer2)
      .set_verifying_key_id(verifying_key2.key_id());

  absl::StatusOr<VerifySignedPayloadResult> result =
      VerifySignedPayload(signed_payload, {&verifying_key2}, {}, absl::Now());
  EXPECT_THAT(result, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("failed to parse verifying key"));
}

TEST(VerifySignedPayloadTest, UnsupportedSignatureOneof) {
  auto [signer, verifying_key] = CreateSignerAndVerifyingKey();

  SignedPayload signed_payload;
  signed_payload.set_payload("payload");
  SignedPayload::Signature::Headers headers;
  headers.set_algorithm(verifying_key.algorithm());
  AddSignature(signed_payload, headers, signer)
      .set_verifying_key_id(verifying_key.key_id());

  // raw_signature or log_entry is required.
  signed_payload.mutable_signatures(0)->clear_raw_signature();

  absl::StatusOr<VerifySignedPayloadResult> result =
      VerifySignedPayload(signed_payload, {&verifying_key}, {}, absl::Now());
  EXPECT_THAT(result, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("unsupported Signature.signature"));
}

TEST(VerifySignedPayloadTest, UnsupportedSignatureVerifier) {
  auto [signer, verifying_key] = CreateSignerAndVerifyingKey();

  SignedPayload signed_payload;
  signed_payload.set_payload("payload");
  SignedPayload::Signature::Headers headers;
  headers.set_algorithm(verifying_key.algorithm());
  // Don't set verifying_key_id or verifying_key.
  AddSignature(signed_payload, headers, signer);

  absl::StatusOr<VerifySignedPayloadResult> result =
      VerifySignedPayload(signed_payload, {&verifying_key}, {}, absl::Now());
  EXPECT_THAT(result, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("unsupported Signature.verifier"));
}

TEST(VerifySignedPayloadTest, SignatureWithTimeBounds) {
  auto [signer, verifying_key] = CreateSignerAndVerifyingKey();

  const absl::Time now = absl::Now();

  SignedPayload signed_payload;
  signed_payload.set_payload("payload");
  SignedPayload::Signature::Headers headers;
  headers.set_algorithm(verifying_key.algorithm());
  // So long as not_before is set, it's okay for not_before to be in the future.
  headers.mutable_issued_at()->set_seconds(
      absl::ToUnixSeconds(now + absl::Seconds(1)));
  headers.mutable_not_before()->set_seconds(absl::ToUnixSeconds(now));
  headers.mutable_not_after()->set_seconds(
      absl::ToUnixSeconds(now + absl::Seconds(1)));
  AddSignature(signed_payload, headers, signer)
      .set_verifying_key_id(verifying_key.key_id());

  ASSERT_OK(VerifySignedPayload(signed_payload, {&verifying_key}, {}, now));
}

TEST(VerifySignedPayloadTest, SignatureNotYetValid) {
  auto [signer, verifying_key] = CreateSignerAndVerifyingKey();

  const absl::Time now = absl::Now();

  SignedPayload signed_payload;
  signed_payload.set_payload("payload");
  SignedPayload::Signature::Headers headers;
  headers.set_algorithm(verifying_key.algorithm());
  headers.mutable_not_before()->set_seconds(
      absl::ToUnixSeconds(now + absl::Seconds(1)));
  AddSignature(signed_payload, headers, signer)
      .set_verifying_key_id(verifying_key.key_id());

  absl::StatusOr<VerifySignedPayloadResult> result =
      VerifySignedPayload(signed_payload, {&verifying_key}, {}, now);
  EXPECT_THAT(result, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("not_before is in the future"));

  // If not_before isn't set, issued_at is used as the start time.
  *headers.mutable_issued_at() = headers.not_before();
  headers.clear_not_before();
  signed_payload.clear_signatures();
  AddSignature(signed_payload, headers, signer)
      .set_verifying_key_id(verifying_key.key_id());
  result = VerifySignedPayload(signed_payload, {&verifying_key}, {}, now);
  EXPECT_THAT(result, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("issued_at is in the future"));
}

TEST(VerifySignedPayloadTest, SignatureExpired) {
  auto [signer, verifying_key] = CreateSignerAndVerifyingKey();

  const absl::Time now = absl::Now();

  SignedPayload signed_payload;
  signed_payload.set_payload("payload");
  SignedPayload::Signature::Headers headers;
  headers.set_algorithm(verifying_key.algorithm());
  headers.mutable_not_after()->set_seconds(
      absl::ToUnixSeconds(now - absl::Seconds(1)));
  AddSignature(signed_payload, headers, signer)
      .set_verifying_key_id(verifying_key.key_id());

  absl::StatusOr<VerifySignedPayloadResult> result =
      VerifySignedPayload(signed_payload, {&verifying_key}, {}, now);
  EXPECT_THAT(result, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(), HasSubstr("not_after is in the past"));
}

TEST(VerifySignedPayloadTest, MissingTransparencyLogEntry) {
  auto [signer, verifying_key] = CreateSignerAndVerifyingKey();
  auto [rekor_signer, rekor_verifying_key] = CreateSignerAndVerifyingKey();

  SignedPayload signed_payload;
  signed_payload.set_payload("payload");
  SignedPayload::Signature::Headers headers;
  headers.set_algorithm(verifying_key.algorithm());
  AddSignature(signed_payload, headers, signer)
      .set_verifying_key_id(verifying_key.key_id());

  AccessPolicyEndorsementOptions::TransparencyLogOptions
      transparency_log_options;
  transparency_log_options.set_require_transparency_log_entry(true);
  *transparency_log_options.add_rekor_verifying_keys() = rekor_verifying_key;

  absl::StatusOr<VerifySignedPayloadResult> result = VerifySignedPayload(
      signed_payload, {&verifying_key}, transparency_log_options, absl::Now());
  EXPECT_THAT(result, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Signature does not have a transparency log entry"));
}

TEST(VerifySignedPayloadTest, ValidRekorLogEntry) {
  auto [signer, verifying_key] = CreateSignerAndVerifyingKey();
  auto [rekor_signer, rekor_verifying_key] = CreateSignerAndVerifyingKey();

  SignedPayload signed_payload;
  signed_payload.set_payload("payload");
  SignedPayload::Signature::Headers headers;
  headers.set_algorithm(verifying_key.algorithm());
  AddRekorLogEntry(signed_payload, headers, signer, rekor_signer,
                   rekor_verifying_key)
      .set_verifying_key_id(verifying_key.key_id());

  AccessPolicyEndorsementOptions::TransparencyLogOptions
      transparency_log_options;
  *transparency_log_options.add_rekor_verifying_keys() = rekor_verifying_key;
  // The Rekor log entry should be accepted regardless of whether a transparency
  // log entry is required.
  for (bool require_transparency_log_entry : {false, true}) {
    SCOPED_TRACE(absl::StrCat("require_transparency_log_entry: ",
                              require_transparency_log_entry));
    transparency_log_options.set_require_transparency_log_entry(
        require_transparency_log_entry);

    absl::StatusOr<VerifySignedPayloadResult> result =
        VerifySignedPayload(signed_payload, {&verifying_key},
                            transparency_log_options, absl::Now());
    ASSERT_OK(result);
    EXPECT_THAT(result->headers, ElementsAre(EqualsProto(headers)));
  }
}

TEST(VerifySignedPayloadTest, NonLeafRekorLogEntry) {
  SignedPayload signed_payload;
  signed_payload.set_payload("payload");

  auto [signer1, verifying_key1] = CreateSignerAndVerifyingKey();
  SignedPayload::Signature::Headers headers1;
  headers1.set_algorithm(verifying_key1.algorithm());
  headers1.add_claims("key1");
  auto [rekor_signer, rekor_verifying_key] = CreateSignerAndVerifyingKey();
  SignedPayload* signed_verifying_key =
      AddRekorLogEntry(signed_payload, headers1, signer1, rekor_signer,
                       rekor_verifying_key)
          .mutable_verifying_key();

  auto [signer2, verifying_key2] = CreateSignerAndVerifyingKey();
  verifying_key1.SerializeToString(signed_verifying_key->mutable_payload());
  SignedPayload::Signature::Headers headers2;
  headers2.set_algorithm(verifying_key2.algorithm());
  headers2.add_claims("key2");
  AddSignature(*signed_verifying_key, headers2, signer2)
      .set_verifying_key_id(verifying_key2.key_id());

  AccessPolicyEndorsementOptions::TransparencyLogOptions
      transparency_log_options;
  *transparency_log_options.add_rekor_verifying_keys() = rekor_verifying_key;
  absl::StatusOr<VerifySignedPayloadResult> result = VerifySignedPayload(
      signed_payload, {&verifying_key2}, transparency_log_options, absl::Now());
  ASSERT_OK(result);
  EXPECT_THAT(result->headers,
              ElementsAre(EqualsProto(headers2), EqualsProto(headers1)));

  // While a Rekor log entry part way through the signature chain is supported,
  // it isn't sufficient to satisfy require_transparency_log_entry, which needs
  // a Rekor log entry at the end of the chain.
  transparency_log_options.set_require_transparency_log_entry(true);
  result = VerifySignedPayload(signed_payload, {&verifying_key2},
                               transparency_log_options, absl::Now());
  EXPECT_THAT(result, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Signature does not have a transparency log entry"));
}

TEST(VerifySignedPayloadTest, InvalidRekorLogEntry) {
  auto [signer, verifying_key] = CreateSignerAndVerifyingKey();
  auto [rekor_signer, rekor_verifying_key] = CreateSignerAndVerifyingKey();

  SignedPayload signed_payload;
  signed_payload.set_payload("payload");
  SignedPayload::Signature::Headers headers;
  headers.set_algorithm(verifying_key.algorithm());
  AddRekorLogEntry(signed_payload, headers, signer, rekor_signer,
                   rekor_verifying_key)
      .set_verifying_key_id(verifying_key.key_id());

  // Corrupt the log entry.
  signed_payload.mutable_signatures(0)
      ->mutable_log_entry()
      ->mutable_rekor()
      ->set_log_index(
          signed_payload.signatures(0).log_entry().rekor().log_index() + 1);

  AccessPolicyEndorsementOptions::TransparencyLogOptions
      transparency_log_options;
  *transparency_log_options.add_rekor_verifying_keys() = rekor_verifying_key;
  EXPECT_THAT(VerifySignedPayload(signed_payload, {&verifying_key},
                                  transparency_log_options, absl::Now()),
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(PayloadTransparencyTest, GetSignedPayloadSigStructureEmitter) {
  std::string to_sign;
  GetSignedPayloadSigStructureEmitter("hdrs", "payload")(
      [&to_sign](absl::string_view part) { absl::StrAppend(&to_sign, part); });
  EXPECT_EQ(to_sign, "\15SignedPayload\4hdrs\7payload");
}

}  // namespace
}  // namespace fcp::confidential_compute::payload_transparency
