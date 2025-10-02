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

#include "fcp/client/attestation/attestation_transparency_verifier.h"

#include <string>
#include <tuple>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "fcp/client/attestation/log_attestation_records.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/confidentialcompute/payload_transparency/payload_transparency.h"
#include "fcp/protos/confidentialcompute/access_policy_endorsement_options.pb.h"
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"
#include "fcp/protos/confidentialcompute/signed_endorsements.pb.h"
#include "fcp/protos/federatedcompute/confidential_encryption_config.pb.h"
#include "fcp/testing/testing.h"

namespace fcp::client::attestation {
namespace {

using ::fcp::confidential_compute::EcdsaP256R1Signer;
using ::fcp::confidential_compute::payload_transparency::
    GetSignedPayloadSigStructureEmitter;
using ::fcp::confidentialcompute::AccessPolicyEndorsementOptions;
using ::fcp::confidentialcompute::Key;
using ::fcp::confidentialcompute::SignedEndorsements;
using ::fcp::confidentialcompute::SignedPayload;
using ::google::internal::federatedcompute::v1::ConfidentialEncryptionConfig;
using ::testing::HasSubstr;
using ::testing::VariantWith;

// Creates a SignedPayload message with a single signature, returning both the
// SignedPayload and the signature's verifying key.
std::tuple<SignedPayload, Key> CreateSignedPayload(
    std::string payload, SignedPayload::Signature::Headers headers = {}) {
  auto signer = EcdsaP256R1Signer::Create();
  Key verifying_key;
  verifying_key.set_algorithm(Key::ECDSA_P256);
  verifying_key.set_purpose(Key::VERIFY);
  verifying_key.set_key_material(signer.GetPublicKey());
  verifying_key.set_key_id(verifying_key.key_material().substr(0, 6));

  // Make sure the headers have the correct algorithm.
  headers.set_algorithm(verifying_key.algorithm());

  SignedPayload signed_payload;
  signed_payload.set_payload(std::move(payload));
  auto* signature = signed_payload.add_signatures();
  signature->set_headers(headers.SerializeAsString());
  signature->set_verifying_key_id(verifying_key.key_id());

  std::string to_sign;
  GetSignedPayloadSigStructureEmitter(signature->headers(),
                                      signed_payload.payload())(
      [&to_sign](absl::string_view part) { absl::StrAppend(&to_sign, part); });
  signature->set_raw_signature(signer.Sign(to_sign));

  return {std::move(signed_payload), std::move(verifying_key)};
}

// Returns valid headers for the public key signature.
SignedPayload::Signature::Headers GetValidPublicKeyHeaders(
    std::string access_policy_sha256) {
  SignedPayload::Signature::Headers headers;
  headers.set_access_policy_sha256(std::move(access_policy_sha256));
  headers.add_claims(
      "https://github.com/project-oak/oak/blob/main/docs/tr/claim/92939.md");
  // The oak_application_signature field is required and must have several
  // headers set, but the library doesn't actually verify the signature since
  // it'd require downloading and extracting the key from the Oak
  // EndorsedEvidence, which can't be done reliably on client devices due to
  // the evidence format not being stable.
  SignedPayload::Signature::Headers oak_headers;
  oak_headers.set_endorsed_evidence_sha256("endorsed evidence hash");
  headers.mutable_oak_application_signature()->set_headers(
      oak_headers.SerializeAsString());
  return headers;
}

TEST(AttestationTransparencyVerifierTest,
     DefaultValuesDoNotVerifySuccessfully) {
  AttestationTransparencyVerifier verifier(AccessPolicyEndorsementOptions(),
                                           LogPrettyPrintedVerificationRecord);
  EXPECT_THAT(verifier.Verify({}, {}, {}),
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(AttestationTransparencyVerifierTest, Success) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(public_key.SerializeAsString(),
                          GetValidPublicKeyHeaders("access policy hash"));

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  ASSERT_OK(result);
  EXPECT_THAT(result->public_key, VariantWith<Key>(EqualsProto(public_key)));
  EXPECT_EQ(result->key_id, public_key.key_id());
  EXPECT_EQ(result->access_policy_sha256, "access policy hash");
}

TEST(AttestationTransparencyVerifierTest, SuccessWithSignatureChain) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  // Add a chain of signatures. The access policy hash should be in the first
  // signature's headers, and all other fields should be in the second.
  SignedPayload::Signature::Headers headers1;
  headers1.set_access_policy_sha256("access policy hash");
  auto [signed_payload, verifying_key] =
      CreateSignedPayload(public_key.SerializeAsString(), headers1);
  SignedPayload::Signature::Headers headers2 = GetValidPublicKeyHeaders("");
  headers2.clear_access_policy_sha256();
  std::tie(*signed_payload.mutable_signatures(0)->mutable_verifying_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(verifying_key.SerializeAsString(), headers2);
  *encryption_config.mutable_encryption_key() = std::move(signed_payload);

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  ASSERT_OK(result);
  EXPECT_THAT(result->public_key, VariantWith<Key>(EqualsProto(public_key)));
  EXPECT_EQ(result->key_id, public_key.key_id());
  EXPECT_EQ(result->access_policy_sha256, "access policy hash");
}

TEST(AttestationTransparencyVerifierTest, RequireTransparencyLogEntry) {
  AccessPolicyEndorsementOptions options;
  options.mutable_transparency_log_options()
      ->set_require_transparency_log_entry(true);

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(public_key.SerializeAsString(),
                          GetValidPublicKeyHeaders("access policy hash"));

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(), HasSubstr("transparency log entry"));
}

TEST(AttestationTransparencyVerifierTest, InvalidPipelineConfigSignature) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());
  // Corrupt the signature.
  signed_endorsements.mutable_pipeline_configuration()
      ->mutable_signatures(0)
      ->set_raw_signature("invalid");

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(public_key.SerializeAsString(),
                          GetValidPublicKeyHeaders("access policy hash"));

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(AttestationTransparencyVerifierTest, InvalidPipelineConfig) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload("invalid");

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(public_key.SerializeAsString(),
                          GetValidPublicKeyHeaders("access policy hash"));

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("failed to parse pipeline configuration"));
}

TEST(AttestationTransparencyVerifierTest, MismatchedAccessPolicySha256) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(public_key.SerializeAsString(),
                          GetValidPublicKeyHeaders("other policy hash"));

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("access policy SHA-256 does not match"));
}

TEST(AttestationTransparencyVerifierTest, InvalidEncryptionKeySignature) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(public_key.SerializeAsString(),
                          GetValidPublicKeyHeaders("access policy hash"));
  // Corrupt the signature.
  encryption_config.mutable_encryption_key()
      ->mutable_signatures(0)
      ->set_raw_signature("invalid");

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(AttestationTransparencyVerifierTest, InvalidEncryptionKey) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());

  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload("invalid",
                          GetValidPublicKeyHeaders("access policy hash"));

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("failed to parse encryption key"));
}

TEST(AttestationTransparencyVerifierTest,
     InvalidOakApplicationSignatureHeaders) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());

  auto headers = GetValidPublicKeyHeaders("access policy hash");
  headers.mutable_oak_application_signature()->set_headers("invalid");

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(public_key.SerializeAsString(), headers);

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("failed to parse oak application signature headers"));
}

TEST(AttestationTransparencyVerifierTest, MissingAccessPolicySha256) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());

  auto headers = GetValidPublicKeyHeaders("access policy hash");
  headers.clear_access_policy_sha256();

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(public_key.SerializeAsString(), headers);

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      result.status().message(),
      HasSubstr("encryption key headers missing access policy SHA-256"));
}

TEST(AttestationTransparencyVerifierTest,
     MissingEndorsedEvidenceSha256InEncryptionKeyHeaders) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());

  auto headers = GetValidPublicKeyHeaders("access policy hash");
  SignedPayload::Signature::Headers oak_headers;
  ASSERT_TRUE(oak_headers.ParseFromString(
      headers.oak_application_signature().headers()));
  oak_headers.clear_endorsed_evidence_sha256();
  oak_headers.SerializeToString(
      headers.mutable_oak_application_signature()->mutable_headers());

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(public_key.SerializeAsString(), headers);

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(),
              HasSubstr("oak application signature headers missing endorsed "
                        "evidence SHA-256"));
}

TEST(AttestationTransparencyVerifierTest, MissingClaimsInEncryptionKeyHeaders) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());

  auto headers = GetValidPublicKeyHeaders("access policy hash");
  headers.clear_claims();

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(public_key.SerializeAsString(), headers);

  AttestationTransparencyVerifier verifier(std::move(options),
                                           LogPrettyPrintedVerificationRecord);
  auto result = verifier.Verify({}, signed_endorsements, encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result.status().message(), HasSubstr("missing required claims"));
}

// Tests whether the AttestationVerificationRecord emitted by the
// AttestationTransparencyVerifier contains sufficient information to allow
// someone to re-do the verification (e.g. on their computer, by calling into
// the Oak Attestation Verification library themselves).
TEST(AttestationTransparencyVerifierTest,
     AttestationVerificationRecordContainsEnoughInfoToReplayVerification) {
  AccessPolicyEndorsementOptions options;

  SignedEndorsements::PipelineConfiguration pipeline_configuration;
  pipeline_configuration.set_access_policy_sha256("access policy hash");
  SignedEndorsements signed_endorsements;
  std::tie(*signed_endorsements.mutable_pipeline_configuration(),
           *options.add_access_policy_verifying_keys()) =
      CreateSignedPayload(pipeline_configuration.SerializeAsString());

  Key public_key;
  public_key.set_key_id("public-key");
  public_key.set_key_material("key-material");
  ConfidentialEncryptionConfig encryption_config;
  std::tie(*encryption_config.mutable_encryption_key(),
           *options.add_kms_verifying_keys()) =
      CreateSignedPayload(public_key.SerializeAsString(),
                          GetValidPublicKeyHeaders("access policy hash"));

  confidentialcompute::AttestationVerificationRecord verification_record;
  AttestationTransparencyVerifier verifier(
      std::move(options),
      [&verification_record](
          confidentialcompute::AttestationVerificationRecord record) {
        verification_record = record;
      });
  ASSERT_OK(verifier.Verify({}, signed_endorsements, encryption_config));

  // Ensure that the verification record logger was called and provided the
  // relevant information.
  EXPECT_THAT(verification_record.pipeline_configuration(),
              EqualsProto(signed_endorsements.pipeline_configuration()));
  EXPECT_THAT(verification_record.encryption_key(),
              EqualsProto(encryption_config.encryption_key()));

  // The pipeline_configuration contains the access policy digest, which can be
  // used to retrieve the full access policy (see proto field documentation).

  // The encryption_key contains the endorsed evidence digest, which can be used
  // to retrieve the full EndorsedEvidence message (see proto field
  // documentation). This evidence can then be validated using the same process
  // shown in oak_rust_attestation_verifier_test.cc.
}

}  // namespace
}  // namespace fcp::client::attestation
