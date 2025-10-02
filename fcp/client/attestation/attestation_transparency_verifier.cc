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
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/payload_transparency/payload_transparency.h"
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "fcp/protos/confidentialcompute/signed_endorsements.pb.h"
#include "fcp/protos/confidentialcompute/verification_record.pb.h"
#include "fcp/protos/federatedcompute/confidential_encryption_config.pb.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace fcp::client::attestation {
namespace {

using ::fcp::confidential_compute::payload_transparency::VerifySignedPayload;
using ::fcp::confidential_compute::payload_transparency::
    VerifySignedPayloadResult;
using ::fcp::confidentialcompute::Key;
using ::fcp::confidentialcompute::SignedEndorsements;
using ::fcp::confidentialcompute::SignedPayload;
using ::google::internal::federatedcompute::v1::ConfidentialEncryptionConfig;

// Returns an error if `actual_claims` is not a superset of `expected_claims`.
absl::Status CheckRequiredClaims(
    const google::protobuf::RepeatedPtrField<std::string>& actual_claims,
    const std::vector<std::string>& expected_claims) {
  std::vector<absl::string_view> missing_claims;
  for (const auto& expected : expected_claims) {
    // The number of claims is expected to be small enough that repeated linear
    // searches are fast enough.
    if (absl::c_find(actual_claims, expected) == actual_claims.end()) {
      missing_claims.push_back(expected);
    }
  }
  return missing_claims.empty() ? absl::OkStatus()
                                : absl::InvalidArgumentError(absl::StrCat(
                                      "missing required claims: ",
                                      absl::StrJoin(missing_claims, ", ")));
}

}  // namespace

absl::StatusOr<AttestationTransparencyVerifier::VerificationResult>
AttestationTransparencyVerifier::Verify(
    // The `access_policy` parameter is ignored because the pipeline
    // configuration in `signed_endorsements` and the upload encryption key in
    // `encryption_config` contain the SHA-256 digest of the access policy,
    // ensuring the same level of transparency is maintained while reducing
    // download sizes. The full policy can be retrieved using the digest (see
    // proto field comments).
    const absl::Cord& access_policy,
    const SignedEndorsements& signed_endorsements,
    const ConfidentialEncryptionConfig& encryption_config) {
  const absl::Time now = absl::Now();

  // Verify that the encryption key is properly signed, included in a
  // transparency log (if required by `transparency_log_options`), and valid.
  FCP_ASSIGN_OR_RETURN(
      VerifySignedPayloadResult encryption_key_result,
      VerifySignedPayload(encryption_config.encryption_key(),
                          options_.kms_verifying_keys(),
                          options_.transparency_log_options(), now));
  Key key;
  if (!key.ParseFromString(encryption_config.encryption_key().payload())) {
    return absl::InvalidArgumentError("failed to parse encryption key");
  }
  if (encryption_key_result.headers.empty()) {
    return absl::InternalError("VerifySignedPayload returned no headers");
  }
  // The access policy hash should be in the encryption key's signature's
  // headers.
  absl::string_view access_policy_sha256 =
      encryption_key_result.headers.back().access_policy_sha256();
  if (access_policy_sha256.empty()) {
    return absl::InvalidArgumentError(
        "encryption key headers missing access policy SHA-256");
  }
  // The Oak application signature should be in the most deeply nested
  // Signature.verifying_key's headers, which will be the first entry in
  // `encryption_key_result.headers` per the VerifySignedPayload documentation.
  SignedPayload::Signature::Headers oak_application_signature_headers;
  if (!oak_application_signature_headers.ParseFromString(
          encryption_key_result.headers.front()
              .oak_application_signature()
              .headers())) {
    return absl::InvalidArgumentError(
        "failed to parse oak application signature headers");
  }
  if (oak_application_signature_headers.endorsed_evidence_sha256().empty()) {
    return absl::InvalidArgumentError(
        "oak application signature headers missing endorsed evidence SHA-256");
  }
  FCP_RETURN_IF_ERROR(CheckRequiredClaims(
      encryption_key_result.headers.front().claims(),
      {
          // Built from open source.
          "https://github.com/project-oak/oak/blob/main/docs/tr/claim/92939.md",
      }));

  // Verify that the pipeline configuration is properly signed, included in a
  // transparency log, and valid.
  FCP_ASSIGN_OR_RETURN(
      VerifySignedPayloadResult pipeline_config_result,
      VerifySignedPayload(signed_endorsements.pipeline_configuration(),
                          options_.access_policy_verifying_keys(),
                          options_.transparency_log_options(), now));
  SignedEndorsements::PipelineConfiguration config;
  if (!config.ParseFromString(
          signed_endorsements.pipeline_configuration().payload())) {
    return absl::InvalidArgumentError("failed to parse pipeline configuration");
  }
  if (config.access_policy_sha256() != access_policy_sha256) {
    return absl::InvalidArgumentError(
        "pipeline configuration access policy SHA-256 does not match "
        "encryption key");
  }
  // TODO: b/444302065 - Check required claims for the access policy.

  // Verification is now complete! Log the key information used to perform the
  // verification to the provided logger. This allows someone observing these
  // logs to replay the same verification, as well as look up the binaries that
  // will process the encrypted data. See the
  // AttestationVerificationRecordContainsEnoughInfoToReplayVerification test in
  // attestation_transparency_verifier_test.cc for an example of how this
  // information can be used for that purpose.
  confidentialcompute::AttestationVerificationRecord verification_record;
  *verification_record.mutable_encryption_key() =
      encryption_config.encryption_key();
  *verification_record.mutable_pipeline_configuration() =
      signed_endorsements.pipeline_configuration();
  record_logger_(verification_record);

  std::string key_id = key.key_id();
  return VerificationResult{
      .public_key = std::move(key),
      .key_id = std::move(key_id),
      .access_policy_sha256 = std::move(*config.mutable_access_policy_sha256()),
  };
}

}  // namespace fcp::client::attestation
