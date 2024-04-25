/*
 * Copyright 2024 Google LLC
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

#include "fcp/client/attestation/oak_rust_attestation_verifier.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "fcp/base/digest.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/rust/oak_attestation_verification_ffi.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/access_policy.pb.h"
#include "fcp/protos/confidentialcompute/verification_record.pb.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"
#include "proto/attestation/endorsement.pb.h"
#include "proto/attestation/evidence.pb.h"
#include "proto/attestation/reference_value.pb.h"
#include "proto/attestation/verification.pb.h"

namespace fcp::client::attestation {
using ::fcp::confidential_compute::EcdsaP256R1SignatureVerifier;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidentialcompute::DataAccessPolicy;
using ::google::internal::federatedcompute::v1::ConfidentialEncryptionConfig;
using ::oak::attestation::v1::AttestationResults;

// See https://www.iana.org/assignments/cose/cose.xhtml.
constexpr int64_t kAlgorithmES256 = -7;

absl::StatusOr<OkpKey> OakRustAttestationVerifier::Verify(
    const absl::Cord& access_policy,
    const ConfidentialEncryptionConfig& encryption_config) {
  // Validate the attestation evidence provided in the encryption config, using
  // the `public_key_reference_values_` provided to us at construction time.
  FCP_ASSIGN_OR_RETURN(
      AttestationResults attestation_results,
      fcp::client::rust::oak_attestation_verification_ffi::VerifyAttestation(
          absl::Now(), encryption_config.attestation_evidence(),
          encryption_config.attestation_endorsements(),
          public_key_reference_values_));

  if (attestation_results.status() != AttestationResults::STATUS_SUCCESS) {
    return absl::FailedPreconditionError(absl::Substitute(
        "Attestation verification failed (status: $0, reason: $1).",
        attestation_results.status(), attestation_results.reason()));
  }

  // Ensure that the provided data access policy parses correctly.
  DataAccessPolicy parsed_access_policy;
  if (!parsed_access_policy.ParseFromCord(access_policy)) {
    return absl::FailedPreconditionError("DataAccessPolicy failed to parse.");
  }

  // Next, validate whether the provided access policy is allowlisted.
  //
  // Note that we must calculate the hash over the unparsed absl::Cord holding
  // the serialized access policy bytes, rather than calculating it over
  // `parsed_access_policy.SerializeAsString()`, because the re-serialized
  // representation may differ slightly from the original serialization (e.g.
  // due non-deterministic map serialization).
  auto access_policy_hash =
      absl::BytesToHexString(ComputeSHA256(access_policy));
  if (!allowlisted_access_policy_hashes_.contains(access_policy_hash)) {
    return absl::FailedPreconditionError(absl::Substitute(
        "Data access policy not in allowlist ($0).", access_policy_hash));
  }

  // Next, let's validate the CWT-encoded public key.
  //
  // First, we'll parse it and check that is signed by the expected ECDSA256
  // signing algorithm. Then validate that it CWT is correctly signed by the
  // signing key from the AttestationResults, and then we'll return the key to
  // the caller.
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(encryption_config.public_key());
  if (!cwt.ok()) {
    return absl::Status(
        cwt.status().code(),
        absl::Substitute(
            "Parsing confidential aggregation public key failed ($0).",
            cwt.status().message()));
  }

  // Ensure that the CWT is signed using the expected ECDSA256 signing
  // algorithm.
  if (!cwt->algorithm.has_value() || *cwt->algorithm != kAlgorithmES256) {
    return absl::FailedPreconditionError(absl::Substitute(
        "Unsupported COSE signing algorithm ($0).", *cwt->algorithm));
  }

  // Extract the protected parts of the CWT (the COSE Sig_structure), which are
  // the parts of the CWT that are covered by the signature.
  absl::StatusOr<std::string> cwt_sig_structure =
      OkpCwt::GetSigStructureForVerifying(
          /*cwt=*/encryption_config.public_key(), /*aad=*/"");
  if (!cwt_sig_structure.ok()) {
    return absl::Status(
        cwt.status().code(),
        absl::Substitute("Failed to extract CWT Sig_structure ($0).",
                         cwt_sig_structure.status().message()));
  }

  // Now verify the CWT signature.
  auto signature_verifier = EcdsaP256R1SignatureVerifier::Create(
      attestation_results.signing_public_key());
  if (!signature_verifier.ok()) {
    return absl::Status(
        signature_verifier.status().code(),
        absl::Substitute("Failed to create signature verifier ($0).",
                         signature_verifier.status().message()));
  }
  auto sig_verify_result =
      signature_verifier->Verify(*cwt_sig_structure, cwt->signature);
  if (!sig_verify_result.ok()) {
    return absl::FailedPreconditionError(absl::Substitute(
        "Signature verification failed (code: $0, msg: $1).",
        sig_verify_result.code(), sig_verify_result.message()));
  }

  // Verification of the attestation evidence, the access policy, and the public
  // encryption key's signature succeeded!

  // We now log the key information we used to perform the verification to the
  // provided logger. This allows someone observing these logs to replay the
  // same verification, as well as look up the binaries that will process the
  // encrypted data. See the
  // AttestationVerificationRecordContainsEnoughInfoToReplayVerification test in
  // oak_rust_attestation_verifier_test.cc for an example of how this
  // information can be used for that purpose.
  confidentialcompute::AttestationVerificationRecord verification_record;
  *verification_record.mutable_attestation_evidence() =
      encryption_config.attestation_evidence();
  *verification_record.mutable_attestation_endorsements() =
      encryption_config.attestation_endorsements();
  *verification_record.mutable_data_access_policy() =
      std::move(parsed_access_policy);
  record_logger_(verification_record);

  // Return the public key with which the caller can now safely encrypt data
  // to be uploaded. Only the attested server binary will have access to the
  // decryption key, and the it will only allow the decryption key to be
  // used by binaries/applications allowed by the data access policy.
  return *std::move(cwt->public_key);
}

}  // namespace fcp::client::attestation
