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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/rust/oak_attestation_verification_ffi.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"
#include "third_party/oak/proto/attestation/endorsement.pb.h"
#include "third_party/oak/proto/attestation/evidence.pb.h"
#include "third_party/oak/proto/attestation/reference_value.pb.h"
#include "third_party/oak/proto/attestation/verification.pb.h"

namespace fcp::client::attestation {
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidential_compute::OkpKey;
using ::google::internal::federatedcompute::v1::ConfidentialEncryptionConfig;
using ::oak::attestation::v1::AttestationResults;

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

  // TODO: b/307312707 -  Validate the data access policy, before proceeding to
  // validate the public key provided in the ConfidentialEncryptionConfig.

  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(encryption_config.public_key());
  if (!cwt.ok()) {
    std::string error_msg =
        "Parsing confidential aggregation public key failed.";
    return absl::Status(cwt.status().code(), error_msg);
  }

  // TODO: b/307312707 -  Validate that the public key we're about to return was
  // actually signed by the application layer signing key from the (validated)
  // attestation evidence.

  return *std::move(cwt->public_key);
}

}  // namespace fcp::client::attestation
