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

#include "fcp/client/attestation/attestation_verifier.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/base/digest.h"
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"

namespace fcp::client::attestation {
using ::fcp::confidentialcompute::Key;
using ::google::internal::federatedcompute::v1::ConfidentialEncryptionConfig;

absl::StatusOr<AttestationVerifier::VerificationResult>
AlwaysFailingAttestationVerifier::Verify(
    const absl::Cord& access_policy,
    const confidentialcompute::SignedEndorsements& signed_endorsements,
    const ConfidentialEncryptionConfig& encryption_config) {
  return absl::FailedPreconditionError(
      "Attestation verification failed unconditionally.");
}

absl::StatusOr<AttestationVerifier::VerificationResult>
AlwaysPassingAttestationVerifier::Verify(
    const absl::Cord& access_policy,
    const confidentialcompute::SignedEndorsements& signed_endorsements,
    const ConfidentialEncryptionConfig& encryption_config) {
  if (!encryption_config.has_encryption_key()) {
    return absl::InvalidArgumentError("Missing encryption key");
  }
  Key key;
  if (!key.ParseFromString(encryption_config.encryption_key().payload())) {
    return absl::InvalidArgumentError("failed to parse encryption key");
  }
  std::string key_id = key.key_id();
  return VerificationResult{
      .public_key = std::move(key),
      .key_id = std::move(key_id),
      .access_policy_sha256 = ComputeSHA256(access_policy),
  };
}

}  // namespace fcp::client::attestation
