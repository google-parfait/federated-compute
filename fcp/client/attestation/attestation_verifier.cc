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
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"

namespace fcp::client::attestation {
using ::fcp::confidentialcompute::Key;
using ::fcp::confidentialcompute::SignedPayload;

absl::StatusOr<AttestationVerifier::VerificationResult>
AlwaysFailingAttestationVerifier::Verify(const SignedPayload& pipeline_config,
                                         const SignedPayload& encryption_key) {
  return absl::FailedPreconditionError(
      "Attestation verification failed unconditionally.");
}

absl::StatusOr<AttestationVerifier::VerificationResult>
AlwaysPassingAttestationVerifier::Verify(const SignedPayload& pipeline_config,
                                         const SignedPayload& encryption_key) {
  if (encryption_key.payload().empty()) {
    return absl::InvalidArgumentError("Missing encryption key");
  }
  Key key;
  if (!key.ParseFromString(encryption_key.payload())) {
    return absl::InvalidArgumentError("failed to parse encryption key");
  }
  std::string key_id = key.key_id();
  return VerificationResult{
      .public_key = std::move(key),
      .key_id = std::move(key_id),
      .access_policy_sha256 = "",
  };
}

}  // namespace fcp::client::attestation
