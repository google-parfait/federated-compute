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

#ifndef FCP_CLIENT_ATTESTATION_ATTESTATION_VERIFIER_H_
#define FCP_CLIENT_ATTESTATION_ATTESTATION_VERIFIER_H_

#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/protos/confidentialcompute/signed_endorsements.pb.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"

namespace fcp::client::attestation {

// Verifies whether a given confidential data access policy and encryption
// config are valid.
class AttestationVerifier {
 public:
  virtual ~AttestationVerifier() = default;

  // Validates the given access policy and encryption config, and if both are
  // considered valid returns a public key extracted from the encryption config
  // with which confidential data can be encrypted. Returns an error if the
  // inputs are not considered a valid confidential configuration.
  virtual absl::StatusOr<fcp::confidential_compute::OkpKey> Verify(
      const absl::Cord& access_policy,
      const confidentialcompute::SignedEndorsements& signed_endorsements,
      const google::internal::federatedcompute::v1::
          ConfidentialEncryptionConfig& encryption_config) = 0;
};

// An AttestationVerifier implementation that always returns a verification
// failure.
class AlwaysFailingAttestationVerifier : public AttestationVerifier {
 public:
  absl::StatusOr<fcp::confidential_compute::OkpKey> Verify(
      const absl::Cord& access_policy,
      const confidentialcompute::SignedEndorsements& signed_endorsements,
      const google::internal::federatedcompute::v1::
          ConfidentialEncryptionConfig& encryption_config) override;
};

// An AttestationVerifier implementation that does no actual validation of the
// access policy or encryption config, but still extracts and returns the
// public encryption key from the encryption config (hence acting as if the
// validation actually passed).
class AlwaysPassingAttestationVerifier : public AttestationVerifier {
 public:
  absl::StatusOr<fcp::confidential_compute::OkpKey> Verify(
      const absl::Cord& access_policy,
      const confidentialcompute::SignedEndorsements& signed_endorsements,
      const google::internal::federatedcompute::v1::
          ConfidentialEncryptionConfig& encryption_config) override;
};

}  // namespace fcp::client::attestation

#endif  // FCP_CLIENT_ATTESTATION_ATTESTATION_VERIFIER_H_
