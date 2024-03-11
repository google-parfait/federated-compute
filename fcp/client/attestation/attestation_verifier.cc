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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"

namespace fcp::client::attestation {
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidential_compute::OkpKey;
using ::google::internal::federatedcompute::v1::ConfidentialEncryptionConfig;

absl::StatusOr<OkpKey> AlwaysFailingAttestationVerifier::Verify(
    const absl::Cord& access_policy,
    const ConfidentialEncryptionConfig& encryption_config) {
  return absl::FailedPreconditionError(
      "Attestation verification failed unconditionally.");
}

absl::StatusOr<OkpKey> AlwaysPassingAttestationVerifier::Verify(
    const absl::Cord& access_policy,
    const ConfidentialEncryptionConfig& encryption_config) {
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(encryption_config.public_key());
  if (!cwt.ok()) {
    std::string error_msg =
        "Parsing confidential aggregation public key failed.";
    return absl::Status(cwt.status().code(), error_msg);
  }
  return *cwt->public_key;
}

}  // namespace fcp::client::attestation
