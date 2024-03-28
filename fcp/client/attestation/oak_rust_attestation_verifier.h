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

#ifndef FCP_CLIENT_ATTESTATION_OAK_RUST_ATTESTATION_VERIFIER_H_
#define FCP_CLIENT_ATTESTATION_OAK_RUST_ATTESTATION_VERIFIER_H_

#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "fcp/client/attestation/attestation_verifier.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/protos/confidentialcompute/verification_record.pb.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"
#include "third_party/oak/proto/attestation/endorsement.pb.h"
#include "third_party/oak/proto/attestation/evidence.pb.h"
#include "third_party/oak/proto/attestation/reference_value.pb.h"
#include "third_party/oak/proto/attestation/verification.pb.h"

namespace fcp::client::attestation {

// An AttestationVerifier implementation that is backed by the Oak Attestation
// Verification library written in Rust.
class OakRustAttestationVerifier : public AttestationVerifier {
 public:
  // Creates a new verifier that will use the given `ReferenceValues` to
  // validate the attestation evidence and will allow the given access policies.
  //
  // The `record_logger` parameter will be called with a record of every
  // successful attestation verification.
  OakRustAttestationVerifier(
      oak::attestation::v1::ReferenceValues public_key_reference_values,
      absl::flat_hash_set<std::string> allowlisted_access_policy_hashes,
      absl::AnyInvocable<
          void(const fcp::confidentialcompute::AttestationVerificationRecord&)>
          record_logger)
      : public_key_reference_values_(std::move(public_key_reference_values)),
        allowlisted_access_policy_hashes_(
            std::move(allowlisted_access_policy_hashes)),
        record_logger_(std::move(record_logger)) {}

  absl::StatusOr<::fcp::confidential_compute::OkpKey> Verify(
      const absl::Cord& access_policy,
      const google::internal::federatedcompute::v1::
          ConfidentialEncryptionConfig& encryption_config) override;

 private:
  oak::attestation::v1::ReferenceValues public_key_reference_values_;
  absl::flat_hash_set<std::string> allowlisted_access_policy_hashes_;
  absl::AnyInvocable<void(
      const fcp::confidentialcompute::AttestationVerificationRecord&)>
      record_logger_;
};

// A verification record logger which simply logs the pretty printed
// `AttestationVerificationRecord` to `FCP_LOG(INFO)`.
//
// Note: this is not intended for use in production, but is convenient for use
// in tests etc.
void LogPrettyPrintedVerificationRecord(
    const fcp::confidentialcompute::AttestationVerificationRecord& record);

// A verification record logger which serializes, compresses, and base64-encodes
// the record before logging it to either `FCP_VLOG(1)` (non-Android builds) or
// to Android Logcat using the DEBUG log level (for Android builds).
void LogSerializedVerificationRecord(
    const fcp::confidentialcompute::AttestationVerificationRecord& record);

namespace internal {

// A helper function that serializes, compresses, and base64-encodes the record
// and chunks the result up into small chunks which are handed to the `logger`
// for actual printing.
void LogSerializedVerificationRecordWith(
    const fcp::confidentialcompute::AttestationVerificationRecord& record,
    absl::FunctionRef<void(absl::string_view message,
                           bool enclose_with_brackets)>
        logger);
}  // namespace internal

}  // namespace fcp::client::attestation

#endif  // FCP_CLIENT_ATTESTATION_OAK_RUST_ATTESTATION_VERIFIER_H_
