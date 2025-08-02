// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![no_std]

use oak_proto_rust::oak::attestation::v1::{
    AttestationResults, Endorsements, Evidence, ReferenceValues,
};

/// Verifies Oak attestation evidence and endorsements given reference values.
pub fn verify_attestation(
    now_utc_millis: i64,
    evidence: &Evidence,
    endorsements: &Endorsements,
    reference_values: &ReferenceValues,
) -> AttestationResults {
    // TODO: b/432726860 - switch to AmdSevSnpDiceAttestationVerifier.
    oak_attestation_verification::verifier::to_attestation_results(
        &oak_attestation_verification::verifier::verify(
            now_utc_millis,
            &evidence,
            &endorsements,
            &reference_values,
        ),
    )
}
