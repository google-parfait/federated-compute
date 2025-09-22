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

#ifndef FCP_CONFIDENTIALCOMPUTE_PAYLOAD_TRANSPARENCY_REKOR_H_
#define FCP_CONFIDENTIALCOMPUTE_PAYLOAD_TRANSPARENCY_REKOR_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "fcp/confidentialcompute/payload_transparency/signatures.h"
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"

namespace fcp::confidential_compute::payload_transparency {

// Verifies that a Rekor log entry is for the provided `signed_data`, signed by
// one of the `verifying_keys`, and included in the Rekor log. The
// `verifying_keys` should be pre-filtered to those that could plausibly verify
// the log entry (e.g. by checking the key id).
absl::Status VerifyRekorLogEntry(
    const confidentialcompute::RekorLogEntry& log_entry,
    absl::Span<const confidentialcompute::Key* const> verifying_keys,
    absl::Span<const confidentialcompute::Key* const> rekor_keys,
    SignedData signed_data);

}  // namespace fcp::confidential_compute::payload_transparency

#endif  // FCP_CONFIDENTIALCOMPUTE_PAYLOAD_TRANSPARENCY_REKOR_H_
