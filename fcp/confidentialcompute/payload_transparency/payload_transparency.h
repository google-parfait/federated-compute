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

// Utilities for verifying fcp::confidentialcompute::SignedPayload messages.

#ifndef FCP_CONFIDENTIALCOMPUTE_PAYLOAD_TRANSPARENCY_PAYLOAD_TRANSPARENCY_H_
#define FCP_CONFIDENTIALCOMPUTE_PAYLOAD_TRANSPARENCY_PAYLOAD_TRANSPARENCY_H_

#include <functional>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "fcp/protos/confidentialcompute/access_policy_endorsement_options.pb.h"
#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"

namespace fcp::confidential_compute::payload_transparency {

// The result of a VerifySignedPayload call.
struct VerifySignedPayloadResult {
  // The list of headers verified as part of signature verification. If the
  // SignedPayload message contained multiple signatures, only headers from the
  // successfully validated signature will be included. The headers will be
  // returned from most deeply nested to least. The list will never be empty.
  std::vector<confidentialcompute::SignedPayload::Signature::Headers> headers;
};

// Verifies a SignedPayload message, returning information extracted during
// verification. To complete successfully, at least one of the SignedPayload's
// signatures must satisfy the following:
//
//   1. The payload must be signed either by a key in `verifying_keys` or by a
//      key in a SignedPayload that itself satisfies these conditions (i.e. a
//      signature chain).
//   2. If `transparency_log_options.require_transparency_log_entry` is true,
//      then the last signature in the chain must be included in a transparency
//      log. If false, it MAY be included in a transparency log.
//   3. All signature time constraints (issued at, not before, not after) must
//      be satisfied.
absl::StatusOr<VerifySignedPayloadResult> VerifySignedPayload(
    const confidentialcompute::SignedPayload& signed_payload,
    absl::Span<const confidentialcompute::Key* const> verifying_keys,
    const confidentialcompute::AccessPolicyEndorsementOptions::
        TransparencyLogOptions& transparency_log_options,
    absl::Time now);

// Returns a function that emits the components of the to-be-signed string for a
// SignedPayload. The return value is compatible with the SignedData variant and
// can be directly passed to ComputeDigest or VerifySignature.
//
// All strings must outlive the returned function.
//
// This function has been exposed to help construct valid SignedPayload messages
// in tests.
std::function<void(absl::FunctionRef<void(absl::string_view)>)>
GetSignedPayloadSigStructureEmitter(absl::string_view headers,
                                    absl::string_view payload);
}  // namespace fcp::confidential_compute::payload_transparency

#endif  // FCP_CONFIDENTIALCOMPUTE_PAYLOAD_TRANSPARENCY_PAYLOAD_TRANSPARENCY_H_
