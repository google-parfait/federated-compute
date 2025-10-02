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

#include "fcp/confidentialcompute/payload_transparency/payload_transparency.h"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/timestamp.pb.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/payload_transparency/rekor.h"
#include "fcp/confidentialcompute/payload_transparency/signatures.h"
#include "fcp/protos/confidentialcompute/access_policy_endorsement_options.pb.h"
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"
#include "google/protobuf/io/coded_stream.h"

namespace fcp::confidential_compute::payload_transparency {
namespace {

using ::fcp::confidentialcompute::AccessPolicyEndorsementOptions;
using ::fcp::confidentialcompute::Key;
using ::fcp::confidentialcompute::LogEntry;
using ::fcp::confidentialcompute::SignedPayload;

// Converts a google::protobuf::Timestamp to an absl::Time value.
absl::Time ParseTimestamp(const google::protobuf::Timestamp& timestamp) {
  return absl::FromUnixSeconds(timestamp.seconds()) +
         absl::Nanoseconds(timestamp.nanos());
}

// Verifies common header fields (if set).
absl::Status VerifyHeaders(const SignedPayload::Signature::Headers& headers,
                           absl::Time now) {
  if (headers.has_not_before() && now < ParseTimestamp(headers.not_before())) {
    return absl::InvalidArgumentError("not_before is in the future");
  }
  // issued_at should be used as the start time if not_before is unset.
  if (!headers.has_not_before() && headers.has_issued_at() &&
      now < ParseTimestamp(headers.issued_at())) {
    return absl::InvalidArgumentError("issued_at is in the future");
  }
  if (headers.has_not_after() && now > ParseTimestamp(headers.not_after())) {
    return absl::InvalidArgumentError("not_after is in the past");
  }
  return absl::OkStatus();
}

// Verifies a transparency log entry.
absl::Status VerifyLogEntry(
    const LogEntry& log_entry, absl::Span<const Key* const> verifying_keys,
    const AccessPolicyEndorsementOptions::TransparencyLogOptions&
        transparency_log_options,
    SignedData signed_data) {
  switch (log_entry.kind_case()) {
    case LogEntry::kRekor:
      return VerifyRekorLogEntry(
          log_entry.rekor(), verifying_keys,
          transparency_log_options.rekor_verifying_keys(), signed_data);
    default:
      return absl::UnimplementedError("unsupported log entry type");
  }
}

// Verifies a single signature in a SignedPayload message.
absl::StatusOr<VerifySignedPayloadResult> VerifySignedPayloadSignature(
    const SignedPayload& signed_payload,
    const SignedPayload::Signature& signature,
    absl::Span<const Key* const> verifying_keys,
    const AccessPolicyEndorsementOptions::TransparencyLogOptions&
        transparency_log_options,
    absl::Time now) {
  SignedPayload::Signature::Headers headers;
  if (!headers.ParseFromString(signature.headers())) {
    return absl::InvalidArgumentError("failed to parse signature headers");
  }
  FCP_RETURN_IF_ERROR(VerifyHeaders(headers, now));

  // Find the key(s) to use for verification. There should usually only be one
  // matching key, but multiple matches are possible since key_ids are not
  // guaranteed to be unique. Since claims come from the innermost signature,
  // this also determines the VerifySignedPayloadResult.
  VerifySignedPayloadResult verify_result;
  Key owned_key;  // For use with kSigningKey.
  absl::InlinedVector<const Key*, 1> filtered_verifying_keys;
  switch (signature.verifier_case()) {
    // A verifying_key_id is a reference to one of the provided verifying keys.
    case SignedPayload::Signature::kVerifyingKeyId:
      for (const Key* key : verifying_keys) {
        if (key->algorithm() == headers.algorithm() &&
            key->key_id() == signature.verifying_key_id()) {
          filtered_verifying_keys.push_back(key);
        }
      }
      break;

    // A verifying_key is a nested SignedPayload message containing the Key,
    // which can only be trusted if SignedPayload verification succeeds.
    case SignedPayload::Signature::kVerifyingKey: {
      FCP_ASSIGN_OR_RETURN(
          verify_result,
          VerifySignedPayload(signature.verifying_key(), verifying_keys,
                              transparency_log_options, now));
      if (!owned_key.ParseFromString(signature.verifying_key().payload())) {
        return absl::InvalidArgumentError("failed to parse verifying key");
      }
      filtered_verifying_keys.push_back(&owned_key);
      break;
    }

    default:
      return absl::InvalidArgumentError("unsupported Signature.verifier");
  }

  // Construct the payload to be signed.
  auto signed_data = GetSignedPayloadSigStructureEmitter(
      signature.headers(), signed_payload.payload());

  // Verify the signature using the key identifier above.
  switch (signature.signature_case()) {
    case SignedPayload::Signature::kRawSignature:
      // If this is the last signature in the chain, it may require a
      // transparency log entry.
      if (!signature.has_verifying_key() &&
          transparency_log_options.require_transparency_log_entry()) {
        return absl::InvalidArgumentError(
            "SignedPayload.Signature does not have a transparency log entry");
      }
      FCP_RETURN_IF_ERROR(
          VerifySignature(signature.raw_signature(), SignatureFormat::kP1363,
                          filtered_verifying_keys, signed_data));
      break;

    case SignedPayload::Signature::kLogEntry:
      FCP_RETURN_IF_ERROR(
          VerifyLogEntry(signature.log_entry(), filtered_verifying_keys,
                         transparency_log_options, signed_data));
      break;

    default:
      return absl::InvalidArgumentError("unsupported Signature.signature");
  }

  verify_result.headers.push_back(std::move(headers));
  return verify_result;
}

}  // namespace

absl::StatusOr<VerifySignedPayloadResult> VerifySignedPayload(
    const SignedPayload& signed_payload,
    absl::Span<const Key* const> verifying_keys,
    const AccessPolicyEndorsementOptions::TransparencyLogOptions&
        transparency_log_options,
    absl::Time now) {
  // Succeed if any signature can be successfully verified. On failure, return
  // the error for each signature.
  std::vector<absl::Status> errors;
  errors.reserve(signed_payload.signatures().size());
  for (const auto& signature : signed_payload.signatures()) {
    if (auto result = VerifySignedPayloadSignature(
            signed_payload, signature, verifying_keys, transparency_log_options,
            now);
        result.ok()) {
      return *result;
    } else {
      errors.push_back(std::move(result).status());
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "signature verification failed: [",
      absl::StrJoin(errors, ", ",
                    [](std::string* out, const absl::Status& status) {
                      absl::StrAppend(out, status.ToString());
                    }),
      "]"));
}

std::function<void(absl::FunctionRef<void(absl::string_view)>)>
GetSignedPayloadSigStructureEmitter(absl::string_view headers,
                                    absl::string_view payload) {
  // The format is {"SignedPayload", headers, payload}, where each component is
  // preceeded by its varint-encoded length. The "SignedPayload" context string
  // is included to avoid ambiguity with other signatures; this is equivalent to
  // "Signature" or "Signature1" in RFC 9052 section 4.4.
  return
      [payload, headers](absl::FunctionRef<void(absl::string_view)> emitter) {
        uint8_t buffer[10];  // Maximum length of varint-encoded uint64_t.
        for (absl::string_view part : std::initializer_list<absl::string_view>{
                 "SignedPayload", headers, payload}) {
          uint8_t* end = google::protobuf::io::CodedOutputStream::WriteVarint64ToArray(
              part.size(), buffer);
          emitter(absl::string_view(reinterpret_cast<const char*>(buffer),
                                    end - buffer));
          emitter(part);
        }
      };
}

}  // namespace fcp::confidential_compute::payload_transparency
