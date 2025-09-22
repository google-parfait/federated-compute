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

#include "fcp/confidentialcompute/payload_transparency/rekor.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/bind_front.h"
#include "absl/functional/function_ref.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/payload_transparency/signatures.h"
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"
#include "openssl/base.h"
#include "openssl/digest.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace fcp::confidential_compute::payload_transparency {
namespace {

using ::fcp::confidentialcompute::Key;
using ::fcp::confidentialcompute::RekorLogEntry;

// Returns the portion of `payload` between the first match of `start` and the
// next match of `end`, or `std::nullopt` if the substring is not found. This is
// used as an alternative to JSON parsing to avoid extra client dependencies.
std::optional<absl::string_view> ExtractSubstring(absl::string_view payload,
                                                  absl::string_view start,
                                                  absl::string_view end) {
  size_t i = payload.find(start);
  if (i == absl::string_view::npos) return std::nullopt;
  size_t substring_start = i + start.size();
  size_t j = payload.find(end, substring_start);
  if (j == absl::string_view::npos) return std::nullopt;
  return payload.substr(substring_start, j - substring_start);
}

// Extracts the data hash field from the log entry.
absl::StatusOr<std::string> ExtractDataHash(absl::string_view body) {
  // Naive substring extraction is safe because the JSON is canonicalized by
  // Rekor. Additionally, the hash value is base16 encoded, so it will never
  // contain a '"' character.
  std::optional<absl::string_view> data_hash_b16 =
      ExtractSubstring(body, R"("hash":{"algorithm":"sha256","value":")", "\"");
  if (!data_hash_b16) {
    return absl::InvalidArgumentError("log entry is missing data hash");
  }
  return absl::HexStringToBytes(*data_hash_b16);
}

// Extracts the signature field from the log entry.
absl::StatusOr<std::string> ExtractSignature(absl::string_view body) {
  // Naive substring extraction is safe because the JSON is canonicalized by
  // Rekor. Additionally, the hash value is base64 encoded, so it will never
  // contain a '"' character.
  std::optional<absl::string_view> signature_b64 =
      ExtractSubstring(body, R"("signature":{"content":")", "\"");
  if (!signature_b64) {
    return absl::InvalidArgumentError("log entry is missing signature");
  }
  std::string signature;
  if (!absl::Base64Unescape(*signature_b64, &signature)) {
    return absl::InvalidArgumentError("signature is not a valid base64 string");
  }
  return signature;
}

// Computes the root hash of a Merkle tree given an audit path. See RFC 6962
// section 2.1.1.
absl::StatusOr<absl::FixedArray<uint8_t>> ComputeRootHash(
    absl::string_view leaf_content, uint64_t leaf_index, uint64_t tree_size,
    const google::protobuf::RepeatedPtrField<std::string>& hashes) {
  // Ensure that the leaf index is valid. This also ensures that the tree is
  // non-empty.
  if (leaf_index >= tree_size) {
    return absl::InvalidArgumentError("leaf index larger than tree size");
  }

  // The structure of Rekor's Merkle trees can be defined recursively:
  //   * If the number of leaves N is a power of 2, then it's structured as a
  //     perfect tree.
  //   * Otherwise, the left child is a tree with k leaves and the right child
  //     is a tree with N-k leaves, where k is the largest power of 2 < N.
  // For a diagram of this structure, see
  // https://github.com/transparency-dev/merkle/blob/c91e4db1928aad36787de967e8c7d22ff92031a5/docs/data_model.md.
  //
  // The audit path for a leaf node is the hash of each sibling node on the path
  // from the leaf to the root (see RFC 6962). Combined with the leaf node's
  // hash, this is the set of hashes needed to re-compute the root hash. The
  // audit path does not specify whether each hash is for a left- or right-
  // sibling, but this can be determined from the size of the tree and the leaf
  // node's index.

  // The length of a node's audit path is equal to that node's depth in the
  // tree. The maximum depth in a Merkle tree with N nodes is ⌈log2(N - 1)⌉.
  // For convenience, we construct the audit path starting at the root
  std::vector<bool> reversed_audit_path;  // false=left, true=right
  reversed_audit_path.reserve(absl::bit_width(tree_size - 1));

  // Step 1: Find the path to the (largest) perfect tree containing the leaf.
  // While the tree isn't perfect...
  while (!absl::has_single_bit(tree_size)) {
    // If the node is in the left subtree, this step is complete.
    const uint64_t left_tree_size = absl::bit_floor(tree_size);
    if (leaf_index < left_tree_size) {
      reversed_audit_path.push_back(false /* left */);
      tree_size = left_tree_size;
      break;
    }

    // Otherwise, descend into the right subtree and repeat.
    reversed_audit_path.push_back(true /* right */);
    leaf_index -= left_tree_size;
    tree_size -= left_tree_size;
  }

  // Step 2: Find the path to the leaf node in the perfect subtree. The series
  // of left and right branches is given by the leaf_index's bit pattern.
  for (tree_size >>= 1; tree_size > 0; tree_size >>= 1) {
    reversed_audit_path.push_back(leaf_index & tree_size);
  }

  if (hashes.size() != reversed_audit_path.size()) {
    return absl::InvalidArgumentError("unexpected number of hashes");
  }

  // Reconstruct the root hash, starting at the leaf. 0x00 is prepended to leaf
  // hashes and 0x01 to interior hashes (see RFC 6962 section 2.1).
  std::optional<absl::FixedArray<uint8_t>> hash = ComputeDigest(
      EVP_sha256(), [&](absl::FunctionRef<void(absl::string_view)> emitter) {
        emitter(absl::string_view("\x00", 1));  // leaf node
        emitter(leaf_content);
      });
  for (int i = 0; i < hashes.size(); ++i) {
    hash.emplace(ComputeDigest(
        EVP_sha256(), [&](absl::FunctionRef<void(absl::string_view)> emitter) {
          absl::string_view hash_str(
              reinterpret_cast<const char*>(hash->data()), hash->size());
          emitter("\x01");  // non-leaf node
          if (reversed_audit_path.rbegin()[i]) {
            emitter(hashes[i]);
            emitter(hash_str);
          } else {
            emitter(hash_str);
            emitter(hashes[i]);
          }
        }));
  }
  return *std::move(hash);
}

// Constructs the Rekor checkpoint description:
//
//   <checkpoint_origin>
//   <tree_size>
//   <base64(root_hash)>
//   <optional other_content>
void BuildCheckpointDescription(
    const RekorLogEntry& log_entry, absl::Span<const uint8_t> root_hash,
    absl::FunctionRef<void(absl::string_view)> emitter) {
  constexpr absl::string_view kNewline = "\n";
  emitter(log_entry.checkpoint_origin());
  emitter(kNewline);

  emitter(absl::StrCat(log_entry.tree_size()));
  emitter(kNewline);

  emitter(absl::Base64Escape(absl::string_view(
      reinterpret_cast<const char*>(root_hash.data()), root_hash.size())));
  emitter(kNewline);

  for (absl::string_view content : log_entry.checkpoint_other_contents()) {
    emitter(content);
    emitter(kNewline);
  }
}

}  // namespace

absl::Status VerifyRekorLogEntry(const RekorLogEntry& log_entry,
                                 absl::Span<const Key* const> verifying_keys,
                                 absl::Span<const Key* const> rekor_keys,
                                 SignedData signed_data) {
  if (!absl::StrContains(log_entry.body(), R"("kind":"hashedrekord")")) {
    return absl::InvalidArgumentError("unsupported log entry kind");
  }

  // Extract important fields from the log entry.
  FCP_ASSIGN_OR_RETURN(std::string data_hash,
                       ExtractDataHash(log_entry.body()));
  FCP_ASSIGN_OR_RETURN(std::string signature,
                       ExtractSignature(log_entry.body()));

  // Verify that the log entry's data hash matches the payload.
  absl::FixedArray<uint8_t> digest = ComputeDigest(EVP_sha256(), signed_data);
  if (data_hash !=
      absl::string_view(reinterpret_cast<const char*>(digest.data()),
                        digest.size())) {
    return absl::InvalidArgumentError(
        "log entry data hash does not match payload");
  }

  // Check that the signature in the log entry is valid, reusing the digest
  // computed above.
  if (auto s = VerifySignature(signature, SignatureFormat::kAsn1,
                               verifying_keys, digest);
      !s.ok()) {
    return absl::Status(
        s.code(),
        absl::StrCat("log entry signature is invalid: ", s.message()));
  }

  // Verify the inclusion proof.
  FCP_ASSIGN_OR_RETURN(
      absl::FixedArray<uint8_t> root_hash,
      ComputeRootHash(log_entry.body(), log_entry.log_index(),
                      log_entry.tree_size(), log_entry.hashes()));
  absl::InlinedVector<const Key*, 2> filtered_rekor_keys;
  for (const Key* key : rekor_keys) {
    if (key->key_id() == log_entry.checkpoint_signature_key_id()) {
      filtered_rekor_keys.push_back(key);
    }
  }
  if (auto s = VerifySignature(
          log_entry.checkpoint_signature(), SignatureFormat::kP1363,
          filtered_rekor_keys,
          absl::bind_front(&BuildCheckpointDescription, log_entry, root_hash));
      !s.ok()) {
    return absl::Status(
        s.code(), absl::StrCat("log entry checkpoint signature is invalid: ",
                               s.message()));
  }
  return absl::OkStatus();
}

}  // namespace fcp::confidential_compute::payload_transparency
