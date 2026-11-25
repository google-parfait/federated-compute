// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.fcp.client.confidentialcompute.internal.crypto;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.CharMatcher;
import com.google.common.io.BaseEncoding;
import com.google.fcp.client.confidentialcompute.internal.crypto.EcdsaSignatureVerifier.SignatureFormat;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.intelligence.fcp.confidentialcompute.RekorLogEntry;
import com.google.protobuf.ByteString;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;

/**
 * Verifies a Rekor transparency log entry for a signed payload, matching {@code
 * VerifyRekorLogEntry} in {@code fcp/confidentialcompute/payload_transparency/rekor.cc}.
 *
 * <p>A {@link RekorLogEntry} proves that a signature over the payload was published in a <a
 * href="https://github.com/sigstore/rekor">Rekor</a> log: its {@code body} is the published
 * hashedrekord entry (the SHA-256 of the signed data and a DER signature over it), and the rest of
 * the message is the Merkle inclusion proof of that body in a tree whose checkpoint Rekor signed.
 */
final class RekorLogEntryVerifier {

  // Fields of the canonical hashedrekord JSON body that the verifier reads.
  private static final String HASHEDREKORD_KIND = "\"kind\":\"hashedrekord\"";
  private static final String DATA_HASH_START = "\"hash\":{\"algorithm\":\"sha256\",\"value\":\"";
  private static final String SIGNATURE_START = "\"signature\":{\"content\":\"";
  private static final String FIELD_END = "\"";

  // RFC 6962 section 2.1 domain separation prefixes for leaf and interior Merkle tree nodes.
  private static final byte MERKLE_LEAF_PREFIX = 0x00;
  private static final byte MERKLE_NODE_PREFIX = 0x01;

  private RekorLogEntryVerifier() {}

  /**
   * Verifies that {@code logEntry} is a valid, included Rekor entry for a signature over {@code
   * signedData} by one of {@code verifyingKeys}.
   *
   * <p>The steps follow C++ {@code VerifyRekorLogEntry}:
   *
   * <ol>
   *   <li>The body must be a hashedrekord entry, and its data hash must equal the SHA-256 of {@code
   *       signedData}, so the entry is about this payload and not some other.
   *   <li>The DER signature in the body must verify over {@code signedData} with one of {@code
   *       verifyingKeys}; this is the signature the log entry stands in for.
   *   <li>The Merkle root recomputed from the body and the audit path must be the one in the
   *       checkpoint, and the checkpoint signature must verify with the Rekor key named by {@code
   *       checkpoint_signature_key_id}, so the entry really is in the log.
   * </ol>
   *
   * @param logEntry the log entry to verify
   * @param verifyingKeys keys the entry's signature may have been made with
   * @param rekorKeys the trusted Rekor checkpoint signing keys, from {@code
   *     TransparencyLogOptions.rekor_verifying_keys}
   * @param signedData the bytes the signature covers
   * @throws GeneralSecurityException if any step fails
   */
  static void verify(
      RekorLogEntry logEntry, List<Key> verifyingKeys, List<Key> rekorKeys, byte[] signedData)
      throws GeneralSecurityException {
    byte[] bodyBytes = logEntry.getBody().toByteArray();
    String body = logEntry.getBody().toStringUtf8();
    if (!body.contains(HASHEDREKORD_KIND)) {
      throw new GeneralSecurityException("unsupported log entry kind");
    }
    byte[] dataHash = extractDataHash(body);
    byte[] signature = extractSignature(body);

    // 1. The log entry's data hash must match the payload.
    if (!MessageDigest.isEqual(dataHash, sha256(signedData))) {
      throw new GeneralSecurityException("log entry data hash does not match payload");
    }

    // 2. The signature recorded in the log entry must be valid. Rekor stores it DER encoded.
    try {
      EcdsaSignatureVerifier.verify(signature, SignatureFormat.ASN1, verifyingKeys, signedData);
    } catch (GeneralSecurityException e) {
      throw new GeneralSecurityException("log entry signature is invalid: " + e.getMessage(), e);
    }

    // 3. The inclusion proof must lead to a root that Rekor signed.
    byte[] rootHash =
        computeRootHash(
            bodyBytes, logEntry.getLogIndex(), logEntry.getTreeSize(), logEntry.getHashesList());
    List<Key> filteredRekorKeys = new ArrayList<>();
    for (Key key : rekorKeys) {
      if (key.getKeyId().equals(logEntry.getCheckpointSignatureKeyId())) {
        filteredRekorKeys.add(key);
      }
    }
    try {
      EcdsaSignatureVerifier.verify(
          logEntry.getCheckpointSignature().toByteArray(),
          SignatureFormat.P1363,
          filteredRekorKeys,
          buildCheckpointDescription(logEntry, rootHash));
    } catch (GeneralSecurityException e) {
      throw new GeneralSecurityException(
          "log entry checkpoint signature is invalid: " + e.getMessage(), e);
    }
  }

  /**
   * Returns the SHA-256 data hash from the hashedrekord body.
   *
   * @throws GeneralSecurityException if the field is missing or not hex
   */
  private static byte[] extractDataHash(String body) throws GeneralSecurityException {
    String dataHashB16 =
        extractField(body, DATA_HASH_START, FIELD_END, "log entry is missing data hash");
    try {
      return BaseEncoding.base16().ignoreCase().decode(dataHashB16);
    } catch (IllegalArgumentException e) {
      throw new GeneralSecurityException("data hash is not a valid base16 string", e);
    }
  }

  /**
   * Returns the DER signature from the hashedrekord body.
   *
   * @throws GeneralSecurityException if the field is missing or not base64
   */
  private static byte[] extractSignature(String body) throws GeneralSecurityException {
    String signatureB64 =
        extractField(body, SIGNATURE_START, FIELD_END, "log entry is missing signature");
    try {
      // absl::Base64Unescape, which rekor.cc uses, skips whitespace; Guava does not.
      return BaseEncoding.base64().decode(CharMatcher.whitespace().removeFrom(signatureB64));
    } catch (IllegalArgumentException e) {
      throw new GeneralSecurityException("signature is not a valid base64 string", e);
    }
  }

  /**
   * Returns the text between the first occurrence of {@code start} in {@code body} and the next
   * occurrence of {@code end} after it.
   *
   * <p>This is substring search rather than JSON parsing, as in {@code rekor.cc}: Rekor
   * canonicalizes the JSON it serves, and the two values read this way are hex and base64, which
   * never contain a quote, so no parser is needed on the client.
   *
   * @throws GeneralSecurityException with {@code missingMessage} if either marker is not found
   */
  private static String extractField(String body, String start, String end, String missingMessage)
      throws GeneralSecurityException {
    int startIndex = body.indexOf(start);
    if (startIndex < 0) {
      throw new GeneralSecurityException(missingMessage);
    }
    int fieldStart = startIndex + start.length();
    int fieldEnd = body.indexOf(end, fieldStart);
    if (fieldEnd < 0) {
      throw new GeneralSecurityException(missingMessage);
    }
    return body.substring(fieldStart, fieldEnd);
  }

  /**
   * Recomputes the Merkle tree root from a leaf and its audit path, matching {@code
   * ComputeRootHash} in {@code rekor.cc}. See RFC 6962 section 2.1.1.
   *
   * <p>Rekor's tree of {@code N} leaves is a perfect tree if {@code N} is a power of two; otherwise
   * its left subtree has the largest power of two below {@code N} leaves and its right subtree the
   * rest, recursively. The audit path lists the sibling hash at each level from the leaf up but not
   * whether that sibling is on the left or the right, so that is derived here from {@code treeSize}
   * and {@code leafIndex}, walking from the root down: first through the imperfect levels to the
   * perfect subtree that holds the leaf, then through the perfect subtree, where the leaf index's
   * bits give the branches.
   *
   * @param leafContent the leaf, hashed with the {@code 0x00} leaf prefix
   * @param leafIndex the leaf's index in the tree, as an unsigned long
   * @param treeSize the number of leaves in the tree, as an unsigned long
   * @param hashes the sibling hashes from the leaf up, with the {@code 0x01} node prefix applied
   *     here
   * @throws GeneralSecurityException if the index is out of range or the path has the wrong length
   */
  private static byte[] computeRootHash(
      byte[] leafContent, long leafIndex, long treeSize, List<ByteString> hashes)
      throws GeneralSecurityException {
    if (Long.compareUnsigned(leafIndex, treeSize) >= 0) {
      throw new GeneralSecurityException("leaf index larger than tree size");
    }

    // The branch at each level from the root down; false is left, true is right.
    List<Boolean> rootToLeafBranches = new ArrayList<>();

    // Descend through the imperfect levels until reaching the perfect subtree that holds the leaf.
    while (Long.bitCount(treeSize) != 1) {
      long leftTreeSize = Long.highestOneBit(treeSize);
      if (Long.compareUnsigned(leafIndex, leftTreeSize) < 0) {
        rootToLeafBranches.add(false);
        treeSize = leftTreeSize;
        break;
      }
      rootToLeafBranches.add(true);
      leafIndex -= leftTreeSize;
      treeSize -= leftTreeSize;
    }

    // Within the perfect subtree the leaf index's bits, high to low, give the branches.
    for (treeSize >>>= 1; treeSize > 0; treeSize >>>= 1) {
      rootToLeafBranches.add((leafIndex & treeSize) != 0);
    }

    if (hashes.size() != rootToLeafBranches.size()) {
      throw new GeneralSecurityException("unexpected number of hashes");
    }

    // Hash from the leaf up. The audit path is ordered leaf to root, so it pairs with the branches
    // in reverse: a right branch means the sibling in the path is on the left.
    byte[] hash = sha256(MERKLE_LEAF_PREFIX, leafContent);
    for (int i = 0; i < hashes.size(); i++) {
      byte[] sibling = hashes.get(i).toByteArray();
      boolean siblingIsLeft = rootToLeafBranches.get(rootToLeafBranches.size() - 1 - i);
      hash =
          siblingIsLeft
              ? sha256(MERKLE_NODE_PREFIX, sibling, hash)
              : sha256(MERKLE_NODE_PREFIX, hash, sibling);
    }
    return hash;
  }

  /**
   * Builds the text Rekor signs for a checkpoint, matching {@code BuildCheckpointDescription} in
   * {@code rekor.cc}: the origin, the tree size, the base64 root hash, then each line of other
   * content, every line ending in a newline.
   */
  private static byte[] buildCheckpointDescription(RekorLogEntry logEntry, byte[] rootHash) {
    StringBuilder description = new StringBuilder();
    description.append(logEntry.getCheckpointOrigin()).append('\n');
    description.append(Long.toUnsignedString(logEntry.getTreeSize())).append('\n');
    description.append(BaseEncoding.base64().encode(rootHash)).append('\n');
    for (String content : logEntry.getCheckpointOtherContentsList()) {
      description.append(content).append('\n');
    }
    return description.toString().getBytes(UTF_8);
  }

  /** Returns the SHA-256 digest of {@code data}. */
  private static byte[] sha256(byte[] data) throws GeneralSecurityException {
    return MessageDigest.getInstance("SHA-256").digest(data);
  }

  /**
   * Returns the SHA-256 digest of {@code prefix} followed by the concatenation of {@code parts}.
   */
  private static byte[] sha256(byte prefix, byte[]... parts) throws GeneralSecurityException {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    digest.update(prefix);
    for (byte[] part : parts) {
      digest.update(part);
    }
    return digest.digest();
  }
}
