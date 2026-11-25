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

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.fcp.client.confidentialcompute.internal.crypto.EcdsaSignatureVerifier.SignatureFormat;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions.TransparencyLogOptions;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.intelligence.fcp.confidentialcompute.LogEntry;
import com.google.intelligence.fcp.confidentialcompute.SignedPayload;
import com.google.intelligence.fcp.confidentialcompute.SignedPayload.Signature.Headers;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.Timestamp;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.security.GeneralSecurityException;
import java.time.DateTimeException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Verifies {@code SignedPayload} messages, matching {@code VerifySignedPayload} in {@code
 * fcp/confidentialcompute/payload_transparency/payload_transparency.cc}.
 *
 * <p>A {@code SignedPayload} carries a payload and one or more signatures over it. Each signature
 * names its verifying key either by ID, referring to one of the caller's trusted keys, or as a
 * nested {@code SignedPayload} whose payload is the key, which must itself verify, forming a chain
 * down to a trusted key. The signature bytes are either a raw ECDSA signature or a transparency log
 * entry that stands in for one.
 */
final class SignedPayloadVerifier {

  /** Context string of the signed structure; see {@link #constructSignedData}. */
  private static final ByteString SIGNED_PAYLOAD_CONTEXT = ByteString.copyFromUtf8("SignedPayload");

  private SignedPayloadVerifier() {}

  /**
   * Verifies that at least one signature on {@code signedPayload} is valid and returns the headers
   * of the verified signature chain.
   *
   * <p>As in C++ {@code VerifySignedPayload}, the signatures are tried in order and the first one
   * that verifies wins; if none does, the error of every signature is reported. The returned list
   * is ordered from the innermost signature (the one closest to a trusted key, whose headers carry
   * the Oak application signature and claims) to the outermost (the signature directly on {@code
   * signedPayload}, whose headers carry the access policy hash). It is never empty on success.
   *
   * @param signedPayload the message to verify
   * @param verifyingKeys the trusted keys that may terminate a signature chain
   * @param transparencyLogOptions whether a transparency log entry is required, and the Rekor keys
   * @param now the time to check the signature headers' validity period against
   * @return the verified chain's headers, innermost first; callers that only need to know the
   *     payload verifies may ignore it
   * @throws GeneralSecurityException if no signature verifies
   */
  @CanIgnoreReturnValue
  static List<Headers> verify(
      SignedPayload signedPayload,
      List<Key> verifyingKeys,
      TransparencyLogOptions transparencyLogOptions,
      Instant now)
      throws GeneralSecurityException {
    List<String> errors = new ArrayList<>();
    for (SignedPayload.Signature signature : signedPayload.getSignaturesList()) {
      try {
        return verifySignature(
            signedPayload, signature, verifyingKeys, transparencyLogOptions, now);
      } catch (GeneralSecurityException e) {
        errors.add(e.getMessage());
      }
    }
    throw new GeneralSecurityException("signature verification failed: " + errors);
  }

  /**
   * Verifies one signature of {@code signedPayload}, matching C++ {@code
   * VerifySignedPayloadSignature}, and returns the headers of its chain, innermost first.
   *
   * @throws GeneralSecurityException if the headers are invalid or out of their validity period,
   *     the verifier cannot be resolved, a transparency log entry is required but absent, or the
   *     signature does not verify
   */
  private static List<Headers> verifySignature(
      SignedPayload signedPayload,
      SignedPayload.Signature signature,
      List<Key> verifyingKeys,
      TransparencyLogOptions transparencyLogOptions,
      Instant now)
      throws GeneralSecurityException {
    Headers headers;
    try {
      headers = Headers.parseFrom(signature.getHeaders(), ExtensionRegistryLite.getEmptyRegistry());
    } catch (InvalidProtocolBufferException e) {
      throw new GeneralSecurityException("failed to parse signature headers", e);
    }
    verifyValidityPeriod(headers, now);

    // Resolve the key(s) to verify with. There is usually one, but key IDs are not guaranteed to be
    // unique, so several of the trusted keys may match. A nested verifying_key is only trusted once
    // its own chain verifies; its headers come first in the result because they are innermost.
    List<Headers> headerChain = new ArrayList<>();
    List<Key> filteredVerifyingKeys = new ArrayList<>();
    switch (signature.getVerifierCase()) {
      case VERIFYING_KEY_ID -> {
        for (Key key : verifyingKeys) {
          if (key.getAlgorithmValue() == headers.getAlgorithmValue()
              && key.getKeyId().equals(signature.getVerifyingKeyId())) {
            filteredVerifyingKeys.add(key);
          }
        }
      }
      case VERIFYING_KEY -> {
        headerChain.addAll(
            verify(signature.getVerifyingKey(), verifyingKeys, transparencyLogOptions, now));
        try {
          filteredVerifyingKeys.add(
              Key.parseFrom(
                  signature.getVerifyingKey().getPayload(),
                  ExtensionRegistryLite.getEmptyRegistry()));
        } catch (InvalidProtocolBufferException e) {
          throw new GeneralSecurityException("failed to parse verifying key", e);
        }
      }
      default -> throw new GeneralSecurityException("unsupported Signature.verifier");
    }

    byte[] signedData =
        constructSignedData(
            signature.getHeaders().toByteArray(), signedPayload.getPayload().toByteArray());

    switch (signature.getSignatureCase()) {
      case RAW_SIGNATURE -> {
        // Only the last signature in a chain, the one made with a trusted key, can be required to
        // have been published in a transparency log.
        if (!signature.hasVerifyingKey()
            && transparencyLogOptions.getRequireTransparencyLogEntry()) {
          throw new GeneralSecurityException(
              "SignedPayload.Signature does not have a transparency log entry");
        }
        EcdsaSignatureVerifier.verify(
            signature.getRawSignature().toByteArray(),
            SignatureFormat.P1363,
            filteredVerifyingKeys,
            signedData);
      }
      case LOG_ENTRY ->
          verifyLogEntry(
              signature.getLogEntry(), filteredVerifyingKeys, transparencyLogOptions, signedData);
      default -> throw new GeneralSecurityException("unsupported Signature.signature");
    }

    headerChain.add(headers);
    return headerChain;
  }

  /**
   * Checks the validity period in {@code headers} against {@code now}, matching C++ {@code
   * VerifyHeaders}. Every field is optional; {@code issued_at} serves as the start of the period
   * only when {@code not_before} is unset, and the period has no end without {@code not_after}.
   *
   * @throws GeneralSecurityException if {@code now} is outside the period
   */
  private static void verifyValidityPeriod(Headers headers, Instant now)
      throws GeneralSecurityException {
    if (headers.hasNotBefore() && now.isBefore(toInstant(headers.getNotBefore()))) {
      throw new GeneralSecurityException("not_before is in the future");
    }
    if (!headers.hasNotBefore()
        && headers.hasIssuedAt()
        && now.isBefore(toInstant(headers.getIssuedAt()))) {
      throw new GeneralSecurityException("issued_at is in the future");
    }
    if (headers.hasNotAfter() && now.isAfter(toInstant(headers.getNotAfter()))) {
      throw new GeneralSecurityException("not_after is in the past");
    }
  }

  /**
   * Converts a protobuf {@link Timestamp} to an {@link Instant}. Seconds outside the range of
   * {@link Instant} saturate to {@link Instant#MAX} or {@link Instant#MIN}, so that any int64 is
   * accepted as in C++ {@code ParseTimestamp} and compares the same way against the current time.
   */
  private static Instant toInstant(Timestamp timestamp) {
    try {
      return Instant.ofEpochSecond(timestamp.getSeconds(), timestamp.getNanos());
    } catch (DateTimeException | ArithmeticException e) {
      return timestamp.getSeconds() > 0 ? Instant.MAX : Instant.MIN;
    }
  }

  /**
   * Verifies a transparency log entry in place of a raw signature, matching C++ {@code
   * VerifyLogEntry}. Rekor is the only log supported.
   *
   * @throws GeneralSecurityException if the entry is of another kind or does not verify
   */
  private static void verifyLogEntry(
      LogEntry logEntry,
      List<Key> verifyingKeys,
      TransparencyLogOptions transparencyLogOptions,
      byte[] signedData)
      throws GeneralSecurityException {
    switch (logEntry.getKindCase()) {
      case REKOR ->
          RekorLogEntryVerifier.verify(
              logEntry.getRekor(),
              verifyingKeys,
              transparencyLogOptions.getRekorVerifyingKeysList(),
              signedData);
      default -> throw new GeneralSecurityException("unsupported log entry type");
    }
  }

  /**
   * Builds the bytes a {@code SignedPayload} signature covers, matching {@code
   * GetSignedPayloadSigStructureEmitter} in {@code payload_transparency.cc}: {@code
   * "SignedPayload"}, the serialized headers and the payload, each preceded by its length as a
   * varint. The context string keeps these signatures from being confused with signatures over
   * other structures, like {@code "Signature1"} in COSE (RFC 9052 section 4.4).
   */
  private static byte[] constructSignedData(byte[] headers, byte[] payload)
      throws GeneralSecurityException {
    try {
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      CodedOutputStream out = CodedOutputStream.newInstance(bytes);
      for (byte[] part : new byte[][] {SIGNED_PAYLOAD_CONTEXT.toByteArray(), headers, payload}) {
        out.writeUInt64NoTag(part.length);
        out.writeRawBytes(part);
      }
      out.flush();
      return bytes.toByteArray();
    } catch (IOException e) {
      throw new GeneralSecurityException("failed to construct signed data", e);
    }
  }
}
