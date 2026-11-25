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

import com.google.crypto.tink.AccessesPartialKey;
import com.google.crypto.tink.Configuration;
import com.google.crypto.tink.KeysetHandle;
import com.google.crypto.tink.PublicKeyVerify;
import com.google.crypto.tink.signature.EcdsaParameters;
import com.google.crypto.tink.signature.EcdsaPublicKey;
import com.google.crypto.tink.signature.SignatureConfig2026;
import com.google.intelligence.fcp.confidentialcompute.Key;
import java.math.BigInteger;
import java.security.GeneralSecurityException;
import java.security.spec.ECPoint;
import java.util.Arrays;
import java.util.List;

/**
 * ECDSA P-256 signature verification over {@code fcp.confidentialcompute.Key} protos, matching
 * {@code VerifySignature} in {@code fcp/confidentialcompute/payload_transparency/signatures.cc}.
 *
 * <p>The C++ code hashes the data with SHA-256 and calls BoringSSL {@code ECDSA_verify} or {@code
 * ECDSA_verify_p1363} with each candidate key in turn. Here every usable key is imported into one
 * Tink keyset and Tink's {@link PublicKeyVerify} tries each enabled key, which has the same
 * outcome: success if any key verifies the signature.
 */
final class EcdsaSignatureVerifier {

  /** Signature encodings of the protocol; {@code SignatureFormat} in {@code signatures.h}. */
  enum SignatureFormat {
    /** ASN.1 DER {@code ECDSA-Sig-Value}, used by Rekor log entries. */
    ASN1(EcdsaParameters.SignatureEncoding.DER),
    /** Fixed-width IEEE P1363 {@code r || s}, used by {@code SignedPayload.raw_signature}. */
    P1363(EcdsaParameters.SignatureEncoding.IEEE_P1363);

    private final EcdsaParameters parameters;

    SignatureFormat(EcdsaParameters.SignatureEncoding encoding) {
      try {
        this.parameters =
            EcdsaParameters.builder()
                .setCurveType(EcdsaParameters.CurveType.NIST_P256)
                .setHashType(EcdsaParameters.HashType.SHA256)
                .setSignatureEncoding(encoding)
                .setVariant(EcdsaParameters.Variant.NO_PREFIX)
                .build();
      } catch (GeneralSecurityException e) {
        // Only thrown for parameter combinations Tink does not support; these are fixed.
        throw new IllegalStateException(e);
      }
    }
  }

  /** Tink's open-source signature configuration, which provides {@link PublicKeyVerify}. */
  private static final Configuration SIGNATURE_CONFIG = SignatureConfig2026.get();

  /** Length of an uncompressed SEC 1 P-256 point: {@code 0x04 || X (32) || Y (32)}. */
  private static final int UNCOMPRESSED_P256_POINT_LENGTH = 65;

  private static final int P256_COORDINATE_LENGTH = 32;

  private EcdsaSignatureVerifier() {}

  /**
   * Verifies {@code signature} over {@code data} with any of {@code keys}.
   *
   * <p>As in C++ {@code VerifySignature}, keys that cannot be used (purpose other than {@code
   * VERIFY}, algorithm other than {@code ECDSA_P256}, or key material that is not a point on P-256)
   * do not fail verification on their own; they are skipped and the remaining keys are tried. The
   * first such error is reported only if no key is usable.
   *
   * @param signature the signature bytes, in {@code format}
   * @param format the encoding of {@code signature}
   * @param keys candidate verifying keys, usually pre-filtered by key ID
   * @param data the signed bytes; they are hashed with SHA-256 as part of verification
   * @throws GeneralSecurityException if {@code keys} is empty, no key is usable, or no key verifies
   *     the signature
   */
  static void verify(byte[] signature, SignatureFormat format, List<Key> keys, byte[] data)
      throws GeneralSecurityException {
    if (keys.isEmpty()) {
      throw new GeneralSecurityException("no matching signing keys");
    }
    KeysetHandle.Builder keysetBuilder = KeysetHandle.newBuilder();
    GeneralSecurityException firstKeyError = null;
    boolean hasPrimary = false;
    for (Key key : keys) {
      EcdsaPublicKey tinkKey;
      try {
        tinkKey = toTinkEcdsaPublicKey(key, format.parameters);
      } catch (GeneralSecurityException e) {
        if (firstKeyError == null) {
          firstKeyError = e;
        }
        continue;
      }
      KeysetHandle.Builder.Entry entry = KeysetHandle.importKey(tinkKey).withRandomId();
      // A keyset needs a primary even though PublicKeyVerify tries every enabled key.
      if (!hasPrimary) {
        entry.makePrimary();
        hasPrimary = true;
      }
      keysetBuilder.addEntry(entry);
    }
    if (!hasPrimary) {
      throw firstKeyError;
    }
    PublicKeyVerify verifier =
        keysetBuilder.build().getPrimitive(SIGNATURE_CONFIG, PublicKeyVerify.class);
    try {
      verifier.verify(signature, data);
    } catch (GeneralSecurityException e) {
      throw new GeneralSecurityException("invalid signature", e);
    }
  }

  /**
   * Converts a {@link Key} with {@code Algorithm.ECDSA_P256}, {@code Purpose.VERIFY} and
   * uncompressed SEC 1 point key material into a Tink {@link EcdsaPublicKey}.
   *
   * <p>The checks match {@code VerifySignatureWithKey} in {@code signatures.cc}: the purpose and
   * algorithm checks are explicit there, and Tink's {@link EcdsaPublicKey.Builder#build} rejects a
   * point that is not on the curve as {@code EC_KEY_oct2key} does.
   *
   * <p>One difference: {@code EC_KEY_oct2key} also accepts compressed points ({@code 0x02} or
   * {@code 0x03} followed by X), while this accepts only the 65-byte uncompressed form. {@code
   * key.proto} specifies the uncompressed X9.62 format for {@code ECDSA_P256} key material, so a
   * compressed key is not valid input.
   *
   * @throws GeneralSecurityException if the key is not a usable ECDSA P-256 verifying key
   */
  @AccessesPartialKey
  private static EcdsaPublicKey toTinkEcdsaPublicKey(Key key, EcdsaParameters parameters)
      throws GeneralSecurityException {
    if (key.getPurpose() != Key.Purpose.VERIFY) {
      throw new GeneralSecurityException("key is not a verifying key");
    }
    if (key.getAlgorithm() != Key.Algorithm.ECDSA_P256) {
      throw new GeneralSecurityException("unsupported key algorithm: " + key.getAlgorithmValue());
    }
    byte[] rawPoint = key.getKeyMaterial().toByteArray();
    if (rawPoint.length != UNCOMPRESSED_P256_POINT_LENGTH || rawPoint[0] != 0x04) {
      throw new GeneralSecurityException("invalid public key: not an uncompressed P-256 point");
    }
    ECPoint publicPoint =
        new ECPoint(
            new BigInteger(1, Arrays.copyOfRange(rawPoint, 1, 1 + P256_COORDINATE_LENGTH)),
            new BigInteger(
                1,
                Arrays.copyOfRange(
                    rawPoint, 1 + P256_COORDINATE_LENGTH, UNCOMPRESSED_P256_POINT_LENGTH)));
    try {
      return EcdsaPublicKey.builder().setParameters(parameters).setPublicPoint(publicPoint).build();
    } catch (GeneralSecurityException e) {
      throw new GeneralSecurityException("invalid public key: " + e.getMessage(), e);
    }
  }
}
