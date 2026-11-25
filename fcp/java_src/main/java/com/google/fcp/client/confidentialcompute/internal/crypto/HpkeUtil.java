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

import com.google.common.primitives.Bytes;
import com.google.crypto.tink.hybrid.HpkeParameters;
import java.security.GeneralSecurityException;

/**
 * Collection of helper functions for HPKE.
 *
 * <p>Branched from Tink's {@code com.google.crypto.tink.hybrid.internal.HpkeUtil}
 * (src/main/java/com/google/crypto/tink/hybrid/internal/HpkeUtil.java in tink-java), which is not
 * public Tink API, and reduced to the one ciphersuite this package supports: {@code DHKEM(X25519,
 * HKDF-SHA256) / HKDF-SHA256 / AES-128-GCM} in {@code mode_base}. Differences to Tink: byte-array
 * concatenation is Guava's {@code Bytes.concat} (so nothing here throws), and the {@code byte[]}
 * identifiers are named in lowerCamelCase as the style guide requires for mutable fields.
 */
final class HpkeUtil {
  // HPKE mode identifiers.
  static final byte[] baseMode = intToByteArray(1, 0x0);

  // HPKE KEM algorithm identifiers.
  static final byte[] x25519HkdfSha256KemId = intToByteArray(2, 0x20);

  // HPKE KDF algorithm identifiers.
  static final byte[] hkdfSha256KdfId = intToByteArray(2, 0x1);

  // HPKE AEAD algorithm identifiers.
  static final byte[] aes128GcmAeadId = intToByteArray(2, 0x1);

  static final byte[] emptySalt = new byte[0];

  private static final byte[] kemLabel = "KEM".getBytes(UTF_8);
  private static final byte[] hpkeLabel = "HPKE".getBytes(UTF_8);
  private static final byte[] hpkeV1Label = "HPKE-v1".getBytes(UTF_8);

  /**
   * Transforms a passed value to an MSB first byte array with the size of the specified capacity.
   * (i.e., {@code com.google.crypto.tink.subtle.Bytes#intToByteArray(int, int)} with MSB first
   * instead of LSB first).
   *
   * <p>The HPKE standard defines this function as I2OSP(n, w) where w = capacity and n = value.
   *
   * <p>https://www.rfc-editor.org/rfc/rfc9180.html#name-notation
   *
   * @param capacity size of the resulting byte array
   * @param value that should be represented as a byte array
   */
  static byte[] intToByteArray(int capacity, int value) {
    if ((capacity > 4) || (capacity < 0)) {
      throw new IllegalArgumentException("capacity must be between 0 and 4");
    }
    // Check that 0 <= value < 256^capacity.
    // For capacity == 4, all positive values are valid.
    if (value < 0 || (capacity < 4 && (value >= 1 << (8 * capacity)))) {
      throw new IllegalArgumentException("value too large");
    }
    final byte[] result = new byte[capacity];
    for (int i = 0; i < capacity; i++) {
      result[i] = (byte) ((value >> (8 * (capacity - i - 1))) & 0xFF);
    }
    return result;
  }

  /**
   * Generates KEM suite id from {@code kemId} according to the definition in
   * https://www.rfc-editor.org/rfc/rfc9180.html#section-4.1-5. Only used for KEM suite id.
   */
  static byte[] kemSuiteId(byte[] kemId) {
    return Bytes.concat(kemLabel, kemId);
  }

  /**
   * Generates HPKE suite id from {@code kemId}, {@code kdfId}, and {@code aeadId} according to the
   * definition in https://www.rfc-editor.org/rfc/rfc9180.html#section-5.1-8. Used for any non-KEM
   * suite id.
   */
  static byte[] hpkeSuiteId(byte[] kemId, byte[] kdfId, byte[] aeadId) {
    return Bytes.concat(hpkeLabel, kemId, kdfId, aeadId);
  }

  /**
   * Transforms {@code ikm} into labeled ikm using {@code label} and {@code suiteId} according to
   * {@code LabeledExtract()} defined in https://www.rfc-editor.org/rfc/rfc9180.html#section-4.
   */
  static byte[] labelIkm(String label, byte[] ikm, byte[] suiteId) {
    return Bytes.concat(hpkeV1Label, suiteId, label.getBytes(UTF_8), ikm);
  }

  /**
   * Transforms {@code info} into labeled info using {@code label}, {@code suiteId}, and {@code
   * length} according to {@code LabeledExpand()} defined in
   * https://www.rfc-editor.org/rfc/rfc9180.html#section-4.
   */
  static byte[] labelInfo(String label, byte[] info, byte[] suiteId, int length) {
    return Bytes.concat(
        intToByteArray(2, length), hpkeV1Label, suiteId, label.getBytes(UTF_8), info);
  }

  /**
   * Returns the encapsulated key length (in bytes) for the specified {@code kemId}. This value
   * corresponds to the 'Nenc' column in the following table.
   *
   * <p>https://www.rfc-editor.org/rfc/rfc9180.html#name-key-encapsulation-mechanism.
   */
  static int encodingSizeInBytes(HpkeParameters.KemId kemId) throws GeneralSecurityException {
    if (kemId == HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256) {
      return 32;
    }
    throw new GeneralSecurityException("Unsupported HPKE KEM: " + kemId);
  }

  private HpkeUtil() {}
}
