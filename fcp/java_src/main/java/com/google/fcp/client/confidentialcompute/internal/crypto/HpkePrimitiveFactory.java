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

import com.google.crypto.tink.hybrid.HpkeParameters;
import java.security.GeneralSecurityException;

/**
 * Helper class for creating HPKE primitives from {@link HpkeParameters} identifiers.
 *
 * <p>Branched from Tink's {@code com.google.crypto.tink.hybrid.internal.HpkePrimitiveFactory},
 * reduced to the one ciphersuite this package supports ({@code DHKEM(X25519, HKDF-SHA256) /
 * HKDF-SHA256 / AES-128-GCM}, the ciphersuite of {@code fcp/confidentialcompute/crypto.cc}); every
 * other identifier is rejected with a {@link GeneralSecurityException}, so a key with an
 * unsupported suite fails at primitive creation rather than being used with the wrong algorithm.
 */
final class HpkePrimitiveFactory {
  /** Returns an {@link HpkeKem} primitive corresponding to {@code kemId}. */
  static HpkeKem createKem(HpkeParameters.KemId kemId) throws GeneralSecurityException {
    if (kemId == HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256) {
      return new X25519HpkeKem(HkdfHpkeKdf.create("HmacSha256"));
    }
    throw new GeneralSecurityException("Unsupported HPKE KEM: " + kemId);
  }

  /** Returns an {@link HpkeKdf} primitive corresponding to {@code kdfId}. */
  static HpkeKdf createKdf(HpkeParameters.KdfId kdfId) throws GeneralSecurityException {
    if (kdfId == HpkeParameters.KdfId.HKDF_SHA256) {
      return HkdfHpkeKdf.create("HmacSha256");
    }
    throw new GeneralSecurityException("Unsupported HPKE KDF: " + kdfId);
  }

  /** Returns an {@link HpkeAead} primitive corresponding to {@code aeadId}. */
  static HpkeAead createAead(HpkeParameters.AeadId aeadId) throws GeneralSecurityException {
    if (aeadId == HpkeParameters.AeadId.AES_128_GCM) {
      return new AesGcmHpkeAead(AesGcmHpkeAead.AES_128_GCM_KEY_LENGTH);
    }
    throw new GeneralSecurityException("Unsupported HPKE AEAD: " + aeadId);
  }

  private HpkePrimitiveFactory() {}
}
