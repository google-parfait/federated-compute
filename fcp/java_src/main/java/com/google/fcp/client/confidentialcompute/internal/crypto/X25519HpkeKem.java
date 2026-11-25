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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.primitives.Bytes;
import com.google.crypto.tink.AccessesPartialKey;
import com.google.crypto.tink.InsecureSecretKeyAccess;
import com.google.crypto.tink.hybrid.HpkePrivateKey;
import com.google.crypto.tink.subtle.X25519;
import com.google.errorprone.annotations.Immutable;
import java.security.GeneralSecurityException;
import java.util.Arrays;

/**
 * Diffie-Hellman-based X25519-HKDF HPKE KEM variant.
 *
 * <p>Branched from Tink's {@code com.google.crypto.tink.hybrid.internal.X25519HpkeKem}, without the
 * authenticated ({@code mode_auth}) operations. It uses Tink's public {@link X25519} directly.
 */
@Immutable
final class X25519HpkeKem implements HpkeKem {
  private final HkdfHpkeKdf hkdf;

  /** Construct X25519-HKDF HPKE KEM using {@code hkdf}. */
  X25519HpkeKem(HkdfHpkeKdf hkdf) {
    this.hkdf = hkdf;
  }

  private byte[] deriveKemSharedSecret(
      byte[] dhSharedSecret, byte[] senderEphemeralPublicKey, byte[] recipientPublicKey)
      throws GeneralSecurityException {
    byte[] kemContext = Bytes.concat(senderEphemeralPublicKey, recipientPublicKey);
    return extractAndExpand(dhSharedSecret, kemContext);
  }

  private byte[] extractAndExpand(byte[] dhSharedSecret, byte[] kemContext)
      throws GeneralSecurityException {
    byte[] kemSuiteId = HpkeUtil.kemSuiteId(HpkeUtil.x25519HkdfSha256KemId);
    return hkdf.extractAndExpand(
        /* salt= */ null,
        dhSharedSecret,
        "eae_prk",
        kemContext,
        "shared_secret",
        kemSuiteId,
        hkdf.getMacLength());
  }

  /** Helper function factored out to facilitate unit testing. */
  HpkeKemEncapOutput encapsulateWithFixedEphemeralKey(
      byte[] recipientPublicKey, byte[] ephemeralPrivateKey, byte[] ephemeralPublicKey)
      throws GeneralSecurityException {
    byte[] dhSharedSecret = X25519.computeSharedSecret(ephemeralPrivateKey, recipientPublicKey);
    byte[] kemSharedSecret =
        deriveKemSharedSecret(dhSharedSecret, ephemeralPublicKey, recipientPublicKey);
    return new HpkeKemEncapOutput(kemSharedSecret, ephemeralPublicKey);
  }

  @Override
  public HpkeKemEncapOutput encapsulate(byte[] recipientPublicKey) throws GeneralSecurityException {
    checkNotNull(recipientPublicKey);
    byte[] ephemeralPrivateKey = X25519.generatePrivateKey();
    byte[] ephemeralPublicKey = X25519.publicFromPrivate(ephemeralPrivateKey);
    return encapsulateWithFixedEphemeralKey(
        recipientPublicKey, ephemeralPrivateKey, ephemeralPublicKey);
  }

  @AccessesPartialKey
  @Override
  public byte[] decapsulate(byte[] encapsulatedKey, HpkePrivateKey recipientPrivateKey)
      throws GeneralSecurityException {
    checkNotNull(encapsulatedKey);
    checkNotNull(recipientPrivateKey);
    byte[] dhSharedSecret =
        X25519.computeSharedSecret(
            recipientPrivateKey.getPrivateKeyBytes().toByteArray(InsecureSecretKeyAccess.get()),
            encapsulatedKey);
    return deriveKemSharedSecret(
        dhSharedSecret,
        encapsulatedKey,
        recipientPrivateKey.getPublicKey().getPublicKeyBytes().toByteArray());
  }

  @Override
  public byte[] getKemId() throws GeneralSecurityException {
    if (Arrays.equals(hkdf.getKdfId(), HpkeUtil.hkdfSha256KdfId)) {
      return HpkeUtil.x25519HkdfSha256KemId;
    }
    throw new GeneralSecurityException("Could not determine HPKE KEM ID");
  }
}
