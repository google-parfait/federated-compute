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

import com.google.crypto.tink.AccessesPartialKey;
import com.google.crypto.tink.hybrid.HpkeParameters;
import com.google.crypto.tink.hybrid.HpkePublicKey;
import com.google.crypto.tink.util.Bytes;
import com.google.errorprone.annotations.Immutable;
import java.security.GeneralSecurityException;
import javax.annotation.Nullable;

/**
 * Hybrid Public Key Encryption (HPKE) encryption with an AEAD associated data input, backed by a
 * Tink {@link HpkePublicKey}.
 *
 * <p>HPKE RFC: https://www.rfc-editor.org/rfc/rfc9180.html
 *
 * <p>Branched from Tink's {@code com.google.crypto.tink.hybrid.internal.HpkeEncrypt}, i.e. the
 * "user-side" equivalent of adding {@code encryptWithAssociatedData} to it, with two differences:
 * the AEAD {@code associatedData} is a parameter instead of always being empty, and only {@code
 * NO_PREFIX} keys are accepted, so the output is the raw HPKE {@code enc || ct} of {@code
 * fcp/confidentialcompute/crypto.cc} (with an empty {@code associatedData} it is byte-compatible
 * with Tink's {@code HpkeEncrypt} for {@code NO_PREFIX} keys). The ciphersuite comes from the key's
 * parameters; see {@link HpkePrimitiveFactory} for the supported one.
 */
@Immutable
public final class HpkeEncryptWithAssociatedData implements HybridEncryptWithAssociatedData {
  @SuppressWarnings("Immutable") // We copy this on creation and never output it.
  private final byte[] recipientPublicKey;

  private final HpkeKem kem;
  private final HpkeKdf kdf;
  private final HpkeAead aead;

  private HpkeEncryptWithAssociatedData(
      Bytes recipientPublicKey, HpkeKem kem, HpkeKdf kdf, HpkeAead aead) {
    this.recipientPublicKey = recipientPublicKey.toByteArray();
    this.kem = kem;
    this.kdf = kdf;
    this.aead = aead;
  }

  /**
   * Creates a {@link HybridEncryptWithAssociatedData} for a single {@code NO_PREFIX} {@link
   * HpkePublicKey}.
   *
   * <p>Most code should not call this directly but obtain the primitive from a keyset: {@code
   * publicKeysetHandle.getPrimitive(FcpTinkConfiguration.get(), HybridEncryptWithAssociatedData
   * .class)}.
   */
  @AccessesPartialKey
  public static HybridEncryptWithAssociatedData create(HpkePublicKey key)
      throws GeneralSecurityException {
    HpkeParameters parameters = key.getParameters();
    HpkeParameters.Variant variant = parameters.getVariant();
    if (!variant.equals(HpkeParameters.Variant.NO_PREFIX)) {
      throw new GeneralSecurityException(
          "only NO_PREFIX HPKE keys are supported, got variant " + variant);
    }
    return new HpkeEncryptWithAssociatedData(
        key.getPublicKeyBytes(),
        HpkePrimitiveFactory.createKem(parameters.getKemId()),
        HpkePrimitiveFactory.createKdf(parameters.getKdfId()),
        HpkePrimitiveFactory.createAead(parameters.getAeadId()));
  }

  @Override
  public byte[] encrypt(
      byte[] plaintext, @Nullable byte[] contextInfo, @Nullable byte[] associatedData)
      throws GeneralSecurityException {
    checkNotNull(plaintext);
    byte[] info = contextInfo;
    if (info == null) {
      info = new byte[0];
    }
    byte[] aad = associatedData;
    if (aad == null) {
      aad = new byte[0];
    }
    HpkeContext context = HpkeContext.createSenderContext(recipientPublicKey, kem, kdf, aead, info);
    byte[] encapsulatedKey = context.getEncapsulatedKey();
    int ciphertextOffset = encapsulatedKey.length;
    byte[] encapsulatedKeyAndCiphertext = context.seal(plaintext, ciphertextOffset, aad);
    // In encapsulatedKeyAndCiphertext, the ciphertext starts at ciphertextOffset.
    // Copy the encapsulatedKey to the beginning of encapsulatedKeyAndCiphertext.
    System.arraycopy(
        /* src= */ encapsulatedKey,
        /* srcPos= */ 0,
        /* dest= */ encapsulatedKeyAndCiphertext,
        /* destPos= */ 0,
        /* length= */ encapsulatedKey.length);
    return encapsulatedKeyAndCiphertext;
  }
}
