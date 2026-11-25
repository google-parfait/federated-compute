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

import com.google.errorprone.annotations.Immutable;
import java.security.GeneralSecurityException;
import java.security.InvalidAlgorithmParameterException;
import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;

/**
 * AES-GCM HPKE AEAD variant.
 *
 * <p>Branched from Tink's {@code com.google.crypto.tink.hybrid.internal.AesGcmHpkeAead}, reduced to
 * AES-128-GCM (the AEAD of the ciphersuite {@code fcp/confidentialcompute/crypto.cc} uses) and
 * using the JCE {@code AES/GCM/NoPadding} cipher directly instead of Tink's internal {@code
 * InsecureNonceAesGcmJce}.
 */
@Immutable
final class AesGcmHpkeAead implements HpkeAead {
  /** {@code Nk} of AES-128-GCM (RFC 9180, Section 7.3). */
  static final int AES_128_GCM_KEY_LENGTH = 16;

  /** {@code Nn} of AES-GCM (RFC 9180, Section 7.3). */
  static final int NONCE_LENGTH = 12;

  /** {@code Nt} of AES-GCM (RFC 9180, Section 7.3). */
  static final int TAG_LENGTH = 16;

  private final int keyLength;

  AesGcmHpkeAead(int keyLength) throws InvalidAlgorithmParameterException {
    if (keyLength != AES_128_GCM_KEY_LENGTH) {
      throw new InvalidAlgorithmParameterException("Unsupported key length: " + keyLength);
    }
    this.keyLength = keyLength;
  }

  @Override
  public byte[] seal(
      byte[] key, byte[] nonce, byte[] plaintext, int ciphertextOffset, byte[] associatedData)
      throws GeneralSecurityException {
    checkNotNull(key);
    checkNotNull(nonce);
    checkNotNull(plaintext);
    if (key.length != keyLength) {
      throw new InvalidAlgorithmParameterException("Unexpected key length: " + key.length);
    }
    if (nonce.length != NONCE_LENGTH) {
      throw new GeneralSecurityException("nonce is wrong size");
    }
    Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
    cipher.init(
        Cipher.ENCRYPT_MODE,
        new SecretKeySpec(key, "AES"),
        new GCMParameterSpec(8 * TAG_LENGTH, nonce));
    if (associatedData != null && associatedData.length != 0) {
      cipher.updateAAD(associatedData);
    }
    int ciphertextSize = cipher.getOutputSize(plaintext.length);
    // Check that outputSize is not longer than the max. size of a Java array.
    if (ciphertextSize > Integer.MAX_VALUE - ciphertextOffset) {
      throw new GeneralSecurityException("plaintext too long");
    }
    byte[] output = new byte[ciphertextOffset + ciphertextSize];
    int written = cipher.doFinal(plaintext, 0, plaintext.length, output, ciphertextOffset);
    if (written != ciphertextSize) {
      throw new GeneralSecurityException("not enough data written");
    }
    return output;
  }

  @Override
  public byte[] open(
      byte[] key, byte[] nonce, byte[] ciphertext, int ciphertextOffset, byte[] associatedData)
      throws GeneralSecurityException {
    checkNotNull(key);
    checkNotNull(nonce);
    checkNotNull(ciphertext);
    if (key.length != keyLength) {
      throw new InvalidAlgorithmParameterException("Unexpected key length: " + key.length);
    }
    if (nonce.length != NONCE_LENGTH) {
      throw new GeneralSecurityException("nonce is wrong size");
    }
    if (ciphertext.length < ciphertextOffset + TAG_LENGTH) {
      throw new GeneralSecurityException("ciphertext too short");
    }
    Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
    cipher.init(
        Cipher.DECRYPT_MODE,
        new SecretKeySpec(key, "AES"),
        new GCMParameterSpec(8 * TAG_LENGTH, nonce));
    if (associatedData != null && associatedData.length != 0) {
      cipher.updateAAD(associatedData);
    }
    return cipher.doFinal(ciphertext, ciphertextOffset, ciphertext.length - ciphertextOffset);
  }

  @Override
  public byte[] getAeadId() throws GeneralSecurityException {
    switch (keyLength) {
      case AES_128_GCM_KEY_LENGTH:
        return HpkeUtil.aes128GcmAeadId;
      default:
        throw new GeneralSecurityException("Could not determine HPKE AEAD ID");
    }
  }

  @Override
  public int getKeyLength() {
    return keyLength;
  }

  @Override
  public int getNonceLength() {
    return NONCE_LENGTH;
  }
}
