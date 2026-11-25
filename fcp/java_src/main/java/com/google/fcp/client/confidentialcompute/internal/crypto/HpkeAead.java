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

import com.google.errorprone.annotations.Immutable;
import java.security.GeneralSecurityException;

/**
 * Interface for Hybrid Public Key Encryption (HPKE) authenticated encryption with associated data
 * (AEAD).
 *
 * <p>HPKE RFC is available at https://www.rfc-editor.org/rfc/rfc9180.html.
 *
 * <p>Branched from Tink's {@code com.google.crypto.tink.hybrid.internal.HpkeAead}.
 */
@Immutable
interface HpkeAead {
  /**
   * Performs authenticated encryption of {@code plaintext} and {@code associatedData} using {@code
   * key} and {@code nonce} according to the HPKE AEAD specification.
   *
   * <p>More details available at
   * https://www.rfc-editor.org/rfc/rfc9180.html#name-encryption-and-decryption.
   */
  default byte[] seal(byte[] key, byte[] nonce, byte[] plaintext, byte[] associatedData)
      throws GeneralSecurityException {
    return seal(key, nonce, plaintext, /* ciphertextOffset= */ 0, associatedData);
  }

  /**
   * Like {@link #seal(byte[], byte[], byte[], byte[])}, but the ciphertext starts at {@code
   * ciphertextOffset} in the returned array (the leading bytes are left zero for the caller to
   * fill, e.g. with the encapsulated key).
   */
  byte[] seal(
      byte[] key, byte[] nonce, byte[] plaintext, int ciphertextOffset, byte[] associatedData)
      throws GeneralSecurityException;

  /**
   * Performs authenticated decryption of {@code ciphertext} and {@code associatedData} using {@code
   * key} and {@code nonce} according to the HPKE AEAD specification.
   *
   * <p>More details available at
   * https://www.rfc-editor.org/rfc/rfc9180.html#name-encryption-and-decryption.
   */
  default byte[] open(byte[] key, byte[] nonce, byte[] ciphertext, byte[] associatedData)
      throws GeneralSecurityException {
    return open(key, nonce, ciphertext, /* ciphertextOffset= */ 0, associatedData);
  }

  /**
   * Like {@link #open(byte[], byte[], byte[], byte[])}, but the ciphertext starts at {@code
   * ciphertextOffset} in {@code ciphertext}.
   */
  byte[] open(
      byte[] key, byte[] nonce, byte[] ciphertext, int ciphertextOffset, byte[] associatedData)
      throws GeneralSecurityException;

  /**
   * Returns the HPKE AEAD algorithm identifier for the underlying AEAD implementation.
   *
   * <p>More details at
   * https://www.rfc-editor.org/rfc/rfc9180.html#name-authenticated-encryption-wi.
   */
  byte[] getAeadId() throws GeneralSecurityException;

  /** Returns key length (in bytes) for this algorithm (i.e., parameter 'Nk' in HPKE RFC). */
  int getKeyLength();

  /** Returns nonce length (in bytes) for this algorithm (i.e., parameter 'Nn' in HPKE RFC). */
  int getNonceLength();
}
