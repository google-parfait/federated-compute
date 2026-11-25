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

import java.security.GeneralSecurityException;
import javax.annotation.Nullable;

/**
 * An AEAD whose nonce is chosen by the caller instead of by the primitive.
 *
 * <p>This is a <em>custom primitive</em>: it is not part of Tink's public primitive set, but it can
 * still be backed by Tink key objects and obtained from a Tink {@code KeysetHandle} via a custom
 * {@link com.google.crypto.tink.Configuration} (see {@code FcpTinkConfiguration}, added at the end
 * of this stack).
 *
 * <p>Ciphertexts are exactly what the underlying algorithm produces for the given nonce (for
 * AES-GCM-SIV: {@code ciphertext || tag}). There is no Tink output prefix: implementations only
 * accept {@code NO_PREFIX} keys and reject keys of any other {@code Variant} when the primitive is
 * created. The nonce is <b>not</b> part of the ciphertext either; the caller is responsible for
 * transmitting or agreeing on it out of band.
 *
 * <p><b>Security warning:</b> callers are responsible for nonce management. For AES-GCM-SIV,
 * reusing a nonce with the same key leaks only whether two (plaintext, associated data) pairs are
 * equal; for AES-GCM it would be catastrophic. Never use this interface with a non-misuse-resistant
 * AEAD and a fixed nonce.
 */
public interface CallerProvidedNonceAead {

  /**
   * Encrypts {@code plaintext} with {@code associatedData} under the caller-provided {@code nonce}.
   *
   * @param associatedData the associated data to authenticate; {@code null} is treated as empty
   * @return the raw ciphertext of the underlying algorithm (for AES-GCM-SIV: {@code ciphertext ||
   *     tag}), without an output prefix or the nonce
   */
  byte[] encrypt(byte[] nonce, byte[] plaintext, @Nullable byte[] associatedData)
      throws GeneralSecurityException;

  /**
   * Decrypts {@code ciphertext} (as produced by {@link #encrypt}) with {@code associatedData} under
   * the caller-provided {@code nonce}.
   *
   * @param associatedData the associated data to authenticate; {@code null} is treated as empty
   */
  byte[] decrypt(byte[] nonce, byte[] ciphertext, @Nullable byte[] associatedData)
      throws GeneralSecurityException;
}
