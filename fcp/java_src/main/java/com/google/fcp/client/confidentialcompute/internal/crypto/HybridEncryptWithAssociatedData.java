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
 * Hybrid encryption that, in addition to the HPKE {@code info} ("context info"), binds an AEAD
 * associated data string to the ciphertext.
 *
 * <p>This is a one-shot API: it does not allow creating a sequence of encryptions once a shared
 * secret is established. When {@link #encrypt} is called again, a new shared secret is derived.
 *
 * <p>Tink's {@link com.google.crypto.tink.HybridEncrypt} always uses an empty AEAD {@code aad} in
 * the HPKE {@code Seal} operation (RFC 9180, Section 6.1), which is incompatible with protocols
 * that authenticate additional data at the AEAD layer (e.g. {@code
 * fcp/confidentialcompute/crypto.cc}). This is a <em>custom primitive</em> that exposes the {@code
 * aad} parameter while being backed by Tink's {@link com.google.crypto.tink.hybrid.HpkePublicKey}.
 *
 * <p>Ciphertexts are the raw HPKE output {@code enc || ct}, where {@code enc} is the KEM
 * encapsulated key (32 bytes for X25519) and {@code ct} the AEAD ciphertext including the 16-byte
 * tag. There is no Tink output prefix: implementations only accept {@code NO_PREFIX} keys and
 * reject keys of any other {@code Variant} when the primitive is created.
 */
public interface HybridEncryptWithAssociatedData {

  /**
   * Encrypts {@code plaintext}, binding {@code contextInfo} via the HPKE key schedule ({@code
   * info}) and {@code associatedData} via the AEAD ({@code aad}).
   *
   * @param contextInfo the HPKE {@code info}; {@code null} is treated as empty
   * @param associatedData the AEAD associated data; {@code null} is treated as empty
   */
  byte[] encrypt(byte[] plaintext, @Nullable byte[] contextInfo, @Nullable byte[] associatedData)
      throws GeneralSecurityException;
}
