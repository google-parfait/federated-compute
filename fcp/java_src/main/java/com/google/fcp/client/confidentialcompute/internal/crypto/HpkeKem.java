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

import com.google.crypto.tink.hybrid.HpkePrivateKey;
import com.google.errorprone.annotations.Immutable;
import java.security.GeneralSecurityException;

/**
 * Interface for Hybrid Public Key Encryption (HPKE) key encapsulation mechanism (KEM).
 *
 * <p>HPKE RFC is available at https://www.rfc-editor.org/rfc/rfc9180.html.
 *
 * <p>Branched from Tink's {@code com.google.crypto.tink.hybrid.internal.HpkeKem}, without the
 * authenticated ({@code mode_auth}) operations, which this package does not use.
 */
@Immutable
interface HpkeKem {
  /**
   * Generates and encapsulates a shared secret using the {@code recipientPublicKey}. Returns a
   * {@link HpkeKemEncapOutput} object that contains the raw shared secret and the encapsulated key.
   * The HPKE RFC refers to this method as Encap(), which is used by the sender.
   *
   * @throws GeneralSecurityException when either the shared secret cannot be generated or the
   *     shared secret cannot be encapsulated.
   */
  HpkeKemEncapOutput encapsulate(byte[] recipientPublicKey) throws GeneralSecurityException;

  /**
   * Extracts the shared secret from {@code encapsulatedKey} using {@code recipientPrivateKey}.
   * Returns the raw shared secret. The HPKE RFC refers to this method as Decap(), which is used by
   * the recipient.
   *
   * @throws GeneralSecurityException if the shared secret cannot be extracted.
   */
  byte[] decapsulate(byte[] encapsulatedKey, HpkePrivateKey recipientPrivateKey)
      throws GeneralSecurityException;

  /**
   * Returns the HPKE KEM algorithm identifier for the underlying KEM implementation.
   *
   * <p>More details at
   * https://www.rfc-editor.org/rfc/rfc9180.html#name-key-encapsulation-mechanism.
   */
  byte[] getKemId() throws GeneralSecurityException;
}
