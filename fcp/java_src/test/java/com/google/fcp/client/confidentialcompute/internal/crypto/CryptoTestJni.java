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
import java.util.Arrays;

/**
 * Test-only JNI bindings for {@code fcp/confidentialcompute/crypto.cc} decryption functions ({@code
 * crypto_internal::UnwrapSymmetricKey} and {@code MessageDecryptor::Decrypt}).
 */
final class CryptoTestJni {

  private static final int X25519_ENCAPSULATED_KEY_SIZE = 32;

  static {
    System.loadLibrary("crypto_test_jni");
  }

  /** Returns whether the native library was compiled with CBOR/COSE support enabled. */
  static boolean isCborSupported() {
    return nativeIsCborSupported();
  }

  /**
   * Splits {@code encapsulatedKeyAndCiphertext} into the 32-byte X25519 {@code enc} prefix and the
   * HPKE ciphertext, and unwraps it with {@code crypto_internal::UnwrapSymmetricKey} from {@code
   * crypto.cc} (which uses an empty HPKE {@code info}).
   */
  static byte[] unwrapSymmetricKey(
      byte[] recipientPrivateKey, byte[] encapsulatedKeyAndCiphertext, byte[] associatedData)
      throws GeneralSecurityException {
    if (encapsulatedKeyAndCiphertext.length < X25519_ENCAPSULATED_KEY_SIZE) {
      throw new GeneralSecurityException("ciphertext is shorter than encapsulated key");
    }
    byte[] encappedKey = Arrays.copyOf(encapsulatedKeyAndCiphertext, X25519_ENCAPSULATED_KEY_SIZE);
    byte[] encryptedSymmetricKey =
        Arrays.copyOfRange(
            encapsulatedKeyAndCiphertext,
            X25519_ENCAPSULATED_KEY_SIZE,
            encapsulatedKeyAndCiphertext.length);
    return nativeUnwrapSymmetricKey(
        recipientPrivateKey, encappedKey, encryptedSymmetricKey, associatedData);
  }

  /**
   * Decrypts a full confidential-compute payload using {@code MessageDecryptor::Decrypt} from
   * {@code crypto.cc}.
   */
  static byte[] decryptMessage(
      byte[] recipientPrivateKey,
      byte[] keyId,
      byte[] ciphertext,
      byte[] ciphertextAssociatedData,
      byte[] encryptedSymmetricKey,
      byte[] encryptedSymmetricKeyAssociatedData,
      byte[] encappedKey)
      throws GeneralSecurityException {
    return nativeDecryptMessage(
        recipientPrivateKey,
        keyId,
        ciphertext,
        ciphertextAssociatedData,
        encryptedSymmetricKey,
        encryptedSymmetricKeyAssociatedData,
        encappedKey);
  }

  private static native boolean nativeIsCborSupported();

  private static native byte[] nativeUnwrapSymmetricKey(
      byte[] recipientPrivateKey,
      byte[] encappedKey,
      byte[] encryptedSymmetricKey,
      byte[] associatedData)
      throws GeneralSecurityException;

  private static native byte[] nativeDecryptMessage(
      byte[] recipientPrivateKey,
      byte[] keyId,
      byte[] ciphertext,
      byte[] ciphertextAssociatedData,
      byte[] encryptedSymmetricKey,
      byte[] encryptedSymmetricKeyAssociatedData,
      byte[] encappedKey)
      throws GeneralSecurityException;

  private CryptoTestJni() {}
}
