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

import android.os.Build;
import com.google.common.io.BaseEncoding;
import com.google.crypto.tink.AccessesPartialKey;
import com.google.crypto.tink.InsecureSecretKeyAccess;
import com.google.crypto.tink.aead.AesGcmSivKey;
import com.google.crypto.tink.aead.AesGcmSivParameters;
import java.security.GeneralSecurityException;
import java.util.Arrays;
import javax.annotation.Nullable;
import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;

/**
 * AES-128-GCM-SIV (RFC 8452) with a caller-provided 12-byte nonce, backed by a Tink {@link
 * AesGcmSivKey}.
 *
 * <p>Only 16-byte keys are supported: {@code fcp/confidentialcompute/crypto.cc} uses
 * AES-128-GCM-SIV only (COSE algorithm {@code AEAD_AES_128_GCM_SIV}), so {@link #create} rejects
 * the 32-byte keys that {@link AesGcmSivParameters} would also allow.
 *
 * <p>Only {@code NO_PREFIX} keys are supported: {@link #encrypt} returns exactly {@code ciphertext
 * || tag}, the format of {@code fcp/confidentialcompute/crypto.cc}, and {@link #decrypt} expects
 * the same. {@link #create} rejects keys of the {@code TINK} and {@code CRUNCHY} variants.
 *
 * <p>Requires a JCE provider that implements {@code AES/GCM-SIV/NoPadding}:
 *
 * <ul>
 *   <li>On Android API level 30 and above, the platform Conscrypt provider ({@code AndroidOpenSSL})
 *       implements it.
 *   <li>On Android API level 29 and below, the Conscrypt that shipped with the platform does not.
 *       An implementation is then only present if the app has installed Google Play services'
 *       Conscrypt ({@code ProviderInstaller.installIfNeeded()}) or bundles its own {@code
 *       org.conscrypt} provider. API level 29 is a special case: Conscrypt became an updatable
 *       Mainline module in Android 10, and while the Android 10 version of the module lacks
 *       GCM-SIV, the Android 11 and later versions of the module can be installed on Android 10
 *       devices and do implement it. Whether a given API level 29 device has GCM-SIV therefore
 *       depends on the module version it received.
 *   <li>On the JVM, the Conscrypt provider implements it; the JDK's SunJCE does not.
 * </ul>
 *
 * <p>When no provider implements it, {@code Cipher.getInstance("AES/GCM-SIV/NoPadding")} does
 * <em>not</em> fail on Android 10 (API level 29) and older: the lookup falls through to the
 * platform's BouncyCastle provider, which until Android 11 treated every mode name starting with
 * "GCM" as GCM, and so hands out a plain AES-GCM cipher. On those Android versions ({@code
 * Build.VERSION.SDK_INT < Build.VERSION_CODES.R}), like Tink's {@code AesGcmSiv}, this class
 * detects that with a known-answer test (RFC 8452) and refuses to run. Because {@code
 * Cipher.getInstance} consults the provider list anew on every call, on those versions the test is
 * run not only when the primitive is created but also on the very {@code Cipher} instance used by
 * each {@link #encrypt} and {@link #decrypt} call.
 */
public final class AesGcmSivCallerProvidedNonceAead implements CallerProvidedNonceAead {

  private static final String CIPHER_TRANSFORMATION = "AES/GCM-SIV/NoPadding";
  private static final int KEY_SIZE_IN_BYTES = 16;
  private static final int NONCE_SIZE_IN_BYTES = 12;
  private static final int TAG_SIZE_IN_BYTES = 16;

  // Test vector from https://www.rfc-editor.org/rfc/rfc8452.html#appendix-C.1, used as a
  // known-answer test to make sure the provider really implements GCM-SIV rather than GCM.
  private static final BaseEncoding HEX = BaseEncoding.base16().lowerCase();
  private static final byte[] testPlaintext = HEX.decode("7a806c");
  private static final byte[] testAad = HEX.decode("46bb91c3c5");
  private static final byte[] testKey = HEX.decode("36864200e0eaf5284d884a0e77d31646");
  private static final byte[] testNonce = HEX.decode("bae8e37fc83441b16034566b");
  private static final byte[] testResult = HEX.decode("af60eb711bd85bc1e4d3e0a462e074eea428a8");

  private final SecretKeySpec keySpec;

  /**
   * Creates a {@link CallerProvidedNonceAead} for a single {@code NO_PREFIX} {@link AesGcmSivKey}.
   *
   * <p>Most code should not call this directly but obtain the primitive from a keyset: {@code
   * keysetHandle.getPrimitive(FcpTinkConfiguration.get(), CallerProvidedNonceAead.class)}.
   */
  @AccessesPartialKey
  public static CallerProvidedNonceAead create(AesGcmSivKey key) throws GeneralSecurityException {
    AesGcmSivParameters.Variant variant = key.getParameters().getVariant();
    if (!variant.equals(AesGcmSivParameters.Variant.NO_PREFIX)) {
      throw new GeneralSecurityException(
          "only NO_PREFIX AesGcmSivKeys are supported, got variant " + variant);
    }
    // Fail fast if no provider implements AES/GCM-SIV/NoPadding, or (on Android 10 and older) if
    // the provider hands out a plain AES-GCM cipher; encrypt and decrypt repeat this check on the
    // Cipher instance they actually use.
    Cipher cipher = Cipher.getInstance(CIPHER_TRANSFORMATION);
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
      checkIsAesGcmSivCipher(cipher);
    }
    byte[] keyBytes = key.getKeyBytes().toByteArray(InsecureSecretKeyAccess.get());
    if (keyBytes.length != KEY_SIZE_IN_BYTES) {
      throw new GeneralSecurityException(
          "only 16-byte (AES-128-GCM-SIV) keys are supported, got " + keyBytes.length + " bytes");
    }
    return new AesGcmSivCallerProvidedNonceAead(new SecretKeySpec(keyBytes, "AES"));
  }

  private AesGcmSivCallerProvidedNonceAead(SecretKeySpec keySpec) {
    this.keySpec = keySpec;
  }

  @Override
  public byte[] encrypt(byte[] nonce, byte[] plaintext, @Nullable byte[] associatedData)
      throws GeneralSecurityException {
    checkNotNull(nonce);
    checkNotNull(plaintext);
    validateNonce(nonce);
    // Unlike in Tink's AesGcmSiv, the ciphertext contains neither the nonce (the caller provides
    // it) nor an output prefix (NO_PREFIX keys only), so it is exactly TAG_SIZE_IN_BYTES longer
    // than the plaintext.
    if (plaintext.length > Integer.MAX_VALUE - TAG_SIZE_IN_BYTES) {
      throw new GeneralSecurityException("plaintext too long");
    }
    Cipher cipher = Cipher.getInstance(CIPHER_TRANSFORMATION);
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
      checkIsAesGcmSivCipher(cipher);
    }
    cipher.init(Cipher.ENCRYPT_MODE, keySpec, new GCMParameterSpec(8 * TAG_SIZE_IN_BYTES, nonce));
    if (associatedData != null && associatedData.length != 0) {
      cipher.updateAAD(associatedData);
    }
    byte[] ciphertext = cipher.doFinal(plaintext);
    // AES-GCM-SIV always adds a tag of length TAG_SIZE_IN_BYTES.
    if (ciphertext.length != plaintext.length + TAG_SIZE_IN_BYTES) {
      int actualTagSize = ciphertext.length - plaintext.length;
      throw new GeneralSecurityException(
          String.format(
              "encryption failed; AES-GCM-SIV tag must be %s bytes, but got only %s bytes",
              TAG_SIZE_IN_BYTES, actualTagSize));
    }
    return ciphertext;
  }

  @Override
  public byte[] decrypt(byte[] nonce, byte[] ciphertext, @Nullable byte[] associatedData)
      throws GeneralSecurityException {
    checkNotNull(nonce);
    checkNotNull(ciphertext);
    validateNonce(nonce);
    // See encrypt: the ciphertext contains neither the nonce nor an output prefix, only the tag.
    if (ciphertext.length < TAG_SIZE_IN_BYTES) {
      throw new GeneralSecurityException("ciphertext too short");
    }
    Cipher cipher = Cipher.getInstance(CIPHER_TRANSFORMATION);
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
      checkIsAesGcmSivCipher(cipher);
    }
    cipher.init(Cipher.DECRYPT_MODE, keySpec, new GCMParameterSpec(8 * TAG_SIZE_IN_BYTES, nonce));
    if (associatedData != null && associatedData.length != 0) {
      cipher.updateAAD(associatedData);
    }
    return cipher.doFinal(ciphertext);
  }

  private static void validateNonce(byte[] nonce) throws GeneralSecurityException {
    if (nonce.length != NONCE_SIZE_IN_BYTES) {
      throw new GeneralSecurityException("nonce must have 12 bytes, got " + nonce.length);
    }
  }

  /**
   * Runs the RFC 8452 known-answer test on {@code cipher} and throws unless it really implements
   * AES-GCM-SIV. The cipher is left initialized with the test key; callers must {@code init} it
   * again before use.
   */
  static void checkIsAesGcmSivCipher(Cipher cipher) throws GeneralSecurityException {
    byte[] decrypted;
    try {
      cipher.init(
          Cipher.DECRYPT_MODE,
          new SecretKeySpec(testKey, "AES"),
          new GCMParameterSpec(8 * TAG_SIZE_IN_BYTES, testNonce));
      cipher.updateAAD(testAad);
      decrypted = cipher.doFinal(testResult);
    } catch (GeneralSecurityException e) {
      throw new GeneralSecurityException(
          "AES/GCM-SIV/NoPadding cipher failed the RFC 8452 known-answer test", e);
    }
    // Public test vector: a plain comparison is fine, there is nothing to hide from timing.
    if (!Arrays.equals(decrypted, testPlaintext)) {
      throw new GeneralSecurityException(
          "AES/GCM-SIV/NoPadding cipher failed the RFC 8452 known-answer test");
    }
  }
}
