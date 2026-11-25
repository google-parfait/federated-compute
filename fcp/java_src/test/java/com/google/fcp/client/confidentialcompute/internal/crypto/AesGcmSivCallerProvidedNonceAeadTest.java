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

import static com.google.common.truth.Truth.assertThat;
import static com.google.fcp.client.confidentialcompute.internal.crypto.AesGcmSivRfcTestVectors.aes128GcmSivVectors;
import static com.google.fcp.client.confidentialcompute.internal.crypto.AesGcmSivRfcTestVectors.hex;
import static com.google.fcp.client.confidentialcompute.internal.crypto.AesGcmSivRfcTestVectors.nonEdgeCaseVector;
import static com.google.fcp.client.confidentialcompute.internal.crypto.AesGcmSivRfcTestVectors.vectorWithEmptyAad;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import android.os.Build;
import com.google.common.primitives.Bytes;
import com.google.crypto.tink.Aead;
import com.google.crypto.tink.InsecureSecretKeyAccess;
import com.google.crypto.tink.aead.AesGcmSivKey;
import com.google.crypto.tink.aead.AesGcmSivParameters;
import com.google.crypto.tink.aead.subtle.AesGcmSivAead;
import com.google.crypto.tink.util.SecretBytes;
import com.google.fcp.client.confidentialcompute.internal.crypto.AesGcmSivRfcTestVectors.TestVector;
import java.security.AlgorithmParameters;
import java.security.GeneralSecurityException;
import java.security.Key;
import java.security.Provider;
import java.security.SecureRandom;
import java.security.Security;
import java.security.spec.AlgorithmParameterSpec;
import java.util.Arrays;
import javax.crypto.Cipher;
import javax.crypto.CipherSpi;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

/**
 * Tests for {@link AesGcmSivCallerProvidedNonceAead}.
 *
 * <p>Runs under Robolectric, which installs Conscrypt as the first security provider like Android
 * does; {@link AesGcmSivCallerProvidedNonceAeadWithoutConscryptTest} covers the environment without
 * it.
 */
@RunWith(RobolectricTestRunner.class)
public final class AesGcmSivCallerProvidedNonceAeadTest {

  private static final SecureRandom random = new SecureRandom();

  @Before
  public void environmentHasConscryptWithAesGcmSiv() throws Exception {
    // The primitive needs a provider that really implements AES-GCM-SIV. On Android that is the
    // platform Conscrypt ("AndroidOpenSSL") from API level 30 on; here it is the Conscrypt that
    // Robolectric installs ("Conscrypt"). Fail loudly if the environment is not like that, rather
    // than silently testing something else.
    Cipher cipher = Cipher.getInstance("AES/GCM-SIV/NoPadding");
    assertThat(cipher.getProvider().getName()).isAnyOf("Conscrypt", "AndroidOpenSSL");
    // ... and that this Conscrypt is recent enough to implement GCM-SIV rather than falling back to
    // AES-GCM: the known-answer test the primitive itself runs on Android 10 and older.
    AesGcmSivCallerProvidedNonceAead.checkIsAesGcmSivCipher(cipher);
  }

  private static byte[] randomBytes(int size) {
    byte[] bytes = new byte[size];
    random.nextBytes(bytes);
    return bytes;
  }

  private static AesGcmSivKey noPrefixKey(byte[] keyBytes) throws GeneralSecurityException {
    return AesGcmSivKey.builder()
        .setParameters(
            AesGcmSivParameters.builder()
                .setKeySizeBytes(keyBytes.length)
                .setVariant(AesGcmSivParameters.Variant.NO_PREFIX)
                .build())
        .setKeyBytes(SecretBytes.copyFrom(keyBytes, InsecureSecretKeyAccess.get()))
        .build();
  }

  @Test
  public void encryptDecrypt_rfcTestVectors_works() throws Exception {
    for (TestVector v : aes128GcmSivVectors) {
      CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(v.key));

      // NO_PREFIX key: the output is exactly ciphertext || tag, nothing else.
      assertThat(aead.encrypt(v.nonce, v.plaintext, v.aad)).isEqualTo(v.ciphertextAndTag());
      assertThat(aead.decrypt(v.nonce, v.ciphertextAndTag(), v.aad)).isEqualTo(v.plaintext);
    }
  }

  @Test
  public void encryptDecrypt_works() throws Exception {
    CallerProvidedNonceAead aead =
        AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(randomBytes(16)));
    byte[] nonce = hex("74df8fd4be34af647f5e54f6");
    byte[] plaintext = "hello confidential compute".getBytes(UTF_8);
    byte[] aad = "associated data".getBytes(UTF_8);

    byte[] ciphertext = aead.encrypt(nonce, plaintext, aad);

    assertThat(ciphertext).hasLength(plaintext.length + 16);
    assertThat(aead.decrypt(nonce, ciphertext, aad)).isEqualTo(plaintext);
  }

  @Test
  public void encryptDecrypt_longMessage_works() throws Exception {
    // Like Tink's AesGcmSivTest.testLongMessages: Android's BouncyCastle once broke GCM for
    // messages longer than 8192 bytes, so use one well beyond that.
    CallerProvidedNonceAead aead =
        AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(randomBytes(16)));
    byte[] nonce = randomBytes(12);
    byte[] plaintext = randomBytes(1 << 14);
    byte[] aad = randomBytes(1 << 10);

    byte[] ciphertext = aead.encrypt(nonce, plaintext, aad);

    assertThat(ciphertext).hasLength(plaintext.length + 16);
    assertThat(aead.decrypt(nonce, ciphertext, aad)).isEqualTo(plaintext);
  }

  @Test
  public void encryptDecrypt_nullAssociatedData_isTreatedAsEmpty() throws Exception {
    TestVector vector = vectorWithEmptyAad();
    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(vector.key));

    assertThat(aead.encrypt(vector.nonce, vector.plaintext, null))
        .isEqualTo(vector.ciphertextAndTag());
    assertThat(aead.decrypt(vector.nonce, vector.ciphertextAndTag(), null))
        .isEqualTo(vector.plaintext);
  }

  @Test
  public void encrypt_nullNonceOrPlaintext_throwsNullPointerException() throws Exception {
    TestVector vector = nonEdgeCaseVector();
    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(vector.key));

    assertThrows(
        NullPointerException.class, () -> aead.encrypt(null, vector.plaintext, vector.aad));
    assertThrows(NullPointerException.class, () -> aead.encrypt(vector.nonce, null, vector.aad));
    assertThrows(NullPointerException.class, () -> aead.encrypt(vector.nonce, null, null));
  }

  @Test
  public void decrypt_nullNonceOrCiphertext_throwsNullPointerException() throws Exception {
    TestVector vector = nonEdgeCaseVector();
    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(vector.key));
    byte[] ciphertext = vector.ciphertextAndTag();

    assertThrows(NullPointerException.class, () -> aead.decrypt(null, ciphertext, vector.aad));
    assertThrows(NullPointerException.class, () -> aead.decrypt(vector.nonce, null, vector.aad));
    assertThrows(NullPointerException.class, () -> aead.decrypt(vector.nonce, null, null));
  }

  @Test
  public void decrypt_wrongNonce_fails() throws Exception {
    TestVector vector = nonEdgeCaseVector();
    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(vector.key));
    byte[] ciphertext = vector.ciphertextAndTag();
    byte[] otherNonce = vector.nonce.clone();
    otherNonce[0] ^= 1;

    assertThrows(
        GeneralSecurityException.class, () -> aead.decrypt(otherNonce, ciphertext, vector.aad));
  }

  @Test
  public void decrypt_wrongAssociatedData_fails() throws Exception {
    TestVector vector = nonEdgeCaseVector();
    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(vector.key));
    byte[] ciphertext = vector.ciphertextAndTag();
    byte[] flippedAad = vector.aad.clone();
    flippedAad[0] ^= 1;
    byte[] longerAad = Arrays.copyOf(vector.aad, vector.aad.length + 1);
    byte[] shorterAad = Arrays.copyOf(vector.aad, vector.aad.length - 1);

    assertThrows(
        GeneralSecurityException.class, () -> aead.decrypt(vector.nonce, ciphertext, flippedAad));
    assertThrows(
        GeneralSecurityException.class, () -> aead.decrypt(vector.nonce, ciphertext, longerAad));
    assertThrows(
        GeneralSecurityException.class, () -> aead.decrypt(vector.nonce, ciphertext, shorterAad));
    assertThrows(
        GeneralSecurityException.class, () -> aead.decrypt(vector.nonce, ciphertext, new byte[0]));
  }

  @Test
  public void decrypt_modifiedCiphertext_fails() throws Exception {
    // One example of each corruption (a flipped bit, a missing byte, an extra byte), once in the
    // ciphertext part and once in the tag part of ciphertext || tag.
    TestVector vector = nonEdgeCaseVector();
    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(vector.key));
    byte[] ciphertext = vector.ciphertextAndTag();
    int tagStart = vector.ciphertext.length;

    byte[] flippedCiphertextBit = ciphertext.clone();
    flippedCiphertextBit[0] ^= 1;
    byte[] flippedTagBit = ciphertext.clone();
    flippedTagBit[tagStart] ^= 1;
    byte[] missingCiphertextByte = Arrays.copyOfRange(ciphertext, 1, ciphertext.length);
    byte[] missingTagByte = Arrays.copyOf(ciphertext, ciphertext.length - 1);
    byte[] extraCiphertextByte = Bytes.concat(new byte[1], ciphertext);
    byte[] extraTagByte = Arrays.copyOf(ciphertext, ciphertext.length + 1);

    for (byte[] modified :
        new byte[][] {
          flippedCiphertextBit,
          flippedTagBit,
          missingCiphertextByte,
          missingTagByte,
          extraCiphertextByte,
          extraTagByte
        }) {
      assertThrows(
          GeneralSecurityException.class, () -> aead.decrypt(vector.nonce, modified, vector.aad));
    }
  }

  @Test
  public void encrypt_invalidNonceLength_fails() throws Exception {
    TestVector vector = nonEdgeCaseVector();
    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(vector.key));

    GeneralSecurityException shortNonce =
        assertThrows(
            GeneralSecurityException.class,
            () -> aead.encrypt(new byte[11], vector.plaintext, vector.aad));
    assertThat(shortNonce).hasMessageThat().contains("nonce must have 12 bytes");
    GeneralSecurityException longNonce =
        assertThrows(
            GeneralSecurityException.class,
            () -> aead.encrypt(new byte[16], vector.plaintext, vector.aad));
    assertThat(longNonce).hasMessageThat().contains("nonce must have 12 bytes");
  }

  @Test
  public void decrypt_invalidNonceLength_fails() throws Exception {
    TestVector vector = nonEdgeCaseVector();
    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(vector.key));
    byte[] ciphertext = vector.ciphertextAndTag();

    GeneralSecurityException shortNonce =
        assertThrows(
            GeneralSecurityException.class,
            () -> aead.decrypt(new byte[11], ciphertext, vector.aad));
    assertThat(shortNonce).hasMessageThat().contains("nonce must have 12 bytes");
    GeneralSecurityException longNonce =
        assertThrows(
            GeneralSecurityException.class,
            () -> aead.decrypt(new byte[16], ciphertext, vector.aad));
    assertThat(longNonce).hasMessageThat().contains("nonce must have 12 bytes");
  }

  @Test
  public void create_rejectsKeysWithOutputPrefix() throws Exception {
    for (AesGcmSivParameters.Variant variant :
        Arrays.asList(AesGcmSivParameters.Variant.TINK, AesGcmSivParameters.Variant.CRUNCHY)) {
      AesGcmSivKey key =
          AesGcmSivKey.builder()
              .setParameters(
                  AesGcmSivParameters.builder().setKeySizeBytes(16).setVariant(variant).build())
              .setKeyBytes(SecretBytes.copyFrom(randomBytes(16), InsecureSecretKeyAccess.get()))
              .setIdRequirement(0x01020304)
              .build();

      assertThrows(
          GeneralSecurityException.class, () -> AesGcmSivCallerProvidedNonceAead.create(key));
    }
  }

  @Test
  public void create_rejectsAes256GcmSivKeys() throws Exception {
    // Tink's AesGcmSivParameters also allow 32-byte keys, but crypto.cc only uses AES-128-GCM-SIV.
    AesGcmSivKey aes256Key = noPrefixKey(randomBytes(32));

    GeneralSecurityException e =
        assertThrows(
            GeneralSecurityException.class,
            () -> AesGcmSivCallerProvidedNonceAead.create(aes256Key));
    assertThat(e).hasMessageThat().contains("16-byte");
  }

  @Test
  public void checkIsAesGcmSivCipher_rejectsPlainAesGcmCipher() throws Exception {
    // What the BouncyCastle of Android 10 and older hands out for "AES/GCM-SIV/NoPadding": AES-GCM.
    Cipher gcmCipher = Cipher.getInstance("AES/GCM/NoPadding");

    assertThrows(
        GeneralSecurityException.class,
        () -> AesGcmSivCallerProvidedNonceAead.checkIsAesGcmSivCipher(gcmCipher));
  }

  @Test
  public void encrypt_cipherReturningWrongTagSize_fails() throws Exception {
    TestVector vector = nonEdgeCaseVector();
    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(noPrefixKey(vector.key));
    Provider brokenProvider = newBrokenAesGcmSivProvider();
    Security.insertProviderAt(brokenProvider, 1);
    try {
      GeneralSecurityException e =
          assertThrows(
              GeneralSecurityException.class,
              () -> aead.encrypt(vector.nonce, vector.plaintext, vector.aad));
      assertThat(e)
          .hasMessageThat()
          .contains("AES-GCM-SIV tag must be 16 bytes, but got only 0 bytes");
    } finally {
      Security.removeProvider(brokenProvider.getName());
    }
  }

  @Test
  @Config(sdk = Build.VERSION_CODES.Q)
  public void beforeAndroid11_runsKatOnCreateEncryptAndDecrypt() throws Exception {
    TestVector vector = nonEdgeCaseVector();
    AesGcmSivKey key = noPrefixKey(vector.key);
    byte[] ciphertext = vector.ciphertextAndTag();
    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(key);
    assertThat(aead.encrypt(vector.nonce, vector.plaintext, vector.aad)).isEqualTo(ciphertext);
    assertThat(aead.decrypt(vector.nonce, ciphertext, vector.aad)).isEqualTo(vector.plaintext);

    Provider brokenProvider = newBrokenAesGcmSivProvider();
    Security.insertProviderAt(brokenProvider, 1);
    try {
      GeneralSecurityException createException =
          assertThrows(
              GeneralSecurityException.class, () -> AesGcmSivCallerProvidedNonceAead.create(key));
      assertThat(createException).hasMessageThat().contains("known-answer test");
      GeneralSecurityException encryptException =
          assertThrows(
              GeneralSecurityException.class,
              () -> aead.encrypt(vector.nonce, vector.plaintext, vector.aad));
      assertThat(encryptException).hasMessageThat().contains("known-answer test");
      GeneralSecurityException decryptException =
          assertThrows(
              GeneralSecurityException.class,
              () -> aead.decrypt(vector.nonce, ciphertext, vector.aad));
      assertThat(decryptException).hasMessageThat().contains("known-answer test");
    } finally {
      Security.removeProvider(brokenProvider.getName());
    }
  }

  @Test
  public void interopWithTinkAesGcmSivAead() throws Exception {
    // Tink's Aead ciphertext format for a NO_PREFIX key is nonce || ciphertext || tag, with a
    // random nonce. The same key therefore interoperates with this primitive by moving the nonce.
    AesGcmSivKey key = noPrefixKey(nonEdgeCaseVector().key);
    Aead tinkAead = AesGcmSivAead.create(key);
    CallerProvidedNonceAead callerProvidedNonceAead = AesGcmSivCallerProvidedNonceAead.create(key);
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] aad = "aad".getBytes(UTF_8);

    // Tink -> this primitive.
    byte[] tinkCiphertext = tinkAead.encrypt(plaintext, aad);
    byte[] nonce = Arrays.copyOf(tinkCiphertext, 12);
    byte[] rest = Arrays.copyOfRange(tinkCiphertext, 12, tinkCiphertext.length);
    assertThat(callerProvidedNonceAead.decrypt(nonce, rest, aad)).isEqualTo(plaintext);

    // This primitive -> Tink.
    byte[] ourNonce = hex("74df8fd4be34af647f5e54f6");
    byte[] ourCiphertext = callerProvidedNonceAead.encrypt(ourNonce, plaintext, aad);
    assertThat(tinkAead.decrypt(Bytes.concat(ourNonce, ourCiphertext), aad)).isEqualTo(plaintext);
  }

  private static Provider newBrokenAesGcmSivProvider() {
    Provider provider = new Provider("BrokenAesGcmSivProvider", "1.0", "Test provider") {};
    provider.put("Cipher.AES/GCM-SIV/NoPadding", ShortTagCipherSpi.class.getName());
    return provider;
  }

  /**
   * A fake {@link CipherSpi} whose {@code doFinal} returns the input unchanged (i.e. with a 0-byte
   * tag instead of a 16-byte tag, and without decrypting the RFC 8452 KAT vector).
   */
  public static final class ShortTagCipherSpi extends CipherSpi {
    public ShortTagCipherSpi() {}

    @Override
    protected void engineSetMode(String mode) {}

    @Override
    protected void engineSetPadding(String padding) {}

    @Override
    protected int engineGetBlockSize() {
      return 16;
    }

    @Override
    protected int engineGetOutputSize(int inputLen) {
      return inputLen;
    }

    @Override
    protected byte[] engineGetIV() {
      return new byte[12];
    }

    @Override
    protected AlgorithmParameters engineGetParameters() {
      return null;
    }

    @Override
    protected void engineInit(int opmode, Key key, SecureRandom random) {}

    @Override
    protected void engineInit(
        int opmode, Key key, AlgorithmParameterSpec params, SecureRandom random) {}

    @Override
    protected void engineInit(
        int opmode, Key key, AlgorithmParameters params, SecureRandom random) {}

    @Override
    protected void engineUpdateAAD(byte[] src, int offset, int len) {}

    @Override
    protected byte[] engineUpdate(byte[] input, int inputOffset, int inputLen) {
      return new byte[0];
    }

    @Override
    protected int engineUpdate(
        byte[] input, int inputOffset, int inputLen, byte[] output, int outputOffset) {
      return 0;
    }

    @Override
    protected byte[] engineDoFinal(byte[] input, int inputOffset, int inputLen) {
      return Arrays.copyOfRange(input, inputOffset, inputOffset + inputLen);
    }

    @Override
    protected int engineDoFinal(
        byte[] input, int inputOffset, int inputLen, byte[] output, int outputOffset) {
      return 0;
    }
  }
}
