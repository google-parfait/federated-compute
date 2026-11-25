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
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.hex;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

import com.google.crypto.tink.HybridDecrypt;
import com.google.crypto.tink.InsecureSecretKeyAccess;
import com.google.crypto.tink.aead.AesGcmSivKey;
import com.google.crypto.tink.aead.AesGcmSivParameters;
import com.google.crypto.tink.hybrid.HpkeParameters;
import com.google.crypto.tink.hybrid.HpkePrivateKey;
import com.google.crypto.tink.hybrid.HpkePublicKey;
import com.google.crypto.tink.hybrid.subtle.HpkeDecrypt;
import com.google.crypto.tink.subtle.X25519;
import com.google.crypto.tink.util.Bytes;
import com.google.crypto.tink.util.SecretBytes;
import java.security.GeneralSecurityException;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

@RunWith(RobolectricTestRunner.class)
public final class HpkeWithAssociatedDataTest {

  // Constants of the confidential compute protocol, copied from fcp/confidentialcompute/crypto.cc.
  private static final byte[] fixedNonce = {
    0x74,
    (byte) 0xDF,
    (byte) 0x8F,
    (byte) 0xD4,
    (byte) 0xBE,
    0x34,
    (byte) 0xAF,
    0x64,
    0x7F,
    0x5E,
    0x54,
    (byte) 0xF6
  };
  // COSE_Key header for a 16-byte symmetric AES-128-GCM-SIV key; the raw key follows it.
  private static final byte[] cborKeyPrefix = {
    (byte) 0xa4, 0x01, 0x04, 0x03, 0x3a, 0x00, 0x01, 0x00, 0x01, 0x04, (byte) 0x81, 0x04, 0x20, 0x50
  };
  private static final int ENCAPSULATED_KEY_SIZE = 32;

  private static HpkeParameters parameters(HpkeParameters.AeadId aeadId, HpkeParameters.Variant v)
      throws GeneralSecurityException {
    return HpkeParameters.builder()
        .setKemId(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256)
        .setKdfId(HpkeParameters.KdfId.HKDF_SHA256)
        .setAeadId(aeadId)
        .setVariant(v)
        .build();
  }

  private static HpkePrivateKey privateKey(
      HpkeParameters parameters, byte[] privateKeyBytes, Integer idRequirement)
      throws GeneralSecurityException {
    HpkePublicKey publicKey =
        HpkePublicKey.create(
            parameters, Bytes.copyFrom(X25519.publicFromPrivate(privateKeyBytes)), idRequirement);
    return HpkePrivateKey.create(
        publicKey, SecretBytes.copyFrom(privateKeyBytes, InsecureSecretKeyAccess.get()));
  }

  @Test
  public void encrypt_decryptsWithNativeUnwrapSymmetricKey() throws Exception {
    byte[] privateKeyBytes = X25519.generatePrivateKey();
    HpkePrivateKey privateKey =
        privateKey(
            parameters(HpkeParameters.AeadId.AES_128_GCM, HpkeParameters.Variant.NO_PREFIX),
            privateKeyBytes,
            /* idRequirement= */ null);
    HybridEncryptWithAssociatedData encrypt =
        HpkeEncryptWithAssociatedData.create(privateKey.getPublicKey());
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] emptyInfo = new byte[0];
    byte[] aad = "aad".getBytes(UTF_8);

    byte[] ciphertext = encrypt.encrypt(plaintext, emptyInfo, aad);

    // Exactly enc || ct, no output prefix.
    assertThat(ciphertext).hasLength(ENCAPSULATED_KEY_SIZE + plaintext.length + 16);
    assertThat(CryptoTestJni.unwrapSymmetricKey(privateKeyBytes, ciphertext, aad))
        .isEqualTo(plaintext);
    // Both the HPKE info and the AEAD associated data are bound to the ciphertext.
    byte[] nonEmptyInfoCiphertext = encrypt.encrypt(plaintext, "other info".getBytes(UTF_8), aad);
    assertThrows(
        GeneralSecurityException.class,
        () -> CryptoTestJni.unwrapSymmetricKey(privateKeyBytes, nonEmptyInfoCiphertext, aad));
    assertThrows(
        GeneralSecurityException.class,
        () ->
            CryptoTestJni.unwrapSymmetricKey(
                privateKeyBytes, ciphertext, "other aad".getBytes(UTF_8)));
    assertThrows(
        GeneralSecurityException.class,
        () ->
            CryptoTestJni.unwrapSymmetricKey(
                privateKeyBytes, Arrays.copyOf(ciphertext, ciphertext.length - 1), aad));
  }

  @Test
  public void encrypt_isCompatibleWithNativeMessageDecryptor() throws Exception {
    assumeTrue("C++ CBOR/COSE support is disabled", CryptoTestJni.isCborSupported());
    byte[] recipientPrivateKeyBytes = X25519.generatePrivateKey();
    HpkePrivateKey recipientKey =
        privateKey(
            parameters(HpkeParameters.AeadId.AES_128_GCM, HpkeParameters.Variant.NO_PREFIX),
            recipientPrivateKeyBytes,
            /* idRequirement= */ null);
    HybridEncryptWithAssociatedData hpkeEncrypt =
        HpkeEncryptWithAssociatedData.create(recipientKey.getPublicKey());

    AesGcmSivParameters aesParameters =
        AesGcmSivParameters.builder()
            .setKeySizeBytes(16)
            .setVariant(AesGcmSivParameters.Variant.NO_PREFIX)
            .build();
    AesGcmSivKey symmetricKey =
        AesGcmSivKey.builder()
            .setParameters(aesParameters)
            .setKeyBytes(SecretBytes.randomBytes(16))
            .build();
    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(symmetricKey);

    byte[] plaintext = "confidential payload".getBytes(UTF_8);
    byte[] aad = "associated data".getBytes(UTF_8);
    byte[] keyId = "key-id".getBytes(UTF_8);

    byte[] cborSymmetricKey =
        com.google.common.primitives.Bytes.concat(
            cborKeyPrefix, symmetricKey.getKeyBytes().toByteArray(InsecureSecretKeyAccess.get()));
    byte[] hpkeOutput = hpkeEncrypt.encrypt(cborSymmetricKey, /* contextInfo= */ new byte[0], aad);
    byte[] encappedKey = Arrays.copyOf(hpkeOutput, ENCAPSULATED_KEY_SIZE);
    byte[] encryptedSymmetricKey =
        Arrays.copyOfRange(hpkeOutput, ENCAPSULATED_KEY_SIZE, hpkeOutput.length);
    byte[] ciphertext = aead.encrypt(fixedNonce, plaintext, aad);

    assertThat(
            CryptoTestJni.decryptMessage(
                recipientPrivateKeyBytes,
                keyId,
                ciphertext,
                aad,
                encryptedSymmetricKey,
                aad,
                encappedKey))
        .isEqualTo(plaintext);
    assertThrows(
        GeneralSecurityException.class,
        () ->
            CryptoTestJni.decryptMessage(
                recipientPrivateKeyBytes,
                keyId,
                ciphertext,
                aad,
                encryptedSymmetricKey,
                "wrong aad".getBytes(UTF_8),
                encappedKey));
  }

  @Test
  public void encrypt_producesDifferentCiphertextEveryTime() throws Exception {
    HpkePrivateKey privateKey =
        privateKey(
            parameters(HpkeParameters.AeadId.AES_128_GCM, HpkeParameters.Variant.NO_PREFIX),
            X25519.generatePrivateKey(),
            /* idRequirement= */ null);
    HybridEncryptWithAssociatedData encrypt =
        HpkeEncryptWithAssociatedData.create(privateKey.getPublicKey());
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] info = "info".getBytes(UTF_8);
    byte[] aad = "aad".getBytes(UTF_8);

    byte[] first = encrypt.encrypt(plaintext, info, aad);
    byte[] second = encrypt.encrypt(plaintext, info, aad);

    assertThat(first).isNotEqualTo(second);
  }

  @Test
  public void encrypt_nullContextInfoOrAssociatedData_isTreatedAsEmpty() throws Exception {
    byte[] privateKeyBytes = X25519.generatePrivateKey();
    HpkePrivateKey privateKey =
        privateKey(
            parameters(HpkeParameters.AeadId.AES_128_GCM, HpkeParameters.Variant.NO_PREFIX),
            privateKeyBytes,
            /* idRequirement= */ null);
    HybridEncryptWithAssociatedData encrypt =
        HpkeEncryptWithAssociatedData.create(privateKey.getPublicKey());
    HybridDecrypt tinkDecrypt = HpkeDecrypt.create(privateKey);
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] info = "info".getBytes(UTF_8);
    byte[] aad = "aad".getBytes(UTF_8);

    byte[] nullInfoCiphertext = encrypt.encrypt(plaintext, null, aad);
    assertThat(CryptoTestJni.unwrapSymmetricKey(privateKeyBytes, nullInfoCiphertext, aad))
        .isEqualTo(plaintext);

    byte[] nullAadCiphertext = encrypt.encrypt(plaintext, info, null);
    assertThat(tinkDecrypt.decrypt(nullAadCiphertext, info)).isEqualTo(plaintext);

    byte[] bothNullCiphertext = encrypt.encrypt(plaintext, null, null);
    assertThat(CryptoTestJni.unwrapSymmetricKey(privateKeyBytes, bothNullCiphertext, new byte[0]))
        .isEqualTo(plaintext);
    assertThat(tinkDecrypt.decrypt(bothNullCiphertext, new byte[0])).isEqualTo(plaintext);
  }

  @Test
  public void encrypt_nullPlaintext_throwsNullPointerException() throws Exception {
    HpkePrivateKey privateKey =
        privateKey(
            parameters(HpkeParameters.AeadId.AES_128_GCM, HpkeParameters.Variant.NO_PREFIX),
            X25519.generatePrivateKey(),
            /* idRequirement= */ null);
    HybridEncryptWithAssociatedData encrypt =
        HpkeEncryptWithAssociatedData.create(privateKey.getPublicKey());
    byte[] info = "info".getBytes(UTF_8);
    byte[] aad = "aad".getBytes(UTF_8);

    assertThrows(NullPointerException.class, () -> encrypt.encrypt(null, info, aad));
    assertThrows(NullPointerException.class, () -> encrypt.encrypt(null, null, null));
  }

  @Test
  public void create_rejectsKeysWithOutputPrefix() throws Exception {
    for (HpkeParameters.Variant variant :
        new HpkeParameters.Variant[] {
          HpkeParameters.Variant.TINK, HpkeParameters.Variant.CRUNCHY
        }) {
      HpkePrivateKey privateKey =
          privateKey(
              parameters(HpkeParameters.AeadId.AES_128_GCM, variant),
              X25519.generatePrivateKey(),
              /* idRequirement= */ 0x0a0b0c0d);
      HpkePublicKey publicKey = privateKey.getPublicKey();

      GeneralSecurityException encryptException =
          assertThrows(
              GeneralSecurityException.class,
              () -> HpkeEncryptWithAssociatedData.create(publicKey));
      assertThat(encryptException).hasMessageThat().contains("NO_PREFIX");
    }
  }

  @Test
  public void emptyAssociatedData_interopsWithTinkHpke() throws Exception {
    // Tink's HybridEncrypt/HybridDecrypt is HPKE with an empty aad. With an empty aad this
    // primitive must therefore be byte-compatible with Tink's HpkeDecrypt (for NO_PREFIX keys, the
    // only variant it accepts).
    HpkePrivateKey privateKey =
        privateKey(
            parameters(HpkeParameters.AeadId.AES_128_GCM, HpkeParameters.Variant.NO_PREFIX),
            X25519.generatePrivateKey(),
            /* idRequirement= */ null);
    HybridDecrypt tinkDecrypt = HpkeDecrypt.create(privateKey);
    HybridEncryptWithAssociatedData ourEncrypt =
        HpkeEncryptWithAssociatedData.create(privateKey.getPublicKey());
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] info = "context info".getBytes(UTF_8);

    assertThat(tinkDecrypt.decrypt(ourEncrypt.encrypt(plaintext, info, new byte[0]), info))
        .isEqualTo(plaintext);
    // ... and a non-empty aad is exactly what Tink cannot express.
    byte[] withAad = ourEncrypt.encrypt(plaintext, info, "aad".getBytes(UTF_8));
    assertThrows(GeneralSecurityException.class, () -> tinkDecrypt.decrypt(withAad, info));
  }

  @Test
  public void unsupportedCiphersuite_isRejectedAtCreation() throws Exception {
    HpkeParameters p256 =
        HpkeParameters.builder()
            .setKemId(HpkeParameters.KemId.DHKEM_P256_HKDF_SHA256)
            .setKdfId(HpkeParameters.KdfId.HKDF_SHA256)
            .setAeadId(HpkeParameters.AeadId.AES_128_GCM)
            .setVariant(HpkeParameters.Variant.NO_PREFIX)
            .build();
    // Uncompressed P-256 point of the NIST P-256 generator.
    HpkePublicKey p256Key =
        HpkePublicKey.create(
            p256,
            Bytes.copyFrom(
                hex(
                    "046b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"
                        + "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5")),
            null);
    assertThrows(
        GeneralSecurityException.class, () -> HpkeEncryptWithAssociatedData.create(p256Key));

    // AES-256-GCM is deliberately unsupported (crypto.cc only uses AES-128-GCM), as is ChaCha20.
    for (HpkeParameters.AeadId aeadId :
        new HpkeParameters.AeadId[] {
          HpkeParameters.AeadId.AES_256_GCM, HpkeParameters.AeadId.CHACHA20_POLY1305
        }) {
      HpkePrivateKey key =
          privateKey(
              parameters(aeadId, HpkeParameters.Variant.NO_PREFIX),
              X25519.generatePrivateKey(),
              null);
      assertThrows(
          GeneralSecurityException.class,
          () -> HpkeEncryptWithAssociatedData.create(key.getPublicKey()));
    }
  }
}
