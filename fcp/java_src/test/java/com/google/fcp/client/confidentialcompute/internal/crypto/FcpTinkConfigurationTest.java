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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.crypto.tink.Aead;
import com.google.crypto.tink.Configuration;
import com.google.crypto.tink.HybridDecrypt;
import com.google.crypto.tink.HybridEncrypt;
import com.google.crypto.tink.InsecureSecretKeyAccess;
import com.google.crypto.tink.KeyStatus;
import com.google.crypto.tink.KeysetHandle;
import com.google.crypto.tink.aead.AesGcmSivKey;
import com.google.crypto.tink.aead.AesGcmSivParameters;
import com.google.crypto.tink.hybrid.HpkeParameters;
import com.google.crypto.tink.hybrid.HpkePrivateKey;
import com.google.crypto.tink.hybrid.HpkePublicKey;
import com.google.crypto.tink.hybrid.HybridConfig2026;
import com.google.crypto.tink.signature.EcdsaParameters;
import com.google.crypto.tink.subtle.X25519;
import com.google.crypto.tink.util.Bytes;
import com.google.crypto.tink.util.SecretBytes;
import java.security.GeneralSecurityException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/** Runs under Robolectric, which installs Conscrypt (needed for AES-GCM-SIV) like Android does. */
@RunWith(RobolectricTestRunner.class)
public final class FcpTinkConfigurationTest {

  private static final Configuration CONFIG = FcpTinkConfiguration.get();

  private static final byte[] nonce = {
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

  private static final AesGcmSivParameters AES_GCM_SIV_NO_PREFIX_PARAMETERS;
  private static final AesGcmSivParameters AES_GCM_SIV_TINK_PARAMETERS;
  private static final AesGcmSivParameters AES_GCM_SIV_CRUNCHY_PARAMETERS;
  private static final AesGcmSivParameters AES_256_GCM_SIV_NO_PREFIX_PARAMETERS;
  private static final HpkeParameters HPKE_NO_PREFIX_PARAMETERS;
  private static final HpkeParameters HPKE_TINK_PARAMETERS;
  private static final HpkeParameters HPKE_CRUNCHY_PARAMETERS;

  static {
    try {
      AES_GCM_SIV_NO_PREFIX_PARAMETERS =
          AesGcmSivParameters.builder()
              .setKeySizeBytes(16)
              .setVariant(AesGcmSivParameters.Variant.NO_PREFIX)
              .build();
      AES_GCM_SIV_TINK_PARAMETERS =
          AesGcmSivParameters.builder()
              .setKeySizeBytes(16)
              .setVariant(AesGcmSivParameters.Variant.TINK)
              .build();
      AES_GCM_SIV_CRUNCHY_PARAMETERS =
          AesGcmSivParameters.builder()
              .setKeySizeBytes(16)
              .setVariant(AesGcmSivParameters.Variant.CRUNCHY)
              .build();
      AES_256_GCM_SIV_NO_PREFIX_PARAMETERS =
          AesGcmSivParameters.builder()
              .setKeySizeBytes(32)
              .setVariant(AesGcmSivParameters.Variant.NO_PREFIX)
              .build();
      HPKE_NO_PREFIX_PARAMETERS =
          HpkeParameters.builder()
              .setKemId(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256)
              .setKdfId(HpkeParameters.KdfId.HKDF_SHA256)
              .setAeadId(HpkeParameters.AeadId.AES_128_GCM)
              .setVariant(HpkeParameters.Variant.NO_PREFIX)
              .build();
      HPKE_TINK_PARAMETERS =
          HpkeParameters.builder()
              .setKemId(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256)
              .setKdfId(HpkeParameters.KdfId.HKDF_SHA256)
              .setAeadId(HpkeParameters.AeadId.AES_128_GCM)
              .setVariant(HpkeParameters.Variant.TINK)
              .build();
      HPKE_CRUNCHY_PARAMETERS =
          HpkeParameters.builder()
              .setKemId(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256)
              .setKdfId(HpkeParameters.KdfId.HKDF_SHA256)
              .setAeadId(HpkeParameters.AeadId.AES_128_GCM)
              .setVariant(HpkeParameters.Variant.CRUNCHY)
              .build();
    } catch (GeneralSecurityException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  @Test
  public void generateNew_aesGcmSiv_createsUsableKeyset() throws Exception {
    KeysetHandle handle = KeysetHandle.generateNew(AES_GCM_SIV_NO_PREFIX_PARAMETERS, CONFIG);

    assertThat(handle.size()).isEqualTo(1);
    assertThat(handle.getPrimary().getKey()).isInstanceOf(AesGcmSivKey.class);
    assertThat(((AesGcmSivKey) handle.getPrimary().getKey()).getKeyBytes().size()).isEqualTo(16);
    assertThat(handle.getPrimary().getKey().getIdRequirementOrNull()).isNull();

    CallerProvidedNonceAead aead = handle.getPrimitive(CONFIG, CallerProvidedNonceAead.class);
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] aad = "aad".getBytes(UTF_8);
    byte[] ciphertext = aead.encrypt(nonce, plaintext, aad);
    assertThat(ciphertext).hasLength(plaintext.length + 16); // NO_PREFIX: just ct || tag.
    assertThat(aead.decrypt(nonce, ciphertext, aad)).isEqualTo(plaintext);
  }

  @Test
  public void generateNew_hpke_createsUsableKeysets() throws Exception {
    KeysetHandle privateHandle = KeysetHandle.generateNew(HPKE_NO_PREFIX_PARAMETERS, CONFIG);
    KeysetHandle publicHandle = privateHandle.getPublicKeysetHandle();

    assertThat(privateHandle.size()).isEqualTo(1);
    assertThat(privateHandle.getPrimary().getKey()).isInstanceOf(HpkePrivateKey.class);
    assertThat(privateHandle.getPrimary().getKey().getIdRequirementOrNull()).isNull();
    assertThat(publicHandle.getPrimary().getKey()).isInstanceOf(HpkePublicKey.class);

    HybridEncryptWithAssociatedData encrypt =
        publicHandle.getPrimitive(CONFIG, HybridEncryptWithAssociatedData.class);
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] info = new byte[0];
    byte[] aad = "aad".getBytes(UTF_8);
    byte[] ciphertext = encrypt.encrypt(plaintext, info, aad);
    assertThat(ciphertext).hasLength(32 + plaintext.length + 16); // NO_PREFIX: enc || ct.
  }

  @Test
  public void generateNew_hpke_decryptsWithJni() throws Exception {
    KeysetHandle privateHandle = KeysetHandle.generateNew(HPKE_NO_PREFIX_PARAMETERS, CONFIG);
    KeysetHandle publicHandle = privateHandle.getPublicKeysetHandle();

    // Extracting a single key via getAt(0) only works here because we know which exact key was used
    // for encryption; otherwise Tink's approach with wrapped Decrypt primitives should be used (see
    // {@link #generateNew_hpke_emptyAssociatedData_decryptsWithTinkHybridDecrypt}).
    byte[] privateKeyBytes =
        ((HpkePrivateKey) privateHandle.getAt(0).getKey())
            .getPrivateKeyBytes()
            .toByteArray(InsecureSecretKeyAccess.get());

    HybridEncryptWithAssociatedData encrypt =
        publicHandle.getPrimitive(CONFIG, HybridEncryptWithAssociatedData.class);
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] info = new byte[0];
    byte[] aad = "aad".getBytes(UTF_8);
    byte[] ciphertext = encrypt.encrypt(plaintext, info, aad);
    assertThat(CryptoTestJni.unwrapSymmetricKey(privateKeyBytes, ciphertext, aad))
        .isEqualTo(plaintext);
    assertThrows(
        GeneralSecurityException.class,
        () -> CryptoTestJni.unwrapSymmetricKey(privateKeyBytes, ciphertext, new byte[0]));
  }

  @Test
  public void generateNew_hpke_emptyAssociatedData_decryptsWithTinkHybridDecrypt()
      throws Exception {
    KeysetHandle privateHandle = KeysetHandle.generateNew(HPKE_NO_PREFIX_PARAMETERS, CONFIG);
    KeysetHandle publicHandle = privateHandle.getPublicKeysetHandle();

    HybridEncryptWithAssociatedData encrypt =
        publicHandle.getPrimitive(CONFIG, HybridEncryptWithAssociatedData.class);
    HybridDecrypt decrypt = privateHandle.getPrimitive(HybridConfig2026.get(), HybridDecrypt.class);
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] info = "context-info".getBytes(UTF_8);
    byte[] ciphertext = encrypt.encrypt(plaintext, info, /* associatedData= */ new byte[0]);
    assertThat(decrypt.decrypt(ciphertext, info)).isEqualTo(plaintext);
  }

  @Test
  public void keyRotation_aesGcmSiv_encryptsWithPrimaryDecryptsWithAllEnabled() throws Exception {
    // CallerProvidedNonceAead only supports NO_PREFIX keys, so there is no prefix to pick the right
    // key on decryption: all enabled keys are tried in turn.
    KeysetHandle oldHandle =
        KeysetHandle.newBuilder()
            .setConfiguration(CONFIG)
            .addEntry(
                KeysetHandle.generateEntryFromParameters(AES_GCM_SIV_NO_PREFIX_PARAMETERS)
                    .withFixedId(1001)
                    .makePrimary())
            .build();
    CallerProvidedNonceAead old = oldHandle.getPrimitive(CONFIG, CallerProvidedNonceAead.class);
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] aad = "aad".getBytes(UTF_8);
    byte[] oldCiphertext = old.encrypt(nonce, plaintext, aad);

    // Rotate: add a new primary and another enabled key.
    KeysetHandle rotatedHandle =
        KeysetHandle.newBuilder(oldHandle)
            .setConfiguration(CONFIG)
            .addEntry(
                KeysetHandle.generateEntryFromParameters(AES_GCM_SIV_NO_PREFIX_PARAMETERS)
                    .withFixedId(1002)
                    .makePrimary())
            .addEntry(
                KeysetHandle.generateEntryFromParameters(AES_GCM_SIV_NO_PREFIX_PARAMETERS)
                    .withFixedId(1003))
            .build();
    CallerProvidedNonceAead rotated =
        rotatedHandle.getPrimitive(CONFIG, CallerProvidedNonceAead.class);

    byte[] newCiphertext = rotated.encrypt(nonce, plaintext, aad);
    assertThat(newCiphertext).hasLength(plaintext.length + 16); // Still just ct || tag.
    assertThat(newCiphertext).isNotEqualTo(oldCiphertext); // Encrypted with the new primary.
    assertThat(rotated.decrypt(nonce, newCiphertext, aad)).isEqualTo(plaintext);
    assertThat(rotated.decrypt(nonce, oldCiphertext, aad)).isEqualTo(plaintext);
    assertThrows(GeneralSecurityException.class, () -> old.decrypt(nonce, newCiphertext, aad));
  }

  @Test
  public void keyRotation_aesGcmSiv_doesNotDecryptWithDisabledKey() throws Exception {
    KeysetHandle disabledKeyHandle =
        KeysetHandle.generateNew(AES_GCM_SIV_NO_PREFIX_PARAMETERS, CONFIG);
    CallerProvidedNonceAead disabledKeyAead =
        disabledKeyHandle.getPrimitive(CONFIG, CallerProvidedNonceAead.class);
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] aad = "aad".getBytes(UTF_8);
    byte[] disabledCiphertext = disabledKeyAead.encrypt(nonce, plaintext, aad);

    KeysetHandle rotatedHandle =
        KeysetHandle.newBuilder()
            .setConfiguration(CONFIG)
            .addEntry(
                KeysetHandle.importKey(disabledKeyHandle.getAt(0).getKey())
                    .withFixedId(1001)
                    .setStatus(KeyStatus.DISABLED))
            .addEntry(
                KeysetHandle.generateEntryFromParameters(AES_GCM_SIV_NO_PREFIX_PARAMETERS)
                    .withFixedId(1002)
                    .makePrimary())
            .build();
    CallerProvidedNonceAead rotated =
        rotatedHandle.getPrimitive(CONFIG, CallerProvidedNonceAead.class);

    assertThrows(
        GeneralSecurityException.class, () -> rotated.decrypt(nonce, disabledCiphertext, aad));
  }

  @Test
  public void callerProvidedNonceAead_rejectsKeysetsWithPrefixedKeys() throws Exception {
    AesGcmSivKey tinkKey =
        AesGcmSivKey.builder()
            .setParameters(AES_GCM_SIV_TINK_PARAMETERS)
            .setKeyBytes(SecretBytes.randomBytes(16))
            .setIdRequirement(1001)
            .build();
    AesGcmSivKey crunchyKey =
        AesGcmSivKey.builder()
            .setParameters(AES_GCM_SIV_CRUNCHY_PARAMETERS)
            .setKeyBytes(SecretBytes.randomBytes(16))
            .setIdRequirement(1002)
            .build();
    AesGcmSivKey noPrefixKey =
        AesGcmSivKey.builder()
            .setParameters(AES_GCM_SIV_NO_PREFIX_PARAMETERS)
            .setKeyBytes(SecretBytes.randomBytes(16))
            .build();

    KeysetHandle tinkVariantHandle =
        KeysetHandle.newBuilder().addEntry(KeysetHandle.importKey(tinkKey).makePrimary()).build();
    KeysetHandle crunchyVariantHandle =
        KeysetHandle.newBuilder()
            .addEntry(KeysetHandle.importKey(crunchyKey).makePrimary())
            .build();
    // A prefixed key anywhere in the keyset (here: a non-primary one) is rejected as well.
    KeysetHandle mixedHandle =
        KeysetHandle.newBuilder()
            .addEntry(KeysetHandle.importKey(tinkKey))
            .addEntry(KeysetHandle.importKey(noPrefixKey).withFixedId(1003).makePrimary())
            .build();

    assertThrows(
        GeneralSecurityException.class,
        () -> tinkVariantHandle.getPrimitive(CONFIG, CallerProvidedNonceAead.class));
    assertThrows(
        GeneralSecurityException.class,
        () -> crunchyVariantHandle.getPrimitive(CONFIG, CallerProvidedNonceAead.class));
    assertThrows(
        GeneralSecurityException.class,
        () -> mixedHandle.getPrimitive(CONFIG, CallerProvidedNonceAead.class));
  }

  @Test
  public void keyRotation_hpke_encryptsWithPrimaryKey() throws Exception {
    KeysetHandle oldHandle = KeysetHandle.generateNew(HPKE_NO_PREFIX_PARAMETERS, CONFIG);
    KeysetHandle rotatedHandle =
        KeysetHandle.newBuilder(oldHandle)
            .setConfiguration(CONFIG)
            .addEntry(
                KeysetHandle.generateEntryFromParameters(HPKE_NO_PREFIX_PARAMETERS)
                    .withRandomId()
                    .makePrimary())
            .build();
    byte[] oldPrivateKeyBytes =
        ((HpkePrivateKey) oldHandle.getPrimary().getKey())
            .getPrivateKeyBytes()
            .toByteArray(InsecureSecretKeyAccess.get());
    byte[] newPrivateKeyBytes =
        ((HpkePrivateKey) rotatedHandle.getPrimary().getKey())
            .getPrivateKeyBytes()
            .toByteArray(InsecureSecretKeyAccess.get());
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    byte[] info = new byte[0];
    byte[] aad = "aad".getBytes(UTF_8);

    byte[] newCiphertext =
        rotatedHandle
            .getPublicKeysetHandle()
            .getPrimitive(CONFIG, HybridEncryptWithAssociatedData.class)
            .encrypt(plaintext, info, aad);

    assertThat(newCiphertext).hasLength(32 + plaintext.length + 16); // Still just enc || ct.
    assertThat(CryptoTestJni.unwrapSymmetricKey(newPrivateKeyBytes, newCiphertext, aad))
        .isEqualTo(plaintext);
    assertThrows(
        GeneralSecurityException.class,
        () -> CryptoTestJni.unwrapSymmetricKey(oldPrivateKeyBytes, newCiphertext, aad));
  }

  @Test
  public void hybridWithAssociatedData_rejectsKeysetsWithPrefixedKeys() throws Exception {
    Bytes publicKeyBytes = Bytes.copyFrom(X25519.publicFromPrivate(X25519.generatePrivateKey()));
    HpkePublicKey tinkKey =
        HpkePublicKey.create(HPKE_TINK_PARAMETERS, publicKeyBytes, /* idRequirement= */ 1001);
    HpkePublicKey crunchyKey =
        HpkePublicKey.create(HPKE_CRUNCHY_PARAMETERS, publicKeyBytes, /* idRequirement= */ 1002);
    HpkePublicKey noPrefixKey =
        HpkePublicKey.create(HPKE_NO_PREFIX_PARAMETERS, publicKeyBytes, /* idRequirement= */ null);

    KeysetHandle tinkVariantHandle =
        KeysetHandle.newBuilder().addEntry(KeysetHandle.importKey(tinkKey).makePrimary()).build();
    KeysetHandle crunchyVariantHandle =
        KeysetHandle.newBuilder()
            .addEntry(KeysetHandle.importKey(crunchyKey).makePrimary())
            .build();
    // A prefixed key anywhere in the keyset (here: a non-primary one) is rejected as well.
    KeysetHandle mixedHandle =
        KeysetHandle.newBuilder()
            .addEntry(KeysetHandle.importKey(tinkKey))
            .addEntry(KeysetHandle.importKey(noPrefixKey).withFixedId(1003).makePrimary())
            .build();

    for (KeysetHandle publicHandle :
        new KeysetHandle[] {tinkVariantHandle, crunchyVariantHandle, mixedHandle}) {
      assertThrows(
          GeneralSecurityException.class,
          () -> publicHandle.getPrimitive(CONFIG, HybridEncryptWithAssociatedData.class));
    }
  }

  @Test
  public void createPrimitive_unsupportedPrimitive_fails() throws Exception {
    KeysetHandle aesHandle = KeysetHandle.generateNew(AES_GCM_SIV_NO_PREFIX_PARAMETERS, CONFIG);
    assertThrows(GeneralSecurityException.class, () -> aesHandle.getPrimitive(CONFIG, Aead.class));

    KeysetHandle hpkeHandle = KeysetHandle.generateNew(HPKE_NO_PREFIX_PARAMETERS, CONFIG);
    KeysetHandle hpkePublicHandle = hpkeHandle.getPublicKeysetHandle();
    assertThrows(
        GeneralSecurityException.class,
        () -> hpkePublicHandle.getPrimitive(CONFIG, HybridEncrypt.class));
    assertThrows(
        GeneralSecurityException.class, () -> hpkeHandle.getPrimitive(CONFIG, HybridDecrypt.class));
  }

  @Test
  public void getOrNull_throwsUnsupportedOperationException() {
    assertThrows(UnsupportedOperationException.class, () -> CONFIG.getOrNull(Object.class));
  }

  @Test
  public void wrongKeyTypeForPrimitive_fails() throws Exception {
    KeysetHandle aesHandle = KeysetHandle.generateNew(AES_GCM_SIV_NO_PREFIX_PARAMETERS, CONFIG);
    KeysetHandle hpkeHandle = KeysetHandle.generateNew(HPKE_NO_PREFIX_PARAMETERS, CONFIG);
    KeysetHandle hpkePublicHandle = hpkeHandle.getPublicKeysetHandle();

    assertThrows(
        GeneralSecurityException.class,
        () -> aesHandle.getPrimitive(CONFIG, HybridEncryptWithAssociatedData.class));
    assertThrows(
        GeneralSecurityException.class,
        () -> hpkeHandle.getPrimitive(CONFIG, CallerProvidedNonceAead.class));
    assertThrows(
        GeneralSecurityException.class,
        () -> hpkeHandle.getPrimitive(CONFIG, HybridEncryptWithAssociatedData.class));
    assertThrows(
        GeneralSecurityException.class,
        () -> hpkePublicHandle.getPrimitive(CONFIG, CallerProvidedNonceAead.class));
  }

  @Test
  public void createKey_unsupportedParameters_fails() throws Exception {
    EcdsaParameters ecdsaParameters =
        EcdsaParameters.builder()
            .setCurveType(EcdsaParameters.CurveType.NIST_P256)
            .setHashType(EcdsaParameters.HashType.SHA256)
            .setSignatureEncoding(EcdsaParameters.SignatureEncoding.IEEE_P1363)
            .setVariant(EcdsaParameters.Variant.NO_PREFIX)
            .build();
    assertThrows(
        GeneralSecurityException.class, () -> KeysetHandle.generateNew(ecdsaParameters, CONFIG));

    HpkeParameters p256Hpke =
        HpkeParameters.builder()
            .setKemId(HpkeParameters.KemId.DHKEM_P256_HKDF_SHA256)
            .setKdfId(HpkeParameters.KdfId.HKDF_SHA256)
            .setAeadId(HpkeParameters.AeadId.AES_128_GCM)
            .setVariant(HpkeParameters.Variant.NO_PREFIX)
            .build();
    assertThrows(GeneralSecurityException.class, () -> KeysetHandle.generateNew(p256Hpke, CONFIG));

    HpkeParameters sha384Hpke =
        HpkeParameters.builder()
            .setKemId(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256)
            .setKdfId(HpkeParameters.KdfId.HKDF_SHA384)
            .setAeadId(HpkeParameters.AeadId.AES_128_GCM)
            .setVariant(HpkeParameters.Variant.NO_PREFIX)
            .build();
    assertThrows(
        GeneralSecurityException.class, () -> KeysetHandle.generateNew(sha384Hpke, CONFIG));

    HpkeParameters aes256Hpke =
        HpkeParameters.builder()
            .setKemId(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256)
            .setKdfId(HpkeParameters.KdfId.HKDF_SHA256)
            .setAeadId(HpkeParameters.AeadId.AES_256_GCM)
            .setVariant(HpkeParameters.Variant.NO_PREFIX)
            .build();
    assertThrows(
        GeneralSecurityException.class, () -> KeysetHandle.generateNew(aes256Hpke, CONFIG));

    assertThrows(
        GeneralSecurityException.class,
        () -> KeysetHandle.generateNew(HPKE_TINK_PARAMETERS, CONFIG));
    assertThrows(
        GeneralSecurityException.class,
        () -> KeysetHandle.generateNew(HPKE_CRUNCHY_PARAMETERS, CONFIG));

    // Tink's parameters allow 32-byte or prefixed AES-GCM-SIV keys, but the primitive (and
    // crypto.cc) only use 16-byte NO_PREFIX AES-128-GCM-SIV, so such keys are not generated either.
    assertThrows(
        GeneralSecurityException.class,
        () -> KeysetHandle.generateNew(AES_256_GCM_SIV_NO_PREFIX_PARAMETERS, CONFIG));
    assertThrows(
        GeneralSecurityException.class,
        () -> KeysetHandle.generateNew(AES_GCM_SIV_TINK_PARAMETERS, CONFIG));
    assertThrows(
        GeneralSecurityException.class,
        () -> KeysetHandle.generateNew(AES_GCM_SIV_CRUNCHY_PARAMETERS, CONFIG));
  }
}
