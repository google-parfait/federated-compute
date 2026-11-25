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
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1BaseNonce;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1Encryptions;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1Info;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1Key;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1PkE;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1PkR;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1Pt;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1SharedSecret;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1SkE;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1SkR;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.primitives.Bytes;
import com.google.crypto.tink.InsecureSecretKeyAccess;
import com.google.crypto.tink.hybrid.HpkeParameters;
import com.google.crypto.tink.hybrid.HpkePrivateKey;
import com.google.crypto.tink.hybrid.HpkePublicKey;
import com.google.crypto.tink.subtle.X25519;
import com.google.crypto.tink.util.SecretBytes;
import java.security.GeneralSecurityException;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class HpkeContextTest {

  private static final HpkeKem KEM = createKem();
  private static final HpkeKdf KDF = createKdf();
  private static final HpkeAead AEAD = createAead();
  private static final HpkeParameters PARAMETERS = createParameters();

  private static HpkeKem createKem() {
    try {
      return HpkePrimitiveFactory.createKem(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256);
    } catch (GeneralSecurityException e) {
      throw new AssertionError(e);
    }
  }

  private static HpkeKdf createKdf() {
    try {
      return HpkePrimitiveFactory.createKdf(HpkeParameters.KdfId.HKDF_SHA256);
    } catch (GeneralSecurityException e) {
      throw new AssertionError(e);
    }
  }

  private static HpkeAead createAead() {
    try {
      return HpkePrimitiveFactory.createAead(HpkeParameters.AeadId.AES_128_GCM);
    } catch (GeneralSecurityException e) {
      throw new AssertionError(e);
    }
  }

  private static HpkeParameters createParameters() {
    try {
      return HpkeParameters.builder()
          .setKemId(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256)
          .setKdfId(HpkeParameters.KdfId.HKDF_SHA256)
          .setAeadId(HpkeParameters.AeadId.AES_128_GCM)
          .setVariant(HpkeParameters.Variant.NO_PREFIX)
          .build();
    } catch (GeneralSecurityException e) {
      throw new AssertionError(e);
    }
  }

  private static HpkePrivateKey recipientPrivateKey(byte[] privateKey, byte[] publicKey)
      throws GeneralSecurityException {
    HpkePublicKey recipientPublicKey =
        HpkePublicKey.create(
            PARAMETERS,
            com.google.crypto.tink.util.Bytes.copyFrom(publicKey),
            /* idRequirement= */ null);
    return HpkePrivateKey.create(
        recipientPublicKey, SecretBytes.copyFrom(privateKey, InsecureSecretKeyAccess.get()));
  }

  @Test
  public void keySchedule() throws Exception {
    HpkeContext context =
        HpkeContext.createContext(HpkeUtil.baseMode, a1PkE, a1SharedSecret, KEM, KDF, AEAD, a1Info);

    assertThat(context.getEncapsulatedKey()).isEqualTo(a1PkE);
    assertThat(context.getKey()).isEqualTo(a1Key);
    assertThat(context.getBaseNonce()).isEqualTo(a1BaseNonce);
  }

  @Test
  public void encryptions_senderContextSealsEveryVector() throws Exception {
    // The sender context derived from the fixed ephemeral key of A.1 ...
    HpkeKemEncapOutput encap =
        ((X25519HpkeKem) KEM).encapsulateWithFixedEphemeralKey(a1PkR, a1SkE, a1PkE);
    HpkeContext sender =
        HpkeContext.createContext(
            HpkeUtil.baseMode,
            encap.getEncapsulatedKey(),
            encap.getSharedSecret(),
            KEM,
            KDF,
            AEAD,
            a1Info);
    HpkeContext recipient =
        HpkeContext.createRecipientContext(
            a1PkE, recipientPrivateKey(a1SkR, a1PkR), KEM, KDF, AEAD, a1Info);

    // ... reproduces the A.1.1.1 ciphertexts in sequence; the gaps (3, 5..254) are skipped by
    // encrypting and decrypting the same plaintext, which advances both sequence numbers.
    int sequenceNumber = 0;
    for (HpkeRfcTestVectors.Encryption encryption : a1Encryptions) {
      while (sequenceNumber < encryption.sequenceNumber) {
        byte[] skipped = sender.seal(a1Pt, encryption.aad);
        assertThat(recipient.open(skipped, encryption.aad)).isEqualTo(a1Pt);
        sequenceNumber++;
      }
      byte[] ciphertext = sender.seal(a1Pt, encryption.aad);
      assertThat(ciphertext).isEqualTo(encryption.ct);
      assertThat(recipient.open(ciphertext, encryption.aad)).isEqualTo(a1Pt);
      sequenceNumber++;
    }
  }

  @Test
  public void senderAndRecipientContexts_roundTripWithFreshKeys() throws Exception {
    byte[] recipientSk = X25519.generatePrivateKey();
    byte[] recipientPk = X25519.publicFromPrivate(recipientSk);
    byte[] info = "info".getBytes(UTF_8);
    byte[] aad = "aad".getBytes(UTF_8);

    HpkeContext sender = HpkeContext.createSenderContext(recipientPk, KEM, KDF, AEAD, info);
    HpkeContext recipient =
        HpkeContext.createRecipientContext(
            sender.getEncapsulatedKey(),
            recipientPrivateKey(recipientSk, recipientPk),
            KEM,
            KDF,
            AEAD,
            info);

    byte[] ciphertext = sender.seal(a1Pt, aad);
    assertThat(ciphertext).hasLength(a1Pt.length + AesGcmHpkeAead.TAG_LENGTH);
    assertThat(recipient.open(ciphertext, aad)).isEqualTo(a1Pt);
  }

  @Test
  public void seal_producesDifferentCiphertextEveryTime() throws Exception {
    byte[] recipientSk = X25519.generatePrivateKey();
    byte[] recipientPk = X25519.publicFromPrivate(recipientSk);
    byte[] info = "info".getBytes(UTF_8);
    byte[] aad = "aad".getBytes(UTF_8);

    HpkeContext sender = HpkeContext.createSenderContext(recipientPk, KEM, KDF, AEAD, info);

    byte[] first = sender.seal(a1Pt, aad);
    byte[] second = sender.seal(a1Pt, aad);
    assertThat(first).isNotEqualTo(second);
  }

  @Test
  public void open_wrongInfoOrAadOrModifiedCiphertext_fails() throws Exception {
    HpkeRfcTestVectors.Encryption first = a1Encryptions[0];
    HpkePrivateKey skR = recipientPrivateKey(a1SkR, a1PkR);
    byte[] otherInfo = Bytes.concat(a1Info, new byte[] {0});
    byte[] otherAad = Bytes.concat(first.aad, new byte[] {0});
    byte[] truncated = Arrays.copyOf(first.ct, first.ct.length - 1);
    byte[] flippedBit = first.ct.clone();
    flippedBit[flippedBit.length - 1] ^= 1;

    // A fresh context per attempt: a failed open still consumes a sequence number.
    HpkeContext wrongInfo =
        HpkeContext.createRecipientContext(a1PkE, skR, KEM, KDF, AEAD, otherInfo);
    HpkeContext forWrongAad =
        HpkeContext.createRecipientContext(a1PkE, skR, KEM, KDF, AEAD, a1Info);
    HpkeContext forTruncated =
        HpkeContext.createRecipientContext(a1PkE, skR, KEM, KDF, AEAD, a1Info);
    HpkeContext forFlippedBit =
        HpkeContext.createRecipientContext(a1PkE, skR, KEM, KDF, AEAD, a1Info);

    // Both the HPKE info and the AEAD associated data are bound to the ciphertext.
    assertThrows(GeneralSecurityException.class, () -> wrongInfo.open(first.ct, first.aad));
    assertThrows(GeneralSecurityException.class, () -> forWrongAad.open(first.ct, otherAad));
    assertThrows(GeneralSecurityException.class, () -> forTruncated.open(truncated, first.aad));
    assertThrows(GeneralSecurityException.class, () -> forFlippedBit.open(flippedBit, first.aad));
  }

  @Test
  public void sealWithCiphertextOffset_leavesRoomForTheEncapsulatedKey() throws Exception {
    HpkeRfcTestVectors.Encryption first = a1Encryptions[0];
    HpkeKemEncapOutput encap =
        ((X25519HpkeKem) KEM).encapsulateWithFixedEphemeralKey(a1PkR, a1SkE, a1PkE);
    HpkeContext sender =
        HpkeContext.createContext(
            HpkeUtil.baseMode,
            encap.getEncapsulatedKey(),
            encap.getSharedSecret(),
            KEM,
            KDF,
            AEAD,
            a1Info);

    byte[] withRoom = sender.seal(a1Pt, a1PkE.length, first.aad);

    assertThat(Arrays.copyOf(withRoom, a1PkE.length)).isEqualTo(new byte[a1PkE.length]);
    assertThat(Arrays.copyOfRange(withRoom, a1PkE.length, withRoom.length)).isEqualTo(first.ct);
  }
}
