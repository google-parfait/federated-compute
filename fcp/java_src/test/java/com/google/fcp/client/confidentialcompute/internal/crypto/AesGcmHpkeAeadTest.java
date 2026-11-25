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
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1Encryptions;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1Key;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1Pt;
import static org.junit.Assert.assertThrows;

import java.security.GeneralSecurityException;
import java.security.InvalidAlgorithmParameterException;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class AesGcmHpkeAeadTest {

  private static AesGcmHpkeAead aead() throws Exception {
    return new AesGcmHpkeAead(16);
  }

  @Test
  public void encryptions_sealAndOpen() throws Exception {
    for (HpkeRfcTestVectors.Encryption encryption : a1Encryptions) {
      assertThat(aead().seal(a1Key, encryption.nonce, a1Pt, encryption.aad))
          .isEqualTo(encryption.ct);
      assertThat(aead().open(a1Key, encryption.nonce, encryption.ct, encryption.aad))
          .isEqualTo(a1Pt);
    }
  }

  @Test
  public void sealAndOpen_withCiphertextOffset() throws Exception {
    HpkeRfcTestVectors.Encryption first = a1Encryptions[0];
    int offset = 5;

    byte[] withOffset = aead().seal(a1Key, first.nonce, a1Pt, offset, first.aad);

    assertThat(Arrays.copyOf(withOffset, offset)).isEqualTo(new byte[offset]);
    assertThat(Arrays.copyOfRange(withOffset, offset, withOffset.length)).isEqualTo(first.ct);
    assertThat(aead().open(a1Key, first.nonce, withOffset, offset, first.aad)).isEqualTo(a1Pt);
  }

  @Test
  public void open_tamperedCiphertextOrWrongInputs_fails() throws Exception {
    HpkeRfcTestVectors.Encryption first = a1Encryptions[0];
    byte[] flippedBit = first.ct.clone();
    flippedBit[0] ^= 1;
    byte[] otherKey = a1Key.clone();
    otherKey[0] ^= 1;
    byte[] otherNonce = first.nonce.clone();
    otherNonce[0] ^= 1;
    byte[] otherAad = Arrays.copyOf(first.aad, first.aad.length + 1);
    byte[] shorterThanTheTag = new byte[15];
    AesGcmHpkeAead aead = aead();

    assertThrows(
        GeneralSecurityException.class, () -> aead.open(a1Key, first.nonce, flippedBit, first.aad));
    assertThrows(
        GeneralSecurityException.class,
        () -> aead.open(otherKey, first.nonce, first.ct, first.aad));
    assertThrows(
        GeneralSecurityException.class, () -> aead.open(a1Key, otherNonce, first.ct, first.aad));
    assertThrows(
        GeneralSecurityException.class, () -> aead.open(a1Key, first.nonce, first.ct, otherAad));
    assertThrows(
        GeneralSecurityException.class,
        () -> aead.open(a1Key, first.nonce, shorterThanTheTag, first.aad));
  }

  @Test
  public void wrongKeyOrNonceLength_isRejected() throws Exception {
    HpkeRfcTestVectors.Encryption first = a1Encryptions[0];
    byte[] aes256Key = new byte[32];
    byte[] longNonce = new byte[16];
    AesGcmHpkeAead aead = aead();

    assertThrows(InvalidAlgorithmParameterException.class, () -> new AesGcmHpkeAead(32));
    assertThrows(
        GeneralSecurityException.class, () -> aead.seal(aes256Key, first.nonce, a1Pt, first.aad));
    assertThrows(
        GeneralSecurityException.class, () -> aead.seal(a1Key, longNonce, a1Pt, first.aad));
  }

  @Test
  public void identifiersAndLengths_matchRfc9180() throws Exception {
    assertThat(aead().getAeadId()).isEqualTo(new byte[] {0x00, 0x01});
    assertThat(aead().getKeyLength()).isEqualTo(16);
    assertThat(aead().getNonceLength()).isEqualTo(12);
  }

  @Test
  public void seal_nullArguments_throwsNullPointerException() throws Exception {
    HpkeRfcTestVectors.Encryption first = a1Encryptions[0];
    AesGcmHpkeAead aead = aead();

    assertThrows(NullPointerException.class, () -> aead.seal(null, first.nonce, a1Pt, first.aad));
    assertThrows(NullPointerException.class, () -> aead.seal(a1Key, null, a1Pt, first.aad));
    assertThrows(NullPointerException.class, () -> aead.seal(a1Key, first.nonce, null, first.aad));
  }

  @Test
  public void open_nullArguments_throwsNullPointerException() throws Exception {
    HpkeRfcTestVectors.Encryption first = a1Encryptions[0];
    AesGcmHpkeAead aead = aead();

    assertThrows(
        NullPointerException.class, () -> aead.open(null, first.nonce, first.ct, first.aad));
    assertThrows(NullPointerException.class, () -> aead.open(a1Key, null, first.ct, first.aad));
    assertThrows(NullPointerException.class, () -> aead.open(a1Key, first.nonce, null, first.aad));
  }
}
