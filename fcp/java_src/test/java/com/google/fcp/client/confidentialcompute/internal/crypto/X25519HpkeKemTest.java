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
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1PkE;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1PkR;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1SharedSecret;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1SkE;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1SkR;
import static org.junit.Assert.assertThrows;

import com.google.crypto.tink.InsecureSecretKeyAccess;
import com.google.crypto.tink.hybrid.HpkeParameters;
import com.google.crypto.tink.hybrid.HpkePrivateKey;
import com.google.crypto.tink.hybrid.HpkePublicKey;
import com.google.crypto.tink.subtle.X25519;
import com.google.crypto.tink.util.Bytes;
import com.google.crypto.tink.util.SecretBytes;
import java.security.GeneralSecurityException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class X25519HpkeKemTest {

  private static final X25519HpkeKem KEM = createKem();
  private static final HpkeParameters PARAMETERS = createParameters();

  private static X25519HpkeKem createKem() {
    try {
      return new X25519HpkeKem(HkdfHpkeKdf.create("HmacSha256"));
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

  private static HpkePrivateKey privateKey(byte[] privateKey, byte[] publicKey)
      throws GeneralSecurityException {
    HpkePublicKey recipientPublicKey =
        HpkePublicKey.create(PARAMETERS, Bytes.copyFrom(publicKey), /* idRequirement= */ null);
    return HpkePrivateKey.create(
        recipientPublicKey, SecretBytes.copyFrom(privateKey, InsecureSecretKeyAccess.get()));
  }

  @Test
  public void encapsulateWithFixedEphemeralKey() throws Exception {
    HpkeKemEncapOutput output = KEM.encapsulateWithFixedEphemeralKey(a1PkR, a1SkE, a1PkE);

    assertThat(output.getEncapsulatedKey()).isEqualTo(a1PkE);
    assertThat(output.getSharedSecret()).isEqualTo(a1SharedSecret);
  }

  @Test
  public void decapsulate() throws Exception {
    assertThat(KEM.decapsulate(a1PkE, privateKey(a1SkR, a1PkR))).isEqualTo(a1SharedSecret);
  }

  @Test
  public void encapsulate_producesFreshEphemeralResultEveryTime() throws Exception {
    byte[] recipientSk = X25519.generatePrivateKey();
    byte[] recipientPk = X25519.publicFromPrivate(recipientSk);

    HpkeKemEncapOutput first = KEM.encapsulate(recipientPk);
    HpkeKemEncapOutput second = KEM.encapsulate(recipientPk);

    assertThat(first.getEncapsulatedKey()).hasLength(32);
    assertThat(first.getSharedSecret()).hasLength(32);
    assertThat(second.getEncapsulatedKey()).hasLength(32);
    assertThat(second.getSharedSecret()).hasLength(32);
    assertThat(first.getEncapsulatedKey()).isNotEqualTo(second.getEncapsulatedKey());
    assertThat(first.getSharedSecret()).isNotEqualTo(second.getSharedSecret());
  }

  @Test
  public void encapsulateDecapsulate_roundTrips() throws Exception {
    byte[] recipientSk = X25519.generatePrivateKey();
    byte[] recipientPk = X25519.publicFromPrivate(recipientSk);
    HpkePrivateKey recipientPrivateKey = privateKey(recipientSk, recipientPk);

    HpkeKemEncapOutput output = KEM.encapsulate(recipientPk);

    assertThat(KEM.decapsulate(output.getEncapsulatedKey(), recipientPrivateKey))
        .isEqualTo(output.getSharedSecret());
  }

  @Test
  public void decapsulate_withWrongPrivateKey_givesDifferentSecret() throws Exception {
    byte[] otherSk = X25519.generatePrivateKey();
    byte[] otherPk = X25519.publicFromPrivate(otherSk);

    assertThat(KEM.decapsulate(a1PkE, privateKey(otherSk, otherPk))).isNotEqualTo(a1SharedSecret);
  }

  @Test
  public void getKemId_isDhkemX25519HkdfSha256() throws Exception {
    assertThat(KEM.getKemId()).isEqualTo(new byte[] {0x00, 0x20});
  }

  @Test
  public void decapsulate_rejectsMalformedEncapsulatedKey() throws Exception {
    HpkePrivateKey skR = privateKey(a1SkR, a1PkR);
    byte[] shortEncapsulatedKey = new byte[31];

    assertThrows(GeneralSecurityException.class, () -> KEM.decapsulate(shortEncapsulatedKey, skR));
  }

  @Test
  public void encapsulate_nullRecipientPublicKey_throwsNullPointerException() {
    assertThrows(NullPointerException.class, () -> KEM.encapsulate(null));
  }

  @Test
  public void decapsulate_nullArguments_throwsNullPointerException() throws Exception {
    HpkePrivateKey skR = privateKey(a1SkR, a1PkR);

    assertThrows(NullPointerException.class, () -> KEM.decapsulate(null, skR));
    assertThrows(NullPointerException.class, () -> KEM.decapsulate(a1PkE, null));
  }
}
