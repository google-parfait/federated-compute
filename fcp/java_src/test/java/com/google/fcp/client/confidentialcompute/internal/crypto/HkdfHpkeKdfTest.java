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
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1Info;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1Key;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1PkE;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1PkR;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1SharedSecret;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.a1SkE;
import static com.google.fcp.client.confidentialcompute.internal.crypto.HpkeRfcTestVectors.hex;
import static org.junit.Assert.assertThrows;

import com.google.common.primitives.Bytes;
import com.google.crypto.tink.subtle.X25519;
import java.security.GeneralSecurityException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class HkdfHpkeKdfTest {

  private static HkdfHpkeKdf kdf() throws GeneralSecurityException {
    return HkdfHpkeKdf.create("HmacSha256");
  }

  // Section 4.1 of RFC 9180: the KEM shared secret is ExtractAndExpand(dh, kem_context) with the
  // KEM suite id; A.1 gives dh implicitly (skE, pkR) and the resulting shared_secret.
  @Test
  public void extractAndExpand_derivesTheKemSharedSecret() throws Exception {
    HkdfHpkeKdf kdf = kdf();
    byte[] dh = X25519.computeSharedSecret(a1SkE, a1PkR);
    byte[] kemContext = Bytes.concat(a1PkE, a1PkR);

    byte[] sharedSecret =
        kdf.extractAndExpand(
            /* salt= */ null,
            dh,
            "eae_prk",
            kemContext,
            "shared_secret",
            HpkeUtil.kemSuiteId(HpkeUtil.x25519HkdfSha256KemId),
            kdf.getMacLength());

    assertThat(sharedSecret).isEqualTo(a1SharedSecret);
  }

  // Section 5.1 of RFC 9180 (KeySchedule) with the A.1 intermediate values.
  @Test
  public void labeledExtractAndExpand_deriveKeyAndBaseNonce() throws Exception {
    HkdfHpkeKdf kdf = kdf();
    byte[] suiteId =
        HpkeUtil.hpkeSuiteId(
            HpkeUtil.x25519HkdfSha256KemId, HpkeUtil.hkdfSha256KdfId, HpkeUtil.aes128GcmAeadId);
    byte[] pskIdHash = kdf.labeledExtract(HpkeUtil.emptySalt, new byte[0], "psk_id_hash", suiteId);
    byte[] infoHash = kdf.labeledExtract(HpkeUtil.emptySalt, a1Info, "info_hash", suiteId);
    byte[] keyScheduleContext = Bytes.concat(HpkeUtil.baseMode, pskIdHash, infoHash);
    byte[] secret = kdf.labeledExtract(a1SharedSecret, new byte[0], "secret", suiteId);

    assertThat(keyScheduleContext)
        .isEqualTo(
            hex(
                "00725611c9d98c07c03f60095cd32d400d8347d45ed67097bbad50fc56da742d07cb6cffde367bb05"
                    + "65ba28bb02c90744a20f5ef37f30523526106f637abb05449"));
    assertThat(secret)
        .isEqualTo(hex("12fff91991e93b48de37e7daddb52981084bd8aa64289c3788471d9a9712f397"));
    assertThat(kdf.labeledExpand(secret, keyScheduleContext, "key", suiteId, 16)).isEqualTo(a1Key);
    assertThat(kdf.labeledExpand(secret, keyScheduleContext, "base_nonce", suiteId, 12))
        .isEqualTo(a1BaseNonce);
  }

  @Test
  public void labeledExpand_rejectsOutputLongerThanHkdfAllows() throws Exception {
    HkdfHpkeKdf kdf = kdf();
    byte[] suiteId = HpkeUtil.kemSuiteId(HpkeUtil.x25519HkdfSha256KemId);
    byte[] prk = kdf.labeledExtract(HpkeUtil.emptySalt, new byte[] {1}, "eae_prk", suiteId);

    assertThat(kdf.labeledExpand(prk, new byte[0], "x", suiteId, 255 * 32)).hasLength(255 * 32);
    assertThrows(
        GeneralSecurityException.class,
        () -> kdf.labeledExpand(prk, new byte[0], "x", suiteId, 255 * 32 + 1));
  }

  @Test
  public void getKdfIdAndMacLength() throws Exception {
    HkdfHpkeKdf kdf = kdf();
    assertThat(kdf.getKdfId()).isEqualTo(new byte[] {0x00, 0x01});
    assertThat(kdf.getMacLength()).isEqualTo(32);
  }

  @Test
  public void create_unsupportedMacAlgorithm_fails() {
    assertThrows(GeneralSecurityException.class, () -> HkdfHpkeKdf.create("HmacSha512"));
  }
}
