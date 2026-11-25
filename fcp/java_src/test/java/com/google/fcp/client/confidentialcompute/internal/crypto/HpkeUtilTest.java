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

import com.google.common.primitives.Bytes;
import com.google.crypto.tink.hybrid.HpkeParameters;
import java.security.GeneralSecurityException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link HpkeUtil}. Branched from Tink's {@code HpkeUtilTest}
 * (src/test/java/com/google/crypto/tink/hybrid/internal/HpkeUtilTest.java in tink-java); the {@code
 * intToByteArray} tests are Tink's, the remaining ones cover the helpers of the reduced ciphersuite
 * set.
 */
@RunWith(JUnit4.class)
public final class HpkeUtilTest {

  @Test
  public void intToByteArray_works() {
    assertThat(HpkeUtil.intToByteArray(0, 0)).isEqualTo(new byte[] {});
    assertThat(HpkeUtil.intToByteArray(1, 42)).isEqualTo(new byte[] {(byte) 42});
    assertThat(HpkeUtil.intToByteArray(2, 0x0102)).isEqualTo(new byte[] {(byte) 01, (byte) 0x02});

    assertThat(HpkeUtil.intToByteArray(1, 0xaa)).isEqualTo(new byte[] {(byte) 0xaa});
    assertThat(HpkeUtil.intToByteArray(1, 256 - 1)).isEqualTo(new byte[] {(byte) 0xff});
    assertThat(HpkeUtil.intToByteArray(2, 0xaabb)).isEqualTo(new byte[] {(byte) 0xaa, (byte) 0xbb});
    assertThat(HpkeUtil.intToByteArray(2, 256 * 256 - 1))
        .isEqualTo(new byte[] {(byte) 0xff, (byte) 0xff});
    assertThat(HpkeUtil.intToByteArray(3, 0xaabbcc))
        .isEqualTo(new byte[] {(byte) 0xaa, (byte) 0xbb, (byte) 0xcc});
    assertThat(HpkeUtil.intToByteArray(3, 256 * 256 * 256 - 1))
        .isEqualTo(new byte[] {(byte) 0xff, (byte) 0xff, (byte) 0xff});
    assertThat(HpkeUtil.intToByteArray(4, 0x0abbccdd))
        .isEqualTo(new byte[] {(byte) 0x0a, (byte) 0xbb, (byte) 0xcc, (byte) 0xdd});
    assertThat(HpkeUtil.intToByteArray(4, Integer.MAX_VALUE))
        .isEqualTo(new byte[] {(byte) 0x7f, (byte) 0xff, (byte) 0xff, (byte) 0xff});
  }

  @Test
  public void intToByteArray_failsWithInvalidCapacity() {
    assertThrows(IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(5, 0));
    assertThrows(IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(-1, 0));
  }

  @Test
  public void intToByteArray_valueTooLong_fails() {
    assertThrows(IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(0, 1));
    assertThrows(IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(0, -1));
    assertThrows(IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(1, 256));
    assertThrows(IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(1, -1));
    assertThrows(IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(2, 256 * 256));
    assertThrows(IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(2, -1));
    assertThrows(IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(3, 256 * 256 * 256));
    assertThrows(IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(3, -1));
    assertThrows(IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(4, -1));
    assertThrows(
        IllegalArgumentException.class, () -> HpkeUtil.intToByteArray(4, Integer.MIN_VALUE));
  }

  @Test
  public void algorithmIdentifiers_matchRfc9180() {
    // RFC 9180, Sections 7.1, 7.2 and 7.3.
    assertThat(Bytes.asList(HpkeUtil.baseMode)).containsExactly((byte) 0x00);
    assertThat(Bytes.asList(HpkeUtil.x25519HkdfSha256KemId))
        .containsExactly((byte) 0x00, (byte) 0x20)
        .inOrder();
    assertThat(Bytes.asList(HpkeUtil.hkdfSha256KdfId))
        .containsExactly((byte) 0x00, (byte) 0x01)
        .inOrder();
    assertThat(Bytes.asList(HpkeUtil.aes128GcmAeadId))
        .containsExactly((byte) 0x00, (byte) 0x01)
        .inOrder();
  }

  @Test
  public void suiteIds_followRfc9180() {
    // RFC 9180, Section 4.1 ("KEM" || I2OSP(kem_id, 2)) and Section 5.1 ("HPKE" || ids).
    assertThat(HpkeUtil.kemSuiteId(HpkeUtil.x25519HkdfSha256KemId))
        .isEqualTo(Bytes.concat("KEM".getBytes(UTF_8), new byte[] {0x00, 0x20}));
    assertThat(
            HpkeUtil.hpkeSuiteId(
                HpkeUtil.x25519HkdfSha256KemId, HpkeUtil.hkdfSha256KdfId, HpkeUtil.aes128GcmAeadId))
        .isEqualTo(
            Bytes.concat("HPKE".getBytes(UTF_8), new byte[] {0x00, 0x20, 0x00, 0x01, 0x00, 0x01}));
  }

  @Test
  public void labels_followRfc9180() {
    // RFC 9180, Section 4: LabeledExtract / LabeledExpand.
    byte[] suiteId = HpkeUtil.kemSuiteId(HpkeUtil.x25519HkdfSha256KemId);
    byte[] ikm = new byte[] {1, 2, 3};
    byte[] info = new byte[] {4, 5};

    assertThat(HpkeUtil.labelIkm("eae_prk", ikm, suiteId))
        .isEqualTo(
            Bytes.concat("HPKE-v1".getBytes(UTF_8), suiteId, "eae_prk".getBytes(UTF_8), ikm));
    assertThat(HpkeUtil.labelInfo("shared_secret", info, suiteId, 32))
        .isEqualTo(
            Bytes.concat(
                new byte[] {0x00, 0x20},
                "HPKE-v1".getBytes(UTF_8),
                suiteId,
                "shared_secret".getBytes(UTF_8),
                info));
  }

  @Test
  public void encodingSizeInBytes_x25519Only() throws Exception {
    assertThat(HpkeUtil.encodingSizeInBytes(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256))
        .isEqualTo(32);
    assertThrows(
        GeneralSecurityException.class,
        () -> HpkeUtil.encodingSizeInBytes(HpkeParameters.KemId.DHKEM_P256_HKDF_SHA256));
  }
}
