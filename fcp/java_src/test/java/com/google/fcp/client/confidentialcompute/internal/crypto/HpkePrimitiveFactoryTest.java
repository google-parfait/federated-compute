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
import static org.junit.Assert.assertThrows;

import com.google.crypto.tink.hybrid.HpkeParameters;
import java.security.GeneralSecurityException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class HpkePrimitiveFactoryTest {

  @Test
  public void createsTheSupportedCiphersuite() throws Exception {
    HpkeKem kem = HpkePrimitiveFactory.createKem(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256);
    HpkeKdf kdf = HpkePrimitiveFactory.createKdf(HpkeParameters.KdfId.HKDF_SHA256);
    HpkeAead aead = HpkePrimitiveFactory.createAead(HpkeParameters.AeadId.AES_128_GCM);

    assertThat(kem).isInstanceOf(X25519HpkeKem.class);
    assertThat(kem.getKemId()).isEqualTo(HpkeUtil.x25519HkdfSha256KemId);
    assertThat(kdf).isInstanceOf(HkdfHpkeKdf.class);
    assertThat(kdf.getKdfId()).isEqualTo(HpkeUtil.hkdfSha256KdfId);
    assertThat(aead).isInstanceOf(AesGcmHpkeAead.class);
    assertThat(aead.getAeadId()).isEqualTo(HpkeUtil.aes128GcmAeadId);
    assertThat(aead.getKeyLength()).isEqualTo(16);
  }

  @Test
  public void rejectsEveryOtherIdentifier() throws Exception {
    for (HpkeParameters.KemId kemId :
        new HpkeParameters.KemId[] {
          HpkeParameters.KemId.DHKEM_P256_HKDF_SHA256,
          HpkeParameters.KemId.DHKEM_P384_HKDF_SHA384,
          HpkeParameters.KemId.DHKEM_P521_HKDF_SHA512
        }) {
      assertThrows(GeneralSecurityException.class, () -> HpkePrimitiveFactory.createKem(kemId));
    }
    for (HpkeParameters.KdfId kdfId :
        new HpkeParameters.KdfId[] {
          HpkeParameters.KdfId.HKDF_SHA384, HpkeParameters.KdfId.HKDF_SHA512
        }) {
      assertThrows(GeneralSecurityException.class, () -> HpkePrimitiveFactory.createKdf(kdfId));
    }
    // AES-256-GCM is deliberately unsupported: crypto.cc only uses AES-128-GCM.
    for (HpkeParameters.AeadId aeadId :
        new HpkeParameters.AeadId[] {
          HpkeParameters.AeadId.AES_256_GCM, HpkeParameters.AeadId.CHACHA20_POLY1305
        }) {
      assertThrows(GeneralSecurityException.class, () -> HpkePrimitiveFactory.createAead(aeadId));
    }
  }
}
