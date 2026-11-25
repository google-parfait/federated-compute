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
import static com.google.fcp.client.confidentialcompute.internal.crypto.AesGcmSivRfcTestVectors.nonEdgeCaseVector;
import static org.junit.Assert.assertThrows;

import com.google.crypto.tink.InsecureSecretKeyAccess;
import com.google.crypto.tink.aead.AesGcmSivKey;
import com.google.crypto.tink.aead.AesGcmSivParameters;
import com.google.crypto.tink.util.SecretBytes;
import com.google.fcp.client.confidentialcompute.internal.crypto.AesGcmSivRfcTestVectors.TestVector;
import java.security.GeneralSecurityException;
import java.security.NoSuchAlgorithmException;
import java.security.Provider;
import java.security.Security;
import java.util.ArrayList;
import java.util.List;
import javax.crypto.Cipher;
import javax.crypto.NoSuchPaddingException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.ConscryptMode;

/**
 * {@link AesGcmSivCallerProvidedNonceAead} in environments without Conscrypt.
 *
 * <p>{@code ConscryptMode.OFF} makes Robolectric install BouncyCastle only. Robolectric's
 * BouncyCastle is recent and implements GCM-SIV, which the primitive accepts. Removing that
 * provider too leaves no GCM-SIV implementation at all, so the primitive must refuse to be created.
 */
@RunWith(RobolectricTestRunner.class)
@ConscryptMode(ConscryptMode.Mode.OFF)
public final class AesGcmSivCallerProvidedNonceAeadWithoutConscryptTest {

  private static final String CIPHER_TRANSFORMATION = "AES/GCM-SIV/NoPadding";

  @Before
  public void environmentHasNoConscrypt() {
    // Neither Robolectric's Conscrypt nor Android's platform Conscrypt provider.
    for (Provider provider : Security.getProviders()) {
      assertThat(provider.getName()).isNoneOf("Conscrypt", "AndroidOpenSSL");
    }
  }

  private static AesGcmSivKey key(TestVector vector) throws GeneralSecurityException {
    return AesGcmSivKey.builder()
        .setParameters(
            AesGcmSivParameters.builder()
                .setKeySizeBytes(16)
                .setVariant(AesGcmSivParameters.Variant.NO_PREFIX)
                .build())
        .setKeyBytes(SecretBytes.copyFrom(vector.key, InsecureSecretKeyAccess.get()))
        .build();
  }

  @Test
  public void create_withAnotherGenuineGcmSivProvider_works() throws Exception {
    assertThat(Cipher.getInstance(CIPHER_TRANSFORMATION).getProvider().getName()).isEqualTo("BC");
    TestVector vector = nonEdgeCaseVector();

    CallerProvidedNonceAead aead = AesGcmSivCallerProvidedNonceAead.create(key(vector));

    assertThat(aead.encrypt(vector.nonce, vector.plaintext, vector.aad))
        .isEqualTo(vector.ciphertextAndTag());
    assertThat(aead.decrypt(vector.nonce, vector.ciphertextAndTag(), vector.aad))
        .isEqualTo(vector.plaintext);
  }

  @Test
  public void create_withoutAnyGcmSivProvider_fails() throws Exception {
    // Remove every provider that implements the transformation (BouncyCastle here); the JDK's own
    // providers never did.
    AesGcmSivKey key = key(nonEdgeCaseVector());
    List<Provider> removed = new ArrayList<>();
    for (Provider provider : Security.getProviders()) {
      try {
        Cipher.getInstance(CIPHER_TRANSFORMATION, provider);
      } catch (NoSuchAlgorithmException | NoSuchPaddingException e) {
        continue;
      }
      removed.add(provider);
      Security.removeProvider(provider.getName());
    }
    try {
      assertThat(removed).isNotEmpty();
      assertThrows(NoSuchAlgorithmException.class, () -> Cipher.getInstance(CIPHER_TRANSFORMATION));

      assertThrows(
          GeneralSecurityException.class, () -> AesGcmSivCallerProvidedNonceAead.create(key));
    } finally {
      for (Provider provider : removed) {
        Security.addProvider(provider);
      }
    }
  }
}
