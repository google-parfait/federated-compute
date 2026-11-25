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

import com.google.crypto.tink.AccessesPartialKey;
import com.google.crypto.tink.Configuration;
import com.google.crypto.tink.InsecureSecretKeyAccess;
import com.google.crypto.tink.Key;
import com.google.crypto.tink.KeyStatus;
import com.google.crypto.tink.KeysetHandleInterface;
import com.google.crypto.tink.Parameters;
import com.google.crypto.tink.aead.AesGcmSivKey;
import com.google.crypto.tink.aead.AesGcmSivParameters;
import com.google.crypto.tink.hybrid.HpkeParameters;
import com.google.crypto.tink.hybrid.HpkePrivateKey;
import com.google.crypto.tink.hybrid.HpkePublicKey;
import com.google.crypto.tink.subtle.X25519;
import com.google.crypto.tink.util.Bytes;
import com.google.crypto.tink.util.SecretBytes;
import com.google.errorprone.annotations.Immutable;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A Tink {@link Configuration} for the custom primitives of this package ({@link
 * CallerProvidedNonceAead} and {@link HybridEncryptWithAssociatedData}).
 *
 * <p>Usage:
 *
 * <pre>{@code
 * Configuration config = FcpTinkConfiguration.get();
 *
 * // Symmetric part: a NO_PREFIX AES-128-GCM-SIV key, used with a caller-chosen nonce.
 * KeysetHandle symmetricKey = KeysetHandle.generateNew(aes128GcmSivNoPrefixParameters, config);
 * CallerProvidedNonceAead aead = symmetricKey.getPrimitive(config, CallerProvidedNonceAead.class);
 * byte[] ciphertext = aead.encrypt(nonce, plaintext, associatedData);
 *
 * // Hybrid part: HPKE with associated data, from the recipient's public keyset.
 * HybridEncryptWithAssociatedData hpke =
 *     recipientPublicKeyset.getPrimitive(config, HybridEncryptWithAssociatedData.class);
 * byte[] encapsulatedKeyAndCiphertext = hpke.encrypt(plaintext, contextInfo, associatedData);
 * }</pre>
 *
 * <p>This is the extension point Tink offers instead of public API changes: {@link Configuration}
 * is a public interface, {@link com.google.crypto.tink.KeysetHandle#getPrimitive(Configuration,
 * Class)} dispatches to it for <em>any</em> primitive class, and {@link
 * com.google.crypto.tink.KeysetHandle#generateNew(Parameters, Configuration)} uses its {@link
 * #createKey}. Keysets created or parsed with this configuration are regular Tink keysets ({@code
 * AesGcmSivKey}, {@code HpkePublicKey}, {@code HpkePrivateKey}) and interoperate with Tink's own
 * primitives, serialization and key rotation.
 *
 * <p>Keyset semantics mirror Tink's wrappers: encryption uses the primary key; decryption tries all
 * {@code ENABLED} keys in turn. Both custom primitives only support {@code NO_PREFIX} keys (see
 * {@link AesGcmSivCallerProvidedNonceAead} and {@link HpkeEncryptWithAssociatedData}): their
 * ciphertexts carry no Tink output prefix, requesting one of them for a keyset that contains a
 * {@code TINK} or {@code CRUNCHY} key fails, and there is no prefix to select the key on decryption
 * with a rotated keyset.
 *
 * <p>Only the two custom primitives above are supported by {@link #createPrimitive}; standard Tink
 * primitives (such as {@code Aead} or {@code HybridEncrypt}) should be obtained using Tink's own
 * configurations.
 *
 * <p>Limitations compared to Tink-native primitives: no Tink monitoring/logging integration, and
 * {@code HpkeParameters} key generation is limited to the X25519 KEM.
 *
 * <p><b>Thread safety:</b> {@code FcpTinkConfiguration} is stateless, immutable, and thread-safe.
 * Like Tink's own {@link Configuration} implementations (such as {@code HybridConfig2026}), it
 * holds no mutable state, so the single shared instance returned by {@link #get()} can be used
 * concurrently across threads without external synchronization or locking. The {@link
 * CallerProvidedNonceAead} and {@link HybridEncryptWithAssociatedData} primitives returned by
 * {@link #createPrimitive} are likewise thread-safe and require no locking (each operation creates
 * its own per-call {@code Cipher} or {@link HpkeContext}).
 */
@Immutable
public final class FcpTinkConfiguration implements Configuration {

  private static final FcpTinkConfiguration INSTANCE = new FcpTinkConfiguration();

  /**
   * Returns the stateless, thread-safe singleton instance, which can be shared across threads
   * without locking.
   */
  public static Configuration get() {
    return INSTANCE;
  }

  private FcpTinkConfiguration() {}

  @Override
  public <P> P createPrimitive(KeysetHandleInterface keysetHandle, Class<P> clazz)
      throws GeneralSecurityException {
    if (clazz.equals(CallerProvidedNonceAead.class)) {
      return clazz.cast(new CallerProvidedNonceAeadWrapper(keysetHandle));
    }
    if (clazz.equals(HybridEncryptWithAssociatedData.class)) {
      return clazz.cast(new HybridEncryptWithAssociatedDataWrapper(keysetHandle));
    }
    throw new GeneralSecurityException(
        "FcpTinkConfiguration does not support primitive " + clazz.getName());
  }

  @Override
  @Nullable
  public <P> P getOrNull(Class<P> clazz) {
    throw new UnsupportedOperationException("Not implemented");
  }

  /**
   * Generates keys for 16-byte {@code NO_PREFIX} {@link AesGcmSivParameters} and {@code NO_PREFIX}
   * (X25519 / HKDF-SHA256 / AES-128-GCM) {@link HpkeParameters}.
   *
   * <p>Implemented natively (rather than delegated to the registry) so that key generation needs
   * neither {@code AeadConfig.register()} nor the restricted {@code Configuration.createKey} API of
   * another configuration.
   */
  @Override
  @AccessesPartialKey
  public Key createKey(Parameters parameters, @Nullable Integer idRequirement)
      throws GeneralSecurityException {
    if (parameters instanceof AesGcmSivParameters aesGcmSivParameters) {
      // AesGcmSivCallerProvidedNonceAead only accepts 16-byte NO_PREFIX keys (AES-128-GCM-SIV,
      // like crypto.cc), so do not generate keys it would then reject.
      if (aesGcmSivParameters.getKeySizeBytes() != 16
          || !aesGcmSivParameters.getVariant().equals(AesGcmSivParameters.Variant.NO_PREFIX)) {
        throw new GeneralSecurityException(
            "Unsupported AES-GCM-SIV parameters (only 16-byte NO_PREFIX is supported): "
                + aesGcmSivParameters);
      }
      return AesGcmSivKey.builder()
          .setParameters(aesGcmSivParameters)
          .setKeyBytes(SecretBytes.randomBytes(aesGcmSivParameters.getKeySizeBytes()))
          .setIdRequirement(idRequirement)
          .build();
    }
    if (parameters instanceof HpkeParameters hpkeParameters) {
      // Rejects anything but NO_PREFIX DHKEM_X25519_HKDF_SHA256 / HKDF_SHA256 / AES_128_GCM.
      if (!hpkeParameters.getVariant().equals(HpkeParameters.Variant.NO_PREFIX)
          || !hpkeParameters.getKemId().equals(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256)
          || !hpkeParameters.getKdfId().equals(HpkeParameters.KdfId.HKDF_SHA256)
          || !hpkeParameters.getAeadId().equals(HpkeParameters.AeadId.AES_128_GCM)) {
        throw new GeneralSecurityException("Unsupported HPKE parameters: " + hpkeParameters);
      }
      byte[] privateKey = X25519.generatePrivateKey();
      byte[] publicKey = X25519.publicFromPrivate(privateKey);
      HpkePublicKey hpkePublicKey =
          HpkePublicKey.create(hpkeParameters, Bytes.copyFrom(publicKey), idRequirement);
      return HpkePrivateKey.create(
          hpkePublicKey, SecretBytes.copyFrom(privateKey, InsecureSecretKeyAccess.get()));
    }
    throw new GeneralSecurityException(
        "FcpTinkConfiguration cannot create keys for parameters " + parameters);
  }

  // ---------------------------------------------------------------------------------------------
  // Keyset wrapping. These play the role of Tink's PrimitiveWrapper implementations.
  // ---------------------------------------------------------------------------------------------

  /**
   * All {@code ENABLED} keys, in keyset order. The custom primitives only accept {@code NO_PREFIX}
   * keys, so there is no output prefix to select a key by: decryption tries them in turn.
   */
  private static List<Key> enabledKeys(KeysetHandleInterface keysetHandle) {
    List<Key> keys = new ArrayList<>();
    for (int i = 0; i < keysetHandle.size(); i++) {
      KeysetHandleInterface.Entry entry = keysetHandle.getAt(i);
      if (entry.getStatus().equals(KeyStatus.ENABLED)) {
        keys.add(entry.getKey());
      }
    }
    return keys;
  }

  private static final class CallerProvidedNonceAeadWrapper implements CallerProvidedNonceAead {
    private final CallerProvidedNonceAead primary;
    private final List<CallerProvidedNonceAead> all = new ArrayList<>();

    CallerProvidedNonceAeadWrapper(KeysetHandleInterface keysetHandle)
        throws GeneralSecurityException {
      if (!(keysetHandle.getPrimary().getKey() instanceof AesGcmSivKey primaryKey)) {
        throw new GeneralSecurityException(
            "CallerProvidedNonceAead requires AesGcmSivKey, got "
                + keysetHandle.getPrimary().getKey().getClass().getName());
      }
      this.primary = AesGcmSivCallerProvidedNonceAead.create(primaryKey);
      for (Key key : enabledKeys(keysetHandle)) {
        if (!(key instanceof AesGcmSivKey aesGcmSivKey)) {
          throw new GeneralSecurityException(
              "CallerProvidedNonceAead requires AesGcmSivKey, got " + key.getClass().getName());
        }
        all.add(AesGcmSivCallerProvidedNonceAead.create(aesGcmSivKey));
      }
    }

    @Override
    public byte[] encrypt(byte[] nonce, byte[] plaintext, byte[] associatedData)
        throws GeneralSecurityException {
      return primary.encrypt(nonce, plaintext, associatedData);
    }

    @Override
    public byte[] decrypt(byte[] nonce, byte[] ciphertext, byte[] associatedData)
        throws GeneralSecurityException {
      for (CallerProvidedNonceAead aead : all) {
        try {
          return aead.decrypt(nonce, ciphertext, associatedData);
        } catch (GeneralSecurityException e) {
          // Wrong key (authentication failure): try the next one.
        }
      }
      throw new GeneralSecurityException("decryption failed");
    }
  }

  private static final class HybridEncryptWithAssociatedDataWrapper
      implements HybridEncryptWithAssociatedData {
    private final HybridEncryptWithAssociatedData primary;

    HybridEncryptWithAssociatedDataWrapper(KeysetHandleInterface keysetHandle)
        throws GeneralSecurityException {
      // Like Tink's wrappers, validate every enabled key (e.g. reject prefixed ones) even though
      // only the primary is used for encryption.
      for (Key key : enabledKeys(keysetHandle)) {
        if (!(key instanceof HpkePublicKey hpkePublicKey)) {
          throw new GeneralSecurityException(
              "HybridEncryptWithAssociatedData requires HpkePublicKey, got "
                  + key.getClass().getName());
        }
        HpkeParameters parameters = hpkePublicKey.getParameters();
        if (!parameters.getVariant().equals(HpkeParameters.Variant.NO_PREFIX)
            || !parameters.getKemId().equals(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256)
            || !parameters.getKdfId().equals(HpkeParameters.KdfId.HKDF_SHA256)
            || !parameters.getAeadId().equals(HpkeParameters.AeadId.AES_128_GCM)) {
          throw new GeneralSecurityException("Unsupported HPKE parameters: " + parameters);
        }
      }
      if (!(keysetHandle.getPrimary().getKey() instanceof HpkePublicKey primaryKey)) {
        throw new GeneralSecurityException(
            "HybridEncryptWithAssociatedData requires HpkePublicKey, got "
                + keysetHandle.getPrimary().getKey().getClass().getName());
      }
      this.primary = HpkeEncryptWithAssociatedData.create(primaryKey);
    }

    @Override
    public byte[] encrypt(byte[] plaintext, byte[] contextInfo, byte[] associatedData)
        throws GeneralSecurityException {
      return primary.encrypt(plaintext, contextInfo, associatedData);
    }
  }
}
