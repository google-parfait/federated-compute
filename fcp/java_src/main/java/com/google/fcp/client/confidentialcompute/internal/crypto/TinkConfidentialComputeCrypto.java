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

import com.google.common.io.BaseEncoding;
import com.google.crypto.tink.AccessesPartialKey;
import com.google.crypto.tink.Configuration;
import com.google.crypto.tink.InsecureSecretKeyAccess;
import com.google.crypto.tink.KeysetHandle;
import com.google.crypto.tink.aead.AesGcmSivKey;
import com.google.crypto.tink.aead.AesGcmSivParameters;
import com.google.crypto.tink.hybrid.HpkeParameters;
import com.google.crypto.tink.hybrid.HpkePublicKey;
import com.google.crypto.tink.util.Bytes;
import com.google.errorprone.annotations.Immutable;
import com.google.fcp.client.confidentialcompute.ConfidentialComputeCrypto;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.intelligence.fcp.confidentialcompute.SignedEndorsements;
import com.google.intelligence.fcp.confidentialcompute.SignedPayload;
import com.google.intelligence.fcp.confidentialcompute.SignedPayload.Signature.Headers;
import com.google.internal.federatedcompute.v1.ConfidentialEncryptionConfig;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protos.fcp.confidentialcompute.AttestationVerificationRecord;
import java.security.GeneralSecurityException;
import java.time.Instant;
import java.util.List;

/**
 * Pure-Java {@link ConfidentialComputeCrypto} built on Tink and {@link FcpTinkConfiguration},
 * without JNI.
 *
 * <p>{@link #verify} matches {@code AttestationTransparencyVerifier::Verify} in {@code
 * fcp/client/attestation/attestation_transparency_verifier.cc}, with {@link SignedPayloadVerifier}
 * and {@link RekorLogEntryVerifier} standing in for {@code payload_transparency.cc} and {@code
 * rekor.cc}. {@link #encrypt} matches {@code MessageEncryptor::EncryptInternal} and {@code
 * crypto_internal::WrapSymmetricKey} in {@code fcp/confidentialcompute/crypto.cc}. Where the C++
 * calls BoringSSL directly, this uses Tink: {@code PublicKeyVerify} for ECDSA, {@link
 * CallerProvidedNonceAead} for AES-128-GCM-SIV with the protocol's fixed nonce, and {@link
 * HybridEncryptWithAssociatedData} for HPKE with associated data.
 */
@Immutable
public final class TinkConfidentialComputeCrypto implements ConfidentialComputeCrypto {

  private static final Configuration FCP_CONFIG = FcpTinkConfiguration.get();

  /**
   * The claim every encryption key's Oak application signature must carry: the key was produced by
   * a binary built from open source. {@code AttestationTransparencyVerifier::Verify} requires it
   * through {@code CheckRequiredClaims}.
   */
  private static final String REQUIRED_OAK_CLAIM =
      "https://github.com/project-oak/oak/blob/main/docs/tr/claim/92939.md";

  /**
   * The AES-GCM-SIV nonce, {@code kNonce} in {@code crypto.cc}. A fixed nonce is safe because each
   * symmetric key encrypts exactly one message and AES-GCM-SIV is nonce-misuse resistant.
   */
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

  /**
   * The COSE_Key that {@code SymmetricKey::Encode} in {@code cose.cc} produces for the symmetric
   * key, up to the key bytes: a 4-entry map with {@code kty: Symmetric (4)}, {@code alg:
   * AES-128-GCM-SIV with fixed nonce (-65538)}, {@code key_ops: [decrypt (4)]} and {@code k:} a
   * 16-byte string, whose bytes follow this prefix.
   */
  private static final ByteString CBOR_KEY_PREFIX =
      ByteString.copyFrom(
          new byte[] {
            (byte) 0xa4,
            0x01,
            0x04,
            0x03,
            0x3a,
            0x00,
            0x01,
            0x00,
            0x01,
            0x04,
            (byte) 0x81,
            0x04,
            0x20,
            0x50
          });

  /** AES-128 key length; {@code EVP_AEAD_key_length(EVP_aead_aes_128_gcm_siv())}. */
  private static final int SYMMETRIC_KEY_SIZE = 16;

  /**
   * X25519 encapsulated key length; the {@code enc} output of {@code EVP_HPKE_CTX_setup_sender}.
   */
  private static final int ENCAPSULATED_KEY_SIZE = 32;

  /** The HPKE info, {@code kInfo} in {@code crypto.cc}, which is empty. */
  private static final byte[] EMPTY_INFO = new byte[0];

  /** DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM; the HPKE suite of {@code crypto.cc}. */
  private static final HpkeParameters HPKE_PARAMETERS;

  /** AES-128-GCM-SIV without a Tink key-ID prefix, so the ciphertext is just {@code ct || tag}. */
  private static final AesGcmSivParameters AES_GCM_SIV_PARAMETERS;

  static {
    try {
      HPKE_PARAMETERS =
          HpkeParameters.builder()
              .setKemId(HpkeParameters.KemId.DHKEM_X25519_HKDF_SHA256)
              .setKdfId(HpkeParameters.KdfId.HKDF_SHA256)
              .setAeadId(HpkeParameters.AeadId.AES_128_GCM)
              .setVariant(HpkeParameters.Variant.NO_PREFIX)
              .build();
      AES_GCM_SIV_PARAMETERS =
          AesGcmSivParameters.builder()
              .setKeySizeBytes(SYMMETRIC_KEY_SIZE)
              .setVariant(AesGcmSivParameters.Variant.NO_PREFIX)
              .build();
    } catch (GeneralSecurityException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  /**
   * Verifies that the upload encryption key and the pipeline configuration are endorsed as {@code
   * options} requires, and returns the key to encrypt to.
   *
   * <p>Matches {@code AttestationTransparencyVerifier::Verify}. The C++ receives the two {@code
   * SignedPayload}s directly and hands the verification record to a logger; here they arrive inside
   * their serialized containers and the record is returned for the caller to log.
   *
   * @param options the trusted KMS and access-policy verifying keys, and the transparency log
   *     requirements
   * @param signedEndorsementsBytes a serialized {@code SignedEndorsements}; its {@code
   *     pipeline_configuration} is verified against {@code access_policy_verifying_keys}
   * @param encryptionConfigBytes a serialized {@code ConfidentialEncryptionConfig}; its {@code
   *     encryption_key} is verified against {@code kms_verifying_keys}
   * @return the endorsed key, its ID, the SHA-256 of the access policy both endorsements name, and
   *     the serialized {@link AttestationVerificationRecord}
   * @throws GeneralSecurityException if either endorsement is missing or fails to verify, or the
   *     two name different access policies
   * @throws IllegalArgumentException if either input is not a valid serialized message
   */
  @Override
  public VerificationResult verify(
      AccessPolicyEndorsementOptions options,
      ByteString signedEndorsementsBytes,
      ByteString encryptionConfigBytes)
      throws GeneralSecurityException {
    // One reading of the clock for the whole verification, like `absl::Now()` at the top of the
    // C++ function, so every signature in both chains is judged at the same instant.
    Instant now = Instant.now();

    SignedEndorsements endorsements;
    ConfidentialEncryptionConfig encryptionConfig;
    try {
      endorsements =
          SignedEndorsements.parseFrom(
              signedEndorsementsBytes, ExtensionRegistryLite.getEmptyRegistry());
      encryptionConfig =
          ConfidentialEncryptionConfig.parseFrom(
              encryptionConfigBytes, ExtensionRegistryLite.getEmptyRegistry());
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException("Failed to parse input protos", e);
    }
    SignedPayload encryptionKeyPayload = encryptionConfig.getEncryptionKey();
    SignedPayload pipelineConfigPayload = endorsements.getPipelineConfiguration();

    // 1. The encryption key must be signed, through a chain ending at a trusted KMS key, and the
    //    chain must be in a transparency log if the options say so. The headers come back innermost
    //    first: index 0 is the signature closest to the KMS key, the last is the signature directly
    //    on the key payload.
    List<Headers> encryptionKeyHeaders =
        SignedPayloadVerifier.verify(
            encryptionKeyPayload,
            options.getKmsVerifyingKeysList(),
            options.getTransparencyLogOptions(),
            now);
    Key key;
    try {
      key =
          Key.parseFrom(
              encryptionKeyPayload.getPayload(), ExtensionRegistryLite.getEmptyRegistry());
    } catch (InvalidProtocolBufferException e) {
      throw new GeneralSecurityException("failed to parse encryption key", e);
    }

    // 2. The outermost headers name the access policy the key was issued for. The TEE's key
    //    management service puts the hash there when it signs the key.
    ByteString accessPolicySha256 =
        encryptionKeyHeaders.get(encryptionKeyHeaders.size() - 1).getAccessPolicySha256();
    if (accessPolicySha256.isEmpty()) {
      throw new GeneralSecurityException("encryption key headers missing access policy SHA-256");
    }

    // 3. The innermost headers carry the Oak application signature: the TEE's own signature over
    //    the signing key, whose headers hash the endorsed attestation evidence so it can be looked
    //    up and replayed. The signature itself is not re-checked here, as in the C++; the chain's
    //    verification already covers these headers, and the claims it carries must say the TEE
    //    binary was built from open source.
    Headers innermostHeaders = encryptionKeyHeaders.get(0);
    Headers oakApplicationSignatureHeaders;
    try {
      oakApplicationSignatureHeaders =
          Headers.parseFrom(
              innermostHeaders.getOakApplicationSignature().getHeaders(),
              ExtensionRegistryLite.getEmptyRegistry());
    } catch (InvalidProtocolBufferException e) {
      throw new GeneralSecurityException("failed to parse oak application signature headers", e);
    }
    if (oakApplicationSignatureHeaders.getEndorsedEvidenceSha256().isEmpty()) {
      throw new GeneralSecurityException(
          "oak application signature headers missing endorsed evidence SHA-256");
    }
    if (!innermostHeaders.getClaimsList().contains(REQUIRED_OAK_CLAIM)) {
      throw new GeneralSecurityException("missing required claims: " + REQUIRED_OAK_CLAIM);
    }

    // 4. The pipeline configuration must be signed by a trusted access-policy key, under the same
    //    transparency log requirements. Only the fact that it verifies matters: unlike the
    //    encryption key, nothing in its headers is inspected, in C++ or here, so the returned
    //    headers are dropped. A failure throws out of verify() like any other.
    SignedPayloadVerifier.verify(
        pipelineConfigPayload,
        options.getAccessPolicyVerifyingKeysList(),
        options.getTransparencyLogOptions(),
        now);
    SignedEndorsements.PipelineConfiguration pipelineConfig;
    try {
      pipelineConfig =
          SignedEndorsements.PipelineConfiguration.parseFrom(
              pipelineConfigPayload.getPayload(), ExtensionRegistryLite.getEmptyRegistry());
    } catch (InvalidProtocolBufferException e) {
      throw new GeneralSecurityException("failed to parse pipeline configuration", e);
    }

    // 5. Both endorsements must be about the same access policy. Comparing the two hashes ties the
    //    key to the pipeline without downloading the policy; the full policy can be fetched by hash
    //    when needed.
    if (!pipelineConfig.getAccessPolicySha256().equals(accessPolicySha256)) {
      throw new GeneralSecurityException(
          "pipeline configuration is endorsed for access policy "
              + hex(pipelineConfig.getAccessPolicySha256())
              + " but the encryption key for access policy "
              + hex(accessPolicySha256));
    }

    // 6. Record what was verified so an observer of the logs can replay the verification and look
    //    up the binaries that will process the data. The C++ passes this to its record logger; the
    //    encoder does that with the returned bytes.
    AttestationVerificationRecord verificationRecord =
        AttestationVerificationRecord.newBuilder()
            .setEncryptionKey(encryptionKeyPayload)
            .setPipelineConfiguration(pipelineConfigPayload)
            .build();
    return VerificationResult.create(
        key,
        key.getKeyId(),
        pipelineConfig.getAccessPolicySha256(),
        verificationRecord.toByteString());
  }

  /**
   * Encrypts {@code plaintext} for {@code recipientPublicKey} with a fresh symmetric key that is
   * itself encrypted to the recipient with HPKE.
   *
   * <p>Matches {@code MessageEncryptor::EncryptInternal} without a release token, which the client
   * never requests. The result is what {@code MessageDecryptor::Decrypt} on the server expects.
   *
   * @param plaintext the message
   * @param recipientPublicKey the {@code HPKE_X25519_SHA256_AES128_GCM} key from {@link #verify};
   *     its {@code key_material} is the raw 32-byte X25519 public key
   * @param associatedData bound to both the message ciphertext and the wrapped key; the serialized
   *     {@code BlobHeader} in uploads
   * @return the AES-GCM-SIV ciphertext, the HPKE encapsulated key, and the HPKE-encrypted COSE_Key
   * @throws GeneralSecurityException if the key has the wrong algorithm or purpose, or Tink fails
   */
  @Override
  @AccessesPartialKey
  public EncryptionResult encrypt(byte[] plaintext, Key recipientPublicKey, byte[] associatedData)
      throws GeneralSecurityException {
    // The same two checks `EncryptInternal` makes on the recipient key.
    if (recipientPublicKey.getAlgorithm() != Key.Algorithm.HPKE_X25519_SHA256_AES128_GCM) {
      throw new GeneralSecurityException(
          "Unsupported public key algorithm: " + recipientPublicKey.getAlgorithmValue());
    }
    if (recipientPublicKey.hasPurpose() && recipientPublicKey.getPurpose() != Key.Purpose.ENCRYPT) {
      throw new GeneralSecurityException(
          "Public key disallows encrypt operation: " + recipientPublicKey.getPurposeValue());
    }

    // 1. A single-use 16-byte AES-128-GCM-SIV key, where the C++ fills `SymmetricKey.k` from
    //    `RAND_bytes`. Tink generates it inside a keyset so the AEAD below never sees raw bytes.
    KeysetHandle symmetricKeyset = KeysetHandle.generateNew(AES_GCM_SIV_PARAMETERS, FCP_CONFIG);

    // 2. The COSE_Key the recipient will unwrap, as `SymmetricKey::Encode` lays it out. This is the
    //    one place the raw key bytes are needed, to put them after the fixed CBOR header.
    AesGcmSivKey symmetricKey = (AesGcmSivKey) symmetricKeyset.getPrimary().getKey();
    byte[] serializedSymmetricKey =
        CBOR_KEY_PREFIX
            .concat(
                ByteString.copyFrom(
                    symmetricKey.getKeyBytes().toByteArray(InsecureSecretKeyAccess.get())))
            .toByteArray();

    // 3. Seal the message with the fixed nonce and the associated data, as `EVP_AEAD_CTX_seal`
    //    does. NO_PREFIX makes the output exactly `ciphertext || tag`.
    CallerProvidedNonceAead aead =
        symmetricKeyset.getPrimitive(FCP_CONFIG, CallerProvidedNonceAead.class);
    byte[] ciphertext = aead.encrypt(fixedNonce, plaintext, associatedData);

    // 4. Wrap the COSE_Key for the recipient with HPKE, binding the same associated data.
    WrappedSymmetricKey wrappedKey =
        wrapSymmetricKey(
            serializedSymmetricKey,
            recipientPublicKey.getKeyMaterial().toByteArray(),
            associatedData);

    return EncryptionResult.create(
        ciphertext, wrappedKey.encappedKey, wrappedKey.encryptedSymmetricKey);
  }

  /** The two outputs of HPKE sealing; {@code WrapSymmetricKeyResult} in {@code crypto.h}. */
  private static final class WrappedSymmetricKey {
    final ByteString encappedKey;
    final ByteString encryptedSymmetricKey;

    WrappedSymmetricKey(ByteString encappedKey, ByteString encryptedSymmetricKey) {
      this.encappedKey = encappedKey;
      this.encryptedSymmetricKey = encryptedSymmetricKey;
    }
  }

  /**
   * Encrypts the serialized symmetric key to the recipient with HPKE, matching {@code
   * crypto_internal::WrapSymmetricKey}: {@code EVP_HPKE_CTX_setup_sender} with the empty info
   * produces the encapsulated key, then {@code EVP_HPKE_CTX_seal} with {@code associatedData}
   * produces the ciphertext. Tink's {@link HybridEncryptWithAssociatedData} returns the two
   * concatenated, so the output is split at the fixed X25519 encapsulated key length.
   *
   * @param serializedSymmetricKey the COSE_Key to wrap
   * @param recipientPublicKeyBytes the raw 32-byte X25519 public key
   * @param associatedData bound to the wrapped key
   * @throws GeneralSecurityException if the public key is not a valid X25519 key or Tink fails
   */
  @AccessesPartialKey
  private static WrappedSymmetricKey wrapSymmetricKey(
      byte[] serializedSymmetricKey, byte[] recipientPublicKeyBytes, byte[] associatedData)
      throws GeneralSecurityException {
    HpkePublicKey recipientKey =
        HpkePublicKey.create(
            HPKE_PARAMETERS, Bytes.copyFrom(recipientPublicKeyBytes), /* idRequirement= */ null);
    KeysetHandle recipientKeyset =
        KeysetHandle.newBuilder()
            .addEntry(KeysetHandle.importKey(recipientKey).withRandomId().makePrimary())
            .build();
    HybridEncryptWithAssociatedData hpke =
        recipientKeyset.getPrimitive(FCP_CONFIG, HybridEncryptWithAssociatedData.class);
    byte[] encappedKeyAndCiphertext =
        hpke.encrypt(serializedSymmetricKey, EMPTY_INFO, associatedData);
    return new WrappedSymmetricKey(
        ByteString.copyFrom(encappedKeyAndCiphertext, 0, ENCAPSULATED_KEY_SIZE),
        ByteString.copyFrom(
            encappedKeyAndCiphertext,
            ENCAPSULATED_KEY_SIZE,
            encappedKeyAndCiphertext.length - ENCAPSULATED_KEY_SIZE));
  }

  /** Formats a hash as lowercase hex for error messages. */
  private static String hex(ByteString value) {
    return BaseEncoding.base16().lowerCase().encode(value.toByteArray());
  }
}
