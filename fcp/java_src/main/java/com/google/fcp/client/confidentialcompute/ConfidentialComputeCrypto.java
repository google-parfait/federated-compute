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
package com.google.fcp.client.confidentialcompute;

import com.google.auto.value.AutoValue;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.protobuf.ByteString;
import java.security.GeneralSecurityException;

/**
 * Cryptography for confidential compute uploads, corresponding to {@code
 * AttestationTransparencyVerifier} ({@code
 * fcp/client/attestation/attestation_transparency_verifier.h}) and {@code MessageEncryptor} ({@code
 * fcp/confidentialcompute/crypto.h}).
 */
public interface ConfidentialComputeCrypto {

  /**
   * Verifies that the encryption key and data access policy are endorsed as required by {@code
   * options}, and returns the key to encrypt uploads to.
   *
   * <p>Matches {@code AttestationTransparencyVerifier::Verify} in {@code
   * fcp/client/attestation/attestation_transparency_verifier.h}.
   *
   * @param options the trusted KMS and access-policy verifying keys, and whether transparency log
   *     entries are required
   * @param signedEndorsements a serialized {@code SignedEndorsements}
   * @param encryptionConfig a serialized {@code ConfidentialEncryptionConfig}
   * @throws GeneralSecurityException if verification fails
   * @throws IllegalArgumentException if {@code signedEndorsements} or {@code encryptionConfig} is
   *     not a valid serialized message
   */
  VerificationResult verify(
      AccessPolicyEndorsementOptions options,
      ByteString signedEndorsements,
      ByteString encryptionConfig)
      throws GeneralSecurityException;

  /**
   * Output of {@link #verify}, corresponding to {@code AttestationVerifier::VerificationResult} in
   * {@code fcp/client/attestation/attestation_verifier.h}.
   */
  @AutoValue
  abstract class VerificationResult {
    public static VerificationResult create(
        Key publicKey,
        ByteString keyId,
        ByteString accessPolicySha256,
        ByteString serializedVerificationRecord) {
      return new AutoValue_ConfidentialComputeCrypto_VerificationResult(
          publicKey, keyId, accessPolicySha256, serializedVerificationRecord);
    }

    /** Verified public key (including algorithm) to encrypt uploads to. */
    public abstract Key publicKey();

    /** Key ID written to {@code BlobHeader.key_id}. */
    public abstract ByteString keyId();

    /**
     * SHA-256 digest of the access policy verified by the endorsements; written to {@code
     * BlobHeader.access_policy_sha256}.
     */
    public abstract ByteString accessPolicySha256();

    /**
     * Serialized {@code fcp.confidentialcompute.AttestationVerificationRecord} describing the
     * verified evidence, endorsements, and access policy, or empty if not produced.
     */
    public abstract ByteString serializedVerificationRecord();
  }

  /**
   * Encrypts {@code plaintext} to {@code recipientPublicKey} with a per-message symmetric key that
   * is HPKE-encrypted to the recipient using the algorithm specified by {@code recipientPublicKey}.
   *
   * <p>Matches {@code MessageEncryptor::Encrypt} in {@code fcp/confidentialcompute/crypto.h}.
   *
   * <p>{@code plaintext} and {@code associatedData} are {@code byte[]} rather than {@link
   * ByteString} because the encryptor implementations both take arrays. A {@link ByteString}
   * parameter would force a full copy of the compressed payload on every call, and the payload is
   * the largest buffer in the upload path. Implementations must not modify either array.
   *
   * @param recipientPublicKey the {@link VerificationResult#publicKey()} from {@link #verify}
   * @param associatedData authenticated by both the AEAD and HPKE layers
   * @throws GeneralSecurityException if the key's algorithm is unsupported or encryption fails
   */
  EncryptionResult encrypt(byte[] plaintext, Key recipientPublicKey, byte[] associatedData)
      throws GeneralSecurityException;

  /**
   * Output of {@link #encrypt}, corresponding to {@code EncryptMessageResult} in {@code
   * fcp/confidentialcompute/crypto.h}.
   */
  @AutoValue
  abstract class EncryptionResult {
    @SuppressWarnings("mutable")
    public static EncryptionResult create(
        byte[] ciphertext, ByteString encappedKey, ByteString encryptedSymmetricKey) {
      return new AutoValue_ConfidentialComputeCrypto_EncryptionResult(
          ciphertext, encappedKey, encryptedSymmetricKey);
    }

    /**
     * The plaintext encrypted with AES-128-GCM-SIV under the fresh symmetric key, with the
     * authentication tag appended. This is the body of the CAv1 envelope.
     *
     * <p>This is the array the AEAD returned, not a copy, so the payload-sized ciphertext is not
     * duplicated before it is framed into the envelope. Callers must not modify it.
     */
    @SuppressWarnings("mutable")
    public abstract byte[] ciphertext();

    /**
     * The HPKE encapsulated key (for X25519, the sender's ephemeral public key) the recipient
     * combines with its private key to unwrap {@link #encryptedSymmetricKey()}.
     */
    public abstract ByteString encappedKey();

    /**
     * The fresh symmetric key, wrapped in a COSE_Key and HPKE-encrypted to the recipient's public
     * key with the associated data bound in. Carried in the CAv1 envelope header next to the
     * encapsulated key.
     */
    public abstract ByteString encryptedSymmetricKey();
  }
}
