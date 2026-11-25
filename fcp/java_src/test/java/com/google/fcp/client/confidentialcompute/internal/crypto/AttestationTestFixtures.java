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

import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.io.BaseEncoding;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.intelligence.fcp.confidentialcompute.LogEntry;
import com.google.intelligence.fcp.confidentialcompute.RekorLogEntry;
import com.google.intelligence.fcp.confidentialcompute.SignedEndorsements;
import com.google.intelligence.fcp.confidentialcompute.SignedPayload;
import com.google.internal.federatedcompute.v1.ConfidentialEncryptionConfig;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.security.GeneralSecurityException;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.MessageDigest;
import java.security.PublicKey;
import java.security.Signature;
import java.security.interfaces.ECPublicKey;
import java.security.spec.ECGenParameterSpec;
import java.security.spec.ECPoint;
import java.util.Arrays;

/**
 * Builds genuinely signed attestation material for tests.
 *
 * <p>Verification cannot be disabled in production code, so tests that need to get past the
 * verifier must present real signatures. This generates a fresh set of P-256 roots per instance and
 * mints endorsements that satisfy the attestation verifier.
 */
public final class AttestationTestFixtures {

  /** The "built from open source" Oak claim that the verifier requires. */
  private static final String OAK_CLAIM =
      "https://github.com/project-oak/oak/blob/main/docs/tr/claim/92939.md";

  private static final String KMS_KEY_ID = "kms-key-id";
  private static final String ACCESS_POLICY_KEY_ID = "access-policy-key-id";
  private static final String REKOR_KEY_ID = "rekor-key-id";
  private static final String REKOR_CHECKPOINT_ORIGIN = "rekor.test - 1";

  private final KeyPair kmsKeyPair;
  private final KeyPair accessPolicyKeyPair;
  private final KeyPair rekorKeyPair;
  private final Key kmsKeyProto;
  private final Key accessPolicyKeyProto;
  private final Key rekorKeyProto;

  public AttestationTestFixtures() throws GeneralSecurityException {
    this.kmsKeyPair = generateKeyPair();
    this.accessPolicyKeyPair = generateKeyPair();
    this.rekorKeyPair = generateKeyPair();

    this.kmsKeyProto = verifyingKeyProto(kmsKeyPair.getPublic(), KMS_KEY_ID);
    this.accessPolicyKeyProto =
        verifyingKeyProto(accessPolicyKeyPair.getPublic(), ACCESS_POLICY_KEY_ID);
    this.rekorKeyProto = verifyingKeyProto(rekorKeyPair.getPublic(), REKOR_KEY_ID);
  }

  /** Endorsement options carrying the roots that match the material minted below. */
  public AccessPolicyEndorsementOptions endorsementOptions() {
    return AccessPolicyEndorsementOptions.newBuilder()
        .addKmsVerifyingKeys(kmsKeyProto)
        .addAccessPolicyVerifyingKeys(accessPolicyKeyProto)
        .build();
  }

  /** Endorsement options requiring transparency log entries. */
  public AccessPolicyEndorsementOptions endorsementOptionsWithTransparencyLogRequired() {
    return endorsementOptions().toBuilder()
        .setTransparencyLogOptions(
            AccessPolicyEndorsementOptions.TransparencyLogOptions.newBuilder()
                .setRequireTransparencyLogEntry(true))
        .build();
  }

  /**
   * Endorsement options requiring transparency log entries and trusting this fixture's Rekor key,
   * as production endorsement options do.
   */
  public AccessPolicyEndorsementOptions endorsementOptionsWithRekor() {
    return endorsementOptions().toBuilder()
        .setTransparencyLogOptions(
            AccessPolicyEndorsementOptions.TransparencyLogOptions.newBuilder()
                .setRequireTransparencyLogEntry(true)
                .addRekorVerifyingKeys(rekorKeyProto))
        .build();
  }

  /** Like {@link #endorsementOptionsWithRekor} but trusting a Rekor key that signed nothing. */
  public AccessPolicyEndorsementOptions endorsementOptionsWithWrongRekorKey()
      throws GeneralSecurityException {
    return endorsementOptions().toBuilder()
        .setTransparencyLogOptions(
            AccessPolicyEndorsementOptions.TransparencyLogOptions.newBuilder()
                .setRequireTransparencyLogEntry(true)
                .addRekorVerifyingKeys(
                    verifyingKeyProto(generateKeyPair().getPublic(), REKOR_KEY_ID)))
        .build();
  }

  /** Endorsement options where the KMS key purpose is not set to VERIFY. */
  public AccessPolicyEndorsementOptions endorsementOptionsWithoutVerifyPurpose() {
    return AccessPolicyEndorsementOptions.newBuilder()
        .addKmsVerifyingKeys(kmsKeyProto.toBuilder().clearPurpose())
        .addAccessPolicyVerifyingKeys(accessPolicyKeyProto)
        .build();
  }

  /** Builds a {@link ConfidentialEncryptionConfig} whose encryption key is properly endorsed. */
  public ConfidentialEncryptionConfig signedEncryptionConfig(
      ByteString teePublicKeyMaterial, String teeKeyId, ByteString accessPolicySha256)
      throws GeneralSecurityException, IOException {
    return signedEncryptionConfigWithHeaders(
        teePublicKeyMaterial, teeKeyId, encryptionKeyHeadersBuilder(accessPolicySha256).build());
  }

  /** Builds a {@link ConfidentialEncryptionConfig} with custom signature headers. */
  public ConfidentialEncryptionConfig signedEncryptionConfigWithHeaders(
      ByteString teePublicKeyMaterial, String teeKeyId, SignedPayload.Signature.Headers headers)
      throws GeneralSecurityException, IOException {
    Key teeKey = teeKey(teePublicKeyMaterial, teeKeyId);

    return ConfidentialEncryptionConfig.newBuilder()
        .setEncryptionKey(
            createSignedPayload(teeKey.toByteString(), headers, kmsKeyPair, KMS_KEY_ID))
        .build();
  }

  /**
   * Builds a {@link ConfidentialEncryptionConfig} with an intermediate key signature chain: KMS
   * signs intermediate key, which signs the encryption key.
   */
  public ConfidentialEncryptionConfig signedEncryptionConfigWithSignatureChain(
      ByteString teePublicKeyMaterial, String teeKeyId, ByteString accessPolicySha256)
      throws GeneralSecurityException, IOException {
    KeyPair intermediateKeyPair = generateKeyPair();
    Key intermediateKeyProto =
        verifyingKeyProto(intermediateKeyPair.getPublic(), "intermediate-key-id");

    Key teeKey = teeKey(teePublicKeyMaterial, teeKeyId);

    SignedPayload.Signature.Headers keyHeaders =
        SignedPayload.Signature.Headers.newBuilder()
            .setAlgorithm(Key.Algorithm.ECDSA_P256)
            .setAccessPolicySha256(accessPolicySha256)
            .build();

    SignedPayload.Signature.Headers intermediateHeaders =
        encryptionKeyHeadersBuilder(ByteString.EMPTY).build();

    SignedPayload signedIntermediatePayload =
        createSignedPayload(
            intermediateKeyProto.toByteString(), intermediateHeaders, kmsKeyPair, KMS_KEY_ID);

    return ConfidentialEncryptionConfig.newBuilder()
        .setEncryptionKey(
            createSignedPayload(
                teeKey.toByteString(), keyHeaders, intermediateKeyPair, signedIntermediatePayload))
        .build();
  }

  /**
   * Builds a {@link ConfidentialEncryptionConfig} with an intermediate key signature chain backed
   * by a Rekor log entry. The KMS endorses an intermediate key through a transparency log entry and
   * the intermediate key signs the encryption key.
   */
  public ConfidentialEncryptionConfig signedEncryptionConfigWithSignatureChainAndLogEntry(
      ByteString teePublicKeyMaterial, String teeKeyId, ByteString accessPolicySha256)
      throws GeneralSecurityException, IOException {
    KeyPair intermediateKeyPair = generateKeyPair();
    Key intermediateKeyProto =
        verifyingKeyProto(intermediateKeyPair.getPublic(), "intermediate-key-id");

    Key teeKey = teeKey(teePublicKeyMaterial, teeKeyId);

    SignedPayload.Signature.Headers keyHeaders =
        SignedPayload.Signature.Headers.newBuilder()
            .setAlgorithm(Key.Algorithm.ECDSA_P256)
            .setAccessPolicySha256(accessPolicySha256)
            .build();

    SignedPayload.Signature.Headers intermediateHeaders =
        encryptionKeyHeadersBuilder(ByteString.EMPTY).build();

    SignedPayload signedIntermediatePayload =
        createLogEntrySignedPayload(
            intermediateKeyProto.toByteString(), intermediateHeaders, kmsKeyPair, KMS_KEY_ID);

    return ConfidentialEncryptionConfig.newBuilder()
        .setEncryptionKey(
            createSignedPayload(
                teeKey.toByteString(), keyHeaders, intermediateKeyPair, signedIntermediatePayload))
        .build();
  }

  /** Builds {@link SignedEndorsements} whose pipeline configuration is properly endorsed. */
  public SignedEndorsements signedEndorsements(ByteString accessPolicySha256)
      throws GeneralSecurityException, IOException {
    SignedEndorsements.PipelineConfiguration pipelineConfig =
        SignedEndorsements.PipelineConfiguration.newBuilder()
            .setAccessPolicySha256(accessPolicySha256)
            .build();

    return signedEndorsementsWithPipelinePayload(pipelineConfig.toByteString());
  }

  /** Builds {@link SignedEndorsements} with custom pipeline configuration payload bytes. */
  public SignedEndorsements signedEndorsementsWithPipelinePayload(ByteString payload)
      throws GeneralSecurityException, IOException {
    return SignedEndorsements.newBuilder()
        .setPipelineConfiguration(
            createSignedPayload(
                payload,
                SignedPayload.Signature.Headers.newBuilder()
                    .setAlgorithm(Key.Algorithm.ECDSA_P256)
                    .build(),
                accessPolicyKeyPair,
                ACCESS_POLICY_KEY_ID))
        .build();
  }

  /**
   * Builds a {@link ConfidentialEncryptionConfig} whose encryption key is endorsed through a Rekor
   * transparency log entry (a {@code log_entry} signature) rather than a raw signature, as
   * production endorsements are.
   */
  public ConfidentialEncryptionConfig signedEncryptionConfigWithLogEntry(
      ByteString teePublicKeyMaterial, String teeKeyId, ByteString accessPolicySha256)
      throws GeneralSecurityException, IOException {
    Key teeKey = teeKey(teePublicKeyMaterial, teeKeyId);
    return ConfidentialEncryptionConfig.newBuilder()
        .setEncryptionKey(
            createLogEntrySignedPayload(
                teeKey.toByteString(),
                encryptionKeyHeadersBuilder(accessPolicySha256).build(),
                kmsKeyPair,
                KMS_KEY_ID))
        .build();
  }

  /** Builds {@link SignedEndorsements} whose pipeline configuration has a Rekor log entry. */
  public SignedEndorsements signedEndorsementsWithLogEntry(ByteString accessPolicySha256)
      throws GeneralSecurityException, IOException {
    SignedEndorsements.PipelineConfiguration pipelineConfig =
        SignedEndorsements.PipelineConfiguration.newBuilder()
            .setAccessPolicySha256(accessPolicySha256)
            .build();
    return SignedEndorsements.newBuilder()
        .setPipelineConfiguration(
            createLogEntrySignedPayload(
                pipelineConfig.toByteString(),
                SignedPayload.Signature.Headers.newBuilder()
                    .setAlgorithm(Key.Algorithm.ECDSA_P256)
                    .build(),
                accessPolicyKeyPair,
                ACCESS_POLICY_KEY_ID))
        .build();
  }

  /** A P1363 checkpoint signature and the key that verifies it, for direct Rekor tests. */
  static final class SignedCheckpoint {
    final byte[] signature;
    final Key key;

    private SignedCheckpoint(byte[] signature, Key key) {
      this.signature = signature;
      this.key = key;
    }
  }

  /** Signs a Rekor checkpoint description with a fresh key, like {@code SignCheckpoint} in C++. */
  static SignedCheckpoint signCheckpoint(String checkpoint) throws GeneralSecurityException {
    KeyPair keyPair = generateKeyPair();
    return new SignedCheckpoint(
        signP1363(keyPair, checkpoint.getBytes(UTF_8)),
        verifyingKeyProto(keyPair.getPublic(), "key id"));
  }

  /** Builds an ECDSA P-256 verifying {@link Key} from an uncompressed SEC 1 point in hex. */
  static Key verifyingKeyFromHex(String hexPoint, String keyId) {
    return Key.newBuilder()
        .setAlgorithm(Key.Algorithm.ECDSA_P256)
        .setPurpose(Key.Purpose.VERIFY)
        .setKeyMaterial(ByteString.copyFrom(BaseEncoding.base16().lowerCase().decode(hexPoint)))
        .setKeyId(ByteString.copyFromUtf8(keyId))
        .build();
  }

  /** Returns the SHA-256 digest of the concatenation of {@code parts}. */
  static byte[] sha256(byte[]... parts) throws GeneralSecurityException {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    for (byte[] part : parts) {
      digest.update(part);
    }
    return digest.digest();
  }

  /** Corrupts the first signature in a {@link ConfidentialEncryptionConfig}. */
  static ConfidentialEncryptionConfig corruptSignature(ConfidentialEncryptionConfig config) {
    SignedPayload signedKey = config.getEncryptionKey();
    return config.toBuilder()
        .setEncryptionKey(
            signedKey.toBuilder()
                .setSignatures(
                    0,
                    signedKey.getSignatures(0).toBuilder()
                        .setRawSignature(ByteString.copyFromUtf8("invalid"))))
        .build();
  }

  /** Corrupts the first signature in a {@link SignedEndorsements}. */
  static SignedEndorsements corruptSignature(SignedEndorsements endorsements) {
    SignedPayload pipelineConfig = endorsements.getPipelineConfiguration();
    return endorsements.toBuilder()
        .setPipelineConfiguration(
            pipelineConfig.toBuilder()
                .setSignatures(
                    0,
                    pipelineConfig.getSignatures(0).toBuilder()
                        .setRawSignature(ByteString.copyFromUtf8("invalid"))))
        .build();
  }

  /** Returns a builder initialized with valid headers for the encryption key. */
  SignedPayload.Signature.Headers.Builder encryptionKeyHeadersBuilder(
      ByteString accessPolicySha256) {
    SignedPayload.Signature.Headers oakSignatureHeaders =
        SignedPayload.Signature.Headers.newBuilder()
            .setAlgorithm(Key.Algorithm.ECDSA_P256)
            .setEndorsedEvidenceSha256(ByteString.copyFromUtf8("endorsed evidence hash"))
            .build();

    return SignedPayload.Signature.Headers.newBuilder()
        .setAlgorithm(Key.Algorithm.ECDSA_P256)
        .addClaims(OAK_CLAIM)
        .setAccessPolicySha256(accessPolicySha256)
        .setOakApplicationSignature(
            SignedPayload.Signature.newBuilder().setHeaders(oakSignatureHeaders.toByteString()));
  }

  /** The TEE's HPKE encryption key as it appears in an endorsed {@code encryption_key} payload. */
  private static Key teeKey(ByteString material, String keyId) {
    return Key.newBuilder()
        .setAlgorithm(Key.Algorithm.HPKE_X25519_SHA256_AES128_GCM)
        .setPurpose(Key.Purpose.ENCRYPT)
        .setKeyMaterial(material)
        .setKeyId(ByteString.copyFromUtf8(keyId))
        .build();
  }

  /** Generates a fresh NIST P-256 keypair, the only curve used by this protocol. */
  private static KeyPair generateKeyPair() throws GeneralSecurityException {
    KeyPairGenerator keyPairGenerator = KeyPairGenerator.getInstance("EC");
    keyPairGenerator.initialize(new ECGenParameterSpec("secp256r1"));
    return keyPairGenerator.generateKeyPair();
  }

  /** Wraps a P-256 public key as an {@code ECDSA_P256} verifying {@link Key} with {@code keyId}. */
  private static Key verifyingKeyProto(PublicKey publicKey, String keyId)
      throws GeneralSecurityException {
    return Key.newBuilder()
        .setAlgorithm(Key.Algorithm.ECDSA_P256)
        .setPurpose(Key.Purpose.VERIFY)
        .setKeyMaterial(ByteString.copyFrom(rawEcPublicKey(publicKey)))
        .setKeyId(ByteString.copyFromUtf8(keyId))
        .build();
  }

  /**
   * Signs {@code payload} under {@code headers} with {@code signerKeyPair} and returns the
   * signature with its {@code raw_signature} set; the caller sets the verifier.
   */
  private static SignedPayload.Signature.Builder sign(
      ByteString payload, SignedPayload.Signature.Headers headers, KeyPair signerKeyPair)
      throws GeneralSecurityException, IOException {
    byte[] headersBytes = headers.toByteArray();
    byte[] signedData = constructSignedData(headersBytes, payload.toByteArray());
    return SignedPayload.Signature.newBuilder()
        .setHeaders(ByteString.copyFrom(headersBytes))
        .setRawSignature(ByteString.copyFrom(signP1363(signerKeyPair, signedData)));
  }

  /** Signs {@code data} with SHA256withECDSA, returning the ASN.1 DER encoded signature. */
  private static byte[] signDer(KeyPair signerKeyPair, byte[] data)
      throws GeneralSecurityException {
    Signature signer = Signature.getInstance("SHA256withECDSA");
    signer.initSign(signerKeyPair.getPrivate());
    signer.update(data);
    return signer.sign();
  }

  /** Signs {@code data} with SHA256withECDSA, returning the IEEE P1363 encoded signature. */
  private static byte[] signP1363(KeyPair signerKeyPair, byte[] data)
      throws GeneralSecurityException {
    return derToP1363(signDer(signerKeyPair, data));
  }

  /**
   * Signs {@code payload} with {@code signerKeyPair} and wraps the DER signature in a hashedrekord
   * entry of a single-leaf Rekor tree whose checkpoint is signed by this fixture's Rekor key.
   */
  private SignedPayload createLogEntrySignedPayload(
      ByteString payload,
      SignedPayload.Signature.Headers headers,
      KeyPair signerKeyPair,
      String keyId)
      throws GeneralSecurityException, IOException {
    byte[] headersBytes = headers.toByteArray();
    byte[] signedData = constructSignedData(headersBytes, payload.toByteArray());
    String body =
        "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\"hash\":{\"algorithm\":"
            + "\"sha256\",\"value\":\""
            + BaseEncoding.base16().lowerCase().encode(sha256(signedData))
            + "\"}},\"signature\":{\"content\":\""
            + BaseEncoding.base64().encode(signDer(signerKeyPair, signedData))
            + "\",\"publicKey\":{\"content\":\"\"}}}}";
    byte[] rootHash = sha256(new byte[] {0x00}, body.getBytes(UTF_8));
    String checkpoint =
        REKOR_CHECKPOINT_ORIGIN + "\n1\n" + BaseEncoding.base64().encode(rootHash) + "\n";
    RekorLogEntry logEntry =
        RekorLogEntry.newBuilder()
            .setBody(ByteString.copyFromUtf8(body))
            .setLogIndex(0)
            .setTreeSize(1)
            .setCheckpointOrigin(REKOR_CHECKPOINT_ORIGIN)
            .setCheckpointSignature(
                ByteString.copyFrom(signP1363(rekorKeyPair, checkpoint.getBytes(UTF_8))))
            .setCheckpointSignatureKeyId(rekorKeyProto.getKeyId())
            .build();
    return SignedPayload.newBuilder()
        .setPayload(payload)
        .addSignatures(
            SignedPayload.Signature.newBuilder()
                .setHeaders(ByteString.copyFrom(headersBytes))
                .setVerifyingKeyId(ByteString.copyFromUtf8(keyId))
                .setLogEntry(LogEntry.newBuilder().setRekor(logEntry)))
        .build();
  }

  /** Signs {@code payload} with {@code signerKeyPair}, emitting an IEEE P1363 raw signature. */
  private static SignedPayload createSignedPayload(
      ByteString payload,
      SignedPayload.Signature.Headers headers,
      KeyPair signerKeyPair,
      String keyId)
      throws GeneralSecurityException, IOException {
    return SignedPayload.newBuilder()
        .setPayload(payload)
        .addSignatures(
            sign(payload, headers, signerKeyPair).setVerifyingKeyId(ByteString.copyFromUtf8(keyId)))
        .build();
  }

  /**
   * Signs {@code payload} with {@code signerKeyPair}, whose key is itself endorsed by the nested
   * {@code verifyingKey} payload, forming a signature chain.
   */
  private static SignedPayload createSignedPayload(
      ByteString payload,
      SignedPayload.Signature.Headers headers,
      KeyPair signerKeyPair,
      SignedPayload verifyingKey)
      throws GeneralSecurityException, IOException {
    return SignedPayload.newBuilder()
        .setPayload(payload)
        .addSignatures(sign(payload, headers, signerKeyPair).setVerifyingKey(verifyingKey))
        .build();
  }

  /**
   * Builds the canonical signed structure: {@code varint(len) || "SignedPayload" || varint(len) ||
   * headers || varint(len) || payload}.
   */
  private static byte[] constructSignedData(byte[] headers, byte[] payload) throws IOException {
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bos);
    byte[] contextString = "SignedPayload".getBytes(UTF_8);
    codedOut.writeUInt64NoTag(contextString.length);
    codedOut.writeRawBytes(contextString);
    codedOut.writeUInt64NoTag(headers.length);
    codedOut.writeRawBytes(headers);
    codedOut.writeUInt64NoTag(payload.length);
    codedOut.writeRawBytes(payload);
    codedOut.flush();
    return bos.toByteArray();
  }

  /** Converts a DER-encoded ECDSA signature into the fixed-width IEEE P1363 form. */
  private static byte[] derToP1363(byte[] der) {
    checkState(der.length >= 8 && der[0] == 0x30, "Invalid DER signature format");
    int rOffset = 2;
    checkState(der[rOffset] == 0x02, "Invalid DER signature: missing integer 1");
    int rLen = der[rOffset + 1];
    byte[] r = Arrays.copyOfRange(der, rOffset + 2, rOffset + 2 + rLen);

    int sOffset = rOffset + 2 + rLen;
    checkState(der[sOffset] == 0x02, "Invalid DER signature: missing integer 2");
    int sLen = der[sOffset + 1];
    byte[] s = Arrays.copyOfRange(der, sOffset + 2, sOffset + 2 + sLen);

    byte[] p1363 = new byte[64];
    System.arraycopy(toFixedLengthBytes(r), 0, p1363, 0, 32);
    System.arraycopy(toFixedLengthBytes(s), 0, p1363, 32, 32);
    return p1363;
  }

  /** Encodes a P-256 public key as an uncompressed SEC 1 point ({@code 0x04 || X || Y}). */
  private static byte[] rawEcPublicKey(PublicKey key) throws GeneralSecurityException {
    ECPoint point = ((ECPublicKey) key).getW();
    byte[] raw = new byte[65];
    raw[0] = 0x04;
    System.arraycopy(toFixedLengthBytes(point.getAffineX().toByteArray()), 0, raw, 1, 32);
    System.arraycopy(toFixedLengthBytes(point.getAffineY().toByteArray()), 0, raw, 33, 32);
    return raw;
  }

  /** Left-pads or trims a big-endian magnitude to 32 bytes, dropping a sign byte if present. */
  private static byte[] toFixedLengthBytes(byte[] raw) {
    if (raw.length == 32) {
      return raw;
    }
    byte[] fixed = new byte[32];
    if (raw.length > 32) {
      System.arraycopy(raw, raw.length - 32, fixed, 0, 32);
    } else {
      System.arraycopy(raw, 0, fixed, 32 - raw.length, raw.length);
    }
    return fixed;
  }
}
