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
import static org.junit.Assume.assumeTrue;

import com.google.crypto.tink.subtle.X25519;
import com.google.fcp.client.confidentialcompute.ConfidentialComputeCrypto.EncryptionResult;
import com.google.fcp.client.confidentialcompute.ConfidentialComputeCrypto.VerificationResult;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.intelligence.fcp.confidentialcompute.SignedEndorsements;
import com.google.intelligence.fcp.confidentialcompute.SignedPayload;
import com.google.internal.federatedcompute.v1.ConfidentialEncryptionConfig;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.Timestamp;
import com.google.protos.fcp.confidentialcompute.AttestationVerificationRecord;
import java.security.GeneralSecurityException;
import java.time.Instant;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

@RunWith(RobolectricTestRunner.class)
public final class TinkConfidentialComputeCryptoTest {
  private static final ByteString TEE_PUBLIC_KEY = ByteString.copyFromUtf8("tee-public-key");
  private static final ByteString ACCESS_POLICY_SHA256 = ByteString.copyFromUtf8("policy-sha256");
  private static final byte[] keyId = "tee-key-id".getBytes(UTF_8);

  private final TinkConfidentialComputeCrypto crypto = new TinkConfidentialComputeCrypto();

  @Test
  public void verify_validMaterial_returnsKey() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    ConfidentialEncryptionConfig config =
        fixtures.signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256);
    SignedEndorsements endorsements = fixtures.signedEndorsements(ACCESS_POLICY_SHA256);

    VerificationResult result =
        crypto.verify(
            fixtures.endorsementOptions(), endorsements.toByteString(), config.toByteString());

    assertThat(result.publicKey().getKeyMaterial()).isEqualTo(TEE_PUBLIC_KEY);
    assertThat(result.publicKey().getAlgorithm())
        .isEqualTo(Key.Algorithm.HPKE_X25519_SHA256_AES128_GCM);
    assertThat(result.keyId()).isEqualTo(ByteString.copyFromUtf8("tee-key-id"));
    assertThat(result.accessPolicySha256()).isEqualTo(ACCESS_POLICY_SHA256);

    assertThat(result.serializedVerificationRecord().isEmpty()).isFalse();
    AttestationVerificationRecord record =
        AttestationVerificationRecord.parseFrom(
            result.serializedVerificationRecord(), ExtensionRegistryLite.getEmptyRegistry());
    assertThat(record.getEncryptionKey()).isEqualTo(config.getEncryptionKey());
    assertThat(record.getPipelineConfiguration())
        .isEqualTo(endorsements.getPipelineConfiguration());
  }

  @Test
  public void verify_corruptSignature_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    AccessPolicyEndorsementOptions options = fixtures.endorsementOptions();
    ByteString signedEndorsements =
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256).toByteString();
    ByteString encryptionConfig =
        AttestationTestFixtures.corruptSignature(
                fixtures.signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256))
            .toByteString();

    assertThrows(
        GeneralSecurityException.class,
        () -> crypto.verify(options, signedEndorsements, encryptionConfig));
  }

  @Test
  public void verify_intermediateKeySignatureChain_returnsKey() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    ConfidentialEncryptionConfig config =
        fixtures.signedEncryptionConfigWithSignatureChain(
            TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256);

    VerificationResult result =
        crypto.verify(
            fixtures.endorsementOptions(),
            fixtures.signedEndorsements(ACCESS_POLICY_SHA256).toByteString(),
            config.toByteString());

    assertThat(result.publicKey().getKeyMaterial()).isEqualTo(TEE_PUBLIC_KEY);
    assertThat(result.accessPolicySha256()).isEqualTo(ACCESS_POLICY_SHA256);
  }

  @Test
  public void verify_transparencyLogRequiredButMissing_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    AccessPolicyEndorsementOptions options =
        fixtures.endorsementOptionsWithTransparencyLogRequired();
    ByteString signedEndorsements =
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256).toByteString();
    ByteString encryptionConfig =
        fixtures
            .signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256)
            .toByteString();

    GeneralSecurityException e =
        assertThrows(
            GeneralSecurityException.class,
            () -> crypto.verify(options, signedEndorsements, encryptionConfig));

    assertThat(e).hasMessageThat().contains("does not have a transparency log entry");
  }

  @Test
  public void verify_kmsKeyWithoutVerifyPurpose_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    AccessPolicyEndorsementOptions options = fixtures.endorsementOptionsWithoutVerifyPurpose();
    ByteString signedEndorsements =
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256).toByteString();
    ByteString encryptionConfig =
        fixtures
            .signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256)
            .toByteString();

    GeneralSecurityException e =
        assertThrows(
            GeneralSecurityException.class,
            () -> crypto.verify(options, signedEndorsements, encryptionConfig));

    assertThat(e).hasMessageThat().contains("key is not a verifying key");
  }

  @Test
  public void verify_missingOakClaim_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    AccessPolicyEndorsementOptions options = fixtures.endorsementOptions();
    ByteString signedEndorsements =
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256).toByteString();
    ByteString encryptionConfig =
        fixtures
            .signedEncryptionConfigWithHeaders(
                TEE_PUBLIC_KEY,
                "tee-key-id",
                fixtures.encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256).clearClaims().build())
            .toByteString();

    GeneralSecurityException e =
        assertThrows(
            GeneralSecurityException.class,
            () -> crypto.verify(options, signedEndorsements, encryptionConfig));

    assertThat(e).hasMessageThat().contains("missing required claims");
  }

  @Test
  public void verify_corruptEndorsementsSignature_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    AccessPolicyEndorsementOptions options = fixtures.endorsementOptions();
    ByteString signedEndorsements =
        AttestationTestFixtures.corruptSignature(fixtures.signedEndorsements(ACCESS_POLICY_SHA256))
            .toByteString();
    ByteString encryptionConfig =
        fixtures
            .signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256)
            .toByteString();

    GeneralSecurityException e =
        assertThrows(
            GeneralSecurityException.class,
            () -> crypto.verify(options, signedEndorsements, encryptionConfig));

    assertThat(e).hasMessageThat().contains("signature verification failed");
  }

  @Test
  public void verify_endorsementsForDifferentAccessPolicy_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    AccessPolicyEndorsementOptions options = fixtures.endorsementOptions();
    ByteString signedEndorsements =
        fixtures.signedEndorsements(ByteString.copyFromUtf8("other-policy-sha256")).toByteString();
    ByteString encryptionConfig =
        fixtures
            .signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256)
            .toByteString();

    GeneralSecurityException e =
        assertThrows(
            GeneralSecurityException.class,
            () -> crypto.verify(options, signedEndorsements, encryptionConfig));

    assertThat(e).hasMessageThat().contains("but the encryption key for access policy");
  }

  @Test
  public void verify_unparseablePipelineConfiguration_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    AccessPolicyEndorsementOptions options = fixtures.endorsementOptions();
    ByteString notAProto = ByteString.copyFrom(new byte[] {(byte) 0xFF});
    ByteString signedEndorsements =
        fixtures.signedEndorsementsWithPipelinePayload(notAProto).toByteString();
    ByteString encryptionConfig =
        fixtures
            .signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256)
            .toByteString();

    GeneralSecurityException e =
        assertThrows(
            GeneralSecurityException.class,
            () -> crypto.verify(options, signedEndorsements, encryptionConfig));

    assertThat(e).hasMessageThat().contains("failed to parse pipeline configuration");
  }

  @Test
  public void verify_malformedInput_throwsIllegalArgumentException() throws Exception {
    AccessPolicyEndorsementOptions options = new AttestationTestFixtures().endorsementOptions();
    ByteString notAProto = ByteString.copyFromUtf8("not a proto");

    assertThrows(
        IllegalArgumentException.class, () -> crypto.verify(options, notAProto, notAProto));
  }

  @Test
  public void verify_calledTwice_succeedsBothTimes() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    ConfidentialEncryptionConfig config =
        fixtures.signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256);
    AccessPolicyEndorsementOptions options = fixtures.endorsementOptions();
    ByteString signedEndorsements =
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256).toByteString();
    ByteString encryptionConfig = config.toByteString();

    VerificationResult first = crypto.verify(options, signedEndorsements, encryptionConfig);
    VerificationResult second = crypto.verify(options, signedEndorsements, encryptionConfig);

    assertThat(second).isEqualTo(first);
  }

  @Test
  public void encrypt_x25519Key_decryptsWithCppServer() throws Exception {
    assumeTrue("C++ CBOR/COSE support is disabled", CryptoTestJni.isCborSupported());
    byte[] rawPrivateKey = X25519.generatePrivateKey();
    byte[] rawPublicKey = X25519.publicFromPrivate(rawPrivateKey);
    byte[] plaintext = "secret payload".getBytes(UTF_8);
    byte[] associatedData = "blob header".getBytes(UTF_8);

    EncryptionResult result =
        crypto.encrypt(plaintext, hpkeKey(ByteString.copyFrom(rawPublicKey)), associatedData);

    assertThat(result.ciphertext()).hasLength(plaintext.length + 16);
    assertThat(result.encappedKey().size()).isEqualTo(32);
    assertThat(result.encryptedSymmetricKey().isEmpty()).isFalse();

    byte[] decrypted =
        CryptoTestJni.decryptMessage(
            rawPrivateKey,
            keyId,
            result.ciphertext(),
            associatedData,
            result.encryptedSymmetricKey().toByteArray(),
            associatedData,
            result.encappedKey().toByteArray());
    assertThat(decrypted).isEqualTo(plaintext);
  }

  @Test
  public void encrypt_differentAssociatedData_failsCppDecryption() throws Exception {
    assumeTrue("C++ CBOR/COSE support is disabled", CryptoTestJni.isCborSupported());
    byte[] rawPrivateKey = X25519.generatePrivateKey();
    byte[] rawPublicKey = X25519.publicFromPrivate(rawPrivateKey);
    EncryptionResult result =
        crypto.encrypt(
            "secret payload".getBytes(UTF_8),
            hpkeKey(ByteString.copyFrom(rawPublicKey)),
            "blob header".getBytes(UTF_8));

    byte[] otherAad = "other".getBytes(UTF_8);
    assertThrows(
        GeneralSecurityException.class,
        () ->
            CryptoTestJni.decryptMessage(
                rawPrivateKey,
                keyId,
                result.ciphertext(),
                otherAad,
                result.encryptedSymmetricKey().toByteArray(),
                otherAad,
                result.encappedKey().toByteArray()));
  }

  @Test
  public void encrypt_calledTwice_producesDistinctOutputs() throws Exception {
    byte[] rawPublicKey = X25519.publicFromPrivate(X25519.generatePrivateKey());
    Key key = hpkeKey(ByteString.copyFrom(rawPublicKey));
    byte[] plaintext = "secret payload".getBytes(UTF_8);
    byte[] associatedData = "blob header".getBytes(UTF_8);

    EncryptionResult first = crypto.encrypt(plaintext, key, associatedData);
    EncryptionResult second = crypto.encrypt(plaintext, key, associatedData);

    assertThat(first.ciphertext()).isNotEqualTo(second.ciphertext());
    assertThat(first.encappedKey()).isNotEqualTo(second.encappedKey());
    assertThat(first.encryptedSymmetricKey()).isNotEqualTo(second.encryptedSymmetricKey());
  }

  @Test
  public void encrypt_invalidKeyMaterial_throws() {
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    Key shortKey = hpkeKey(ByteString.copyFromUtf8("short"));

    assertThrows(
        GeneralSecurityException.class, () -> crypto.encrypt(plaintext, shortKey, new byte[0]));
  }

  @Test
  public void encrypt_unsupportedAlgorithm_throws() throws Exception {
    byte[] rawPublicKey = X25519.publicFromPrivate(X25519.generatePrivateKey());
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    Key signingKey =
        Key.newBuilder()
            .setAlgorithm(Key.Algorithm.ECDSA_P256)
            .setKeyMaterial(ByteString.copyFrom(rawPublicKey))
            .build();

    assertThrows(
        GeneralSecurityException.class, () -> crypto.encrypt(plaintext, signingKey, new byte[0]));
  }

  @Test
  public void encrypt_wrongKeyPurpose_throws() throws Exception {
    byte[] rawPublicKey = X25519.publicFromPrivate(X25519.generatePrivateKey());
    Key decryptOnlyKey =
        Key.newBuilder()
            .setAlgorithm(Key.Algorithm.HPKE_X25519_SHA256_AES128_GCM)
            .setPurpose(Key.Purpose.DECRYPT)
            .setKeyMaterial(ByteString.copyFrom(rawPublicKey))
            .build();

    assertThrows(
        GeneralSecurityException.class,
        () -> crypto.encrypt("plaintext".getBytes(UTF_8), decryptOnlyKey, new byte[0]));
  }

  @Test
  public void encrypt_unrecognizedAlgorithmValue_throwsGeneralSecurityException() throws Exception {
    byte[] rawPublicKey = X25519.publicFromPrivate(X25519.generatePrivateKey());
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    Key key =
        Key.newBuilder()
            .setAlgorithmValue(9999)
            .setKeyMaterial(ByteString.copyFrom(rawPublicKey))
            .build();

    assertThrows(GeneralSecurityException.class, () -> crypto.encrypt(plaintext, key, new byte[0]));
  }

  @Test
  public void encrypt_unrecognizedPurposeValue_throwsGeneralSecurityException() throws Exception {
    byte[] rawPublicKey = X25519.publicFromPrivate(X25519.generatePrivateKey());
    byte[] plaintext = "plaintext".getBytes(UTF_8);
    Key key =
        Key.newBuilder()
            .setAlgorithm(Key.Algorithm.HPKE_X25519_SHA256_AES128_GCM)
            .setPurposeValue(9999)
            .setKeyMaterial(ByteString.copyFrom(rawPublicKey))
            .build();

    assertThrows(GeneralSecurityException.class, () -> crypto.encrypt(plaintext, key, new byte[0]));
  }

  // ---------------------------------------------------------------------------------------------
  // Signature header time bounds (VerifyHeaders in payload_transparency.cc).
  // ---------------------------------------------------------------------------------------------

  @Test
  public void verify_headersWithinTimeBounds_returnsKey() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    SignedPayload.Signature.Headers headers =
        fixtures
            .encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256)
            .setIssuedAt(timestamp(Instant.now().minusSeconds(3600)))
            .setNotBefore(timestamp(Instant.now().minusSeconds(60)))
            .setNotAfter(timestamp(Instant.now().plusSeconds(3600)))
            .build();

    VerificationResult result =
        crypto.verify(
            fixtures.endorsementOptions(),
            fixtures.signedEndorsements(ACCESS_POLICY_SHA256).toByteString(),
            fixtures
                .signedEncryptionConfigWithHeaders(TEE_PUBLIC_KEY, "tee-key-id", headers)
                .toByteString());

    assertThat(result.publicKey().getKeyMaterial()).isEqualTo(TEE_PUBLIC_KEY);
  }

  @Test
  public void verify_notAfterInPast_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    SignedPayload.Signature.Headers headers =
        fixtures
            .encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256)
            .setNotAfter(timestamp(Instant.now().minusSeconds(60)))
            .build();

    GeneralSecurityException e = assertVerifyThrows(fixtures, headers);

    assertThat(e).hasMessageThat().contains("not_after is in the past");
  }

  @Test
  public void verify_notBeforeInFuture_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    SignedPayload.Signature.Headers headers =
        fixtures
            .encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256)
            .setNotBefore(timestamp(Instant.now().plusSeconds(3600)))
            .build();

    GeneralSecurityException e = assertVerifyThrows(fixtures, headers);

    assertThat(e).hasMessageThat().contains("not_before is in the future");
  }

  @Test
  public void verify_issuedAtInFutureWithoutNotBefore_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    SignedPayload.Signature.Headers headers =
        fixtures
            .encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256)
            .setIssuedAt(timestamp(Instant.now().plusSeconds(3600)))
            .build();

    GeneralSecurityException e = assertVerifyThrows(fixtures, headers);

    assertThat(e).hasMessageThat().contains("issued_at is in the future");
  }

  @Test
  public void verify_issuedAtInFutureWithNotBeforeInPast_returnsKey() throws Exception {
    // not_before takes precedence over issued_at as the start time.
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    SignedPayload.Signature.Headers headers =
        fixtures
            .encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256)
            .setIssuedAt(timestamp(Instant.now().plusSeconds(3600)))
            .setNotBefore(timestamp(Instant.now().minusSeconds(60)))
            .build();

    VerificationResult result =
        crypto.verify(
            fixtures.endorsementOptions(),
            fixtures.signedEndorsements(ACCESS_POLICY_SHA256).toByteString(),
            fixtures
                .signedEncryptionConfigWithHeaders(TEE_PUBLIC_KEY, "tee-key-id", headers)
                .toByteString());

    assertThat(result.publicKey().getKeyMaterial()).isEqualTo(TEE_PUBLIC_KEY);
  }

  // ---------------------------------------------------------------------------------------------
  // Rekor transparency log entries through verify().
  // ---------------------------------------------------------------------------------------------

  @Test
  public void verify_rekorLogEntryWithTransparencyLogRequired_returnsKey() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    ConfidentialEncryptionConfig config =
        fixtures.signedEncryptionConfigWithLogEntry(
            TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256);
    SignedEndorsements endorsements = fixtures.signedEndorsementsWithLogEntry(ACCESS_POLICY_SHA256);

    VerificationResult result =
        crypto.verify(
            fixtures.endorsementOptionsWithRekor(),
            endorsements.toByteString(),
            config.toByteString());

    assertThat(result.publicKey().getKeyMaterial()).isEqualTo(TEE_PUBLIC_KEY);
    assertThat(result.keyId()).isEqualTo(ByteString.copyFromUtf8("tee-key-id"));
    assertThat(result.accessPolicySha256()).isEqualTo(ACCESS_POLICY_SHA256);
  }

  @Test
  public void verify_rekorLogEntryWithUntrustedRekorKey_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    AccessPolicyEndorsementOptions options = fixtures.endorsementOptionsWithWrongRekorKey();
    ByteString signedEndorsements =
        fixtures.signedEndorsementsWithLogEntry(ACCESS_POLICY_SHA256).toByteString();
    ByteString encryptionConfig =
        fixtures
            .signedEncryptionConfigWithLogEntry(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256)
            .toByteString();

    GeneralSecurityException e =
        assertThrows(
            GeneralSecurityException.class,
            () -> crypto.verify(options, signedEndorsements, encryptionConfig));

    assertThat(e).hasMessageThat().contains("log entry checkpoint signature is invalid");
  }

  @Test
  public void verify_rekorLogEntryWithoutRekorKeys_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    AccessPolicyEndorsementOptions options =
        fixtures.endorsementOptionsWithTransparencyLogRequired();
    ByteString signedEndorsements =
        fixtures.signedEndorsementsWithLogEntry(ACCESS_POLICY_SHA256).toByteString();
    ByteString encryptionConfig =
        fixtures
            .signedEncryptionConfigWithLogEntry(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256)
            .toByteString();

    GeneralSecurityException e =
        assertThrows(
            GeneralSecurityException.class,
            () -> crypto.verify(options, signedEndorsements, encryptionConfig));

    assertThat(e).hasMessageThat().contains("no matching signing keys");
  }

  // ---------------------------------------------------------------------------------------------
  // Additional verification edge cases.
  // ---------------------------------------------------------------------------------------------

  @Test
  public void verify_rawSignatures_succeeds() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    assertVerifyAccepts(
        fixtures.endorsementOptions(),
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256));
  }

  @Test
  public void verify_rekorLogEntries_succeeds() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    assertVerifyAccepts(
        fixtures.endorsementOptionsWithRekor(),
        fixtures.signedEndorsementsWithLogEntry(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfigWithLogEntry(
            TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256));
  }

  @Test
  public void verify_rekorLogEntriesWithUntrustedRekorKey_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    assertVerifyRejects(
        fixtures.endorsementOptionsWithWrongRekorKey(),
        fixtures.signedEndorsementsWithLogEntry(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfigWithLogEntry(
            TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256),
        "log entry checkpoint signature is invalid");
  }

  @Test
  public void verify_rawSignatureWithTransparencyLogRequired_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    assertVerifyRejects(
        fixtures.endorsementOptionsWithTransparencyLogRequired(),
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256),
        "does not have a transparency log entry");
  }

  @Test
  public void verify_expiredHeaders_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    assertVerifyRejects(
        fixtures.endorsementOptions(),
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfigWithHeaders(
            TEE_PUBLIC_KEY,
            "tee-key-id",
            fixtures
                .encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256)
                .setNotAfter(timestamp(Instant.now().minusSeconds(60)))
                .build()),
        "not_after is in the past");
  }

  @Test
  public void verify_futureIssuedAt_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    assertVerifyRejects(
        fixtures.endorsementOptions(),
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfigWithHeaders(
            TEE_PUBLIC_KEY,
            "tee-key-id",
            fixtures
                .encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256)
                .setIssuedAt(timestamp(Instant.now().plusSeconds(3600)))
                .build()),
        "issued_at is in the future");
  }

  @Test
  public void verify_signatureChain_succeeds() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    assertVerifyAccepts(
        fixtures.endorsementOptions(),
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfigWithSignatureChain(
            TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256));
  }

  @Test
  public void verify_notAfterAtInt64Max_succeeds() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    SignedPayload.Signature.Headers headers =
        fixtures
            .encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256)
            .setNotAfter(Timestamp.newBuilder().setSeconds(Long.MAX_VALUE).build())
            .build();
    assertVerifyAccepts(
        fixtures.endorsementOptions(),
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfigWithHeaders(TEE_PUBLIC_KEY, "tee-key-id", headers));
  }

  @Test
  public void verify_notBeforeAtInt64Max_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    SignedPayload.Signature.Headers headers =
        fixtures
            .encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256)
            .setNotBefore(Timestamp.newBuilder().setSeconds(Long.MAX_VALUE).build())
            .build();
    assertVerifyRejects(
        fixtures.endorsementOptions(),
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfigWithHeaders(TEE_PUBLIC_KEY, "tee-key-id", headers),
        "not_before is in the future");
  }

  @Test
  public void verify_notAfterAtInt64Min_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    SignedPayload.Signature.Headers headers =
        fixtures
            .encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256)
            .setNotAfter(Timestamp.newBuilder().setSeconds(Long.MIN_VALUE).build())
            .build();
    assertVerifyRejects(
        fixtures.endorsementOptions(),
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfigWithHeaders(TEE_PUBLIC_KEY, "tee-key-id", headers),
        "not_after is in the past");
  }

  @Test
  public void verify_unknownSignatureAlgorithm_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    SignedPayload.Signature.Headers headers =
        fixtures.encryptionKeyHeadersBuilder(ACCESS_POLICY_SHA256).setAlgorithmValue(9999).build();
    AccessPolicyEndorsementOptions options =
        fixtures.endorsementOptions().toBuilder()
            .setKmsVerifyingKeys(
                0,
                fixtures.endorsementOptions().getKmsVerifyingKeys(0).toBuilder()
                    .setAlgorithmValue(9999))
            .build();
    assertVerifyRejects(
        options,
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfigWithHeaders(TEE_PUBLIC_KEY, "tee-key-id", headers),
        "unsupported key algorithm");
  }

  @Test
  public void verify_unusableKeyBeforeMatchingKey_succeeds() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    Key kms = fixtures.endorsementOptions().getKmsVerifyingKeys(0);
    AccessPolicyEndorsementOptions options =
        fixtures.endorsementOptions().toBuilder()
            .clearKmsVerifyingKeys()
            .addKmsVerifyingKeys(kms.toBuilder().clearPurpose())
            .addKmsVerifyingKeys(kms)
            .build();
    assertVerifyAccepts(
        options,
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256));
  }

  @Test
  public void verify_wrongKeyBeforeMatchingKey_succeeds() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    Key kms = fixtures.endorsementOptions().getKmsVerifyingKeys(0);
    Key other = new AttestationTestFixtures().endorsementOptions().getKmsVerifyingKeys(0);
    AccessPolicyEndorsementOptions options =
        fixtures.endorsementOptions().toBuilder()
            .clearKmsVerifyingKeys()
            .addKmsVerifyingKeys(other)
            .addKmsVerifyingKeys(kms)
            .build();
    assertVerifyAccepts(
        options,
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256));
  }

  @Test
  public void verify_invalidSignatureBeforeValidSignature_succeeds() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    ConfidentialEncryptionConfig config =
        fixtures.signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256);
    SignedPayload key = config.getEncryptionKey();
    SignedPayload.Signature valid = key.getSignatures(0);
    ConfidentialEncryptionConfig configWithInvalidSignature =
        config.toBuilder()
            .setEncryptionKey(
                key.toBuilder()
                    .clearSignatures()
                    .addSignatures(
                        valid.toBuilder().setRawSignature(ByteString.copyFromUtf8("invalid")))
                    .addSignatures(valid))
            .build();
    assertVerifyAccepts(
        fixtures.endorsementOptions(),
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        configWithInvalidSignature);
  }

  @Test
  public void verify_missingEncryptionKey_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    assertVerifyRejects(
        fixtures.endorsementOptions(),
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256),
        ConfidentialEncryptionConfig.getDefaultInstance(),
        "signature verification failed");
  }

  @Test
  public void verify_missingPipelineConfiguration_throws() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    assertVerifyRejects(
        fixtures.endorsementOptions(),
        SignedEndorsements.getDefaultInstance(),
        fixtures.signedEncryptionConfig(TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256),
        "signature verification failed");
  }

  @Test
  public void verify_signatureChainWithRekorLogEntry_succeeds() throws Exception {
    AttestationTestFixtures fixtures = new AttestationTestFixtures();
    assertVerifyAccepts(
        fixtures.endorsementOptionsWithRekor(),
        fixtures.signedEndorsementsWithLogEntry(ACCESS_POLICY_SHA256),
        fixtures.signedEncryptionConfigWithSignatureChainAndLogEntry(
            TEE_PUBLIC_KEY, "tee-key-id", ACCESS_POLICY_SHA256));
  }

  // ---------------------------------------------------------------------------------------------
  // Helpers.
  // ---------------------------------------------------------------------------------------------

  /**
   * Verifies a properly endorsed configuration whose encryption key carries {@code
   * encryptionKeyHeaders}, asserts it fails, and returns the exception.
   */
  private GeneralSecurityException assertVerifyThrows(
      AttestationTestFixtures fixtures, SignedPayload.Signature.Headers encryptionKeyHeaders)
      throws Exception {
    AccessPolicyEndorsementOptions options = fixtures.endorsementOptions();
    ByteString signedEndorsements =
        fixtures.signedEndorsements(ACCESS_POLICY_SHA256).toByteString();
    ByteString encryptionConfig =
        fixtures
            .signedEncryptionConfigWithHeaders(TEE_PUBLIC_KEY, "tee-key-id", encryptionKeyHeaders)
            .toByteString();
    return assertThrows(
        GeneralSecurityException.class,
        () -> crypto.verify(options, signedEndorsements, encryptionConfig));
  }

  /** Runs {@code verify} and asserts it accepts with the expected key, key ID, and policy hash. */
  private void assertVerifyAccepts(
      AccessPolicyEndorsementOptions options,
      SignedEndorsements endorsements,
      ConfidentialEncryptionConfig config)
      throws Exception {
    VerificationResult result =
        crypto.verify(options, endorsements.toByteString(), config.toByteString());

    assertThat(result.publicKey().getKeyMaterial()).isEqualTo(TEE_PUBLIC_KEY);
    assertThat(result.keyId()).isEqualTo(ByteString.copyFromUtf8("tee-key-id"));
    assertThat(result.accessPolicySha256()).isEqualTo(ACCESS_POLICY_SHA256);
    AttestationVerificationRecord record =
        AttestationVerificationRecord.parseFrom(
            result.serializedVerificationRecord(), ExtensionRegistryLite.getEmptyRegistry());
    assertThat(record.getEncryptionKey()).isEqualTo(config.getEncryptionKey());
    assertThat(record.getPipelineConfiguration())
        .isEqualTo(endorsements.getPipelineConfiguration());
  }

  /** Runs {@code verify} and asserts it rejects with {@code expectedMessage}. */
  private void assertVerifyRejects(
      AccessPolicyEndorsementOptions options,
      SignedEndorsements endorsements,
      ConfidentialEncryptionConfig config,
      String expectedMessage) {
    GeneralSecurityException error =
        assertThrows(
            GeneralSecurityException.class,
            () -> crypto.verify(options, endorsements.toByteString(), config.toByteString()));

    assertThat(error).hasMessageThat().contains(expectedMessage);
  }

  /** Converts an {@link Instant} to a protobuf {@link Timestamp}. */
  private static Timestamp timestamp(Instant instant) {
    return Timestamp.newBuilder()
        .setSeconds(instant.getEpochSecond())
        .setNanos(instant.getNano())
        .build();
  }

  /** An {@code HPKE_X25519_SHA256_AES128_GCM} encryption key with the given raw public key. */
  private static Key hpkeKey(ByteString keyMaterial) {
    return Key.newBuilder()
        .setAlgorithm(Key.Algorithm.HPKE_X25519_SHA256_AES128_GCM)
        .setPurpose(Key.Purpose.ENCRYPT)
        .setKeyMaterial(keyMaterial)
        .build();
  }
}
