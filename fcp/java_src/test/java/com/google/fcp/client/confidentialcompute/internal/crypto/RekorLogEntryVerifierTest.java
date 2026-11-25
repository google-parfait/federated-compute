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

import com.google.common.collect.ImmutableList;
import com.google.common.io.BaseEncoding;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.intelligence.fcp.confidentialcompute.RekorLogEntry;
import com.google.protobuf.ByteString;
import java.security.GeneralSecurityException;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/**
 * Tests for {@link RekorLogEntryVerifier}, ported from {@code
 * fcp/confidentialcompute/payload_transparency/rekor_test.cc}. They use that test's 11-leaf Merkle
 * tree (root {@code 8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=}) and its even and odd signing
 * keys, so the leaf bodies and audit paths are fixed data; only the checkpoint is signed afresh.
 */
@RunWith(RobolectricTestRunner.class)
public final class RekorLogEntryVerifierTest {

  private static final String ROOT_11_B64 = "8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=";

  private static final String LEAF_PREFIX =
      "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\"hash\":{\"algorithm\":"
          + "\"sha256\",\"value\":\"";
  private static final String EVEN_PUBLIC_KEY_PEM_B64 =
      "LS0tLS1CRUdJTiBQVUJMSUMgS0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFOC8rNTd"
          + "mWWh1YUM4Vit1TUZPOWtkVEIzM05TWgo1cS9IYk1XY01FeS8wZGNYRnlXbmFqZi9Kdy8rQTlwa3RVc1h0K2k2Wit5"
          + "TXhZOGp1YXRIaENPM3BBPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t";
  private static final String ODD_PUBLIC_KEY_PEM_B64 =
      "LS0tLS1CRUdJTiBQVUJMSUMgS0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFWlhJaFN"
          + "CeFEyalhSZzhHazlMUjliZjZGM28yYQo2L1VnYitOS1dWR3h3S3YxZU92UDYzTWlOYVNaQTY5VjM0eGtrYUpsVm1s"
          + "MFJzMXVMZkwxWERtL0hnPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t";

  /** Builds a canonical hashedrekord body, as Rekor serves it, from its three variable fields. */
  private static String leaf(String dataHash, String signatureB64, String publicKeyB64) {
    return LEAF_PREFIX
        + dataHash
        + "\"}},\"signature\":{\"content\":\""
        + signatureB64
        + "\",\"publicKey\":{\"content\":\""
        + publicKeyB64
        + "\"}}}}";
  }

  private static final String LEAF_0 =
      leaf(
          "20e325f06280f9d0d193fed01a0eda5bef79063f2e602d93e3605cbe825d96ad",
          "MEYCIQCIBNw3Y/epRrQws36IqLbm2P37wjcrHOeMtngWJTjYwQIhAJudknwUm3+dGFpsaBW9ytHioRARkE50vp37MKi3ODwN",
          EVEN_PUBLIC_KEY_PEM_B64);
  private static final String LEAF_5 =
      leaf(
          "5df183a7928a0b9a8a923c39009e89847ab5f53d07547a362bd2b30e26ee2e16",
          "MEUCIQDxzsRvijGh5n4EykMBO6XH2je/0FxRrPNHnHg3Z33DlQIgcgwi5MXVSxf0zFOSV87Y3qekrz210ac66IAR/KOfjZs=",
          ODD_PUBLIC_KEY_PEM_B64);
  private static final String LEAF_7 =
      leaf(
          "3fc6f659908e60026f20aeb6f1a90a14cacb8a27d870aec882a0529f2c829bb2",
          "MEUCIGTUxt6wDhOP1dQFJ30n5HbgQ6uv6bdEaHYvo+rAzFshAiEAmqA5j6ZnP40/guzZLcnZhvOUb3fngxZk5FuXPF/kDdo=",
          ODD_PUBLIC_KEY_PEM_B64);
  private static final String LEAF_8 =
      leaf(
          "3779790b00cf4dd355ba895adf36838158baa398f0e70fce7018503dbade253d",
          "MEUCIFy+2n1e9Cp5SFxpR8etJa7i/NpUHZhA8Zk7yqCGgB1EAiEAl9HBuEum6BXLyXmb7x3mS1cHU3GI2FuVpwH6ZKItXcg=",
          EVEN_PUBLIC_KEY_PEM_B64);
  private static final String LEAF_10_DATA_HASH =
      "b789e63ed0a8dd152a5b020b53e0c9d7a2522b93702c255d2847575ebea98a92";
  private static final String LEAF_10_SIGNATURE_B64 =
      "MEYCIQDyifHM2YsfHTnw+CZoEZeO9ELWadxfORWL0P6dCh/OPQIhAIklmtzK/g0+cwSjRviNp4LFn5ka2AvMjA4bvJJXMMau";
  private static final String LEAF_10 =
      leaf(LEAF_10_DATA_HASH, LEAF_10_SIGNATURE_B64, EVEN_PUBLIC_KEY_PEM_B64);

  private static final ImmutableList<String> LEAF_10_AUDIT_PATH =
      ImmutableList.of(
          "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
          "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe");

  /** The key used to sign even-index log entries in rekor_test.cc. */
  private static Key evenVerifyingKey() {
    return AttestationTestFixtures.verifyingKeyFromHex(
        "04f3ffb9edf621b9a0bc57eb8c14ef64753077dcd499e6afc76cc59c304cbfd1d7171725"
            + "a76a37ff270ffe03da64b54b17b7e8ba67ec8cc58f23b9ab478423b7a4",
        "even");
  }

  /** The key used to sign odd-index log entries in rekor_test.cc. */
  private static Key oddVerifyingKey() {
    return AttestationTestFixtures.verifyingKeyFromHex(
        "04657221481c50da35d183c1a4f4b47d6dfe85de8d9aebf5206fe34a5951b1c0abf578eb"
            + "cfeb732235a49903af55df8c6491a26556697446cd6e2df2f55c39bf1e",
        "odd");
  }

  /** A log entry for {@code body} at {@code logIndex} in a tree of {@code treeSize} leaves. */
  private static RekorLogEntry.Builder logEntry(
      String body, long logIndex, long treeSize, List<String> hashesHex) {
    RekorLogEntry.Builder builder =
        RekorLogEntry.newBuilder()
            .setBody(ByteString.copyFromUtf8(body))
            .setLogIndex(logIndex)
            .setTreeSize(treeSize)
            .setCheckpointOrigin("origin");
    for (String hash : hashesHex) {
      builder.addHashes(ByteString.copyFrom(BaseEncoding.base16().lowerCase().decode(hash)));
    }
    return builder;
  }

  /** Signs the checkpoint for {@code logEntry} with a fresh Rekor key and returns both. */
  private static RekorLogEntryWithKey signed(RekorLogEntry.Builder logEntry, String checkpoint)
      throws GeneralSecurityException {
    AttestationTestFixtures.SignedCheckpoint signedCheckpoint =
        AttestationTestFixtures.signCheckpoint(checkpoint);
    return new RekorLogEntryWithKey(
        logEntry
            .setCheckpointSignature(ByteString.copyFrom(signedCheckpoint.signature))
            .setCheckpointSignatureKeyId(signedCheckpoint.key.getKeyId())
            .build(),
        signedCheckpoint.key);
  }

  /** A log entry together with the Rekor key that signed its checkpoint. */
  private static final class RekorLogEntryWithKey {
    final RekorLogEntry logEntry;
    final Key rekorKey;

    RekorLogEntryWithKey(RekorLogEntry logEntry, Key rekorKey) {
      this.logEntry = logEntry;
      this.rekorKey = rekorKey;
    }
  }

  /**
   * Verifies {@code entry} as a log entry for a signature over {@code data} by {@code
   * verifyingKey}.
   */
  private static void verifyRekorLogEntry(RekorLogEntryWithKey entry, Key verifyingKey, String data)
      throws GeneralSecurityException {
    RekorLogEntryVerifier.verify(
        entry.logEntry,
        ImmutableList.of(verifyingKey),
        ImmutableList.of(entry.rekorKey),
        data.getBytes(UTF_8));
  }

  /** Asserts that {@link #verifyRekorLogEntry} fails and returns the exception. */
  private static GeneralSecurityException assertRekorLogEntryThrows(
      RekorLogEntryWithKey entry, Key verifyingKey, String data) {
    return assertThrows(
        GeneralSecurityException.class, () -> verifyRekorLogEntry(entry, verifyingKey, data));
  }

  // Leaf 0 is on the far left of the size 8 perfect subtree.
  @Test
  public void verifyRekorLogEntry_leaf0() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(
                LEAF_0,
                0,
                11,
                ImmutableList.of(
                    "92fb1f3d84bef84d8f14f7055c3c0a9d0a882811e51489f6161fb995160193f0",
                    "dfa09bb23f87a6729fd0728a13b22141adb3da53c706b4fac0ac8edf19d1fafa",
                    "c5ebc562f49755eccc26679d2aa22a700c76311a2e97b0be0c3c3e7a5c502786",
                    "23e18ca02de317968435d21756f6ff8189abbb44b864f5357264d100d4e44a97")),
            "origin\n11\n" + ROOT_11_B64 + "\n");

    verifyRekorLogEntry(entry, evenVerifyingKey(), "leaf 0");
  }

  // Leaf 5 is in the middle of the size 8 perfect subtree.
  @Test
  public void verifyRekorLogEntry_leaf5() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(
                LEAF_5,
                5,
                11,
                ImmutableList.of(
                    "97fd43db9b236927e7921731ed75c06796047ab78b3e64b0ae0d50f96193d4e8",
                    "e7f2620cfc19210f52498d84198bd3eff6dee1c8807710e3cae284d1d0f68a2f",
                    "f625e204b44ef052ae52ec92695dd9648b83a9b601ba6cf76205124d522a8e15",
                    "23e18ca02de317968435d21756f6ff8189abbb44b864f5357264d100d4e44a97")),
            "origin\n11\n" + ROOT_11_B64 + "\n");

    verifyRekorLogEntry(entry, oddVerifyingKey(), "leaf 5");
  }

  // Leaf 7 is on the right of the size 8 perfect subtree.
  @Test
  public void verifyRekorLogEntry_leaf7() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(
                LEAF_7,
                7,
                11,
                ImmutableList.of(
                    "5642602157cc4d5b6f09dfe6c3122c12a620d922dc1e8579dfa52c7a7fec0668",
                    "d5d85c38cf23702cc8015fc104abbfe6c1085a0b8ad031d4c679fb62492e9197",
                    "f625e204b44ef052ae52ec92695dd9648b83a9b601ba6cf76205124d522a8e15",
                    "23e18ca02de317968435d21756f6ff8189abbb44b864f5357264d100d4e44a97")),
            "origin\n11\n" + ROOT_11_B64 + "\n");

    verifyRekorLogEntry(entry, oddVerifyingKey(), "leaf 7");
  }

  // Leaf 8 is inside the size 2 perfect subtree.
  @Test
  public void verifyRekorLogEntry_leaf8() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(
                LEAF_8,
                8,
                11,
                ImmutableList.of(
                    "8e70aedc03a3784c0fa0988dde490ab7b44dec3d4894c52d35d660e5e947d806",
                    "c0e6be9a317ce82b482b91ac12c7a90555208bdfc6741f5f67207e2a7593ecc6",
                    "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe")),
            "origin\n11\n" + ROOT_11_B64 + "\n");

    verifyRekorLogEntry(entry, evenVerifyingKey(), "leaf 8");
  }

  // Leaf 10 is in the size 1 perfect subtree.
  @Test
  public void verifyRekorLogEntry_leaf10() throws Exception {
    RekorLogEntryWithKey entry =
        signed(logEntry(LEAF_10, 10, 11, LEAF_10_AUDIT_PATH), "origin\n11\n" + ROOT_11_B64 + "\n");

    verifyRekorLogEntry(entry, evenVerifyingKey(), "leaf 10");
  }

  @Test
  public void verifyRekorLogEntry_singleElementTree() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(LEAF_0, 0, 1, ImmutableList.of()),
            "origin\n1\nNfgamq7LbfhbNrVZ62YMK71b5igmnDUrZvJcmxY9+m4=\n");

    verifyRekorLogEntry(entry, evenVerifyingKey(), "leaf 0");
  }

  @Test
  public void verifyRekorLogEntry_checkpointOtherContents() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(LEAF_10, 10, 11, LEAF_10_AUDIT_PATH)
                .addCheckpointOtherContents("extra 1")
                .addCheckpointOtherContents("extra 2"),
            "origin\n11\n" + ROOT_11_B64 + "\nextra 1\nextra 2\n");

    verifyRekorLogEntry(entry, evenVerifyingKey(), "leaf 10");
  }

  @Test
  public void verifyRekorLogEntry_unsupportedLogEntryKind_throws() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(LEAF_10.replace("hashedrekord", "dsse"), 10, 11, LEAF_10_AUDIT_PATH),
            "origin\n11\nLCHRxFbKsvf4Jzr+px0Pqd4reWgwC+dRl4VcdjkLBQk=\n");

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, evenVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("unsupported log entry kind");
  }

  @Test
  public void verifyRekorLogEntry_missingDataHash_throws() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(
                LEAF_10.replace(",\"value\":\"" + LEAF_10_DATA_HASH + "\"", ""),
                10,
                11,
                LEAF_10_AUDIT_PATH),
            "origin\n11\n/VSaEwzcUfM0Be3XuW7MM8QQmOOMc2/C6GFQ78unNpE=\n");

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, evenVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("log entry is missing data hash");
  }

  @Test
  public void verifyRekorLogEntry_invalidDataHashEncoding_throws() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(
                LEAF_10.replace(LEAF_10_DATA_HASH, "invalid base16"), 10, 11, LEAF_10_AUDIT_PATH),
            "origin\n11\nN84K5iEwOrqXSfGdSM/cJEjE4FnBrugXnDwzFpCXG/U=\n");

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, evenVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("data hash is not a valid base16 string");
  }

  @Test
  public void verifyRekorLogEntry_mismatchedDataHash_throws() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(
                LEAF_10.replace(
                    LEAF_10_DATA_HASH,
                    "0000000000000000000000000000000000000000000000000000000000000000"),
                10,
                11,
                LEAF_10_AUDIT_PATH),
            "origin\n11\n3BtuumfZR9RqgbP0NEvON8vVk5XcHeNkE+8FcmzPDhk=\n");

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, evenVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("log entry data hash does not match payload");
  }

  @Test
  public void verifyRekorLogEntry_missingSignature_throws() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(
                LEAF_10.replace("\"content\":\"" + LEAF_10_SIGNATURE_B64 + "\",", ""),
                10,
                11,
                LEAF_10_AUDIT_PATH),
            "origin\n11\n9XUVBPhvc+eDGGyhi5wymRxas5oPyO2BRSgdsouEoSQ=\n");

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, evenVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("log entry is missing signature");
  }

  @Test
  public void verifyRekorLogEntry_invalidSignatureEncoding_throws() throws Exception {
    RekorLogEntryWithKey entry =
        signed(
            logEntry(
                LEAF_10.replace(LEAF_10_SIGNATURE_B64, "invalid base64!"),
                10,
                11,
                LEAF_10_AUDIT_PATH),
            "origin\n11\n2pObl5Tee5UwYch1FsUM1gHihTLC3hg94LdqSRqDdEA=\n");

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, evenVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("signature is not a valid base64 string");
  }

  @Test
  public void verifyRekorLogEntry_mismatchedSignature_throws() throws Exception {
    // Leaf 0's signature in place of leaf 10's.
    RekorLogEntryWithKey entry =
        signed(
            logEntry(
                LEAF_10.replace(
                    LEAF_10_SIGNATURE_B64,
                    "MEYCIQCIBNw3Y/epRrQws36IqLbm2P37wjcrHOeMtngWJTjYwQIhAJudknwUm3+dGFpsaBW9ytHioRARkE50vp37MKi3ODwN"),
                10,
                11,
                LEAF_10_AUDIT_PATH),
            "origin\n11\nQyAqzby0n2ekpgChOfnecBtbiBQKPO5qFKMMx9g4dZQ=\n");

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, evenVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("log entry signature is invalid");
  }

  @Test
  public void verifyRekorLogEntry_wrongVerifyingKey_throws() throws Exception {
    RekorLogEntryWithKey entry =
        signed(logEntry(LEAF_10, 10, 11, LEAF_10_AUDIT_PATH), "origin\n11\n" + ROOT_11_B64 + "\n");

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, oddVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("log entry signature is invalid");
  }

  @Test
  public void verifyRekorLogEntry_leafIndexTooLarge_throws() throws Exception {
    String checkpoint = "origin\n11\n" + ROOT_11_B64 + "\n";

    GeneralSecurityException atTreeSize =
        assertRekorLogEntryThrows(
            signed(logEntry(LEAF_10, 11, 11, LEAF_10_AUDIT_PATH), checkpoint),
            evenVerifyingKey(),
            "leaf 10");
    GeneralSecurityException pastTreeSize =
        assertRekorLogEntryThrows(
            signed(logEntry(LEAF_10, 12, 11, LEAF_10_AUDIT_PATH), checkpoint),
            evenVerifyingKey(),
            "leaf 10");

    assertThat(atTreeSize).hasMessageThat().contains("leaf index larger than tree size");
    assertThat(pastTreeSize).hasMessageThat().contains("leaf index larger than tree size");
  }

  @Test
  public void verifyRekorLogEntry_wrongNumberOfHashes_throws() throws Exception {
    String checkpoint = "origin\n11\n" + ROOT_11_B64 + "\n";

    GeneralSecurityException tooMany =
        assertRekorLogEntryThrows(
            signed(
                logEntry(
                    LEAF_10,
                    10,
                    11,
                    ImmutableList.<String>builder()
                        .addAll(LEAF_10_AUDIT_PATH)
                        .add("0000000000000000000000000000000000000000000000000000000000000000")
                        .build()),
                checkpoint),
            evenVerifyingKey(),
            "leaf 10");
    GeneralSecurityException tooFew =
        assertRekorLogEntryThrows(
            signed(logEntry(LEAF_10, 10, 11, ImmutableList.of()), checkpoint),
            evenVerifyingKey(),
            "leaf 10");

    assertThat(tooMany).hasMessageThat().contains("unexpected number of hashes");
    assertThat(tooFew).hasMessageThat().contains("unexpected number of hashes");
  }

  @Test
  public void verifyRekorLogEntry_mismatchedCheckpointSignature_throws() throws Exception {
    RekorLogEntryWithKey signedEntry =
        signed(logEntry(LEAF_10, 10, 11, LEAF_10_AUDIT_PATH), "origin\n11\n" + ROOT_11_B64 + "\n");
    byte[] corrupted = signedEntry.logEntry.getCheckpointSignature().toByteArray();
    corrupted[0] ^= (byte) 0xff;
    RekorLogEntryWithKey entry =
        new RekorLogEntryWithKey(
            signedEntry.logEntry.toBuilder()
                .setCheckpointSignature(ByteString.copyFrom(corrupted))
                .build(),
            signedEntry.rekorKey);

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, evenVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("checkpoint signature is invalid");
  }

  @Test
  public void verifyRekorLogEntry_wrongRootHashInCheckpoint_throws() throws Exception {
    // The checkpoint is signed over a different root, so the recomputed root does not verify.
    RekorLogEntryWithKey entry =
        signed(
            logEntry(LEAF_10, 10, 11, LEAF_10_AUDIT_PATH),
            "origin\n11\nNfgamq7LbfhbNrVZ62YMK71b5igmnDUrZvJcmxY9+m4=\n");

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, evenVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("checkpoint signature is invalid");
  }

  @Test
  public void verifyRekorLogEntry_noMatchingRekorKey_throws() throws Exception {
    RekorLogEntryWithKey signedEntry =
        signed(logEntry(LEAF_10, 10, 11, LEAF_10_AUDIT_PATH), "origin\n11\n" + ROOT_11_B64 + "\n");
    RekorLogEntryWithKey entry =
        new RekorLogEntryWithKey(
            signedEntry.logEntry.toBuilder()
                .setCheckpointSignatureKeyId(
                    signedEntry
                        .logEntry
                        .getCheckpointSignatureKeyId()
                        .concat(ByteString.copyFromUtf8("X")))
                .build(),
            signedEntry.rekorKey);

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, evenVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("checkpoint signature is invalid");
  }

  @Test
  public void verifyRekorLogEntry_treeSizeAboveInt64Max() throws Exception {
    String siblingHex = "1111111111111111111111111111111111111111111111111111111111111111";
    byte[] sibling = BaseEncoding.base16().lowerCase().decode(siblingHex);
    byte[] leafHash = AttestationTestFixtures.sha256(new byte[] {0x00}, LEAF_10.getBytes(UTF_8));
    byte[] root = AttestationTestFixtures.sha256(new byte[] {0x01}, sibling, leafHash);
    String checkpoint = "origin\n9223372036854775809\n" + BaseEncoding.base64().encode(root) + "\n";
    RekorLogEntryWithKey entry =
        signed(
            logEntry(LEAF_10, Long.MIN_VALUE, Long.MIN_VALUE + 1, ImmutableList.of(siblingHex)),
            checkpoint);

    verifyRekorLogEntry(entry, evenVerifyingKey(), "leaf 10");
  }

  @Test
  public void verifyRekorLogEntry_leafIndexAboveInt64MaxAndTreeSize_throws() throws Exception {
    String siblingHex = "1111111111111111111111111111111111111111111111111111111111111111";
    String checkpoint = "origin\n9223372036854775809\n" + ROOT_11_B64 + "\n";
    RekorLogEntryWithKey entry =
        signed(
            logEntry(LEAF_10, -1L, Long.MIN_VALUE + 1, ImmutableList.of(siblingHex)), checkpoint);

    GeneralSecurityException e = assertRekorLogEntryThrows(entry, evenVerifyingKey(), "leaf 10");

    assertThat(e).hasMessageThat().contains("leaf index larger than tree size");
  }

  @Test
  public void verifyRekorLogEntry_whitespaceInSignature_isIgnored() throws Exception {
    String body = LEAF_0.replace("MEYCIQCIBN", "MEYCI\nQCIBN");
    assertThat(body).isNotEqualTo(LEAF_0);
    byte[] root = AttestationTestFixtures.sha256(new byte[] {0x00}, body.getBytes(UTF_8));
    String checkpoint = "origin\n1\n" + BaseEncoding.base64().encode(root) + "\n";
    RekorLogEntryWithKey entry = signed(logEntry(body, 0, 1, ImmutableList.of()), checkpoint);

    verifyRekorLogEntry(entry, evenVerifyingKey(), "leaf 0");
  }
}
