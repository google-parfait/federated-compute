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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.collect.ImmutableMap;
import com.google.fcp.client.confidentialcompute.proto.ConfidentialAggregationEncryption;
import com.google.fcp.client.confidentialcompute.proto.LegacyFcpCompatProto;
import com.google.fcp.client.confidentialcompute.proto.UploadTokensProto;
import com.google.fcp.client.confidentialcompute.proto.WorkloadSetSelectorSpecProto;
import com.google.fcp.client.confidentialcompute.proto.WorkloadSetStateProto;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class WorkloadSetTest {

  private static final String TEST_WORKLOAD_SET_NAME = "telemetry/test_task";
  private static final ByteString TEST_SIGNED_ENDORSEMENTS =
      ByteString.copyFromUtf8("test_signed_endorsements");
  private static final ByteString TEST_ENCRYPTION_CONFIG =
      ByteString.copyFromUtf8("test_encryption_config");

  private static UploadTokens testTokens() {
    return UploadTokens.builder()
        .setAggregationId("agg-123")
        .setClientToken("client-tok-789")
        .setUploadUri("https://scotty.google.com/upload")
        .setResourceName("sessions/agg-123/blobs/test")
        .setSubmitUriPrefix("https://aggregator.google.com")
        .setSubmitExtraHeaders(ImmutableMap.of("X-Custom", "Val"))
        .setUploadExtraHeaders(ImmutableMap.of("X-Upload-Header", "UploadVal"))
        .build();
  }

  private static WorkloadSet testWorkloadSet() {
    return WorkloadSet.builder()
        .setName(TEST_WORKLOAD_SET_NAME)
        .setSelectorSpec(WorkloadSetSelectorSpec.of(ByteString.copyFromUtf8("criteria-bytes")))
        .setLegacyFcpCompat(LegacyFcpCompat.of("app:/test_collection"))
        .setSignedEndorsements(TEST_SIGNED_ENDORSEMENTS)
        .setEncryptionConfig(TEST_ENCRYPTION_CONFIG)
        .setUploadTokens(testTokens())
        .setLogSourceName("test_log_source")
        .setFetchId(42)
        .build();
  }

  @Test
  public void testWorkloadSetCreationAndGetters() {
    WorkloadSet workloadSet = testWorkloadSet();

    assertThat(workloadSet.name()).isEqualTo(TEST_WORKLOAD_SET_NAME);
    assertThat(workloadSet.signedEndorsements()).isEqualTo(TEST_SIGNED_ENDORSEMENTS);
    assertThat(workloadSet.encryptionConfig()).isEqualTo(TEST_ENCRYPTION_CONFIG);
    assertThat(workloadSet.selectorSpec().rawCriteria())
        .isEqualTo(ByteString.copyFromUtf8("criteria-bytes"));
    assertThat(workloadSet.legacyFcpCompat().collectionUri()).isEqualTo("app:/test_collection");
    assertThat(workloadSet.uploadTokens().aggregationId()).isEqualTo("agg-123");
    assertThat(workloadSet.uploadTokens().clientToken()).isEqualTo("client-tok-789");
    assertThat(workloadSet.uploadTokens().uploadUri())
        .isEqualTo("https://scotty.google.com/upload");
    assertThat(workloadSet.uploadTokens().resourceName()).isEqualTo("sessions/agg-123/blobs/test");
    assertThat(workloadSet.uploadTokens().submitUriPrefix())
        .isEqualTo("https://aggregator.google.com");
    assertThat(workloadSet.uploadTokens().submitExtraHeaders()).containsExactly("X-Custom", "Val");
    assertThat(workloadSet.uploadTokens().uploadExtraHeaders())
        .containsExactly("X-Upload-Header", "UploadVal");
  }

  @Test
  public void testWorkloadSetSelectorSpec() {
    WorkloadSetSelectorSpec custom = WorkloadSetSelectorSpec.of(ByteString.copyFromUtf8("hello"));
    assertThat(custom.rawCriteria()).isEqualTo(ByteString.copyFromUtf8("hello"));
  }

  @Test
  public void testLegacyFcpCompat() {
    LegacyFcpCompat custom = LegacyFcpCompat.of("app:/collection");
    assertThat(custom.collectionUri()).isEqualTo("app:/collection");
    assertThat(custom.outputVectorNames()).isEmpty();
  }

  @Test
  public void testSerializationRoundTrip_keepsOutputVectorNames() throws Exception {
    LegacyFcpCompat compat =
        LegacyFcpCompat.of(
            "app:/collection",
            ImmutableMap.of(
                "upload_query/entry", "upload_query/entry",
                "upload_query/confidential_compute_event_time",
                    "upload_query/confidential_compute_event_time"));
    WorkloadSet original =
        WorkloadSet.builder()
            .setName(TEST_WORKLOAD_SET_NAME)
            .setLegacyFcpCompat(compat)
            .setSignedEndorsements(TEST_SIGNED_ENDORSEMENTS)
            .setEncryptionConfig(TEST_ENCRYPTION_CONFIG)
            .setUploadTokens(testTokens())
            .setLogSourceName("test_log_source")
            .setFetchId(42)
            .build();

    WorkloadSet inflated = WorkloadSet.fromByteString(original.toByteString());

    assertThat(inflated.legacyFcpCompat()).isEqualTo(compat);
  }

  @Test
  public void testWorkloadSetSerializationRoundTrip() throws Exception {
    WorkloadSet original = testWorkloadSet();

    ByteString serialized = original.toByteString();
    WorkloadSet inflated = WorkloadSet.fromByteString(serialized);

    assertThat(inflated).isEqualTo(original);
    assertThat(inflated.name()).isEqualTo(TEST_WORKLOAD_SET_NAME);
    assertThat(inflated.signedEndorsements()).isEqualTo(TEST_SIGNED_ENDORSEMENTS);
    assertThat(inflated.encryptionConfig()).isEqualTo(TEST_ENCRYPTION_CONFIG);
    assertThat(inflated.selectorSpec())
        .isEqualTo(WorkloadSetSelectorSpec.of(ByteString.copyFromUtf8("criteria-bytes")));
    assertThat(inflated.legacyFcpCompat()).isEqualTo(LegacyFcpCompat.of("app:/test_collection"));
    assertThat(inflated.uploadTokens().aggregationId()).isEqualTo("agg-123");
    assertThat(inflated.uploadTokens().clientToken()).isEqualTo("client-tok-789");
    assertThat(inflated.uploadTokens().uploadUri()).isEqualTo("https://scotty.google.com/upload");
    assertThat(inflated.uploadTokens().resourceName()).isEqualTo("sessions/agg-123/blobs/test");
    assertThat(inflated.uploadTokens().submitUriPrefix())
        .isEqualTo("https://aggregator.google.com");
    assertThat(inflated.uploadTokens().submitExtraHeaders()).containsExactly("X-Custom", "Val");
    assertThat(inflated.uploadTokens().uploadExtraHeaders())
        .containsExactly("X-Upload-Header", "UploadVal");
    assertThat(inflated.toByteString()).isEqualTo(serialized);
  }

  @Test
  public void testWorkloadSetFromByteString_malformedInputThrows() {
    ByteString malformed = ByteString.copyFrom(new byte[] {(byte) 0xFF, (byte) 0xFF, (byte) 0xFF});
    assertThrows(InvalidProtocolBufferException.class, () -> WorkloadSet.fromByteString(malformed));
  }

  @Test
  public void fromByteString_missingName_throws() throws Exception {
    WorkloadSetStateProto proto = createValidWorkloadSetStateProto();
    ByteString bytes = proto.toBuilder().clearName().build().toByteString();
    InvalidProtocolBufferException e =
        assertThrows(InvalidProtocolBufferException.class, () -> WorkloadSet.fromByteString(bytes));
    assertThat(e).hasMessageThat().contains("name");
  }

  @Test
  public void fromByteString_encryptionNotSet_throws() throws Exception {
    WorkloadSetStateProto proto = createValidWorkloadSetStateProto();
    ByteString bytes = proto.toBuilder().clearEncryption().build().toByteString();
    InvalidProtocolBufferException e =
        assertThrows(InvalidProtocolBufferException.class, () -> WorkloadSet.fromByteString(bytes));
    assertThat(e).hasMessageThat().contains("confidential_aggregation");
  }

  @Test
  public void fromByteString_missingEncryptionConfig_throws() throws Exception {
    WorkloadSetStateProto proto = createValidWorkloadSetStateProto();
    ByteString bytes =
        proto.toBuilder()
            .setConfidentialAggregation(
                proto.getConfidentialAggregation().toBuilder().clearEncryptionConfig())
            .build()
            .toByteString();
    InvalidProtocolBufferException e =
        assertThrows(InvalidProtocolBufferException.class, () -> WorkloadSet.fromByteString(bytes));
    assertThat(e).hasMessageThat().contains("encryption_config");
  }

  @Test
  public void fromByteString_emptyEndorsements_succeeds() throws Exception {
    WorkloadSetStateProto proto = createValidWorkloadSetStateProto();
    ByteString bytes =
        proto.toBuilder()
            .setConfidentialAggregation(
                proto.getConfidentialAggregation().toBuilder().clearSignedEndorsements())
            .build()
            .toByteString();

    WorkloadSet inflated = WorkloadSet.fromByteString(bytes);

    assertThat(inflated.signedEndorsements()).isEmpty();
  }

  @Test
  public void fromByteString_missingUploadTokens_throws() throws Exception {
    WorkloadSetStateProto proto = createValidWorkloadSetStateProto();
    ByteString bytes = proto.toBuilder().clearUploadTokens().build().toByteString();
    InvalidProtocolBufferException e =
        assertThrows(InvalidProtocolBufferException.class, () -> WorkloadSet.fromByteString(bytes));
    assertThat(e).hasMessageThat().contains("upload_tokens");
  }

  @Test
  public void fromByteString_missingClientToken_throws() throws Exception {
    WorkloadSetStateProto proto = createValidWorkloadSetStateProto();
    ByteString bytes =
        proto.toBuilder()
            .setUploadTokens(proto.getUploadTokens().toBuilder().clearClientToken())
            .build()
            .toByteString();
    InvalidProtocolBufferException e =
        assertThrows(InvalidProtocolBufferException.class, () -> WorkloadSet.fromByteString(bytes));
    assertThat(e).hasMessageThat().contains("client_token");
  }


  private static WorkloadSetStateProto createValidWorkloadSetStateProto() {
    return WorkloadSetStateProto.newBuilder()
        .setName(TEST_WORKLOAD_SET_NAME)
        .setConfidentialAggregation(
            ConfidentialAggregationEncryption.newBuilder()
                .setSignedEndorsements(TEST_SIGNED_ENDORSEMENTS)
                .setEncryptionConfig(TEST_ENCRYPTION_CONFIG))
        .setSelectorSpec(
            WorkloadSetSelectorSpecProto.newBuilder()
                .setRawCriteria(ByteString.copyFromUtf8("criteria-bytes")))
        .setLegacyFcpCompat(
            LegacyFcpCompatProto.newBuilder().setCollectionUri("app:/test_collection"))
        .setUploadTokens(
            UploadTokensProto.newBuilder()
                .setAggregationId("agg-123")
                .setClientToken("client-tok-789")
                .setUploadUri("https://scotty.google.com/upload")
                .setResourceName("sessions/agg-123/blobs/test")
                .setSubmitUriPrefix("https://aggregator.google.com")
                .putSubmitExtraHeaders("X-Custom", "Val")
                .putUploadExtraHeaders("X-Upload-Header", "UploadVal"))
        .build();
  }

  @Test
  public void selectorSpec_toString_omitsCriteria() {
    String secretCriteria = "secret-criteria-query-text";
    WorkloadSetSelectorSpec spec =
        WorkloadSetSelectorSpec.of(ByteString.copyFromUtf8(secretCriteria));
    String specString = spec.toString();
    assertThat(specString).isEqualTo("WorkloadSetSelectorSpec{rawCriteria=26 bytes}");
    assertThat(specString).doesNotContain(secretCriteria);
  }

  @Test
  public void testToString_omitsSecrets() {
    String secretClientToken = "secret-client-token-xyz";
    String secretUploadUri = "https://scotty.google.com/secret-upload-endpoint";
    String secretResourceName = "sessions/agg-secret/blobs/secret-resource-xyz";
    String secretSubmitPrefix = "https://aggregator.google.com/secret-submit";
    String secretEndorsements = "secret-endorsements-123";
    String secretEncryptionConfig = "secret-encryption-config-456";
    String secretCriteria = "secret-criteria-789";

    UploadTokens tokens =
        UploadTokens.builder()
            .setAggregationId("agg-safe-id")
            .setClientToken(secretClientToken)
            .setUploadUri(secretUploadUri)
            .setResourceName(secretResourceName)
            .setSubmitUriPrefix(secretSubmitPrefix)
            .build();

    WorkloadSet workloadSet =
        WorkloadSet.builder()
            .setName("safe-workload-name")
            .setSelectorSpec(WorkloadSetSelectorSpec.of(ByteString.copyFromUtf8(secretCriteria)))
            .setLegacyFcpCompat(LegacyFcpCompat.of("app:/safe-collection"))
            .setSignedEndorsements(ByteString.copyFromUtf8(secretEndorsements))
            .setEncryptionConfig(ByteString.copyFromUtf8(secretEncryptionConfig))
            .setUploadTokens(tokens)
            .setLogSourceName("log_source")
            .setFetchId(1)
            .build();

    String workloadSetString = workloadSet.toString();
    assertThat(workloadSetString).isEqualTo("WorkloadSet{name=safe-workload-name}");
    assertThat(workloadSetString).doesNotContain(secretEndorsements);
    assertThat(workloadSetString).doesNotContain(secretEncryptionConfig);
    assertThat(workloadSetString).doesNotContain(secretClientToken);
    assertThat(workloadSetString).doesNotContain(secretResourceName);
    assertThat(workloadSetString).doesNotContain(secretCriteria);

    String tokensString = tokens.toString();
    assertThat(tokensString).isEqualTo("UploadTokens{aggregationId=agg-safe-id, <redacted>}");
    assertThat(tokensString).doesNotContain(secretClientToken);
    assertThat(tokensString).doesNotContain(secretUploadUri);
    assertThat(tokensString).doesNotContain(secretResourceName);
    assertThat(tokensString).doesNotContain(secretSubmitPrefix);
  }
}
