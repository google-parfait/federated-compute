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
package com.google.fcp.client.confidentialcompute.internal.encode;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static org.junit.Assert.assertThrows;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.base.Ticker;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.confidentialcompute.ConfidentialComputeCrypto.VerificationResult;
import com.google.fcp.client.confidentialcompute.EncodedPayload;
import com.google.fcp.client.confidentialcompute.LegacyFcpCompat;
import com.google.fcp.client.confidentialcompute.PayloadEncodingException;
import com.google.fcp.client.confidentialcompute.TimestampedInput;
import com.google.fcp.client.confidentialcompute.UploadTokens;
import com.google.fcp.client.confidentialcompute.WorkloadSet;
import com.google.fcp.client.confidentialcompute.WorkloadSetSelectorSpec;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent.Kind;
import com.google.fcp.client.stabilitymonitor.RecordingStabilityMonitor;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protos.fcp.confidentialcompute.BlobHeader;
import java.security.GeneralSecurityException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link ConfidentialPayloadEncoderImpl}. */
@RunWith(AndroidJUnit4.class)
public final class ConfidentialPayloadEncoderTest {
  private static final Key PUBLIC_KEY =
      Key.newBuilder()
          .setAlgorithm(Key.Algorithm.HPKE_X25519_SHA256_AES128_GCM)
          .setKeyMaterial(ByteString.copyFromUtf8("public-key"))
          .build();
  private static final ByteString KEY_ID = ByteString.copyFromUtf8("key-id");
  private static final ByteString ACCESS_POLICY_SHA256 = ByteString.copyFromUtf8("policy-sha256");
  private static final ByteString VERIFICATION_RECORD = ByteString.copyFromUtf8("record");
  private static final ByteString SIGNED_ENDORSEMENTS = ByteString.copyFromUtf8("endorsements");
  private static final ByteString ENCRYPTION_CONFIG = ByteString.copyFromUtf8("encryption-config");
  private static final AccessPolicyEndorsementOptions ENDORSEMENT_OPTIONS =
      AccessPolicyEndorsementOptions.newBuilder()
          .addKmsVerifyingKeys(Key.newBuilder().setKeyId(ByteString.copyFromUtf8("kms-key")))
          .build();
  private static final ImmutableMap<String, String> OUTPUT_VECTOR_NAMES =
      ImmutableMap.of(
          "upload_query/entry", "upload_query/entry",
          "upload_query/confidential_compute_event_time",
              "upload_query/confidential_compute_event_time");

  private static final Instant NOW = Instant.parse("2026-11-12T12:00:00Z");
  private final TimeSource timeSource = () -> NOW;
  private final FakeConfidentialComputeCrypto crypto = new FakeConfidentialComputeCrypto();
  private final List<ByteString> loggedRecords = new ArrayList<>();
  private final RecordingStabilityMonitor monitor = new RecordingStabilityMonitor();
  private final AtomicLong tickerNanos = new AtomicLong(1_000_000_000L);
  private final Ticker ticker =
      new Ticker() {
        @Override
        public long read() {
          return tickerNanos.getAndAdd(500_000_000L);
        }
      };
  private final ConfidentialPayloadEncoderImpl encoder =
      new ConfidentialPayloadEncoderImpl(
          crypto, loggedRecords::add, monitor, ticker, timeSource, directExecutor());

  @Before
  public void setUp() {
    crypto.verificationResult =
        VerificationResult.create(PUBLIC_KEY, KEY_ID, ACCESS_POLICY_SHA256, VERIFICATION_RECORD);
  }

  @Test
  public void encode_encryptsCheckpointToVerifiedKey() throws Exception {
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(
            TimestampedInput.of(ByteString.copyFromUtf8("entry-data-1"), "2026-09-14T08:00:00Z"),
            TimestampedInput.of(ByteString.copyFromUtf8("entry-data-2"), "2026-09-14T14:30:00Z"));

    EncodedPayload payload = encoder.encode(workloadSet(), ENDORSEMENT_OPTIONS, entries).get();

    assertThat(crypto.verifiedOptions).isEqualTo(ENDORSEMENT_OPTIONS);
    assertThat(crypto.verifiedSignedEndorsements).isEqualTo(SIGNED_ENDORSEMENTS);
    assertThat(crypto.verifiedEncryptionConfig).isEqualTo(ENCRYPTION_CONFIG);
    assertThat(loggedRecords).containsExactly(VERIFICATION_RECORD);
    BlobHeader blobHeader =
        BlobHeader.parseFrom(crypto.associatedData, ExtensionRegistryLite.getEmptyRegistry());
    assertThat(blobHeader.getKeyId()).isEqualTo(KEY_ID);
    assertThat(blobHeader.getAccessPolicySha256()).isEqualTo(ACCESS_POLICY_SHA256);
    assertThat(blobHeader.getBlobId().size()).isEqualTo(16);
    assertThat(crypto.recipientPublicKey).isEqualTo(PUBLIC_KEY);
    assertThat(payload.encryptedEnvelope().isEmpty()).isFalse();
    assertThat(FakeConfidentialComputeCrypto.gunzip(crypto.plaintext))
        .isEqualTo(FederatedComputeCheckpointBuilder.buildCheckpoint(entries, OUTPUT_VECTOR_NAMES));
  }

  @Test
  public void encode_logsEncodeStartedThenCompleted() throws Exception {
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("data"), "2026-09-14T08:00Z"));
    WorkloadSet workloadSet = workloadSet();

    var unused = encoder.encode(workloadSet, ENDORSEMENT_OPTIONS, entries).get();

    assertThat(monitor.kinds())
        .containsExactly(Kind.ENCODE_STARTED, Kind.ENCODE_COMPLETED)
        .inOrder();
    for (DiagnosticEvent event : monitor.events()) {
      assertThat(event.logSourceName()).isEqualTo(workloadSet.logSourceName());
      assertThat(event.fetchId()).isEqualTo(workloadSet.fetchId());
      assertThat(event.workloadSetName()).isEqualTo(workloadSet.name());
    }
    assertThat(monitor.event(Kind.ENCODE_COMPLETED).duration()).isEqualTo(Duration.ofMillis(500));
    assertThat(monitor.event(Kind.ENCODE_COMPLETED).timestamp()).isEqualTo(NOW);
  }

  @Test
  public void encode_verificationFails_logsEncodeError() {
    crypto.throwOnVerify = true;
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("data"), "2026-09-14T08:00Z"));

    ListenableFuture<EncodedPayload> future =
        encoder.encode(workloadSet(), ENDORSEMENT_OPTIONS, entries);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(monitor.kinds()).containsExactly(Kind.ENCODE_STARTED, Kind.ENCODE_ERROR).inOrder();
    assertThat(monitor.event(Kind.ENCODE_ERROR).error()).isSameInstanceAs(thrown.getCause());
  }

  @Test
  public void encode_noPlanOutputVectors_usesBareColumnNames() throws Exception {
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("data"), "2026-09-14T08:00Z"));

    var unused =
        encoder.encode(workloadSet(LegacyFcpCompat.of("")), ENDORSEMENT_OPTIONS, entries).get();

    assertThat(FakeConfidentialComputeCrypto.gunzip(crypto.plaintext))
        .isEqualTo(FederatedComputeCheckpointBuilder.buildCheckpoint(entries, ImmutableMap.of()));
  }

  @Test
  public void encode_planOutputVectorIsNotAColumn_failsWithPayloadEncodingException() {
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("data"), "2026-09-14T08:00Z"));
    WorkloadSet workloadSet =
        workloadSet(LegacyFcpCompat.of("", ImmutableMap.of("q/unknown", "q/unknown")));

    ListenableFuture<EncodedPayload> future =
        encoder.encode(workloadSet, ENDORSEMENT_OPTIONS, entries);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(PayloadEncodingException.class);
  }

  @Test
  public void encode_verificationFails_failsWithPayloadEncodingExceptionWithoutEncrypting() {
    crypto.throwOnVerify = true;
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("data"), "2026-09-14T08:00Z"));

    ListenableFuture<EncodedPayload> future =
        encoder.encode(workloadSet(), ENDORSEMENT_OPTIONS, entries);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(PayloadEncodingException.class);
    assertThat(thrown).hasCauseThat().hasCauseThat().isInstanceOf(GeneralSecurityException.class);
    assertThat(crypto.plaintext).isNull();
    assertThat(loggedRecords).isEmpty();
  }

  @Test
  public void encode_emptyVerificationRecord_isNotLogged() throws Exception {
    crypto.verificationResult =
        VerificationResult.create(PUBLIC_KEY, KEY_ID, ACCESS_POLICY_SHA256, ByteString.EMPTY);
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("data"), "2026-09-14T08:00Z"));

    var unused = encoder.encode(workloadSet(), ENDORSEMENT_OPTIONS, entries).get();

    assertThat(loggedRecords).isEmpty();
  }

  @Test
  public void encode_verificationRecordLoggerThrows_stillEncodes() throws Exception {
    ConfidentialPayloadEncoderImpl encoder =
        new ConfidentialPayloadEncoderImpl(
            crypto,
            record -> {
              throw new IllegalStateException("logger failure");
            },
            monitor,
            ticker,
            timeSource,
            directExecutor());
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("data"), "2026-09-14T08:00Z"));

    var unused = encoder.encode(workloadSet(), ENDORSEMENT_OPTIONS, entries).get();

    assertThat(crypto.plaintext).isNotNull();
  }

  @Test
  public void encode_emptyInputs_failsWithPayloadEncodingException() {
    ListenableFuture<EncodedPayload> future =
        encoder.encode(workloadSet(), ENDORSEMENT_OPTIONS, ImmutableList.of());

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(PayloadEncodingException.class);
    assertThat(thrown).hasCauseThat().hasMessageThat().contains("No inputs to encode");
  }

  @Test
  public void encode_cryptoThrowsGeneralSecurityException_failsWithPayloadEncodingException() {
    crypto.throwOnEncrypt = true;
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(
            TimestampedInput.of(ByteString.copyFromUtf8("entry-data"), "2026-09-14T08:00:00Z"));

    ListenableFuture<EncodedPayload> future =
        encoder.encode(workloadSet(), ENDORSEMENT_OPTIONS, entries);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(PayloadEncodingException.class);
    assertThat(thrown).hasCauseThat().hasCauseThat().isInstanceOf(GeneralSecurityException.class);
  }

  @Test
  public void encode_executorShutDown_returnsFailedFuture() {
    ExecutorService shutDownExecutor = Executors.newSingleThreadExecutor();
    shutDownExecutor.shutdown();
    ConfidentialPayloadEncoderImpl encoder =
        new ConfidentialPayloadEncoderImpl(
            crypto, loggedRecords::add, monitor, ticker, timeSource, shutDownExecutor);
    ImmutableList<TimestampedInput> entries =
        ImmutableList.of(TimestampedInput.of(ByteString.copyFromUtf8("data"), "2026-09-14T08:00Z"));

    ListenableFuture<EncodedPayload> future =
        encoder.encode(workloadSet(), ENDORSEMENT_OPTIONS, entries);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(RejectedExecutionException.class);
    assertThat(crypto.plaintext).isNull();
  }

  private static WorkloadSet workloadSet() {
    return workloadSet(LegacyFcpCompat.of("app:/collection", OUTPUT_VECTOR_NAMES));
  }

  private static WorkloadSet workloadSet(LegacyFcpCompat legacyFcpCompat) {
    return WorkloadSet.builder()
        .setName("workloadSet/test")
        .setSelectorSpec(WorkloadSetSelectorSpec.of(ByteString.copyFromUtf8("criteria")))
        .setLegacyFcpCompat(legacyFcpCompat)
        .setSignedEndorsements(SIGNED_ENDORSEMENTS)
        .setEncryptionConfig(ENCRYPTION_CONFIG)
        .setUploadTokens(
            UploadTokens.builder()
                .setAggregationId("agg-123")
                .setClientToken("client-789")
                .setUploadUri("https://scotty.google.com/upload")
                .setResourceName("sessions/agg-123/blobs/test")
                .setSubmitUriPrefix("https://aggregator.google.com")
                .setSubmitExtraHeaders(ImmutableMap.of())
                .build())
        .setLogSourceName("log_source")
        .setFetchId(1)
        .build();
  }
}
