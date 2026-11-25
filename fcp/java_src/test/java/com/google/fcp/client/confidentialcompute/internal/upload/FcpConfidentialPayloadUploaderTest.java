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
package com.google.fcp.client.confidentialcompute.internal.upload;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import androidx.datastore.guava.GuavaDataStore;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.base.Ticker;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.fcp.client.FakeTimeSource;
import com.google.fcp.client.confidentialcompute.EncodedPayload;
import com.google.fcp.client.confidentialcompute.UploadAbortedException;
import com.google.fcp.client.confidentialcompute.UploadTokens;
import com.google.fcp.client.confidentialcompute.WorkloadSet;
import com.google.fcp.client.confidentialcompute.internal.fetch.AggregationAbortSender;
import com.google.fcp.client.confidentialcompute.internal.http.FakeHttpUrlConnectionEngine;
import com.google.fcp.client.confidentialcompute.internal.http.FakeHttpUrlConnectionEngine.RecordedRequest;
import com.google.fcp.client.confidentialcompute.internal.swor.SworStore;
import com.google.fcp.client.confidentialcompute.internal.swor.SworTracker;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent.Kind;
import com.google.fcp.client.stabilitymonitor.RecordingStabilityMonitor;
import com.google.intelligence.fcp.client.ProdDiagCode;
import com.google.internal.federatedcompute.v1.AbortConfidentialAggregationRequest;
import com.google.internal.federatedcompute.v1.SubmitConfidentialAggregationResultRequest;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.rpc.Code;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class FcpConfidentialPayloadUploaderTest {

  private static final String TEST_WORKLOAD_SET_NAME = "telemetry/uploader_test";
  private static final String TEST_UPLOAD_URI = "https://scotty.google.com";
  private static final String TEST_SUBMIT_URI = "https://coordinator.google.com";
  private static final String TEST_AGGREGATION_ID = "agg-001";
  private static final String TEST_CLIENT_TOKEN = "tok-001";
  private static final String TEST_RESOURCE_NAME = "sessions/agg-001/blobs/b-001";
  private static final String TEST_API_KEY = "test-api-key";

  @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

  private final FakeTimeSource timeSource = new FakeTimeSource();
  private FakeHttpUrlConnectionEngine fakeEngine;
  private SworStore sworStore;
  private FcpConfidentialPayloadUploader payloadUploader;

  @Before
  public void setUp() {
    File sworFile = new File(temporaryFolder.getRoot(), "swor.pb");
    sworStore =
        new SworStore(new GuavaDataStore.Builder<>(SworStore.serializer(), () -> sworFile).build());
    fakeEngine = new FakeHttpUrlConnectionEngine();
    fakeEngine.setDefaultResponse(200, new byte[0]);
    payloadUploader =
        new FcpConfidentialPayloadUploader(
            FcpConfidentialPayloadUploaderDeps.builder()
                .setHttpEngine(fakeEngine)
                .setExecutor(directExecutor())
                .setApiKey(TEST_API_KEY)
                .setRequestHeaders(ImmutableMap.of("X-Request-Header", "ReqVal"))
                .setSworTracker(SworTracker.DISABLED)
                .build());
  }

  private static WorkloadSet createTestWorkloadSet() {
    UploadTokens tokens =
        UploadTokens.builder()
            .setAggregationId(TEST_AGGREGATION_ID)
            .setClientToken(TEST_CLIENT_TOKEN)
            .setUploadUri(TEST_UPLOAD_URI)
            .setResourceName(TEST_RESOURCE_NAME)
            .setSubmitUriPrefix(TEST_SUBMIT_URI)
            .setUploadExtraHeaders(ImmutableMap.of("X-Upload-Extra", "UploadVal"))
            .setSubmitExtraHeaders(ImmutableMap.of("X-Submit-Extra", "SubmitVal"))
            .build();
    return createTestWorkloadSet(tokens);
  }

  private static WorkloadSet createTestWorkloadSet(UploadTokens tokens) {
    return WorkloadSet.builder()
        .setName(TEST_WORKLOAD_SET_NAME)
        .setSignedEndorsements(ByteString.EMPTY)
        .setEncryptionConfig(ByteString.copyFromUtf8("encryption-config"))
        .setUploadTokens(tokens)
        .setLogSourceName("log_source")
        .setFetchId(42)
        .build();
  }

  private static EncodedPayload createTestPayload(ByteString data) {
    return EncodedPayload.create(data);
  }

  @Test
  public void upload_success() throws Exception {
    WorkloadSet workloadSet = createTestWorkloadSet();
    ByteString payloadBytes = ByteString.copyFromUtf8("encrypted-test-data");
    EncodedPayload payload = createTestPayload(payloadBytes);

    ListenableFuture<Void> future = payloadUploader.upload(workloadSet, payload);
    future.get();

    assertThat(fakeEngine.requests()).hasSize(2);

    RecordedRequest uploadConn = fakeEngine.requests().get(0);
    assertThat(uploadConn.url())
        .isEqualTo(
            "https://scotty.google.com/upload/v1/media/sessions/agg-001/blobs/b-001?upload_protocol=raw");
    assertThat(uploadConn.headers()).containsEntry("Content-Type", "application/octet-stream");
    assertThat(uploadConn.headers()).doesNotContainKey("Accept");
    assertThat(uploadConn.headers()).doesNotContainKey("x-goog-blob-header");
    assertThat(uploadConn.headers()).containsEntry("x-goog-api-key", TEST_API_KEY);
    assertThat(uploadConn.headers()).containsEntry("X-Upload-Extra", "UploadVal");
    assertThat(uploadConn.headers()).containsEntry("X-Request-Header", "ReqVal");
    assertThat(uploadConn.body()).isEqualTo(payloadBytes.toByteArray());

    RecordedRequest submitConn = fakeEngine.requests().get(1);
    assertThat(submitConn.url())
        .isEqualTo(
            "https://coordinator.google.com/v1/confidentialaggregations/agg-001/clients/tok-001:submit?%24alt=proto");
    assertThat(submitConn.headers()).containsEntry("Content-Type", "application/x-protobuf");
    // No Accept header: like the C++ client, the proto encoding is selected by $alt=proto.
    assertThat(submitConn.headers()).doesNotContainKey("Accept");
    assertThat(submitConn.headers()).containsEntry("x-goog-api-key", TEST_API_KEY);
    assertThat(submitConn.headers()).containsEntry("X-Submit-Extra", "SubmitVal");
    assertThat(submitConn.headers()).containsEntry("X-Request-Header", "ReqVal");

    SubmitConfidentialAggregationResultRequest submitRequest =
        SubmitConfidentialAggregationResultRequest.parseFrom(
            submitConn.body(), ExtensionRegistryLite.getEmptyRegistry());
    assertThat(submitRequest.getResourceName()).isEqualTo(TEST_RESOURCE_NAME);
    assertThat(submitRequest.getAggregationId()).isEmpty();
    assertThat(submitRequest.getClientToken()).isEmpty();
  }

  @Test
  public void upload_sworTracker_recordsUploadTime() throws Exception {
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));

    sworUploader().upload(createTestWorkloadSet(), payload).get();

    assertThat(sworStore.lastUpload(TEST_WORKLOAD_SET_NAME).get()).hasValue(timeSource.now());
  }

  @Test
  public void upload_sworTrackerAndOutcomeUnknown_keepsUploadTime() throws Exception {
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.respondTo("/upload/v1/media/", 503, "Service Unavailable".getBytes(UTF_8));

    ListenableFuture<Void> future = sworUploader().upload(createTestWorkloadSet(), payload);

    assertThrows(ExecutionException.class, future::get);
    assertThat(sworStore.lastUpload(TEST_WORKLOAD_SET_NAME).get()).hasValue(timeSource.now());
  }

  @Test
  public void upload_sworTrackerAndBlobAbortedByServer_restoresPreviousUploadTime()
      throws Exception {
    Instant previous = timeSource.now();
    sworStore.recordUpload(TEST_WORKLOAD_SET_NAME, previous).get();
    timeSource.advance(Duration.ofDays(2));
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.respondTo("/upload/v1/media/", 409, "ABORTED".getBytes(UTF_8));

    ListenableFuture<Void> future = sworUploader().upload(createTestWorkloadSet(), payload);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(UploadAbortedException.class);
    assertThat(sworStore.lastUpload(TEST_WORKLOAD_SET_NAME).get()).hasValue(previous);
  }

  @Test
  public void upload_blobRefusedByServer_restoresSworSendsNoAbortAndFailsWithUploadAborted()
      throws Exception {
    Instant previous = timeSource.now();
    sworStore.recordUpload(TEST_WORKLOAD_SET_NAME, previous).get();
    timeSource.advance(Duration.ofDays(2));
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.respondTo("/upload/v1/media/", 403, "PERMISSION_DENIED".getBytes(UTF_8));

    ListenableFuture<Void> future = sworUploader().upload(createTestWorkloadSet(), payload);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(UploadAbortedException.class);
    assertThat(sworStore.lastUpload(TEST_WORKLOAD_SET_NAME).get()).hasValue(previous);
    assertThat(fakeEngine.requests()).hasSize(1);
  }

  @Test
  public void upload_sworRecordFails_makesNoNetworkCallAndFailsWithUploadAbortedException()
      throws Exception {
    File notADirectory = temporaryFolder.newFile("not_a_directory");
    SworStore unwritableStore =
        new SworStore(
            new GuavaDataStore.Builder<>(
                    SworStore.serializer(), () -> new File(notADirectory, "swor.pb"))
                .build());
    FcpConfidentialPayloadUploader uploader =
        new FcpConfidentialPayloadUploader(
            FcpConfidentialPayloadUploaderDeps.builder()
                .setHttpEngine(fakeEngine)
                .setExecutor(directExecutor())
                .setApiKey(TEST_API_KEY)
                .setSworTracker(new SworTracker(unwritableStore, timeSource))
                .build());
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));

    ListenableFuture<Void> future = uploader.upload(createTestWorkloadSet(), payload);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(UploadAbortedException.class);
    assertThat(fakeEngine.requests()).isEmpty();
  }

  @Test
  public void upload_executorShutDown_returnsFailedFuture() {
    ExecutorService shutDownExecutor = Executors.newSingleThreadExecutor();
    shutDownExecutor.shutdown();
    FcpConfidentialPayloadUploader uploader =
        new FcpConfidentialPayloadUploader(
            FcpConfidentialPayloadUploaderDeps.builder()
                .setHttpEngine(fakeEngine)
                .setExecutor(shutDownExecutor)
                .setApiKey(TEST_API_KEY)
                .setSworTracker(SworTracker.DISABLED)
                .build());
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));

    ListenableFuture<Void> future = uploader.upload(createTestWorkloadSet(), payload);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(RejectedExecutionException.class);
    assertThat(fakeEngine.requests()).isEmpty();
  }

  @Test
  public void upload_uploadFailure_sendsAbortAndFailsWithOriginalException() throws Exception {
    WorkloadSet workloadSet = createTestWorkloadSet();
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.respondTo("/upload/v1/media/", 503, "Service Unavailable".getBytes(UTF_8));

    ListenableFuture<Void> future = payloadUploader.upload(workloadSet, payload);
    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    // The abort is acknowledged (default 200), but the server does not discard a stored blob on
    // abort, so the outcome stays unknown and the original failure is surfaced.
    assertThat(thrown).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(thrown).hasCauseThat().isNotInstanceOf(UploadAbortedException.class);
    assertThat(thrown).hasCauseThat().hasMessageThat().contains("503");

    assertThat(fakeEngine.requests()).hasSize(2);
    RecordedRequest abortConn = fakeEngine.requests().get(1);
    assertThat(abortConn.url())
        .isEqualTo(
            "https://coordinator.google.com/v1/confidentialaggregations/agg-001/clients/tok-001:abort?%24alt=proto");
    assertThat(abortConn.headers()).containsEntry("Content-Type", "application/x-protobuf");
    assertThat(abortConn.headers()).containsEntry("x-goog-api-key", TEST_API_KEY);
    assertThat(abortConn.headers()).containsEntry("X-Submit-Extra", "SubmitVal");
    assertThat(abortConn.headers()).containsEntry("X-Request-Header", "ReqVal");

    AbortConfidentialAggregationRequest abortRequest = parseAbortRequest(abortConn.body());
    assertThat(abortRequest.getStatus().getCode()).isEqualTo(Code.UNAVAILABLE_VALUE);
    assertThat(abortRequest.getStatus().getMessage())
        .isEqualTo(AggregationAbortSender.ABORT_ERROR_MESSAGE);
    assertThat(abortRequest.getAggregationId()).isEmpty();
    assertThat(abortRequest.getClientToken()).isEmpty();
  }

  @Test
  public void upload_blobAbortedByServer_failsWithServerAbortedAndSendsNoAbort() {
    WorkloadSet workloadSet = createTestWorkloadSet();
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.respondTo("/upload/v1/media/", 409, "ABORTED".getBytes(UTF_8));

    ListenableFuture<Void> future = payloadUploader.upload(workloadSet, payload);
    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(UploadAbortedException.class);
    UploadAbortedException cause = (UploadAbortedException) thrown.getCause();
    assertThat(cause).hasCauseThat().isInstanceOf(IOException.class);

    assertThat(fakeEngine.requests()).hasSize(1);
  }

  @Test
  public void upload_blobFailsAndAbortNotAcknowledged_failsWithOriginalException() {
    WorkloadSet workloadSet = createTestWorkloadSet();
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.respondTo("/upload/v1/media/", 503, "Service Unavailable".getBytes(UTF_8));
    fakeEngine.respondTo(":abort", 503, "Service Unavailable".getBytes(UTF_8));

    ListenableFuture<Void> future = payloadUploader.upload(workloadSet, payload);
    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(thrown).hasCauseThat().isNotInstanceOf(UploadAbortedException.class);

    assertThat(fakeEngine.requests()).hasSize(2);
    assertThat(fakeEngine.requests().get(0).url()).contains("/upload/v1/media/");
    assertThat(fakeEngine.requests().get(1).url()).contains(":abort");
  }

  @Test
  public void upload_submitFailure_failsFutureWithoutAbort() {
    WorkloadSet workloadSet = createTestWorkloadSet();
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.respondTo(":submit", 500, "Internal error".getBytes(UTF_8));

    ListenableFuture<Void> future = payloadUploader.upload(workloadSet, payload);
    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(thrown).hasCauseThat().isNotInstanceOf(UploadAbortedException.class);

    assertThat(fakeEngine.requests()).hasSize(2);
    assertThat(fakeEngine.requests().get(0).url()).contains("/upload/v1/media/");
    assertThat(fakeEngine.requests().get(1).url()).contains(":submit");
  }

  @Test
  public void upload_submitAbortedByServer_failsWithOriginalExceptionAndSendsNoAbort() {
    WorkloadSet workloadSet = createTestWorkloadSet();
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.respondTo(":submit", 409, "ABORTED".getBytes(UTF_8));

    ListenableFuture<Void> future = payloadUploader.upload(workloadSet, payload);
    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    // The blob was already stored, so an ABORTED submit leaves the outcome unknown.
    assertThat(thrown).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(thrown).hasCauseThat().isNotInstanceOf(UploadAbortedException.class);

    assertThat(fakeEngine.requests()).hasSize(2);
    assertThat(fakeEngine.requests().get(0).url()).contains("/upload/v1/media/");
    assertThat(fakeEngine.requests().get(1).url()).contains(":submit");
  }

  @Test
  public void upload_uploadFailure404_sendsAbortWithNotFound() throws Exception {
    WorkloadSet workloadSet = createTestWorkloadSet();
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.respondTo("/upload/v1/media/", 404, "Not found".getBytes(UTF_8));

    ListenableFuture<Void> future = payloadUploader.upload(workloadSet, payload);
    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(thrown).hasCauseThat().isNotInstanceOf(UploadAbortedException.class);

    assertThat(fakeEngine.requests()).hasSize(2);
    AbortConfidentialAggregationRequest abortRequest =
        parseAbortRequest(fakeEngine.requests().get(1).body());
    assertThat(abortRequest.getStatus().getCode()).isEqualTo(Code.NOT_FOUND_VALUE);
    assertThat(abortRequest.getStatus().getMessage())
        .isEqualTo(AggregationAbortSender.ABORT_ERROR_MESSAGE);
  }

  @Test
  public void upload_blobTransportIOException_sendsAbortWithUnavailable() throws Exception {
    WorkloadSet workloadSet = createTestWorkloadSet();
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.failWith("/upload/v1/media/", new IOException("connection reset"));

    ListenableFuture<Void> future = payloadUploader.upload(workloadSet, payload);
    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(thrown).hasCauseThat().isNotInstanceOf(UploadAbortedException.class);
    assertThat(thrown).hasCauseThat().hasMessageThat().contains("connection reset");

    assertThat(fakeEngine.requests()).hasSize(1);
    RecordedRequest abortConn = fakeEngine.requests().get(0);
    assertThat(abortConn.url()).contains(":abort");
    AbortConfidentialAggregationRequest abortRequest = parseAbortRequest(abortConn.body());
    assertThat(abortRequest.getStatus().getCode()).isEqualTo(Code.UNAVAILABLE_VALUE);
    assertThat(abortRequest.getStatus().getMessage())
        .isEqualTo(AggregationAbortSender.ABORT_ERROR_MESSAGE);
  }

  @Test
  public void upload_blobThrowsRuntimeException_sendsAbortWithInternalAndRethrows()
      throws Exception {
    WorkloadSet workloadSet = createTestWorkloadSet();
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.failWith("/upload/v1/media/", new IllegalStateException("Connection failed"));

    ListenableFuture<Void> future = payloadUploader.upload(workloadSet, payload);
    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(IllegalStateException.class);

    assertThat(fakeEngine.requests()).hasSize(1);
    RecordedRequest abortConn = fakeEngine.requests().get(0);
    assertThat(abortConn.url()).contains(":abort");
    AbortConfidentialAggregationRequest abortRequest = parseAbortRequest(abortConn.body());
    assertThat(abortRequest.getStatus().getCode()).isEqualTo(Code.INTERNAL_VALUE);
  }

  @Test
  public void upload_blobThrowsRuntimeExceptionAndAbortNotAcknowledged_failsWithRuntimeException() {
    WorkloadSet workloadSet = createTestWorkloadSet();
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));
    fakeEngine.failWith("/upload/v1/media/", new IllegalStateException("Connection failed"));
    fakeEngine.respondTo(":abort", 503, "Service Unavailable".getBytes(UTF_8));

    ListenableFuture<Void> future = payloadUploader.upload(workloadSet, payload);
    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(IllegalStateException.class);
    assertThat(thrown).hasCauseThat().isNotInstanceOf(UploadAbortedException.class);

    assertThat(fakeEngine.requests()).hasSize(1);
    RecordedRequest abortConn = fakeEngine.requests().get(0);
    assertThat(abortConn.url()).contains(":abort");
  }

  @Test
  public void upload_resourceNameWithSlashAndSpace_keepsSlashAndEncodesSpace() throws Exception {
    UploadTokens tokens =
        UploadTokens.builder()
            .setAggregationId(TEST_AGGREGATION_ID)
            .setClientToken(TEST_CLIENT_TOKEN)
            .setUploadUri(TEST_UPLOAD_URI)
            .setResourceName("sessions/agg-001/blobs/b 001")
            .setSubmitUriPrefix(TEST_SUBMIT_URI)
            .build();

    WorkloadSet workloadSet = createTestWorkloadSet(tokens);

    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));

    payloadUploader.upload(workloadSet, payload).get();

    RecordedRequest uploadConn = fakeEngine.requests().get(0);
    assertThat(uploadConn.url())
        .isEqualTo(
            "https://scotty.google.com/upload/v1/media/sessions/agg-001/blobs/b%20001?upload_protocol=raw");
  }

  @Test
  public void upload_forwardingHeadersOverrideCallerHeaders() throws Exception {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.setDefaultResponse(200, new byte[0]);
    FcpConfidentialPayloadUploader uploader =
        new FcpConfidentialPayloadUploader(
            FcpConfidentialPayloadUploaderDeps.builder()
                .setHttpEngine(engine)
                .setExecutor(directExecutor())
                .setApiKey(TEST_API_KEY)
                .setRequestHeaders(ImmutableMap.of("x-test-header", "caller"))
                .setSworTracker(SworTracker.DISABLED)
                .build());

    UploadTokens tokens =
        UploadTokens.builder()
            .setAggregationId(TEST_AGGREGATION_ID)
            .setClientToken(TEST_CLIENT_TOKEN)
            .setUploadUri(TEST_UPLOAD_URI)
            .setResourceName(TEST_RESOURCE_NAME)
            .setSubmitUriPrefix(TEST_SUBMIT_URI)
            .setUploadExtraHeaders(ImmutableMap.of("x-test-header", "server"))
            .setSubmitExtraHeaders(ImmutableMap.of("x-test-header", "server"))
            .build();

    WorkloadSet workloadSet = createTestWorkloadSet(tokens);

    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("encrypted-test-data"));

    uploader.upload(workloadSet, payload).get();

    assertThat(engine.requests()).hasSize(2);
    RecordedRequest uploadConn = engine.requests().get(0);
    assertThat(uploadConn.headers()).containsEntry("x-test-header", "server");

    RecordedRequest submitConn = engine.requests().get(1);
    assertThat(submitConn.headers()).containsEntry("x-test-header", "server");
  }

  @Test
  public void upload_apiKeyOverridesCallerApiKeyHeader() throws Exception {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.setDefaultResponse(200, new byte[0]);
    FcpConfidentialPayloadUploader uploader =
        new FcpConfidentialPayloadUploader(
            FcpConfidentialPayloadUploaderDeps.builder()
                .setHttpEngine(engine)
                .setExecutor(directExecutor())
                .setApiKey(TEST_API_KEY)
                .setRequestHeaders(ImmutableMap.of("x-goog-api-key", "caller-key"))
                .setSworTracker(SworTracker.DISABLED)
                .build());
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));

    uploader.upload(createTestWorkloadSet(), payload).get();

    assertThat(engine.requests()).hasSize(2);
    for (RecordedRequest request : engine.requests()) {
      assertThat(request.headers()).containsEntry("x-goog-api-key", TEST_API_KEY);
    }
  }

  @Test
  public void upload_success_logsStartedThenCompletedWithBodyBytes() throws Exception {
    RecordingStabilityMonitor monitor = new RecordingStabilityMonitor();
    byte[] responseBody = "ok".getBytes(UTF_8);
    fakeEngine.setDefaultResponse(200, responseBody);
    ByteString payloadBytes = ByteString.copyFromUtf8("encrypted-test-data");
    EncodedPayload payload = createTestPayload(payloadBytes);

    monitoredUploader(monitor).upload(createTestWorkloadSet(), payload).get();

    assertThat(monitor.kinds())
        .containsExactly(Kind.UPLOAD_STARTED, Kind.UPLOAD_COMPLETED)
        .inOrder();
    for (DiagnosticEvent event : monitor.events()) {
      assertThat(event.workloadSetName()).isEqualTo(TEST_WORKLOAD_SET_NAME);
      assertThat(event.logSourceName()).isEqualTo("log_source");
      assertThat(event.fetchId()).isEqualTo(42);
      assertThat(event.error()).isNull();
    }
    DiagnosticEvent completed = monitor.event(Kind.UPLOAD_COMPLETED);
    // The blob upload and the submit.
    assertThat(completed.bytesUploaded())
        .isEqualTo(payloadBytes.size() + fakeEngine.requests().get(1).body().length);
    assertThat(completed.bytesDownloaded()).isEqualTo(2L * responseBody.length);
    assertThat(completed.duration().isNegative()).isFalse();
  }

  @Test
  public void upload_blobAbortedByServer_logsAborted() {
    RecordingStabilityMonitor monitor = new RecordingStabilityMonitor();
    byte[] abortedBody = "ABORTED".getBytes(UTF_8);
    fakeEngine.respondTo("/upload/v1/media/", 409, abortedBody);
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));

    ListenableFuture<Void> future =
        monitoredUploader(monitor).upload(createTestWorkloadSet(), payload);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(monitor.kinds()).containsExactly(Kind.UPLOAD_STARTED, Kind.UPLOAD_ABORTED).inOrder();
    DiagnosticEvent aborted = monitor.event(Kind.UPLOAD_ABORTED);
    assertThat(aborted.error()).isSameInstanceAs(thrown.getCause());
    assertThat(aborted.bytesUploaded()).isEqualTo(4);
    assertThat(aborted.bytesDownloaded()).isEqualTo(abortedBody.length);
  }

  @Test
  public void upload_eventsUseTheInjectedTickerAndTimeSource() {
    AtomicLong nanos = new AtomicLong();
    Ticker ticker =
        new Ticker() {
          @Override
          public long read() {
            return nanos.getAndAdd(Duration.ofMillis(7).toNanos());
          }
        };
    Instant now = Instant.parse("2026-11-12T12:00:00Z");
    RecordingStabilityMonitor monitor = new RecordingStabilityMonitor();
    FcpConfidentialPayloadUploader timedUploader =
        new FcpConfidentialPayloadUploader(
            FcpConfidentialPayloadUploaderDeps.builder()
                .setHttpEngine(fakeEngine)
                .setExecutor(directExecutor())
                .setApiKey(TEST_API_KEY)
                .setSworTracker(SworTracker.DISABLED)
                .setStabilityMonitor(monitor)
                .setTicker(ticker)
                .setTimeSource(new FakeTimeSource(now))
                .build());
    fakeEngine.respondTo("/upload/v1/media/", 503, new byte[0]);
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));

    assertThrows(
        ExecutionException.class,
        () -> timedUploader.upload(createTestWorkloadSet(), payload).get());

    DiagnosticEvent error = monitor.event(Kind.UPLOAD_ERROR);
    assertThat(error.duration()).isEqualTo(Duration.ofMillis(7));
    assertThat(error.timestamp()).isEqualTo(now);
  }

  @Test
  public void upload_blobRefusedByServer_logsAborted() {
    RecordingStabilityMonitor monitor = new RecordingStabilityMonitor();
    byte[] refusedBody = "PERMISSION_DENIED".getBytes(UTF_8);
    fakeEngine.respondTo("/upload/v1/media/", 403, refusedBody);
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));

    ListenableFuture<Void> future =
        monitoredUploader(monitor).upload(createTestWorkloadSet(), payload);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(monitor.kinds()).containsExactly(Kind.UPLOAD_STARTED, Kind.UPLOAD_ABORTED).inOrder();
    DiagnosticEvent aborted = monitor.event(Kind.UPLOAD_ABORTED);
    assertThat(aborted.error()).isSameInstanceAs(thrown.getCause());
    assertThat(aborted.bytesUploaded()).isEqualTo(4);
    assertThat(aborted.bytesDownloaded()).isEqualTo(refusedBody.length);
  }

  @Test
  public void upload_blobFailsAndAbortFails_logsAbortRequestFailedDiagCode() {
    RecordingStabilityMonitor monitor = new RecordingStabilityMonitor();
    fakeEngine.respondTo("/upload/v1/media/", 503, new byte[0]);
    fakeEngine.respondTo(":abort", 503, new byte[0]);
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));

    ListenableFuture<Void> future =
        monitoredUploader(monitor).upload(createTestWorkloadSet(), payload);

    assertThrows(ExecutionException.class, future::get);
    assertThat(monitor.diagCodes())
        .containsExactly(ProdDiagCode.HTTP_CANCELLATION_OR_ABORT_REQUEST_FAILED);
    assertThat(monitor.diagFetchIds()).containsExactly(42L);
  }

  @Test
  public void upload_blobFailure_logsError() {
    RecordingStabilityMonitor monitor = new RecordingStabilityMonitor();
    fakeEngine.respondTo("/upload/v1/media/", 503, "Service Unavailable".getBytes(UTF_8));
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));

    ListenableFuture<Void> future =
        monitoredUploader(monitor).upload(createTestWorkloadSet(), payload);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(monitor.kinds()).containsExactly(Kind.UPLOAD_STARTED, Kind.UPLOAD_ERROR).inOrder();
    assertThat(monitor.event(Kind.UPLOAD_ERROR).error()).isSameInstanceAs(thrown.getCause());
  }

  @Test
  public void upload_submitFailure_logsError() {
    RecordingStabilityMonitor monitor = new RecordingStabilityMonitor();
    fakeEngine.respondTo(":submit", 500, "Internal error".getBytes(UTF_8));
    EncodedPayload payload = createTestPayload(ByteString.copyFromUtf8("data"));

    ListenableFuture<Void> future =
        monitoredUploader(monitor).upload(createTestWorkloadSet(), payload);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(monitor.kinds()).containsExactly(Kind.UPLOAD_STARTED, Kind.UPLOAD_ERROR).inOrder();
    assertThat(monitor.event(Kind.UPLOAD_ERROR).error()).isSameInstanceAs(thrown.getCause());
  }

  private FcpConfidentialPayloadUploader monitoredUploader(RecordingStabilityMonitor monitor) {
    return new FcpConfidentialPayloadUploader(
        FcpConfidentialPayloadUploaderDeps.builder()
            .setHttpEngine(fakeEngine)
            .setExecutor(directExecutor())
            .setApiKey(TEST_API_KEY)
            .setSworTracker(SworTracker.DISABLED)
            .setStabilityMonitor(monitor)
            .build());
  }

  private static AbortConfidentialAggregationRequest parseAbortRequest(byte[] bytes)
      throws InvalidProtocolBufferException {
    return AbortConfidentialAggregationRequest.parseFrom(
        bytes, ExtensionRegistryLite.getEmptyRegistry());
  }

  private FcpConfidentialPayloadUploader sworUploader() {
    return new FcpConfidentialPayloadUploader(
        FcpConfidentialPayloadUploaderDeps.builder()
            .setHttpEngine(fakeEngine)
            .setExecutor(directExecutor())
            .setApiKey(TEST_API_KEY)
            .setSworTracker(new SworTracker(sworStore, timeSource))
            .build());
  }
}
