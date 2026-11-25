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
package com.google.fcp.client.privatelogger.fcpconnector;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.truth.Truth.assertThat;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.base.Ticker;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.confidentialcompute.ConfidentialCompute;
import com.google.fcp.client.confidentialcompute.ConfidentialPayloadEncoder;
import com.google.fcp.client.confidentialcompute.ConfidentialPayloadUploader;
import com.google.fcp.client.confidentialcompute.EncodedPayload;
import com.google.fcp.client.confidentialcompute.PayloadEncodingException;
import com.google.fcp.client.confidentialcompute.UploadAbortedException;
import com.google.fcp.client.confidentialcompute.UploadTokens;
import com.google.fcp.client.confidentialcompute.WorkloadSet;
import com.google.fcp.client.confidentialcompute.WorkloadSetFetcher;
import com.google.fcp.client.privatelogger.LogEntry;
import com.google.fcp.client.privatelogger.PrivateLoggerUnavailableException;
import com.google.fcp.client.privatelogger.impl.DataProvider;
import com.google.fcp.client.privatelogger.impl.UploadOutcome;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent.Kind;
import com.google.fcp.client.stabilitymonitor.RecordingStabilityMonitor;
import com.google.intelligence.fcp.client.FederatedComputation;
import com.google.intelligence.fcp.client.SelectorContext;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class FcpPrivateLoggerConnectorTest {

  private static final String LOG_SOURCE = "test_log_source";
  private static final long FETCH_ID = 42;
  private static final String WORKLOAD_SET = "workload_set_one";
  private static final String OTHER_WORKLOAD_SET = "workload_set_two";

  /**
   * Records uploads. An upload is answered from {@link #uploadFutures} by workload set name, else
   * from {@link #uploadQueue}, else succeeds.
   */
  private static final class FakeUploader implements ConfidentialPayloadUploader {
    final Map<String, ListenableFuture<Void>> uploadFutures = new ConcurrentHashMap<>();
    final Queue<ListenableFuture<Void>> uploadQueue = new ConcurrentLinkedQueue<>();
    final List<String> uploadedWorkloadSets = new CopyOnWriteArrayList<>();
    volatile CountDownLatch uploadsStarted = new CountDownLatch(0);
    volatile Throwable uploadThrows = null;

    @Override
    public ListenableFuture<Void> upload(WorkloadSet workloadSet, EncodedPayload payload) {
      uploadedWorkloadSets.add(workloadSet.name());
      uploadsStarted.countDown();
      if (uploadThrows != null) {
        if (uploadThrows instanceof RuntimeException re) {
          throw re;
        } else if (uploadThrows instanceof Error err) {
          throw err;
        }
        throw new RuntimeException(uploadThrows);
      }
      if (uploadFutures.containsKey(workloadSet.name())) {
        return uploadFutures.get(workloadSet.name());
      }
      return uploadQueue.isEmpty() ? Futures.immediateVoidFuture() : uploadQueue.remove();
    }

    /** Waits until {@code count} uploads have started. */
    void awaitUploads(int count) throws InterruptedException {
      int remaining = count - uploadedWorkloadSets.size();
      if (remaining <= 0) {
        return;
      }
      CountDownLatch latch = new CountDownLatch(remaining);
      uploadsStarted = latch;
      if (uploadedWorkloadSets.size() >= count) {
        return;
      }
      assertThat(latch.await(10, SECONDS)).isTrue();
    }
  }

  private static final class TestError extends Error {
    TestError(String message) {
      super(message);
    }
  }

  private final RecordingStabilityMonitor stabilityMonitor = new RecordingStabilityMonitor();
  private FakeUploader uploader;
  private ImmutableList<WorkloadSet> resolvedWorkloadSets;
  private ListenableFuture<ImmutableList<WorkloadSet>> fetchFuture;
  private Exception encodingFailure;
  private FcpPrivateLoggerConnector connector;
  private static final Instant NOW = Instant.parse("2026-11-12T12:00:00Z");
  private final TimeSource timeSource = () -> NOW;
  private final AtomicLong tickerNanos = new AtomicLong(1_000_000_000L);
  private final Ticker ticker =
      new Ticker() {
        @Override
        public long read() {
          return tickerNanos.getAndAdd(500_000_000L);
        }
      };

  @Before
  public void setUp() {
    uploader = new FakeUploader();
    resolvedWorkloadSets = ImmutableList.of();
    fetchFuture = null;
    encodingFailure = null;

    connector = new FcpPrivateLoggerConnector(depsBuilder().build());
  }

  private FcpPrivateLoggerConnectorDeps.Builder depsBuilder() {
    return depsBuilder(
        (ws, options, inputs) -> {
          if (encodingFailure != null) {
            return Futures.immediateFailedFuture(encodingFailure);
          }
          return Futures.immediateFuture(createPayload());
        });
  }

  /**
   * Deps whose confidential compute client has the fake fetcher and uploader and {@code encoder}.
   */
  private FcpPrivateLoggerConnectorDeps.Builder depsBuilder(ConfidentialPayloadEncoder encoder) {
    WorkloadSetFetcher fetcher =
        logSourceName ->
            fetchFuture != null ? fetchFuture : Futures.immediateFuture(resolvedWorkloadSets);
    ConfidentialCompute confidentialCompute =
        ConfidentialCompute.builder()
            .setWorkloadSetFetcher(fetcher)
            .setPayloadEncoder(encoder)
            .setPayloadUploader(uploader)
            .setTicker(ticker)
            .setTimeSource(timeSource)
            .setStabilityMonitor(stabilityMonitor)
            .build();

    return FcpPrivateLoggerConnectorDeps.builder()
        .setConfidentialCompute(confidentialCompute)
        .setEndorsementOptions(AccessPolicyEndorsementOptions.getDefaultInstance());
  }

  @Test
  public void upload_happyPath_contributed() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(outcomes).containsExactly(outcome(WORKLOAD_SET, UploadOutcome.Status.CONTRIBUTED));
    assertThat(uploader.uploadedWorkloadSets).containsExactly(WORKLOAD_SET);
  }

  @Test
  public void upload_passesEndorsementOptionsToEncoder() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    AccessPolicyEndorsementOptions endorsementOptions =
        AccessPolicyEndorsementOptions.newBuilder()
            .addKmsVerifyingKeys(Key.newBuilder().setKeyId(ByteString.copyFromUtf8("kms-key")))
            .build();
    List<AccessPolicyEndorsementOptions> encodedWith = new CopyOnWriteArrayList<>();
    ConfidentialPayloadEncoder encoder =
        (ws, options, inputs) -> {
          encodedWith.add(options);
          return Futures.immediateFuture(createPayload());
        };
    FcpPrivateLoggerConnector customConnector =
        new FcpPrivateLoggerConnector(
            depsBuilder(encoder).setEndorsementOptions(endorsementOptions).build());

    var unused = customConnector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(encodedWith).containsExactly(endorsementOptions);
  }

  @Test
  public void upload_uploadFailsWithUnknownOutcome_contributed() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    uploader.uploadQueue.add(Futures.immediateFailedFuture(new IOException("Upload failed")));

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(outcomes).containsExactly(outcome(WORKLOAD_SET, UploadOutcome.Status.CONTRIBUTED));
    assertThat(stabilityMonitor.crashes()).isEmpty();
  }

  @Test
  public void upload_uploadAbortedByServer_notContributed() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    uploader.uploadQueue.add(
        Futures.immediateFailedFuture(
            new UploadAbortedException("aborted", new IOException("409"))));

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(outcomes)
        .containsExactly(outcome(WORKLOAD_SET, UploadOutcome.Status.NOT_CONTRIBUTED));
  }

  @Test
  public void upload_uploadFailsWithRuntimeException_contributed() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    IllegalStateException bug = new IllegalStateException("boom");
    uploader.uploadQueue.add(Futures.immediateFailedFuture(bug));

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(outcomes).containsExactly(outcome(WORKLOAD_SET, UploadOutcome.Status.CONTRIBUTED));
    // The outcome stops a second upload; the RuntimeException is still reported as a bug.
    assertThat(stabilityMonitor.crashes()).containsExactly(bug);
  }

  @Test
  public void upload_uploaderThrowsSynchronously_contributed() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    IllegalStateException bug = new IllegalStateException("synchronous crash");
    uploader.uploadThrows = bug;

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(outcomes).containsExactly(outcome(WORKLOAD_SET, UploadOutcome.Status.CONTRIBUTED));
    assertThat(stabilityMonitor.crashes()).containsExactly(bug);
  }

  @Test
  public void upload_runtimeExceptionInOneSetsUpload_otherSetsOutcomesStillDelivered()
      throws Exception {
    resolvedWorkloadSets =
        ImmutableList.of(createWorkloadSet(WORKLOAD_SET), createWorkloadSet(OTHER_WORKLOAD_SET));
    IllegalStateException bug = new IllegalStateException("bug");
    uploader.uploadFutures.put(WORKLOAD_SET, Futures.immediateFailedFuture(bug));
    uploader.uploadFutures.put(
        OTHER_WORKLOAD_SET,
        Futures.immediateFailedFuture(
            new UploadAbortedException("aborted", new IOException("409"))));

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    // The bug counts as contributed for its own set, and the other set's rollback still reaches
    // PrivateLoggerImpl.
    assertThat(outcomes)
        .containsExactly(
            outcome(WORKLOAD_SET, UploadOutcome.Status.CONTRIBUTED),
            outcome(OTHER_WORKLOAD_SET, UploadOutcome.Status.NOT_CONTRIBUTED));
    assertThat(stabilityMonitor.crashes()).containsExactly(bug);
  }

  @Test
  public void upload_encoderThrowsRuntimeExceptionForOneSet_onlyThatSetNotContributed()
      throws Exception {
    resolvedWorkloadSets =
        ImmutableList.of(createWorkloadSet(WORKLOAD_SET), createWorkloadSet(OTHER_WORKLOAD_SET));
    IllegalStateException bug = new IllegalStateException("bug");
    ConfidentialPayloadEncoder encoder =
        (ws, options, inputs) -> {
          if (ws.name().equals(WORKLOAD_SET)) {
            throw bug;
          }
          return Futures.immediateFuture(createPayload());
        };
    FcpPrivateLoggerConnector customConnector =
        new FcpPrivateLoggerConnector(depsBuilder(encoder).build());

    List<UploadOutcome> outcomes = customConnector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(outcomes)
        .containsExactly(
            outcome(WORKLOAD_SET, UploadOutcome.Status.NOT_CONTRIBUTED),
            outcome(OTHER_WORKLOAD_SET, UploadOutcome.Status.CONTRIBUTED));
    assertThat(uploader.uploadedWorkloadSets).containsExactly(OTHER_WORKLOAD_SET);
    assertThat(stabilityMonitor.crashes()).containsExactly(bug);
  }

  @Test
  public void upload_noData_notContributed() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    DataProvider emptyProvider = selectorContext -> Futures.immediateFuture(ImmutableList.of());

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, emptyProvider).get();

    assertThat(outcomes)
        .containsExactly(outcome(WORKLOAD_SET, UploadOutcome.Status.NOT_CONTRIBUTED));
    assertThat(uploader.uploadedWorkloadSets).isEmpty();
  }

  @Test
  public void upload_encodingFails_notContributed() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    encodingFailure = new PayloadEncodingException("boom", new IOException());

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(outcomes)
        .containsExactly(outcome(WORKLOAD_SET, UploadOutcome.Status.NOT_CONTRIBUTED));
    assertThat(uploader.uploadedWorkloadSets).isEmpty();
    // A checked exception is an expected failure, not a crash.
    assertThat(stabilityMonitor.crashes()).isEmpty();
  }

  @Test
  public void upload_fetchFailsWithIoException_completesWithEmptyOutcomes() throws Exception {
    fetchFuture = Futures.immediateFailedFuture(new IOException("Fetch RPC failed"));

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(outcomes).isEmpty();
    assertThat(uploader.uploadedWorkloadSets).isEmpty();
    assertThat(stabilityMonitor.crashes()).isEmpty();
  }

  @Test
  public void upload_fetchFailsWithRuntimeException_reportsCrash() {
    IllegalStateException bug = new IllegalStateException("bug");
    fetchFuture = Futures.immediateFailedFuture(bug);

    ListenableFuture<List<UploadOutcome>> future =
        connector.upload(LOG_SOURCE, sampleDataProvider());
    ExecutionException e = assertThrows(ExecutionException.class, future::get);

    assertThat(e).hasCauseThat().isInstanceOf(PrivateLoggerUnavailableException.class);
    assertThat(stabilityMonitor.crashes()).containsExactly(bug);
  }

  @Test
  public void upload_queryFails_notContributed() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    IllegalStateException bug = new IllegalStateException("db closed");
    DataProvider failingProvider = selectorContext -> Futures.immediateFailedFuture(bug);

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, failingProvider).get();

    assertThat(outcomes)
        .containsExactly(outcome(WORKLOAD_SET, UploadOutcome.Status.NOT_CONTRIBUTED));
    assertThat(uploader.uploadedWorkloadSets).isEmpty();
    // The outcome keeps the rollback; the RuntimeException is still reported as a bug.
    assertThat(stabilityMonitor.crashes()).containsExactly(bug);
  }

  @Test
  public void upload_selectorContextHasTaskNameAndConfidentialAggregation() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    List<SelectorContext> contexts = new CopyOnWriteArrayList<>();
    DataProvider provider =
        selectorContext -> {
          contexts.add(selectorContext);
          return Futures.immediateFuture(ImmutableList.of());
        };

    var unused = connector.upload(LOG_SOURCE, provider).get();

    FederatedComputation federated =
        Iterables.getOnlyElement(contexts).getComputationProperties().getFederated();
    assertThat(federated.getTaskName()).isEqualTo(WORKLOAD_SET);
    assertThat(federated.hasConfidentialAggregation()).isTrue();
    assertThat(Iterables.getOnlyElement(contexts).getComputationProperties().getRunId())
        .isEqualTo(FETCH_ID);
  }

  @Test
  public void upload_dataProviderThrowsForOneSet_onlyThatSetNotContributed() throws Exception {
    resolvedWorkloadSets =
        ImmutableList.of(createWorkloadSet(WORKLOAD_SET), createWorkloadSet(OTHER_WORKLOAD_SET));
    DataProvider sampleProvider = sampleDataProvider();
    DataProvider provider =
        selectorContext -> {
          String taskName = selectorContext.getComputationProperties().getFederated().getTaskName();
          checkState(!taskName.equals(WORKLOAD_SET), "db closed");
          return sampleProvider.getData(selectorContext);
        };

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, provider).get();

    assertThat(outcomes)
        .containsExactly(
            outcome(WORKLOAD_SET, UploadOutcome.Status.NOT_CONTRIBUTED),
            outcome(OTHER_WORKLOAD_SET, UploadOutcome.Status.CONTRIBUTED));
    assertThat(uploader.uploadedWorkloadSets).containsExactly(OTHER_WORKLOAD_SET);
  }

  @Test
  public void upload_pendingUpload_returnedFutureCompletesWithUpload() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    SettableFuture<Void> pendingUpload = SettableFuture.create();
    uploader.uploadQueue.add(pendingUpload);

    ListenableFuture<List<UploadOutcome>> result =
        connector.upload(LOG_SOURCE, sampleDataProvider());
    uploader.awaitUploads(1);

    // The uploader has been invoked, but nothing blocks waiting for it.
    assertThat(uploader.uploadedWorkloadSets).containsExactly(WORKLOAD_SET);
    assertThat(result.isDone()).isFalse();
    // The SWOR record is made at upload start.

    pendingUpload.set(null);

    assertThat(result.get())
        .containsExactly(outcome(WORKLOAD_SET, UploadOutcome.Status.CONTRIBUTED));
  }

  @Test
  public void upload_multipleWorkloadSets_uploadedConcurrently() throws Exception {
    resolvedWorkloadSets =
        ImmutableList.of(createWorkloadSet(WORKLOAD_SET), createWorkloadSet(OTHER_WORKLOAD_SET));
    SettableFuture<Void> firstUpload = SettableFuture.create();
    SettableFuture<Void> secondUpload = SettableFuture.create();
    uploader.uploadFutures.put(WORKLOAD_SET, firstUpload);
    uploader.uploadFutures.put(OTHER_WORKLOAD_SET, secondUpload);

    ListenableFuture<List<UploadOutcome>> result =
        connector.upload(LOG_SOURCE, sampleDataProvider());
    uploader.awaitUploads(2);

    // Both uploads are in flight before either has completed.
    assertThat(uploader.uploadedWorkloadSets).containsExactly(WORKLOAD_SET, OTHER_WORKLOAD_SET);
    assertThat(result.isDone()).isFalse();

    secondUpload.setException(new UploadAbortedException("aborted", new IOException()));
    assertThat(result.isDone()).isFalse();
    firstUpload.set(null);

    assertThat(result.get())
        .containsExactly(
            outcome(WORKLOAD_SET, UploadOutcome.Status.CONTRIBUTED),
            outcome(OTHER_WORKLOAD_SET, UploadOutcome.Status.NOT_CONTRIBUTED))
        .inOrder();
  }





  @Test
  public void upload_encoderThrowsError_futureFailsWithError() {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    ConfidentialPayloadEncoder failingEncoder =
        (ws, options, inputs) -> {
          throw new TestError("Simulated error");
        };

    FcpPrivateLoggerConnector customConnector =
        new FcpPrivateLoggerConnector(depsBuilder(failingEncoder).build());

    ListenableFuture<List<UploadOutcome>> future =
        customConnector.upload(LOG_SOURCE, sampleDataProvider());

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(TestError.class);
    assertThat(uploader.uploadedWorkloadSets).isEmpty();
    // Errors are not caught, so they are not reported as crashes either.
    assertThat(stabilityMonitor.crashes()).isEmpty();
  }

  @Test
  public void upload_fetcherExecutorShutDown_throwsPrivateLoggerUnavailableException() {
    fetchFuture = Futures.immediateFailedFuture(new RejectedExecutionException("shut down"));

    ListenableFuture<List<UploadOutcome>> future =
        connector.upload(LOG_SOURCE, sampleDataProvider());

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(PrivateLoggerUnavailableException.class);
    assertThat(e).hasCauseThat().hasCauseThat().isInstanceOf(RejectedExecutionException.class);
    assertThat(stabilityMonitor.crashes()).isEmpty();
  }

  @Test
  public void upload_encoderExecutorShutDown_notContributed() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    encodingFailure = new RejectedExecutionException("shut down");

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(outcomes)
        .containsExactly(outcome(WORKLOAD_SET, UploadOutcome.Status.NOT_CONTRIBUTED));
    assertThat(uploader.uploadedWorkloadSets).isEmpty();
    assertThat(stabilityMonitor.crashes()).isEmpty();
  }

  @Test
  public void upload_uploaderExecutorShutDown_contributed() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    uploader.uploadQueue.add(
        Futures.immediateFailedFuture(new RejectedExecutionException("shut down")));

    List<UploadOutcome> outcomes = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(outcomes).containsExactly(outcome(WORKLOAD_SET, UploadOutcome.Status.CONTRIBUTED));
    assertThat(stabilityMonitor.crashes()).isEmpty();
  }

  @Test
  public void upload_uploaderThrowsOutOfMemoryError_futureFailsWithOutOfMemoryError() {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    uploader.uploadThrows = new OutOfMemoryError("simulated OOM");

    ListenableFuture<List<UploadOutcome>> future =
        connector.upload(LOG_SOURCE, sampleDataProvider());

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(OutOfMemoryError.class);
    assertThat(stabilityMonitor.crashes()).isEmpty();
  }

  @Test
  public void upload_happyPath_logsQueryStartedThenCompleted() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));

    var unused = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(stabilityMonitor.kinds())
        .containsExactly(Kind.QUERY_STARTED, Kind.QUERY_COMPLETED)
        .inOrder();
    for (DiagnosticEvent event : stabilityMonitor.events()) {
      assertThat(event.logSourceName()).isEqualTo(LOG_SOURCE);
      assertThat(event.workloadSetName()).isEqualTo(WORKLOAD_SET);
      assertThat(event.error()).isNull();
    }
    DiagnosticEvent completed = stabilityMonitor.event(Kind.QUERY_COMPLETED);
    assertThat(completed.inputCount()).isEqualTo(1);
    assertThat(completed.inputBytes()).isEqualTo("sample-log".length());
    assertThat(completed.duration()).isEqualTo(Duration.ofSeconds(1));
    assertThat(completed.timestamp()).isEqualTo(NOW);
  }

  @Test
  public void upload_twoSets_eachQueryEventHasItsSetAndTheFetchId() throws Exception {
    resolvedWorkloadSets =
        ImmutableList.of(createWorkloadSet(WORKLOAD_SET), createWorkloadSet(OTHER_WORKLOAD_SET));

    var unused = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    assertThat(stabilityMonitor.events()).hasSize(4);
    for (DiagnosticEvent event : stabilityMonitor.events()) {
      assertThat(event.fetchId()).isEqualTo(FETCH_ID);
      assertThat(event.workloadSetName()).isAnyOf(WORKLOAD_SET, OTHER_WORKLOAD_SET);
    }
  }

  @Test
  public void upload_noData_logsQueryNoData() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    DataProvider emptyProvider = selectorContext -> Futures.immediateFuture(ImmutableList.of());

    var unused = connector.upload(LOG_SOURCE, emptyProvider).get();

    assertThat(stabilityMonitor.kinds())
        .containsExactly(Kind.QUERY_STARTED, Kind.QUERY_NO_DATA)
        .inOrder();
    DiagnosticEvent noData = stabilityMonitor.event(Kind.QUERY_NO_DATA);
    assertThat(noData.logSourceName()).isEqualTo(LOG_SOURCE);
    assertThat(noData.workloadSetName()).isEqualTo(WORKLOAD_SET);
    assertThat(noData.inputCount()).isEqualTo(0);
  }

  @Test
  public void upload_encodingFails_logsNoQueryError() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    encodingFailure = new PayloadEncodingException("boom", new IOException());

    var unused = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    // Encode events are the encoder's to log; the fake encoder logs none.
    assertThat(stabilityMonitor.kinds())
        .containsExactly(Kind.QUERY_STARTED, Kind.QUERY_COMPLETED)
        .inOrder();
  }

  @Test
  public void upload_queryFails_logsQueryError() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    IllegalStateException readFailure = new IllegalStateException("db closed");
    DataProvider failingProvider = selectorContext -> Futures.immediateFailedFuture(readFailure);

    var unused = connector.upload(LOG_SOURCE, failingProvider).get();

    assertThat(stabilityMonitor.kinds())
        .containsExactly(Kind.QUERY_STARTED, Kind.QUERY_ERROR)
        .inOrder();
    assertThat(stabilityMonitor.event(Kind.QUERY_ERROR).error()).isSameInstanceAs(readFailure);
  }

  @Test
  public void upload_uploadFails_logsNoQueryError() throws Exception {
    resolvedWorkloadSets = ImmutableList.of(createWorkloadSet(WORKLOAD_SET));
    uploader.uploadQueue.add(Futures.immediateFailedFuture(new IOException("Upload failed")));

    var unused = connector.upload(LOG_SOURCE, sampleDataProvider()).get();

    // Upload events are the uploader's to log; the fake uploader logs none.
    assertThat(stabilityMonitor.kinds())
        .containsExactly(Kind.QUERY_STARTED, Kind.QUERY_COMPLETED)
        .inOrder();
  }

  private static UploadOutcome outcome(String workloadSetName, UploadOutcome.Status status) {
    return UploadOutcome.builder().setTaskName(workloadSetName).setStatus(status).build();
  }

  private static WorkloadSet createWorkloadSet(String name) {
    return WorkloadSet.builder()
        .setName(name)
        .setLogSourceName(LOG_SOURCE)
        .setFetchId(FETCH_ID)
        .setSignedEndorsements(ByteString.EMPTY)
        .setEncryptionConfig(ByteString.copyFromUtf8("encryption-config"))
        .setUploadTokens(
            UploadTokens.builder()
                .setAggregationId("agg-id-" + name)
                .setClientToken("client-token")
                .setUploadUri("https://scotty.google.com")
                .setResourceName("resource-name")
                .setSubmitUriPrefix("https://submit.google.com")
                .build())
        .build();
  }

  private static EncodedPayload createPayload() {
    return EncodedPayload.create(ByteString.copyFrom(new byte[] {1, 2, 3}));
  }

  private static DataProvider sampleDataProvider() {
    return selectorContext ->
        Futures.immediateFuture(
            ImmutableList.of(
                LogEntry.builder()
                    .setValue(ByteString.copyFromUtf8("sample-log"))
                    .setTimestamp("2026-09-24T10:00:00Z")
                    .build()));
  }
}
