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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.FluentFuture;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.fcp.client.confidentialcompute.EncodedPayload;
import com.google.fcp.client.confidentialcompute.UploadAbortedException;
import com.google.fcp.client.confidentialcompute.WorkloadSet;
import com.google.fcp.client.privatelogger.LogEntry;
import com.google.fcp.client.privatelogger.PrivateLoggerUnavailableException;
import com.google.fcp.client.privatelogger.impl.DataProvider;
import com.google.fcp.client.privatelogger.impl.PrivateLoggerConnector;
import com.google.fcp.client.privatelogger.impl.UploadOutcome;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent.Kind;
import com.google.intelligence.fcp.client.ConfidentialAggregation;
import com.google.intelligence.fcp.client.FederatedComputation;
import com.google.intelligence.fcp.client.QueryTimeComputationProperties;
import com.google.intelligence.fcp.client.SelectorContext;
import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.RejectedExecutionException;

/**
 * {@link PrivateLoggerConnector} that uploads through the unbundled confidential compute client. It
 * fetches the workload sets for the log source, then for each set reads the entries, encodes them
 * and uploads the payload.
 *
 * <p>Each workload set is handled on its own and reports one of these back to {@code
 * PrivateLoggerImpl}, which rolls back the entries of every set reported as {@code
 * NOT_CONTRIBUTED}:
 *
 * <ul>
 *   <li>No entries, or reading or encoding failed: {@code NOT_CONTRIBUTED}.
 *   <li>Uploaded: {@code CONTRIBUTED}.
 *   <li>Upload failed with {@link UploadAbortedException}: {@code NOT_CONTRIBUTED}.
 *   <li>Upload failed in any other way: {@code CONTRIBUTED}, because the server may have accepted
 *       the payload and the entries must never be uploaded twice.
 * </ul>
 *
 * <p>This applies to bugs too: any {@link Exception} in a set, even a {@link RuntimeException},
 * ends up as one of the outcomes above rather than failing the future. That is deliberate. {@code
 * PrivateLoggerImpl} only rolls back entries for sets that report {@code NOT_CONTRIBUTED}; if the
 * whole future failed, no set would be rolled back, and entries read for sets that never uploaded
 * would stay marked as contributed.
 *
 * <p>A {@link RuntimeException} is still a bug, so the connector also reports it to the {@link
 * ConfidentialCompute#stabilityMonitor()}, which by default crashes the process. A {@link
 * RejectedExecutionException} from a shut-down executor is not a bug and is only mapped. {@link
 * Error}s are not caught and fail the returned future.
 *
 * <p>No abort is sent for a set that fails before the upload; its session expires on the server. If
 * the fetch fails with an {@link IOException}, such as when the device is offline, the returned
 * future completes with no outcomes and the entries are offered again on the next upload. Any other
 * fetch failure fails the returned future with {@link PrivateLoggerUnavailableException}, and an
 * {@link Error} thrown by a dependency fails it with that error. In every case {@code
 * PrivateLoggerImpl} rolls nothing back, so no entry is uploaded twice.
 *
 * <p>Threading: the connector has no executor of its own. The fetcher, encoder and uploader run
 * their blocking work on the executor they were built with, and return a failed future rather than
 * throwing if it has been shut down; the data provider is asynchronous too. The connector only
 * chains the futures they return, on {@code directExecutor()}, so each of its own steps is cheap
 * bookkeeping on whichever thread completed the previous one. A shut-down executor before the
 * upload starts surfaces as {@code NOT_CONTRIBUTED}, and the entries are offered again later; once
 * the upload has started it surfaces as {@code CONTRIBUTED}, like any other upload failure.
 *
 * <p>Corresponds to the multiple task assignment flow in {@code fcp/client/fl_runner.cc}.
 */
public final class FcpPrivateLoggerConnector implements PrivateLoggerConnector {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final FcpPrivateLoggerConnectorDeps deps;

  /** Creates a new connector with the given dependencies. */
  public FcpPrivateLoggerConnector(FcpPrivateLoggerConnectorDeps deps) {
    this.deps = Objects.requireNonNull(deps);
  }

  @Override
  public ListenableFuture<List<UploadOutcome>> upload(
      String privateLogSourceName, DataProvider dataProvider) {
    Objects.requireNonNull(privateLogSourceName);
    Objects.requireNonNull(dataProvider);

    return FluentFuture.from(fetchWorkloadSets(privateLogSourceName))
        .transformAsync(
            workloadSets -> processWorkloadSets(privateLogSourceName, workloadSets, dataProvider),
            directExecutor());
  }

  /**
   * Fetches the workload sets. An {@link IOException} yields an empty list; any other failure,
   * including an exception thrown by the fetcher, surfaces as {@link
   * PrivateLoggerUnavailableException}.
   */
  private ListenableFuture<ImmutableList<WorkloadSet>> fetchWorkloadSets(String logSourceName) {
    logger.atFine().log("Fetching workload sets for source: %s", logSourceName);
    // submitAsync so that a synchronous throw from fetch() is handled like a failed fetch future
    // and upload() never throws.
    return FluentFuture.from(
            Futures.submitAsync(
                () -> deps.confidentialCompute().workloadSetFetcher().fetch(logSourceName),
                directExecutor()))
        .catchingAsync(
            Exception.class,
            e -> {
              logger.atWarning().withCause(e).log("Fetch failed for %s", logSourceName);
              if (e instanceof IOException) {
                return immediateFuture(ImmutableList.of());
              }
              reportIfUnexpected(e);
              return immediateFailedFuture(
                  new PrivateLoggerUnavailableException("Fetch failed for " + logSourceName, e));
            },
            directExecutor());
  }

  /**
   * Kicks off every workload set and returns. Nothing runs in the loop itself: each {@link
   * #processWorkloadSet} call queues that set's read and hands back a future. From there the reads
   * run one after another on the data provider's executor, and the encodes and uploads one after
   * another on the {@code ConfidentialCompute} executor, so one set is usually being read while
   * another is being encoded or uploaded. A failure in one set never affects the others.
   */
  private ListenableFuture<List<UploadOutcome>> processWorkloadSets(
      String logSourceName, ImmutableList<WorkloadSet> workloadSets, DataProvider dataProvider) {
    return Futures.allAsList(
        workloadSets.stream()
            .map(workloadSet -> processWorkloadSet(logSourceName, workloadSet, dataProvider))
            .collect(toImmutableList()));
  }

  /**
   * Returns the outcome for {@code workloadSet}. The returned future only fails with an {@link
   * Error}.
   */
  private ListenableFuture<UploadOutcome> processWorkloadSet(
      String logSourceName, WorkloadSet workloadSet, DataProvider dataProvider) {
    // submitAsync so that a synchronous throw from the encoder fails only this workload set, like a
    // failed future from it would.
    return FluentFuture.from(
            Futures.submitAsync(
                () -> readAndUpload(logSourceName, workloadSet, dataProvider), directExecutor()))
        .catching(
            Exception.class,
            e -> {
              // Once the upload has started, uploadPayload turns every exception into an outcome,
              // so this one happened before anything was uploaded: a failed read or encode.
              // PrivateLoggerImpl rolls the entries back.
              logger.atWarning().withCause(e).log(
                  "Failed to process workloadSet %s", workloadSet.name());
              reportIfUnexpected(e);
              return notContributed(workloadSet.name());
            },
            directExecutor());
  }

  private ListenableFuture<UploadOutcome> readAndUpload(
      String logSourceName, WorkloadSet workloadSet, DataProvider dataProvider) {
    return Futures.transformAsync(
        query(logSourceName, workloadSet, dataProvider),
        entries ->
            entries.isEmpty()
                ? immediateFuture(notContributed(workloadSet.name()))
                : encodeAndUpload(workloadSet, entries),
        directExecutor());
  }

  /** Queries the inputs for {@code workloadSet}. */
  private ListenableFuture<List<LogEntry>> query(
      String logSourceName, WorkloadSet workloadSet, DataProvider dataProvider) {
    QueryTimer timer = new QueryTimer(logSourceName, workloadSet);
    timer.log(timer.builder(Kind.QUERY_STARTED).setDuration(Duration.ZERO));
    // submitAsync so that a synchronous throw from the data provider is logged like a failed query.
    return FluentFuture.from(
            Futures.submitAsync(
                () -> dataProvider.getData(selectorContext(workloadSet)), directExecutor()))
        .transform(
            entries -> {
              if (entries.isEmpty()) {
                logger.atFine().log("No log entries for workloadSet: %s", workloadSet.name());
                timer.log(timer.builder(Kind.QUERY_NO_DATA));
              } else {
                timer.log(
                    timer
                        .builder(Kind.QUERY_COMPLETED)
                        .setInputCount(entries.size())
                        .setInputBytes(entries.stream().mapToLong(e -> e.value().size()).sum()));
              }
              return entries;
            },
            directExecutor())
        .catchingAsync(
            Exception.class,
            e -> {
              timer.log(timer.builder(Kind.QUERY_ERROR).setError(e));
              return immediateFailedFuture(e);
            },
            directExecutor());
  }

  /**
   * Returns the context the data provider selects entries by. SelectorContext is only used for
   * legacy support: PrivateLoggerImpl keys contributions by FederatedComputation.task_name, and the
   * workload set name plays that role. Confidential aggregation is set because the platform's
   * private compute checks expect it on uploads. The fetch id is set as the run id so the data
   * provider can join its own logging with the set's diagnostic events.
   */
  private static SelectorContext selectorContext(WorkloadSet workloadSet) {
    // TODO: b/563220328 - Replace SelectorContext in DataProvider with a leaner type.
    return SelectorContext.newBuilder()
        .setComputationProperties(
            QueryTimeComputationProperties.newBuilder()
                .setRunId(workloadSet.fetchId())
                .setFederated(
                    FederatedComputation.newBuilder()
                        .setTaskName(workloadSet.name())
                        .setConfidentialAggregation(ConfidentialAggregation.getDefaultInstance())))
        .build();
  }

  /** Encodes the entries, then uploads them. */
  private ListenableFuture<UploadOutcome> encodeAndUpload(
      WorkloadSet workloadSet, List<LogEntry> entries) {
    return FluentFuture.from(
            deps.confidentialCompute()
                .payloadEncoder()
                .encode(workloadSet, deps.endorsementOptions(), entries))
        .transformAsync(payload -> uploadPayload(workloadSet, payload), directExecutor());
  }

  /**
   * Uploads the payload and maps the result to an outcome. This is the point of no return: from
   * here on the entries are reported as contributed unless the server aborts the upload. The
   * returned future never fails.
   */
  private ListenableFuture<UploadOutcome> uploadPayload(
      WorkloadSet workloadSet, EncodedPayload payload) {
    // If the upload fails, the uploader aborts the aggregation itself. submitAsync so that a
    // synchronous throw from upload() counts as CONTRIBUTED below, like a failed upload future: the
    // upload may have started before the throw.
    ListenableFuture<Void> uploadFuture =
        Futures.submitAsync(
            () -> deps.confidentialCompute().payloadUploader().upload(workloadSet, payload),
            directExecutor());
    return FluentFuture.from(uploadFuture)
        .transform(
            unused -> {
              logger.atFine().log("Upload succeeded for workloadSet: %s", workloadSet.name());
              return toOutcome(workloadSet.name(), UploadOutcome.Status.CONTRIBUTED);
            },
            directExecutor())
        .catching(
            UploadAbortedException.class,
            e -> {
              // The server rejected the blob, so the entries were not stored and can be uploaded
              // again later.
              logger.atWarning().withCause(e).log(
                  "Upload of %s was aborted; rolling back", workloadSet.name());
              return notContributed(workloadSet.name());
            },
            directExecutor())
        .catching(
            Exception.class,
            e -> {
              logger.atWarning().withCause(e).log(
                  "Upload of %s failed with the outcome unknown; reporting CONTRIBUTED so the"
                      + " entries are never uploaded twice",
                  workloadSet.name());
              reportIfUnexpected(e);
              return toOutcome(workloadSet.name(), UploadOutcome.Status.CONTRIBUTED);
            },
            directExecutor());
  }

  /**
   * Reports a {@link RuntimeException} as a bug. Checked exceptions and a shut-down executor are
   * expected failures; the caller only logs them.
   */
  private void reportIfUnexpected(Exception e) {
    if (e instanceof RuntimeException && !(e instanceof RejectedExecutionException)) {
      deps.confidentialCompute().stabilityMonitor().reportCrash(e);
    }
  }

  /**
   * Times one workload set's query and builds its events, from its creation with the {@link
   * ConfidentialCompute} ticker. The duration runs from the {@code getData} call to its result, so
   * it includes any time the data provider keeps the query queued.
   */
  private final class QueryTimer {
    private final String logSourceName;
    private final WorkloadSet workloadSet;
    private final long startNanos = deps.confidentialCompute().ticker().read();

    QueryTimer(String logSourceName, WorkloadSet workloadSet) {
      this.logSourceName = logSourceName;
      this.workloadSet = workloadSet;
    }

    DiagnosticEvent.Builder builder(Kind kind) {
      return DiagnosticEvent.builder(kind)
          .setLogSourceName(logSourceName)
          .setFetchId(workloadSet.fetchId())
          .setWorkloadSetName(workloadSet.name())
          .setDuration(Duration.ofNanos(deps.confidentialCompute().ticker().read() - startNanos));
    }

    void log(DiagnosticEvent.Builder event) {
      deps.confidentialCompute()
          .stabilityMonitor()
          .logDiagnosticEvent(
              event.setTimestamp(deps.confidentialCompute().timeSource().now()).build());
    }
  }

  private static UploadOutcome toOutcome(String workloadSetName, UploadOutcome.Status status) {
    return UploadOutcome.builder().setTaskName(workloadSetName).setStatus(status).build();
  }

  private static UploadOutcome notContributed(String workloadSetName) {
    return toOutcome(workloadSetName, UploadOutcome.Status.NOT_CONTRIBUTED);
  }
}
