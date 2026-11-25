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
package com.google.fcp.client.confidentialcompute.internal.fetch;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;

import androidx.annotation.Nullable;
import com.google.auto.value.AutoValue;
import com.google.common.base.Ticker;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.confidentialcompute.DeviceAttestationProvider;
import com.google.fcp.client.confidentialcompute.LegacyFcpCompat;
import com.google.fcp.client.confidentialcompute.UploadTokens;
import com.google.fcp.client.confidentialcompute.WorkloadSet;
import com.google.fcp.client.confidentialcompute.WorkloadSetFetcher;
import com.google.fcp.client.confidentialcompute.WorkloadSetSelectorSpec;
import com.google.fcp.client.confidentialcompute.internal.http.FcpHttpClient;
import com.google.fcp.client.confidentialcompute.internal.http.NetworkPhase;
import com.google.fcp.client.confidentialcompute.internal.http.UriEncoding;
import com.google.fcp.client.confidentialcompute.internal.swor.SworTracker;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent.Kind;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import com.google.intelligence.fcp.client.ProdDiagCode;
import com.google.internal.federated.plan.ClientOnlyPlan;
import com.google.internal.federated.plan.EligibilityPolicyEvalSpec;
import com.google.internal.federated.plan.ExampleQuerySpec;
import com.google.internal.federated.plan.PopulationEligibilitySpec;
import com.google.internal.federated.plan.PopulationEligibilitySpec.TaskInfo;
import com.google.internal.federated.plan.PopulationEligibilitySpec.TaskInfo.TaskAssignmentMode;
import com.google.internal.federatedcompute.v1.AttestationMeasurement;
import com.google.internal.federatedcompute.v1.ByteStreamResource;
import com.google.internal.federatedcompute.v1.ClientVersion;
import com.google.internal.federatedcompute.v1.EligibilityEvalTask;
import com.google.internal.federatedcompute.v1.EligibilityEvalTaskCapabilities;
import com.google.internal.federatedcompute.v1.EligibilityEvalTaskRequest;
import com.google.internal.federatedcompute.v1.EligibilityEvalTaskResponse;
import com.google.internal.federatedcompute.v1.ForwardingInfo;
import com.google.internal.federatedcompute.v1.PerformMultipleTaskAssignmentsRequest;
import com.google.internal.federatedcompute.v1.PerformMultipleTaskAssignmentsResponse;
import com.google.internal.federatedcompute.v1.ReportEligibilityEvalTaskResultRequest;
import com.google.internal.federatedcompute.v1.Resource;
import com.google.internal.federatedcompute.v1.ResourceCapabilities;
import com.google.internal.federatedcompute.v1.ResourceCompressionFormat;
import com.google.internal.federatedcompute.v1.StartConfidentialAggregationDataUploadRequest;
import com.google.internal.federatedcompute.v1.StartConfidentialAggregationDataUploadResponse;
import com.google.internal.federatedcompute.v1.TaskAssignment;
import com.google.longrunning.Operation;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.rpc.Code;
import java.io.IOException;
import java.security.SecureRandom;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeoutException;

/**
 * {@link WorkloadSetFetcher} implementation over the FCP HTTP protocol, corresponding to {@code
 * HttpFederatedProtocol} ({@code fcp/client/http/http_federated_protocol.cc}) and the check-in flow
 * in {@code fcp/client/fl_runner.cc}.
 *
 * <p>Checks in via {@code EligibilityEvalCheckin}, extracts the {@code
 * TASK_ASSIGNMENT_MODE_MULTIPLE} tasks from the returned {@link PopulationEligibilitySpec}, reports
 * the eligibility eval result when policies are present, requests assignments via {@code
 * PerformMultipleTaskAssignments}, and for each assigned task starts a confidential aggregation
 * upload session ({@code StartConfidentialAggregationDataUpload}) and downloads its encryption
 * config. (Unlike C++ {@code HttpFederatedProtocol}, which opens the upload session during {@code
 * ReportCompleted}, this fetcher opens it during {@link #fetch} so each returned {@link
 * WorkloadSet} carries upload tokens.) The fetcher does not verify the encryption config; the
 * encoder does, as C++ {@code HttpFederatedProtocol::UploadResult} does.
 *
 * <p>Tasks within their SWOR period according to {@link FcpWorkloadSetFetcherDeps#sworTracker()}
 * are left out of the task assignment request.
 *
 * <p>The task assignment request carries the measurement from {@link
 * FcpWorkloadSetFetcherDeps#deviceAttestationProvider()}, obtained once per fetch when tasks
 * remain.
 *
 * <p>Check-in and assignment {@link DiagnosticEvent}s go to {@link
 * FcpWorkloadSetFetcherDeps#stabilityMonitor()}.
 *
 * <p>A check-in rejection ({@code rejection_info}) or a spec with no multiple-assignment tasks
 * resolves to an empty list. Transport errors or malformed responses before per-task resolution
 * fail the returned future with {@link IOException}. Per-task I/O or timeout failures skip only
 * that workload set (sending a best-effort {@code AbortConfidentialAggregation} if a session was
 * already opened). Server retry windows are ignored; scheduling is managed by the caller.
 */
public final class FcpWorkloadSetFetcher implements WorkloadSetFetcher {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final int MAX_LRO_POLL_ATTEMPTS = 30;

  /**
   * The client version reported at check-in and task assignment. The server maps it to the {@code
   * android_unbundled} runtime. It carries no app or library version.
   */
  public static final String CLIENT_VERSION = "android_unbundled";

  private static final SecureRandom SECURE_RANDOM = new SecureRandom();

  private final String serverUrl;
  private final FcpHttpClient httpClient;
  private final ImmutableMap<String, String> requestHeaders;
  private final Executor executor;

  private final Duration pollingInterval;
  private final SworTracker sworTracker;
  private final StabilityMonitor stabilityMonitor;
  private final Ticker ticker;
  private final TimeSource timeSource;
  private final DeviceAttestationProvider deviceAttestationProvider;

  /**
   * Creates a fetcher that talks to deps.serverUrl() through deps.httpEngine() and runs each fetch
   * on deps.executor().
   */
  public FcpWorkloadSetFetcher(FcpWorkloadSetFetcherDeps deps) {
    this.serverUrl = stripTrailingSlash(deps.serverUrl());
    checkArgument(!this.serverUrl.isEmpty(), "serverUrl must not be empty");
    this.httpClient =
        new FcpHttpClient(deps.httpEngine(), deps.httpConnectTimeout(), deps.httpReadTimeout());
    Map<String, String> mergedHeaders = new HashMap<>(deps.requestHeaders());
    mergedHeaders.put("x-goog-api-key", deps.apiKey());
    this.requestHeaders = ImmutableMap.copyOf(mergedHeaders);
    this.executor = deps.executor();
    this.pollingInterval = deps.pollingInterval();
    this.sworTracker = deps.sworTracker();
    this.stabilityMonitor = deps.stabilityMonitor();
    this.ticker = deps.ticker();
    this.timeSource = deps.timeSource();
    this.deviceAttestationProvider = deps.deviceAttestationProvider();
  }

  /**
   * Resource capabilities advertised on check-in and task assignment requests (see {@code
   * PerformEligibilityEvalTaskRequest} and {@code
   * PerformMultipleTaskAssignmentsAndReportEligibilityEvalResult} in {@code
   * fcp/client/http/http_federated_protocol.cc}).
   */
  private static ResourceCapabilities createResourceCapabilities() {
    return ResourceCapabilities.newBuilder()
        .addSupportedCompressionFormats(ResourceCompressionFormat.RESOURCE_COMPRESSION_FORMAT_GZIP)
        .setSupportsConfidentialAggregation(true)
        .setSupportsAttestationTransparencyVerifier(true)
        .build();
  }

  @Override
  public ListenableFuture<ImmutableList<WorkloadSet>> fetch(String privateLogSourceName) {
    // Futures.submit throws if the executor is shut down; report that like any other failure.
    try {
      return Futures.submit(() -> fetchSynchronously(privateLogSourceName), executor);
    } catch (RejectedExecutionException e) {
      return immediateFailedFuture(e);
    }
  }

  private ImmutableList<WorkloadSet> fetchSynchronously(String privateLogSourceName)
      throws IOException, InterruptedException {
    logger.atFine().log("Starting workloadSet resolution for source: %s", privateLogSourceName);
    // Random and positive; 0 means no fetch id.
    long fetchId = 0;
    while (fetchId == 0) {
      fetchId = SECURE_RANDOM.nextLong() & Long.MAX_VALUE;
    }

    logDiagnosticEvent(
        DiagnosticEvent.builder(Kind.CHECKIN_STARTED), privateLogSourceName, fetchId);
    NetworkPhase checkin = NetworkPhase.start(httpClient, ticker);
    EligibilityEvalTaskResponse eetResponse;
    String taskAssignPrefix;
    ParsedEligibilitySpec parsedSpec;
    try {
      // Step 1: Request EligibilityEvalTask from server
      eetResponse = requestEligibilityEvalTask(privateLogSourceName);
      switch (eetResponse.getResultCase()) {
        case REJECTION_INFO -> {
          // Matches Rejection in HttpFederatedProtocol::HandleEligibilityEvalTaskResponse.
          logger.atFine().log("EET check-in rejected for source: %s", privateLogSourceName);
          logDiagnosticEvent(checkin.event(Kind.CHECKIN_REJECTED), privateLogSourceName, fetchId);
          return ImmutableList.of();
        }
        case NO_ELIGIBILITY_EVAL_CONFIGURED ->
            // Multiple task assignment requires a PopulationEligibilitySpec listing the task names
            // (in C++ fl_runner.cc, NO_ELIGIBILITY_EVAL_CONFIGURED only supports single-task
            // Checkin).
            throw new IOException(
                "Server returned no_eligibility_eval_configured for population "
                    + privateLogSourceName
                    + "; multiple task assignment requires a population eligibility spec listing"
                    + " the tasks");
        case RESULT_NOT_SET ->
            // Matches the default branch in
            // HttpFederatedProtocol::HandleEligibilityEvalTaskResponse.
            throw new IOException(
                "EligibilityEvalTaskResponse has no result set for population "
                    + privateLogSourceName);
        case ELIGIBILITY_EVAL_TASK -> {}
      }
      logDiagnosticEvent(checkin.event(Kind.CHECKIN_RECEIVED), privateLogSourceName, fetchId);

      taskAssignPrefix =
          resolveForwardingPrefix(serverUrl, eetResponse.getTaskAssignmentForwardingInfo());

      // Step 2: Parse the EligibilityEvalTask spec
      parsedSpec =
          parseEligibilityEvalSpec(eetResponse.getEligibilityEvalTask(), privateLogSourceName);
    } catch (IOException | RuntimeException e) {
      logDiagnosticEvent(
          checkin.event(Kind.CHECKIN_ERROR).setError(e), privateLogSourceName, fetchId);
      throw e;
    }
    logDiagnosticEvent(checkin.event(Kind.CHECKIN_COMPLETED), privateLogSourceName, fetchId);

    String checkInSession = eetResponse.getSessionId();
    Map<String, String> taskAssignHeaders = new HashMap<>(this.requestHeaders);
    taskAssignHeaders.putAll(
        eetResponse.getTaskAssignmentForwardingInfo().getExtraRequestHeadersMap());
    // Match C++ HttpFederatedProtocol::HandleEligibilityEvalTaskResponse, which enables
    // ReportEligibilityEvalTaskResult only when population_eligibility_spec.eligibility_policies
    // is non-empty.
    if (parsedSpec.hasEligibilityPolicies()) {
      reportEetResult(privateLogSourceName, checkInSession, fetchId);
    }
    if (parsedSpec.multipleAssignmentTaskNames().isEmpty()) {
      logger.atFine().log(
          "No multiple-assignment tasks in spec for source: %s", privateLogSourceName);
      return ImmutableList.of();
    }

    // Step 3: Perform multiple task assignment
    ImmutableList<WorkloadSet> resolvedWorkloadSets =
        assignAndResolveWorkloadSets(
            privateLogSourceName,
            fetchId,
            checkInSession,
            taskAssignPrefix,
            taskAssignHeaders,
            parsedSpec);

    logger.atFine().log(
        "Resolved %d workloadSets for source: %s",
        resolvedWorkloadSets.size(), privateLogSourceName);
    return resolvedWorkloadSets;
  }

  private EligibilityEvalTaskResponse requestEligibilityEvalTask(String privateLogSourceName)
      throws IOException {
    String encodedLogSource = UriEncoding.encodePathSegment(privateLogSourceName);
    // The device attestation measurement is left unset here, as the C++ client's host sends it
    // empty; the server treats unset and empty the same.
    EligibilityEvalTaskRequest eetRequest =
        EligibilityEvalTaskRequest.newBuilder()
            .setPopulationName(privateLogSourceName)
            .setClientVersion(ClientVersion.newBuilder().setVersionCode(CLIENT_VERSION))
            .setResourceCapabilities(createResourceCapabilities())
            .setEligibilityEvalTaskCapabilities(
                EligibilityEvalTaskCapabilities.newBuilder()
                    .setSupportsMultipleTaskAssignment(true)
                    .setSupportsNativeEets(true))
            .build();

    byte[] eetResponseBytes =
        httpClient.post(
            String.format(
                "%s/v1/eligibilityevaltasks/%s:request?%%24alt=proto", serverUrl, encodedLogSource),
            eetRequest.toByteString(),
            requestHeaders);

    return EligibilityEvalTaskResponse.parseFrom(
        eetResponseBytes, ExtensionRegistryLite.getEmptyRegistry());
  }

  private ParsedEligibilitySpec parseEligibilityEvalSpec(
      EligibilityEvalTask eet, String privateLogSourceName)
      throws IOException, InterruptedException {
    if (!eet.hasPopulationEligibilitySpec()) {
      throw new IOException(
          "EligibilityEvalTask has no population eligibility spec for population "
              + privateLogSourceName);
    }
    byte[] popSpecBytes = downloadResource(eet.getPopulationEligibilitySpec());
    PopulationEligibilitySpec popSpec =
        PopulationEligibilitySpec.parseFrom(popSpecBytes, ExtensionRegistryLite.getEmptyRegistry());

    ImmutableList<String> multipleAssignmentTasks =
        popSpec.getTaskInfoList().stream()
            .filter(taskInfo ->
                taskInfo.getTaskAssignmentMode()
                    == TaskAssignmentMode.TASK_ASSIGNMENT_MODE_MULTIPLE)
            .map(TaskInfo::getTaskName)
            .distinct()
            .collect(toImmutableList());
    ImmutableMap<String, Duration> sworPeriods = getMultipleAssignmentSworPeriods(popSpec);
    // Skip sets that have been uploaded to within their SWOR period, so that no task assignment is
    // requested for them. Sets with no SWOR policy are always requested.
    ImmutableSet<String> withinSworPeriod = sworTracker.workloadSetsWithinPeriod(sworPeriods);
    ImmutableList<String> tasksToRequest =
        multipleAssignmentTasks.stream()
            .filter(taskName -> !withinSworPeriod.contains(taskName))
            .collect(toImmutableList());

    logger.atFine().log(
        "EET check-in succeeded for source %s. Multiple-assignment tasks: %s",
        privateLogSourceName, tasksToRequest);
    return ParsedEligibilitySpec.create(
        tasksToRequest, /* hasEligibilityPolicies= */ popSpec.getEligibilityPoliciesCount() > 0);
  }

  /**
   * Returns the TASK_ASSIGNMENT_MODE_MULTIPLE tasks in {@code popSpec} that have a SWOR policy with
   * a min period, in spec order, mapped to their SWOR periods. Tasks without a SWOR policy are not
   * checked (see {@code ComputeEligibility} in {@code fcp/client/eligibility_decider.cc}).
   */
  private static ImmutableMap<String, Duration> getMultipleAssignmentSworPeriods(
      PopulationEligibilitySpec popSpec) {
    ImmutableMap.Builder<String, Duration> sworPeriods = ImmutableMap.builder();
    for (TaskInfo taskInfo : popSpec.getTaskInfoList()) {
      if (taskInfo.getTaskAssignmentMode() == TaskAssignmentMode.TASK_ASSIGNMENT_MODE_MULTIPLE) {
        getSworPeriod(taskInfo, popSpec)
            .ifPresent(period -> sworPeriods.put(taskInfo.getTaskName(), period));
      }
    }
    return sworPeriods.buildKeepingLast();
  }

  /**
   * Returns the sampling-without-replacement period from {@code taskInfo}'s eligibility policies in
   * {@code popSpec} (see {@code ComputeEligibility} in {@code fcp/client/eligibility_decider.cc}),
   * or empty if none is configured.
   */
  private static Optional<Duration> getSworPeriod(
      TaskInfo taskInfo, PopulationEligibilitySpec popSpec) {
    Optional<Duration> sworPeriod = Optional.empty();
    for (int policyIndex : taskInfo.getEligibilityPolicyIndicesList()) {
      if (policyIndex < 0 || policyIndex >= popSpec.getEligibilityPoliciesCount()) {
        continue;
      }
      EligibilityPolicyEvalSpec policySpec = popSpec.getEligibilityPolicies(policyIndex);
      if (policySpec.getSworPolicy().hasMinPeriod()) {
        sworPeriod =
            Optional.of(
                Duration.ofSeconds(
                    policySpec.getSworPolicy().getMinPeriod().getSeconds(),
                    policySpec.getSworPolicy().getMinPeriod().getNanos()));
      }
    }
    return sworPeriod;
  }

  private ImmutableList<WorkloadSet> assignAndResolveWorkloadSets(
      String privateLogSourceName,
      long fetchId,
      String checkInSession,
      String taskAssignPrefix,
      Map<String, String> taskAssignHeaders,
      ParsedEligibilitySpec parsedSpec)
      throws IOException, InterruptedException {
    logDiagnosticEvent(
        DiagnosticEvent.builder(Kind.ASSIGNMENT_STARTED), privateLogSourceName, fetchId);
    NetworkPhase assignment = NetworkPhase.start(httpClient, ticker);
    List<TaskAssignment> workloadSetAssignments;
    try {
      ListenableFuture<String> measurementFuture =
          deviceAttestationProvider.getAttestationMeasurement(
              privateLogSourceName, checkInSession, fetchId);
      String deviceAttestationMeasurement;
      try {
        deviceAttestationMeasurement = measurementFuture.get();
      } catch (InterruptedException e) {
        measurementFuture.cancel(/* mayInterruptIfRunning= */ true);
        throw e;
      } catch (ExecutionException e) {
        if (e.getCause() instanceof Error error) {
          throw error;
        }
        throw new IOException("Failed to get the device attestation measurement", e.getCause());
      } catch (CancellationException e) {
        throw new IOException("Device attestation measurement was cancelled", e);
      }
      checkNotNull(
          deviceAttestationMeasurement,
          "DeviceAttestationProvider.getAttestationMeasurement() completed with null");
      workloadSetAssignments =
          performMultipleTaskAssignments(
              privateLogSourceName,
              checkInSession,
              taskAssignPrefix,
              taskAssignHeaders,
              parsedSpec,
              deviceAttestationMeasurement);
    } catch (IOException | RuntimeException e) {
      logDiagnosticEvent(
          assignment.event(Kind.ASSIGNMENT_ERROR).setError(e), privateLogSourceName, fetchId);
      throw e;
    }
    logDiagnosticEvent(assignment.event(Kind.ASSIGNMENT_RECEIVED), privateLogSourceName, fetchId);

    if (workloadSetAssignments.isEmpty()) {
      logDiagnosticEvent(assignment.event(Kind.ASSIGNMENT_REJECTED), privateLogSourceName, fetchId);
      return ImmutableList.of();
    }

    // Resolve each assigned workload set independently (see IssueMultipleTaskAssignments in
    // fcp/client/fl_runner.cc) so an I/O or timeout failure on one set does not discard the
    // others. Unchecked exceptions and interrupts fail the whole fetch(), leaving any
    // sessions already opened to expire on the server.
    List<WorkloadSet> resolvedWorkloadSets = new ArrayList<>();
    for (TaskAssignment workloadSetAssignment : workloadSetAssignments) {
      String workloadSetName = workloadSetAssignment.getTaskName();
      logger.atFine().log("Processing workload set assignment: %s", workloadSetName);
      try {
        String aggregationDataPrefix =
            resolveForwardingPrefix(
                taskAssignPrefix, workloadSetAssignment.getAggregationDataForwardingInfo());
        WorkloadSet workloadSet =
            resolveWorkloadSetAssignment(
                WorkloadSetResolutionContext.create(
                    workloadSetAssignment, aggregationDataPrefix, privateLogSourceName, fetchId));
        if (workloadSet != null) {
          resolvedWorkloadSets.add(workloadSet);
        }
      } catch (IOException | TimeoutException e) {
        logger.atWarning().withCause(e).log(
            "Failed to resolve workload set %s; skipping it", workloadSetName);
      }
    }
    // As in IssueMultipleTaskAssignments, completed means every assigned set was resolved.
    if (resolvedWorkloadSets.size() == workloadSetAssignments.size()) {
      logDiagnosticEvent(
          assignment.event(Kind.ASSIGNMENT_COMPLETED), privateLogSourceName, fetchId);
    } else {
      logDiagnosticEvent(
          assignment.event(Kind.ASSIGNMENT_PARTIALLY_COMPLETED), privateLogSourceName, fetchId);
    }
    return ImmutableList.copyOf(resolvedWorkloadSets);
  }

  private void logDiagnosticEvent(
      DiagnosticEvent.Builder event, String privateLogSourceName, long fetchId) {
    stabilityMonitor.logDiagnosticEvent(
        event
            .setTimestamp(timeSource.now())
            .setLogSourceName(privateLogSourceName)
            .setFetchId(fetchId)
            .build());
  }

  private List<TaskAssignment> performMultipleTaskAssignments(
      String privateLogSourceName,
      String checkInSession,
      String taskAssignPrefix,
      Map<String, String> taskAssignHeaders,
      ParsedEligibilitySpec parsedSpec,
      String deviceAttestationMeasurement)
      throws IOException {
    String encodedLogSource = UriEncoding.encodePathSegment(privateLogSourceName);
    String encodedCheckInSession = UriEncoding.encodePathSegment(checkInSession);

    // C++ leaves session_id and population_name unset in the body since they are in the URL.
    PerformMultipleTaskAssignmentsRequest assignRequest =
        PerformMultipleTaskAssignmentsRequest.newBuilder()
            .setSessionId(checkInSession)
            .setPopulationName(privateLogSourceName)
            .setClientVersion(ClientVersion.newBuilder().setVersionCode(CLIENT_VERSION))
            .addAllTaskNames(parsedSpec.multipleAssignmentTaskNames())
            .setResourceCapabilities(createResourceCapabilities())
            // C++ sets the measurement whenever the environment returned one, including "".
            .setAttestationMeasurement(
                AttestationMeasurement.newBuilder().setValue(deviceAttestationMeasurement))
            .build();

    byte[] assignResponseBytes =
        httpClient.post(
            String.format(
                "%s/v1/populations/%s/taskassignments/%s:performmultiple?%%24alt=proto",
                taskAssignPrefix, encodedLogSource, encodedCheckInSession),
            assignRequest.toByteString(),
            taskAssignHeaders);

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.parseFrom(
            assignResponseBytes, ExtensionRegistryLite.getEmptyRegistry());

    logger.atFine().log(
        "PerformMultipleTaskAssignments returned %d assignments.",
        assignResponse.getTaskAssignmentsCount());
    return assignResponse.getTaskAssignmentsList();
  }

  /**
   * Polls a long-running {@link Operation} until done, sleeping {@link
   * FcpWorkloadSetFetcherDeps#pollingInterval()} between polls (see {@code
   * PollOperationResponseUntilDone} and {@code CreateGetOperationRequest} in {@code
   * fcp/client/http/protocol_request_helper.cc}).
   *
   * <p>Blocks the caller's {@link FcpWorkloadSetFetcherDeps#executor()} thread (where the
   * synchronous fetch and HTTP calls already run) with a sleep bounded by {@link
   * #MAX_LRO_POLL_ATTEMPTS}; cancelling the fetch future interrupts the sleep.
   */
  // Thread.sleep(Duration) requires Android API 37 (NewApi).
  @SuppressWarnings("PreferJavaTimeOverload")
  private Operation pollOperationUntilDone(
      Operation initialOp, String baseUrlPrefix, Map<String, String> headers, String operationLabel)
      throws IOException, InterruptedException, TimeoutException {
    Operation op = initialOp;
    int pollAttempts = 0;
    while (!op.getDone()) {
      if (++pollAttempts > MAX_LRO_POLL_ATTEMPTS) {
        throw new TimeoutException("Timed out waiting for " + operationLabel + " operation");
      }
      if (!op.getName().startsWith("operations/")) {
        throw new IOException("Invalid operation name: " + op.getName());
      }
      if (!pollingInterval.isZero()) {
        Thread.sleep(pollingInterval.toMillis());
      }
      // Mirrors C++ CreateGetOperationRequest (protocol_request_helper.cc), which percent-encodes
      // the operation name and always requests $alt=proto.
      byte[] getBytes =
          httpClient.get(
              baseUrlPrefix
                  + "/v1/"
                  + UriEncoding.encodePathSegments(op.getName())
                  + "?%24alt=proto",
              headers);
      op = Operation.parseFrom(getBytes, ExtensionRegistryLite.getEmptyRegistry());
    }
    return op;
  }

  /**
   * Best-effort report; a failed report is logged, with the diag code C++ logs, and never fails the
   * fetch (see {@code PerformMultipleTaskAssignmentsAndReportEligibilityEvalResult} in {@code
   * fcp/client/http/http_federated_protocol.cc}).
   */
  private void reportEetResult(String privateLogSourceName, String checkInSession, long fetchId) {
    try {
      String encodedLogSource = UriEncoding.encodePathSegment(privateLogSourceName);
      String encodedCheckInSession = UriEncoding.encodePathSegment(checkInSession);
      ReportEligibilityEvalTaskResultRequest reportRequest =
          ReportEligibilityEvalTaskResultRequest.newBuilder()
              .setPopulationName(privateLogSourceName)
              .setSessionId(checkInSession)
              .setStatusCode(Code.OK)
              .build();
      var unused =
          httpClient.post(
              String.format(
                  "%s/v1/populations/%s/eligibilityevaltasks/%s:reportresult?%%24alt=proto",
                  serverUrl, encodedLogSource, encodedCheckInSession),
              reportRequest.toByteString(),
              requestHeaders);
    } catch (IOException e) {
      logger.atWarning().withCause(e).log(
          "Failed to report EET result for source: %s", privateLogSourceName);
      stabilityMonitor.logDiagCode(
          ProdDiagCode.HTTP_REPORT_ELIGIBILITY_EVAL_RESULT_REQUEST_FAILED_VALUE, fetchId);
    }
  }

  @AutoValue
  abstract static class WorkloadSetResolutionContext {
    abstract TaskAssignment workloadSetAssignment();

    abstract String aggregationDataPrefix();

    abstract String logSourceName();

    abstract long fetchId();

    static WorkloadSetResolutionContext create(
        TaskAssignment workloadSetAssignment,
        String aggregationDataPrefix,
        String logSourceName,
        long fetchId) {
      return new AutoValue_FcpWorkloadSetFetcher_WorkloadSetResolutionContext(
          workloadSetAssignment, aggregationDataPrefix, logSourceName, fetchId);
    }
  }

  /**
   * Resolves a single {@link TaskAssignment} into a {@link WorkloadSet} (corresponding to {@code
   * HandleMultipleTaskAssignmentsInnerResponse} and {@code CreatePerUploadInfo} in {@code
   * fcp/client/http/http_federated_protocol.cc}).
   */
  @Nullable
  private WorkloadSet resolveWorkloadSetAssignment(WorkloadSetResolutionContext ctx)
      throws IOException, InterruptedException, TimeoutException {
    TaskAssignment workloadSetAssignment = ctx.workloadSetAssignment();
    if (workloadSetAssignment.getAggregationId().isEmpty()
        || workloadSetAssignment.getAuthorizationToken().isEmpty()) {
      logger.atWarning().log("Workload set assignment missing aggregationId or authorizationToken");
      return null;
    }

    // Download and parse the plan and signed endorsements before opening the aggregation session
    // so that any download or parse failure skips this workload set without needing to abort an
    // open session.
    ExampleQuerySpec.ExampleQuery query = extractExampleQuery(workloadSetAssignment);

    byte[] endorsementsBytes = new byte[0];
    if (workloadSetAssignment.getConfidentialAggregationInfo().hasSignedEndorsements()) {
      endorsementsBytes =
          downloadResource(
              workloadSetAssignment.getConfidentialAggregationInfo().getSignedEndorsements());
    }

    // Open the aggregation session during fetch so we can obtain the TEE's encryption config and
    // upload tokens. If the caller does not end up uploading to a returned WorkloadSet, its session
    // expires on the server.
    StartConfidentialAggregationDataUploadResponse uploadResp = startDataUpload(ctx);
    if (uploadResp == null) {
      return null;
    }

    // Resolve the aggregation protocol (submit/abort) and data upload URI prefixes against this
    // workload set's aggregation data prefix. If the aggregation protocol prefix is missing or
    // invalid, we cannot send an abort request, so let the session expire on the server.
    String submitPrefix =
        resolveForwardingPrefix(
            ctx.aggregationDataPrefix(), uploadResp.getAggregationProtocolForwardingInfo());
    String uploadPrefix;
    try {
      uploadPrefix =
          resolveForwardingPrefix(
              ctx.aggregationDataPrefix(), uploadResp.getResource().getDataUploadForwardingInfo());
    } catch (IOException e) {
      abortSession(ctx, uploadResp, submitPrefix, Code.INTERNAL_VALUE);
      throw e;
    }

    byte[] encConfigBytes;
    try {
      encConfigBytes = downloadResource(uploadResp.getEncryptionConfig());
    } catch (IOException | RuntimeException e) {
      abortSession(ctx, uploadResp, submitPrefix, Code.INTERNAL_VALUE);
      throw e;
    }

    try {
      return buildWorkloadSet(
          ctx, uploadResp, endorsementsBytes, encConfigBytes, query, submitPrefix, uploadPrefix);
    } catch (RuntimeException e) {
      abortSession(ctx, uploadResp, submitPrefix, Code.INTERNAL_VALUE);
      throw e;
    }
  }

  private Map<String, String> headersFor(ForwardingInfo info) {
    Map<String, String> headers = new HashMap<>(this.requestHeaders);
    headers.putAll(info.getExtraRequestHeadersMap());
    return headers;
  }

  /**
   * Sends a best-effort abort for a session opened during fetch that will not be returned to the
   * caller. Never throws: a failed abort is logged, with the diag code C++ logs, so that the
   * caller's own error handling runs.
   */
  private void abortSession(
      WorkloadSetResolutionContext ctx,
      StartConfidentialAggregationDataUploadResponse uploadResp,
      String submitPrefix,
      int rpcStatusCode) {
    TaskAssignment workloadSetAssignment = ctx.workloadSetAssignment();
    try {
      Map<String, String> headers = headersFor(uploadResp.getAggregationProtocolForwardingInfo());
      AggregationAbortSender.sendAbort(
          httpClient,
          submitPrefix,
          headers,
          workloadSetAssignment.getAggregationId(),
          uploadResp.getClientToken(),
          rpcStatusCode);
    } catch (IOException | RuntimeException e) {
      logger.atWarning().withCause(e).log(
          "Best-effort abort failed for workload set %s", workloadSetAssignment.getTaskName());
      stabilityMonitor.logDiagCode(
          ProdDiagCode.HTTP_CANCELLATION_OR_ABORT_REQUEST_FAILED_VALUE, ctx.fetchId());
    }
  }

  /** Returns null when the operation completed with an error, i.e. no session was opened. */
  @Nullable
  private StartConfidentialAggregationDataUploadResponse startDataUpload(
      WorkloadSetResolutionContext ctx) throws IOException, InterruptedException, TimeoutException {
    TaskAssignment workloadSetAssignment = ctx.workloadSetAssignment();
    String encodedAggregationId =
        UriEncoding.encodePathSegment(workloadSetAssignment.getAggregationId());
    String encodedAuthToken =
        UriEncoding.encodePathSegment(workloadSetAssignment.getAuthorizationToken());
    String targetUriPrefix = ctx.aggregationDataPrefix();

    Map<String, String> caHeaders =
        headersFor(workloadSetAssignment.getAggregationDataForwardingInfo());

    StartConfidentialAggregationDataUploadRequest startUploadRequest =
        StartConfidentialAggregationDataUploadRequest.newBuilder()
            .setAggregationId(workloadSetAssignment.getAggregationId())
            .build();

    byte[] startUploadRespBytes =
        httpClient.post(
            targetUriPrefix
                + "/v1/confidentialaggregations/"
                + encodedAggregationId
                + "/clients/"
                + encodedAuthToken
                + ":startdataupload?%24alt=proto",
            startUploadRequest.toByteString(),
            caHeaders);

    Operation uploadOp =
        pollOperationUntilDone(
            Operation.parseFrom(startUploadRespBytes, ExtensionRegistryLite.getEmptyRegistry()),
            targetUriPrefix,
            caHeaders,
            "StartConfidentialAggregationDataUpload");

    if (uploadOp.hasError()) {
      logger.atWarning().log(
          "Failed to start confidential data upload for workload set %s: %s",
          workloadSetAssignment.getTaskName(), uploadOp.getError().getMessage());
      return null;
    }

    return StartConfidentialAggregationDataUploadResponse.parseFrom(
        uploadOp.getResponse().getValue(), ExtensionRegistryLite.getEmptyRegistry());
  }

  private WorkloadSet buildWorkloadSet(
      WorkloadSetResolutionContext ctx,
      StartConfidentialAggregationDataUploadResponse uploadResponse,
      byte[] endorsementsBytes,
      byte[] encConfigBytes,
      @Nullable ExampleQuerySpec.ExampleQuery query,
      String submitPrefix,
      String uploadPrefix) {
    TaskAssignment workloadSetAssignment = ctx.workloadSetAssignment();
    ByteStreamResource byteStreamResource = uploadResponse.getResource();

    UploadTokens uploadTokens =
        UploadTokens.builder()
            .setAggregationId(workloadSetAssignment.getAggregationId())
            .setClientToken(uploadResponse.getClientToken())
            .setUploadUri(uploadPrefix)
            .setResourceName(byteStreamResource.getResourceName())
            .setSubmitUriPrefix(submitPrefix)
            .setSubmitExtraHeaders(
                uploadResponse.getAggregationProtocolForwardingInfo().getExtraRequestHeadersMap())
            .setUploadExtraHeaders(
                byteStreamResource.getDataUploadForwardingInfo().getExtraRequestHeadersMap())
            .build();

    WorkloadSetSelectorSpec selectorSpec = WorkloadSetSelectorSpec.of(ByteString.EMPTY);
    LegacyFcpCompat legacyFcpCompat = LegacyFcpCompat.of("");
    if (query != null) {
      // The encoder names the checkpoint tensors after these, as the C++ client does in
      // ExampleQueryPlanEngine::RunPlan (fcp/client/engine/example_query_plan_engine.cc).
      ImmutableMap.Builder<String, String> outputVectorNames = ImmutableMap.builder();
      query
          .getOutputVectorSpecsMap()
          .forEach((outputName, spec) -> outputVectorNames.put(outputName, spec.getVectorName()));
      legacyFcpCompat =
          LegacyFcpCompat.of(
              query.getExampleSelector().getCollectionUri(), outputVectorNames.buildOrThrow());
      ByteString criteriaBytes = query.getExampleSelector().getCriteria().toByteString();
      if (!criteriaBytes.isEmpty()) {
        selectorSpec = WorkloadSetSelectorSpec.of(criteriaBytes);
      }
    }

    return WorkloadSet.builder()
        .setName(workloadSetAssignment.getTaskName())
        .setLogSourceName(ctx.logSourceName())
        .setFetchId(ctx.fetchId())
        .setSelectorSpec(selectorSpec)
        .setLegacyFcpCompat(legacyFcpCompat)
        .setSignedEndorsements(ByteString.copyFrom(endorsementsBytes))
        .setEncryptionConfig(ByteString.copyFrom(encConfigBytes))
        .setUploadTokens(uploadTokens)
        .build();
  }

  /**
   * Extracts the first {@link ExampleQuerySpec.ExampleQuery} from the task's {@link ClientOnlyPlan}
   * (see {@code CreateCheckinResultFromTaskAssignment} in {@code fcp/client/fl_runner.cc}), or
   * returns {@code null} if no plan or query is present.
   */
  @Nullable
  private ExampleQuerySpec.ExampleQuery extractExampleQuery(TaskAssignment workloadSetAssignment)
      throws IOException {
    if (!workloadSetAssignment.hasPlan()) {
      return null;
    }
    byte[] planBytes = downloadResource(workloadSetAssignment.getPlan());
    if (planBytes.length == 0) {
      return null;
    }
    ClientOnlyPlan plan =
        ClientOnlyPlan.parseFrom(planBytes, ExtensionRegistryLite.getEmptyRegistry());
    if (plan.getPhase().getExampleQuerySpec().getExampleQueriesCount() > 0) {
      return plan.getPhase().getExampleQuerySpec().getExampleQueries(0);
    }
    return null;
  }

  private byte[] downloadResource(Resource resource) throws IOException {
    // Mirrors ConvertResourceToUriOrInlineData in fcp/client/http/http_federated_protocol.cc.
    return switch (resource.getResourceCase()) {
      case URI -> {
        if (resource.getUri().isEmpty()) {
          throw new IOException("Resource.uri must be non-empty when set");
        }
        yield httpClient.getResource(resource.getUri());
      }
      case INLINE_RESOURCE -> {
        byte[] data = resource.getInlineResource().getData().toByteArray();
        yield switch (resource.getInlineResource().getCompressionFormat()) {
          case RESOURCE_COMPRESSION_FORMAT_UNSPECIFIED -> data;
          case RESOURCE_COMPRESSION_FORMAT_GZIP -> httpClient.gunzip(data);
          case UNRECOGNIZED -> throw new IOException("Unknown ResourceCompressionFormat");
        };
      }
      // Neither field set is treated as empty inline data, as in the C++ client.
      case RESOURCE_NOT_SET -> new byte[0];
    };
  }

  /**
   * Resolves {@code forwardingInfo.getTargetUriPrefix()} against {@code previousBaseUrl} and strips
   * any trailing slash, matching {@code
   * GetNextTargetUriPrefixAndMaybeUpdateMostRecentForwardingPrefix} in {@code
   * fcp/client/http/http_federated_protocol.cc} and {@code ProtocolRequestCreator::Create} in
   * {@code fcp/client/http/protocol_request_helper.cc}.
   *
   * <p>An absolute prefix is used as-is; a relative prefix (starting with {@code '/'}) is resolved
   * against the scheme and authority of {@code previousBaseUrl}, replacing its path.
   *
   * @throws IOException if {@code target_uri_prefix} is empty
   */
  static String resolveForwardingPrefix(String previousBaseUrl, ForwardingInfo forwardingInfo)
      throws IOException {
    String prefix = forwardingInfo.getTargetUriPrefix();
    if (prefix.isEmpty()) {
      throw new IOException("Missing ForwardingInfo.target_uri_prefix");
    }
    if (prefix.startsWith("/")) {
      int authorityStart = previousBaseUrl.indexOf("//");
      int pathStart = previousBaseUrl.indexOf('/', authorityStart < 0 ? 0 : authorityStart + 2);
      String origin = pathStart < 0 ? previousBaseUrl : previousBaseUrl.substring(0, pathStart);
      prefix = origin + prefix;
    }
    return stripTrailingSlash(prefix);
  }

  /**
   * Returns the base URI to which "/v1/..." suffixes are appended, with a single trailing '/'
   * removed, as {@code JoinBaseUriWithSuffix} in {@code fcp/client/http/http_client_util.cc} does.
   */
  static String stripTrailingSlash(String baseUrl) {
    return baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
  }

  @AutoValue
  abstract static class ParsedEligibilitySpec {
    abstract ImmutableList<String> multipleAssignmentTaskNames();

    abstract boolean hasEligibilityPolicies();

    static ParsedEligibilitySpec create(
        ImmutableList<String> multipleAssignmentTaskNames, boolean hasEligibilityPolicies) {
      return new AutoValue_FcpWorkloadSetFetcher_ParsedEligibilitySpec(
          multipleAssignmentTaskNames, hasEligibilityPolicies);
    }
  }
}
