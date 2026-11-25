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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateCancelledFuture;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import androidx.datastore.guava.GuavaDataStore;
import com.google.common.base.Ticker;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.fcp.client.FakeTimeSource;
import com.google.fcp.client.confidentialcompute.WorkloadSet;
import com.google.fcp.client.confidentialcompute.internal.http.FakeHttpUrlConnectionEngine;
import com.google.fcp.client.confidentialcompute.internal.swor.SworStore;
import com.google.fcp.client.confidentialcompute.internal.swor.SworTracker;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent.Kind;
import com.google.fcp.client.stabilitymonitor.RecordingStabilityMonitor;
import com.google.intelligence.fcp.client.ProdDiagCode;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.intelligence.fcp.confidentialcompute.SignedPayload;
import com.google.internal.federated.plan.ClientOnlyPlan;
import com.google.internal.federated.plan.ClientPhase;
import com.google.internal.federated.plan.EligibilityPolicyEvalSpec;
import com.google.internal.federated.plan.ExampleQuerySpec;
import com.google.internal.federated.plan.ExampleSelector;
import com.google.internal.federated.plan.PopulationEligibilitySpec;
import com.google.internal.federated.plan.PopulationEligibilitySpec.TaskInfo;
import com.google.internal.federated.plan.PopulationEligibilitySpec.TaskInfo.TaskAssignmentMode;
import com.google.internal.federated.plan.SamplingWithoutReplacementPolicy;
import com.google.internal.federatedcompute.v1.AbortConfidentialAggregationRequest;
import com.google.internal.federatedcompute.v1.ByteStreamResource;
import com.google.internal.federatedcompute.v1.ConfidentialEncryptionConfig;
import com.google.internal.federatedcompute.v1.EligibilityEvalTask;
import com.google.internal.federatedcompute.v1.EligibilityEvalTaskRequest;
import com.google.internal.federatedcompute.v1.EligibilityEvalTaskResponse;
import com.google.internal.federatedcompute.v1.ForwardingInfo;
import com.google.internal.federatedcompute.v1.NoEligibilityEvalConfigured;
import com.google.internal.federatedcompute.v1.PerformMultipleTaskAssignmentsRequest;
import com.google.internal.federatedcompute.v1.PerformMultipleTaskAssignmentsResponse;
import com.google.internal.federatedcompute.v1.RejectionInfo;
import com.google.internal.federatedcompute.v1.ReportEligibilityEvalTaskResultRequest;
import com.google.internal.federatedcompute.v1.Resource;
import com.google.internal.federatedcompute.v1.ResourceCompressionFormat;
import com.google.internal.federatedcompute.v1.StartConfidentialAggregationDataUploadResponse;
import com.google.internal.federatedcompute.v1.TaskAssignment;
import com.google.longrunning.Operation;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.rpc.Code;
import com.google.rpc.Status;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.GZIPOutputStream;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

@RunWith(RobolectricTestRunner.class)
public class FcpWorkloadSetFetcherTest {
  @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

  private final Executor executor = directExecutor();
  private final RecordingStabilityMonitor monitor = new RecordingStabilityMonitor();
  private FakeHttpUrlConnectionEngine engine;

  private FcpWorkloadSetFetcher fetcher;

  private static final String API_KEY = "test-api-key";
  private static final String SERVER_URL = "https://fake.server.google.com";
  private static final String PRIVATE_LOG_SOURCE_NAME = "test_population";
  private static final String TASK_1 = "test_source/task1";
  private static final String TASK_2 = "test_source/task2";
  private static final String TASK_3 = "test_source/task3";
  private static final Duration SWOR_PERIOD = Duration.ofDays(7);


  private FcpWorkloadSetFetcher newFetcher(
      FakeHttpUrlConnectionEngine engine, ImmutableMap<String, String> headers) {
    return newFetcher(engine, headers, SworTracker.DISABLED);
  }

  private FcpWorkloadSetFetcher newFetcher(
      FakeHttpUrlConnectionEngine engine,
      ImmutableMap<String, String> headers,
      SworTracker sworTracker) {
    return new FcpWorkloadSetFetcher(depsBuilder(engine, headers, sworTracker).build());
  }

  private FcpWorkloadSetFetcherDeps.Builder depsBuilder(
      FakeHttpUrlConnectionEngine engine,
      ImmutableMap<String, String> headers,
      SworTracker sworTracker) {
    return FcpWorkloadSetFetcherDeps.builder()
        .setApiKey(API_KEY)
        .setServerUrl(SERVER_URL)
        .setHttpEngine(engine)
        .setRequestHeaders(headers)
        .setExecutor(executor)
        .setPollingInterval(Duration.ZERO)
        .setSworTracker(sworTracker)
        .setStabilityMonitor(monitor);
  }

  @Before
  public void setUp() throws Exception {
    engine = new FakeHttpUrlConnectionEngine();
    engine.setDefaultResponse(200, new byte[0]);
    engine.respondTo(":abort", 200, new byte[0]);
    fetcher = newFetcher(engine, ImmutableMap.of());
  }

  @Test
  public void fetch_apiKeyHeaderNotOverriddenByCallerHeaders() throws Exception {
    FcpWorkloadSetFetcher localFetcher =
        newFetcher(
            engine,
            ImmutableMap.of(
                "x-goog-api-key", "caller-api-key",
                "x-other-header", "other-val"));
    engine.respondTo(
        ":request",
        200,
        eetResponseWithSpec(PopulationEligibilitySpec.getDefaultInstance()).toByteArray());

    localFetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    String requestUrl = engine.onlyRequestContaining(":request").url();
    assertThat(engine.onlyRequestContaining(requestUrl).headers())
        .containsEntry("x-goog-api-key", API_KEY);
    assertThat(engine.onlyRequestContaining(requestUrl).headers())
        .containsEntry("x-other-header", "other-val");
  }

  @Test
  public void fetch_twoTasks_returnsBothWorkloadSets() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    TaskAssignment assignment1 = createTaskAssignment(TASK_1, "agg_1", "auth_1");
    TaskAssignment assignment2 = createTaskAssignment(TASK_2, "agg_2", "auth_2");

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(assignment1)
            .addTaskAssignments(assignment2)
            .build();

    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    Operation uploadOp1 = createUploadOperation("key_material_1", "client_token_1", "res_1");
    Operation uploadOp2 = createUploadOperation("key_material_2", "client_token_2", "res_2");

    engine.respondTo("agg_1/clients/auth_1:startdataupload", 200, uploadOp1.toByteArray());
    engine.respondTo("agg_2/clients/auth_2:startdataupload", 200, uploadOp2.toByteArray());

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();
    assertThat(fetchResult).hasSize(2);

    WorkloadSet workloadSet1 = fetchResult.get(0);
    assertThat(workloadSet1.name()).isEqualTo(TASK_1);
    assertThat(workloadSet1.encryptionConfig())
        .isEqualTo(encryptionConfig("key_material_1").toByteString());
    assertThat(workloadSet1.signedEndorsements())
        .isEqualTo(ByteString.copyFromUtf8("signed_endorsements"));
    assertThat(workloadSet1.uploadTokens().aggregationId()).isEqualTo("agg_1");
    assertThat(workloadSet1.uploadTokens().clientToken()).isEqualTo("client_token_1");
    assertThat(workloadSet1.uploadTokens().resourceName()).isEqualTo("res_1");

    WorkloadSet workloadSet2 = fetchResult.get(1);
    assertThat(workloadSet2.name()).isEqualTo(TASK_2);
    assertThat(workloadSet2.encryptionConfig())
        .isEqualTo(encryptionConfig("key_material_2").toByteString());
    assertThat(workloadSet2.uploadTokens().aggregationId()).isEqualTo("agg_2");

    byte[] reportReportBytes = engine.requestsContaining(":reportresult").get(0).body();
    ReportEligibilityEvalTaskResultRequest reportRequest =
        ReportEligibilityEvalTaskResultRequest.parseFrom(
            reportReportBytes, ExtensionRegistryLite.getEmptyRegistry());
    assertThat(reportRequest.getStatusCode()).isEqualTo(Code.OK);
    assertThat(reportRequest.getSessionId()).isEqualTo("session_123");

    int reportResultIdx = -1;
    int performMultipleIdx = -1;
    for (int i = 0; i < engine.requestUrls().size(); i++) {
      if (engine.requestUrls().get(i).contains(":reportresult")) {
        reportResultIdx = i;
      } else if (engine.requestUrls().get(i).contains(":performmultiple")) {
        performMultipleIdx = i;
      }
    }
    assertThat(reportResultIdx).isAtLeast(0);
    assertThat(performMultipleIdx).isAtLeast(0);
    assertThat(reportResultIdx).isLessThan(performMultipleIdx);

    byte[] performMultipleReqBytes = engine.onlyRequestContaining(":performmultiple").body();
    PerformMultipleTaskAssignmentsRequest assignReq =
        PerformMultipleTaskAssignmentsRequest.parseFrom(
            performMultipleReqBytes, ExtensionRegistryLite.getEmptyRegistry());
    assertThat(assignReq.getTaskNamesList()).containsExactly(TASK_1, TASK_2).inOrder();
    assertThat(assignReq.getPopulationName()).isEqualTo(PRIVATE_LOG_SOURCE_NAME);
    assertThat(assignReq.getSessionId()).isEqualTo("session_123");
  }

  @Test
  public void fetch_performMultipleReturnsNoAssignments_returnsEmptyList() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.getDefaultInstance().toByteArray());

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).isEmpty();
    assertThat(engine.anyRequestContains(":start?")).isFalse();
  }

  @Test
  public void fetch_startDataUploadReturnsError_skipsOnlyThatTask() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .addTaskAssignments(createTaskAssignment(TASK_2, "agg_2", "auth_2"))
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    Operation errorOp =
        Operation.newBuilder()
            .setDone(true)
            .setError(
                Status.newBuilder()
                    .setCode(Code.INTERNAL_VALUE)
                    .setMessage("Upload start rejected"))
            .build();
    engine.respondTo("agg_1/clients/auth_1:startdataupload", 200, errorOp.toByteArray());
    engine.respondTo(
        "agg_2/clients/auth_2:startdataupload",
        200,
        createUploadOperation("key_2", "tok_2", "res_2").toByteArray());

    ImmutableList<WorkloadSet> resolved = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).hasSize(1);
    assertThat(resolved.get(0).name()).isEqualTo(TASK_2);
  }

  @Test
  public void fetch_lroTimeout_skipsOnlyThatTask() throws Exception {
    FcpWorkloadSetFetcher fastFetcher =
        new FcpWorkloadSetFetcher(
            FcpWorkloadSetFetcherDeps.builder()
                .setApiKey("test-api-key")
                .setServerUrl(SERVER_URL)
                .setHttpEngine(engine)
                .setRequestHeaders(ImmutableMap.of())
                .setExecutor(executor)
                .setPollingInterval(Duration.ZERO)
                .setSworTracker(SworTracker.DISABLED)
                .build());

    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .addTaskAssignments(createTaskAssignment(TASK_2, "agg_2", "auth_2"))
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    Operation incompleteOp =
        Operation.newBuilder().setDone(false).setName("operations/op-incomplete").build();
    engine.respondTo("agg_1/clients/auth_1:startdataupload", 200, incompleteOp.toByteArray());
    engine.respondTo("operations/op-incomplete", 200, incompleteOp.toByteArray());
    engine.respondTo(
        "agg_2/clients/auth_2:startdataupload",
        200,
        createUploadOperation("key_2", "tok_2", "res_2").toByteArray());

    ImmutableList<WorkloadSet> resolved = fastFetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).hasSize(1);
    assertThat(resolved.get(0).name()).isEqualTo(TASK_2);
  }

  @Test
  public void fetch_http500OnEligibilityEval_failsWithIOException() {
    engine.respondTo(":request", 500, new byte[0]);

    ExecutionException thrown =
        assertThrows(ExecutionException.class, () -> fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get());

    assertThat(thrown).hasCauseThat().isInstanceOf(IOException.class);
  }

  @Test
  public void fetch_ioExceptionResolvingTask2_returnsTask1Only() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .addTaskAssignments(createTaskAssignment(TASK_2, "agg_2", "auth_2"))
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload",
        200,
        createUploadOperation("key_1", "tok_1", "res_1").toByteArray());
    engine.respondTo("agg_2/clients/auth_2:startdataupload", 500, new byte[0]);

    ImmutableList<WorkloadSet> resolved = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).hasSize(1);
    assertThat(resolved.get(0).name()).isEqualTo(TASK_1);
  }

  @Test
  public void fetch_unparseableEncryptionConfig_returnsItUnverifiedWithoutAbort() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    ByteString garbageConfig = ByteString.copyFromUtf8("not an encryption config");
    StartConfidentialAggregationDataUploadResponse uploadResponse =
        createUploadResponse("key_1", "tok_1", "res_1").toBuilder()
            .setEncryptionConfig(
                Resource.newBuilder()
                    .setInlineResource(Resource.InlineResource.newBuilder().setData(garbageConfig)))
            .build();
    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload",
        200,
        wrapUploadResponse(uploadResponse).toByteArray());

    ImmutableList<WorkloadSet> resolved = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    // The fetcher does not verify the config; the encoder does.
    assertThat(resolved).hasSize(1);
    assertThat(resolved.get(0).encryptionConfig()).isEqualTo(garbageConfig);
    assertThat(engine.anyRequestContains(":abort")).isFalse();
  }

  @Test
  public void fetch_assignmentWithoutSignedEndorsements_returnsEmptyEndorsements()
      throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    TaskAssignment assignment =
        createTaskAssignment(TASK_1, "agg_1", "auth_1").toBuilder()
            .clearConfidentialAggregationInfo()
            .build();
    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder().addTaskAssignments(assignment).build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());
    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload",
        200,
        createUploadOperation("key_1", "tok_1", "res_1").toByteArray());

    ImmutableList<WorkloadSet> resolved = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).hasSize(1);
    assertThat(resolved.get(0).signedEndorsements()).isEmpty();
  }

  @Test
  public void fetch_eetRejected_returnsEmptyList() throws Exception {
    EligibilityEvalTaskResponse rejectionResponse =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("session_rejected")
            .setRejectionInfo(RejectionInfo.getDefaultInstance())
            .build();

    engine.respondTo(":request", 200, rejectionResponse.toByteArray());

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).isEmpty();
  }

  @Test
  public void stripTrailingSlash_noSlash_returnsSame() {
    assertThat(FcpWorkloadSetFetcher.stripTrailingSlash("https://b.com"))
        .isEqualTo("https://b.com");
  }

  @Test
  public void stripTrailingSlash_oneSlash_removed() {
    assertThat(FcpWorkloadSetFetcher.stripTrailingSlash("https://a.com/ca/"))
        .isEqualTo("https://a.com/ca");
  }

  @Test
  public void resolveForwardingPrefix_absolutePrefix_usedAsIsWithoutTrailingSlash()
      throws Exception {
    ForwardingInfo forwardingInfo =
        ForwardingInfo.newBuilder().setTargetUriPrefix("https://b.example.com/").build();

    assertThat(
            FcpWorkloadSetFetcher.resolveForwardingPrefix("https://a.example.com", forwardingInfo))
        .isEqualTo("https://b.example.com");
  }

  @Test
  public void resolveForwardingPrefix_rootRelativePrefix_resolvesAgainstPreviousHost()
      throws Exception {
    ForwardingInfo forwardingInfo = ForwardingInfo.newBuilder().setTargetUriPrefix("/").build();

    assertThat(
            FcpWorkloadSetFetcher.resolveForwardingPrefix("https://a.example.com", forwardingInfo))
        .isEqualTo("https://a.example.com");
  }

  @Test
  public void resolveForwardingPrefix_pathRelativePrefix_dropsPreviousPath() throws Exception {
    ForwardingInfo forwardingInfo =
        ForwardingInfo.newBuilder().setTargetUriPrefix("/some/path/").build();

    assertThat(
            FcpWorkloadSetFetcher.resolveForwardingPrefix(
                "https://a.example.com/base/path", forwardingInfo))
        .isEqualTo("https://a.example.com/some/path");
  }

  @Test
  public void resolveForwardingPrefix_emptyPrefix_throwsIOException() {
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                FcpWorkloadSetFetcher.resolveForwardingPrefix(
                    "https://a.example.com", ForwardingInfo.getDefaultInstance()));
    assertThat(e).hasMessageThat().contains("target_uri_prefix");
  }

  @Test
  public void fetch_populationSpecResourceWithEmptyUri_failsWithIOException() {
    EligibilityEvalTaskResponse eetResponse =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("session_empty_uri")
            .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
            .setEligibilityEvalTask(
                EligibilityEvalTask.newBuilder()
                    .setPopulationEligibilitySpec(Resource.newBuilder().setUri("")))
            .build();
    engine.respondTo(":request", 200, eetResponse.toByteArray());

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(e).hasCauseThat().hasMessageThat().contains("must be non-empty");
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
  }

  @Test
  public void fetch_noTasksInSpec_returnsEmptyListWithoutTaskAssignment() throws Exception {
    PopulationEligibilitySpec emptySpec = PopulationEligibilitySpec.getDefaultInstance();

    EligibilityEvalTaskResponse eetResponse =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("empty_session")
            .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
            .setEligibilityEvalTask(
                EligibilityEvalTask.newBuilder()
                    .setPopulationEligibilitySpec(
                        Resource.newBuilder()
                            .setInlineResource(
                                Resource.InlineResource.newBuilder()
                                    .setData(emptySpec.toByteString()))))
            .build();

    engine.respondTo(":request", 200, eetResponse.toByteArray());

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).isEmpty();
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
    assertThat(engine.anyRequestContains(":reportresult")).isFalse();
  }

  @Test
  public void fetch_noEligibilityPolicies_skipsEetReportAndAssignsTasks() throws Exception {
    PopulationEligibilitySpec popSpec =
        PopulationEligibilitySpec.newBuilder().addTaskInfo(multipleAssignmentTask(TASK_1)).build();
    engine.respondTo(":request", 200, eetResponseWithSpec(popSpec).toByteArray());
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.getDefaultInstance().toByteArray());

    fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(engine.anyRequestContains(":reportresult")).isFalse();
    assertThat(engine.anyRequestContains(":performmultiple")).isTrue();
  }

  @Test
  public void fetch_gzipResource_advertisesGzipAndAttestationTransparency() throws Exception {
    PopulationEligibilitySpec popSpec =
        PopulationEligibilitySpec.newBuilder().addTaskInfo(multipleAssignmentTask(TASK_1)).build();
    ByteArrayOutputStream compressedPopSpec = new ByteArrayOutputStream();
    try (GZIPOutputStream gzipOut = new GZIPOutputStream(compressedPopSpec)) {
      popSpec.writeTo(gzipOut);
    }

    EligibilityEvalTaskResponse eetResp =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("session-1")
            .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
            .setEligibilityEvalTask(
                EligibilityEvalTask.newBuilder()
                    .setPopulationEligibilitySpec(
                        Resource.newBuilder()
                            .setInlineResource(
                                Resource.InlineResource.newBuilder()
                                    .setCompressionFormat(
                                        ResourceCompressionFormat.RESOURCE_COMPRESSION_FORMAT_GZIP)
                                    .setData(
                                        ByteString.copyFrom(compressedPopSpec.toByteArray())))))
            .build();

    TaskAssignment assignment = createTaskAssignment(TASK_1, "agg-multi", "auth-multi");
    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder().addTaskAssignments(assignment).build();
    Operation uploadOp = createUploadOperation("key-multi", "client-tok-multi", "res-multi");

    engine.respondTo(":request", 200, eetResp.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());
    engine.respondTo(":startdataupload", 200, uploadOp.toByteArray());

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();
    assertThat(fetchResult).isNotEmpty();

    byte[] eetReqBytes = engine.onlyRequestContaining(":request").body();
    EligibilityEvalTaskRequest capturedEetReq =
        EligibilityEvalTaskRequest.parseFrom(eetReqBytes, ExtensionRegistryLite.getEmptyRegistry());
    assertThat(capturedEetReq.getResourceCapabilities().getSupportedCompressionFormatsList())
        .contains(ResourceCompressionFormat.RESOURCE_COMPRESSION_FORMAT_GZIP);
    assertThat(
            capturedEetReq.getResourceCapabilities().getSupportsAttestationTransparencyVerifier())
        .isTrue();

    assertThat(fetchResult).hasSize(1);
    assertThat(fetchResult.get(0).name()).isEqualTo(TASK_1);
  }

  @Test
  public void fetch_noEligibilityEvalConfigured_failsWithIOException() {
    EligibilityEvalTaskResponse noEetResponse =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("session_no_eet")
            .setNoEligibilityEvalConfigured(NoEligibilityEvalConfigured.getDefaultInstance())
            .build();
    engine.respondTo(":request", 200, noEetResponse.toByteArray());

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);
    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);

    assertThat(thrown).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(thrown).hasCauseThat().hasMessageThat().contains("no_eligibility_eval_configured");
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
  }

  @Test
  public void fetch_extractsSelectionCriteriaFromClientOnlyPlan() throws Exception {
    Any expectedCriteria =
        Any.newBuilder()
            .setTypeUrl("type.googleapis.com/test.SelectionCriteria")
            .setValue(ByteString.copyFromUtf8("sql_filter_bytes_123"))
            .build();

    ClientOnlyPlan clientOnlyPlan =
        ClientOnlyPlan.newBuilder()
            .setPhase(
                ClientPhase.newBuilder()
                    .setExampleQuerySpec(
                        ExampleQuerySpec.newBuilder()
                            .addExampleQueries(
                                ExampleQuerySpec.ExampleQuery.newBuilder()
                                    .setExampleSelector(
                                        ExampleSelector.newBuilder()
                                            .setCollectionUri("app:/test_collection")
                                            .setCriteria(expectedCriteria))
                                    .putOutputVectorSpecs(
                                        "upload_query/entry",
                                        ExampleQuerySpec.OutputVectorSpec.newBuilder()
                                            .setVectorName("upload_query/entry")
                                            .build())
                                    .putOutputVectorSpecs(
                                        "upload_query/confidential_compute_event_time",
                                        ExampleQuerySpec.OutputVectorSpec.newBuilder()
                                            .setVectorName(
                                                "upload_query/confidential_compute_event_time")
                                            .build()))))
            .build();

    PopulationEligibilitySpec popSpec =
        PopulationEligibilitySpec.newBuilder().addTaskInfo(multipleAssignmentTask(TASK_1)).build();

    EligibilityEvalTaskResponse eetResp =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("sess-criteria")
            .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
            .setEligibilityEvalTask(
                EligibilityEvalTask.newBuilder()
                    .setPopulationEligibilitySpec(
                        Resource.newBuilder()
                            .setInlineResource(
                                Resource.InlineResource.newBuilder()
                                    .setData(popSpec.toByteString()))))
            .build();

    TaskAssignment assignment =
        createTaskAssignment(TASK_1, "agg-criteria", "auth-criteria").toBuilder()
            .setPlan(
                Resource.newBuilder()
                    .setInlineResource(
                        Resource.InlineResource.newBuilder()
                            .setData(clientOnlyPlan.toByteString())))
            .build();

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder().addTaskAssignments(assignment).build();
    Operation uploadOp = createUploadOperation("key-criteria", "client-tok-crit", "res-crit");

    engine.respondTo(":request", 200, eetResp.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());
    engine.respondTo(":startdataupload", 200, uploadOp.toByteArray());

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).hasSize(1);
    WorkloadSet ws = fetchResult.get(0);
    assertThat(ws.name()).isEqualTo(TASK_1);
    assertThat(ws.selectorSpec().rawCriteria()).isEqualTo(expectedCriteria.toByteString());
    assertThat(ws.legacyFcpCompat().collectionUri()).isEqualTo("app:/test_collection");
    assertThat(ws.legacyFcpCompat().outputVectorNames())
        .containsExactly(
            "upload_query/entry", "upload_query/entry",
            "upload_query/confidential_compute_event_time",
                "upload_query/confidential_compute_event_time");
  }

  @Test
  public void fetch_reportResultFails_stillAssignsTasks() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 500, new byte[0]);

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    ImmutableList<WorkloadSet> resolved = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).isEmpty();
    assertThat(engine.anyRequestContains(":performmultiple")).isTrue();
    assertThat(monitor.diagCodes())
        .containsExactly(ProdDiagCode.HTTP_REPORT_ELIGIBILITY_EVAL_RESULT_REQUEST_FAILED);
    assertThat(monitor.diagFetchIds())
        .containsExactly(monitor.event(Kind.CHECKIN_STARTED).fetchId());
  }

  @Test
  public void fetch_reportResultThrowsRuntimeException_failsFetch() {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.failWith(":reportresult", new IllegalStateException("report result boom"));

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(IllegalStateException.class);
  }

  @Test
  public void fetch_taskAssignmentForwardingInfo_usesTargetPrefixAndServerHeadersWin()
      throws Exception {
    FcpWorkloadSetFetcher localFetcher =
        newFetcher(
            engine, ImmutableMap.of("x-shared-header", "caller", "x-caller-header", "caller"));
    EligibilityEvalTaskResponse eetResponse =
        buildEetResponseWithTwoTasks().toBuilder()
            .setTaskAssignmentForwardingInfo(
                ForwardingInfo.newBuilder()
                    .setTargetUriPrefix("https://taskassignment.google.com/")
                    .putExtraRequestHeaders("x-shared-header", "server")
                    .putExtraRequestHeaders("x-server-header", "server"))
            .build();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.getDefaultInstance().toByteArray());

    localFetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    String performMultipleUrl = engine.onlyRequestContaining(":performmultiple").url();
    assertThat(performMultipleUrl).startsWith("https://taskassignment.google.com/v1/populations/");
    assertThat(engine.onlyRequestContaining(performMultipleUrl).headers())
        .containsAtLeast(
            "x-goog-api-key", "test-api-key",
            "x-caller-header", "caller",
            "x-shared-header", "server",
            "x-server-header", "server");
    // The forwarding info only applies to task assignment; the EET report goes to the base URL with
    // the caller's headers.
    String reportUrl = engine.onlyRequestContaining(":reportresult").url();
    assertThat(reportUrl).startsWith(SERVER_URL + "/v1/populations/");
    assertThat(engine.onlyRequestContaining(reportUrl).headers())
        .containsEntry("x-shared-header", "caller");
    assertThat(engine.onlyRequestContaining(reportUrl).headers())
        .doesNotContainKey("x-server-header");
  }

  @Test
  public void fetch_relativeTaskAssignmentPrefix_resolvesAgainstServerUrl() throws Exception {
    EligibilityEvalTaskResponse eetResponse =
        buildEetResponseWithTwoTasks().toBuilder()
            .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
            .build();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.getDefaultInstance().toByteArray());

    fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    String performMultipleUrl = engine.onlyRequestContaining(":performmultiple").url();
    assertThat(performMultipleUrl).startsWith(SERVER_URL + "/v1/populations/");
  }

  @Test
  public void fetch_relativeTaskAssignmentPathPrefix_resolvesAgainstServerHost() throws Exception {
    EligibilityEvalTaskResponse eetResponse =
        buildEetResponseWithTwoTasks().toBuilder()
            .setTaskAssignmentForwardingInfo(
                ForwardingInfo.newBuilder().setTargetUriPrefix("/some/path/"))
            .build();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.getDefaultInstance().toByteArray());

    fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    String performMultipleUrl = engine.onlyRequestContaining(":performmultiple").url();
    assertThat(performMultipleUrl).startsWith(SERVER_URL + "/some/path/v1/populations/");
  }

  @Test
  public void fetch_missingTaskAssignmentForwardingInfo_failsWithIOException() {
    EligibilityEvalTaskResponse eetResponse =
        buildEetResponseWithTwoTasks().toBuilder().clearTaskAssignmentForwardingInfo().build();
    engine.respondTo(":request", 200, eetResponse.toByteArray());

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(e).hasCauseThat().hasMessageThat().contains("target_uri_prefix");
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
    assertThat(engine.anyRequestContains(":reportresult")).isFalse();
  }

  @Test
  public void fetch_noTasksAndMissingForwardingInfo_failsWithIOException() {
    PopulationEligibilitySpec emptySpec = PopulationEligibilitySpec.getDefaultInstance();
    EligibilityEvalTaskResponse eetResponse =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("empty_session")
            .setEligibilityEvalTask(
                EligibilityEvalTask.newBuilder()
                    .setPopulationEligibilitySpec(
                        Resource.newBuilder()
                            .setInlineResource(
                                Resource.InlineResource.newBuilder()
                                    .setData(emptySpec.toByteString()))))
            .build();
    engine.respondTo(":request", 200, eetResponse.toByteArray());

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(e).hasCauseThat().hasMessageThat().contains("target_uri_prefix");
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
    assertThat(engine.anyRequestContains(":reportresult")).isFalse();
  }

  @Test
  public void fetch_populationSpecByUri_downloadsAndAssigns() throws Exception {
    PopulationEligibilitySpec popSpec =
        PopulationEligibilitySpec.newBuilder()
            .addTaskInfo(multipleAssignmentTask(TASK_1))
            .addTaskInfo(multipleAssignmentTask(TASK_2))
            .build();
    EligibilityEvalTaskResponse eetResponse =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("session_uri")
            .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
            .setEligibilityEvalTask(
                EligibilityEvalTask.newBuilder()
                    .setPopulationEligibilitySpec(
                        Resource.newBuilder().setUri(SERVER_URL + "/popspec")))
            .build();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo("/popspec", 200, popSpec.toByteArray());
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.getDefaultInstance().toByteArray());

    fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    byte[] performMultipleReqBytes = engine.onlyRequestContaining(":performmultiple").body();
    PerformMultipleTaskAssignmentsRequest assignReq =
        PerformMultipleTaskAssignmentsRequest.parseFrom(
            performMultipleReqBytes, ExtensionRegistryLite.getEmptyRegistry());
    assertThat(assignReq.getTaskNamesList()).containsExactly(TASK_1, TASK_2).inOrder();
  }

  @Test
  public void fetch_populationSpecUnknownCompression_failsWithIOException() {
    EligibilityEvalTaskResponse eetResponse =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("session_unknown_compression")
            .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
            .setEligibilityEvalTask(
                EligibilityEvalTask.newBuilder()
                    .setPopulationEligibilitySpec(
                        Resource.newBuilder()
                            .setInlineResource(
                                Resource.InlineResource.newBuilder()
                                    .setCompressionFormatValue(99)
                                    .setData(ByteString.copyFromUtf8("invalid")))))
            .build();
    engine.respondTo(":request", 200, eetResponse.toByteArray());

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(e).hasCauseThat().hasMessageThat().contains("Unknown ResourceCompressionFormat");
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
  }

  @Test
  public void fetch_populationSpecResourceUnset_returnsEmptyList() throws Exception {
    EligibilityEvalTaskResponse eetResponse =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("session_unset_resource")
            .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
            .setEligibilityEvalTask(
                EligibilityEvalTask.newBuilder()
                    .setPopulationEligibilitySpec(Resource.getDefaultInstance()))
            .build();
    engine.respondTo(":request", 200, eetResponse.toByteArray());

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).isEmpty();
    assertThat(engine.anyRequestContains(":reportresult")).isFalse();
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
  }

  @Test
  public void fetch_mixedAssignmentModes_requestsOnlyMultipleAssignmentTasks() throws Exception {
    PopulationEligibilitySpec popSpec =
        PopulationEligibilitySpec.newBuilder()
            .addTaskInfo(multipleAssignmentTask(TASK_1))
            .addTaskInfo(taskWithMode(TASK_2, TaskAssignmentMode.TASK_ASSIGNMENT_MODE_SINGLE))
            .addTaskInfo(taskWithMode(TASK_3, TaskAssignmentMode.TASK_ASSIGNMENT_MODE_UNSPECIFIED))
            .build();
    engine.respondTo(":request", 200, eetResponseWithSpec(popSpec).toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.getDefaultInstance().toByteArray());

    fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    PerformMultipleTaskAssignmentsRequest assignReq =
        PerformMultipleTaskAssignmentsRequest.parseFrom(
            engine.onlyRequestContaining(":performmultiple").body(),
            ExtensionRegistryLite.getEmptyRegistry());
    assertThat(assignReq.getTaskNamesList()).containsExactly(TASK_1);
  }

  @Test
  public void fetch_noMultipleAssignmentTasks_skipsTaskAssignmentAndReturnsEmptyList()
      throws Exception {
    PopulationEligibilitySpec popSpec =
        PopulationEligibilitySpec.newBuilder()
            .addTaskInfo(taskWithMode(TASK_1, TaskAssignmentMode.TASK_ASSIGNMENT_MODE_SINGLE))
            .addTaskInfo(taskWithMode(TASK_2, TaskAssignmentMode.TASK_ASSIGNMENT_MODE_UNSPECIFIED))
            .build();
    engine.respondTo(":request", 200, eetResponseWithSpec(popSpec).toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).isEmpty();
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
  }

  @Test
  public void fetch_taskWithinSworPeriod_isNotRequestedForAssignment() throws Exception {
    FakeTimeSource timeSource = new FakeTimeSource();
    File file = new File(temporaryFolder.getRoot(), "swor.pb");
    SworStore sworStore =
        new SworStore(new GuavaDataStore.Builder<>(SworStore.serializer(), () -> file).build());
    sworStore.recordUpload(TASK_1, timeSource.now()).get();
    timeSource.advance(Duration.ofDays(1));
    FcpWorkloadSetFetcher sworFetcher =
        newFetcher(engine, ImmutableMap.of(), new SworTracker(sworStore, timeSource));
    engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_2, "agg_2", "auth_2"))
            .build()
            .toByteArray());
    engine.respondTo(
        "agg_2/clients/auth_2:startdataupload",
        200,
        createUploadOperation("key_material_2", "client_token_2", "res_2").toByteArray());

    ImmutableList<WorkloadSet> fetchResult = sworFetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).hasSize(1);
    assertThat(fetchResult.get(0).name()).isEqualTo(TASK_2);
    PerformMultipleTaskAssignmentsRequest assignRequest =
        PerformMultipleTaskAssignmentsRequest.parseFrom(
            engine.onlyRequestContaining(":performmultiple").body(),
            ExtensionRegistryLite.getEmptyRegistry());
    assertThat(assignRequest.getTaskNamesList()).containsExactly(TASK_2);
  }

  @Test
  public void fetch_sworStoreUnreadable_skipsSworTaskAndRequestsTaskWithoutSworPolicy()
      throws Exception {
    File file = temporaryFolder.newFile("unreadable_swor.pb");
    byte[] notAProto = new byte[12];
    Arrays.fill(notAProto, (byte) 0xff);
    Files.write(file.toPath(), notAProto);
    SworStore unreadableStore =
        new SworStore(new GuavaDataStore.Builder<>(SworStore.serializer(), () -> file).build());
    FcpWorkloadSetFetcher sworFetcher =
        newFetcher(
            engine, ImmutableMap.of(), new SworTracker(unreadableStore, new FakeTimeSource()));
    engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_2, "agg_2", "auth_2"))
            .build()
            .toByteArray());
    engine.respondTo(
        "agg_2/clients/auth_2:startdataupload",
        200,
        createUploadOperation("key_material_2", "client_token_2", "res_2").toByteArray());

    ImmutableList<WorkloadSet> fetchResult = sworFetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    // The task with SWOR policy fails closed, but the task without SWOR policy is still requested.
    assertThat(fetchResult).hasSize(1);
    assertThat(fetchResult.get(0).name()).isEqualTo(TASK_2);
    PerformMultipleTaskAssignmentsRequest assignRequest =
        PerformMultipleTaskAssignmentsRequest.parseFrom(
            engine.onlyRequestContaining(":performmultiple").body(),
            ExtensionRegistryLite.getEmptyRegistry());
    assertThat(assignRequest.getTaskNamesList()).containsExactly(TASK_2);
  }

  @Test
  public void fetch_defaultDeps_sendsEmptyDeviceAttestationMeasurement() throws Exception {
    var unused = respondWithTwoTasks(buildEetResponseWithTwoTasks());

    fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    PerformMultipleTaskAssignmentsRequest assignRequest =
        PerformMultipleTaskAssignmentsRequest.parseFrom(
            engine.onlyRequestContaining(":performmultiple").body(),
            ExtensionRegistryLite.getEmptyRegistry());
    assertThat(assignRequest.hasAttestationMeasurement()).isTrue();
    assertThat(assignRequest.getAttestationMeasurement().getValue()).isEmpty();
  }

  @Test
  public void fetch_deviceAttestationProvider_sendsMeasurementBoundToCheckInSession()
      throws Exception {
    List<String> providerArgs = new ArrayList<>();
    List<Long> providerFetchIds = new ArrayList<>();
    FcpWorkloadSetFetcher attestingFetcher =
        new FcpWorkloadSetFetcher(
            depsBuilder(engine, ImmutableMap.of(), SworTracker.DISABLED)
                .setDeviceAttestationProvider((logSourceName, contentBinding, fetchId) -> {
                  providerArgs.add(logSourceName);
                  providerArgs.add(contentBinding);
                  providerFetchIds.add(fetchId);
                  return immediateFuture("measurement");
                })
                .build());
    var unused = respondWithTwoTasks(buildEetResponseWithTwoTasks());

    attestingFetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(providerArgs).containsExactly(PRIVATE_LOG_SOURCE_NAME, "session_123").inOrder();
    assertThat(providerFetchIds).containsExactly(monitor.event(Kind.ASSIGNMENT_STARTED).fetchId());
    PerformMultipleTaskAssignmentsRequest assignRequest =
        PerformMultipleTaskAssignmentsRequest.parseFrom(
            engine.onlyRequestContaining(":performmultiple").body(),
            ExtensionRegistryLite.getEmptyRegistry());
    assertThat(assignRequest.getAttestationMeasurement().getValue()).isEqualTo("measurement");
  }

  @Test
  public void fetch_deviceAttestationProviderFails_failsWithIOExceptionBeforeTaskAssignment() {
    IllegalStateException providerFailure = new IllegalStateException("no measurement");
    FcpWorkloadSetFetcher attestingFetcher =
        new FcpWorkloadSetFetcher(
            depsBuilder(engine, ImmutableMap.of(), SworTracker.DISABLED)
                .setDeviceAttestationProvider((logSourceName, contentBinding, fetchId) ->
                    immediateFailedFuture(providerFailure))
                .build());
    engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    ListenableFuture<ImmutableList<WorkloadSet>> future =
        attestingFetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(e.getCause()).hasCauseThat().isSameInstanceAs(providerFailure);
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
    assertThat(monitor.kinds())
        .containsExactly(
            Kind.CHECKIN_STARTED,
            Kind.CHECKIN_RECEIVED,
            Kind.CHECKIN_COMPLETED,
            Kind.ASSIGNMENT_STARTED,
            Kind.ASSIGNMENT_ERROR)
        .inOrder();
    assertThat(monitor.event(Kind.ASSIGNMENT_ERROR).error()).isSameInstanceAs(e.getCause());
  }

  @Test
  public void fetch_deviceAttestationProviderFailsWithError_propagatesErrorWithoutWrapping() {
    AssertionError providerError = new AssertionError("fatal attestation error");
    FcpWorkloadSetFetcher attestingFetcher =
        new FcpWorkloadSetFetcher(
            depsBuilder(engine, ImmutableMap.of(), SworTracker.DISABLED)
                .setDeviceAttestationProvider((logSourceName, contentBinding, fetchId) ->
                    immediateFailedFuture(providerError))
                .build());
    engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    ListenableFuture<ImmutableList<WorkloadSet>> future =
        attestingFetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isSameInstanceAs(providerError);
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
    assertThat(monitor.kinds())
        .containsExactly(
            Kind.CHECKIN_STARTED,
            Kind.CHECKIN_RECEIVED,
            Kind.CHECKIN_COMPLETED,
            Kind.ASSIGNMENT_STARTED)
        .inOrder();
  }

  @Test
  public void fetch_cancelledWhileWaitingForDeviceAttestation_cancelsProviderFuture()
      throws Exception {
    SettableFuture<String> providerFuture = SettableFuture.create();
    CountDownLatch providerCalled = new CountDownLatch(1);
    CountDownLatch providerCancelled = new CountDownLatch(1);
    providerFuture.addListener(
        () -> {
          if (providerFuture.isCancelled()) {
            providerCancelled.countDown();
          }
        },
        directExecutor());
    ExecutorService worker = Executors.newSingleThreadExecutor();
    try {
      FcpWorkloadSetFetcher attestingFetcher =
          new FcpWorkloadSetFetcher(
              depsBuilder(engine, ImmutableMap.of(), SworTracker.DISABLED)
                  .setExecutor(worker)
                  .setDeviceAttestationProvider((logSourceName, contentBinding, fetchId) -> {
                    providerCalled.countDown();
                    return providerFuture;
                  })
                  .build());
      engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
      engine.respondTo(":reportresult", 200, new byte[0]);

      ListenableFuture<ImmutableList<WorkloadSet>> future =
          attestingFetcher.fetch(PRIVATE_LOG_SOURCE_NAME);
      assertThat(providerCalled.await(5, SECONDS)).isTrue();

      assertThat(future.cancel(/* mayInterruptIfRunning= */ true)).isTrue();

      assertThat(providerCancelled.await(5, SECONDS)).isTrue();
      assertThat(providerFuture.isCancelled()).isTrue();
      assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
    } finally {
      worker.shutdownNow();
    }
  }

  @Test
  public void fetch_deviceAttestationProviderCancelled_failsWithIOExceptionBeforeTaskAssignment() {
    FcpWorkloadSetFetcher attestingFetcher =
        new FcpWorkloadSetFetcher(
            depsBuilder(engine, ImmutableMap.of(), SworTracker.DISABLED)
                .setDeviceAttestationProvider(
                    (logSourceName, contentBinding, fetchId) -> immediateCancelledFuture())
                .build());
    engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    ListenableFuture<ImmutableList<WorkloadSet>> future =
        attestingFetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(e.getCause()).hasCauseThat().isInstanceOf(CancellationException.class);
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
    assertThat(monitor.event(Kind.ASSIGNMENT_ERROR).error()).isSameInstanceAs(e.getCause());
  }

  @Test
  public void fetch_deviceAttestationProviderReturnsNull_failsWithNullPointerException() {
    FcpWorkloadSetFetcher attestingFetcher =
        new FcpWorkloadSetFetcher(
            depsBuilder(engine, ImmutableMap.of(), SworTracker.DISABLED)
                .setDeviceAttestationProvider(
                    (logSourceName, contentBinding, fetchId) -> immediateFuture(null))
                .build());
    engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    ListenableFuture<ImmutableList<WorkloadSet>> future =
        attestingFetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(NullPointerException.class);
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
  }

  @Test
  public void fetch_noEligibilityPolicies_callsDeviceAttestationProviderWithCheckInSession()
      throws Exception {
    List<String> contentBindings = new ArrayList<>();
    FcpWorkloadSetFetcher attestingFetcher =
        new FcpWorkloadSetFetcher(
            depsBuilder(engine, ImmutableMap.of(), SworTracker.DISABLED)
                .setDeviceAttestationProvider((logSourceName, contentBinding, fetchId) -> {
                  contentBindings.add(contentBinding);
                  return immediateFuture("measurement");
                })
                .build());
    PopulationEligibilitySpec popSpec =
        PopulationEligibilitySpec.newBuilder().addTaskInfo(multipleAssignmentTask(TASK_1)).build();
    engine.respondTo(":request", 200, eetResponseWithSpec(popSpec).toByteArray());
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.getDefaultInstance().toByteArray());

    attestingFetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(contentBindings).containsExactly("session_123");
  }

  @Test
  public void fetch_noTasksRemain_doesNotCallDeviceAttestationProvider() throws Exception {
    List<String> providerArgs = new ArrayList<>();
    FcpWorkloadSetFetcher attestingFetcher =
        new FcpWorkloadSetFetcher(
            depsBuilder(engine, ImmutableMap.of(), SworTracker.DISABLED)
                .setDeviceAttestationProvider((logSourceName, contentBinding, fetchId) -> {
                  providerArgs.add(contentBinding);
                  return immediateFuture("measurement");
                })
                .build());
    engine.respondTo(
        ":request",
        200,
        eetResponseWithSpec(PopulationEligibilitySpec.getDefaultInstance()).toByteArray());

    ImmutableList<WorkloadSet> fetchResult = attestingFetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).isEmpty();
    assertThat(providerArgs).isEmpty();
  }

  @Test
  public void fetch_http500OnPerformMultiple_failsWithIOException() {
    engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(":performmultiple", 500, new byte[0]);

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(IOException.class);
  }

  @Test
  public void fetch_eligibilityEvalTaskWithoutPopulationSpec_failsWithIOException() {
    EligibilityEvalTaskResponse eetResponse =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("session_no_spec")
            .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
            .setEligibilityEvalTask(EligibilityEvalTask.getDefaultInstance())
            .build();
    engine.respondTo(":request", 200, eetResponse.toByteArray());

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(e).hasCauseThat().hasMessageThat().contains("no population eligibility spec");
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
    assertThat(engine.anyRequestContains(":reportresult")).isFalse();
  }

  @Test
  public void fetch_twoTasks_logsCheckinAndAssignmentEvents() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    PerformMultipleTaskAssignmentsResponse assignResponse = respondWithTwoTasks(eetResponse);

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).hasSize(2);
    long fetchId = fetchResult.get(0).fetchId();
    assertThat(fetchId).isNotEqualTo(0L);
    assertThat(fetchResult.get(1).fetchId()).isEqualTo(fetchId);
    assertThat(fetchResult.get(0).logSourceName()).isEqualTo(PRIVATE_LOG_SOURCE_NAME);
    assertThat(monitor.kinds())
        .containsExactly(
            Kind.CHECKIN_STARTED,
            Kind.CHECKIN_RECEIVED,
            Kind.CHECKIN_COMPLETED,
            Kind.ASSIGNMENT_STARTED,
            Kind.ASSIGNMENT_RECEIVED,
            Kind.ASSIGNMENT_COMPLETED)
        .inOrder();
    for (DiagnosticEvent event : monitor.events()) {
      assertThat(event.logSourceName()).isEqualTo(PRIVATE_LOG_SOURCE_NAME);
      assertThat(event.fetchId()).isEqualTo(fetchId);
      assertThat(event.workloadSetName()).isNull();
      assertThat(event.error()).isNull();
      assertThat(event.duration().isNegative()).isFalse();
    }

    DiagnosticEvent checkinReceived = monitor.event(Kind.CHECKIN_RECEIVED);
    assertThat(checkinReceived.bytesUploaded())
        .isEqualTo(engine.onlyRequestContaining(":request").body().length);
    assertThat(checkinReceived.bytesDownloaded()).isEqualTo(eetResponse.getSerializedSize());
    // The spec is inline, so the spec download adds no bytes.
    DiagnosticEvent checkinCompleted = monitor.event(Kind.CHECKIN_COMPLETED);
    assertThat(checkinCompleted.bytesUploaded()).isEqualTo(checkinReceived.bytesUploaded());
    assertThat(checkinCompleted.bytesDownloaded()).isEqualTo(checkinReceived.bytesDownloaded());

    DiagnosticEvent assignmentReceived = monitor.event(Kind.ASSIGNMENT_RECEIVED);
    assertThat(assignmentReceived.bytesUploaded())
        .isEqualTo(engine.onlyRequestContaining(":performmultiple").body().length);
    assertThat(assignmentReceived.bytesDownloaded()).isEqualTo(assignResponse.getSerializedSize());
    // Resolving the sets starts their upload sessions.
    DiagnosticEvent assignmentCompleted = monitor.event(Kind.ASSIGNMENT_COMPLETED);
    assertThat(assignmentCompleted.bytesUploaded())
        .isGreaterThan(assignmentReceived.bytesUploaded());
    assertThat(assignmentCompleted.bytesDownloaded())
        .isGreaterThan(assignmentReceived.bytesDownloaded());
  }

  @Test
  public void fetch_eetRejected_logsCheckinRejected() throws Exception {
    EligibilityEvalTaskResponse rejectionResponse =
        EligibilityEvalTaskResponse.newBuilder()
            .setSessionId("session_rejected")
            .setRejectionInfo(RejectionInfo.getDefaultInstance())
            .build();
    engine.respondTo(":request", 200, rejectionResponse.toByteArray());

    var unused = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(monitor.kinds())
        .containsExactly(Kind.CHECKIN_STARTED, Kind.CHECKIN_REJECTED)
        .inOrder();
    assertThat(monitor.event(Kind.CHECKIN_REJECTED).bytesDownloaded())
        .isEqualTo(rejectionResponse.getSerializedSize());
  }

  @Test
  public void fetch_eventsUseTheInjectedTickerAndTimeSource() {
    AtomicLong nanos = new AtomicLong();
    Ticker ticker =
        new Ticker() {
          @Override
          public long read() {
            return nanos.getAndAdd(Duration.ofMillis(7).toNanos());
          }
        };
    Instant now = Instant.parse("2026-11-12T12:00:00Z");
    FcpWorkloadSetFetcher timedFetcher =
        new FcpWorkloadSetFetcher(
            FcpWorkloadSetFetcherDeps.builder()
                .setApiKey(API_KEY)
                .setServerUrl(SERVER_URL)
                .setHttpEngine(engine)
                .setRequestHeaders(ImmutableMap.of())
                .setExecutor(executor)
                .setPollingInterval(Duration.ZERO)
                .setSworTracker(SworTracker.DISABLED)
                .setStabilityMonitor(monitor)
                .setTicker(ticker)
                .setTimeSource(new FakeTimeSource(now))
                .build());
    engine.respondTo(":request", 500, new byte[0]);

    assertThrows(ExecutionException.class, () -> timedFetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get());

    DiagnosticEvent error = monitor.event(Kind.CHECKIN_ERROR);
    assertThat(error.duration()).isEqualTo(Duration.ofMillis(7));
    assertThat(error.timestamp()).isEqualTo(now);
  }

  @Test
  public void fetch_http500OnEligibilityEval_logsCheckinError() {
    engine.respondTo(":request", 500, new byte[0]);

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(monitor.kinds()).containsExactly(Kind.CHECKIN_STARTED, Kind.CHECKIN_ERROR).inOrder();
    DiagnosticEvent error = monitor.event(Kind.CHECKIN_ERROR);
    assertThat(error.error()).isSameInstanceAs(thrown.getCause());
    assertThat(error.logSourceName()).isEqualTo(PRIVATE_LOG_SOURCE_NAME);
    assertThat(error.bytesUploaded())
        .isEqualTo(engine.onlyRequestContaining(":request").body().length);
  }

  @Test
  public void fetch_runtimeExceptionOnEligibilityEval_logsCheckinError() {
    engine.failWith(":request", new IllegalStateException("boom"));

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(monitor.kinds()).containsExactly(Kind.CHECKIN_STARTED, Kind.CHECKIN_ERROR).inOrder();
    assertThat(monitor.event(Kind.CHECKIN_ERROR).error()).isSameInstanceAs(thrown.getCause());
  }

  @Test
  public void fetch_http500OnPerformMultiple_logsAssignmentError() {
    engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(":performmultiple", 500, new byte[0]);

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(monitor.kinds())
        .containsExactly(
            Kind.CHECKIN_STARTED,
            Kind.CHECKIN_RECEIVED,
            Kind.CHECKIN_COMPLETED,
            Kind.ASSIGNMENT_STARTED,
            Kind.ASSIGNMENT_ERROR)
        .inOrder();
    assertThat(monitor.event(Kind.ASSIGNMENT_ERROR).error()).isSameInstanceAs(thrown.getCause());
  }

  @Test
  public void fetch_oneTaskFailsToResolve_logsAssignmentPartiallyCompleted() throws Exception {
    engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .addTaskAssignments(createTaskAssignment(TASK_2, "agg_2", "auth_2"))
            .build()
            .toByteArray());
    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload",
        200,
        createUploadOperation("key_1", "tok_1", "res_1").toByteArray());
    engine.respondTo("agg_2/clients/auth_2:startdataupload", 500, new byte[0]);

    ImmutableList<WorkloadSet> resolved = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).hasSize(1);
    assertThat(monitor.kinds())
        .containsExactly(
            Kind.CHECKIN_STARTED,
            Kind.CHECKIN_RECEIVED,
            Kind.CHECKIN_COMPLETED,
            Kind.ASSIGNMENT_STARTED,
            Kind.ASSIGNMENT_RECEIVED,
            Kind.ASSIGNMENT_PARTIALLY_COMPLETED)
        .inOrder();
  }

  @Test
  public void fetch_zeroTasksAssigned_logsAssignmentRejected() throws Exception {
    engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.getDefaultInstance().toByteArray());

    ImmutableList<WorkloadSet> resolved = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).isEmpty();
    assertThat(monitor.kinds())
        .containsExactly(
            Kind.CHECKIN_STARTED,
            Kind.CHECKIN_RECEIVED,
            Kind.CHECKIN_COMPLETED,
            Kind.ASSIGNMENT_STARTED,
            Kind.ASSIGNMENT_RECEIVED,
            Kind.ASSIGNMENT_REJECTED)
        .inOrder();
  }

  /** Serves a check-in with {@code eetResponse} and assigns and opens sessions for two tasks. */
  private PerformMultipleTaskAssignmentsResponse respondWithTwoTasks(
      EligibilityEvalTaskResponse eetResponse) throws Exception {
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .addTaskAssignments(createTaskAssignment(TASK_2, "agg_2", "auth_2"))
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());
    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload",
        200,
        createUploadOperation("key_material_1", "client_token_1", "res_1").toByteArray());
    engine.respondTo(
        "agg_2/clients/auth_2:startdataupload",
        200,
        createUploadOperation("key_material_2", "client_token_2", "res_2").toByteArray());
    return assignResponse;
  }

  private static TaskInfo multipleAssignmentTask(String taskName) {
    return taskWithMode(taskName, TaskAssignmentMode.TASK_ASSIGNMENT_MODE_MULTIPLE);
  }

  private static TaskInfo taskWithMode(String taskName, TaskAssignmentMode mode) {
    return TaskInfo.newBuilder().setTaskName(taskName).setTaskAssignmentMode(mode).build();
  }

  private static EligibilityEvalTaskResponse eetResponseWithSpec(PopulationEligibilitySpec spec) {
    return EligibilityEvalTaskResponse.newBuilder()
        .setSessionId("session_123")
        .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
        .setEligibilityEvalTask(
            EligibilityEvalTask.newBuilder()
                .setPopulationEligibilitySpec(
                    Resource.newBuilder()
                        .setInlineResource(
                            Resource.InlineResource.newBuilder().setData(spec.toByteString()))))
        .build();
  }

  private EligibilityEvalTaskResponse buildEetResponseWithTwoTasks() {
    com.google.protobuf.Duration protoDur =
        com.google.protobuf.Duration.newBuilder()
            .setSeconds(SWOR_PERIOD.getSeconds())
            .setNanos(SWOR_PERIOD.getNano())
            .build();

    EligibilityPolicyEvalSpec sworPolicySpec =
        EligibilityPolicyEvalSpec.newBuilder()
            .setName("swor_policy")
            .setSworPolicy(
                SamplingWithoutReplacementPolicy.newBuilder().setMinPeriod(protoDur).build())
            .build();

    PopulationEligibilitySpec popSpec =
        PopulationEligibilitySpec.newBuilder()
            .addEligibilityPolicies(sworPolicySpec)
            .addTaskInfo(
                TaskInfo.newBuilder()
                    .setTaskName(TASK_1)
                    .addEligibilityPolicyIndices(0)
                    .setTaskAssignmentMode(TaskAssignmentMode.TASK_ASSIGNMENT_MODE_MULTIPLE))
            .addTaskInfo(
                TaskInfo.newBuilder()
                    .setTaskName(TASK_2)
                    .setTaskAssignmentMode(TaskAssignmentMode.TASK_ASSIGNMENT_MODE_MULTIPLE))
            .addEligibilityPolicies(
                EligibilityPolicyEvalSpec.newBuilder()
                    .setSworPolicy(SamplingWithoutReplacementPolicy.getDefaultInstance()))
            .build();

    return EligibilityEvalTaskResponse.newBuilder()
        .setSessionId("session_123")
        .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
        .setEligibilityEvalTask(
            EligibilityEvalTask.newBuilder()
                .setPopulationEligibilitySpec(
                    Resource.newBuilder()
                        .setInlineResource(
                            Resource.InlineResource.newBuilder().setData(popSpec.toByteString()))))
        .build();
  }

  private TaskAssignment createTaskAssignment(
      String taskName, String aggregationId, String authToken) {
    return TaskAssignment.newBuilder()
        .setTaskName(taskName)
        .setAggregationId(aggregationId)
        .setAuthorizationToken(authToken)
        .setAggregationDataForwardingInfo(
            ForwardingInfo.newBuilder().setTargetUriPrefix(SERVER_URL))
        .setConfidentialAggregationInfo(
            TaskAssignment.ConfidentialAggregationInfo.newBuilder()
                .setSignedEndorsements(
                    Resource.newBuilder()
                        .setInlineResource(
                            Resource.InlineResource.newBuilder()
                                .setData(ByteString.copyFromUtf8("signed_endorsements")))))
        .build();
  }

  private static ConfidentialEncryptionConfig encryptionConfig(String keyMaterial) {
    Key key =
        Key.newBuilder()
            .setAlgorithm(Key.Algorithm.HPKE_X25519_SHA256_AES128_GCM)
            .setKeyMaterial(ByteString.copyFromUtf8(keyMaterial))
            .setKeyId(ByteString.copyFromUtf8("tee-key-id"))
            .build();
    return ConfidentialEncryptionConfig.newBuilder()
        .setEncryptionKey(SignedPayload.newBuilder().setPayload(key.toByteString()))
        .build();
  }

  private StartConfidentialAggregationDataUploadResponse createUploadResponse(
      String keyMaterial, String clientToken, String resourceName) {
    return StartConfidentialAggregationDataUploadResponse.newBuilder()
        .setClientToken(clientToken)
        .setEncryptionConfig(
            Resource.newBuilder()
                .setInlineResource(
                    Resource.InlineResource.newBuilder()
                        .setData(encryptionConfig(keyMaterial).toByteString())))
        .setResource(
            ByteStreamResource.newBuilder()
                .setDataUploadForwardingInfo(
                    ForwardingInfo.newBuilder().setTargetUriPrefix(SERVER_URL))
                .setResourceName(resourceName))
        .setAggregationProtocolForwardingInfo(
            ForwardingInfo.newBuilder().setTargetUriPrefix(SERVER_URL))
        .build();
  }

  private Operation wrapUploadResponse(StartConfidentialAggregationDataUploadResponse response) {
    return Operation.newBuilder()
        .setDone(true)
        .setResponse(
            Any.newBuilder()
                .setTypeUrl(
                    "type.googleapis.com/com.google.internal.federatedcompute.v1.StartConfidentialAggregationDataUploadResponse")
                .setValue(response.toByteString())
                .build())
        .build();
  }

  private Operation createUploadOperation(
      String keyMaterial, String clientToken, String resourceName) {
    return wrapUploadResponse(createUploadResponse(keyMaterial, clientToken, resourceName));
  }

  private AbortConfidentialAggregationRequest abortRequestFor(String urlFragment) throws Exception {
    byte[] abortReqBytes = engine.requestsContaining(urlFragment).get(0).body();
    return AbortConfidentialAggregationRequest.parseFrom(
        abortReqBytes, ExtensionRegistryLite.getEmptyRegistry());
  }

  private static StartConfidentialAggregationDataUploadResponse extractUploadResponse(
      Operation op) {
    try {
      return StartConfidentialAggregationDataUploadResponse.parseFrom(
          op.getResponse().getValue(), ExtensionRegistryLite.getEmptyRegistry());
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  @Test
  public void fetch_executorShutDown_returnsFailedFuture() {
    ExecutorService shutDownExecutor = Executors.newSingleThreadExecutor();
    shutDownExecutor.shutdown();
    FcpWorkloadSetFetcher fetcherWithShutDownExecutor =
        new FcpWorkloadSetFetcher(
            FcpWorkloadSetFetcherDeps.builder()
                .setApiKey(API_KEY)
                .setServerUrl(SERVER_URL)
                .setHttpEngine(engine)
                .setRequestHeaders(ImmutableMap.of())
                .setExecutor(shutDownExecutor)
                .setPollingInterval(Duration.ZERO)
                .setSworTracker(SworTracker.DISABLED)
                .build());

    ListenableFuture<ImmutableList<WorkloadSet>> future =
        fetcherWithShutDownExecutor.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(RejectedExecutionException.class);
    assertThat(engine.requests()).isEmpty();
  }

  @Test
  public void fetch_lroPolling_encodesPathSegmentsAndRequestsProtoAlt() throws Exception {
    FcpWorkloadSetFetcher fastFetcher =
        new FcpWorkloadSetFetcher(
            FcpWorkloadSetFetcherDeps.builder()
                .setApiKey("test-api-key")
                .setServerUrl(SERVER_URL)
                .setHttpEngine(engine)
                .setRequestHeaders(ImmutableMap.of())
                .setExecutor(executor)
                .setPollingInterval(Duration.ZERO)
                .setSworTracker(SworTracker.DISABLED)
                .build());

    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    Operation pendingOp =
        Operation.newBuilder().setDone(false).setName("operations/sessions/abc def").build();
    engine.respondTo("agg_1/clients/auth_1:startdataupload", 200, pendingOp.toByteArray());

    Operation doneOp = createUploadOperation("key_1", "tok_1", "res_1");
    engine.respondTo("operations/sessions/abc%20def", 200, doneOp.toByteArray());

    ImmutableList<WorkloadSet> resolved = fastFetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).hasSize(1);
    assertThat(resolved.get(0).name()).isEqualTo(TASK_1);

    String expectedPollUrl = SERVER_URL + "/v1/operations/sessions/abc%20def?%24alt=proto";
    assertThat(engine.requestUrls()).contains(expectedPollUrl);
  }

  @Test
  public void fetch_eetResponseWithNoResultSet_failsWithIOException() {
    EligibilityEvalTaskResponse emptyEetResponse =
        EligibilityEvalTaskResponse.newBuilder().setSessionId("session_empty_eet").build();
    engine.respondTo(":request", 200, emptyEetResponse.toByteArray());

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);
    ExecutionException thrown = assertThrows(ExecutionException.class, future::get);

    assertThat(thrown).hasCauseThat().isInstanceOf(IOException.class);
    assertThat(thrown).hasCauseThat().hasMessageThat().contains("no result set");
    assertThat(engine.anyRequestContains(":performmultiple")).isFalse();
    assertThat(engine.anyRequestContains(":reportresult")).isFalse();
  }

  @Test
  public void fetch_planDownloadFailure_skipsTaskWithoutOpeningSession() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    TaskAssignment assignment1 = createTaskAssignment(TASK_1, "agg_1", "auth_1");
    TaskAssignment assignment2 =
        createTaskAssignment(TASK_2, "agg_2", "auth_2").toBuilder()
            .setPlan(Resource.newBuilder().setUri(SERVER_URL + "/task2_plan"))
            .build();

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(assignment1)
            .addTaskAssignments(assignment2)
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload",
        200,
        createUploadOperation("key_1", "tok_1", "res_1").toByteArray());
    engine.respondTo(
        "agg_2/clients/auth_2:startdataupload",
        200,
        createUploadOperation("key_2", "tok_2", "res_2").toByteArray());
    engine.respondTo("task2_plan", 500, new byte[0]);

    ImmutableList<WorkloadSet> resolved = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).hasSize(1);
    assertThat(resolved.get(0).name()).isEqualTo(TASK_1);

    // The plan is read before the session is opened, so task 2 never opens a session and nothing
    // is aborted.
    assertThat(engine.anyRequestContains("agg_2/clients/")).isFalse();
    assertThat(engine.anyRequestContains(":abort")).isFalse();
  }

  @Test
  public void fetch_endorsementsDownloadFailure_skipsTaskWithoutOpeningSession() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    TaskAssignment assignment1 = createTaskAssignment(TASK_1, "agg_1", "auth_1");
    TaskAssignment assignment2 =
        createTaskAssignment(TASK_2, "agg_2", "auth_2").toBuilder()
            .setConfidentialAggregationInfo(
                TaskAssignment.ConfidentialAggregationInfo.newBuilder()
                    .setSignedEndorsements(
                        Resource.newBuilder().setUri(SERVER_URL + "/task2_endorsements")))
            .build();

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(assignment1)
            .addTaskAssignments(assignment2)
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload",
        200,
        createUploadOperation("key_1", "tok_1", "res_1").toByteArray());
    engine.respondTo(
        "agg_2/clients/auth_2:startdataupload",
        200,
        createUploadOperation("key_2", "tok_2", "res_2").toByteArray());
    engine.respondTo("task2_endorsements", 500, new byte[0]);

    ImmutableList<WorkloadSet> resolved = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).hasSize(1);
    assertThat(resolved.get(0).name()).isEqualTo(TASK_1);

    // The endorsements are fetched with the plan, before the session is opened, so task 2 never
    // opens one and there is nothing to abort.
    assertThat(engine.anyRequestContains("agg_2/clients/auth_2:startdataupload")).isFalse();
    assertThat(engine.anyRequestContains(":abort")).isFalse();
  }

  @Test
  public void fetch_runtimeExceptionReadingTask2Plan_failsFetchWithoutAborts() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    TaskAssignment assignment1 = createTaskAssignment(TASK_1, "agg_1", "auth_1");
    TaskAssignment assignment2 =
        createTaskAssignment(TASK_2, "agg_2", "auth_2").toBuilder()
            .setPlan(Resource.newBuilder().setUri(SERVER_URL + "/task2_plan"))
            .build();

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(assignment1)
            .addTaskAssignments(assignment2)
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload",
        200,
        createUploadOperation("key_1", "tok_1", "res_1").toByteArray());
    engine.respondTo(
        "agg_2/clients/auth_2:startdataupload",
        200,
        createUploadOperation("key_2", "tok_2", "res_2").toByteArray());
    engine.failWith("task2_plan", new IllegalStateException("boom"));

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(IllegalStateException.class);

    // The already-resolved task 1 is dropped without an abort and its session expires on the
    // server; task 2 failed before opening a session.
    assertThat(engine.anyRequestContains("agg_2/clients/")).isFalse();
    assertThat(engine.anyRequestContains(":abort")).isFalse();
  }

  @Test
  public void fetch_relativeAggregationPrefixes_resolveAgainstPreviousPhaseHost() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    TaskAssignment assignment =
        createTaskAssignment(TASK_1, "agg_1", "auth_1").toBuilder()
            .setAggregationDataForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
            .build();

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder().addTaskAssignments(assignment).build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    StartConfidentialAggregationDataUploadResponse baseResp =
        extractUploadResponse(createUploadOperation("key_1", "tok_1", "res_1"));
    StartConfidentialAggregationDataUploadResponse uploadResp =
        baseResp.toBuilder()
            .setAggregationProtocolForwardingInfo(
                ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
            .setResource(
                baseResp.getResource().toBuilder()
                    .setDataUploadForwardingInfo(
                        ForwardingInfo.newBuilder().setTargetUriPrefix("/")))
            .build();
    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload", 200, wrapUploadResponse(uploadResp).toByteArray());

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).hasSize(1);
    WorkloadSet workloadSet = fetchResult.get(0);
    assertThat(engine.onlyRequestContaining(":startdataupload").url())
        .startsWith(SERVER_URL + "/v1/confidentialaggregations/");
    assertThat(workloadSet.uploadTokens().uploadUri()).isEqualTo(SERVER_URL);
    assertThat(workloadSet.uploadTokens().submitUriPrefix()).isEqualTo(SERVER_URL);
  }

  @Test
  public void fetch_missingAggregationDataPrefix_skipsTaskWithoutOpeningSession() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    TaskAssignment assignment1 = createTaskAssignment(TASK_1, "agg_1", "auth_1");
    TaskAssignment assignment2 =
        createTaskAssignment(TASK_2, "agg_2", "auth_2").toBuilder()
            .clearAggregationDataForwardingInfo()
            .build();

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(assignment1)
            .addTaskAssignments(assignment2)
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload",
        200,
        createUploadOperation("key_1", "tok_1", "res_1").toByteArray());

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).hasSize(1);
    assertThat(fetchResult.get(0).name()).isEqualTo(TASK_1);
    assertThat(engine.anyRequestContains("agg_2")).isFalse();
    assertThat(engine.anyRequestContains(":abort")).isFalse();
  }

  @Test
  public void fetch_missingDataUploadPrefix_abortsWithInternalAndSkipsTask() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    StartConfidentialAggregationDataUploadResponse baseResp =
        extractUploadResponse(createUploadOperation("key_1", "tok_1", "res_1"));
    StartConfidentialAggregationDataUploadResponse uploadResp =
        baseResp.toBuilder()
            .setResource(baseResp.getResource().toBuilder().clearDataUploadForwardingInfo())
            .build();
    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload", 200, wrapUploadResponse(uploadResp).toByteArray());

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).isEmpty();

    byte[] abortReqBytes = engine.onlyRequestContaining(":abort").body();
    AbortConfidentialAggregationRequest abortReq =
        AbortConfidentialAggregationRequest.parseFrom(
            abortReqBytes, ExtensionRegistryLite.getEmptyRegistry());
    assertThat(abortReq.getStatus().getCode()).isEqualTo(Code.INTERNAL_VALUE);
  }

  @Test
  public void fetch_abortFails_logsAbortRequestFailedDiagCode() throws Exception {
    engine.respondTo(":request", 200, buildEetResponseWithTwoTasks().toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);
    engine.respondTo(
        ":performmultiple",
        200,
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .build()
            .toByteArray());
    StartConfidentialAggregationDataUploadResponse baseResp =
        extractUploadResponse(createUploadOperation("key_1", "tok_1", "res_1"));
    StartConfidentialAggregationDataUploadResponse uploadResp =
        baseResp.toBuilder()
            .setResource(baseResp.getResource().toBuilder().clearDataUploadForwardingInfo())
            .build();
    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload", 200, wrapUploadResponse(uploadResp).toByteArray());
    engine.respondTo(":abort", 503, new byte[0]);

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).isEmpty();
    assertThat(monitor.diagCodes())
        .containsExactly(ProdDiagCode.HTTP_CANCELLATION_OR_ABORT_REQUEST_FAILED);
    assertThat(monitor.diagFetchIds())
        .containsExactly(monitor.event(Kind.CHECKIN_STARTED).fetchId());
  }

  @Test
  public void fetch_missingAggregationProtocolPrefix_skipsTaskWithoutAbort() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    StartConfidentialAggregationDataUploadResponse baseResp =
        extractUploadResponse(createUploadOperation("key_1", "tok_1", "res_1"));
    StartConfidentialAggregationDataUploadResponse uploadResp =
        baseResp.toBuilder().clearAggregationProtocolForwardingInfo().build();
    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload", 200, wrapUploadResponse(uploadResp).toByteArray());

    ImmutableList<WorkloadSet> fetchResult = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(fetchResult).isEmpty();
    assertThat(engine.anyRequestContains(":abort")).isFalse();
  }

  @Test
  public void fetch_runtimeExceptionDownloadingEncryptionConfig_abortsWithInternalAndFailsFetch()
      throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(createTaskAssignment(TASK_1, "agg_1", "auth_1"))
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    StartConfidentialAggregationDataUploadResponse uploadResp =
        createUploadResponse("key_1", "tok_1", "res_1").toBuilder()
            .setEncryptionConfig(Resource.newBuilder().setUri("https://enc-config"))
            .build();
    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload", 200, wrapUploadResponse(uploadResp).toByteArray());

    // Engine throws IllegalStateException for the encryption config URL
    engine.failWith(
        uploadResp.getEncryptionConfig().getUri(), new IllegalStateException("download failed"));

    ListenableFuture<ImmutableList<WorkloadSet>> future = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME);

    ExecutionException e = assertThrows(ExecutionException.class, future::get);
    assertThat(e).hasCauseThat().isInstanceOf(IllegalStateException.class);

    AbortConfidentialAggregationRequest abortReq = abortRequestFor("agg_1/clients/tok_1:abort");
    assertThat(abortReq.getStatus().getCode()).isEqualTo(Code.INTERNAL_VALUE);
  }

  @Test
  public void fetch_assignmentMissingAggregationId_skipsTaskWithoutRequests() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    TaskAssignment badAssignment =
        createTaskAssignment(TASK_1, "agg_1", "auth_1").toBuilder().clearAggregationId().build();
    TaskAssignment goodAssignment = createTaskAssignment(TASK_2, "agg_2", "auth_2");

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder()
            .addTaskAssignments(badAssignment)
            .addTaskAssignments(goodAssignment)
            .build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    engine.respondTo(
        "agg_2/clients/auth_2:startdataupload",
        200,
        createUploadOperation("key_2", "tok_2", "res_2").toByteArray());

    ImmutableList<WorkloadSet> resolved = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).hasSize(1);
    assertThat(resolved.get(0).name()).isEqualTo(TASK_2);
    assertThat(engine.anyRequestContains("agg_1")).isFalse();
    assertThat(engine.anyRequestContains(":abort")).isFalse();
  }

  @Test
  public void fetch_emptyPlanResource_yieldsEmptySelectorAndCompat() throws Exception {
    EligibilityEvalTaskResponse eetResponse = buildEetResponseWithTwoTasks();
    engine.respondTo(":request", 200, eetResponse.toByteArray());
    engine.respondTo(":reportresult", 200, new byte[0]);

    TaskAssignment assignment =
        createTaskAssignment(TASK_1, "agg_1", "auth_1").toBuilder()
            .setPlan(
                Resource.newBuilder()
                    .setInlineResource(
                        Resource.InlineResource.newBuilder().setData(ByteString.EMPTY)))
            .build();

    PerformMultipleTaskAssignmentsResponse assignResponse =
        PerformMultipleTaskAssignmentsResponse.newBuilder().addTaskAssignments(assignment).build();
    engine.respondTo(":performmultiple", 200, assignResponse.toByteArray());

    engine.respondTo(
        "agg_1/clients/auth_1:startdataupload",
        200,
        createUploadOperation("key_1", "tok_1", "res_1").toByteArray());

    ImmutableList<WorkloadSet> resolved = fetcher.fetch(PRIVATE_LOG_SOURCE_NAME).get();

    assertThat(resolved).hasSize(1);
    WorkloadSet ws = resolved.get(0);
    assertThat(ws.name()).isEqualTo(TASK_1);
    assertThat(ws.selectorSpec().rawCriteria()).isEmpty();
    assertThat(ws.legacyFcpCompat().collectionUri()).isEmpty();
  }

}
