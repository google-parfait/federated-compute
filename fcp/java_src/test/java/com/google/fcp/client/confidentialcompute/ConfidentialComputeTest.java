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
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static org.junit.Assert.assertThrows;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.base.Ticker;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.fcp.client.FakeTimeSource;
import com.google.fcp.client.confidentialcompute.internal.http.FakeHttpUrlConnectionEngine;
import com.google.fcp.client.confidentialcompute.internal.http.FakeHttpUrlConnectionEngine.FakeHttpUrlConnection;
import com.google.fcp.client.flags.FcpClientFlags;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent.Kind;
import com.google.fcp.client.stabilitymonitor.RecordingStabilityMonitor;
import com.google.internal.federated.plan.PopulationEligibilitySpec;
import com.google.internal.federated.plan.PopulationEligibilitySpec.TaskInfo;
import com.google.internal.federated.plan.PopulationEligibilitySpec.TaskInfo.TaskAssignmentMode;
import com.google.internal.federatedcompute.v1.EligibilityEvalTask;
import com.google.internal.federatedcompute.v1.EligibilityEvalTaskResponse;
import com.google.internal.federatedcompute.v1.ForwardingInfo;
import com.google.internal.federatedcompute.v1.Resource;
import com.google.protobuf.ByteString;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link ConfidentialCompute}. */
@RunWith(AndroidJUnit4.class)
public final class ConfidentialComputeTest {

  private static final FcpClientFlags FLAGS =
      new FcpClientFlags() {
        @Override
        public String serverUrl() {
          return "https://flags.example.com";
        }

        @Override
        public Duration httpConnectTimeout() {
          return Duration.ofMillis(1234);
        }

        @Override
        public Duration httpReadTimeout() {
          return Duration.ofMillis(5678);
        }
      };

  @Test
  public void setApiKey_emptyString_throwsIllegalArgumentException() {
    ConfidentialCompute.Builder builder = ConfidentialCompute.builder();
    assertThrows(IllegalArgumentException.class, () -> builder.setApiKey(""));
  }

  @Test
  public void build_withEmptyFlagServerUrl_throwsIllegalArgumentException() {
    ConfidentialCompute.Builder builder =
        ConfidentialCompute.builder()
            .setFlags(
                new FcpClientFlags() {
                  @Override
                  public String serverUrl() {
                    return "";
                  }
                })
            .setApiKey("test-key")
            .disableSwor();
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void build_withFlags_defaultStagesUseFlagServerUrlAndTimeouts() {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.setDefaultResponse(500, new byte[0]);
    ConfidentialCompute cc =
        ConfidentialCompute.builder()
            .setFlags(FLAGS)
            .setApiKey("test-key")
            .disableSwor()
            .setHttpEngine(engine)
            .setExecutor(directExecutor())
            .build();

    ListenableFuture<ImmutableList<WorkloadSet>> fetch =
        cc.workloadSetFetcher().fetch("log_source");
    assertThrows(ExecutionException.class, fetch::get);
    ListenableFuture<Void> upload =
        cc.payloadUploader().upload(createWorkloadSet(), EncodedPayload.create(ByteString.EMPTY));
    assertThrows(ExecutionException.class, upload::get);

    FakeHttpUrlConnection checkin = engine.onlyRequestContaining("flags.example.com").connection();
    assertThat(checkin.getConnectTimeout()).isEqualTo(1234);
    assertThat(checkin.getReadTimeout()).isEqualTo(5678);
    FakeHttpUrlConnection blobUpload =
        engine.onlyRequestContaining("upload.example.com").connection();
    assertThat(blobUpload.getConnectTimeout()).isEqualTo(1234);
    assertThat(blobUpload.getReadTimeout()).isEqualTo(5678);
  }

  @Test
  public void build_withoutApiKey_throwsIllegalStateException() {
    ConfidentialCompute.Builder builder =
        ConfidentialCompute.builder()
            .setContext(ApplicationProvider.getApplicationContext());
    IllegalStateException e = assertThrows(IllegalStateException.class, builder::build);
    assertThat(e).hasMessageThat().contains("apiKey");
  }

  @Test
  public void build_allStagesReplaced_needsNoServerUrlOrApiKey() {
    WorkloadSetFetcher fetcher = logSourceName -> immediateFuture(ImmutableList.of());
    ConfidentialPayloadEncoder encoder =
        (workloadSet, options, inputs) ->
            immediateFailedFuture(new UnsupportedOperationException());
    ConfidentialPayloadUploader uploader =
        (workloadSet, payload) -> immediateFailedFuture(new UnsupportedOperationException());

    ConfidentialCompute cc =
        ConfidentialCompute.builder()
            .setWorkloadSetFetcher(fetcher)
            .setPayloadEncoder(encoder)
            .setPayloadUploader(uploader)
            .build();

    assertThat(cc.workloadSetFetcher()).isSameInstanceAs(fetcher);
    assertThat(cc.payloadEncoder()).isSameInstanceAs(encoder);
    assertThat(cc.payloadUploader()).isSameInstanceAs(uploader);
  }

  @Test
  public void build_encoderReplaced_keepsDefaultFetcherAndUploader() {
    ConfidentialPayloadEncoder encoder =
        (workloadSet, options, inputs) ->
            immediateFailedFuture(new UnsupportedOperationException());

    ConfidentialCompute cc =
        ConfidentialCompute.builder()
            .disableSwor()
            .setApiKey("test-key")
            .setPayloadEncoder(encoder)
            .build();

    assertThat(cc.payloadEncoder()).isSameInstanceAs(encoder);
    assertThat(cc.workloadSetFetcher()).isNotNull();
    assertThat(cc.payloadUploader()).isNotNull();
  }

  @Test
  public void build_withContext_buildsDefaultStages() {
    ConfidentialCompute cc =
        ConfidentialCompute.builder()
            .setApiKey("test-key")
            .setContext(ApplicationProvider.getApplicationContext())
            .build();

    assertThat(cc.workloadSetFetcher()).isNotNull();
    assertThat(cc.payloadUploader()).isNotNull();
  }

  @Test
  public void build_withoutContextWithSwor_throwsIllegalStateException() {
    ConfidentialCompute.Builder builder = ConfidentialCompute.builder().setApiKey("test-key");
    IllegalStateException e = assertThrows(IllegalStateException.class, builder::build);
    assertThat(e).hasMessageThat().contains("context");
  }

  @Test
  public void build_withStabilityMonitor_defaultFetcherAndUploaderLogToIt() {
    RecordingStabilityMonitor monitor = new RecordingStabilityMonitor();
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.setDefaultResponse(500, new byte[0]);
    ConfidentialCompute cc =
        ConfidentialCompute.builder()
            .setApiKey("test-key")
            .disableSwor()
            .setHttpEngine(engine)
            .setExecutor(directExecutor())
            .setStabilityMonitor(monitor)
            .build();

    ListenableFuture<ImmutableList<WorkloadSet>> fetch =
        cc.workloadSetFetcher().fetch("log_source");
    assertThrows(ExecutionException.class, fetch::get);
    ListenableFuture<Void> upload =
        cc.payloadUploader().upload(createWorkloadSet(), EncodedPayload.create(ByteString.EMPTY));
    assertThrows(ExecutionException.class, upload::get);

    assertThat(monitor.kinds())
        .containsExactly(
            Kind.CHECKIN_STARTED, Kind.CHECKIN_ERROR, Kind.UPLOAD_STARTED, Kind.UPLOAD_ERROR)
        .inOrder();
  }

  @Test
  public void build_withClocksAndMonitor_returnsThemAndDefaultStagesUseThem() {
    AtomicLong nanos = new AtomicLong();
    Ticker ticker =
        new Ticker() {
          @Override
          public long read() {
            return nanos.getAndAdd(Duration.ofMillis(7).toNanos());
          }
        };
    Instant now = Instant.parse("2026-11-12T12:00:00Z");
    FakeTimeSource timeSource = new FakeTimeSource(now);
    RecordingStabilityMonitor monitor = new RecordingStabilityMonitor();
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.setDefaultResponse(500, new byte[0]);
    ConfidentialCompute cc =
        ConfidentialCompute.builder()
            .setApiKey("test-key")
            .disableSwor()
            .setHttpEngine(engine)
            .setExecutor(directExecutor())
            .setTicker(ticker)
            .setTimeSource(timeSource)
            .setStabilityMonitor(monitor)
            .build();

    assertThrows(ExecutionException.class, () -> cc.workloadSetFetcher().fetch("log_source").get());
    assertThrows(
        ExecutionException.class,
        () ->
            cc.payloadUploader()
                .upload(createWorkloadSet(), EncodedPayload.create(ByteString.EMPTY))
                .get());

    assertThat(cc.ticker()).isSameInstanceAs(ticker);
    assertThat(cc.timeSource()).isSameInstanceAs(timeSource);
    assertThat(cc.stabilityMonitor()).isSameInstanceAs(monitor);
    for (DiagnosticEvent event : monitor.events()) {
      assertThat(event.timestamp()).isEqualTo(now);
    }
    assertThat(monitor.event(Kind.CHECKIN_ERROR).duration()).isEqualTo(Duration.ofMillis(7));
    assertThat(monitor.event(Kind.UPLOAD_ERROR).duration()).isEqualTo(Duration.ofMillis(7));
  }

  @Test
  public void build_withDeviceAttestationProvider_defaultFetcherUsesIt() {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.setDefaultResponse(500, new byte[0]);
    engine.respondTo(":request", 200, eetResponseWithOneTask().toByteArray());
    List<String> contentBindings = new ArrayList<>();
    ConfidentialCompute cc =
        ConfidentialCompute.builder()
            .setApiKey("test-key")
            .disableSwor()
            .setHttpEngine(engine)
            .setExecutor(directExecutor())
            .setDeviceAttestationProvider((logSourceName, contentBinding, fetchId) -> {
              contentBindings.add(contentBinding);
              return immediateFuture("measurement");
            })
            .build();

    ListenableFuture<ImmutableList<WorkloadSet>> fetch =
        cc.workloadSetFetcher().fetch("log_source");
    assertThrows(ExecutionException.class, fetch::get);

    assertThat(contentBindings).containsExactly("session_id");
  }

  /** A check-in response whose eligibility spec has one multiple-assignment task. */
  private static EligibilityEvalTaskResponse eetResponseWithOneTask() {
    PopulationEligibilitySpec spec =
        PopulationEligibilitySpec.newBuilder()
            .addTaskInfo(
                TaskInfo.newBuilder()
                    .setTaskName("task")
                    .setTaskAssignmentMode(TaskAssignmentMode.TASK_ASSIGNMENT_MODE_MULTIPLE))
            .build();
    return EligibilityEvalTaskResponse.newBuilder()
        .setSessionId("session_id")
        .setTaskAssignmentForwardingInfo(ForwardingInfo.newBuilder().setTargetUriPrefix("/"))
        .setEligibilityEvalTask(
            EligibilityEvalTask.newBuilder()
                .setPopulationEligibilitySpec(
                    Resource.newBuilder()
                        .setInlineResource(
                            Resource.InlineResource.newBuilder().setData(spec.toByteString()))))
        .build();
  }

  private static WorkloadSet createWorkloadSet() {
    return WorkloadSet.builder()
        .setName("workload_set")
        .setSignedEndorsements(ByteString.EMPTY)
        .setEncryptionConfig(ByteString.copyFromUtf8("encryption-config"))
        .setUploadTokens(
            UploadTokens.builder()
                .setAggregationId("agg")
                .setClientToken("token")
                .setUploadUri("https://upload.example.com")
                .setResourceName("resource")
                .setSubmitUriPrefix("https://submit.example.com")
                .build())
        .setLogSourceName("log_source")
        .setFetchId(1)
        .build();
  }

  @Test
  public void build_fetcherReplacedWithDefaultUploader_stillRequiresApiKey() {
    ConfidentialCompute.Builder builder =
        ConfidentialCompute.builder()
            .setWorkloadSetFetcher(logSourceName -> immediateFuture(ImmutableList.of()))
            .setContext(ApplicationProvider.getApplicationContext());
    IllegalStateException e = assertThrows(IllegalStateException.class, builder::build);
    assertThat(e).hasMessageThat().contains("apiKey");
  }
}
