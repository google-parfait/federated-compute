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
package com.google.fcp.client.privatelogger;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.newDirectExecutorService;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static org.junit.Assert.assertThrows;

import android.content.Context;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.fcp.client.confidentialcompute.ConfidentialCompute;
import com.google.fcp.client.confidentialcompute.WorkloadSetFetcher;
import com.google.fcp.client.confidentialcompute.internal.http.FakeHttpUrlConnectionEngine;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;
import com.google.protobuf.StringValue;
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link FcpPrivateLogger}. */
@RunWith(AndroidJUnit4.class)
public final class FcpPrivateLoggerTest {

  private static final String TEST_LOG_SOURCE = "test_fcp_log_source";
  private static final String TEST_API_KEY = "test_api_key";

  private Context context;
  private PrivateLoggerConfig defaultConfig;

  @Before
  public void setUp() {
    context = ApplicationProvider.getApplicationContext();
    defaultConfig =
        PrivateLoggerConfig.builder()
            .setPrivateLogSourceName(TEST_LOG_SOURCE)
            .setRetentionPeriod(Duration.ofDays(7))
            .build();
  }

  @Test
  public void build_withoutConfidentialCompute_throwsIllegalStateException() {
    FcpPrivateLogger.Builder builder =
        FcpPrivateLogger.newBuilder(context, defaultConfig)
            .setEndorsementOptions(AccessPolicyEndorsementOptions.getDefaultInstance());
    assertThrows(IllegalStateException.class, builder::build);
  }

  @Test
  public void build_withoutEndorsementOptions_throwsIllegalStateException() {
    FcpPrivateLogger.Builder builder =
        FcpPrivateLogger.newBuilder(context, defaultConfig)
            .setConfidentialCompute(
                ConfidentialCompute.builder()
                    .disableSwor()
                    .setApiKey(TEST_API_KEY)
                    .build());
    assertThrows(IllegalStateException.class, builder::build);
  }

  @Test
  public void loggingPipeline_upload_invokesFetcherThroughWiredConnector() throws Exception {
    ExecutorService directExec = newDirectExecutorService();
    AtomicBoolean fetcherCalled = new AtomicBoolean(false);
    WorkloadSetFetcher testFetcher =
        logSourceName -> {
          fetcherCalled.set(true);
          return immediateFuture(ImmutableList.of());
        };

    ConfidentialCompute confidentialCompute =
        ConfidentialCompute.builder()
            .disableSwor()
            .setApiKey(TEST_API_KEY)
            .setExecutor(directExec)
            .setWorkloadSetFetcher(testFetcher)
            .build();
    PrivateLogger logger =
        FcpPrivateLogger.newBuilder(context, defaultConfig)
            .setConfidentialCompute(confidentialCompute)
            .setEndorsementOptions(AccessPolicyEndorsementOptions.getDefaultInstance())
            .setDbExecutor(directExec)
            .build();

    logger.log(StringValue.of("sample entry"));
    logger.upload().get();

    assertThat(fetcherCalled.get()).isTrue();
  }

  @Test
  public void loggingPipeline_uploadWithRealFetcher_runsOnHttpExecutorThread() throws Exception {
    FakeHttpUrlConnectionEngine fakeEngine = new FakeHttpUrlConnectionEngine();
    fakeEngine.setDefaultResponse(500, new byte[0]);
    ExecutorService httpExecutor =
        Executors.newSingleThreadExecutor(
            new ThreadFactoryBuilder().setNameFormat("test-http-executor-%d").build());
    try {
      ConfidentialCompute confidentialCompute =
          ConfidentialCompute.builder()
              .setApiKey(TEST_API_KEY)
              .setContext(context)
              .setHttpEngine(fakeEngine)
              .setExecutor(httpExecutor)
              .build();
      PrivateLogger logger =
          FcpPrivateLogger.newBuilder(context, defaultConfig)
              .setConfidentialCompute(confidentialCompute)
              .setEndorsementOptions(AccessPolicyEndorsementOptions.getDefaultInstance())
              .build();

      logger.log(StringValue.of("sample entry"));
      logger.upload().get();

      assertThat(fakeEngine.lastRequest().threadName()).startsWith("test-http-executor-");
      assertThat(new String(fakeEngine.lastRequest().body(), ISO_8859_1))
          .contains("android_unbundled");
    } finally {
      httpExecutor.shutdown();
    }
  }
}
