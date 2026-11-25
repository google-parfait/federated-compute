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

import com.google.auto.value.AutoValue;
import com.google.common.base.Ticker;
import com.google.common.collect.ImmutableMap;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.confidentialcompute.http.HttpUrlConnectionEngine;
import com.google.fcp.client.confidentialcompute.internal.http.ElapsedRealtimeTicker;
import com.google.fcp.client.confidentialcompute.internal.swor.SworTracker;
import com.google.fcp.client.flags.FcpClientFlags;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import java.net.HttpURLConnection;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.Executor;

/** Dependencies required to construct an {@link FcpConfidentialPayloadUploader}. */
@AutoValue
public abstract class FcpConfidentialPayloadUploaderDeps {

  /** Opens {@link HttpURLConnection}s; defaults to {@link HttpUrlConnectionEngine#DEFAULT}. */
  public abstract HttpUrlConnectionEngine httpEngine();

  /** HTTP connect timeout; defaults to {@link FcpClientFlags#httpConnectTimeout()}. */
  public abstract Duration httpConnectTimeout();

  /** HTTP read timeout; defaults to {@link FcpClientFlags#httpReadTimeout()}. */
  public abstract Duration httpReadTimeout();

  /** Executor on which the blocking upload (blob upload, submit, and any abort) runs. */
  public abstract Executor executor();

  /** API key sent as the {@code x-goog-api-key} header on every request. */
  public abstract String apiKey();

  /**
   * Extra headers added to every request. Server-provided forwarding headers from the upload tokens
   * override these.
   */
  public abstract ImmutableMap<String, String> requestHeaders();

  /**
   * Records the SWOR upload time before each upload and restores it if the server aborts the
   * upload. Required; pass {@link SworTracker#DISABLED} to turn SWOR off.
   */
  public abstract SworTracker sworTracker();


  /**
   * Receives the upload diagnostic events. Defaults to {@link
   * StabilityMonitor#DEFAULT_STABILITY_MONITOR}, which ignores them.
   */
  public abstract StabilityMonitor stabilityMonitor();

  /** Clock that diagnostic event durations are measured with. */
  public abstract Ticker ticker();

  /** Wall-clock time source used to timestamp diagnostic events. */
  public abstract TimeSource timeSource();

  /**
   * Returns a builder with defaults ({@link HttpUrlConnectionEngine#DEFAULT}, the {@link
   * FcpClientFlags#DEFAULT} timeouts, empty {@code requestHeaders}, and {@link
   * StabilityMonitor#DEFAULT_STABILITY_MONITOR}).
   */
  public static Builder builder() {
    return new AutoValue_FcpConfidentialPayloadUploaderDeps.Builder()
        .setHttpEngine(HttpUrlConnectionEngine.DEFAULT)
        .setHttpConnectTimeout(FcpClientFlags.DEFAULT_HTTP_CONNECT_TIMEOUT)
        .setHttpReadTimeout(FcpClientFlags.DEFAULT_HTTP_READ_TIMEOUT)
        .setRequestHeaders(ImmutableMap.of())
        .setStabilityMonitor(StabilityMonitor.DEFAULT_STABILITY_MONITOR)
        .setTicker(ElapsedRealtimeTicker.INSTANCE)
        .setTimeSource(TimeSource.REAL_TIME_SOURCE);
  }

  /** Builder for {@link FcpConfidentialPayloadUploaderDeps}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setHttpEngine(HttpUrlConnectionEngine httpEngine);

    public abstract Builder setHttpConnectTimeout(Duration httpConnectTimeout);

    public abstract Builder setHttpReadTimeout(Duration httpReadTimeout);

    public abstract Builder setExecutor(Executor executor);

    public abstract Builder setApiKey(String apiKey);

    public abstract Builder setRequestHeaders(Map<String, String> requestHeaders);

    public abstract Builder setSworTracker(SworTracker sworTracker);

    public abstract Builder setStabilityMonitor(StabilityMonitor stabilityMonitor);

    public abstract Builder setTicker(Ticker ticker);

    public abstract Builder setTimeSource(TimeSource timeSource);

    public abstract FcpConfidentialPayloadUploaderDeps build();
  }
}
