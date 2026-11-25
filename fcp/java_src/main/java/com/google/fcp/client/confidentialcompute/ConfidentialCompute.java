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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.MoreExecutors.newSequentialExecutor;

import android.content.Context;
import androidx.annotation.Nullable;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.base.Ticker;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.confidentialcompute.http.HttpUrlConnectionEngine;
import com.google.fcp.client.confidentialcompute.internal.crypto.TinkConfidentialComputeCrypto;
import com.google.fcp.client.confidentialcompute.internal.encode.AttestationRecordLogger;
import com.google.fcp.client.confidentialcompute.internal.encode.ConfidentialPayloadEncoderImpl;
import com.google.fcp.client.confidentialcompute.internal.fetch.FcpWorkloadSetFetcher;
import com.google.fcp.client.confidentialcompute.internal.fetch.FcpWorkloadSetFetcherDeps;
import com.google.fcp.client.confidentialcompute.internal.http.ElapsedRealtimeTicker;
import com.google.fcp.client.confidentialcompute.internal.swor.SworTracker;
import com.google.fcp.client.confidentialcompute.internal.upload.FcpConfidentialPayloadUploader;
import com.google.fcp.client.confidentialcompute.internal.upload.FcpConfidentialPayloadUploaderDeps;
import com.google.fcp.client.flags.FcpClientFlags;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import java.time.Duration;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Entry point to the unbundled confidential compute client API, providing the three pipeline
 * stages: {@link WorkloadSetFetcher}, {@link ConfidentialPayloadEncoder}, and {@link
 * ConfidentialPayloadUploader}.
 *
 * <p>{@link Builder#build} creates the default implementation of every stage that is not replaced.
 * To replace a stage, pass your own implementation to {@link Builder#setWorkloadSetFetcher}, {@link
 * Builder#setPayloadEncoder} or {@link Builder#setPayloadUploader}. A replacement encoder verifies
 * {@link WorkloadSet#encryptionConfig()} and {@link WorkloadSet#signedEndorsements()} against the
 * endorsement options passed to {@code encode}; a replacement uploader uses {@link
 * WorkloadSet#uploadTokens()}.
 *
 * <p>The caller's data store must only pass fresh data to {@code encode}: records never contributed
 * to that workload set. Either return each record at most once per workload set, or mark records as
 * contributed when they are read and roll that back when the upload fails with {@link
 * UploadAbortedException} (or when nothing is uploaded), as {@code PrivateLogger} does. Any other
 * upload failure may have stored the payload and must count as contributed.
 */
public final class ConfidentialCompute {

  /**
   * The pool every instance built without {@link Builder#setExecutor} runs on. Daemon threads, so
   * that an idle pool never keeps a JVM from exiting (on Android the process is killed regardless).
   * A cached pool releases idle threads after a minute, so it needs no shutdown.
   */
  private static final Supplier<ExecutorService> DEFAULT_POOL =
      Suppliers.memoize(
          () ->
              Executors.newCachedThreadPool(
                  new ThreadFactoryBuilder()
                      .setNameFormat("fcp-confidentialcompute-%d")
                      .setDaemon(true)
                      .build()));

  private final WorkloadSetFetcher workloadSetFetcher;
  private final ConfidentialPayloadEncoder payloadEncoder;
  private final ConfidentialPayloadUploader payloadUploader;
  private final Ticker ticker;
  private final TimeSource timeSource;
  private final StabilityMonitor stabilityMonitor;

  private ConfidentialCompute(
      WorkloadSetFetcher workloadSetFetcher,
      ConfidentialPayloadEncoder payloadEncoder,
      ConfidentialPayloadUploader payloadUploader,
      Ticker ticker,
      TimeSource timeSource,
      StabilityMonitor stabilityMonitor) {
    this.workloadSetFetcher = workloadSetFetcher;
    this.payloadEncoder = payloadEncoder;
    this.payloadUploader = payloadUploader;
    this.ticker = ticker;
    this.timeSource = timeSource;
    this.stabilityMonitor = stabilityMonitor;
  }

  /** Stage 1: obtains workload sets. */
  public WorkloadSetFetcher workloadSetFetcher() {
    return workloadSetFetcher;
  }

  /** Stage 2: verifies a workload set's encryption config and encodes and encrypts inputs. */
  public ConfidentialPayloadEncoder payloadEncoder() {
    return payloadEncoder;
  }

  /** Stage 3: uploads an encoded payload. */
  public ConfidentialPayloadUploader payloadUploader() {
    return payloadUploader;
  }

  /**
   * The clock the default stages time their diagnostic events with. A caller that logs its own
   * events, such as the private logger connector, uses it too.
   */
  public Ticker ticker() {
    return ticker;
  }

  /** The wall-clock time source used to timestamp diagnostic events. */
  public TimeSource timeSource() {
    return timeSource;
  }

  /**
   * The monitor the default stages log their diagnostic events to. A caller that logs its own
   * events should use it too.
   */
  public StabilityMonitor stabilityMonitor() {
    return stabilityMonitor;
  }

  /** Creates a new {@link Builder} to construct a {@link ConfidentialCompute} instance. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for {@link ConfidentialCompute}. */
  public static final class Builder {

    private String apiKey;
    private HttpUrlConnectionEngine httpEngine;
    private ConfidentialComputeCrypto crypto;
    private Executor executor;
    @Nullable private WorkloadSetFetcher workloadSetFetcher;
    @Nullable private ConfidentialPayloadEncoder payloadEncoder;
    @Nullable private ConfidentialPayloadUploader payloadUploader;
    @Nullable private Context context;
    private boolean sworDisabled;
    private StabilityMonitor stabilityMonitor = StabilityMonitor.DEFAULT_STABILITY_MONITOR;
    private Ticker ticker = ElapsedRealtimeTicker.INSTANCE;
    private TimeSource timeSource = TimeSource.REAL_TIME_SOURCE;
    private DeviceAttestationProvider deviceAttestationProvider = DeviceAttestationProvider.NONE;
    private FcpClientFlags flags = FcpClientFlags.DEFAULT;

    private Builder() {}

    /**
     * Sets the API key for backend authentication. Required unless the fetcher and uploader are
     * both replaced.
     */
    @CanIgnoreReturnValue
    public Builder setApiKey(String apiKey) {
      checkNotNull(apiKey, "apiKey must not be null");
      checkArgument(!apiKey.isEmpty(), "apiKey must not be empty");
      this.apiKey = apiKey;
      return this;
    }

    /** Sets the HTTP engine used for all network calls. Defaults to the platform engine. */
    @CanIgnoreReturnValue
    public Builder setHttpEngine(HttpUrlConnectionEngine httpEngine) {
      this.httpEngine = checkNotNull(httpEngine, "httpEngine must not be null");
      return this;
    }

    /**
     * Sets the crypto implementation. Defaults to the pure-Java implementation built on Tink, which
     * matches C++ {@code AttestationTransparencyVerifier} ({@code
     * fcp/client/attestation/attestation_transparency_verifier.cc}) and {@code MessageEncryptor}
     * ({@code fcp/confidentialcompute/crypto.cc}). See the README for its provider requirement on
     * Android API levels below 30.
     */
    @CanIgnoreReturnValue
    public Builder setCrypto(ConfidentialComputeCrypto crypto) {
      this.crypto = checkNotNull(crypto, "crypto must not be null");
      return this;
    }

    /**
     * Sets the executor on which the default stages run their network calls and crypto. The built
     * instance wraps it in a sequential executor, so the default stages run one call at a time
     * whatever executor is given. Defaults to a cached pool of daemon threads that all instances
     * share, created on first use; the default never runs work on the caller's thread. Replacement
     * stages are not run on it. A caller-supplied executor stays owned by the caller.
     */
    @CanIgnoreReturnValue
    public Builder setExecutor(Executor executor) {
      this.executor = checkNotNull(executor, "executor must not be null");
      return this;
    }

    /**
     * Sets the context whose files dir holds the sampling without replacement (SWOR) upload times;
     * only its application context is kept. Required unless the fetcher and uploader are both
     * replaced or {@link #disableSwor} is called.
     */
    @CanIgnoreReturnValue
    public Builder setContext(Context context) {
      this.context = checkNotNull(context, "context must not be null").getApplicationContext();
      return this;
    }

    /**
     * Turns off sampling without replacement (SWOR) in the default fetcher and uploader. SWOR is on
     * by default: the fetcher does not request workload sets uploaded to within their SWOR period,
     * and the uploader records each upload time. Replaced stages are not affected.
     */
    @CanIgnoreReturnValue
    public Builder disableSwor() {
      this.sworDisabled = true;
      return this;
    }

    /**
     * Sets the monitor that receives the diagnostic events of the default stages (see {@link
     * DiagnosticEvent}). Defaults to {@link StabilityMonitor#DEFAULT_STABILITY_MONITOR}, which logs
     * them to logcat at verbose level. Replaced stages log nothing unless they do so themselves.
     */
    @CanIgnoreReturnValue
    public Builder setStabilityMonitor(StabilityMonitor stabilityMonitor) {
      this.stabilityMonitor = checkNotNull(stabilityMonitor, "stabilityMonitor must not be null");
      return this;
    }

    /**
     * Sets the clock that diagnostic event durations are measured with. Defaults to {@code
     * SystemClock.elapsedRealtimeNanos()}, which keeps counting while the device sleeps. For tests.
     */
    @CanIgnoreReturnValue
    public Builder setTicker(Ticker ticker) {
      this.ticker = checkNotNull(ticker, "ticker must not be null");
      return this;
    }

    /** Sets the wall-clock time source used to timestamp diagnostic events. For tests. */
    @CanIgnoreReturnValue
    public Builder setTimeSource(TimeSource timeSource) {
      this.timeSource = checkNotNull(timeSource, "timeSource must not be null");
      return this;
    }

    /**
     * Sets the provider of the device attestation measurement sent with the task assignment
     * request. Used by the default fetcher only. Defaults to {@link
     * DeviceAttestationProvider#NONE}.
     */
    @CanIgnoreReturnValue
    public Builder setDeviceAttestationProvider(
        DeviceAttestationProvider deviceAttestationProvider) {
      this.deviceAttestationProvider =
          checkNotNull(deviceAttestationProvider, "deviceAttestationProvider must not be null");
      return this;
    }

    /**
     * Sets the flags read by the default fetcher and uploader: the server URL and the HTTP
     * timeouts. Defaults to {@link FcpClientFlags#DEFAULT}. Flags are read once in {@link
     * #build()}; a later change to the host's flag values is not seen by an instance that is
     * already built.
     */
    @CanIgnoreReturnValue
    public Builder setFlags(FcpClientFlags flags) {
      this.flags = checkNotNull(flags, "flags must not be null");
      return this;
    }

    /**
     * Replaces the default fetcher. The replacement must follow the {@link WorkloadSetFetcher}
     * contract: return quickly, do blocking work on a thread of its own choosing, and report
     * failures through the returned future.
     */
    @CanIgnoreReturnValue
    public Builder setWorkloadSetFetcher(WorkloadSetFetcher workloadSetFetcher) {
      this.workloadSetFetcher =
          checkNotNull(workloadSetFetcher, "workloadSetFetcher must not be null");
      return this;
    }

    /**
     * Replaces the default encoder. The replacement must follow the {@link
     * ConfidentialPayloadEncoder} contract: verify {@link WorkloadSet#encryptionConfig()} and
     * {@link WorkloadSet#signedEndorsements()} against the endorsement options passed to {@code
     * encode} and encrypt to the verified key.
     */
    @CanIgnoreReturnValue
    public Builder setPayloadEncoder(ConfidentialPayloadEncoder payloadEncoder) {
      this.payloadEncoder = checkNotNull(payloadEncoder, "payloadEncoder must not be null");
      return this;
    }

    /**
     * Replaces the default uploader. The replacement must follow the {@link
     * ConfidentialPayloadUploader} contract and upload to {@link WorkloadSet#uploadTokens()}.
     */
    @CanIgnoreReturnValue
    public Builder setPayloadUploader(ConfidentialPayloadUploader payloadUploader) {
      this.payloadUploader = checkNotNull(payloadUploader, "payloadUploader must not be null");
      return this;
    }

    /**
     * Builds the {@link ConfidentialCompute} instance, creating a default for each stage that was
     * not replaced. The flags are read here.
     *
     * @throws IllegalStateException if a default stage is built without the settings it needs: the
     *     fetcher and uploader need {@code apiKey}, and both need {@code context} unless SWOR is
     *     disabled.
     */
    public ConfidentialCompute build() {
      // Shared by the default stages and created only if one of them is built.
      Supplier<Executor> stageExecutor = Suppliers.memoize(this::createStageExecutor);
      Supplier<ConfidentialComputeCrypto> stageCrypto =
          Suppliers.memoize(() -> crypto != null ? crypto : new TinkConfidentialComputeCrypto());
      Supplier<SworTracker> sworTracker = Suppliers.memoize(this::createSworTracker);
      Duration httpConnectTimeout = null;
      Duration httpReadTimeout = null;
      if (workloadSetFetcher == null || payloadUploader == null) {
        checkState(
            apiKey != null, "apiKey is required unless the fetcher and uploader are replaced");
        httpConnectTimeout = flags.httpConnectTimeout();
        httpReadTimeout = flags.httpReadTimeout();
      }
      return new ConfidentialCompute(
          workloadSetFetcher != null
              ? workloadSetFetcher
              : createFetcher(stageExecutor, sworTracker, httpConnectTimeout, httpReadTimeout),
          payloadEncoder != null
              ? payloadEncoder
              : new ConfidentialPayloadEncoderImpl(
                  stageCrypto.get(),
                  AttestationRecordLogger.LOGCAT,
                  stabilityMonitor,
                  ticker,
                  timeSource,
                  stageExecutor.get()),
          payloadUploader != null
              ? payloadUploader
              : createUploader(stageExecutor, sworTracker, httpConnectTimeout, httpReadTimeout),
          ticker,
          timeSource,
          stabilityMonitor);
    }

    private WorkloadSetFetcher createFetcher(
        Supplier<Executor> stageExecutor,
        Supplier<SworTracker> sworTracker,
        Duration httpConnectTimeout,
        Duration httpReadTimeout) {
      return new FcpWorkloadSetFetcher(
          FcpWorkloadSetFetcherDeps.builder()
              .setServerUrl(flags.serverUrl())
              .setApiKey(apiKey)
              .setHttpEngine(httpEngineOrDefault())
              .setHttpConnectTimeout(httpConnectTimeout)
              .setHttpReadTimeout(httpReadTimeout)
              .setSworTracker(sworTracker.get())
              .setStabilityMonitor(stabilityMonitor)
              .setTicker(ticker)
              .setTimeSource(timeSource)
              .setDeviceAttestationProvider(deviceAttestationProvider)
              .setExecutor(stageExecutor.get())
              .build());
    }

    private ConfidentialPayloadUploader createUploader(
        Supplier<Executor> stageExecutor,
        Supplier<SworTracker> sworTracker,
        Duration httpConnectTimeout,
        Duration httpReadTimeout) {
      return new FcpConfidentialPayloadUploader(
          FcpConfidentialPayloadUploaderDeps.builder()
              .setApiKey(apiKey)
              .setHttpEngine(httpEngineOrDefault())
              .setHttpConnectTimeout(httpConnectTimeout)
              .setHttpReadTimeout(httpReadTimeout)
              .setSworTracker(sworTracker.get())
              .setStabilityMonitor(stabilityMonitor)
              .setTicker(ticker)
              .setTimeSource(timeSource)
              .setExecutor(stageExecutor.get())
              .build());
    }

    /**
     * Returns the executor the default stages share. It runs one network or crypto call at a time,
     * whatever the underlying executor is, so the stages behave the same on a pool as on a single
     * thread.
     */
    private Executor createStageExecutor() {
      return newSequentialExecutor(executor != null ? executor : DEFAULT_POOL.get());
    }

    private SworTracker createSworTracker() {
      if (sworDisabled) {
        return SworTracker.DISABLED;
      }
      checkState(
          context != null,
          "context is required unless the fetcher and uploader are replaced or disableSwor() is"
              + " called");
      return SworTracker.forContext(context, timeSource);
    }

    private HttpUrlConnectionEngine httpEngineOrDefault() {
      return httpEngine != null ? httpEngine : HttpUrlConnectionEngine.DEFAULT;
    }
  }
}
