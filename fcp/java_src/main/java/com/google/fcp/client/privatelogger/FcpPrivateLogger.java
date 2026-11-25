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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import android.content.Context;
import androidx.annotation.Nullable;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.confidentialcompute.ConfidentialCompute;
import com.google.fcp.client.privatelogger.fcpconnector.FcpPrivateLoggerConnector;
import com.google.fcp.client.privatelogger.fcpconnector.FcpPrivateLoggerConnectorDeps;
import com.google.fcp.client.privatelogger.impl.PrivateLoggerImpl;
import com.google.fcp.client.privatelogger.impl.PrivateLoggerImplDeps;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Entry point and builder for creating unbundled {@link PrivateLogger} instances backed by {@link
 * FcpPrivateLoggerConnector}.
 */
public final class FcpPrivateLogger {

  /**
   * The database executor every logger built without {@link Builder#setDbExecutor} shares: one
   * daemon thread, created on first use.
   */
  private static final Supplier<ExecutorService> DEFAULT_DB_EXECUTOR =
      Suppliers.memoize(
          () ->
              Executors.newSingleThreadExecutor(
                  new ThreadFactoryBuilder()
                      .setNameFormat("fcp-privatelogger-db-%d")
                      .setDaemon(true)
                      .build()));

  private FcpPrivateLogger() {}

  /** Creates a new {@link Builder} for a {@link PrivateLogger} instance. */
  public static Builder newBuilder(Context context, PrivateLoggerConfig config) {
    return new Builder(context, config);
  }

  /** Builder for constructing {@link PrivateLogger} instances. */
  public static final class Builder {
    private final Context context;
    private final PrivateLoggerConfig config;
    @Nullable private ExecutorService dbExecutor;
    @Nullable private TimeSource timeSource;
    @Nullable private StabilityMonitor stabilityMonitor;
    @Nullable private AccessPolicyEndorsementOptions endorsementOptions;
    @Nullable private ConfidentialCompute confidentialCompute;

    private Builder(Context context, PrivateLoggerConfig config) {
      this.context = checkNotNull(context, "context must not be null").getApplicationContext();
      this.config = checkNotNull(config, "config must not be null");
    }

    /**
     * Sets the database executor. Caller-supplied executors are owned (and shut down) by the
     * caller; the default is one daemon thread that every logger built without one shares, and that
     * lives for the process.
     */
    @CanIgnoreReturnValue
    public Builder setDbExecutor(ExecutorService dbExecutor) {
      this.dbExecutor = checkNotNull(dbExecutor, "dbExecutor must not be null");
      return this;
    }

    /**
     * Sets the wall clock for storage timestamps. Defaults to the time source of the {@link
     * ConfidentialCompute} passed to {@link #setConfidentialCompute}, which SWOR and diagnostic
     * event timestamps always use.
     */
    @CanIgnoreReturnValue
    public Builder setTimeSource(TimeSource timeSource) {
      this.timeSource = checkNotNull(timeSource, "timeSource must not be null");
      return this;
    }

    /**
     * Sets the handler for crashes of the logger. Defaults to the monitor of the {@link
     * ConfidentialCompute} passed to {@link #setConfidentialCompute}, which the connector always
     * reports to.
     */
    @CanIgnoreReturnValue
    public Builder setStabilityMonitor(StabilityMonitor stabilityMonitor) {
      this.stabilityMonitor = checkNotNull(stabilityMonitor, "stabilityMonitor must not be null");
      return this;
    }

    /** Sets the attestation endorsement options. Required. */
    @CanIgnoreReturnValue
    public Builder setEndorsementOptions(AccessPolicyEndorsementOptions endorsementOptions) {
      this.endorsementOptions =
          checkNotNull(endorsementOptions, "endorsementOptions must not be null");
      return this;
    }

    /**
     * Sets the {@link ConfidentialCompute} that fetches, encodes and uploads. Required. Its builder
     * holds the network settings (API key, server, HTTP engine and executor), and SWOR follows how
     * it was built (see {@link ConfidentialCompute.Builder#disableSwor}).
     */
    @CanIgnoreReturnValue
    public Builder setConfidentialCompute(ConfidentialCompute confidentialCompute) {
      this.confidentialCompute =
          checkNotNull(confidentialCompute, "confidentialCompute must not be null");
      return this;
    }

    /**
     * Builds only the configured {@link FcpPrivateLoggerConnector}, for hosts that wire {@link
     * PrivateLoggerImpl} themselves with their own database executor and stability monitor. Callers
     * that want a complete {@link PrivateLogger} should use {@link #build()}.
     */
    public FcpPrivateLoggerConnector buildConnector() {
      checkState(
          confidentialCompute != null,
          "confidentialCompute is required; call setConfidentialCompute()");
      checkState(
          endorsementOptions != null,
          "endorsementOptions are required; call setEndorsementOptions()");
      return new FcpPrivateLoggerConnector(
          FcpPrivateLoggerConnectorDeps.builder()
              .setConfidentialCompute(confidentialCompute)
              .setEndorsementOptions(endorsementOptions)
              .build());
    }

    /** Builds the configured {@link PrivateLogger} instance. */
    public PrivateLogger build() {
      FcpPrivateLoggerConnector connector = buildConnector();
      ConfidentialCompute resolvedConfidentialCompute = checkNotNull(confidentialCompute);
      PrivateLoggerImplDeps deps =
          PrivateLoggerImplDeps.newBuilder()
              .setContext(context)
              .setConfig(config)
              .setConnector(connector)
              .setDbExecutor(dbExecutor != null ? dbExecutor : DEFAULT_DB_EXECUTOR.get())
              .setTimeSource(
                  timeSource != null ? timeSource : resolvedConfidentialCompute.timeSource())
              .setStabilityMonitor(
                  stabilityMonitor != null
                      ? stabilityMonitor
                      : resolvedConfidentialCompute.stabilityMonitor())
              .build();
      return PrivateLoggerImpl.create(deps);
    }
  }
}
