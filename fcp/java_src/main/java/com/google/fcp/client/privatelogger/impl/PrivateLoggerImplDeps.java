// Copyright 2025 Google LLC
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
package com.google.fcp.client.privatelogger.impl;

import android.content.Context;
import com.google.auto.value.AutoValue;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.privatelogger.PrivateLoggerConfig;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import java.util.concurrent.Executor;

/** Dependencies for {@link PrivateLoggerImpl}. */
@AutoValue
public abstract class PrivateLoggerImplDeps {
  /** The Android {@link Context}. */
  public abstract Context context();

  /** The {@link PrivateLoggerConfig} for the logger. */
  public abstract PrivateLoggerConfig config();

  /** The {@link PrivateLoggerConnector} for uploading data. */
  public abstract PrivateLoggerConnector connector();

  /** The {@link TimeSource} for timestamping log entries. */
  public abstract TimeSource timeSource();

  /** The {@link Executor} for database operations. */
  public abstract Executor dbExecutor();

  /** The {@link StabilityMonitor} for reporting crashes. */
  public abstract StabilityMonitor stabilityMonitor();

  /** Creates a new {@link Builder} instance. */
  public static Builder newBuilder() {
    return new AutoValue_PrivateLoggerImplDeps.Builder()
        .setStabilityMonitor(StabilityMonitor.DEFAULT_STABILITY_MONITOR);
  }

  /** Builder for {@link PrivateLoggerImplDeps}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets the Android {@link Context}. */
    public abstract Builder setContext(Context context);

    /** Sets the {@link PrivateLoggerConfig} for the logger. */
    public abstract Builder setConfig(PrivateLoggerConfig config);

    /** Sets the {@link PrivateLoggerConnector} for uploading data. */
    public abstract Builder setConnector(PrivateLoggerConnector connector);

    /** Sets the {@link TimeSource} for timestamping log entries. */
    public abstract Builder setTimeSource(TimeSource timeSource);

    /** Sets the {@link Executor} for database operations. */
    public abstract Builder setDbExecutor(Executor dbExecutor);

    /** Sets the {@link StabilityMonitor} for reporting crashes. */
    public abstract Builder setStabilityMonitor(StabilityMonitor stabilityMonitor);

    /** Builds the {@link PrivateLoggerImplDeps} instance. */
    public abstract PrivateLoggerImplDeps build();
  }
}
