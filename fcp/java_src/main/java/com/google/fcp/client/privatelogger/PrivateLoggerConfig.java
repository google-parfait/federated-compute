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
package com.google.fcp.client.privatelogger;

import com.google.auto.value.AutoValue;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.time.Duration;

/**
 * Configuration for a {@link PrivateLogger} instance.
 *
 * <p>This class holds settings that are static for the lifetime of the logger instance.
 */
@AutoValue
public abstract class PrivateLoggerConfig {

  /**
   * Gets the target private log source the logger should upload to.
   *
   * @return The name of the Private Log Source.
   */
  public abstract String getPrivateLogSourceName();

  /**
   * Gets the maximum duration for which log entries are stored on the device. Entries older than
   * this period are guaranteed to never be uploaded, and will be deleted upon the next creation of
   * the PrivateLogger instance.
   *
   * @return The retention period.
   */
  public abstract Duration getRetentionPeriod();

  /**
   * Creates a new builder for {@link PrivateLoggerConfig}.
   *
   * @return A new Builder instance.
   */
  public static Builder builder() {
    return new AutoValue_PrivateLoggerConfig.Builder();
  }

  /** Builder for {@link PrivateLoggerConfig}. */
  @AutoValue.Builder
  public abstract static class Builder {

    /**
     * Sets the target Private Log Source to upload to.
     *
     * @param privateLogSourceName The name of the Private Log Source.
     * @return This builder.
     */
    @CanIgnoreReturnValue
    public abstract Builder setPrivateLogSourceName(String privateLogSourceName);

    /**
     * Sets the maximum duration for which log entries are stored.
     *
     * @param retentionPeriod The retention period.
     * @return This builder.
     */
    @CanIgnoreReturnValue
    public abstract Builder setRetentionPeriod(Duration retentionPeriod);

    /**
     * Builds the {@link PrivateLoggerConfig} instance.
     *
     * @return The configured PrivateLoggerConfig.
     */
    public abstract PrivateLoggerConfig build();
  }
}
