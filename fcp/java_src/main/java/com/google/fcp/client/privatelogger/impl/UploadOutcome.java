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

import com.google.auto.value.AutoValue;
import com.google.errorprone.annotations.Immutable;

/** Represents the outcome of an upload attempt for a single task. */
@AutoValue
@Immutable
public abstract class UploadOutcome {

  /** The status of the upload. */
  public enum Status {
    /** Data was likely contributed to the server. */
    CONTRIBUTED,
    /** The upload failed, and no data conclusively reached the server. */
    NOT_CONTRIBUTED
  }

  /** The name of the upload task that was executed. */
  public abstract String taskName();

  /** The result of the upload attempt. */
  public abstract Status status();

  /** Creates a new builder for {@link UploadOutcome}. */
  public static Builder builder() {
    return new AutoValue_UploadOutcome.Builder();
  }

  /** Builder for {@link UploadOutcome}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets the name of the upload task. */
    public abstract Builder setTaskName(String taskName);

    /** Sets the status of the upload. */
    public abstract Builder setStatus(Status status);

    /** Builds the {@link UploadOutcome} instance. */
    public abstract UploadOutcome build();
  }
}
