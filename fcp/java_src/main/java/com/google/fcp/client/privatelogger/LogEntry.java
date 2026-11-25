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
import com.google.errorprone.annotations.Immutable;
import com.google.fcp.client.confidentialcompute.TimestampedInput;
import com.google.protobuf.ByteString;

/**
 * Represents a single entry that is passed to the Federated Compute protocol implementation. This
 * is a data object for data in transit.
 */
@AutoValue
@Immutable
public abstract class LogEntry implements TimestampedInput {

  /** The serialized value of the entry. */
  @Override
  public abstract ByteString value();

  /**
   * The creation timestamp of the entry, as a string formatted according to ISO-8601 with the
   * device's local time-zone offset. For example: "2016-09-12T08:00:00.123-07:00".
   */
  @Override
  public abstract String timestamp();

  /** Creates a new builder for {@link LogEntry}. */
  public static Builder builder() {
    return new AutoValue_LogEntry.Builder();
  }

  /** Builder for {@link LogEntry}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets the serialized value of the entry. */
    public abstract Builder setValue(ByteString value);

    /** Sets the creation timestamp of the entry. */
    public abstract Builder setTimestamp(String timestamp);

    /** Builds the {@link LogEntry} instance. */
    public abstract LogEntry build();
  }
}
