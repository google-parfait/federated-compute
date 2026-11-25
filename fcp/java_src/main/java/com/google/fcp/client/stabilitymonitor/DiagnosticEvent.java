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
package com.google.fcp.client.stabilitymonitor;

import androidx.annotation.Nullable;
import com.google.auto.value.AutoValue;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.time.Duration;
import java.time.Instant;

/**
 * A diagnostic event logged through {@link StabilityMonitor#logDiagnosticEvent} by the default
 * confidential compute stages and the private logger connector.
 *
 * <p>Most kinds correspond to a call on the C++ {@code PhaseLogger} ({@code
 * fcp/client/phase_logger.h}), named in their javadoc, so a host can log similar events for both
 * clients. The kinds are not a fixed mapping and may change. Fields that do not apply to a kind are
 * left at their defaults: null, zero or {@link Duration#ZERO}.
 *
 * <p>Every event carries the log source and the {@link #fetchId()}, and query, encode and upload
 * events name their workload set, so a host can group them without keeping state.
 */
@AutoValue
public abstract class DiagnosticEvent {

  /** What happened. */
  public enum Kind {
    /** The fetcher is about to check in. {@code LogEligibilityEvalCheckinStarted}. */
    CHECKIN_STARTED,
    /**
     * The check-in response arrived. Network stats and duration cover the check-in request. {@code
     * LogEligibilityEvalCheckinPlanUriReceived}.
     */
    CHECKIN_RECEIVED,
    /**
     * The population eligibility spec was downloaded and parsed. Network stats and duration cover
     * the whole check-in. Logged whether or not the spec has eligibility policies; the C++ client
     * logs {@code LogEligibilityEvalNotConfigured} or {@code LogEligibilityEvalCheckinCompleted}.
     */
    CHECKIN_COMPLETED,
    /** The server rejected the check-in. {@code LogEligibilityEvalCheckinTurnedAway}. */
    CHECKIN_REJECTED,
    /** The check-in failed. Has an error. {@code LogEligibilityEvalCheckinIOError}. */
    CHECKIN_ERROR,
    /** The fetcher is about to request assignments. {@code LogMultipleTaskAssignmentsStarted}. */
    ASSIGNMENT_STARTED,
    /**
     * The assignment response arrived. Network stats and duration cover the assignment request.
     * {@code LogMultipleTaskAssignmentsPlanUriReceived}.
     */
    ASSIGNMENT_RECEIVED,
    /**
     * Every assigned workload set was resolved. Network stats and duration cover the assignment
     * request and the resolution. {@code LogMultipleTaskAssignmentsCompleted}.
     */
    ASSIGNMENT_COMPLETED,
    /**
     * Some but not all assigned workload sets were resolved. Network stats and duration cover the
     * assignment request and the resolution. {@code LogMultipleTaskAssignmentsPartialCompleted}.
     */
    ASSIGNMENT_PARTIALLY_COMPLETED,
    /**
     * The server assigned zero workload sets. Network stats and duration cover the assignment
     * request. {@code LogMultipleTaskAssignmentsTurnedAway}.
     */
    ASSIGNMENT_REJECTED,
    /**
     * The assignment request, or the device attestation measurement it carries, failed. Has an
     * error. {@code LogMultipleTaskAssignmentsIOError}.
     */
    ASSIGNMENT_ERROR,
    /**
     * The caller is about to query the inputs for a workload set. {@code LogComputationStarted}.
     */
    QUERY_STARTED,
    /** The query returned inputs. Has their count and bytes. {@code LogComputationCompleted}. */
    QUERY_COMPLETED,
    /**
     * The query returned no inputs, so nothing is encoded or uploaded. {@code
     * LogComputationInsufficientData}.
     */
    QUERY_NO_DATA,
    /** The query failed. Has an error. {@code LogComputationExampleIteratorError}. */
    QUERY_ERROR,
    /**
     * The encoder is about to encode the inputs. No C++ counterpart: the C++ client encrypts the
     * payload as part of the upload.
     */
    ENCODE_STARTED,
    /** The inputs were encoded. */
    ENCODE_COMPLETED,
    /** Encoding failed. Has an error. */
    ENCODE_ERROR,
    /** The uploader is about to upload a payload. {@code LogResultUploadStarted}. */
    UPLOAD_STARTED,
    /** The payload was uploaded and submitted. {@code LogResultUploadCompleted}. */
    UPLOAD_COMPLETED,
    /**
     * The server aborted the upload, so the payload was not stored. Has an error. {@code
     * LogResultUploadServerAborted}.
     */
    UPLOAD_ABORTED,
    /**
     * The upload failed in some other way; the payload may have been stored. Has an error. {@code
     * LogResultUploadIOError}.
     */
    UPLOAD_ERROR,
  }

  public abstract Kind kind();

  /** The log source the workload sets were fetched for. */
  @Nullable
  public abstract String logSourceName();

  /**
   * Id of the fetch the event belongs to (see {@code WorkloadSet.fetchId()}), or 0 if not set. A
   * host can use it to group a fetch's events.
   */
  public abstract long fetchId();

  /** The workload set, for query, encode and upload events. */
  @Nullable
  public abstract String workloadSetName();

  /**
   * Time since the matching {@code *_STARTED} event. Measured with the ConfidentialCompute ticker
   * (monotonic, keeps counting during sleep), not derived from timestamps.
   */
  public abstract Duration duration();

  /**
   * Wall-clock time the event was logged, from the TimeSource the client was built with. Use it to
   * order events or to line them up with other logs; use duration() for how long a step took, which
   * is measured with a monotonic clock and is not affected by wall-clock changes.
   */
  @Nullable
  public abstract Instant timestamp();

  /**
   * HTTP request body bytes sent since the matching {@code *_STARTED} event. Headers are not
   * counted, so this is lower than the C++ client's network stats.
   */
  public abstract long bytesUploaded();

  /**
   * HTTP response body bytes received since the matching {@code *_STARTED} event, as received
   * (before any gzip decoding). Headers are not counted.
   */
  public abstract long bytesDownloaded();

  /** Number of inputs the query returned. */
  public abstract int inputCount();

  /** Total size of the input values the query returned. */
  public abstract long inputBytes();

  /** The failure, for error and abort events. */
  @Nullable
  public abstract Throwable error();

  /** Returns a builder for an event of {@code kind} with every other field at its default. */
  public static Builder builder(Kind kind) {
    return new AutoValue_DiagnosticEvent.Builder()
        .setKind(kind)
        .setFetchId(0)
        .setDuration(Duration.ZERO)
        .setBytesUploaded(0)
        .setBytesDownloaded(0)
        .setInputCount(0)
        .setInputBytes(0);
  }

  /** Builder for {@link DiagnosticEvent}. */
  @AutoValue.Builder
  public abstract static class Builder {
    abstract Builder setKind(Kind kind);

    @CanIgnoreReturnValue
    public abstract Builder setLogSourceName(@Nullable String logSourceName);

    @CanIgnoreReturnValue
    public abstract Builder setFetchId(long fetchId);

    @CanIgnoreReturnValue
    public abstract Builder setWorkloadSetName(@Nullable String workloadSetName);

    @CanIgnoreReturnValue
    public abstract Builder setDuration(Duration duration);

    @CanIgnoreReturnValue
    public abstract Builder setTimestamp(@Nullable Instant timestamp);

    @CanIgnoreReturnValue
    public abstract Builder setBytesUploaded(long bytesUploaded);

    @CanIgnoreReturnValue
    public abstract Builder setBytesDownloaded(long bytesDownloaded);

    @CanIgnoreReturnValue
    public abstract Builder setInputCount(int inputCount);

    @CanIgnoreReturnValue
    public abstract Builder setInputBytes(long inputBytes);

    @CanIgnoreReturnValue
    public abstract Builder setError(@Nullable Throwable error);

    public abstract DiagnosticEvent build();
  }
}
