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
package com.google.fcp.client.confidentialcompute.internal.swor;

import android.content.Context;
import androidx.annotation.Nullable;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.confidentialcompute.UploadAbortedException;
import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutionException;

/**
 * Sampling without replacement (SWOR): a workload set is not contributed to again within its SWOR
 * period. SWOR is needed until every workload in a workload set uses privacy IDs.
 *
 * <p>The default fetcher skips a workload set whose last upload is within its SWOR period before
 * requesting a task assignment for it, and the default uploader records the upload time before the
 * blob upload and restores the previous time if the server aborts the upload.
 *
 * <p>Corresponds to {@code ComputePerTaskSworEligibility} in {@code
 * fcp/client/eligibility_decider.cc}, which reads the time of the last {@code
 * EVENT_KIND_RESULT_UPLOAD_STARTED} for a task name from opstats. Other eligibility policies are
 * not evaluated, as in the legacy client. Records are keyed by workload set name.
 *
 * <p>Every method blocks on the store and is meant to run on the stage's executor.
 */
public final class SworTracker {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** Skips nothing and records nothing. */
  public static final SworTracker DISABLED =
      new SworTracker(/* store= */ null, TimeSource.REAL_TIME_SOURCE);

  @Nullable private final SworStore store;
  private final TimeSource timeSource;

  /**
   * Returns a tracker backed by the process-wide {@link SworStore} for {@code context} and reading
   * the time from {@code timeSource}.
   */
  public static SworTracker forContext(Context context, TimeSource timeSource) {
    return new SworTracker(SworStore.forContext(context), timeSource);
  }

  /** Creates a tracker backed by {@code store} and reading the time from {@code timeSource}. */
  public SworTracker(@Nullable SworStore store, TimeSource timeSource) {
    this.store = store;
    this.timeSource = timeSource;
  }

  /**
   * Returns the workload sets in {@code sworPeriods} (workload set name to SWOR period) that were
   * uploaded to within their period, or whose period the store's records do not cover (see {@link
   * SworStore}). One store read covers all of them. A store error counts as within the period for
   * every set, failing closed like {@code ComputePerTaskSworEligibility}.
   */
  public ImmutableSet<String> workloadSetsWithinPeriod(Map<String, Duration> sworPeriods)
      throws InterruptedException {
    if (store == null || sworPeriods.isEmpty()) {
      return ImmutableSet.of();
    }
    try {
      ImmutableSet<String> withinPeriod =
          store.keysWithinSworPeriod(sworPeriods, timeSource.now()).get();
      if (!withinPeriod.isEmpty()) {
        logger.atFine().log("Workload sets within their SWOR period: %s", withinPeriod);
      }
      return withinPeriod;
    } catch (ExecutionException e) {
      logger.atWarning().withCause(e).log(
          "Failed to check the SWOR periods; skipping all workload sets");
      return ImmutableSet.copyOf(sworPeriods.keySet());
    }
  }

  /**
   * Records the current time as the upload time of {@code workloadSetName}. Throws {@link
   * UploadAbortedException} if the store write fails.
   */
  public SworRecord recordUpload(String workloadSetName)
      throws InterruptedException, UploadAbortedException {
    if (store == null) {
      return SworRecord.NOT_RECORDED;
    }
    Instant now = timeSource.now();
    try {
      Optional<Instant> previous = store.recordUpload(workloadSetName, now).get();
      return new SworRecord(workloadSetName, now, previous);
    } catch (ExecutionException e) {
      logger.atWarning().withCause(e).log(
          "Failed to record the SWOR upload time for %s", workloadSetName);
      throw new UploadAbortedException(
          "Failed to record SWOR upload time for " + workloadSetName, e);
    }
  }

  /**
   * Puts back the upload time that {@link #recordUpload} replaced, after the server aborted the
   * upload, provided no other upload has recorded since. A store error is logged.
   */
  public void restoreUpload(SworRecord record) throws InterruptedException {
    if (store == null || record.workloadSetName == null || record.recordedTime == null) {
      return;
    }
    try {
      store.restoreUpload(record.workloadSetName, record.recordedTime, record.previous).get();
    } catch (ExecutionException e) {
      logger.atWarning().withCause(e).log(
          "Failed to restore the SWOR upload time for %s after an aborted upload",
          record.workloadSetName);
    }
  }

  /** The result of {@link #recordUpload}: the upload time it recorded and the one it replaced. */
  public static final class SworRecord {
    private static final SworRecord NOT_RECORDED =
        new SworRecord(/* workloadSetName= */ null, /* recordedTime= */ null, Optional.empty());

    // Null when nothing was recorded.
    @Nullable private final String workloadSetName;
    @Nullable private final Instant recordedTime;
    private final Optional<Instant> previous;

    private SworRecord(
        @Nullable String workloadSetName,
        @Nullable Instant recordedTime,
        Optional<Instant> previous) {
      this.workloadSetName = workloadSetName;
      this.recordedTime = recordedTime;
      this.previous = previous;
    }
  }
}
