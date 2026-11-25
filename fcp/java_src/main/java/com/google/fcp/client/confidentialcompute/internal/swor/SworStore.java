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

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import android.content.Context;
import androidx.annotation.Nullable;
import androidx.annotation.VisibleForTesting;
import androidx.datastore.core.CorruptionException;
import androidx.datastore.core.Serializer;
import androidx.datastore.core.handlers.ReplaceFileCorruptionHandler;
import androidx.datastore.guava.GuavaDataStore;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import kotlin.Unit;
import kotlin.coroutines.Continuation;

/**
 * Stores the start time of the last upload per workload set, for sampling without replacement
 * (SWOR). Backed by a Jetpack DataStore file holding a {@link SworState}.
 *
 * <p>The store also keeps the time from which its records can be trusted: the first time it is
 * used, or the first use after a corrupt file was replaced; it never moves after that. Before that
 * time the store cannot tell whether an upload happened, so a SWOR period reaching back past it
 * counts as within the period. This matches the opstats {@code earliest_trustworthy_time} check in
 * {@code ComputePerTaskSworEligibility}, and the within-period check matches {@code
 * ExecutionOutsideSworPeriod}, both in {@code fcp/client/eligibility_decider.cc}.
 */
public final class SworStore {

  /**
   * How long a record is kept. Server SWOR periods are at most 31 days, which is why 60 days is
   * enough; a record past it is dropped the next time the store is checked.
   */
  @VisibleForTesting static final Duration RETENTION = Duration.ofDays(60);

  /** File under {@code Context.getFilesDir()/datastore} that holds the state. */
  @VisibleForTesting static final String FILE_NAME = "fcp_swor_store.pb";

  private static final Object lock = new Object();

  // DataStore allows one instance per file per process, so the production store is a singleton.
  // It cannot be final: creating it needs the Context, which is only available in forContext.
  @SuppressWarnings("NonFinalStaticField")
  @GuardedBy("lock")
  @Nullable
  private static GuavaDataStore<SworState> sharedDataStore;

  private final GuavaDataStore<SworState> dataStore;

  /** Returns the store for this process, backed by {@link #FILE_NAME} in the app's files dir. */
  public static SworStore forContext(Context context) {
    synchronized (lock) {
      if (sharedDataStore == null) {
        sharedDataStore =
            new GuavaDataStore.Builder<>(context.getApplicationContext(), FILE_NAME, serializer())
                .setCorruptionHandler(corruptionHandler())
                .build();
      }
      return new SworStore(sharedDataStore);
    }
  }

  /** Returns a serializer for {@link SworState}, for tests that build their own DataStore. */
  @VisibleForTesting
  public static Serializer<SworState> serializer() {
    return SworStateSerializer.INSTANCE;
  }

  private static final class SworStateSerializer implements Serializer<SworState> {
    static final SworStateSerializer INSTANCE = new SworStateSerializer();

    @Override
    public SworState getDefaultValue() {
      return SworState.getDefaultInstance();
    }

    @Override
    public Object readFrom(InputStream input, Continuation<? super SworState> continuation) {
      try {
        return SworState.parseFrom(input, ExtensionRegistryLite.getEmptyRegistry());
      } catch (InvalidProtocolBufferException e) {
        throw rethrowUnchecked(new CorruptionException("Cannot read proto.", e));
      } catch (IOException e) {
        throw rethrowUnchecked(e);
      }
    }

    @Override
    public Object writeTo(
        SworState t, OutputStream output, Continuation<? super Unit> continuation) {
      try {
        t.writeTo(output);
        return Unit.INSTANCE;
      } catch (IOException e) {
        throw rethrowUnchecked(e);
      }
    }

    // Kotlin suspend methods do not declare checked exceptions in Java bytecode, so we rethrow
    // CorruptionException/IOException directly via type-erased generic cast for DataStore to catch.
    @SuppressWarnings("unchecked")
    private static <E extends Throwable> RuntimeException rethrowUnchecked(Throwable t) throws E {
      throw (E) t;
    }
  }

  /**
   * Replaces a file that cannot be parsed with the empty state. Without it a corrupt file would
   * fail every read, and {@link SworTracker} fails closed on a failed read, so no workload set
   * would be uploaded to again.
   */
  @VisibleForTesting
  static ReplaceFileCorruptionHandler<SworState> corruptionHandler() {
    return new ReplaceFileCorruptionHandler<>(unused -> SworState.getDefaultInstance());
  }

  @VisibleForTesting
  public SworStore(GuavaDataStore<SworState> dataStore) {
    this.dataStore = dataStore;
  }

  /**
   * Returns the keys in {@code sworPeriods} (key to SWOR period) whose last recorded upload is
   * within their period, or whose period reaches back past the time the store's records can be
   * trusted from. Also drops records older than {@link #RETENTION}.
   */
  public ListenableFuture<ImmutableSet<String>> keysWithinSworPeriod(
      Map<String, Duration> sworPeriods, Instant now) {
    return Futures.transform(
        dataStore.updateDataAsync(
            state -> withoutExpiredRecords(withTrustworthyTime(state, now), now)),
        state -> {
          ImmutableSet.Builder<String> within = ImmutableSet.builder();
          for (Map.Entry<String, Duration> entry : sworPeriods.entrySet()) {
            if (isWithinSworPeriod(state, entry.getKey(), entry.getValue(), now)) {
              within.add(entry.getKey());
            }
          }
          return within.build();
        },
        directExecutor());
  }

  private static boolean isWithinSworPeriod(
      SworState state, String key, Duration sworPeriod, Instant now) {
    Instant trustworthySince = Instant.ofEpochMilli(state.getEarliestTrustworthyEpochMillis());
    if (Duration.between(trustworthySince, now).compareTo(sworPeriod) < 0) {
      return true;
    }
    Long lastUploadMillis = state.getLastUploadEpochMillisMap().get(key);
    if (lastUploadMillis == null) {
      return false;
    }
    // A record in the future (clock moved back) counts as within the period, as in C++.
    return Duration.between(Instant.ofEpochMilli(lastUploadMillis), now).compareTo(sworPeriod) <= 0;
  }

  /**
   * Records {@code now} as the upload time of {@code key} and returns the time it replaced, if any,
   * so that {@link #restoreUpload} can undo the record.
   */
  public ListenableFuture<Optional<Instant>> recordUpload(String key, Instant now) {
    AtomicReference<Optional<Instant>> previous = new AtomicReference<>(Optional.empty());
    return Futures.transform(
        dataStore.updateDataAsync(
            state -> {
              Long previousMillis = state.getLastUploadEpochMillisMap().get(key);
              previous.set(Optional.ofNullable(previousMillis).map(Instant::ofEpochMilli));
              return withTrustworthyTime(state, now).toBuilder()
                  .putLastUploadEpochMillis(key, now.toEpochMilli())
                  .build();
            }),
        unused -> previous.get(),
        directExecutor());
  }

  /**
   * Puts back the upload time of {@code key} that {@link #recordUpload} returned, provided the
   * stored upload time is still {@code recordedTime}. If another upload has recorded since, the
   * store is left alone.
   */
  public ListenableFuture<Void> restoreUpload(
      String key, Instant recordedTime, Optional<Instant> previous) {
    return Futures.transform(
        dataStore.updateDataAsync(state -> {
          Long currentMillis = state.getLastUploadEpochMillisMap().get(key);
          if (currentMillis == null || currentMillis != recordedTime.toEpochMilli()) {
            return state;
          }
          SworState.Builder builder = state.toBuilder();
          if (previous.isPresent()) {
            builder.putLastUploadEpochMillis(key, previous.get().toEpochMilli());
          } else {
            builder.removeLastUploadEpochMillis(key);
          }
          return builder.build();
        }),
        unused -> null,
        directExecutor());
  }

  /** Returns {@code state} without the records older than {@link #RETENTION}. */
  private static SworState withoutExpiredRecords(SworState state, Instant now) {
    long cutoffMillis = now.minus(RETENTION).toEpochMilli();
    @Nullable SworState.Builder builder = null;
    for (Map.Entry<String, Long> entry : state.getLastUploadEpochMillisMap().entrySet()) {
      if (entry.getValue() < cutoffMillis) {
        if (builder == null) {
          builder = state.toBuilder();
        }
        builder.removeLastUploadEpochMillis(entry.getKey());
      }
    }
    return builder == null ? state : builder.build();
  }

  /** Returns {@code state} with its trustworthy time set to {@code now} if it has none yet. */
  private static SworState withTrustworthyTime(SworState state, Instant now) {
    if (state.getEarliestTrustworthyEpochMillis() != 0) {
      return state;
    }
    return state.toBuilder().setEarliestTrustworthyEpochMillis(now.toEpochMilli()).build();
  }

  /** Returns the last recorded upload time of {@code key}, if any. */
  public ListenableFuture<Optional<Instant>> lastUpload(String key) {
    return Futures.transform(
        dataStore.getDataAsync(),
        state -> {
          Long millis = state.getLastUploadEpochMillisMap().get(key);
          return Optional.ofNullable(millis).map(Instant::ofEpochMilli);
        },
        directExecutor());
  }
}
