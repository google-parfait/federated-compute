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

import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.common.util.concurrent.MoreExecutors.newSequentialExecutor;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.concurrent.TimeUnit.SECONDS;

import android.content.Context;
import android.database.sqlite.SQLiteConstraintException;
import android.database.sqlite.SQLiteException;
import android.util.Log;
import androidx.annotation.GuardedBy;
import androidx.annotation.VisibleForTesting;
import androidx.room.Room;
import com.google.common.collect.Lists;
import com.google.common.hash.Hashing;
import com.google.common.util.concurrent.FluentFuture;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.privatelogger.LogEntry;
import com.google.fcp.client.privatelogger.PrivateLogger;
import com.google.fcp.client.privatelogger.PrivateLoggerConfig;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import com.google.intelligence.fcp.client.SelectorContext;
import com.google.protobuf.ByteString;
import com.google.protobuf.MessageLite;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;

/**
 * Main implementation of the {@link PrivateLogger} interface.
 *
 * <h3>Overview:</h3>
 *
 * <p>This class provides a concrete implementation of {@link PrivateLogger}, handling the storage,
 * retrieval, and management of on-device data for uploads. It uses a single Room database ({@link
 * PrivateLoggerDatabase}) containing four tables:
 *
 * <ul>
 *   <li><b>{@link DbLogEntry}:</b> Stores the raw data logged via {@link #log}.
 *   <li><b>{@link Task}:</b> Maps unique task names to integer IDs.
 *   <li><b>{@link ContributedEntry}:</b> Keeps track of which {@link DbLogEntry} IDs have been
 *       contributed to which task, preventing duplicate uploads.
 *   <li><b>{@link DatabaseMetadata}:</b> Key-value store for database-level flags and migration
 *       state.
 * </ul>
 *
 * <p><b>Important:</b> This implementation requires a concrete {@link PrivateLoggerConnector} to be
 * provided to connect to the platform's federated compute protocol implementation and execute the
 * upload. Without a functional connector, the {@link #upload} method will not be able to send data
 * to the server.
 *
 * <h3>Data Retention and Cleanup:</h3>
 *
 * <p>Data logged to this store is not kept indefinitely. The retention period is defined in the
 * {@link PrivateLoggerConfig} provided during construction. Entries older than the retention period
 * are deleted from the database when the {@link PrivateLoggerImpl} is instantiated, on a
 * best-effort basis after each {@link #upload} call, and periodically during logging (every {@value
 * #LOGS_PER_CLEANUP} logs since initialization).
 *
 * <p>Because private loggers are often long-lived singletons, performing cleanup post-upload and
 * periodically during logging ensures that databases are pruned regularly without relying on
 * frequent logger re-instantiation. The periodic log counter is held in memory and resets each time
 * the logger is instantiated or when cleanup is triggered.
 *
 * <h3>Thread Safety:</h3>
 *
 * <p>This class is thread-safe. All public methods ({@link #log} and {@link #upload}) can be safely
 * called from any thread.
 *
 * <p>All database operations are performed asynchronously on the {@link Executor} provided during
 * construction. To guarantee that database operations are executed in the order they are received,
 * the provided executor is wrapped in a sequential executor.
 *
 * <h3>Workflow:</h3>
 *
 * <ol>
 *   <li>{@link #log(MessageLite)}: Serializes the proto and stores it in {@link
 *       PrivateLoggerDatabase} with the current timestamp.
 *   <li>{@link #upload()}:
 *       <ol>
 *         <li>Invokes {@link PrivateLoggerConnector#upload}, providing a method reference to {@code
 *             getData(SelectorContext)} on an {@link UploadSession}. The connector interacts with
 *             the implementation of the federated compute protocol.
 *         <li>The platform uses the {@code getData} method to fetch a list of log entries for a
 *             specific task.
 *         <li>The {@code getData} method queries for uncontributed entries (checking that no
 *             matching record exists in {@link ContributedEntry}), records pending contributions in
 *             {@link ContributedEntry}, and returns the results.
 *         <li>When the {@code ListenableFuture<List<UploadOutcome>>} returned by the connector
 *             completes, this class handles the result. If any uploads were not contributed, their
 *             contributions are rolled back via {@link ContributedEntryDao#deleteContributions}.
 *         <li>The returned {@link ListenableFuture} from this class's {@code upload} method is
 *             completed when the entire process (including any rollbacks) is finished.
 *       </ol>
 * </ol>
 */
public class PrivateLoggerImpl implements PrivateLogger {

  private static final String TAG = "PrivateLoggerImpl";
  private static final int CLOSE_AFTER_UNUSED_SECONDS = 60;
  @VisibleForTesting static final int LOGS_PER_CLEANUP = 1000;

  private final PrivateLoggerConfig config;
  private final PrivateLoggerConnector connector;
  private final Executor executor;
  private final TimeSource timeSource;
  private final Context context;
  private final PrivateLoggerDatabase database;
  private final StabilityMonitor stabilityMonitor;
  private int logCount = 0;

  private final Object lock = new Object();

  @GuardedBy("lock")
  private ListenableFuture<Void> uploadFuture;

  /**
   * Creates a new PrivateLoggerImpl instance.
   *
   * @param deps The {@link PrivateLoggerImplDeps} with the dependencies for this logger instance.
   */
  public static PrivateLoggerImpl create(PrivateLoggerImplDeps deps) {
    Executor sequentialExecutor = newSequentialExecutor(deps.dbExecutor());
    PrivateLoggerImpl logger =
        new PrivateLoggerImpl(
            deps.context(),
            deps.config(),
            deps.connector(),
            sequentialExecutor,
            deps.timeSource(),
            deps.stabilityMonitor());
    // Fire and forget the cleanup and one-time legacy migration.
    var unused1 = logger.cleanupOldEntries();
    ListenableFuture<Void> unused2 = logger.migrateFromSharedTracker();
    return logger;
  }

  private PrivateLoggerImpl(
      Context context,
      PrivateLoggerConfig config,
      PrivateLoggerConnector connector,
      Executor executor,
      TimeSource timeSource,
      StabilityMonitor stabilityMonitor) {
    this.context = context;
    this.config = config;
    this.connector = connector;
    this.executor = executor;
    this.timeSource = timeSource;
    this.stabilityMonitor = stabilityMonitor;
    this.database = createDatabase();
  }

  /**
   * Generates a unique and filename-safe hash for the Private Log Source name. This is used to
   * ensure the database name is a valid filename, as the original name may contain characters that
   * are not allowed in file names.
   */
  @VisibleForTesting
  static String getDatabaseName(String privateLogSourceName) {
    return "privatelogger-database-" + Hashing.sha256().hashString(privateLogSourceName, UTF_8);
  }

  /** Creates a new instance of the Room database. */
  private PrivateLoggerDatabase createDatabase() {
    return Room.databaseBuilder(
            this.context.getApplicationContext(),
            PrivateLoggerDatabase.class,
            getDatabaseName(config.getPrivateLogSourceName()))
        .allowMainThreadQueries() // A bit dangerous, but useful for testing. This is safe because
        // the main thread queries are only done in tests.
        .setAutoCloseTimeout(CLOSE_AFTER_UNUSED_SECONDS, SECONDS)
        // setAutoCloseTimeout will close the database on the QueryExecutor. The queryExecutor needs
        // to be set to the same executor that we run transactions on.
        .setQueryExecutor(executor)
        .build();
  }

  /**
   * Performs a one-time migration of contribution records from the legacy shared
   * TaskContributionTracker database into this logger's local ContributedEntry table.
   */
  @VisibleForTesting
  ListenableFuture<Void> migrateFromSharedTracker() {
    return LegacyContributionMigration.migrate(
        context, database, config.getPrivateLogSourceName(), executor, stabilityMonitor);
  }

  @Override
  public void log(MessageLite event) {
    log(Collections.singletonList(event));
  }

  @Override
  public void log(List<? extends MessageLite> events) {
    executor.execute(() -> {
      try {
        if (events == null || events.isEmpty()) {
          return;
        }
        Instant now = timeSource.now();
        long timestampMillis = now.toEpochMilli();
        String timestamp = TimestampUtil.getIso8601String(now);

        List<DbLogEntry> dbLogEntries = new ArrayList<>(events.size());
        for (MessageLite event : events) {
          DbLogEntry dbLogEntry = new DbLogEntry();
          dbLogEntry.value = event.toByteArray();
          dbLogEntry.timestampMillis = timestampMillis;
          dbLogEntry.timestamp = timestamp;
          dbLogEntries.add(dbLogEntry);
        }
        database.logEntryDao().insertAll(dbLogEntries);
        logCount += dbLogEntries.size();
        // Periodically perform retention cleanup every LOGS_PER_CLEANUP logs, and reset the
        // counter.
        if (logCount >= LOGS_PER_CLEANUP) {
          logCount %= LOGS_PER_CLEANUP;
          var unused = cleanupOldEntries();
        }
      } catch (RuntimeException e) {
        reportUnlessDatabaseError(e);
      }
    });
  }

  /**
   * Reports {@code e} as a bug unless it is a database error. SQLite throws unchecked exceptions
   * for conditions that are not bugs, such as a full disk or a corrupt file, and the default {@link
   * StabilityMonitor} crashes the process on a report. A database error only fails the operation it
   * happened in. A write on a full disk is dropped. A corrupt database file is deleted by Room (the
   * default {@code SupportSQLiteOpenHelper.Callback#onCorruption}), so the next open starts with an
   * empty database. A constraint violation is a bug and is still reported.
   */
  private void reportUnlessDatabaseError(Throwable e) {
    if (isDatabaseError(e)) {
      Log.w(TAG, "Database error", e);
      return;
    }
    stabilityMonitor.reportCrash(e);
  }

  private static boolean isDatabaseError(Throwable e) {
    return e instanceof SQLiteException && !(e instanceof SQLiteConstraintException);
  }

  @VisibleForTesting
  int getLogCount() {
    return logCount;
  }

  /**
   * Removes entries older than the retention period and unreferenced tasks from the database.
   *
   * <p>This is best-effort: all errors (synchronous and asynchronous) are reported to the {@link
   * StabilityMonitor} and swallowed. The returned future always succeeds.
   */
  @VisibleForTesting
  ListenableFuture<Void> cleanupOldEntries() {
    try {
      ListenableFuture<Void> cleanupFuture =
          Futures.submit(
              () -> {
                Duration retentionPeriod = config.getRetentionPeriod();
                if (retentionPeriod != null) {
                  Instant cutoffInstant = timeSource.now().minus(retentionPeriod);
                  long cutoffMillis = cutoffInstant.toEpochMilli();
                  database.runInTransaction(
                      () -> {
                        database.logEntryDao().deleteOlderThan(cutoffMillis);
                        database.contributedEntryDao().deleteOlderThan(cutoffMillis);
                        database.taskDao().deleteOrphanTasks();
                      });
                }
                return null;
              },
              executor);
      return FluentFuture.from(cleanupFuture)
          .catching(
              Exception.class,
              e -> {
                reportUnlessDatabaseError(e);
                return null;
              },
              directExecutor());
    } catch (RuntimeException e) {
      reportUnlessDatabaseError(e);
      return immediateVoidFuture();
    }
  }

  /**
   * Initiates an upload task.
   *
   * <p>This method may be called on the main thread, but the {@link PrivateLoggerConnector} is
   * responsible for executing the upload asynchronously.
   *
   * @return A {@link ListenableFuture} that completes when the entire upload process finishes.
   *     Cancelling the returned future does not stop the upload.
   */
  @Override
  public ListenableFuture<Void> upload() {
    synchronized (lock) {
      if (uploadFuture != null && !uploadFuture.isDone()) {
        return Futures.nonCancellationPropagating(uploadFuture);
      }
      UploadSession session = new UploadSession();
      ListenableFuture<List<UploadOutcome>> outcomesFuture =
          connector.upload(config.getPrivateLogSourceName(), session::getData);

      ListenableFuture<Void> uploadAndRollbackFuture =
          Futures.transformAsync(
              outcomesFuture,
              outcomes -> {
                List<ListenableFuture<Void>> rollbackFutures = new ArrayList<>();
                for (UploadOutcome outcome : outcomes) {
                  if (outcome.status().equals(UploadOutcome.Status.NOT_CONTRIBUTED)) {
                    rollbackFutures.add(session.rollbackContributions(outcome.taskName()));
                  }
                }
                return Futures.transform(
                    Futures.allAsList(rollbackFutures),
                    // We don't care about the result of the rollbacks, just that they're all done.
                    ignored -> null,
                    directExecutor());
              },
              executor);

      // Best-effort cleanup post-upload: run under the caller's wakelock, ensuring cleanup
      // failures never fail the upload nor mask upload failures.
      ListenableFuture<Void> newUploadFuture =
          Futures.whenAllComplete(uploadAndRollbackFuture)
              .callAsync(
                  () ->
                      FluentFuture.from(cleanupOldEntries())
                          // Return the original upload result so the caller sees the real outcome.
                          .transformAsync(unused -> uploadAndRollbackFuture, directExecutor()),
                  directExecutor());
      // Add a listener to clear the future from the instance variable once it's completed.
      // This ensures that subsequent calls to upload() will trigger a new upload.
      newUploadFuture.addListener(
          () -> {
            synchronized (lock) {
              if (this.uploadFuture == newUploadFuture) {
                this.uploadFuture = null;
              }
            }
          },
          executor);
      this.uploadFuture = newUploadFuture;
      return Futures.nonCancellationPropagating(uploadFuture);
    }
  }

  private class UploadSession {
    private final Object sessionLock = new Object();

    @GuardedBy("sessionLock")
    private final Map<String, List<Long>> committedContributions = new HashMap<>();

    /**
     * Fetches and filters log entries for a given task.
     *
     * @param selectorContext The context for the task, containing metadata like the task name.
     * @return A {@link ListenableFuture} containing the list of filtered {@link LogEntry} protos.
     */
    ListenableFuture<List<LogEntry>> getData(SelectorContext selectorContext) {
      String taskName = selectorContext.getComputationProperties().getFederated().getTaskName();

      return Futures.submit(
          () -> {
            try {
              List<LogEntry> filteredEntries = new ArrayList<>();
              List<Long> entryIdsToCommit = new ArrayList<>();

              database.runInTransaction(() -> {
                // Get or create the task ID.
                long taskId = database.taskDao().getOrCreateTaskId(taskName);
                Duration retentionPeriod = config.getRetentionPeriod();
                Instant now = timeSource.now();
                long cutoffMillis = now.minus(retentionPeriod).toEpochMilli();
                long nowMillis = now.toEpochMilli();
                List<DbLogEntry> uncontributedEntries =
                    database.logEntryDao().getUncontributedEntries(taskId, cutoffMillis);

                // Both getUncontributedEntries and markEntriesAsContributedForTimeRange run in
                // the same transaction with identical WHERE predicates and parameters, so no
                // concurrent writes can interleave and the rows marked as contributed in SQLite
                // match the rows returned and tracked in entryIdsToCommit for rollback.
                if (!uncontributedEntries.isEmpty()) {
                  database
                      .contributedEntryDao()
                      .markEntriesAsContributedForTimeRange(taskId, cutoffMillis, nowMillis);
                }

                for (DbLogEntry dbLogEntry : uncontributedEntries) {
                  LogEntry.Builder logEntryBuilder =
                      LogEntry.builder()
                          .setValue(ByteString.copyFrom(dbLogEntry.value))
                          .setTimestamp(dbLogEntry.timestamp);
                  filteredEntries.add(logEntryBuilder.build());
                  entryIdsToCommit.add((long) dbLogEntry.id);
                }
              });

              // Track committed entry IDs in this session to support rollback if the upload fails.
              if (!entryIdsToCommit.isEmpty()) {
                synchronized (sessionLock) {
                  committedContributions
                      .computeIfAbsent(taskName, k -> new ArrayList<>())
                      .addAll(entryIdsToCommit);
                }
              }

              return filteredEntries;
            } catch (RuntimeException e) {
              if (isDatabaseError(e)) {
                // Not a bug, see reportUnlessDatabaseError. The checked exception tells the
                // connector that the read failed, so it does not report a bug either.
                Log.w(TAG, "Failed to read log entries", e);
                throw new IOException("Failed to read log entries", e);
              }
              stabilityMonitor.reportCrash(e);
              throw e;
            }
          },
          executor);
    }

    private ListenableFuture<Void> rollbackContributions(String taskName) {
      return Futures.submit(
          () -> {
            try {
              List<Long> entryIds;
              synchronized (sessionLock) {
                entryIds = committedContributions.remove(taskName);
              }
              if (entryIds != null && !entryIds.isEmpty()) {
                Long taskId = database.taskDao().getTaskId(taskName);
                if (taskId != null) {
                  database.runInTransaction(() -> {
                    // Safely partition list to stay under SQLite's limit of 999 bound
                    // parameters per query. deleteContributions binds :taskId plus each
                    // entryId in chunk (1 + chunk.size()).
                    for (List<Long> chunk : Lists.partition(entryIds, 900)) {
                      database.contributedEntryDao().deleteContributions(taskId, chunk);
                    }
                  });
                }
              }
              return null;
            } catch (RuntimeException e) {
              reportUnlessDatabaseError(e);
              throw e;
            }
          },
          executor);
    }
  }
}
