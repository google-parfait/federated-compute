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
package com.google.fcp.client.contributiontracker;

import static java.util.concurrent.TimeUnit.SECONDS;

import android.content.Context;
import android.database.sqlite.SQLiteConstraintException;
import androidx.annotation.GuardedBy;
import androidx.annotation.VisibleForTesting;
import androidx.room.Room;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;

/**
 * Manages records of entry contributions to tasks, ensuring that the same entry is not
 * re-contributed to the same task for a given Private Log Source.
 *
 * <h3>Purpose:</h3>
 *
 * <p>This class tracks which data entries (identified by their unique IDs) have been contributed to
 * which task. Tasks are identified by a composite key of {@code (logSource, taskName)}. This helps
 * prevent duplicate data submissions to the same task, especially when tasks might be retried or
 * run multiple times on the same device. This class is designed to work with a single database for
 * all Private Log Sources contributed to by the app.
 *
 * <h3>Lifetime and Workflow:</h3>
 *
 * <p>A `TaskContributionTracker` instance is typically created per upload task execution. The
 * underlying Room database connection is automatically closed after a period of inactivity via
 * auto-close. The typical workflow for a single task run is as follows:
 *
 * <pre>
 * 1. Data provider is called for an upload task.
 *     |
 *     V
 * 2. Data provider selects data entries.
 *     - tracker.commitContributions(logSource, taskName, entryIds)
 *     |
 *     V
 * 3. The upload task processes the selected entries.
 *     |
 *     V
 * 4. Upload task result is successfully uploaded.
 *     - tracker.clearCommittedContributions()
 * </pre>
 *
 * <p>If the upload task fails at any point after `commitContributions` is called, {@link
 * #rollbackContributions(String, String)} should be called to roll back the contributions from the
 * database. This is necessary because the upload task might fail after data has been selected but
 * before it's successfully uploaded. We only want to permanently mark data as contributed if the
 * upload succeeds. Committed contributions are written to the persistent database but also tracked
 * in memory. `rollbackContributions` uses the in-memory list to remove them. If the process crashes
 * after a commit but before a rollback, the in-memory state is lost, and the committed
 * contributions cannot be rolled back. However, if we successfully upload then crash, the
 * contributions will still be tracked in the database, and thus won't be contributed again if the
 * task is retried.
 *
 * <h3>API Usage:</h3>
 *
 * <ol>
 *   <li><b>Commit Contributions:</b> As data entries are selected for a specific task, call {@link
 *       #commitContributions(String, String, List)} for all entries at once. This writes the
 *       records to the database and tracks them in memory for potential rollback.
 *   <li><b>Rollback Contributions:</b> If the upload task fails after commit, call {@link
 *       #rollbackContributions(String, String)} to remove the committed contributions.
 *   <li><b>Check Contributed IDs:</b> Before processing data for a task, use {@link
 *       #getContributedIds(String, String)} to retrieve the set of entry IDs that have already been
 *       committed for this task. This allows the data selection logic to filter out these entries.
 *   <li><b>Cleanup:</b> Old contribution records are automatically cleaned up every time a new
 *       {@code TaskContributionTracker} is created via {@link #create}, based on the
 *       `retentionPeriod` provided. Since a new tracker should be created for each upload task run,
 *       this ensures regular cleanup.
 * </ol>
 *
 * <h3>Database Schema:</h3>
 *
 * <p>This tracker uses a Room database ({@link TaskContributionDatabase}) with two main tables:
 *
 * <ol>
 *   <li><b>Task:</b> Stores information about the tasks.
 *       <ul>
 *         <li>`id`: INTEGER (Primary Key, Auto-generated) - Unique identifier for the task record.
 *         <li>`logSource`: TEXT - The name of the Private Log Source.
 *         <li>`taskName`: TEXT - The name of the task.
 *         <li>Unique constraint on (`logSource`, `taskName`).
 *       </ul>
 *   <li><b>EntryContribution:</b> Stores records of individual data entries contributed to tasks.
 *       <ul>
 *         <li>`id`: INTEGER (Primary Key, Auto-generated) - Unique identifier for the contribution
 *             record.
 *         <li>`taskId`: INTEGER - Foreign key referencing `Task.id`. Indicates which task this
 *             contribution belongs to. CASCADE on delete.
 *         <li>`valueId`: LONG - The ID of the data entry from the source table.
 *         <li>`insertionTimeMillis`: LONG - The timestamp (ms since epoch) when this contribution
 *             record was committed. Used for retention enforcement.
 *       </ul>
 *       <p>Indices are created on `taskId`, (`taskId`, `valueId`) (unique constraint), and
 *       `insertionTimeMillis` to optimize queries and enforce constraints.
 * </ol>
 *
 * <h3>Thread Safety:</h3>
 *
 * <p>This class is thread-safe. All public methods can be safely called from multiple threads.
 * Operations are executed on a {@link Executor} to avoid blocking the calling thread. To ensure
 * that operations are correctly ordered, the provided executor **must** be sequential (e.g.,
 * single-threaded). Critical sections that modify both the database and in-memory state are
 * synchronized to ensure atomicity and prevent race conditions.
 */
public class TaskContributionTracker {

  private static final String DATABASE_NAME = "task-contribution-database";
  private static final int CLOSE_AFTER_UNUSED_SECONDS = 60;

  private final TimeSource timeSource;
  private final Executor executor;

  @GuardedBy("lock")
  @VisibleForTesting
  final TaskDao taskDao;

  @GuardedBy("lock")
  @VisibleForTesting
  final EntryContributionDao entryContributionDao;

  @GuardedBy("lock")
  private final TaskContributionDatabase database;

  // Map of logSource -> taskName -> list of entry IDs.
  @GuardedBy("lock")
  private final Map<String, Map<String, List<Long>>> committedContributions = new HashMap<>();

  private final Duration retentionPeriod;
  private final Object lock = new Object();
  private final StabilityMonitor stabilityMonitor;

  /**
   * Creates a new TaskContributionTracker instance and performs an initial cleanup of old entries.
   *
   * @param context The application context.
   * @param timeSource The time source to use for timestamps.
   * @param executor The executor to use for background tasks. This executor *must* be sequential
   *     (e.g., a single-threaded executor) to guarantee that database operations are correctly
   *     ordered.
   * @param retentionPeriod the duration for which entries are retained. Entries older than this
   *     will be removed.
   * @return The created TaskContributionTracker instance.
   */
  public static TaskContributionTracker create(
      Context context,
      TimeSource timeSource,
      Executor executor,
      Duration retentionPeriod,
      StabilityMonitor stabilityMonitor) {
    TaskContributionTracker tracker =
        new TaskContributionTracker(
            context, timeSource, executor, retentionPeriod, stabilityMonitor);
    // Fire and forget the cleanup.
    ListenableFuture<Void> unused = tracker.cleanupOldEntries();
    return tracker;
  }

  private TaskContributionTracker(
      Context context,
      TimeSource timeSource,
      Executor executor,
      Duration retentionPeriod,
      StabilityMonitor stabilityMonitor) {
    this.database =
        Room.databaseBuilder(context, TaskContributionDatabase.class, DATABASE_NAME)
            .allowMainThreadQueries() // A bit dangerous, but useful for testing.
            .setAutoCloseTimeout(CLOSE_AFTER_UNUSED_SECONDS, SECONDS)
            .setQueryExecutor(executor)
            .build();
    this.taskDao = database.taskDao();
    this.entryContributionDao = database.entryContributionDao();
    this.timeSource = timeSource;
    this.executor = executor;
    this.retentionPeriod = retentionPeriod;
    this.stabilityMonitor = stabilityMonitor;
  }

  /**
   * Cleans up old entries from the database that are older than the retention period. Also removes
   * any tasks that no longer have any associated contributions.
   */
  @VisibleForTesting
  ListenableFuture<Void> cleanupOldEntries() {
    return Futures.submit(
        () -> {
          synchronized (lock) {
            Instant cutoffTime = timeSource.now().minus(retentionPeriod);
            entryContributionDao.deleteOlderThan(cutoffTime.toEpochMilli());

            // Clean up tasks with no more contributions.
            List<Integer> taskIds = taskDao.getAllTaskIds();
            for (int taskId : taskIds) {
              if (entryContributionDao.countContributionsForTask(taskId) == 0) {
                taskDao.deleteTaskById(taskId);
              }
            }
          }
          return null;
        },
        executor);
  }

  /**
   * Commits contributions for the given task.
   *
   * <p>This makes the contributions permanent in the database, but tracks them in memory so they
   * can be rolled back with {@link #rollbackContributions(String, String)}.
   *
   * @param logSource The name of the Private Log Source.
   * @param taskName The name of the task for which to commit contributions.
   * @param entryIds The IDs of the entries being contributed.
   * @return A ListenableFuture that completes when the commit operation is finished.
   */
  public ListenableFuture<Void> commitContributions(
      String logSource, String taskName, List<Long> entryIds) {
    return Futures.submit(
        () -> {
          try {
            synchronized (lock) {
              if (entryIds == null || entryIds.isEmpty()) {
                // No contributions to commit.
                return null;
              }

              Task task = taskDao.getTaskByName(logSource, taskName);
              if (task == null) {
                task = new Task();
                task.logSource = logSource;
                task.taskName = taskName;
                // With a sequential executor, we don't expect a race where another thread inserts
                // the same task between our get and insert calls. The returned task ID should
                // always be valid.
                long taskId = taskDao.insert(task);
                if (taskId == -1) {
                  // This should not happen with a sequential executor, but if it does, it's an
                  // unexpected error state.
                  throw new IllegalStateException(
                      "Failed to insert a new task for "
                          + logSource
                          + "/"
                          + taskName
                          + "; insert returned -1.");
                }
                task.id = (int) taskId;
              }

              long insertionTimeMillis = timeSource.now().toEpochMilli();
              List<EntryContribution> records = new ArrayList<>();
              for (long entryId : entryIds) {
                EntryContribution record = new EntryContribution();
                record.taskId = task.id;
                record.valueId = entryId;
                record.insertionTimeMillis = insertionTimeMillis;
                records.add(record);
              }
              entryContributionDao.insertAll(records);
              committedContributions
                  .computeIfAbsent(logSource, k -> new HashMap<>())
                  .put(taskName, entryIds);
            }
          } catch (SQLiteConstraintException e) {
            // Since exceptions in executor tasks are normally swallowed, we use StabilityMonitor
            // to report crashes to the system's crash reporting mechanism (e.g., DropboxManager)
            // for visibility into system bugs.
            stabilityMonitor.reportCrash(e);
            Thread.currentThread()
                .getUncaughtExceptionHandler()
                .uncaughtException(Thread.currentThread(), e);
            throw e;
          } catch (RuntimeException e) {
            stabilityMonitor.reportCrash(e);
          }
          return null;
        },
        executor);
  }

  /**
   * Rolls back contributions for the given task that were previously committed with {@link
   * #commitContributions(String, String, List)}.
   *
   * <p>This removes the contribution records from the database based on the in-memory list of
   * committed contributions.
   *
   * @param logSource The name of the Private Log Source.
   * @param taskName The name of the task for which to roll back contributions.
   * @return A ListenableFuture that completes when the rollback operation is finished.
   */
  public ListenableFuture<Void> rollbackContributions(String logSource, String taskName) {
    return Futures.submit(
        () -> {
          synchronized (lock) {
            Map<String, List<Long>> privateDataSourceContributions =
                committedContributions.get(logSource);
            if (privateDataSourceContributions == null) {
              return null;
            }
            List<Long> entryIds = privateDataSourceContributions.remove(taskName);
            if (entryIds == null || entryIds.isEmpty()) {
              return null;
            }

            Task task = taskDao.getTaskByName(logSource, taskName);
            if (task != null) {
              entryContributionDao.deleteContributions(task.id, entryIds);
            }
          }
          return null;
        },
        executor);
  }

  /**
   * Returns a list of all entry IDs that have been contributed to the given task.
   *
   * @param logSource The name of the Private Log Source.
   * @param taskName The name of the task.
   * @return A ListenableFuture containing the list of entry IDs.
   */
  public ListenableFuture<List<Long>> getContributedIds(String logSource, String taskName) {
    return Futures.submit(
        () -> {
          synchronized (lock) {
            return entryContributionDao.getContributedEntryIdsForTaskName(logSource, taskName);
          }
        },
        executor);
  }

  /**
   * Clears all in-memory committed contribution records.
   *
   * <p>Should be called once an upload has fully completed (including any rollbacks), at which
   * point the in-memory rollback data is no longer needed. The persistent database records are not
   * affected.
   */
  public void clearCommittedContributions() {
    synchronized (lock) {
      committedContributions.clear();
    }
  }

  /** Returns the number of tasks in the database. For testing only. */
  @VisibleForTesting
  ListenableFuture<Integer> getTaskCount() {
    return Futures.submit(
        () -> {
          synchronized (lock) {
            return taskDao.getAllTaskIds().size();
          }
        },
        executor);
  }

  /** Returns the task with the given name. For testing only. */
  @VisibleForTesting
  ListenableFuture<Task> getTask(String logSource, String taskName) {
    return Futures.submit(
        () -> {
          synchronized (lock) {
            return taskDao.getTaskByName(logSource, taskName);
          }
        },
        executor);
  }
}
