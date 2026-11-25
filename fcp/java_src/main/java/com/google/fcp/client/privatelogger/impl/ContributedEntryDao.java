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
package com.google.fcp.client.privatelogger.impl;

import androidx.annotation.VisibleForTesting;
import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;
import java.util.List;

/**
 * Data Access Object (DAO) for {@link ContributedEntry} entities in the {@link
 * PrivateLoggerDatabase}.
 */
@Dao
public interface ContributedEntryDao {

  /** Inserts a list of contributed entries into the database, ignoring duplicates. */
  // OnConflictStrategy.IGNORE guarantees idempotence. Crucially, this ensures the one-time
  // legacy data migration (LegacyContributionMigration.migrate) is safely retryable.
  // If the app is killed mid-migration before writing the completion flag to DatabaseMetadata,
  // subsequent startup retries will silently ignore already-copied rows instead of throwing
  // fatal SQLiteConstraintExceptions.
  @Insert(onConflict = OnConflictStrategy.IGNORE)
  void insertAll(List<ContributedEntry> entries);

  /**
   * Marks uncontributed {@link DbLogEntry} rows newer than {@code cutoffTimestampMillis} as
   * contributed for the specified task ID.
   *
   * <p>When invoked within the same database transaction as {@link
   * LogEntryDao#getUncontributedEntries(long, long)} with the same {@code taskId} and {@code
   * cutoffTimestampMillis}, the identical {@code WHERE} clause guarantees that the exact set of
   * entries returned to the caller is marked as contributed in SQLite without instantiating {@link
   * ContributedEntry} objects in Java. {@code INSERT OR IGNORE} ensures idempotency against the
   * unique {@code (taskId, entryId)} index.
   *
   * <p>Note: We cannot combine the read and insert into a single {@code INSERT INTO ... RETURNING}
   * query because SQLite's {@code RETURNING} clause requires SQLite 3.35.0+, which is not available
   * on our minimum supported Android SDK level.
   */
  @SuppressWarnings("GoodTime-ApiWithNumericTimeUnit")
  @Query(
      "INSERT OR IGNORE INTO ContributedEntry (taskId, entryId, insertionTimeMillis) "
          + "SELECT :taskId, d.id, :insertionTimeMillis FROM DbLogEntry d "
          + "WHERE d.timestampMillis > :cutoffTimestampMillis "
          + "AND NOT EXISTS ("
          + "  SELECT 1 FROM ContributedEntry c "
          + "  WHERE c.entryId = d.id AND c.taskId = :taskId)")
  void markEntriesAsContributedForTimeRange(
      long taskId, long cutoffTimestampMillis, long insertionTimeMillis);

  /**
   * Deletes contribution records for a given task ID and list of entry IDs (used for rollback).
   *
   * @param taskId The ID of the task.
   * @param entryIds The list of entry IDs to delete.
   */
  @Query("DELETE FROM ContributedEntry WHERE taskId = :taskId AND entryId IN (:entryIds)")
  void deleteContributions(long taskId, List<Long> entryIds);

  /**
   * Deletes contribution records older than the given cutoff time.
   *
   * @param cutoffTimestampMillis The cutoff timestamp in millis since epoch.
   */
  @SuppressWarnings("GoodTime-ApiWithNumericTimeUnit")
  @Query("DELETE FROM ContributedEntry WHERE insertionTimeMillis < :cutoffTimestampMillis")
  void deleteOlderThan(long cutoffTimestampMillis);

  /**
   * Retrieves all contributed entry IDs for a given task ID. For testing only.
   *
   * @param taskId The ID of the task.
   * @return A list of contributed entry IDs.
   */
  @VisibleForTesting(otherwise = VisibleForTesting.NONE)
  @Query("SELECT entryId FROM ContributedEntry WHERE taskId = :taskId")
  List<Long> getContributedEntryIds(long taskId);
}
