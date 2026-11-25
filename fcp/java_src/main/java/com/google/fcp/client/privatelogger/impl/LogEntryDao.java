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

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;
import java.util.List;

/**
 * Data Access Object (DAO) for {@link DbLogEntry} entities. Provides methods to interact with the
 * DbLogEntry table in the {@link PrivateLoggerDatabase}.
 */
@Dao
public interface LogEntryDao {
  /**
   * Inserts a list of log entries into the database in a single transaction.
   *
   * @param dbLogEntries The entries to insert.
   */
  @Insert
  void insertAll(List<DbLogEntry> dbLogEntries);

  /**
   * Retrieves all log entries from the database that are newer than the given timestamp.
   *
   * @param cutoffTimestampMillis The timestamp in millis since epoch. Entries with a later
   *     timestamp will be returned.
   * @return A list of all DbLogEntry objects.
   */
  // Room database APIs work with primitive types, and we need epoch millis for reliable
  // cross-timezone date comparisons.
  @SuppressWarnings("GoodTime-ApiWithNumericTimeUnit")
  @Query("SELECT * FROM DbLogEntry WHERE timestampMillis > :cutoffTimestampMillis")
  List<DbLogEntry> getAllAfter(long cutoffTimestampMillis);

  /**
   * Retrieves all log entries newer than the given cutoff timestamp that have not yet been
   * contributed to the specified task ID.
   *
   * @param taskId The ID of the task.
   * @param cutoffTimestampMillis The timestamp in millis since epoch. Only entries logged after
   *     this timestamp will be returned. By filtering with a cutoff, we ensure that only entries
   *     newer than the retention period are returned even if post-upload cleanup has not yet run.
   * @return A list of uncontributed {@link DbLogEntry} objects.
   */
  @SuppressWarnings("GoodTime-ApiWithNumericTimeUnit")
  @Query(
      "SELECT d.* FROM DbLogEntry d "
          + "WHERE d.timestampMillis > :cutoffTimestampMillis "
          + "AND NOT EXISTS ("
          + "  SELECT 1 FROM ContributedEntry c "
          + "  WHERE c.entryId = d.id AND c.taskId = :taskId)")
  List<DbLogEntry> getUncontributedEntries(long taskId, long cutoffTimestampMillis);

  /**
   * Deletes log entries from the database that are older than the given timestamp.
   *
   * @param cutoffTimestampMillis The timestamp in millis since epoch. Entries with an earlier
   *     timestamp will be deleted.
   */
  // Room database APIs work with primitive types, and we need epoch millis for reliable
  // cross-timezone date comparisons.
  @SuppressWarnings("GoodTime-ApiWithNumericTimeUnit")
  @Query("DELETE FROM DbLogEntry WHERE timestampMillis < :cutoffTimestampMillis")
  void deleteOlderThan(long cutoffTimestampMillis);
}
