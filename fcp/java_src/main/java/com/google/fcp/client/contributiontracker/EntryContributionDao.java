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

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;
import java.util.List;

/** DAO for EntryContribution entities. */
@Dao
public interface EntryContributionDao {
  /**
   * Inserts an entry contribution into the database.
   *
   * @param entryContribution The entry contribution to insert.
   */
  @Insert(onConflict = OnConflictStrategy.IGNORE)
  void insert(EntryContribution entryContribution);

  /**
   * Inserts a list of entry contributions into the database.
   *
   * @param entryContributions The list of entry contributions to insert.
   */
  @Insert(onConflict = OnConflictStrategy.IGNORE)
  void insertAll(List<EntryContribution> entryContributions);

  /**
   * Deletes entry contributions older than the given cutoff time based on their insertion time.
   *
   * @param cutoffTimeMillis The cutoff time in milliseconds since epoch.
   */
  @Query("DELETE FROM EntryContribution WHERE insertionTimeMillis < :cutoffTimeMillis")
  void deleteOlderThan(long cutoffTimeMillis);

  /**
   * Counts the number of contributions for a given task.
   *
   * @param taskId The ID of the task.
   * @return The number of contributions.
   */
  @Query("SELECT COUNT(*) FROM EntryContribution WHERE taskId = :taskId")
  int countContributionsForTask(int taskId);

  /**
   * Retrieves all contributed entry IDs for a given task name using a JOIN.
   *
   * @param logSource The name of the PrivateLogSource.
   * @param taskName The name of the task.
   * @return A list of contributed entry IDs.
   */
  @Query(
      "SELECT ec.valueId FROM EntryContribution ec INNER JOIN Task t ON ec.taskId = t.id WHERE"
          + " t.logSource = :logSource AND t.taskName = :taskName")
  List<Long> getContributedEntryIdsForTaskName(String logSource, String taskName);

  /**
   * Deletes entry contributions for a given task ID and list of entry IDs.
   *
   * @param taskId The ID of the task.
   * @param entryIds The list of entry IDs to delete.
   */
  @Query("DELETE FROM EntryContribution WHERE taskId = :taskId AND valueId IN (:entryIds)")
  void deleteContributions(int taskId, List<Long> entryIds);
}
