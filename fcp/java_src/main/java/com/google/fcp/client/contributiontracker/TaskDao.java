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

/** DAO for Task entities. */
@Dao
public interface TaskDao {
  /**
   * Inserts a task into the database.
   *
   * @param task The task to insert.
   * @return The row ID of the newly inserted task, or -1 if the task already exists.
   */
  @Insert(onConflict = OnConflictStrategy.IGNORE)
  long insert(Task task);

  /**
   * Retrieves a task by its name.
   *
   * @param logSource The name of the PrivateLogSource.
   * @param taskName The name of the task. This is the primary identifier used for looking up
   *     contributions for a given task.
   * @return The Task object, or null if not found.
   */
  @Query("SELECT * FROM Task WHERE logSource = :logSource AND taskName =" + " :taskName")
  Task getTaskByName(String logSource, String taskName);

  /**
   * Deletes a task by its ID.
   *
   * @param taskId The ID of the task to delete.
   */
  @Query("DELETE FROM Task WHERE id = :taskId")
  void deleteTaskById(int taskId);

  /**
   * Retrieves all task IDs from the database.
   *
   * @return A list of all task IDs.
   */
  @Query("SELECT id FROM Task")
  List<Integer> getAllTaskIds();
}
