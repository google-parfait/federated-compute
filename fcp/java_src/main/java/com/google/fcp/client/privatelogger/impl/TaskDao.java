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

import androidx.annotation.Nullable;
import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;
import com.google.errorprone.annotations.CanIgnoreReturnValue;

/** Data Access Object (DAO) for {@link Task} entities in the {@link PrivateLoggerDatabase}. */
@Dao
public interface TaskDao {

  /**
   * Retrieves the task ID for a given task name, or null if not found.
   *
   * @param taskName The name of the task.
   * @return The task ID, or null.
   */
  @Nullable
  @Query("SELECT id FROM Task WHERE taskName = :taskName")
  Long getTaskId(String taskName);

  /**
   * Inserts a new task into the database. If a task with the same name already exists, the insert
   * is ignored and -1 is returned.
   *
   * @param task The task to insert.
   * @return The row ID of the newly inserted task, or -1 if the task already exists.
   */
  @Insert(onConflict = OnConflictStrategy.IGNORE)
  long insert(Task task);

  /**
   * Retrieves the task ID for a given task name, creating the task if it does not already exist.
   *
   * @param taskName The name of the task.
   * @return The task ID.
   */
  default long getOrCreateTaskId(String taskName) {
    Task task = new Task();
    task.taskName = taskName;
    long insertedId = insert(task);
    if (insertedId != -1) {
      return insertedId;
    }
    Long existingId = getTaskId(taskName);
    return existingId != null ? existingId : -1;
  }

  /**
   * Deletes tasks that have no referencing entries in the {@link ContributedEntry} table.
   *
   * @return The number of deleted orphan tasks.
   */
  @CanIgnoreReturnValue
  @Query("DELETE FROM Task WHERE id NOT IN (SELECT DISTINCT taskId FROM ContributedEntry)")
  int deleteOrphanTasks();
}
