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

import androidx.room.AutoMigration;
import androidx.room.Database;
import androidx.room.RoomDatabase;

/**
 * The Room database definition for the {@link com.google.fcp.client.privatelogger.PrivateLogger}.
 * It stores {@link DbLogEntry}, {@link ContributedEntry}, {@link Task}, and {@link
 * DatabaseMetadata} entities.
 */
@Database(
    entities = {DbLogEntry.class, ContributedEntry.class, Task.class, DatabaseMetadata.class},
    version = 2,
    exportSchema = true,
    autoMigrations = {@AutoMigration(from = 1, to = 2)})
public abstract class PrivateLoggerDatabase extends RoomDatabase {
  /**
   * Provides access to the {@link DbLogEntry} table.
   *
   * @return The Data Access Object for DbLogEntry.
   */
  public abstract LogEntryDao logEntryDao();

  /**
   * Provides access to the {@link ContributedEntry} table.
   *
   * @return The Data Access Object for ContributedEntry.
   */
  public abstract ContributedEntryDao contributedEntryDao();

  /**
   * Provides access to the {@link Task} table.
   *
   * @return The Data Access Object for Task.
   */
  public abstract TaskDao taskDao();

  /**
   * Provides access to the {@link DatabaseMetadata} table.
   *
   * @return The Data Access Object for DatabaseMetadata.
   */
  public abstract DatabaseMetadataDao databaseMetadataDao();
}
