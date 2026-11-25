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

import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import androidx.annotation.VisibleForTesting;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;

/**
 * Utility for performing a one-time migration of contribution records from the legacy shared
 * TaskContributionTracker database ("task-contribution-database") into a logger's local Room
 * database.
 *
 * <p>TODO: b/558175230 - Once this migration has been live in production across active clients for
 * 270 days, this migration code will be removed and replaced with a cleanup migration that deletes
 * the legacy database file.
 */
final class LegacyContributionMigration {

  @VisibleForTesting static final String MIGRATION_KEY_SHARED_TRACKER = "shared_tracker_migrated";
  private static final String LEGACY_DB_NAME = "task-contribution-database";
  private static final int BATCH_SIZE = 1000;

  /**
   * Performs the migration asynchronously on the given executor.
   *
   * @param context The Android context used to locate the legacy database file.
   * @param database The local {@link PrivateLoggerDatabase} to migrate records into.
   * @param logSourceName The private log source name for this logger.
   * @param executor The executor to run the migration on.
   * @param stabilityMonitor The stability monitor to report non-fatal errors to.
   * @return A {@link ListenableFuture} that completes when migration finishes or is skipped.
   */
  static ListenableFuture<Void> migrate(
      Context context,
      PrivateLoggerDatabase database,
      String logSourceName,
      Executor executor,
      StabilityMonitor stabilityMonitor) {
    return Futures.submit(
        () -> {
          try {
            // Check if we've already completed this migration. This is a cheap indexed lookup
            // that avoids ever touching the old DB file after the first successful run.
            if (database.databaseMetadataDao().hasKey(MIGRATION_KEY_SHARED_TRACKER)) {
              return null;
            }
            File oldDbFile = context.getDatabasePath(LEGACY_DB_NAME);
            if (!oldDbFile.exists()) {
              // No old DB to migrate from. Mark as completed so we don't check the filesystem
              // again on future startups.
              markMigrationCompleted(database);
              return null;
            }
            try (SQLiteDatabase oldDb =
                    SQLiteDatabase.openDatabase(
                        oldDbFile.getAbsolutePath(), null, SQLiteDatabase.OPEN_READONLY);
                Cursor cursor =
                    oldDb.rawQuery(
                        "SELECT ec.valueId, ec.insertionTimeMillis, t.taskName "
                            + "FROM EntryContribution ec "
                            + "INNER JOIN Task t ON ec.taskId = t.id "
                            + "WHERE t.logSource = ?",
                        new String[] {logSourceName})) {
              database.runInTransaction(
                  () -> {
                    Map<String, Long> taskIdMap = new HashMap<>();
                    List<ContributedEntry> entries = new ArrayList<>();
                    while (cursor.moveToNext()) {
                      long valueId = cursor.getLong(0);
                      long insertionTimeMillis = cursor.getLong(1);
                      String taskName = cursor.getString(2);

                      Long taskId =
                          taskIdMap.computeIfAbsent(
                              taskName, database.taskDao()::getOrCreateTaskId);

                      ContributedEntry entry = new ContributedEntry();
                      entry.taskId = taskId;
                      entry.entryId = valueId;
                      entry.insertionTimeMillis = insertionTimeMillis;
                      entries.add(entry);
                      // Flush in batches to bound peak heap memory usage on memory-constrained
                      // Android devices when migrating large legacy databases, while maintaining
                      // atomicity via the enclosing database transaction.
                      if (entries.size() >= BATCH_SIZE) {
                        database.contributedEntryDao().insertAll(entries);
                        entries.clear();
                      }
                    }
                    if (!entries.isEmpty()) {
                      database.contributedEntryDao().insertAll(entries);
                    }
                    markMigrationCompleted(database);
                  });
            }
          } catch (RuntimeException e) {
            stabilityMonitor.reportCrash(e);
          }
          return null;
        },
        executor);
  }

  private static void markMigrationCompleted(PrivateLoggerDatabase database) {
    DatabaseMetadata metadata = new DatabaseMetadata();
    metadata.key = MIGRATION_KEY_SHARED_TRACKER;
    metadata.value = "true";
    database.databaseMetadataDao().putIfAbsent(metadata);
  }

  private LegacyContributionMigration() {}
}
