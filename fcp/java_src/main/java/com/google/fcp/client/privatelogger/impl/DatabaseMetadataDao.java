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

/** Data Access Object (DAO) for {@link DatabaseMetadata} in the {@link PrivateLoggerDatabase}. */
@Dao
public interface DatabaseMetadataDao {

  /** Returns true if a metadata entry with the given key exists. */
  @Query("SELECT EXISTS(SELECT 1 FROM DatabaseMetadata WHERE `key` = :key)")
  boolean hasKey(String key);

  /** Retrieves the value associated with the given key, or null if not found. */
  @Nullable
  @Query("SELECT value FROM DatabaseMetadata WHERE `key` = :key")
  String getValue(String key);

  /** Inserts or updates a metadata entry. */
  @Insert(onConflict = OnConflictStrategy.REPLACE)
  void put(DatabaseMetadata metadata);

  /** Inserts a metadata entry if it doesn't already exist. */
  @Insert(onConflict = OnConflictStrategy.IGNORE)
  void putIfAbsent(DatabaseMetadata metadata);
}
