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

import androidx.room.Entity;
import androidx.room.Index;
import androidx.room.PrimaryKey;

/**
 * Represents a log entry that has been contributed to a specific task.
 *
 * <p>Used by {@link PrivateLoggerImpl} to track contributions locally per log source, avoiding
 * re-contributing the same data to the same task.
 */
@Entity(
    indices = {
      @Index(
          value = {"taskId", "entryId"},
          unique = true),
      @Index(value = {"insertionTimeMillis"})
    })
public class ContributedEntry {
  /** Automatically generated unique identifier for this contribution record. */
  @PrimaryKey(autoGenerate = true)
  public int id;

  /** The ID of the {@link Task} this entry was contributed to. */
  public long taskId;

  /** The ID of the {@link DbLogEntry} that was contributed. */
  public long entryId;

  /**
   * The timestamp in milliseconds since epoch when this contribution was recorded.
   *
   * <p>This timestamp is used during retention cleanup to garbage-collect expired contribution
   * records whose age exceeds the configured retention period.
   */
  // Room database APIs work with primitive types, and we need epoch millis for reliable
  // cross-timezone date comparisons.
  @SuppressWarnings("GoodTime-ApiWithNumericTimeUnit")
  public long insertionTimeMillis;
}
