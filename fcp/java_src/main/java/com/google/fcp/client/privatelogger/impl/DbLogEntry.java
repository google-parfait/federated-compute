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

import androidx.room.Entity;
import androidx.room.Index;
import androidx.room.PrimaryKey;

/**
 * Represents a single piece of data logged by the {@link
 * com.google.fcp.client.privatelogger.PrivateLogger}. This entity is used by Room to create the
 * underlying database table.
 */
@Entity(indices = {@Index(value = {"timestampMillis"})})
public class DbLogEntry {
  /** Automatically generated unique identifier for this log entry. */
  @PrimaryKey(autoGenerate = true)
  public int id;

  /** The serialized form of the logged proto message. */
  public byte[] value;

  /**
   * The timestamp in milliseconds since epoch when this entry was logged. Used for database
   * operations where we need to compare absolute timestamps.
   */
  // Room database APIs work with primitive types, and we need epoch millis for reliable
  // cross-timezone date comparisons.
  @SuppressWarnings("GoodTime-ApiWithNumericTimeUnit")
  public long timestampMillis;

  /**
   * The timestamp as a string with timezone info when this entry was logged. Formatted according to
   * ISO-8601. E.g. "2016-09-12T08:00:00.123-07:00".
   */
  public String timestamp;
}
