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

import androidx.room.Entity;
import androidx.room.ForeignKey;
import androidx.room.Index;
import androidx.room.PrimaryKey;

/** Entity representing an entry contribution to a task. */
@Entity(
    foreignKeys = {
      @ForeignKey(
          entity = Task.class,
          parentColumns = "id",
          childColumns = "taskId",
          onDelete = ForeignKey.CASCADE)
    },
    indices = {
      @Index(value = {"taskId"}),
      @Index(
          value = {"taskId", "valueId"},
          unique = true),
      @Index(value = {"insertionTimeMillis"})
    })
public class EntryContribution {
  /** Unique identifier for the contribution record. */
  @PrimaryKey(autoGenerate = true)
  public int id;

  /** The ID of the task this contribution is for. */
  public int taskId;

  /** The ID of the entry from the source LogEntry table. */
  public long valueId;

  /** The time this entry was inserted, in milliseconds since epoch. */
  public long insertionTimeMillis;
}
