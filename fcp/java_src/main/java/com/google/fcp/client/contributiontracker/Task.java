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
import androidx.room.Index;
import androidx.room.PrimaryKey;

/** Entity representing a task that has been contributed to. */
@Entity(
    indices = {
      @Index(
          value = {"logSource", "taskName"},
          unique = true)
    })
public class Task {
  /** Unique identifier for the task. */
  @PrimaryKey(autoGenerate = true)
  public int id;

  /** The name of the PrivateLogSource this task contributes to. */
  public String logSource;

  /**
   * The name of the task. This is the primary identifier used for looking up contributions for a
   * given task.
   */
  public String taskName;
  // Add any auxiliary task-level information here.
}
