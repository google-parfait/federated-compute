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

import androidx.annotation.NonNull;
import androidx.room.Entity;
import androidx.room.Index;
import androidx.room.PrimaryKey;

/**
 * Represents a task that contributions can be recorded against.
 *
 * <p>Stores a unique mapping from task name to an integer ID, allowing {@link ContributedEntry}
 * records to reference tasks by integer ID rather than duplicating the task name string.
 */
@Entity(
    indices = {
      @Index(
          value = {"taskName"},
          unique = true)
    })
public class Task {
  /** Automatically generated unique identifier for this task. */
  @PrimaryKey(autoGenerate = true)
  public long id;

  /** The name of the task. */
  @NonNull public String taskName;
}
