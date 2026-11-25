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
import androidx.annotation.Nullable;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

/**
 * A generic key-value metadata table used to store database-level flags, migration states, and
 * configuration within the {@link PrivateLoggerDatabase}.
 */
@Entity
public class DatabaseMetadata {
  /** A unique key identifying the metadata entry. */
  @PrimaryKey @NonNull public String key;

  /** Arbitrary string value or payload associated with this key. */
  @Nullable public String value;
}
