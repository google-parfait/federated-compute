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
package com.google.fcp.client.ossbuilddefs;

import androidx.room.util.SchemaFileResolver;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Bazel sandbox {@link SchemaFileResolver} that reads committed Room schema JSON files from the
 * execution root and discards schema writes so {@code RoomProcessor} does not attempt to overwrite
 * read-only input symlinks.
 */
public final class NoWriteSchemaFileResolver implements SchemaFileResolver {
  @Override
  public InputStream readPath(Path path) throws IOException {
    return Files.newInputStream(path);
  }

  @Override
  public OutputStream writePath(Path path) {
    return OutputStream.nullOutputStream();
  }
}
