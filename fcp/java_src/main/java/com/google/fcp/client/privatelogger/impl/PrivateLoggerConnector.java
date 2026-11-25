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

import com.google.common.util.concurrent.ListenableFuture;
import java.util.List;

/**
 * Interface for the platform-specific connector that bridges the {@link PrivateLogger} to the
 * underlying federated compute protocol implementation.
 */
public interface PrivateLoggerConnector {

  /**
   * Initiates an upload.
   *
   * @param privateLogSourceName The target private log source.
   * @param dataProvider A function that provides access to the on-device data.
   * @return A {@link ListenableFuture} that completes with a list of {@link UploadOutcome}s when
   *     the upload process finishes.
   */
  ListenableFuture<List<UploadOutcome>> upload(
      String privateLogSourceName, DataProvider dataProvider);

}
