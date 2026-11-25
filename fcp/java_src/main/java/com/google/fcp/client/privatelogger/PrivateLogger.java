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
package com.google.fcp.client.privatelogger;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.protobuf.MessageLite;
import java.util.List;

/**
 * Interface for logging data to an on-device store and then uploading it to a private log source.
 * When connected to a Federated Compute protocol implementation
 * (https://github.com/google-parfait/federated-compute) and Confidential Federated Compute
 * (https://github.com/google-parfait/confidential-federated-compute) platform, this implements a
 * verifiably private way to upload the logged data to a private log source.
 *
 * <h3>Usage Workflow:</h3>
 *
 * <ol>
 *   <li><b>Initialization:</b> Obtain an instance of {@link PrivateLogger} (typically via a factory
 *       or dependency injection), configured with a {@link PrivateLoggerConfig}.
 *   <li><b>Logging:</b> Call {@link #log(MessageLite)} or {@link #log(List)} to add data entries to
 *       the store. This is an asynchronous operation.
 *       <pre>{@code
 * MyDataProto event = MyDataProto.newBuilder()...build();
 * logger.log(event);
 * }</pre>
 *   <li><b>Uploading:</b> Calling {@link #upload()} will trigger an upload to the private log
 *       source. The result of the uploading process is available through the returned {@link
 *       ListenableFuture}.
 *   <li><b>Cleanup:</b> Old data entries are automatically removed based on the retention policy
 *       upon logger initialization, post-upload, and periodically during logging.
 * </ol>
 */
public interface PrivateLogger {

  /**
   * Logs a new entry (as a proto message) to the on-device store. This operation is performed
   * asynchronously. Errors are not propagated to the caller, so if there is an error saving the
   * entry, the entry will be dropped. Entries are subject to the retention period defined in the
   * {@link PrivateLoggerConfig}. Best-effort retention cleanup is also automatically triggered
   * periodically during logging (e.g. every 1,000 logs since initialization; the counter resets
   * each time the logger is created).
   *
   * @param event The proto message to log.
   */
  void log(MessageLite event);

  /**
   * Logs a list of entries (as proto messages) to the on-device store in a single transaction. This
   * operation is performed asynchronously. Errors are not propagated to the caller, so if there is
   * an error saving the entries, the entries will be dropped. Entries are subject to the retention
   * period defined in the {@link PrivateLoggerConfig}.
   *
   * @param events The list of proto messages to log.
   */
  // Default implementation to avoid updating all interface implementations.
  default void log(List<? extends MessageLite> events) {
    if (events != null) {
      for (MessageLite event : events) {
        log(event);
      }
    }
  }

  /**
   * Triggers an upload of the locally stored data to the private log source and performs a
   * best-effort cleanup of expired entries based on the retention policy.
   *
   * <p>The caller must hold a wakelock until the future completes, e.g. via a background job.
   *
   * @return A {@link ListenableFuture} that completes when the entire upload process finishes.
   *     Cancelling the returned future does not stop the upload. If the future fails, callers
   *     should expect and gracefully handle {@link PrivateLoggerUnavailableException} for cases
   *     where the private logger implementation cannot be used (e.g. if the implementation was
   *     disabled by flag or the upload was cancelled). Any other exception should be treated as an
   *     unexpected error (e.g. by rethrowing them wrapped in a new UncheckedExecutionException and
   *     letting them trigger a crash report).
   */
  ListenableFuture<Void> upload();
}
