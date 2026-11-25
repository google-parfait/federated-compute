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
package com.google.fcp.client.confidentialcompute;

import com.google.common.util.concurrent.ListenableFuture;

/**
 * Uploads encrypted confidential aggregation payloads via the ByteStream protocol and submits the
 * aggregation result (corresponding to {@code HttpFederatedProtocol::UploadResult} in {@code
 * fcp/client/http/http_federated_protocol.cc}).
 */
public interface ConfidentialPayloadUploader {

  /**
   * Uploads {@code payload} for {@code workloadSet} and submits the aggregation result.
   *
   * <p>Each {@link WorkloadSet} may be uploaded at most once. An unuploaded workload set requires
   * no cleanup; its aggregation session expires on the server.
   *
   * <p>If the blob upload fails for any reason other than {@code ABORTED} or refused authorization,
   * the uploader sends a best-effort {@code AbortConfidentialAggregation} before failing the
   * returned future. A failure during the subsequent {@code SubmitConfidentialAggregationResult}
   * step sends no abort, matching {@code HttpFederatedProtocol::UploadResult}.
   *
   * <p>The returned future fails with {@link UploadAbortedException} when the payload is known not
   * to have been stored: the server answered the blob upload with {@code ABORTED} or refused
   * authorization (such as an expired session), or the upload was not started. Any other failure
   * surfaces the original exception, and the payload may or may not have been stored (the server
   * does not discard a stored blob on abort); callers that must never contribute the same data
   * twice should treat such a failure as contributed.
   *
   * <p>The call itself does not throw when the uploader's executor has been shut down; the returned
   * future fails with {@link java.util.concurrent.RejectedExecutionException} instead.
   */
  ListenableFuture<Void> upload(WorkloadSet workloadSet, EncodedPayload payload);
}
