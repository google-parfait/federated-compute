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
package com.google.fcp.client.confidentialcompute.internal.upload;

import static com.google.common.util.concurrent.Futures.immediateFailedFuture;

import com.google.common.base.Ticker;
import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.confidentialcompute.ConfidentialPayloadUploader;
import com.google.fcp.client.confidentialcompute.EncodedPayload;
import com.google.fcp.client.confidentialcompute.UploadAbortedException;
import com.google.fcp.client.confidentialcompute.UploadTokens;
import com.google.fcp.client.confidentialcompute.WorkloadSet;
import com.google.fcp.client.confidentialcompute.internal.fetch.AggregationAbortSender;
import com.google.fcp.client.confidentialcompute.internal.http.FcpHttpClient;
import com.google.fcp.client.confidentialcompute.internal.http.FcpHttpException;
import com.google.fcp.client.confidentialcompute.internal.http.NetworkPhase;
import com.google.fcp.client.confidentialcompute.internal.http.UriEncoding;
import com.google.fcp.client.confidentialcompute.internal.http.UrlRedaction;
import com.google.fcp.client.confidentialcompute.internal.swor.SworTracker;
import com.google.fcp.client.confidentialcompute.internal.swor.SworTracker.SworRecord;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent.Kind;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import com.google.intelligence.fcp.client.ProdDiagCode;
import com.google.internal.federatedcompute.v1.SubmitConfidentialAggregationResultRequest;
import com.google.rpc.Code;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;

/**
 * Uploads encrypted confidential aggregation payloads and reports aggregation completion.
 *
 * <p>Mirrors the confidential aggregation upload in {@code
 * fcp/client/http/http_federated_protocol.cc}: {@code HttpFederatedProtocol::UploadResult}, which
 * runs {@code UploadDataViaByteStreamProtocol}, calls {@code AbortAggregation} on any failure other
 * than ABORTED or authorization refusal, and otherwise finishes with {@code
 * SubmitAggregationResult}.
 *
 * <p>The SWOR upload time is recorded through {@link
 * FcpConfidentialPayloadUploaderDeps#sworTracker()} before the blob upload and restored if the
 * server aborts or refuses it.
 *
 * <p>Upload {@link DiagnosticEvent}s go to {@link
 * FcpConfidentialPayloadUploaderDeps#stabilityMonitor()}.
 */
public final class FcpConfidentialPayloadUploader implements ConfidentialPayloadUploader {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final FcpHttpClient httpClient;
  private final Executor executor;
  private final String apiKey;
  private final ImmutableMap<String, String> requestHeaders;
  private final SworTracker sworTracker;
  private final StabilityMonitor stabilityMonitor;
  private final Ticker ticker;
  private final TimeSource timeSource;

  public FcpConfidentialPayloadUploader(FcpConfidentialPayloadUploaderDeps deps) {
    this.httpClient =
        new FcpHttpClient(deps.httpEngine(), deps.httpConnectTimeout(), deps.httpReadTimeout());
    this.executor = deps.executor();
    this.apiKey = deps.apiKey();
    this.requestHeaders = deps.requestHeaders();
    this.sworTracker = deps.sworTracker();
    this.stabilityMonitor = deps.stabilityMonitor();
    this.ticker = deps.ticker();
    this.timeSource = deps.timeSource();
  }

  @Override
  public ListenableFuture<Void> upload(WorkloadSet workloadSet, EncodedPayload payload) {
    // Futures.submit throws if the executor is shut down; report that like any other failure.
    try {
      return Futures.submit(() -> uploadSynchronously(workloadSet, payload), executor);
    } catch (RejectedExecutionException e) {
      return immediateFailedFuture(e);
    }
  }

  /**
   * Uploads the encrypted payload blob via the ByteStream protocol and then submits the aggregation
   * result to complete the session.
   *
   * <p>Throws {@link UploadAbortedException} when the payload is known not to have been stored: the
   * server answered the blob upload with {@code ABORTED} (HTTP 409) or refused authorization with
   * {@code PERMISSION_DENIED} (HTTP 403, such as an expired session token), or writing the SWOR
   * record failed before the upload started. Any other blob upload failure sends a best-effort
   * abort, as C++ does, and rethrows the original exception: the blob may still have been stored,
   * and the server does not discard it on abort. A failed submit sends no abort and rethrows too,
   * since the blob is already stored by then.
   */
  private Void uploadSynchronously(WorkloadSet workloadSet, EncodedPayload payload)
      throws IOException, InterruptedException {
    UploadTokens tokens = workloadSet.uploadTokens();
    // Recorded before the upload starts, like EVENT_KIND_RESULT_UPLOAD_STARTED in the C++ client,
    // so that the same data is never uploaded twice.
    SworRecord sworRecord = sworTracker.recordUpload(workloadSet.name());
    logger.atFine().log("Starting payload upload for workload set: %s", workloadSet.name());
    logDiagnosticEvent(DiagnosticEvent.builder(Kind.UPLOAD_STARTED), workloadSet);
    NetworkPhase upload = NetworkPhase.start(httpClient, ticker);
    try {
      uploadBlob(tokens, payload);
    } catch (IOException | RuntimeException e) {
      // No warning is logged here: the failure is rethrown (possibly wrapped) and the caller logs
      // it.
      int statusCode =
          e instanceof IOException ioException
              ? FcpHttpException.toRpcStatusCode(ioException)
              : Code.INTERNAL_VALUE;
      if (statusCode == Code.ABORTED_VALUE || statusCode == Code.PERMISSION_DENIED_VALUE) {
        // The upload server rejected the blob (HTTP 409) or refused authorization (HTTP 403, e.g.
        // an expired upload session token), so it was not stored and there is nothing to abort
        // (an abort would be refused with the same token).
        sworTracker.restoreUpload(sworRecord);
        UploadAbortedException aborted =
            new UploadAbortedException(
                "Server aborted or refused the upload for workload set " + workloadSet.name(), e);
        logDiagnosticEvent(upload.event(Kind.UPLOAD_ABORTED).setError(aborted), workloadSet);
        throw aborted;
      }
      // Same as UploadResult in C++. The abort only tells the server why we stopped; it does not
      // remove a blob that reached storage, so whether the payload was stored stays unknown.
      abort(workloadSet, statusCode);
      logDiagnosticEvent(upload.event(Kind.UPLOAD_ERROR).setError(e), workloadSet);
      throw e;
    }
    // A failure after a successful blob upload sends no abort, as in the C++ client
    // (UploadResult/SubmitAggregationResult for confidential aggregation in
    // http_federated_protocol.cc). The blob is already stored, so the outcome is unknown.
    try {
      submit(tokens);
    } catch (IOException | RuntimeException e) {
      logDiagnosticEvent(upload.event(Kind.UPLOAD_ERROR).setError(e), workloadSet);
      throw e;
    }
    logger.atFine().log("Payload upload completed for workload set: %s", workloadSet.name());
    logDiagnosticEvent(upload.event(Kind.UPLOAD_COMPLETED), workloadSet);
    return null;
  }

  private void logDiagnosticEvent(DiagnosticEvent.Builder event, WorkloadSet workloadSet) {
    stabilityMonitor.logDiagnosticEvent(
        event
            .setTimestamp(timeSource.now())
            .setLogSourceName(workloadSet.logSourceName())
            .setFetchId(workloadSet.fetchId())
            .setWorkloadSetName(workloadSet.name())
            .build());
  }

  /**
   * Uploads the encrypted envelope via raw ByteStream POST (see {@code
   * UploadDataViaByteStreamProtocol} and {@code CreateByteStreamUploadUriSuffix} in {@code
   * fcp/client/http/http_federated_protocol.cc}).
   */
  private void uploadBlob(UploadTokens tokens, EncodedPayload payload) throws IOException {
    String uploadUrl =
        String.format(
            "%s/upload/v1/media/%s?upload_protocol=raw",
            tokens.uploadUri(), UriEncoding.encodePathSegments(tokens.resourceName()));
    Map<String, String> headers = headers(tokens.uploadExtraHeaders());
    // The C++ client sends no Content-Type here; FcpHttpClient would default the body to
    // application/x-protobuf, and the encrypted envelope is not a proto.
    headers.put("Content-Type", "application/octet-stream");

    logger.atFine().log("Uploading blob to %s", UrlRedaction.redact(uploadUrl));
    var unused = httpClient.post(uploadUrl, payload.encryptedEnvelope(), headers);
  }

  /**
   * Notifies the server that the blob upload is complete (see {@code SubmitAggregationResult} and
   * {@code CreateSubmitConfidentialAggregationResultUriSuffix} in {@code
   * fcp/client/http/http_federated_protocol.cc}).
   */
  private void submit(UploadTokens tokens) throws IOException {
    String submitUrl =
        String.format(
            "%s/v1/confidentialaggregations/%s/clients/%s:submit?%%24alt=proto",
            tokens.submitUriPrefix(),
            UriEncoding.encodePathSegment(tokens.aggregationId()),
            UriEncoding.encodePathSegment(tokens.clientToken()));

    SubmitConfidentialAggregationResultRequest request =
        SubmitConfidentialAggregationResultRequest.newBuilder()
            .setResourceName(tokens.resourceName())
            .build();

    logger.atFine().log("Submitting aggregation result to %s", UrlRedaction.redact(submitUrl));
    var unused =
        httpClient.post(submitUrl, request.toByteString(), headers(tokens.submitExtraHeaders()));
  }

  /**
   * Sends a best-effort abort after a failed blob upload. Never throws: a failed abort is logged,
   * with the diag code C++ logs, so that the caller can still report the original blob upload
   * failure.
   */
  private void abort(WorkloadSet workloadSet, int statusCode) {
    UploadTokens tokens = workloadSet.uploadTokens();
    try {
      AggregationAbortSender.sendAbort(
          httpClient,
          tokens.submitUriPrefix(),
          headers(tokens.submitExtraHeaders()),
          tokens.aggregationId(),
          tokens.clientToken(),
          statusCode);
    } catch (IOException | RuntimeException e) {
      logger.atWarning().withCause(e).log("Failed to abort aggregation after upload failure");
      stabilityMonitor.logDiagCode(
          ProdDiagCode.HTTP_CANCELLATION_OR_ABORT_REQUEST_FAILED_VALUE, workloadSet.fetchId());
    }
  }

  /**
   * Returns a mutable map with the caller-supplied request headers, the API key, and the
   * server-provided forwarding headers from the tokens; later entries win, so the API key overrides
   * a caller header of the same name and forwarding headers override both, as in the fetcher and
   * the C++ client.
   */
  private Map<String, String> headers(Map<String, String> tokenHeaders) {
    Map<String, String> headers = new HashMap<>(requestHeaders);
    headers.put("x-goog-api-key", apiKey);
    headers.putAll(tokenHeaders);
    return headers;
  }
}
