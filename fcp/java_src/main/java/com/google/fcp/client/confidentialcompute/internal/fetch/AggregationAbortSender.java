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
package com.google.fcp.client.confidentialcompute.internal.fetch;

import com.google.common.flogger.GoogleLogger;
import com.google.fcp.client.confidentialcompute.internal.http.FcpHttpClient;
import com.google.fcp.client.confidentialcompute.internal.http.UriEncoding;
import com.google.fcp.client.confidentialcompute.internal.http.UrlRedaction;
import com.google.internal.federatedcompute.v1.AbortConfidentialAggregationRequest;
import com.google.rpc.Status;
import java.io.IOException;
import java.util.Map;

/**
 * Sends {@code AbortConfidentialAggregation} requests, matching {@code
 * HttpFederatedProtocol::AbortAggregationInner} in {@code
 * fcp/client/http/http_federated_protocol.cc}.
 */
public final class AggregationAbortSender {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  public static final String ABORT_ERROR_MESSAGE =
      "Upload data via Confidential aggregation failed.";

  private AggregationAbortSender() {}

  /**
   * Sends an {@link AbortConfidentialAggregationRequest} with {@code rpcStatusCode} and {@link
   * #ABORT_ERROR_MESSAGE} to {@code submitUriPrefix}.
   */
  public static void sendAbort(
      FcpHttpClient httpClient,
      String submitUriPrefix,
      Map<String, String> headers,
      String aggregationId,
      String clientToken,
      int rpcStatusCode)
      throws IOException {
    String abortUrl =
        String.format(
            "%s/v1/confidentialaggregations/%s/clients/%s:abort?%%24alt=proto",
            submitUriPrefix,
            UriEncoding.encodePathSegment(aggregationId),
            UriEncoding.encodePathSegment(clientToken));
    logger.atFine().log(
        "Sending confidential aggregation abort to %s", UrlRedaction.redact(abortUrl));
    AbortConfidentialAggregationRequest request =
        AbortConfidentialAggregationRequest.newBuilder()
            .setStatus(Status.newBuilder().setCode(rpcStatusCode).setMessage(ABORT_ERROR_MESSAGE))
            .build();
    var unused = httpClient.post(abortUrl, request.toByteString(), headers);
  }
}
