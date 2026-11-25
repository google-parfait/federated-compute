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
package com.google.fcp.client.confidentialcompute.internal.http;

import com.google.common.base.Ticker;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent;
import java.time.Duration;

/**
 * The time and bytes an {@link FcpHttpClient} has spent since a phase started, for diagnostic
 * events.
 */
public final class NetworkPhase {
  private final FcpHttpClient httpClient;
  private final Ticker ticker;
  private final long startNanos;
  private final long bytesUploadedAtStart;
  private final long bytesDownloadedAtStart;

  private NetworkPhase(FcpHttpClient httpClient, Ticker ticker) {
    this.httpClient = httpClient;
    this.ticker = ticker;
    this.startNanos = ticker.read();
    this.bytesUploadedAtStart = httpClient.bytesUploaded();
    this.bytesDownloadedAtStart = httpClient.bytesDownloaded();
  }

  /** Starts a phase now. */
  public static NetworkPhase start(FcpHttpClient httpClient, Ticker ticker) {
    return new NetworkPhase(httpClient, ticker);
  }

  /** Returns a builder for a {@code kind} event with the time and bytes since the start. */
  public DiagnosticEvent.Builder event(DiagnosticEvent.Kind kind) {
    return DiagnosticEvent.builder(kind)
        .setDuration(Duration.ofNanos(ticker.read() - startNanos))
        .setBytesUploaded(httpClient.bytesUploaded() - bytesUploadedAtStart)
        .setBytesDownloaded(httpClient.bytesDownloaded() - bytesDownloadedAtStart);
  }
}
