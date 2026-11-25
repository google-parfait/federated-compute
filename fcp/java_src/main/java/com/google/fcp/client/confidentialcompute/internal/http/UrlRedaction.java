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

/** Removes bearer material from request URLs before they reach logs or exception messages. */
public final class UrlRedaction {

  private static final String REDACTED = "<redacted>";
  private static final String CLIENTS_MARKER = "/clients/";
  private static final String MEDIA_MARKER = "/media/";

  private UrlRedaction() {}

  /**
   * Returns {@code url} with bearer material removed, for use in logs and exception messages.
   *
   * <p>The query string is dropped (resource URIs are signed), the path segment after {@code
   * /clients/} (authorization or client token) is replaced, and everything after {@code /media/}
   * (the Scotty resource name, itself a capability) is replaced. Session and aggregation IDs are
   * kept because they are needed to correlate with server logs and are not credentials.
   */
  public static String redact(String url) {
    int query = url.indexOf('?');
    String redacted = query >= 0 ? url.substring(0, query) : url;
    redacted = redactSegmentAfter(redacted, CLIENTS_MARKER);
    int media = redacted.indexOf(MEDIA_MARKER);
    if (media >= 0) {
      redacted = redacted.substring(0, media + MEDIA_MARKER.length()) + REDACTED;
    }
    return redacted;
  }

  /**
   * Replaces the path segment following {@code marker}. The segment ends at the next {@code /} or
   * {@code :} (REST custom method separator); tokens are URL-encoded so neither occurs inside them.
   */
  private static String redactSegmentAfter(String url, String marker) {
    int start = url.indexOf(marker);
    if (start < 0) {
      return url;
    }
    int segmentStart = start + marker.length();
    int end = segmentStart;
    while (end < url.length() && url.charAt(end) != '/' && url.charAt(end) != ':') {
      end++;
    }
    return url.substring(0, segmentStart) + REDACTED + url.substring(end);
  }
}
