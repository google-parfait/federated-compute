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

import com.google.common.net.PercentEscaper;

/**
 * RFC 3986 percent-encoding of URI path segments.
 *
 * <p>Mirrors {@code EncodeUriSinglePathSegment} and {@code EncodeUriMultiplePathSegments} from
 * {@code fcp/client/http/http_client_util.cc}: every character except the unreserved set
 * [A-Za-z0-9-._~] is encoded as UTF-8 with uppercase hex. Guava's {@code
 * UrlEscapers.urlPathSegmentEscaper()} is not used because it leaves sub-delimiters such as {@code
 * :} and {@code +} unencoded.
 *
 * <p>Deliberate difference: non-ASCII input is UTF-8 percent-encoded rather than rejected.
 *
 * <p>Public because the fetch and upload packages both build protocol URLs with it.
 */
public final class UriEncoding {

  // Letters and digits are always safe for PercentEscaper.
  private static final PercentEscaper PATH_SEGMENT_ESCAPER =
      new PercentEscaper("-._~", /* plusForSpace= */ false);
  private static final PercentEscaper PATH_SEGMENTS_ESCAPER =
      new PercentEscaper("-._~/", /* plusForSpace= */ false);

  private UriEncoding() {}

  /** Percent-encodes {@code input} as a single path segment; '/' is encoded. */
  public static String encodePathSegment(String input) {
    return PATH_SEGMENT_ESCAPER.escape(input);
  }

  /** Same as {@link #encodePathSegment(String)}, but leaves '/' intact. */
  public static String encodePathSegments(String input) {
    return PATH_SEGMENTS_ESCAPER.escape(input);
  }
}
