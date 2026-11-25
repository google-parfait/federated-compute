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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Strings.nullToEmpty;
import static java.nio.charset.StandardCharsets.UTF_8;

import androidx.annotation.Nullable;
import androidx.annotation.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.fcp.client.confidentialcompute.http.HttpUrlConnectionEngine;
import com.google.protobuf.ByteString;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.CookieHandler;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.time.Duration;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.GZIPInputStream;

/**
 * A lightweight HTTP client that uses {@link HttpUrlConnectionEngine} to perform requests.
 *
 * <p>Because {@link HttpURLConnection} is a low-level per-request object requiring manual stream
 * reading, error stream draining, and timeout configuration on each connection, this class
 * centralizes that boilerplate behind simple {@link #post} and {@link #get} helpers.
 *
 * <p>Refuses to run while a system-wide {@link CookieHandler} is installed (cookies must never be
 * sent or stored), so callers see it on the same IOException path as transport failures.
 *
 * <p>Behaviour follows the C++ {@code HttpClient} contract and {@code
 * in_memory_request_response.cc}: HTTPS only (plus {@code http://localhost} for tests), no caching,
 * redirects followed, a body whose Content-Type ends with {@code +gzip} is gunzipped, no {@code
 * Accept} header is sent, and a connection whose response was read to EOF is left open for reuse.
 *
 * <p>Deliberate differences from the C++ client:
 *
 * <ul>
 *   <li>Request bodies are not gzip-compressed (the protocol requests are small and the upload
 *       payload is already encrypted).
 *   <li>HTTP-level {@code Content-Encoding: gzip} is left to the platform {@link
 *       HttpURLConnection}.
 *   <li>There is no built-in retry loop or resource cache (PrivateLogger uploads are themselves
 *       periodic).
 *   <li>Connect and read timeouts are set on each connection, as passed to the constructor.
 *   <li>Response bodies are capped at 16 MiB (they are buffered in the app heap, and the cap also
 *       bounds gzip inflation) while {@code Content-Length} framing is enforced by {@link
 *       HttpURLConnection} rather than re-checked here.
 * </ul>
 *
 * <p>Counts the request and response body bytes it sends and receives, for diagnostic events; see
 * {@link #bytesUploaded} and {@link #bytesDownloaded}.
 */
public final class FcpHttpClient {

  private static final Duration MIN_TIMEOUT = Duration.ofMillis(1);
  private static final Duration MAX_TIMEOUT = Duration.ofMillis(Integer.MAX_VALUE);
  private static final int MAX_ERROR_BODY_CHARS = 512;
  private static final int MAX_ERROR_BODY_BYTES = 4 * MAX_ERROR_BODY_CHARS;
  private static final String CLIENT_DECODED_GZIP_SUFFIX = "+gzip";

  private static final int MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

  /**
   * The request could not be built (malformed or non-HTTPS URL); a programming or configuration
   * error, never retryable.
   */
  static final class InvalidRequestException extends IOException {
    InvalidRequestException(String message) {
      super(message);
    }

    InvalidRequestException(String message, Throwable cause) {
      super(message, cause);
    }
  }

  private final HttpUrlConnectionEngine httpEngine;
  private final int connectTimeoutMs;
  private final int readTimeoutMs;
  private final int maxResponseBytes;
  private final AtomicLong bytesUploaded = new AtomicLong();
  private final AtomicLong bytesDownloaded = new AtomicLong();

  /** Creates a client that opens connections through {@code httpEngine} with the given timeouts. */
  public FcpHttpClient(
      HttpUrlConnectionEngine httpEngine, Duration connectTimeout, Duration readTimeout) {
    this(httpEngine, connectTimeout, readTimeout, MAX_RESPONSE_BYTES);
  }

  @VisibleForTesting
  FcpHttpClient(
      HttpUrlConnectionEngine httpEngine,
      Duration connectTimeout,
      Duration readTimeout,
      int maxResponseBytes) {
    this.httpEngine = checkNotNull(httpEngine, "httpEngine must not be null");
    this.connectTimeoutMs = toTimeoutMillis(connectTimeout, "connectTimeout");
    this.readTimeoutMs = toTimeoutMillis(readTimeout, "readTimeout");
    this.maxResponseBytes = maxResponseBytes;
  }

  private static int toTimeoutMillis(Duration timeout, String name) {
    checkNotNull(timeout, "%s must not be null", name);
    // HttpURLConnection reads a 0 ms timeout as no timeout.
    checkArgument(
        timeout.compareTo(MIN_TIMEOUT) >= 0, "%s must be at least 1 ms: %s", name, timeout);
    return timeout.compareTo(MAX_TIMEOUT) >= 0 ? Integer.MAX_VALUE : (int) timeout.toMillis();
  }

  /** Sends a protocol POST and returns the 2xx body; non-2xx is an FcpHttpException. */
  public byte[] post(String urlString, ByteString requestBody, Map<String, String> extraHeaders)
      throws IOException {
    return execute("POST", urlString, requestBody, extraHeaders);
  }

  /** Sends a protocol GET and returns the 2xx body; non-2xx is an FcpHttpException. */
  public byte[] get(String urlString, Map<String, String> extraHeaders) throws IOException {
    return execute("GET", urlString, null, extraHeaders);
  }

  /** Fetches a {@code Resource} URI without extra headers. */
  public byte[] getResource(String urlString) throws IOException {
    return get(urlString, /* extraHeaders= */ ImmutableMap.of());
  }

  /**
   * Request body bytes written since this client was created. The counter never resets: callers
   * measure one phase as the difference between two readings, which is what {@link NetworkPhase}
   * does.
   */
  public long bytesUploaded() {
    return bytesUploaded.get();
  }

  /**
   * Response body bytes read since this client was created, error bodies included, counted as
   * received (before gzip decoding). Like {@link #bytesUploaded}, it never resets.
   */
  public long bytesDownloaded() {
    return bytesDownloaded.get();
  }

  /** Inflates a gzip body; also used by the fetcher for gzip-compressed inline resources. */
  public byte[] gunzip(byte[] compressed) throws IOException {
    try (GZIPInputStream gzipIn = new GZIPInputStream(new ByteArrayInputStream(compressed))) {
      return readStream(gzipIn, -1, maxResponseBytes);
    }
  }

  /**
   * Parses {@code urlString}, allowing only HTTPS, plus {@code http://localhost} for local test
   * servers, as {@code InMemoryHttpRequest::Create} in {@code
   * fcp/client/http/in_memory_request_response.cc} does.
   *
   * <p>The parse exceptions' messages contain the full, unredacted URL, so the cause attached to
   * the thrown exception is a copy of the same type built from the redacted URL (see {@link
   * #malformedUrl}).
   */
  private static URL toHttpsUrl(String urlString) throws IOException {
    URI uri;
    try {
      uri = new URI(urlString);
    } catch (URISyntaxException e) {
      String redactedUrl = UrlRedaction.redact(urlString);
      // The index still refers to the original URL.
      throw malformedUrl(
          redactedUrl, new URISyntaxException(redactedUrl, e.getReason(), e.getIndex()), e);
    }
    String scheme = nullToEmpty(uri.getScheme());
    boolean allowed =
        Ascii.equalsIgnoreCase(scheme, "https")
            || (Ascii.equalsIgnoreCase(scheme, "http")
                && Ascii.equalsIgnoreCase(nullToEmpty(uri.getHost()), "localhost"));
    if (!allowed) {
      throw new InvalidRequestException(
          "Non-HTTPS URL rejected: " + UrlRedaction.redact(urlString));
    }
    try {
      return uri.toURL();
    } catch (MalformedURLException e) {
      String redactedUrl = UrlRedaction.redact(urlString);
      throw malformedUrl(redactedUrl, new MalformedURLException(redactedUrl), e);
    }
  }

  /**
   * Returns the exception thrown for an unparseable URL, with {@code redactedCause} as its cause.
   * {@code redactedCause} is a copy of {@code cause} without the raw URL; it takes over {@code
   * cause}'s stack trace.
   */
  private static InvalidRequestException malformedUrl(
      String redactedUrl, Exception redactedCause, Exception cause) {
    redactedCause.setStackTrace(cause.getStackTrace());
    return new InvalidRequestException("Malformed request URL: " + redactedUrl, redactedCause);
  }

  // Per the C++ HttpClient contract (fcp/client/http/http_client.h) the client must neither send
  // nor store cookies: a cookie is a server-assigned persistent identifier that would link a
  // device's check-ins and uploads across rounds. HttpURLConnection consults the process-wide
  // CookieHandler on every request and has no per-connection opt-out, so refuse to run while one
  // is installed, as the existing FCP Java HTTP client (HttpRequestHandleImpl) does, so callers see
  // it on the same IOException path as transport failures.
  private static void checkNoCookieHandler() throws InvalidRequestException {
    if (CookieHandler.getDefault() != null) {
      throw new InvalidRequestException(
          "A system-wide java.net.CookieHandler is installed; the confidential compute client must"
              + " not send or store cookies (see fcp/client/http/http_client.h)");
    }
  }

  private byte[] execute(
      String method,
      String urlString,
      @Nullable ByteString requestBody,
      Map<String, String> extraHeaders)
      throws IOException {
    checkNoCookieHandler();
    URL url = toHttpsUrl(urlString);
    HttpURLConnection connection = httpEngine.openConnection(url);
    boolean responseConsumed = false;

    try {
      connection.setRequestMethod(method);
      boolean hasBody = requestBody != null && !requestBody.isEmpty();
      if (hasBody) {
        connection.setDoOutput(true);
        connection.setFixedLengthStreamingMode(requestBody.size());
      }
      connection.setConnectTimeout(connectTimeoutMs);
      connection.setReadTimeout(readTimeoutMs);
      // Per the C++ HttpClient contract (fcp/client/http/http_client.h): no caching, and follow
      // redirects.
      connection.setUseCaches(false);
      connection.setInstanceFollowRedirects(true);

      // 1. Add default headers
      if (hasBody) {
        connection.setRequestProperty("Content-Type", "application/x-protobuf");
      }

      // 2. Add extra headers
      for (Map.Entry<String, String> entry : extraHeaders.entrySet()) {
        connection.setRequestProperty(entry.getKey(), entry.getValue());
      }

      // 3. Write request body
      if (hasBody) {
        try (OutputStream os = connection.getOutputStream();
            InputStream is = requestBody.newInput()) {
          // Not ByteStreams.copy, which always allocates an 8 KiB buffer: most requests are small
          // RPC bodies, so the buffer is sized to the body and capped at 8 KiB for large uploads.
          byte[] buffer = new byte[Math.min(requestBody.size(), 8192)];
          int read;
          while ((read = is.read(buffer)) != -1) {
            os.write(buffer, 0, read);
          }
        }
        bytesUploaded.addAndGet(requestBody.size());
      }

      // 4. Read response body
      int responseCode = connection.getResponseCode();
      if (responseCode >= 200 && responseCode < 300) {
        byte[] body;
        try (InputStream is = connection.getInputStream()) {
          body = readStream(is, connection.getContentLength(), maxResponseBytes);
        }
        bytesDownloaded.addAndGet(body.length);
        responseConsumed = true;
        boolean gzipContentType =
            Ascii.toLowerCase(nullToEmpty(connection.getContentType()))
                .endsWith(CLIENT_DECODED_GZIP_SUFFIX);
        return gzipContentType ? gunzip(body) : body;
      } else {
        byte[] errorBytes = null;
        try (InputStream err = connection.getErrorStream()) {
          if (err != null) {
            errorBytes = readAtMost(err, MAX_ERROR_BODY_BYTES);
            bytesDownloaded.addAndGet(errorBytes.length);
          }
        } catch (IOException ignored) {
          // Ignored, fallback to throwing general HTTP connection exception below.
        }
        throw new FcpHttpException(
            responseCode,
            String.format(
                "HTTP %s to %s failed with code %d. Error: %s",
                method,
                UrlRedaction.redact(urlString),
                responseCode,
                errorBytes != null ? truncate(new String(errorBytes, UTF_8)) : "none"));
      }
    } finally {
      // As in HttpRequestHandleImpl: a connection whose response body was read to EOF goes back to
      // the platform's keep-alive pool; disconnect() would close the socket instead. Error
      // responses (whose body may be truncated by MAX_ERROR_BODY_BYTES) and failures are still
      // disconnected.
      if (!responseConsumed) {
        connection.disconnect();
      }
    }
  }

  private static String truncate(String s) {
    return s.length() <= MAX_ERROR_BODY_CHARS ? s : s.substring(0, MAX_ERROR_BODY_CHARS) + "...";
  }

  /**
   * Reads {@code is} to EOF. {@code contentLength} is only a size hint: Cronet's {@code
   * HttpURLConnection} decodes a {@code Content-Encoding} body but still reports the encoded {@code
   * Content-Length}, so the stream can be longer or shorter than the header says. The body is read
   * into an array of the hinted size, which is returned without a copy when the stream ends exactly
   * there.
   */
  private static byte[] readStream(InputStream is, int contentLength, int maxBytes)
      throws IOException {
    if (contentLength > maxBytes) {
      throw new IOException("Response body exceeded " + maxBytes + " bytes");
    }
    ByteArrayOutputStream bos;
    int totalBytes;
    if (contentLength >= 0) {
      byte[] body = new byte[contentLength];
      int read = ByteStreams.read(is, body, 0, contentLength);
      if (read < contentLength) {
        return Arrays.copyOf(body, read);
      }
      int next = is.read();
      if (next == -1) {
        return body;
      }
      totalBytes = contentLength + 1;
      if (totalBytes > maxBytes) {
        throw new IOException("Response body exceeded " + maxBytes + " bytes");
      }
      bos = new ByteArrayOutputStream(2 * totalBytes);
      bos.write(body);
      bos.write(next);
    } else {
      bos = new ByteArrayOutputStream();
      totalBytes = 0;
    }
    byte[] buffer = new byte[8192];
    int read;
    while ((read = is.read(buffer)) != -1) {
      totalBytes += read;
      if (totalBytes > maxBytes) {
        throw new IOException("Response body exceeded " + maxBytes + " bytes");
      }
      bos.write(buffer, 0, read);
    }
    return bos.toByteArray();
  }

  private static byte[] readAtMost(InputStream is, int maxBytes) throws IOException {
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    byte[] buffer = new byte[8192];
    int totalBytes = 0;
    int read;
    while (totalBytes < maxBytes
        && (read = is.read(buffer, 0, Math.min(buffer.length, maxBytes - totalBytes))) != -1) {
      totalBytes += read;
      bos.write(buffer, 0, read);
    }
    return bos.toByteArray();
  }
}
