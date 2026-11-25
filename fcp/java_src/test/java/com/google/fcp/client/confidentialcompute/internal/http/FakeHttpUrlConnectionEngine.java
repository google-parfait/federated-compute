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

import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.fcp.client.confidentialcompute.http.HttpUrlConnectionEngine;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Function;

/**
 * A fake {@link HttpUrlConnectionEngine} for testing.
 *
 * <p>Responses and failures are configured by URL substring, where the longest matching substring
 * wins. If the status code is {@code >= 400}, the body is served from {@code getErrorStream()} and
 * {@code getInputStream()} throws {@link IOException}.
 */
public final class FakeHttpUrlConnectionEngine implements HttpUrlConnectionEngine {

  /** A canned response: status, body and response headers. */
  public static final class FakeResponse {
    private final int status;
    private final byte[] body;
    private final ImmutableMap<String, String> headers;

    private FakeResponse(int status, byte[] body, ImmutableMap<String, String> headers) {
      this.status = status;
      this.body = body;
      this.headers = headers;
    }

    /** Creates a response with the given status and body. */
    public static FakeResponse of(int status, byte[] body) {
      return new FakeResponse(status, body, ImmutableMap.of());
    }

    /** Returns a new response adding the given header. */
    public FakeResponse withHeader(String name, String value) {
      ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
      builder.putAll(headers).put(name, value);
      return new FakeResponse(status, body, builder.buildOrThrow());
    }
  }

  /** One call to {@link #openConnection}; method, headers and body are read from the connection. */
  public static final class RecordedRequest {
    private final FakeHttpUrlConnection connection;
    private final String threadName;

    RecordedRequest(FakeHttpUrlConnection connection, String threadName) {
      this.connection = connection;
      this.threadName = threadName;
    }

    /** Returns the requested URL. */
    public String url() {
      return connection.getURL().toString();
    }

    /** Returns the request method. */
    public String method() {
      return connection.getRequestMethod();
    }

    /** Returns the request headers as set by the caller. */
    public ImmutableMap<String, String> headers() {
      return ImmutableMap.copyOf(connection.getRequestPropertiesMap());
    }

    /** Returns the request body written to the output stream. */
    public byte[] body() {
      return connection.getWrittenBody();
    }

    /** Returns the name of the thread that called openConnection. */
    public String threadName() {
      return threadName;
    }

    /** Returns the underlying connection. */
    public FakeHttpUrlConnection connection() {
      return connection;
    }
  }

  /** The connection handed to the code under test; serves the configured {@link FakeResponse}. */
  public static final class FakeHttpUrlConnection extends HttpURLConnection {
    private final Function<RecordedRequest, FakeResponse> responseFn;
    private RecordedRequest request;
    private final Map<String, String> requestProperties = new ConcurrentHashMap<>();
    private final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

    private boolean disconnected = false;
    private FakeResponse evaluatedResponse = null;

    FakeHttpUrlConnection(URL url, Function<RecordedRequest, FakeResponse> responseFn) {
      super(url);
      this.responseFn = responseFn;
    }

    /** Returns whether {@link #disconnect()} was called. */
    public boolean wasDisconnected() {
      return disconnected;
    }

    void setRecordedRequest(RecordedRequest request) {
      this.request = request;
    }

    private synchronized FakeResponse getResponse() {
      if (evaluatedResponse == null) {
        evaluatedResponse = responseFn.apply(request);
      }
      return evaluatedResponse;
    }

    @Override
    public void connect() {}

    @Override
    public void disconnect() {
      disconnected = true;
    }

    @Override
    public boolean usingProxy() {
      return false;
    }

    @Override
    public void setRequestProperty(String key, String value) {
      requestProperties.put(key, value);
      super.setRequestProperty(key, value);
    }

    Map<String, String> getRequestPropertiesMap() {
      return requestProperties;
    }

    byte[] getWrittenBody() {
      return outputStream.toByteArray();
    }

    @Override
    public OutputStream getOutputStream() {
      return outputStream;
    }

    @Override
    public int getResponseCode() {
      return getResponse().status;
    }

    @Override
    public InputStream getInputStream() throws IOException {
      FakeResponse resp = getResponse();
      if (resp.status >= 400) {
        throw new IOException("HTTP error: " + resp.status);
      }
      return new ByteArrayInputStream(resp.body);
    }

    @Override
    public InputStream getErrorStream() {
      FakeResponse resp = getResponse();
      if (resp.status >= 400) {
        return new ByteArrayInputStream(resp.body);
      }
      return null;
    }

    @Override
    public String getHeaderField(String name) {
      for (Map.Entry<String, String> entry : getResponse().headers.entrySet()) {
        if (Ascii.equalsIgnoreCase(entry.getKey(), name)) {
          return entry.getValue();
        }
      }
      return null;
    }

    @Override
    public String getContentType() {
      return getHeaderField("Content-Type");
    }
  }

  private final Map<String, Exception> failures = new ConcurrentHashMap<>();
  private final Map<String, Function<RecordedRequest, FakeResponse>> responses =
      new ConcurrentHashMap<>();
  private FakeResponse defaultResponse = FakeResponse.of(404, new byte[0]);
  private final CopyOnWriteArrayList<RecordedRequest> recordedRequests =
      new CopyOnWriteArrayList<>();

  /** Configures a response for the given URL substring. */
  public void respondTo(String substring, int status, byte[] body) {
    responses.put(substring, req -> FakeResponse.of(status, body));
  }

  /** Configures a response with a content type for the given URL substring. */
  public void respondTo(String substring, int status, byte[] body, String contentType) {
    responses.put(
        substring, req -> FakeResponse.of(status, body).withHeader("Content-Type", contentType));
  }

  /** Configures a dynamic response function for the given URL substring. */
  public void respondTo(String substring, Function<RecordedRequest, FakeResponse> function) {
    responses.put(substring, function);
  }

  /** Sets the default response when no substring matches. */
  public void setDefaultResponse(int status, byte[] body) {
    this.defaultResponse = FakeResponse.of(status, body);
  }

  /** Configures a failure to be thrown from openConnection for the given URL substring. */
  public void failWith(String substring, IOException exception) {
    failures.put(substring, exception);
  }

  /** Configures a failure to be thrown from openConnection for the given URL substring. */
  public void failWith(String substring, RuntimeException exception) {
    failures.put(substring, exception);
  }

  @Override
  public HttpURLConnection openConnection(URL url) throws IOException {
    String urlStr = url.toString();
    String bestMatch = null;

    for (String substring : failures.keySet()) {
      if (urlStr.contains(substring)) {
        if (bestMatch == null || substring.length() > bestMatch.length()) {
          bestMatch = substring;
        }
      }
    }
    for (String substring : responses.keySet()) {
      if (urlStr.contains(substring)) {
        if (bestMatch == null || substring.length() > bestMatch.length()) {
          bestMatch = substring;
        }
      }
    }

    if (bestMatch != null && failures.containsKey(bestMatch)) {
      Exception e = failures.get(bestMatch);
      if (e instanceof IOException ioException) {
        throw ioException;
      }
      if (e instanceof RuntimeException runtimeException) {
        throw runtimeException;
      }
    }

    Function<RecordedRequest, FakeResponse> responseFn = req -> defaultResponse;
    if (bestMatch != null && responses.containsKey(bestMatch)) {
      responseFn = responses.get(bestMatch);
    }

    FakeHttpUrlConnection conn = new FakeHttpUrlConnection(url, responseFn);
    RecordedRequest req = new RecordedRequest(conn, Thread.currentThread().getName());
    conn.setRecordedRequest(req);
    recordedRequests.add(req);
    return conn;
  }

  /** Returns all recorded requests. */
  public ImmutableList<RecordedRequest> requests() {
    return ImmutableList.copyOf(recordedRequests);
  }

  /** Returns recorded requests whose URL contains the given substring. */
  public ImmutableList<RecordedRequest> requestsContaining(String substring) {
    ImmutableList.Builder<RecordedRequest> builder = ImmutableList.builder();
    for (RecordedRequest req : recordedRequests) {
      if (req.url().contains(substring)) {
        builder.add(req);
      }
    }
    return builder.build();
  }

  /**
   * Returns the only recorded request whose URL contains the given substring, asserting there is
   * exactly one.
   */
  public RecordedRequest onlyRequestContaining(String substring) {
    ImmutableList<RecordedRequest> matches = requestsContaining(substring);
    if (matches.size() != 1) {
      throw new AssertionError(
          "Expected exactly one request containing '"
              + substring
              + "', but found "
              + matches.size());
    }
    return matches.get(0);
  }

  /** Returns whether any recorded request's URL contains the given substring. */
  public boolean anyRequestContains(String substring) {
    for (RecordedRequest req : recordedRequests) {
      if (req.url().contains(substring)) {
        return true;
      }
    }
    return false;
  }

  /** Returns the requested URLs in order. */
  public ImmutableList<String> requestUrls() {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    for (RecordedRequest req : recordedRequests) {
      builder.add(req.url());
    }
    return builder.build();
  }

  /** Returns the last recorded request. */
  public RecordedRequest lastRequest() {
    if (recordedRequests.isEmpty()) {
      throw new AssertionError("No requests recorded");
    }
    return recordedRequests.get(recordedRequests.size() - 1);
  }
}
