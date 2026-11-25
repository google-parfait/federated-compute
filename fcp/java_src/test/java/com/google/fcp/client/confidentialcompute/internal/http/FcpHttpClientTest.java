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

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.fcp.client.confidentialcompute.http.HttpUrlConnectionEngine;
import com.google.fcp.client.confidentialcompute.internal.http.FcpHttpClient.InvalidRequestException;
import com.google.protobuf.ByteString;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.CookieHandler;
import java.net.CookieManager;
import java.net.HttpURLConnection;
import java.net.URISyntaxException;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPOutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class FcpHttpClientTest {

  private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(10);
  private static final Duration READ_TIMEOUT = Duration.ofSeconds(30);

  @Test
  public void execute_setsConfiguredTimeouts() throws Exception {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/test", 200, new byte[0]);
    FcpHttpClient client =
        new FcpHttpClient(engine, Duration.ofMillis(1234), Duration.ofMillis(5678));

    var unused = client.get("https://example.com/test", ImmutableMap.of());

    assertThat(engine.lastRequest().connection().getConnectTimeout()).isEqualTo(1234);
    assertThat(engine.lastRequest().connection().getReadTimeout()).isEqualTo(5678);
  }

  @Test
  public void constructor_negativeTimeout_throwsIllegalArgumentException() {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    assertThrows(
        IllegalArgumentException.class,
        () -> new FcpHttpClient(engine, Duration.ofMillis(-1), READ_TIMEOUT));
    assertThrows(
        IllegalArgumentException.class,
        () -> new FcpHttpClient(engine, CONNECT_TIMEOUT, Duration.ofMillis(-1)));
  }

  @Test
  public void constructor_timeoutUnderOneMillisecond_throwsIllegalArgumentException() {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    assertThrows(
        IllegalArgumentException.class,
        () -> new FcpHttpClient(engine, Duration.ZERO, READ_TIMEOUT));
    assertThrows(
        IllegalArgumentException.class,
        () -> new FcpHttpClient(engine, CONNECT_TIMEOUT, Duration.ofNanos(1)));
  }

  @Test
  public void execute_timeoutExceedingMaxInt_saturatesToMaxIntMillis() throws Exception {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/test", 200, new byte[0]);
    FcpHttpClient client =
        new FcpHttpClient(engine, Duration.ofDays(365), ChronoUnit.FOREVER.getDuration());

    var unused = client.get("https://example.com/test", ImmutableMap.of());

    assertThat(engine.lastRequest().connection().getConnectTimeout()).isEqualTo(Integer.MAX_VALUE);
    assertThat(engine.lastRequest().connection().getReadTimeout()).isEqualTo(Integer.MAX_VALUE);
  }

  @Test
  public void execute_get_setsNoCachingAndFollowRedirects() throws Exception {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/test", 200, new byte[0]);
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    var unused = client.get("https://localhost/test", ImmutableMap.of());

    assertThat(engine.lastRequest().connection().getUseCaches()).isFalse();
    assertThat(engine.lastRequest().connection().getInstanceFollowRedirects()).isTrue();
  }

  @Test
  public void execute_post_setsNoCachingAndFollowRedirects() throws Exception {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/test", 200, new byte[0]);
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    var unused =
        client.post("https://localhost/test", ByteString.copyFromUtf8("hello"), ImmutableMap.of());

    assertThat(engine.lastRequest().connection().getUseCaches()).isFalse();
    assertThat(engine.lastRequest().connection().getInstanceFollowRedirects()).isTrue();
    assertThat(engine.lastRequest().method()).isEqualTo("POST");
    assertThat(engine.lastRequest().connection().getDoOutput()).isTrue();
    assertThat(engine.lastRequest().body()).isEqualTo("hello".getBytes(UTF_8));
    assertThat(engine.lastRequest().headers())
        .containsEntry("Content-Type", "application/x-protobuf");
  }

  @Test
  public void execute_countsRequestAndResponseBodyBytes() throws Exception {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/ok", 200, "response".getBytes(UTF_8));
    engine.respondTo("/error", 500, "err".getBytes(UTF_8));
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    var unused =
        client.post("https://example.com/ok", ByteString.copyFromUtf8("hello"), ImmutableMap.of());
    assertThrows(
        FcpHttpException.class,
        () ->
            client.post(
                "https://example.com/error", ByteString.copyFromUtf8("ab"), ImmutableMap.of()));

    assertThat(client.bytesUploaded()).isEqualTo("hello".length() + "ab".length());
    assertThat(client.bytesDownloaded()).isEqualTo("response".length() + "err".length());
  }

  @Test
  public void execute_longErrorBody_truncatedTo512CharsInExceptionMessage() {
    // 100 KiB error body consisting of repeating 'A's
    byte[] largeError = new byte[100 * 1024];
    Arrays.fill(largeError, (byte) 'A');
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/test", 500, largeError);
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    FcpHttpException thrown =
        assertThrows(
            FcpHttpException.class, () -> client.get("https://localhost/test", ImmutableMap.of()));

    String expectedPrefix = new String(new char[512]).replace('\0', 'A');
    assertThat(thrown).hasMessageThat().contains("Error: " + expectedPrefix + "...");
    assertThat(thrown).hasMessageThat().doesNotContain("none");
  }

  @Test
  public void get_withGlobalCookieHandler_throwsIOExceptionWithoutOpeningConnection() {
    CookieHandler previous = CookieHandler.getDefault();
    CookieHandler.setDefault(new CookieManager());
    try {
      FcpHttpClient client =
          new FcpHttpClient(
              url -> {
                throw new AssertionError("openConnection must not be called");
              },
              CONNECT_TIMEOUT,
              READ_TIMEOUT);

      IOException thrown =
          assertThrows(
              IOException.class,
              () -> client.get("https://example.com/v1/test", ImmutableMap.of()));
      assertThat(thrown).isInstanceOf(InvalidRequestException.class);
      assertThat(thrown).hasMessageThat().contains("CookieHandler");
    } finally {
      CookieHandler.setDefault(previous);
    }
  }

  @Test
  public void get_withoutCookieHandler_succeeds() throws Exception {
    byte[] responseBytes = "response".getBytes(UTF_8);
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/test", 200, responseBytes);
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    byte[] response = client.get("https://localhost/test", ImmutableMap.of());

    assertThat(response).isEqualTo(responseBytes);
  }

  @Test
  public void execute_nonHttpsUrl_throwsIOExceptionWithRedactedUrl() {
    FcpHttpClient client = new FcpHttpClient(url -> null, CONNECT_TIMEOUT, READ_TIMEOUT);
    String url = "http://example.com/v1/x/clients/SECRET:abort?k=v";

    IOException thrown = assertThrows(IOException.class, () -> client.get(url, ImmutableMap.of()));

    assertThat(thrown).isInstanceOf(InvalidRequestException.class);
    assertThat(thrown).hasMessageThat().contains("Non-HTTPS URL rejected");
    assertThat(thrown).hasMessageThat().doesNotContain("SECRET");
    assertThat(thrown).hasMessageThat().doesNotContain("k=v");
  }

  @Test
  public void execute_httpLoopbackIp_isRejected() {
    FcpHttpClient client = new FcpHttpClient(url -> null, CONNECT_TIMEOUT, READ_TIMEOUT);

    IOException thrown =
        assertThrows(
            IOException.class,
            () -> client.get("http://127.0.0.1:8080/v1/test", ImmutableMap.of()));

    assertThat(thrown).isInstanceOf(InvalidRequestException.class);
    assertThat(thrown).hasMessageThat().contains("Non-HTTPS URL rejected");
  }

  @Test
  public void execute_httpLocalhost_isAccepted() throws Exception {
    AtomicBoolean reachedEngine = new AtomicBoolean(false);
    HttpUrlConnectionEngine engine =
        url -> {
          reachedEngine.set(true);
          return new HttpURLConnection(url) {
            @Override
            public void connect() {}

            @Override
            public void disconnect() {}

            @Override
            public boolean usingProxy() {
              return false;
            }

            @Override
            public int getResponseCode() {
              return 200;
            }

            @Override
            public InputStream getInputStream() {
              return new ByteArrayInputStream(new byte[0]);
            }
          };
        };
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    byte[] response = client.get("http://localhost:8080/v1/test", ImmutableMap.of());

    assertThat(reachedEngine.get()).isTrue();
    assertThat(response).isEmpty();
  }

  @Test
  public void execute_malformedUrlWithToken_chainsRedactedCause() {
    FcpHttpClient client = new FcpHttpClient(url -> null, CONNECT_TIMEOUT, READ_TIMEOUT);
    String url = "https://exa mple.com/clients/SECRET";

    IOException thrown = assertThrows(IOException.class, () -> client.get(url, ImmutableMap.of()));

    assertThat(thrown).isInstanceOf(InvalidRequestException.class);
    assertThat(thrown).hasMessageThat().contains("Malformed request URL");
    assertThat(thrown).hasMessageThat().doesNotContain("SECRET");
    assertThat(thrown).hasCauseThat().isInstanceOf(URISyntaxException.class);
    assertThat(thrown).hasCauseThat().hasMessageThat().contains("Illegal character");
    assertThat(thrown).hasCauseThat().hasMessageThat().doesNotContain("SECRET");
    assertThat(thrown.getCause().getStackTrace()).isNotEmpty();
    assertThat(thrown.getCause().getStackTrace()[0].getClassName()).startsWith("java.net.URI");
  }

  @Test
  public void execute_responseExceedingCap_throwsIOException() {
    byte[] data = new byte[20];
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/test", 200, data);
    FcpHttpClient client =
        new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT, /* maxResponseBytes= */ 10);

    IOException thrown =
        assertThrows(
            IOException.class, () -> client.get("https://localhost/test", ImmutableMap.of()));

    assertThat(thrown).hasMessageThat().contains("Response body exceeded 10 bytes");
  }

  @Test
  public void execute_extraHeaders_areSetOnConnection() throws Exception {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/test", 200, new byte[0]);
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    var unused =
        client.get(
            "https://localhost/test",
            ImmutableMap.of("x-goog-api-key", "key", "x-custom", "value"));

    assertThat(engine.lastRequest().headers()).containsEntry("x-goog-api-key", "key");
    assertThat(engine.lastRequest().headers()).containsEntry("x-custom", "value");
    assertThat(engine.lastRequest().headers()).doesNotContainKey("Accept");
  }

  @Test
  public void post_emptyBody_sendsNoBodyAndNoContentType() throws Exception {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/test", 200, new byte[0]);
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    byte[] response = client.post("https://localhost/test", ByteString.EMPTY, ImmutableMap.of());

    assertThat(response).isEmpty();
    assertThat(engine.lastRequest().headers()).doesNotContainKey("Content-Type");
    assertThat(engine.lastRequest().connection().getDoOutput()).isFalse();
    assertThat(engine.lastRequest().body()).isEmpty();
  }

  @Test
  public void get_success_doesNotDisconnect() throws Exception {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/test", 200, new byte[0]);
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    var unused = client.get("https://localhost/test", ImmutableMap.of());

    assertThat(engine.lastRequest().connection().wasDisconnected()).isFalse();
  }

  @Test
  public void get_errorResponse_disconnects() {
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("/test", 500, new byte[0]);
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    assertThrows(
        FcpHttpException.class, () -> client.get("https://localhost/test", ImmutableMap.of()));

    assertThat(engine.lastRequest().connection().wasDisconnected()).isTrue();
  }

  @Test
  public void getResource_gzipContentType_returnsGunzippedBody() throws Exception {
    byte[] plain = "resource".getBytes(UTF_8);
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("resource", 200, gzip(plain), "application/octet-stream+gzip");
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    byte[] response = client.getResource("https://localhost/resource");

    assertThat(response).isEqualTo(plain);
  }

  @Test
  public void getResource_gzipBodyDecompressesPastCap_throwsIOException() throws Exception {
    byte[] plain = new byte[1000];
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("resource", 200, gzip(plain), "application/x-protobuf+gzip");
    FcpHttpClient client =
        new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT, /* maxResponseBytes= */ 64);

    IOException thrown =
        assertThrows(IOException.class, () -> client.getResource("https://localhost/resource"));

    assertThat(thrown).hasMessageThat().contains("exceeded");
  }

  @Test
  public void get_gzipContentType_returnsGunzippedBody() throws Exception {
    byte[] plain = "resource".getBytes(UTF_8);
    byte[] compressed = gzip(plain);
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo("resource", 200, compressed, "application/octet-stream+gzip");
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    byte[] response = client.get("https://localhost/resource", ImmutableMap.of());

    assertThat(response).isEqualTo(plain);
  }

  @Test
  public void execute_nullErrorStream_exceptionMessageContainsErrorNone() {
    HttpUrlConnectionEngine engine =
        url ->
            new HttpURLConnection(url) {
              @Override
              public void connect() {}

              @Override
              public void disconnect() {}

              @Override
              public boolean usingProxy() {
                return false;
              }

              @Override
              public int getResponseCode() {
                return 500;
              }

              @Override
              public InputStream getErrorStream() {
                return null;
              }
            };
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    FcpHttpException thrown =
        assertThrows(
            FcpHttpException.class, () -> client.get("https://localhost/test", ImmutableMap.of()));

    assertThat(thrown).hasMessageThat().contains("Error: none");
  }

  @Test
  public void execute_errorStreamThrowsOnRead_exceptionMessageContainsErrorNone() {
    InputStream throwingStream =
        new InputStream() {
          @Override
          public int read() throws IOException {
            throw new IOException("Stream read failed");
          }
        };
    HttpUrlConnectionEngine engine =
        url ->
            new HttpURLConnection(url) {
              @Override
              public void connect() {}

              @Override
              public void disconnect() {}

              @Override
              public boolean usingProxy() {
                return false;
              }

              @Override
              public int getResponseCode() {
                return 500;
              }

              @Override
              public InputStream getErrorStream() {
                return throwingStream;
              }
            };
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    FcpHttpException thrown =
        assertThrows(
            FcpHttpException.class, () -> client.get("https://localhost/test", ImmutableMap.of()));

    assertThat(thrown).hasMessageThat().contains("Error: none");
  }

  @Test
  public void execute_bodyLongerThanContentLength_readsWholeBody() throws Exception {
    // Cronet decodes a Content-Encoding body but keeps the encoded Content-Length.
    byte[] body = "decoded response body".getBytes(UTF_8);
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo(
        "/test",
        req ->
            FakeHttpUrlConnectionEngine.FakeResponse.of(200, body)
                .withHeader("Content-Length", "4"));
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    byte[] response = client.get("https://localhost/test", ImmutableMap.of());

    assertThat(response).isEqualTo(body);
  }

  @Test
  public void execute_bodyShorterThanContentLength_returnsBody() throws Exception {
    byte[] body = "short".getBytes(UTF_8);
    FakeHttpUrlConnectionEngine engine = new FakeHttpUrlConnectionEngine();
    engine.respondTo(
        "/test",
        req ->
            FakeHttpUrlConnectionEngine.FakeResponse.of(200, body)
                .withHeader("Content-Length", "100"));
    FcpHttpClient client = new FcpHttpClient(engine, CONNECT_TIMEOUT, READ_TIMEOUT);

    byte[] response = client.get("https://localhost/test", ImmutableMap.of());

    assertThat(response).isEqualTo(body);
  }

  private static byte[] gzip(byte[] data) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (GZIPOutputStream gzipOut = new GZIPOutputStream(out)) {
      gzipOut.write(data);
    }
    return out.toByteArray();
  }
}
