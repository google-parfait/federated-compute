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

import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

@RunWith(RobolectricTestRunner.class)
public final class FakeHttpUrlConnectionEngineTest {

  private FakeHttpUrlConnectionEngine engine;

  @Before
  public void setUp() {
    engine = new FakeHttpUrlConnectionEngine();
  }

  @Test
  public void longestSubstringWins() throws Exception {
    engine.respondTo(":short", 200, "short".getBytes(UTF_8));
    engine.respondTo(":short_and_long", 201, "long".getBytes(UTF_8));

    FakeHttpUrlConnectionEngine.FakeHttpUrlConnection conn =
        (FakeHttpUrlConnectionEngine.FakeHttpUrlConnection)
            engine.openConnection(URI.create("https://host/api:short_and_long").toURL());

    assertThat(conn.getResponseCode()).isEqualTo(201);

    conn =
        (FakeHttpUrlConnectionEngine.FakeHttpUrlConnection)
            engine.openConnection(URI.create("https://host/api:short_only").toURL());
    assertThat(conn.getResponseCode()).isEqualTo(200);
  }

  @Test
  public void defaultResponseWhenNothingMatches() throws Exception {
    FakeHttpUrlConnectionEngine.FakeHttpUrlConnection conn =
        (FakeHttpUrlConnectionEngine.FakeHttpUrlConnection)
            engine.openConnection(URI.create("https://host/api:unknown").toURL());
    assertThat(conn.getResponseCode()).isEqualTo(404);
    assertThat(conn.getErrorStream().read()).isEqualTo(-1); // empty body
  }

  @Test
  public void errorStatusServesFromErrorStream() throws Exception {
    engine.respondTo(":error", 400, "error body".getBytes(UTF_8));

    FakeHttpUrlConnectionEngine.FakeHttpUrlConnection conn =
        (FakeHttpUrlConnectionEngine.FakeHttpUrlConnection)
            engine.openConnection(URI.create("https://host/api:error").toURL());

    assertThat(conn.getResponseCode()).isEqualTo(400);
    assertThrows(IOException.class, conn::getInputStream);

    byte[] errorBody = new byte[100];
    int len = conn.getErrorStream().read(errorBody);
    assertThat(new String(errorBody, 0, len, UTF_8)).isEqualTo("error body");
  }

  @Test
  public void failWithThrowsExceptions() throws Exception {
    IOException expectedIo = new IOException("io err");
    engine.failWith(":io", expectedIo);

    URL ioUrl = URI.create("https://host/api:io").toURL();
    IOException thrownIo = assertThrows(IOException.class, () -> engine.openConnection(ioUrl));
    assertThat(thrownIo).isSameInstanceAs(expectedIo);

    RuntimeException expectedRt = new RuntimeException("rt err");
    engine.failWith(":rt", expectedRt);

    URL rtUrl = URI.create("https://host/api:rt").toURL();
    RuntimeException thrownRt =
        assertThrows(RuntimeException.class, () -> engine.openConnection(rtUrl));
    assertThat(thrownRt).isSameInstanceAs(expectedRt);
  }

  @Test
  public void responseFunctionSeesRequestAndVariesByCallCount() throws Exception {
    AtomicInteger calls = new AtomicInteger(0);
    engine.respondTo(
        ":dynamic",
        req -> {
          int call = calls.incrementAndGet();
          if (call == 1) {
            return FakeHttpUrlConnectionEngine.FakeResponse.of(200, "first".getBytes(UTF_8));
          } else {
            return FakeHttpUrlConnectionEngine.FakeResponse.of(201, req.method().getBytes(UTF_8));
          }
        });

    FakeHttpUrlConnectionEngine.FakeHttpUrlConnection conn1 =
        (FakeHttpUrlConnectionEngine.FakeHttpUrlConnection)
            engine.openConnection(URI.create("https://host/api:dynamic").toURL());
    conn1.setRequestMethod("POST");
    assertThat(conn1.getResponseCode()).isEqualTo(200);

    FakeHttpUrlConnectionEngine.FakeHttpUrlConnection conn2 =
        (FakeHttpUrlConnectionEngine.FakeHttpUrlConnection)
            engine.openConnection(URI.create("https://host/api:dynamic").toURL());
    conn2.setRequestMethod("PUT");
    assertThat(conn2.getResponseCode()).isEqualTo(201);

    byte[] body2 = new byte[100];
    int len2 = conn2.getInputStream().read(body2);
    assertThat(new String(body2, 0, len2, UTF_8)).isEqualTo("PUT");
  }

  @Test
  public void recordedRequestCapturesDetails() throws Exception {
    engine.respondTo(":record", 200, new byte[0]);
    FakeHttpUrlConnectionEngine.FakeHttpUrlConnection conn =
        (FakeHttpUrlConnectionEngine.FakeHttpUrlConnection)
            engine.openConnection(URI.create("https://host/api:record").toURL());

    conn.setRequestMethod("POST");
    conn.setRequestProperty("X-Custom", "value");
    conn.getOutputStream().write("body data".getBytes(UTF_8));

    FakeHttpUrlConnectionEngine.RecordedRequest req = engine.lastRequest();
    assertThat(req.url()).isEqualTo("https://host/api:record");
    assertThat(req.method()).isEqualTo("POST");
    assertThat(req.headers()).containsEntry("X-Custom", "value");
    assertThat(req.body()).isEqualTo("body data".getBytes(UTF_8));
    assertThat(req.threadName()).isEqualTo(Thread.currentThread().getName());
  }

  @Test
  public void queryMethodsBehave() throws Exception {
    var unused1 = engine.openConnection(URI.create("https://host/api:req1").toURL());
    var unused2 = engine.openConnection(URI.create("https://host/api:req2").toURL());
    var unused3 = engine.openConnection(URI.create("https://host/api:req2_again").toURL());

    assertThat(engine.anyRequestContains(":req1")).isTrue();
    assertThat(engine.anyRequestContains(":req3")).isFalse();

    assertThat(engine.requestUrls())
        .containsExactly(
            "https://host/api:req1", "https://host/api:req2", "https://host/api:req2_again")
        .inOrder();

    assertThat(engine.requestsContaining(":req2")).hasSize(2);

    assertThat(engine.onlyRequestContaining(":req1").url()).isEqualTo("https://host/api:req1");

    assertThrows(AssertionError.class, () -> engine.onlyRequestContaining(":req2"));
    assertThrows(AssertionError.class, () -> engine.onlyRequestContaining(":req3"));
  }
}
