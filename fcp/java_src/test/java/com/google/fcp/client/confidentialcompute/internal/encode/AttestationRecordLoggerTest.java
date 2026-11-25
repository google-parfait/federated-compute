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
package com.google.fcp.client.confidentialcompute.internal.encode;

import static com.google.common.truth.Truth.assertThat;

import android.util.Log;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteStreams;
import com.google.protobuf.ByteString;
import java.io.ByteArrayInputStream;
import java.util.Base64;
import java.util.zip.GZIPInputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.shadows.ShadowLog;

@RunWith(RobolectricTestRunner.class)
public final class AttestationRecordLoggerTest {

  @Test
  public void formatRecord_producesInfoLineChunksAndTerminator() throws Exception {
    // Large enough to need several 200-character chunks after gzip + base64.
    byte[] payload = new byte[4096];
    for (int i = 0; i < payload.length; i++) {
      payload[i] = (byte) (i * 31);
    }
    ByteString record = ByteString.copyFrom(payload);

    ImmutableList<String> lines = AttestationRecordLogger.formatRecord(record);

    assertThat(lines.size()).isAtLeast(4);
    assertThat(lines.get(0)).isEqualTo(AttestationRecordLogger.INFO_MESSAGE);
    assertThat(Iterables.getLast(lines)).isEqualTo("<>");
    StringBuilder encoded = new StringBuilder();
    for (String chunk : lines.subList(1, lines.size() - 1)) {
      assertThat(chunk).startsWith("<");
      assertThat(chunk).endsWith(">");
      String body = chunk.substring(1, chunk.length() - 1);
      assertThat(body.length()).isAtMost(AttestationRecordLogger.CHUNK_LENGTH);
      assertThat(body).isNotEmpty();
      encoded.append(body);
    }
    // Every chunk except the last must be full-length.
    for (String chunk : lines.subList(1, lines.size() - 2)) {
      assertThat(chunk.length() - 2).isEqualTo(AttestationRecordLogger.CHUNK_LENGTH);
    }

    assertThat(gunzip(Base64.getDecoder().decode(encoded.toString()))).isEqualTo(payload);
  }

  @Test
  public void formatRecord_emptyRecord_stillEmitsInfoAndTerminator() throws Exception {
    ImmutableList<String> lines = AttestationRecordLogger.formatRecord(ByteString.EMPTY);

    assertThat(lines.get(0)).isEqualTo(AttestationRecordLogger.INFO_MESSAGE);
    assertThat(Iterables.getLast(lines)).isEqualTo("<>");
    // gzip of an empty input is still non-empty (header + trailer), so one chunk is expected.
    assertThat(lines).hasSize(3);
  }

  @Test
  public void logcatAccept_emptyRecord_logsNothing() throws Exception {
    ShadowLog.setLoggable(AttestationRecordLogger.LOG_TAG, Log.DEBUG);
    AttestationRecordLogger.LOGCAT.accept(ByteString.EMPTY);
    assertThat(ShadowLog.getLogsForTag(AttestationRecordLogger.LOG_TAG)).isEmpty();
  }

  @Test
  public void logcatAccept_notLoggable_logsNothing() throws Exception {
    // ShadowLog default is not loggable
    AttestationRecordLogger.LOGCAT.accept(ByteString.copyFromUtf8("record"));
    assertThat(ShadowLog.getLogsForTag(AttestationRecordLogger.LOG_TAG)).isEmpty();
  }

  @Test
  public void logcatAccept_loggableAndNonEmpty_logsChunksInOrder() throws Exception {
    ShadowLog.setLoggable(AttestationRecordLogger.LOG_TAG, Log.DEBUG);
    ByteString record = ByteString.copyFromUtf8("my_attestation_record");

    AttestationRecordLogger.LOGCAT.accept(record);

    ImmutableList<String> expectedLines = AttestationRecordLogger.formatRecord(record);
    ImmutableList<String> actualLines =
        ShadowLog.getLogsForTag(AttestationRecordLogger.LOG_TAG).stream()
            .map(logItem -> logItem.msg)
            .collect(ImmutableList.toImmutableList());

    assertThat(actualLines).containsExactlyElementsIn(expectedLines).inOrder();
  }

  private static byte[] gunzip(byte[] compressed) throws Exception {
    try (GZIPInputStream in = new GZIPInputStream(new ByteArrayInputStream(compressed))) {
      return ByteStreams.toByteArray(in);
    }
  }
}
