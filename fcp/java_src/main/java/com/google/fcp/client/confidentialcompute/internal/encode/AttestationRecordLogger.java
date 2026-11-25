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

import android.util.Log;
import com.google.common.collect.ImmutableList;
import com.google.protobuf.ByteString;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.function.Consumer;
import java.util.zip.GZIPOutputStream;

/**
 * Logs serialized {@code AttestationVerificationRecord}s to logcat in the format emitted by {@code
 * fcp::client::attestation::LogSerializedVerificationRecord} ({@code
 * fcp/client/attestation/log_attestation_records.cc}) and parsed by {@code
 * fcp/client/attestation/extract_attestation_records.cc}.
 *
 * <p>When {@code adb shell setprop log.tag.fcp.attest D} is enabled, the record is gzipped,
 * base64-encoded, and written directly via {@link Log#d} under {@link #LOG_TAG} as an introductory
 * line followed by 200-character angle-bracketed chunks and a terminating {@code <>} line. Direct
 * {@link Log} calls are used instead of Flogger so that logcat output matches the exact tag and
 * line format expected by {@code extract_attestation_records}.
 */
public final class AttestationRecordLogger implements Consumer<ByteString> {

  static final String LOG_TAG = "fcp.attest";

  static final String INFO_MESSAGE =
      "This device is contributing data via the confidential aggregation protocol. The attestation"
          + " verification record follows.";

  static final int CHUNK_LENGTH = 200;


  /** Logs to logcat under {@link #LOG_TAG}. */
  public static final AttestationRecordLogger LOGCAT = new AttestationRecordLogger();

  private AttestationRecordLogger() {}

  @Override
  public void accept(ByteString serializedRecord) {
    if (serializedRecord.isEmpty() || !Log.isLoggable(LOG_TAG, Log.DEBUG)) {
      return;
    }
    ImmutableList<String> lines;
    try {
      lines = formatRecord(serializedRecord);
    } catch (IOException e) {
      Log.w(LOG_TAG, "Failed to compress attestation verification record", e);
      return;
    }
    for (String line : lines) {
      Log.d(LOG_TAG, line);
    }
  }

  /**
   * Returns the log lines for {@code serializedRecord}: the introductory message, then the gzipped
   * and base64-encoded record in bracketed {@link #CHUNK_LENGTH}-character chunks, then an empty
   * {@code <>} chunk that separates consecutive records.
   */
  static ImmutableList<String> formatRecord(ByteString serializedRecord) throws IOException {
    String encoded = Base64.getEncoder().encodeToString(gzip(serializedRecord));
    ImmutableList.Builder<String> lines = ImmutableList.builder();
    lines.add(INFO_MESSAGE);
    for (int start = 0; start < encoded.length(); start += CHUNK_LENGTH) {
      int end = Math.min(start + CHUNK_LENGTH, encoded.length());
      lines.add("<" + encoded.substring(start, end) + ">");
    }
    return lines.add("<>").build();
  }

  private static byte[] gzip(ByteString bytes) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (GZIPOutputStream gzip = new GZIPOutputStream(out)) {
      bytes.writeTo(gzip);
    }
    return out.toByteArray();
  }
}
