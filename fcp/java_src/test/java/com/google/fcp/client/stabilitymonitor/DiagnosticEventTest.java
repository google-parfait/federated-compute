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
package com.google.fcp.client.stabilitymonitor;

import static com.google.common.truth.Truth.assertThat;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent.Kind;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class DiagnosticEventTest {

  @Test
  public void builder_defaultsEveryFieldButKind() {
    DiagnosticEvent event = DiagnosticEvent.builder(Kind.CHECKIN_STARTED).build();

    assertThat(event.kind()).isEqualTo(Kind.CHECKIN_STARTED);
    assertThat(event.logSourceName()).isNull();
    assertThat(event.workloadSetName()).isNull();
    assertThat(event.duration()).isEqualTo(Duration.ZERO);
    assertThat(event.timestamp()).isNull();
    assertThat(event.bytesUploaded()).isEqualTo(0);
    assertThat(event.bytesDownloaded()).isEqualTo(0);
    assertThat(event.inputCount()).isEqualTo(0);
    assertThat(event.inputBytes()).isEqualTo(0);
    assertThat(event.error()).isNull();
  }

  @Test
  public void builder_setsFields() {
    IOException error = new IOException("boom");
    Instant now = Instant.parse("2026-11-12T12:00:00Z");

    DiagnosticEvent event =
        DiagnosticEvent.builder(Kind.UPLOAD_ERROR)
            .setLogSourceName("source")
            .setWorkloadSetName("set")
            .setDuration(Duration.ofMillis(5))
            .setTimestamp(now)
            .setBytesUploaded(10)
            .setBytesDownloaded(20)
            .setInputCount(3)
            .setInputBytes(30)
            .setError(error)
            .build();

    assertThat(event.kind()).isEqualTo(Kind.UPLOAD_ERROR);
    assertThat(event.logSourceName()).isEqualTo("source");
    assertThat(event.workloadSetName()).isEqualTo("set");
    assertThat(event.duration()).isEqualTo(Duration.ofMillis(5));
    assertThat(event.timestamp()).isEqualTo(now);
    assertThat(event.bytesUploaded()).isEqualTo(10);
    assertThat(event.bytesDownloaded()).isEqualTo(20);
    assertThat(event.inputCount()).isEqualTo(3);
    assertThat(event.inputBytes()).isEqualTo(30);
    assertThat(event.error()).isSameInstanceAs(error);
  }
}
