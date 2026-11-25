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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class UrlRedactionTest {

  @Test
  public void redact_clientTokenInCustomMethodPath_isRedacted() {
    String url =
        "https://fcp.example.com/v1/populations/pop/taskassignments/session-1/clients/"
            + "SECRET_TOKEN:startdataupload";

    assertThat(UrlRedaction.redact(url))
        .isEqualTo(
            "https://fcp.example.com/v1/populations/pop/taskassignments/session-1/clients/"
                + "<redacted>:startdataupload");
  }

  @Test
  public void redact_clientTokenFollowedByMoreSegments_isRedacted() {
    String url = "https://fcp.example.com/v1/sessions/s/clients/SECRET_TOKEN/uploads/agg-1";

    assertThat(UrlRedaction.redact(url))
        .isEqualTo("https://fcp.example.com/v1/sessions/s/clients/<redacted>/uploads/agg-1");
  }

  @Test
  public void redact_queryString_isDropped() {
    String url = "https://storage.example.com/plan/abc?X-Goog-Signature=SIG&Expires=1";

    assertThat(UrlRedaction.redact(url)).isEqualTo("https://storage.example.com/plan/abc");
  }

  @Test
  public void redact_mediaResourceName_isRedacted() {
    String url =
        "https://upload.example.com/upload/v1/media/SCOTTY_RESOURCE_NAME?upload_protocol=raw";

    assertThat(UrlRedaction.redact(url))
        .isEqualTo("https://upload.example.com/upload/v1/media/<redacted>");
  }

  @Test
  public void redact_nothingSensitive_isReturnedAsIs() {
    String url = "https://fcp.example.com/v1/populations/pop/taskassignments:create";

    assertThat(UrlRedaction.redact(url)).isEqualTo(url);
  }
}
