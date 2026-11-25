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
public final class UriEncodingTest {

  @Test
  public void encodePathSegment_unreservedChars_untouched() {
    String unreserved = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~";
    assertThat(UriEncoding.encodePathSegment(unreserved)).isEqualTo(unreserved);
    assertThat(UriEncoding.encodePathSegments(unreserved)).isEqualTo(unreserved);
  }

  @Test
  public void encodePathSegment_space_encodedToPercent20() {
    assertThat(UriEncoding.encodePathSegment("hello world")).isEqualTo("hello%20world");
    assertThat(UriEncoding.encodePathSegments("hello world")).isEqualTo("hello%20world");
  }

  @Test
  public void encodePathSegment_plus_encodedToPercent2B() {
    assertThat(UriEncoding.encodePathSegment("a+b")).isEqualTo("a%2Bb");
    assertThat(UriEncoding.encodePathSegments("a+b")).isEqualTo("a%2Bb");
  }

  @Test
  public void encodePathSegment_star_encodedToPercent2A() {
    assertThat(UriEncoding.encodePathSegment("a*b")).isEqualTo("a%2Ab");
    assertThat(UriEncoding.encodePathSegments("a*b")).isEqualTo("a%2Ab");
  }

  @Test
  public void encodePathSegment_slash_encodedOnlyBySingleSegment() {
    assertThat(UriEncoding.encodePathSegment("a/b/c")).isEqualTo("a%2Fb%2Fc");
    assertThat(UriEncoding.encodePathSegments("a/b/c")).isEqualTo("a/b/c");
  }

  @Test
  public void encodePathSegment_colonAndAt_encoded() {
    assertThat(UriEncoding.encodePathSegment("tok:en@x")).isEqualTo("tok%3Aen%40x");
    assertThat(UriEncoding.encodePathSegments("tok:en@x")).isEqualTo("tok%3Aen%40x");
  }

  @Test
  public void encodePathSegment_nonAsciiChar_utf8PercentEncoded() {
    assertThat(UriEncoding.encodePathSegment("café")).isEqualTo("caf%C3%A9");
    assertThat(UriEncoding.encodePathSegments("café/latte")).isEqualTo("caf%C3%A9/latte");
  }
}
