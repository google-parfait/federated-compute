// Copyright 2025 Google LLC
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
package com.google.fcp.client.privatelogger.impl;

import static com.google.common.truth.Truth.assertThat;

import java.time.Instant;
import java.util.TimeZone;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

@RunWith(RobolectricTestRunner.class)
public class TimestampUtilTest {
  private TimeZone originalTimeZone;

  @Before
  public void setUp() {
    originalTimeZone = TimeZone.getDefault();
  }

  @After
  public void tearDown() {
    TimeZone.setDefault(originalTimeZone);
  }

  @Test
  public void testGetIso8601String_pst() {
    TimeZone.setDefault(TimeZone.getTimeZone("America/Los_Angeles"));
    Instant instant = Instant.parse("2024-01-01T12:00:00Z");
    String formatted = TimestampUtil.getIso8601String(instant);
    assertThat(formatted).isEqualTo("2024-01-01T04:00:00.000-08:00");
  }

  @Test
  public void testGetIso8601String_utc() {
    TimeZone.setDefault(TimeZone.getTimeZone("UTC"));
    Instant instant = Instant.parse("2024-01-01T12:00:00Z");
    String formatted = TimestampUtil.getIso8601String(instant);
    assertThat(formatted).isEqualTo("2024-01-01T12:00:00.000Z");
  }

  @Test
  public void testParseThenFormat_isLossless() throws Exception {
    TimeZone.setDefault(TimeZone.getTimeZone("America/New_York")); // -05:00
    String originalString = "2024-01-01T12:34:56.789-05:00";
    Instant instant = TimestampUtil.parseIso8601String(originalString).toInstant();
    String formattedString = TimestampUtil.getIso8601String(instant);
    assertThat(formattedString).isEqualTo(originalString);
  }

  @Test
  public void testGetIso8601String_positiveOffset() {
    // Set timezone to Central European Time (UTC+1).
    TimeZone.setDefault(TimeZone.getTimeZone("Europe/Paris"));
    Instant instant = Instant.parse("2024-01-01T12:00:00Z");
    String formatted = TimestampUtil.getIso8601String(instant);
    // Should be 1pm in Paris.
    assertThat(formatted).isEqualTo("2024-01-01T13:00:00.000+01:00");
  }

  @Test
  public void testGetIso8601String_halfHourOffset() {
    // Set timezone to India Standard Time (UTC+5:30).
    TimeZone.setDefault(TimeZone.getTimeZone("Asia/Kolkata"));
    Instant instant = Instant.parse("2024-01-01T12:00:00Z");
    String formatted = TimestampUtil.getIso8601String(instant);
    // Should be 5:30pm in Kolkata.
    assertThat(formatted).isEqualTo("2024-01-01T17:30:00.000+05:30");
  }

  @Test
  public void testParseIso8601String_utc() throws Exception {
    String utcString = "2024-01-01T12:00:00.000Z";
    Instant instant = TimestampUtil.parseIso8601String(utcString).toInstant();
    assertThat(instant).isEqualTo(Instant.parse("2024-01-01T12:00:00Z"));
  }

  @Test
  public void testParseIso8601String_negativeOffset() throws Exception {
    String pstString = "2024-01-01T04:00:00.000-08:00";
    Instant instant = TimestampUtil.parseIso8601String(pstString).toInstant();
    // The parsed instant should correspond to 12:00 UTC.
    assertThat(instant).isEqualTo(Instant.parse("2024-01-01T12:00:00Z"));
  }

  @Test
  public void testParseIso8601String_positiveOffset() throws Exception {
    String cetString = "2024-01-01T13:00:00.000+01:00";
    Instant instant = TimestampUtil.parseIso8601String(cetString).toInstant();
    // The parsed instant should correspond to 12:00 UTC.
    assertThat(instant).isEqualTo(Instant.parse("2024-01-01T12:00:00Z"));
  }

  @Test
  public void testParseIso8601String_halfHourOffset() throws Exception {
    String istString = "2024-01-01T17:30:00.000+05:30";
    Instant instant = TimestampUtil.parseIso8601String(istString).toInstant();
    // The parsed instant should correspond to 12:00 UTC.
    assertThat(instant).isEqualTo(Instant.parse("2024-01-01T12:00:00Z"));
  }
}
