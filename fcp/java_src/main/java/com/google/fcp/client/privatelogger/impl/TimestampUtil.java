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

import java.text.ParseException;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Date;
import java.util.Locale;

/** Utility class for Timestamp conversions. */
public final class TimestampUtil {

  // DateTimeFormatter is immutable and thread-safe.
  private static final DateTimeFormatter ISO_8601_FORMATTER =
      DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSXXX", Locale.US);

  private TimestampUtil() {}

  /**
   * Formats an Instant into an ISO-8601 string with a colon in the timezone offset.
   *
   * @param instant The Instant to format.
   * @return The formatted string (e.g. '2016-09-12T08:00:00.123-07:00').
   */
  public static String getIso8601String(Instant instant) {
    return instant.atZone(ZoneId.systemDefault()).format(ISO_8601_FORMATTER);
  }

  /**
   * Parses an ISO-8601 string with a colon in the timezone offset.
   *
   * @param iso8601String The string to parse.
   * @return a Date object.
   * @throws ParseException if the string cannot be parsed.
   */
  public static Date parseIso8601String(String iso8601String) throws ParseException {
    try {
      return Date.from(ISO_8601_FORMATTER.parse(iso8601String, Instant::from));
    } catch (DateTimeParseException e) {
      throw new ParseException(e.getMessage(), e.getErrorIndex());
    }
  }
}
