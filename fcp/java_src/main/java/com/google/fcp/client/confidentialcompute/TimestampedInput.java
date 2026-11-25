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
package com.google.fcp.client.confidentialcompute;

import com.google.auto.value.AutoValue;
import com.google.errorprone.annotations.Immutable;
import com.google.protobuf.ByteString;

/**
 * A single input record paired with its creation timestamp for confidential aggregation.
 *
 * <p>Callers whose own record type implements this interface can pass records to {@link
 * ConfidentialPayloadEncoder#encode} without copying; otherwise use {@link #of}. Implementations
 * must be immutable.
 */
@Immutable
public interface TimestampedInput {

  /** The serialized value of the input. */
  ByteString value();

  /**
   * The creation timestamp of the input, as a string formatted according to ISO-8601 with the
   * device's local time-zone offset. For example: "2016-09-12T08:00:00.123-07:00".
   */
  String timestamp();

  /** Returns a {@link TimestampedInput} holding the given value and timestamp. */
  static TimestampedInput of(ByteString value, String timestamp) {
    return new AutoValue_TimestampedInput_Simple(value, timestamp);
  }

  /** Minimal value-type implementation backing {@link #of}. */
  @AutoValue
  @Immutable
  abstract class Simple implements TimestampedInput {
    Simple() {}
  }
}
