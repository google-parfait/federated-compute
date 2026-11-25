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
package com.google.fcp.client;

import java.time.Duration;
import java.time.Instant;

/** A fake implementation of {@link TimeSource} for testing. */
public class FakeTimeSource implements TimeSource {
  private Instant now;

  /** Creates a new {@link FakeTimeSource} with a default current time. */
  public FakeTimeSource() {
    this(Instant.ofEpochMilli(1000000000));
  }

  /** Creates a new {@link FakeTimeSource} with the given current time. */
  public FakeTimeSource(Instant now) {
    this.now = now;
  }

  /** Returns the current fake time. */
  @Override
  public Instant now() {
    return now;
  }

  /** Advances the current fake time by the given duration. */
  public void advance(Duration duration) {
    now = now.plus(duration);
  }
}
