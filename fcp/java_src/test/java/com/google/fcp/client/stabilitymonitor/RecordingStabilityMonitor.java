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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.intelligence.fcp.client.ProdDiagCode;
import java.util.ArrayList;
import java.util.List;

/** A {@link StabilityMonitor} that records what it receives, for tests. */
public final class RecordingStabilityMonitor implements StabilityMonitor {
  private final List<Throwable> crashes = new ArrayList<>();
  private final List<DiagnosticEvent> events = new ArrayList<>();
  private final List<ProdDiagCode> diagCodes = new ArrayList<>();
  private final List<Long> diagFetchIds = new ArrayList<>();

  @Override
  public synchronized void reportCrash(Throwable t) {
    crashes.add(t);
  }

  @Override
  public synchronized void logDiagnosticEvent(DiagnosticEvent event) {
    events.add(event);
  }

  @Override
  public synchronized void logDiagCode(int code, long fetchId) {
    diagCodes.add(ProdDiagCode.forNumber(code));
    diagFetchIds.add(fetchId);
  }

  public synchronized ImmutableList<Throwable> crashes() {
    return ImmutableList.copyOf(crashes);
  }

  public synchronized ImmutableList<DiagnosticEvent> events() {
    return ImmutableList.copyOf(events);
  }

  /** The logged diag codes, in order. */
  public synchronized ImmutableList<ProdDiagCode> diagCodes() {
    return ImmutableList.copyOf(diagCodes);
  }

  /** The fetch ids of the logged diag codes, in order. */
  public synchronized ImmutableList<Long> diagFetchIds() {
    return ImmutableList.copyOf(diagFetchIds);
  }

  /** The kinds of the logged events, in order. */
  public synchronized ImmutableList<DiagnosticEvent.Kind> kinds() {
    return events.stream().map(DiagnosticEvent::kind).collect(toImmutableList());
  }

  /** The only event of {@code kind}; fails if there is not exactly one. */
  public synchronized DiagnosticEvent event(DiagnosticEvent.Kind kind) {
    ImmutableList<DiagnosticEvent> matching =
        events.stream().filter(e -> e.kind() == kind).collect(toImmutableList());
    if (matching.size() != 1) {
      throw new IllegalStateException("Expected one " + kind + " event, got " + kinds());
    }
    return matching.get(0);
  }
}
