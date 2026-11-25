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

import android.util.Log;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent.Kind;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.shadows.ShadowLog;

@RunWith(AndroidJUnit4.class)
public final class StabilityMonitorTest {

  @Test
  public void defaultMonitor_reportCrash_handsTheCrashToTheUncaughtExceptionHandler() {
    List<Throwable> uncaught = new ArrayList<>();
    Thread thread = Thread.currentThread();
    Thread.UncaughtExceptionHandler original = thread.getUncaughtExceptionHandler();
    thread.setUncaughtExceptionHandler((t, e) -> uncaught.add(e));
    IllegalStateException bug = new IllegalStateException("bug");
    try {
      StabilityMonitor.DEFAULT_STABILITY_MONITOR.reportCrash(bug);
    } finally {
      thread.setUncaughtExceptionHandler(original);
    }

    assertThat(uncaught).containsExactly(bug);
  }

  @Test
  public void defaultMonitor_logDiagnosticEvent_logsTheEventAtVerboseLevel() {
    ShadowLog.setLoggable(StabilityMonitor.TAG, Log.VERBOSE);
    DiagnosticEvent event = DiagnosticEvent.builder(Kind.CHECKIN_STARTED).build();

    StabilityMonitor.DEFAULT_STABILITY_MONITOR.logDiagnosticEvent(event);

    assertThat(ShadowLog.getLogsForTag(StabilityMonitor.TAG)).hasSize(1);
    ShadowLog.LogItem item = ShadowLog.getLogsForTag(StabilityMonitor.TAG).get(0);
    assertThat(item.type).isEqualTo(Log.VERBOSE);
    assertThat(item.msg).isEqualTo(event.toString());
  }

  @Test
  public void defaultMonitor_logDiagCode_logsTheCodeAndFetchIdAtVerboseLevel() {
    ShadowLog.setLoggable(StabilityMonitor.TAG, Log.VERBOSE);

    StabilityMonitor.DEFAULT_STABILITY_MONITOR.logDiagCode(105, 7);

    assertThat(ShadowLog.getLogsForTag(StabilityMonitor.TAG)).hasSize(1);
    ShadowLog.LogItem item = ShadowLog.getLogsForTag(StabilityMonitor.TAG).get(0);
    assertThat(item.type).isEqualTo(Log.VERBOSE);
    assertThat(item.msg).isEqualTo("diag=105 fetchId=7");
  }
}
