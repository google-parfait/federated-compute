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

import android.util.Log;

/**
 * Receives crashes and diagnostic events from the client libraries, so that the host can forward
 * them to its own crash reporting and logging.
 *
 * <p>All methods are called on whichever thread hit the crash or event, and must return quickly and
 * never throw.
 */
public interface StabilityMonitor {
  static final String TAG = "StabilityMonitor";

  /**
   * Reports a bug in the client libraries. The host decides what happens next: the default crashes
   * the process, and a host with its own crash reporting can report and carry on.
   *
   * @param t The throwable representing the crash.
   */
  void reportCrash(Throwable t);

  /**
   * Logs a diagnostic event from the confidential compute client. A monitor that does not override
   * this ignores events.
   *
   * @param event The event, see {@link DiagnosticEvent.Kind} for when each kind is logged.
   */
  default void logDiagnosticEvent(DiagnosticEvent event) {}

  /**
   * Logs a diag code for the fetch {@code fetchId}. The client logs the C++ client's codes ({@code
   * fcp/client/diag_codes.proto}) at the points where it logs them. The code is a plain number, so
   * a host can map it onto its own codes. A monitor that does not override this ignores diag codes.
   */
  default void logDiagCode(int code, long fetchId) {}

  /**
   * Default implementation. It logs a crash and hands it to the thread's uncaught exception
   * handler, which on Android writes the crash report and ends the process. Throwing would not do
   * that: callers report from future callbacks and executor tasks, where a throw is caught. It logs
   * diagnostic events and diag codes to logcat at verbose level.
   */
  public static final StabilityMonitor DEFAULT_STABILITY_MONITOR =
      new StabilityMonitor() {
        @Override
        public void reportCrash(Throwable t) {
          Log.e(TAG, "Uncaught exception", t);
          Thread thread = Thread.currentThread();
          thread.getUncaughtExceptionHandler().uncaughtException(thread, t);
        }

        @Override
        public void logDiagnosticEvent(DiagnosticEvent event) {
          if (Log.isLoggable(TAG, Log.VERBOSE)) {
            Log.v(TAG, event.toString());
          }
        }

        @Override
        public void logDiagCode(int code, long fetchId) {
          if (Log.isLoggable(TAG, Log.VERBOSE)) {
            Log.v(TAG, "diag=" + code + " fetchId=" + fetchId);
          }
        }
      };
}
