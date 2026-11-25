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

import static com.google.common.util.concurrent.Futures.immediateFuture;

import com.google.common.util.concurrent.ListenableFuture;

/**
 * Supplies the device attestation measurement sent with the task assignment request. Corresponds to
 * {@code SimpleTaskEnvironment::GetAttestationMeasurement} ({@code
 * fcp/client/simple_task_environment.h}), which {@code fcp/client/fl_runner.cc} calls once per run
 * between the eligibility eval check-in and task assignment. The measurement is requested inside
 * the assignment phase and a failure is logged as ASSIGNMENT_ERROR.
 */
public interface DeviceAttestationProvider {

  /**
   * Returns the measurement for {@code logSourceName}, bound to {@code contentBinding} (the
   * check-in session id from {@code EligibilityEvalTaskResponse.session_id}). An implementation
   * that binds nothing may ignore {@code contentBinding}. {@code fetchId} is the id of the fetch
   * asking (see {@code DiagnosticEvent.fetchId()}), for an implementation that logs to its own
   * {@code StabilityMonitor}. The result is sent verbatim as {@code
   * PerformMultipleTaskAssignmentsRequest.attestation_measurement.value}; an empty string means no
   * measurement, and the future must not complete with null. A failed future fails the fetch.
   *
   * <p>The fetcher blocks on the future without a timeout, as C++ does, so an implementation that
   * can hang should bound its own future.
   */
  ListenableFuture<String> getAttestationMeasurement(
      String logSourceName, String contentBinding, long fetchId);

  /** Provides no measurement: always returns an empty string. */
  DeviceAttestationProvider NONE =
      new DeviceAttestationProvider() {
        @Override
        public ListenableFuture<String> getAttestationMeasurement(
            String logSourceName, String contentBinding, long fetchId) {
          return immediateFuture("");
        }
      };
}
