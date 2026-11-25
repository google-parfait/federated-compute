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
 * Specification defining the criteria used to select relevant records from local storage for a
 * specific workload set.
 */
@AutoValue
@Immutable
public abstract class WorkloadSetSelectorSpec {

  /** Raw selection criteria the server configured for this workload set. */
  public abstract ByteString rawCriteria();

  /** Omits the criteria bytes, which may carry caller-private query text. */
  @Override
  public final String toString() {
    return "WorkloadSetSelectorSpec{rawCriteria=" + rawCriteria().size() + " bytes}";
  }

  /** Returns a {@link WorkloadSetSelectorSpec} with the given raw criteria; empty means none. */
  public static WorkloadSetSelectorSpec of(ByteString rawCriteria) {
    return new AutoValue_WorkloadSetSelectorSpec(rawCriteria);
  }
}
