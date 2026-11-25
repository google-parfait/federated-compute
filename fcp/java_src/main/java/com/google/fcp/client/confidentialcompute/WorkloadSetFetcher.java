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

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;

/**
 * Fetches eligible {@link WorkloadSet} instances for a private log source.
 *
 * <p>This is the first of the three confidential compute stages (fetch, encode, upload). An
 * implementation performs the server eligibility check-in, opens an upload session per workload
 * set, and returns each set with its unverified encryption config and endorsements.
 */
public interface WorkloadSetFetcher {
  /**
   * Resolves the workload sets the server currently offers for a private log source.
   *
   * <p>The future fails with an IOException on a transport failure, a non-2xx response or a
   * malformed server response, and with {@link java.util.concurrent.RejectedExecutionException} if
   * the fetcher's executor has been shut down (the call itself does not throw); a server rejection
   * of the check-in is not a failure and yields an empty list.
   *
   * @param privateLogSourceName Name of the private log source; this is the FCP population name
   *     used in every request.
   * @return A {@link ListenableFuture} resolving to the {@link WorkloadSet} instances, or an empty
   *     list if none are eligible.
   */
  ListenableFuture<ImmutableList<WorkloadSet>> fetch(String privateLogSourceName);
}
