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
package com.google.fcp.client.flags;

import java.time.Duration;

/**
 * Runtime flags read by the default confidential compute stages. Hosts back it with their own flag
 * system and override the methods whose defaults they change. Flags are read once when {@code
 * ConfidentialCompute} is built; a later change to the host's flag values is not seen by an
 * instance that is already built.
 */
public interface FcpClientFlags {

  /** The flags with their default values. */
  FcpClientFlags DEFAULT = new FcpClientFlags() {};

  /** Default for {@link #httpConnectTimeout()}. */
  Duration DEFAULT_HTTP_CONNECT_TIMEOUT = Duration.ofSeconds(10);

  /** Default for {@link #httpReadTimeout()}. */
  Duration DEFAULT_HTTP_READ_TIMEOUT = Duration.ofSeconds(30);

  /** Federated compute server URL. */
  default String serverUrl() {
    return "https://federatedcompute-pa.googleapis.com";
  }

  /** HTTP connect timeout. */
  default Duration httpConnectTimeout() {
    return DEFAULT_HTTP_CONNECT_TIMEOUT;
  }

  /** HTTP read timeout. */
  default Duration httpReadTimeout() {
    return DEFAULT_HTTP_READ_TIMEOUT;
  }
}
