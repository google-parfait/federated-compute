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
package com.google.fcp.client.privatelogger.fcpconnector;

import com.google.auto.value.AutoValue;
import com.google.fcp.client.confidentialcompute.ConfidentialCompute;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;

/** Dependencies of {@link FcpPrivateLoggerConnector}. */
@AutoValue
public abstract class FcpPrivateLoggerConnectorDeps {

  /** The confidential compute client whose fetcher, encoder and uploader the connector uses. */
  public abstract ConfidentialCompute confidentialCompute();

  /**
   * Endorsement requirements the encoder verifies attestations against. Required: the default
   * instance has no verifying keys, so every verification would fail.
   */
  public abstract AccessPolicyEndorsementOptions endorsementOptions();

  /** Returns a new builder. */
  public static Builder builder() {
    return new AutoValue_FcpPrivateLoggerConnectorDeps.Builder();
  }

  /** Builder for {@link FcpPrivateLoggerConnectorDeps}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setConfidentialCompute(ConfidentialCompute confidentialCompute);

    public abstract Builder setEndorsementOptions(
        AccessPolicyEndorsementOptions endorsementOptions);

    public abstract FcpPrivateLoggerConnectorDeps build();
  }
}
