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

import com.google.common.util.concurrent.ListenableFuture;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;
import java.util.List;

/**
 * Encodes {@link TimestampedInput} records into an encrypted CAv1 payload for a {@link WorkloadSet}
 * (corresponding to {@code FederatedComputeCheckpointBuilder} in {@code
 * tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.cc}
 * and {@code HttpFederatedProtocol::EncryptPayloadForConfidentialAggregation} in {@code
 * fcp/client/http/http_federated_protocol.cc}).
 */
public interface ConfidentialPayloadEncoder {

  /**
   * Verifies the encryption config and endorsements of {@code workloadSet} against {@code
   * endorsementOptions}, then transforms {@code inputs} into a single encrypted CAv1 payload for
   * the verified key. The encoding runs on the encoder's executor, never on the caller's thread.
   *
   * <p>Only pass records that have not been contributed to {@code workloadSet} yet. A record may be
   * passed again only if its earlier upload failed with {@link UploadAbortedException} or never
   * happened. See the package README.
   *
   * <p>The future fails with {@link PayloadEncodingException} if attestation verification of the
   * workload set's encryption config fails, if {@code inputs} is empty, if an output vector of the
   * workload set's plan names no input column, or if the inputs cannot be serialized or encrypted,
   * and with {@link java.util.concurrent.RejectedExecutionException} if the encoder's executor has
   * been shut down (the call itself does not throw).
   *
   * @param workloadSet the fetched workload set to encode for
   * @param endorsementOptions the trusted KMS and access-policy verifying keys, and whether
   *     transparency log entries are required, that the workload set's encryption config is
   *     verified against
   * @param inputs the records to encode
   */
  ListenableFuture<EncodedPayload> encode(
      WorkloadSet workloadSet,
      AccessPolicyEndorsementOptions endorsementOptions,
      List<? extends TimestampedInput> inputs);
}
