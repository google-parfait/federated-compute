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
package com.google.fcp.client.confidentialcompute.internal.encode;

import static com.google.common.util.concurrent.Futures.immediateFailedFuture;

import com.google.common.base.Ticker;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.fcp.client.TimeSource;
import com.google.fcp.client.confidentialcompute.ConfidentialComputeCrypto;
import com.google.fcp.client.confidentialcompute.ConfidentialComputeCrypto.VerificationResult;
import com.google.fcp.client.confidentialcompute.ConfidentialPayloadEncoder;
import com.google.fcp.client.confidentialcompute.EncodedPayload;
import com.google.fcp.client.confidentialcompute.PayloadEncodingException;
import com.google.fcp.client.confidentialcompute.TimestampedInput;
import com.google.fcp.client.confidentialcompute.WorkloadSet;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent;
import com.google.fcp.client.stabilitymonitor.DiagnosticEvent.Kind;
import com.google.fcp.client.stabilitymonitor.StabilityMonitor;
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;
import com.google.protobuf.ByteString;
import com.google.protos.fcp.confidentialcompute.BlobHeader;
import java.io.IOException;
import java.security.GeneralSecurityException;
import java.security.SecureRandom;
import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.Consumer;

/**
 * {@link ConfidentialPayloadEncoder} implementation that verifies the workload set's encryption
 * config, serializes inputs into an FCv1 checkpoint, builds the {@link BlobHeader} associated data,
 * and seals the payload into a CAv1 envelope. The work runs on the executor given at construction.
 *
 * <p>Verification matches {@code HttpFederatedProtocol::UploadResult}, which calls {@code
 * ValidateConfidentialEncryptionConfig} before encrypting, and encryption matches {@code
 * HttpFederatedProtocol::EncryptPayloadForConfidentialAggregation}, both in {@code
 * fcp/client/http/http_federated_protocol.cc}.
 */
public final class ConfidentialPayloadEncoderImpl implements ConfidentialPayloadEncoder {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * 128-bit random blob ID, matching {@code fcp::RandomToken} in {@code fcp/base/random_token.h}.
   */
  private static final int BLOB_ID_BYTES = 16;

  private static final SecureRandom SECURE_RANDOM = new SecureRandom();

  private final ConfidentialComputeCrypto crypto;
  private final ConfidentialComputeEncryptor encryptor;
  private final Consumer<ByteString> verificationRecordLogger;
  private final StabilityMonitor stabilityMonitor;
  private final Ticker ticker;
  private final TimeSource timeSource;
  private final Executor executor;

  /**
   * Creates an encoder that verifies and encrypts with {@code crypto}, passes each serialized
   * {@code AttestationVerificationRecord} to {@code verificationRecordLogger}, and runs on {@code
   * executor}.
   */
  public ConfidentialPayloadEncoderImpl(
      ConfidentialComputeCrypto crypto,
      Consumer<ByteString> verificationRecordLogger,
      StabilityMonitor stabilityMonitor,
      Ticker ticker,
      TimeSource timeSource,
      Executor executor) {
    this.crypto = crypto;
    this.encryptor = new ConfidentialComputeEncryptor(crypto);
    this.verificationRecordLogger = verificationRecordLogger;
    this.stabilityMonitor = Objects.requireNonNull(stabilityMonitor);
    this.ticker = Objects.requireNonNull(ticker);
    this.timeSource = Objects.requireNonNull(timeSource);
    this.executor = Objects.requireNonNull(executor);
  }

  @Override
  public ListenableFuture<EncodedPayload> encode(
      WorkloadSet workloadSet,
      AccessPolicyEndorsementOptions endorsementOptions,
      List<? extends TimestampedInput> inputs) {
    Objects.requireNonNull(workloadSet);
    Objects.requireNonNull(endorsementOptions);
    Objects.requireNonNull(inputs);
    // Futures.submit throws if the executor is shut down; report that like any other failure.
    try {
      return Futures.submit(
          () -> encodeSynchronously(workloadSet, endorsementOptions, inputs), executor);
    } catch (RejectedExecutionException e) {
      return immediateFailedFuture(e);
    }
  }

  /**
   * Runs on the executor, so the encode events time the encode itself and not the wait for the
   * executor.
   */
  private EncodedPayload encodeSynchronously(
      WorkloadSet workloadSet,
      AccessPolicyEndorsementOptions endorsementOptions,
      List<? extends TimestampedInput> inputs)
      throws PayloadEncodingException {
    logDiagnosticEvent(DiagnosticEvent.builder(Kind.ENCODE_STARTED), workloadSet);
    long startNanos = ticker.read();
    EncodedPayload payload;
    try {
      payload = buildPayload(workloadSet, endorsementOptions, inputs);
    } catch (PayloadEncodingException | RuntimeException e) {
      logDiagnosticEvent(
          DiagnosticEvent.builder(Kind.ENCODE_ERROR).setDuration(since(startNanos)).setError(e),
          workloadSet);
      throw e;
    }
    logDiagnosticEvent(
        DiagnosticEvent.builder(Kind.ENCODE_COMPLETED).setDuration(since(startNanos)), workloadSet);
    return payload;
  }

  private Duration since(long startNanos) {
    return Duration.ofNanos(ticker.read() - startNanos);
  }

  private void logDiagnosticEvent(DiagnosticEvent.Builder event, WorkloadSet workloadSet) {
    stabilityMonitor.logDiagnosticEvent(
        event
            .setTimestamp(timeSource.now())
            .setLogSourceName(workloadSet.logSourceName())
            .setFetchId(workloadSet.fetchId())
            .setWorkloadSetName(workloadSet.name())
            .build());
  }

  private EncodedPayload buildPayload(
      WorkloadSet workloadSet,
      AccessPolicyEndorsementOptions endorsementOptions,
      List<? extends TimestampedInput> inputs)
      throws PayloadEncodingException {
    if (inputs.isEmpty()) {
      throw new PayloadEncodingException("No inputs to encode");
    }

    VerificationResult verification;
    try {
      verification =
          crypto.verify(
              endorsementOptions, workloadSet.signedEndorsements(), workloadSet.encryptionConfig());
    } catch (GeneralSecurityException | IllegalArgumentException e) {
      throw new PayloadEncodingException("Attestation verification failed", e);
    }
    logVerificationRecord(verification.serializedVerificationRecord());

    // Build the BlobHeader bound as AEAD/HPKE associated data and carried alongside the upload
    // (see HttpFederatedProtocol::UploadResult in fcp/client/http/http_federated_protocol.cc).
    byte[] blobId = new byte[BLOB_ID_BYTES];
    SECURE_RANDOM.nextBytes(blobId);
    BlobHeader blobHeader =
        BlobHeader.newBuilder()
            .setBlobId(ByteString.copyFrom(blobId))
            .setKeyId(verification.keyId())
            .setAccessPolicySha256(verification.accessPolicySha256())
            .build();

    ByteString serializedBlobHeader = blobHeader.toByteString();

    ByteString checkpointBytes;
    try {
      checkpointBytes =
          FederatedComputeCheckpointBuilder.buildCheckpoint(
              inputs, workloadSet.legacyFcpCompat().outputVectorNames());
    } catch (IOException e) {
      throw new PayloadEncodingException("Failed to serialize FCCheckpoint", e);
    }

    ByteString encryptedEnvelope;
    try {
      encryptedEnvelope =
          encryptor.encrypt(checkpointBytes, verification.publicKey(), serializedBlobHeader);
    } catch (GeneralSecurityException | IOException e) {
      throw new PayloadEncodingException("Failed to seal payload into CAv1 envelope", e);
    }

    return EncodedPayload.create(encryptedEnvelope);
  }

  private void logVerificationRecord(ByteString serializedRecord) {
    if (serializedRecord.isEmpty()) {
      return;
    }
    try {
      verificationRecordLogger.accept(serializedRecord);
    } catch (RuntimeException e) {
      logger.atWarning().withCause(e).log("Failed to log attestation verification record");
    }
  }
}
