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
import com.google.fcp.client.confidentialcompute.proto.ConfidentialAggregationEncryption;
import com.google.fcp.client.confidentialcompute.proto.LegacyFcpCompatProto;
import com.google.fcp.client.confidentialcompute.proto.UploadTokensProto;
import com.google.fcp.client.confidentialcompute.proto.WorkloadSetSelectorSpecProto;
import com.google.fcp.client.confidentialcompute.proto.WorkloadSetStateProto;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.InvalidProtocolBufferException;

/**
 * Client-side representation of a server workload set to which the client can upload data.
 *
 * <p>On the server, workloads (such as analytics pipelines or learning programs) that consume the
 * same client data schema under the same access policy are grouped into a workload set so that one
 * encrypted client upload can feed all of them. (Workloads with different schemas or different
 * access policies belong to separate workload sets, and changing a set's access policy requires
 * deploying a new workload set name if clients should re-contribute previously uploaded data.)
 *
 * <p>On the client, a {@code WorkloadSet} instance carries the state for one assigned workload set
 * across the three client phases:
 *
 * <ul>
 *   <li><b>Fetch ({@code WorkloadSetFetcher}):</b> checks in with the server, opens an upload
 *       session, downloads its encryption config, and returns one {@code WorkloadSet} per eligible
 *       set. Fetch does not verify the encryption config.
 *   <li><b>Caller selection (public fields):</b> the caller reads {@link #name()} to check
 *       contribution history, and applies {@link #selectorSpec()} (and, for legacy callers, {@link
 *       #legacyFcpCompat()}) to select local data matching the set's schema.
 *   <li><b>Encode and upload:</b> {@code ConfidentialPayloadEncoder} verifies {@link
 *       #encryptionConfig()} and {@link #signedEndorsements()} against the endorsement options the
 *       caller passes to {@code encode} and encrypts the selected data to the verified key, and
 *       {@code ConfidentialPayloadUploader} uploads and submits the ciphertext using the single-use
 *       session credentials in {@link #uploadTokens()}. Only encoders and uploaders need these.
 * </ul>
 *
 * <p>Instances can be serialized across process restarts via {@link #toByteString()} and {@link
 * #fromByteString(ByteString)}.
 */
@AutoValue
@Immutable
public abstract class WorkloadSet {

  /** Unique identifier of the workload set, used for contribution tracking. */
  public abstract String name();

  /**
   * Specification guiding the client application on which records to select from the local store
   * for this workload set.
   */
  public abstract WorkloadSetSelectorSpec selectorSpec();

  /** Compatibility metadata for legacy Federated Compute (FCP) plans. */
  public abstract LegacyFcpCompat legacyFcpCompat();

  /**
   * Serialized {@code SignedEndorsements} the server sent with the assignment, or empty if it sent
   * none. Not verified.
   */
  public abstract ByteString signedEndorsements();

  /**
   * Serialized {@code ConfidentialEncryptionConfig} of the upload session, holding the TEE's public
   * key and attestation evidence. Not verified.
   */
  public abstract ByteString encryptionConfig();

  /**
   * The single-use session credentials and endpoints an uploader sends the payload with. The tokens
   * are a bearer credential: do not log them or send them anywhere else.
   */
  public abstract UploadTokens uploadTokens();

  /** The log source the set was fetched for. */
  public abstract String logSourceName();

  /**
   * Id of the fetch that returned the set, set on its diagnostic events. Sets from one fetch share
   * it.
   */
  public abstract long fetchId();

  /**
   * Serializes this WorkloadSet, including its upload tokens, so it can be cached and reused within
   * its token lifetime.
   *
   * <p><b>The result is a bearer credential.</b> Anyone holding these bytes can upload to this
   * aggregation until the tokens expire. Store it with the same protection you would give an auth
   * token: app-private storage, never logged, never sent off device.
   *
   * <p>The serialized form contains live upload credentials: they are bound to one aggregation
   * session, are accepted by the server only for a short time after {@code fetch}, and are consumed
   * by one upload. Cache the bytes only in app-private storage and only to bridge a process
   * restart; re-fetch rather than reuse a stale set.
   */
  public ByteString toByteString() {
    UploadTokens tokens = uploadTokens();
    UploadTokensProto tokensProto =
        UploadTokensProto.newBuilder()
            .setAggregationId(tokens.aggregationId())
            .setClientToken(tokens.clientToken())
            .setUploadUri(tokens.uploadUri())
            .setResourceName(tokens.resourceName())
            .setSubmitUriPrefix(tokens.submitUriPrefix())
            .putAllSubmitExtraHeaders(tokens.submitExtraHeaders())
            .putAllUploadExtraHeaders(tokens.uploadExtraHeaders())
            .build();

    WorkloadSetSelectorSpecProto selectorProto =
        WorkloadSetSelectorSpecProto.newBuilder()
            .setRawCriteria(selectorSpec().rawCriteria())
            .build();

    LegacyFcpCompatProto compatProto =
        LegacyFcpCompatProto.newBuilder()
            .setCollectionUri(legacyFcpCompat().collectionUri())
            .putAllOutputVectorNames(legacyFcpCompat().outputVectorNames())
            .build();

    return WorkloadSetStateProto.newBuilder()
        .setName(name())
        .setSelectorSpec(selectorProto)
        .setLegacyFcpCompat(compatProto)
        .setConfidentialAggregation(
            ConfidentialAggregationEncryption.newBuilder()
                .setSignedEndorsements(signedEndorsements())
                .setEncryptionConfig(encryptionConfig()))
        .setUploadTokens(tokensProto)
        .setLogSourceName(logSourceName())
        .setFetchId(fetchId())
        .build()
        .toByteString();
  }

  /**
   * Inverse of {@link #toByteString}.
   *
   * @throws InvalidProtocolBufferException if the bytes do not parse or a required field (name,
   *     confidential aggregation encryption and its encryption config, or the upload session
   *     tokens) is missing.
   */
  public static WorkloadSet fromByteString(ByteString bytes) throws InvalidProtocolBufferException {
    WorkloadSetStateProto proto =
        WorkloadSetStateProto.parseFrom(bytes, ExtensionRegistryLite.getEmptyRegistry());

    checkRequired(!proto.getName().isEmpty(), "name");
    checkRequired(proto.hasConfidentialAggregation(), "confidential_aggregation");
    ConfidentialAggregationEncryption encryptionProto = proto.getConfidentialAggregation();
    checkRequired(!encryptionProto.getEncryptionConfig().isEmpty(), "encryption_config");
    checkRequired(proto.hasUploadTokens(), "upload_tokens");

    UploadTokensProto tokensProto = proto.getUploadTokens();
    checkRequired(!tokensProto.getAggregationId().isEmpty(), "aggregation_id");
    checkRequired(!tokensProto.getClientToken().isEmpty(), "client_token");
    checkRequired(!tokensProto.getUploadUri().isEmpty(), "upload_uri");
    checkRequired(!tokensProto.getResourceName().isEmpty(), "resource_name");
    checkRequired(!tokensProto.getSubmitUriPrefix().isEmpty(), "submit_uri_prefix");

    UploadTokens tokens =
        UploadTokens.builder()
            .setAggregationId(tokensProto.getAggregationId())
            .setClientToken(tokensProto.getClientToken())
            .setUploadUri(tokensProto.getUploadUri())
            .setResourceName(tokensProto.getResourceName())
            .setSubmitUriPrefix(tokensProto.getSubmitUriPrefix())
            .setSubmitExtraHeaders(tokensProto.getSubmitExtraHeadersMap())
            .setUploadExtraHeaders(tokensProto.getUploadExtraHeadersMap())
            .build();

    WorkloadSetSelectorSpec selectorSpec =
        WorkloadSetSelectorSpec.of(proto.getSelectorSpec().getRawCriteria());

    LegacyFcpCompat legacyFcpCompat =
        LegacyFcpCompat.of(
            proto.getLegacyFcpCompat().getCollectionUri(),
            proto.getLegacyFcpCompat().getOutputVectorNamesMap());

    return WorkloadSet.builder()
        .setName(proto.getName())
        .setSelectorSpec(selectorSpec)
        .setLegacyFcpCompat(legacyFcpCompat)
        .setSignedEndorsements(encryptionProto.getSignedEndorsements())
        .setEncryptionConfig(encryptionProto.getEncryptionConfig())
        .setUploadTokens(tokens)
        .setLogSourceName(proto.getLogSourceName())
        .setFetchId(proto.getFetchId())
        .build();
  }


  private static void checkRequired(boolean present, String field)
      throws InvalidProtocolBufferException {
    if (!present) {
      throw new InvalidProtocolBufferException("WorkloadSet is missing required field: " + field);
    }
  }

  /**
   * Omits tokens, URIs, encryption config and endorsements: a WorkloadSet is a bearer credential
   * and toString output ends up in logs.
   */
  @Override
  public final String toString() {
    return "WorkloadSet{name=" + name() + "}";
  }

  /** Returns a new builder with an empty selector spec and legacy compat preset. */
  public static Builder builder() {
    return new AutoValue_WorkloadSet.Builder()
        .setSelectorSpec(WorkloadSetSelectorSpec.of(ByteString.EMPTY))
        .setLegacyFcpCompat(LegacyFcpCompat.of(""));
  }

  /** Builder for {@link WorkloadSet}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setName(String name);

    public abstract Builder setSelectorSpec(WorkloadSetSelectorSpec selectorSpec);

    public abstract Builder setLegacyFcpCompat(LegacyFcpCompat legacyFcpCompat);

    public abstract Builder setSignedEndorsements(ByteString signedEndorsements);

    public abstract Builder setEncryptionConfig(ByteString encryptionConfig);

    public abstract Builder setUploadTokens(UploadTokens uploadTokens);

    public abstract Builder setLogSourceName(String logSourceName);

    public abstract Builder setFetchId(long fetchId);

    public abstract WorkloadSet build();
  }
}
