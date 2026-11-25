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
import com.google.common.collect.ImmutableMap;
import com.google.errorprone.annotations.Immutable;
import java.util.Map;

/** Ephemeral authorization tokens and routing URIs for data upload and finalization. */
@AutoValue
@Immutable
public abstract class UploadTokens {

  /** Server-generated aggregation session ID. */
  public abstract String aggregationId();

  /** Client participation token returned by StartConfidentialAggregationDataUpload. */
  public abstract String clientToken();

  /** Target URI prefix for Scotty / ByteStream upload. */
  public abstract String uploadUri();

  /** Scotty / ByteStream resource name for payload streaming. */
  public abstract String resourceName();

  /** Target URI prefix for submitting the aggregation completion RPC. */
  public abstract String submitUriPrefix();

  /** Extra HTTP request headers required when calling submit. */
  public abstract ImmutableMap<String, String> submitExtraHeaders();

  /** Extra HTTP request headers required when uploading data via ByteStream. */
  public abstract ImmutableMap<String, String> uploadExtraHeaders();

  /**
   * Omits tokens and URIs: upload tokens are bearer credentials and toString output ends up in
   * logs.
   */
  @Override
  public final String toString() {
    return "UploadTokens{aggregationId=" + aggregationId() + ", <redacted>}";
  }

  /** Returns a new builder with empty header maps preset. */
  public static Builder builder() {
    return new AutoValue_UploadTokens.Builder()
        .setSubmitExtraHeaders(ImmutableMap.of())
        .setUploadExtraHeaders(ImmutableMap.of());
  }

  /** Builder for {@link UploadTokens}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setAggregationId(String aggregationId);

    public abstract Builder setClientToken(String clientToken);

    public abstract Builder setUploadUri(String uploadUri);

    public abstract Builder setResourceName(String resourceName);

    public abstract Builder setSubmitUriPrefix(String submitUriPrefix);

    public abstract Builder setSubmitExtraHeaders(Map<String, String> submitExtraHeaders);

    public abstract Builder setUploadExtraHeaders(Map<String, String> uploadExtraHeaders);

    public abstract UploadTokens build();
  }
}
