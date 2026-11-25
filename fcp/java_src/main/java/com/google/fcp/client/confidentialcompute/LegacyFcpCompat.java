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

/**
 * Compatibility metadata for legacy Federated Compute (FCP) plans, such as the {@code
 * ExampleSelector} collection URI.
 */
@AutoValue
@Immutable
public abstract class LegacyFcpCompat {

  /** Collection URI from the plan's ExampleSelector (e.g. "app:/..."), or empty if not set. */
  public abstract String collectionUri();

  /**
   * The plan's {@code ExampleQuery.output_vector_specs}, as output name to vector name. The server
   * names them {@code <query_name>/<column>}, and the uploaded checkpoint must use the output names
   * as its tensor names. Empty if the plan has none.
   */
  // TODO: b/573865751 - Delete once the server no longer requires the <query_name>/ prefix.
  public abstract ImmutableMap<String, String> outputVectorNames();

  /** Returns a {@link LegacyFcpCompat} with the given collection URI; empty means not set. */
  public static LegacyFcpCompat of(String collectionUri) {
    return of(collectionUri, ImmutableMap.of());
  }

  /** Returns a {@link LegacyFcpCompat} with the given collection URI and output vector names. */
  public static LegacyFcpCompat of(String collectionUri, Map<String, String> outputVectorNames) {
    return new AutoValue_LegacyFcpCompat(collectionUri, ImmutableMap.copyOf(outputVectorNames));
  }
}
