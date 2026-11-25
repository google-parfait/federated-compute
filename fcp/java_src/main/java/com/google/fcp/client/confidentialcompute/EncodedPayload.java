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
 * Encrypted CAv1 envelope (produced by {@code EncodeClientPayload} in {@code
 * fcp/confidentialcompute/client_payload.cc}) ready for byte-stream upload.
 */
@AutoValue
@Immutable
public abstract class EncodedPayload {

  /** The fully framed, encrypted CAv1 payload bytes ready for byte-stream upload. */
  public abstract ByteString encryptedEnvelope();

  /** Creates an encoded payload with the encrypted envelope bytes. */
  public static EncodedPayload create(ByteString encryptedEnvelope) {
    return new AutoValue_EncodedPayload(encryptedEnvelope);
  }
}
