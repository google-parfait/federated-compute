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

import com.google.fcp.client.confidentialcompute.ConfidentialComputeCrypto;
import com.google.fcp.client.confidentialcompute.ConfidentialComputeCrypto.EncryptionResult;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import com.google.protos.fcp.confidentialcompute.AggregationClientPayloadHeader;
import com.google.protos.fcp.confidentialcompute.CompressionType;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.GeneralSecurityException;
import java.util.zip.GZIPOutputStream;

/**
 * Compresses and encrypts a payload into a CAv1 envelope, matching {@code
 * HttpFederatedProtocol::EncryptPayloadForConfidentialAggregation} ({@code
 * fcp/client/http/http_federated_protocol.cc}) and {@code EncodeClientPayload} ({@code
 * fcp/confidentialcompute/client_payload.cc}).
 */
final class ConfidentialComputeEncryptor {

  /** {@code kConfidentialAggPayloadV1MagicBytes} in {@code client_payload.cc}. */
  private static final ByteString MAGIC_BYTES = ByteString.copyFromUtf8("CAv1");

  private final ConfidentialComputeCrypto crypto;

  ConfidentialComputeEncryptor(ConfidentialComputeCrypto crypto) {
    this.crypto = crypto;
  }

  /**
   * Gzip-compresses {@code payload}, encrypts it with {@code serializedBlobHeader} as associated
   * data, and frames the result as {@code "CAv1" || varint(header_len) ||
   * AggregationClientPayloadHeader || ciphertext}.
   */
  ByteString encrypt(ByteString payload, Key recipientPublicKey, ByteString serializedBlobHeader)
      throws GeneralSecurityException, IOException {
    // Compress before encrypting, matching EncryptPayloadForConfidentialAggregation.
    // Use an estimated initial size and stream from payload.newInput() to avoid flattened
    // allocations.
    ByteArrayOutputStream compressed =
        new ByteArrayOutputStream(Math.max(8192, payload.size() / 2));
    try (GZIPOutputStream gzip = new GZIPOutputStream(compressed);
        InputStream payloadInput = payload.newInput()) {
      byte[] buffer = new byte[8192];
      int read;
      while ((read = payloadInput.read(buffer)) != -1) {
        gzip.write(buffer, 0, read);
      }
    }

    byte[] compressedBytes = compressed.toByteArray();
    byte[] blobHeaderBytes = serializedBlobHeader.toByteArray();
    EncryptionResult encrypted =
        crypto.encrypt(compressedBytes, recipientPublicKey, blobHeaderBytes);

    // Frame as "CAv1" || varint32(header_size) || AggregationClientPayloadHeader || ciphertext
    // (writeDelimitedTo writes the varint32 size prefix followed by the serialized header proto,
    // matching EncodeClientPayload in fcp/confidentialcompute/client_payload.cc).
    AggregationClientPayloadHeader header =
        AggregationClientPayloadHeader.newBuilder()
            .setEncryptedSymmetricKey(encrypted.encryptedSymmetricKey())
            .setEncapsulatedPublicKey(encrypted.encappedKey())
            .setBlobHeader(serializedBlobHeader)
            .setCompressionType(CompressionType.COMPRESSION_TYPE_GZIP)
            .build();

    int headerSize = header.getSerializedSize();
    int delimiterSize = CodedOutputStream.computeUInt32SizeNoTag(headerSize);
    byte[] ciphertext = encrypted.ciphertext();
    int envelopeSize = MAGIC_BYTES.size() + delimiterSize + headerSize + ciphertext.length;

    // exactSize makes toByteString() wrap the internal array with no trailing copy.
    ByteString.Output envelope = ByteString.newOutput(envelopeSize);
    MAGIC_BYTES.writeTo(envelope);
    header.writeDelimitedTo(envelope);
    envelope.write(ciphertext, 0, ciphertext.length);
    return envelope.toByteString();
  }
}
