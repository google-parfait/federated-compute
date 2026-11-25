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

import static com.google.common.truth.Truth.assertThat;

import com.google.fcp.client.confidentialcompute.internal.encode.FakeConfidentialComputeCrypto.Envelope;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.protobuf.ByteString;
import com.google.protos.fcp.confidentialcompute.CompressionType;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ConfidentialComputeEncryptor}. */
@RunWith(JUnit4.class)
public final class ConfidentialComputeEncryptorTest {
  private static final ByteString PAYLOAD = ByteString.copyFromUtf8("checkpoint");
  private static final Key PUBLIC_KEY =
      Key.newBuilder()
          .setAlgorithm(Key.Algorithm.HPKE_X25519_SHA256_AES128_GCM)
          .setKeyMaterial(ByteString.copyFromUtf8("public-key"))
          .build();
  private static final ByteString BLOB_HEADER = ByteString.copyFromUtf8("blob-header");

  private final FakeConfidentialComputeCrypto crypto = new FakeConfidentialComputeCrypto();
  private final ConfidentialComputeEncryptor encryptor = new ConfidentialComputeEncryptor(crypto);

  @Test
  public void encrypt_encryptsGzippedPayloadWithBlobHeaderAsAad() throws Exception {
    var unused = encryptor.encrypt(PAYLOAD, PUBLIC_KEY, BLOB_HEADER);

    assertThat(FakeConfidentialComputeCrypto.gunzip(crypto.plaintext)).isEqualTo(PAYLOAD);
    assertThat(crypto.recipientPublicKey).isEqualTo(PUBLIC_KEY);
    assertThat(crypto.associatedData).isEqualTo(BLOB_HEADER);
  }

  @Test
  public void encrypt_framesCav1Envelope() throws Exception {
    Envelope envelope = new Envelope(encryptor.encrypt(PAYLOAD, PUBLIC_KEY, BLOB_HEADER));

    assertThat(envelope.magic.toStringUtf8()).isEqualTo("CAv1");
    assertThat(envelope.header.getEncapsulatedPublicKey())
        .isEqualTo(FakeConfidentialComputeCrypto.ENCAPPED_KEY);
    assertThat(envelope.header.getEncryptedSymmetricKey())
        .isEqualTo(FakeConfidentialComputeCrypto.ENCRYPTED_SYMMETRIC_KEY);
    assertThat(envelope.header.getBlobHeader()).isEqualTo(BLOB_HEADER);
    assertThat(envelope.header.getCompressionType())
        .isEqualTo(CompressionType.COMPRESSION_TYPE_GZIP);
    assertThat(envelope.ciphertext).isEqualTo(crypto.plaintext);
  }
}
