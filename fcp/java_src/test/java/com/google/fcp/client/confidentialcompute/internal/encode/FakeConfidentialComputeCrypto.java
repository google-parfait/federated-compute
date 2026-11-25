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
import com.google.intelligence.fcp.confidentialcompute.AccessPolicyEndorsementOptions;
import com.google.intelligence.fcp.confidentialcompute.Key;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protos.fcp.confidentialcompute.AggregationClientPayloadHeader;
import java.io.IOException;
import java.io.InputStream;
import java.security.GeneralSecurityException;
import java.util.zip.GZIPInputStream;

/**
 * {@link ConfidentialComputeCrypto} that records its verify and encrypt inputs, returns {@link
 * #verificationResult} from verify, and returns the plaintext unchanged from encrypt, so tests can
 * inspect the CAv1 framing.
 */
final class FakeConfidentialComputeCrypto implements ConfidentialComputeCrypto {
  static final ByteString ENCAPPED_KEY = ByteString.copyFromUtf8("encapped-key");
  static final ByteString ENCRYPTED_SYMMETRIC_KEY = ByteString.copyFromUtf8("encrypted-key");

  ByteString plaintext;
  Key recipientPublicKey;
  ByteString associatedData;
  boolean throwOnEncrypt = false;

  AccessPolicyEndorsementOptions verifiedOptions;
  ByteString verifiedSignedEndorsements;
  ByteString verifiedEncryptionConfig;
  VerificationResult verificationResult;
  boolean throwOnVerify = false;

  @Override
  public VerificationResult verify(
      AccessPolicyEndorsementOptions options,
      ByteString signedEndorsements,
      ByteString encryptionConfig)
      throws GeneralSecurityException {
    this.verifiedOptions = options;
    this.verifiedSignedEndorsements = signedEndorsements;
    this.verifiedEncryptionConfig = encryptionConfig;
    if (throwOnVerify) {
      throw new GeneralSecurityException("Fake verification failure");
    }
    return verificationResult;
  }

  @Override
  public EncryptionResult encrypt(byte[] plaintext, Key recipientPublicKey, byte[] associatedData)
      throws GeneralSecurityException {
    if (throwOnEncrypt) {
      throw new GeneralSecurityException("Fake encryption failure");
    }
    this.plaintext = ByteString.copyFrom(plaintext);
    this.recipientPublicKey = recipientPublicKey;
    this.associatedData = ByteString.copyFrom(associatedData);
    return EncryptionResult.create(plaintext, ENCAPPED_KEY, ENCRYPTED_SYMMETRIC_KEY);
  }

  /** A parsed CAv1 envelope. */
  static final class Envelope {
    final ByteString magic;
    final AggregationClientPayloadHeader header;
    final ByteString ciphertext;

    Envelope(ByteString envelope) throws IOException {
      magic = envelope.substring(0, 4);
      InputStream in = envelope.substring(4).newInput();
      header =
          AggregationClientPayloadHeader.parseDelimitedFrom(
              in, ExtensionRegistryLite.getEmptyRegistry());
      ciphertext = ByteString.readFrom(in);
    }
  }

  static ByteString gunzip(ByteString compressed) throws IOException {
    return ByteString.readFrom(new GZIPInputStream(compressed.newInput()));
  }
}
