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
package com.google.fcp.client.confidentialcompute.internal.crypto;

import com.google.common.base.Preconditions;
import com.google.crypto.tink.hybrid.HpkePrivateKey;
import com.google.crypto.tink.subtle.Bytes;
import java.math.BigInteger;
import java.security.GeneralSecurityException;
import java.util.Arrays;
import javax.annotation.concurrent.GuardedBy;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Hybrid Public Key Encryption (HPKE) context for either a sender or a recipient.
 *
 * <p>https://www.rfc-editor.org/rfc/rfc9180.html#name-creating-the-encryption-con
 *
 * <p>Branched from Tink's {@code com.google.crypto.tink.hybrid.internal.HpkeContext}, without the
 * authenticated ({@code mode_auth}) contexts. The {@code associatedData} parameter of {@link #seal}
 * and {@link #open} is the AEAD associated data that Tink's own {@code HpkeEncrypt}/{@code
 * HpkeDecrypt} always leave empty and that this package exposes.
 */
@ThreadSafe
final class HpkeContext {
  private static final byte[] emptyIkm = new byte[0];

  private final HpkeAead aead;
  private final BigInteger maxSequenceNumber;
  private final byte[] key;
  private final byte[] baseNonce;
  private final byte[] encapsulatedKey;

  @GuardedBy("this")
  private BigInteger sequenceNumber;

  private HpkeContext(
      byte[] encapsulatedKey,
      byte[] key,
      byte[] baseNonce,
      BigInteger maxSequenceNumber,
      HpkeAead aead) {
    this.encapsulatedKey = encapsulatedKey;
    this.key = key;
    this.baseNonce = baseNonce;
    this.sequenceNumber = BigInteger.ZERO;
    this.maxSequenceNumber = maxSequenceNumber;
    this.aead = aead;
  }

  /** Helper function factored out to facilitate unit testing. */
  static HpkeContext createContext(
      byte[] mode,
      byte[] encapsulatedKey,
      byte[] sharedSecret,
      HpkeKem kem,
      HpkeKdf kdf,
      HpkeAead aead,
      byte[] info)
      throws GeneralSecurityException {
    byte[] suiteId = HpkeUtil.hpkeSuiteId(kem.getKemId(), kdf.getKdfId(), aead.getAeadId());
    byte[] pskIdHash = kdf.labeledExtract(HpkeUtil.emptySalt, emptyIkm, "psk_id_hash", suiteId);
    byte[] infoHash = kdf.labeledExtract(HpkeUtil.emptySalt, info, "info_hash", suiteId);
    byte[] keyScheduleContext = Bytes.concat(mode, pskIdHash, infoHash);
    byte[] secret = kdf.labeledExtract(sharedSecret, emptyIkm, "secret", suiteId);

    byte[] key = kdf.labeledExpand(secret, keyScheduleContext, "key", suiteId, aead.getKeyLength());
    byte[] baseNonce =
        kdf.labeledExpand(secret, keyScheduleContext, "base_nonce", suiteId, aead.getNonceLength());
    BigInteger maxSeqNo = maxSequenceNumber(aead.getNonceLength());

    return new HpkeContext(encapsulatedKey, key, baseNonce, maxSeqNo, aead);
  }

  /**
   * Creates HPKE sender context according to KeySchedule() defined in
   * https://www.rfc-editor.org/rfc/rfc9180.html#section-5.1-9.
   *
   * @param recipientPublicKey recipient's public key (pkR)
   * @param kem key encapsulation mechanism primitive
   * @param kdf key derivation function primitive
   * @param aead authenticated encryption with associated data primitive
   * @param info application-specific information parameter to influence key generation
   */
  static HpkeContext createSenderContext(
      byte[] recipientPublicKey, HpkeKem kem, HpkeKdf kdf, HpkeAead aead, byte[] info)
      throws GeneralSecurityException {
    HpkeKemEncapOutput encapOutput = kem.encapsulate(recipientPublicKey);
    byte[] encapsulatedKey = encapOutput.getEncapsulatedKey();
    byte[] sharedSecret = encapOutput.getSharedSecret();
    return createContext(HpkeUtil.baseMode, encapsulatedKey, sharedSecret, kem, kdf, aead, info);
  }

  /**
   * Creates HPKE recipient context according to KeySchedule() defined in
   * https://www.rfc-editor.org/rfc/rfc9180.html#section-5.1-9.
   *
   * @param encapsulatedKey encapsulated key (enc)
   * @param recipientPrivateKey recipient's private key (skR)
   * @param kem key encapsulation mechanism primitive
   * @param kdf key derivation function primitive
   * @param aead authenticated encryption with associated data primitive
   * @param info application-specific information parameter to influence key generation
   */
  static HpkeContext createRecipientContext(
      byte[] encapsulatedKey,
      HpkePrivateKey recipientPrivateKey,
      HpkeKem kem,
      HpkeKdf kdf,
      HpkeAead aead,
      byte[] info)
      throws GeneralSecurityException {
    byte[] sharedSecret = kem.decapsulate(encapsulatedKey, recipientPrivateKey);
    return createContext(HpkeUtil.baseMode, encapsulatedKey, sharedSecret, kem, kdf, aead, info);
  }

  private static BigInteger maxSequenceNumber(int nonceLength) {
    return BigInteger.ONE.shiftLeft(8 * nonceLength).subtract(BigInteger.ONE);
  }

  @GuardedBy("this")
  private void incrementSequenceNumber() throws GeneralSecurityException {
    if (sequenceNumber.compareTo(maxSequenceNumber) >= 0) {
      throw new GeneralSecurityException("message limit reached");
    }
    sequenceNumber = sequenceNumber.add(BigInteger.ONE);
  }

  /** ComputeNonce() from https://www.rfc-editor.org/rfc/rfc9180.html#section-5.2-11. */
  @GuardedBy("this")
  private byte[] computeNonce() throws GeneralSecurityException {
    return Bytes.xor(
        baseNonce, toBigEndianBytesOfFixedLength(sequenceNumber, aead.getNonceLength()));
  }

  /** Returns the next nonce to use for seal/open. Also, increments the sequence number. */
  private synchronized byte[] computeNonceAndIncrementSequenceNumber()
      throws GeneralSecurityException {
    byte[] nonce = computeNonce();
    incrementSequenceNumber();
    return nonce;
  }

  /**
   * {@code com.google.crypto.tink.internal.BigIntegerEncoding#toBigEndianBytesOfFixedLength}: the
   * big-endian encoding of the non-negative {@code n} in exactly {@code length} bytes.
   */
  private static byte[] toBigEndianBytesOfFixedLength(BigInteger n, int length)
      throws GeneralSecurityException {
    Preconditions.checkArgument(n.signum() != -1, "integer must be nonnegative");
    byte[] b = n.toByteArray();
    if (b.length == length) {
      return b;
    }
    if (b.length > length + 1 /* potential leading zero */) {
      throw new GeneralSecurityException("integer too large");
    }
    if (b.length == length + 1) {
      if (b[0] == 0 /* leading zero */) {
        return Arrays.copyOfRange(b, 1, b.length);
      } else {
        throw new GeneralSecurityException("integer too large");
      }
    }
    // Left zero pad b.
    byte[] res = new byte[length];
    System.arraycopy(b, 0, res, length - b.length, b.length);
    return res;
  }

  byte[] getKey() {
    return key;
  }

  byte[] getBaseNonce() {
    return baseNonce;
  }

  byte[] getEncapsulatedKey() {
    return encapsulatedKey;
  }

  /**
   * Performs AEAD encryption of {@code plaintext} with {@code associatedData} according to
   * ContextS.Seal() defined in https://www.rfc-editor.org/rfc/rfc9180.html#section-5.2-8.
   *
   * @return ciphertext
   */
  byte[] seal(byte[] plaintext, byte[] associatedData) throws GeneralSecurityException {
    byte[] nonce = computeNonceAndIncrementSequenceNumber();
    return aead.seal(key, nonce, plaintext, associatedData);
  }

  byte[] seal(byte[] plaintext, int ciphertextOffset, byte[] associatedData)
      throws GeneralSecurityException {
    byte[] nonce = computeNonceAndIncrementSequenceNumber();
    return aead.seal(key, nonce, plaintext, ciphertextOffset, associatedData);
  }

  /**
   * Performs AEAD decryption of {@code ciphertext} with {@code associatedData} according to
   * ContextR.Open() defined in https://www.rfc-editor.org/rfc/rfc9180.html#section-5.2-10.
   *
   * @return plaintext
   */
  byte[] open(byte[] ciphertext, byte[] associatedData) throws GeneralSecurityException {
    return open(ciphertext, /* ciphertextOffset= */ 0, associatedData);
  }

  byte[] open(byte[] ciphertext, int ciphertextOffset, byte[] associatedData)
      throws GeneralSecurityException {
    byte[] nonce = computeNonceAndIncrementSequenceNumber();
    return aead.open(key, nonce, ciphertext, ciphertextOffset, associatedData);
  }
}
