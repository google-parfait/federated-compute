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

import com.google.errorprone.annotations.Immutable;
import java.security.GeneralSecurityException;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/**
 * HKDF HPKE KDF variant.
 *
 * <p>Branched from Tink's {@code com.google.crypto.tink.hybrid.internal.HkdfHpkeKdf}, reduced to
 * HKDF-SHA256 and using the JCE {@link Mac} directly instead of Tink's {@code EngineFactory}.
 */
@Immutable
final class HkdfHpkeKdf implements HpkeKdf {
  private final String macAlgorithm;

  private HkdfHpkeKdf(String macAlgorithm) {
    this.macAlgorithm = macAlgorithm;
  }

  static HkdfHpkeKdf create(String macAlgorithm) throws GeneralSecurityException {
    if (!macAlgorithm.equals("HmacSha256")) {
      throw new GeneralSecurityException("Unsupported MAC algorithm: " + macAlgorithm);
    }
    return new HkdfHpkeKdf(macAlgorithm);
  }

  /**
   * Copied from {@code com.google.crypto.tink.subtle.Hkdf#computeHkdf(String, byte[], byte[],
   * byte[], int)}
   */
  private byte[] extract(byte[] ikm, byte[] salt) throws GeneralSecurityException {
    Mac mac = Mac.getInstance(macAlgorithm);
    if (salt == null || salt.length == 0) {
      // According to RFC 5869, Section 2.2 the salt is optional. If no salt is provided
      // then HKDF uses a salt that is an array of zeros of the same length as the hash digest.
      mac.init(new SecretKeySpec(new byte[mac.getMacLength()], macAlgorithm));
    } else {
      mac.init(new SecretKeySpec(salt, macAlgorithm));
    }
    return mac.doFinal(ikm);
  }

  /**
   * Copied from {@code com.google.crypto.tink.subtle.Hkdf#computeHkdf(String, byte[], byte[],
   * byte[], int)}
   */
  private byte[] expand(byte[] prk, byte[] info, int length) throws GeneralSecurityException {
    Mac mac = Mac.getInstance(macAlgorithm);
    if (length > 255 * mac.getMacLength()) {
      throw new GeneralSecurityException("size too large");
    }
    byte[] result = new byte[length];
    int ctr = 1;
    int pos = 0;
    mac.init(new SecretKeySpec(prk, macAlgorithm));
    byte[] digest = new byte[0];
    while (true) {
      mac.update(digest);
      mac.update(info);
      mac.update((byte) ctr);
      digest = mac.doFinal();
      if (pos + digest.length < length) {
        System.arraycopy(digest, 0, result, pos, digest.length);
        pos += digest.length;
        ctr++;
      } else {
        System.arraycopy(digest, 0, result, pos, length - pos);
        break;
      }
    }
    return result;
  }

  @Override
  public byte[] labeledExtract(byte[] salt, byte[] ikm, String ikmLabel, byte[] suiteId)
      throws GeneralSecurityException {
    return extract(HpkeUtil.labelIkm(ikmLabel, ikm, suiteId), salt);
  }

  @Override
  public byte[] labeledExpand(byte[] prk, byte[] info, String infoLabel, byte[] suiteId, int length)
      throws GeneralSecurityException {
    return expand(prk, HpkeUtil.labelInfo(infoLabel, info, suiteId, length), length);
  }

  @Override
  public byte[] extractAndExpand(
      byte[] salt,
      byte[] ikm,
      String ikmLabel,
      byte[] info,
      String infoLabel,
      byte[] suiteId,
      int length)
      throws GeneralSecurityException {
    byte[] prk = extract(HpkeUtil.labelIkm(ikmLabel, ikm, suiteId), salt);
    return expand(prk, HpkeUtil.labelInfo(infoLabel, info, suiteId, length), length);
  }

  @Override
  public byte[] getKdfId() throws GeneralSecurityException {
    switch (macAlgorithm) {
      case "HmacSha256":
        return HpkeUtil.hkdfSha256KdfId;
      default:
        throw new GeneralSecurityException("Could not determine HPKE KDF ID");
    }
  }

  int getMacLength() throws GeneralSecurityException {
    return Mac.getInstance(macAlgorithm).getMacLength();
  }
}
