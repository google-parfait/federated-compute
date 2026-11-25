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

import com.google.common.io.BaseEncoding;

/**
 * RFC 9180, Appendix A.1: DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM, Base Setup, shared
 * by the HPKE tests of this package.
 */
final class HpkeRfcTestVectors {

  static byte[] hex(String hex) {
    return BaseEncoding.base16().lowerCase().decode(hex);
  }

  // A.1.1 Base Setup Information.
  static final byte[] a1Info = hex("4f6465206f6e2061204772656369616e2055726e");
  static final byte[] a1SkE =
      hex("52c4a758a802cd8b936eceea314432798d5baf2d7e9235dc084ab1b9cfa2f736");
  static final byte[] a1PkE =
      hex("37fda3567bdbd628e88668c3c8d7e97d1d1253b6d4ea6d44c150f741f1bf4431");
  static final byte[] a1SkR =
      hex("4612c550263fc8ad58375df3f557aac531d26850903e55a9f23f21d8534e8ac8");
  static final byte[] a1PkR =
      hex("3948cfe0ad1ddb695d780e59077195da6c56506b027329794ab02bca80815c4d");
  static final byte[] a1SharedSecret =
      hex("fe0e18c9f024ce43799ae393c7e8fe8fce9d218875e8227b0187c04e7d2ea1fc");
  static final byte[] a1Key = hex("4531685d41d65f03dc48f6b8302c05b0");
  static final byte[] a1BaseNonce = hex("56d890e5accaaf011cff4b7d");

  // A.1.1.1 Encryptions: the plaintext is the same for every sequence number.
  static final byte[] a1Pt = hex("4265617574792069732074727574682c20747275746820626561757479");

  /** One entry of A.1.1.1: the AEAD inputs and output for a given sequence number. */
  static final class Encryption {
    final int sequenceNumber;
    final byte[] aad;
    final byte[] nonce;
    final byte[] ct;

    Encryption(int sequenceNumber, String aad, String nonce, String ct) {
      this.sequenceNumber = sequenceNumber;
      this.aad = hex(aad);
      this.nonce = hex(nonce);
      this.ct = hex(ct);
    }
  }

  static final Encryption[] a1Encryptions = {
    new Encryption(
        0,
        "436f756e742d30",
        "56d890e5accaaf011cff4b7d",
        "f938558b5d72f1a23810b4be2ab4f84331acc02fc97babc53a52ae8218a355a96d8770ac83d07bea87e13c512a"),
    new Encryption(
        1,
        "436f756e742d31",
        "56d890e5accaaf011cff4b7c",
        "af2d7e9ac9ae7e270f46ba1f975be53c09f8d875bdc8535458c2494e8a6eab251c03d0c22a56b8ca42c2063b84"),
    new Encryption(
        2,
        "436f756e742d32",
        "56d890e5accaaf011cff4b7f",
        "498dfcabd92e8acedc281e85af1cb4e3e31c7dc394a1ca20e173cb72516491588d96a19ad4a683518973dcc180"),
    new Encryption(
        4,
        "436f756e742d34",
        "56d890e5accaaf011cff4b79",
        "583bd32bc67a5994bb8ceaca813d369bca7b2a42408cddef5e22f880b631215a09fc0012bc69fccaa251c0246d"),
    new Encryption(
        255,
        "436f756e742d323535",
        "56d890e5accaaf011cff4b82",
        "7175db9717964058640a3a11fb9007941a5d1757fda1a6935c805c21af32505bf106deefec4a49ac38d71c9e0a"),
    new Encryption(
        256,
        "436f756e742d323536",
        "56d890e5accaaf011cff4a7d",
        "957f9800542b0b8891badb026d79cc54597cb2d225b54c00c5238c25d05c30e3fbeda97d2e0e1aba483a2df9f2"),
  };

  private HpkeRfcTestVectors() {}
}
