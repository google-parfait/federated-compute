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
import com.google.common.primitives.Bytes;

/**
 * The RFC 8452 test vectors for AEAD_AES_128_GCM_SIV: the example of Section 8 and the 24 vectors
 * of Appendix C.1 (https://www.rfc-editor.org/rfc/rfc8452.html#appendix-C.1), in that order.
 *
 * <p>Appendix C.2 (AES-256-GCM-SIV) is deliberately absent: this package only supports 16-byte
 * keys.
 */
final class AesGcmSivRfcTestVectors {

  /** One AEAD test vector. {@code ciphertext} and {@code tag} are separate, as in the RFC. */
  static final class TestVector {
    final byte[] key;
    final byte[] nonce;
    final byte[] aad;
    final byte[] plaintext;
    final byte[] ciphertext;
    final byte[] tag;

    TestVector(
        String key, String nonce, String aad, String plaintext, String ciphertext, String tag) {
      this.key = hex(key);
      this.nonce = hex(nonce);
      this.aad = hex(aad);
      this.plaintext = hex(plaintext);
      this.ciphertext = hex(ciphertext);
      this.tag = hex(tag);
    }

    /** {@code ciphertext || tag}: the output of {@link CallerProvidedNonceAead#encrypt}. */
    byte[] ciphertextAndTag() {
      return Bytes.concat(ciphertext, tag);
    }
  }

  static final TestVector[] aes128GcmSivVectors = {
    // RFC 8452, Section 8 (the example).
    new TestVector(
        /* key= */ "ee8e1ed9ff2540ae8f2ba9f50bc2f27c",
        /* nonce= */ "752abad3e0afb5f434dc4310",
        /* aad= */ "6578616d706c65",
        /* plaintext= */ "48656c6c6f20776f726c64",
        /* ciphertext= */ "5d349ead175ef6b1def6fd",
        /* tag= */ "4fbcdeb7e4793f4a1d7e4faa70100af1"),
    // RFC 8452, Appendix C.1, vector 1.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "",
        /* plaintext= */ "",
        /* ciphertext= */ "",
        /* tag= */ "dc20e2d83f25705bb49e439eca56de25"),
    // RFC 8452, Appendix C.1, vector 2.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "",
        /* plaintext= */ "0100000000000000",
        /* ciphertext= */ "b5d839330ac7b786",
        /* tag= */ "578782fff6013b815b287c22493a364c"),
    // RFC 8452, Appendix C.1, vector 3.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "",
        /* plaintext= */ "010000000000000000000000",
        /* ciphertext= */ "7323ea61d05932260047d942",
        /* tag= */ "a4978db357391a0bc4fdec8b0d106639"),
    // RFC 8452, Appendix C.1, vector 4.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "",
        /* plaintext= */ "01000000000000000000000000000000",
        /* ciphertext= */ "743f7c8077ab25f8624e2e948579cf77",
        /* tag= */ "303aaf90f6fe21199c6068577437a0c4"),
    // RFC 8452, Appendix C.1, vector 5.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "",
        /* plaintext= */ "0100000000000000000000000000000002000000000000000000000000000000",
        /* ciphertext= */ "84e07e62ba83a6585417245d7ec413a9fe427d6315c09b57ce45f2e3936a9445",
        /* tag= */ "1a8e45dcd4578c667cd86847bf6155ff"),
    // RFC 8452, Appendix C.1, vector 6.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "",
        /* plaintext= */ "010000000000000000000000000000000200000000000000"
            + "000000000000000003000000000000000000000000000000",
        /* ciphertext= */ "3fd24ce1f5a67b75bf2351f181a475c7b800a5b4d3dcf701"
            + "06b1eea82fa1d64df42bf7226122fa92e17a40eeaac1201b",
        /* tag= */ "5e6e311dbf395d35b0fe39c2714388f8"),
    // RFC 8452, Appendix C.1, vector 7.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "",
        /* plaintext= */ "0100000000000000000000000000000002000000000000000000000000000000"
            + "0300000000000000000000000000000004000000000000000000000000000000",
        /* ciphertext= */ "2433668f1058190f6d43e360f4f35cd8e475127cfca7028ea8ab5c20f7ab2af0"
            + "2516a2bdcbc08d521be37ff28c152bba36697f25b4cd169c6590d1dd39566d3f",
        /* tag= */ "8a263dd317aa88d56bdf3936dba75bb8"),
    // RFC 8452, Appendix C.1, vector 8.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "01",
        /* plaintext= */ "0200000000000000",
        /* ciphertext= */ "1e6daba35669f427",
        /* tag= */ "3b0a1a2560969cdf790d99759abd1508"),
    // RFC 8452, Appendix C.1, vector 9.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "01",
        /* plaintext= */ "020000000000000000000000",
        /* ciphertext= */ "296c7889fd99f41917f44620",
        /* tag= */ "08299c5102745aaa3a0c469fad9e075a"),
    // RFC 8452, Appendix C.1, vector 10.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "01",
        /* plaintext= */ "02000000000000000000000000000000",
        /* ciphertext= */ "e2b0c5da79a901c1745f700525cb335b",
        /* tag= */ "8f8936ec039e4e4bb97ebd8c4457441f"),
    // RFC 8452, Appendix C.1, vector 11.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "01",
        /* plaintext= */ "0200000000000000000000000000000003000000000000000000000000000000",
        /* ciphertext= */ "620048ef3c1e73e57e02bb8562c416a319e73e4caac8e96a1ecb2933145a1d71",
        /* tag= */ "e6af6a7f87287da059a71684ed3498e1"),
    // RFC 8452, Appendix C.1, vector 12.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "01",
        /* plaintext= */ "020000000000000000000000000000000300000000000000"
            + "000000000000000004000000000000000000000000000000",
        /* ciphertext= */ "50c8303ea93925d64090d07bd109dfd9515a5a33431019c1"
            + "7d93465999a8b0053201d723120a8562b838cdff25bf9d1e",
        /* tag= */ "6a8cc3865f76897c2e4b245cf31c51f2"),
    // RFC 8452, Appendix C.1, vector 13.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "01",
        /* plaintext= */ "0200000000000000000000000000000003000000000000000000000000000000"
            + "0400000000000000000000000000000005000000000000000000000000000000",
        /* ciphertext= */ "2f5c64059db55ee0fb847ed513003746aca4e61c711b5de2e7a77ffd02da42fe"
            + "ec601910d3467bb8b36ebbaebce5fba30d36c95f48a3e7980f0e7ac299332a80",
        /* tag= */ "cdc46ae475563de037001ef84ae21744"),
    // RFC 8452, Appendix C.1, vector 14.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "010000000000000000000000",
        /* plaintext= */ "02000000",
        /* ciphertext= */ "a8fe3e87",
        /* tag= */ "07eb1f84fb28f8cb73de8e99e2f48a14"),
    // RFC 8452, Appendix C.1, vector 15.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "010000000000000000000000000000000200",
        /* plaintext= */ "0300000000000000000000000000000004000000",
        /* ciphertext= */ "6bb0fecf5ded9b77f902c7d5da236a4391dd0297",
        /* tag= */ "24afc9805e976f451e6d87f6fe106514"),
    // RFC 8452, Appendix C.1, vector 16.
    new TestVector(
        /* key= */ "01000000000000000000000000000000",
        /* nonce= */ "030000000000000000000000",
        /* aad= */ "0100000000000000000000000000000002000000",
        /* plaintext= */ "030000000000000000000000000000000400",
        /* ciphertext= */ "44d0aaf6fb2f1f34add5e8064e83e12a2ada",
        /* tag= */ "bff9b2ef00fb47920cc72a0c0f13b9fd"),
    // RFC 8452, Appendix C.1, vector 17.
    new TestVector(
        /* key= */ "e66021d5eb8e4f4066d4adb9c33560e4",
        /* nonce= */ "f46e44bb3da0015c94f70887",
        /* aad= */ "",
        /* plaintext= */ "",
        /* ciphertext= */ "",
        /* tag= */ "a4194b79071b01a87d65f706e3949578"),
    // RFC 8452, Appendix C.1, vector 18.
    new TestVector(
        /* key= */ "36864200e0eaf5284d884a0e77d31646",
        /* nonce= */ "bae8e37fc83441b16034566b",
        /* aad= */ "46bb91c3c5",
        /* plaintext= */ "7a806c",
        /* ciphertext= */ "af60eb",
        /* tag= */ "711bd85bc1e4d3e0a462e074eea428a8"),
    // RFC 8452, Appendix C.1, vector 19.
    new TestVector(
        /* key= */ "aedb64a6c590bc84d1a5e269e4b47801",
        /* nonce= */ "afc0577e34699b9e671fdd4f",
        /* aad= */ "fc880c94a95198874296",
        /* plaintext= */ "bdc66f146545",
        /* ciphertext= */ "bb93a3e34d3c",
        /* tag= */ "d6a9c45545cfc11f03ad743dba20f966"),
    // RFC 8452, Appendix C.1, vector 20.
    new TestVector(
        /* key= */ "d5cc1fd161320b6920ce07787f86743b",
        /* nonce= */ "275d1ab32f6d1f0434d8848c",
        /* aad= */ "046787f3ea22c127aaf195d1894728",
        /* plaintext= */ "1177441f195495860f",
        /* ciphertext= */ "4f37281f7ad12949d0",
        /* tag= */ "1d02fd0cd174c84fc5dae2f60f52fd2b"),
    // RFC 8452, Appendix C.1, vector 21.
    new TestVector(
        /* key= */ "b3fed1473c528b8426a582995929a149",
        /* nonce= */ "9e9ad8780c8d63d0ab4149c0",
        /* aad= */ "c9882e5386fd9f92ec489c8fde2be2cf97e74e93",
        /* plaintext= */ "9f572c614b4745914474e7c7",
        /* ciphertext= */ "f54673c5ddf710c745641c8b",
        /* tag= */ "c1dc2f871fb7561da1286e655e24b7b0"),
    // RFC 8452, Appendix C.1, vector 22.
    new TestVector(
        /* key= */ "2d4ed87da44102952ef94b02b805249b",
        /* nonce= */ "ac80e6f61455bfac8308a2d4",
        /* aad= */ "2950a70d5a1db2316fd568378da107b52b0da55210cc1c1b0a",
        /* plaintext= */ "0d8c8451178082355c9e940fea2f58",
        /* ciphertext= */ "c9ff545e07b88a015f05b274540aa1",
        /* tag= */ "83b3449b9f39552de99dc214a1190b0b"),
    // RFC 8452, Appendix C.1, vector 23.
    new TestVector(
        /* key= */ "bde3b2f204d1e9f8b06bc47f9745b3d1",
        /* nonce= */ "ae06556fb6aa7890bebc18fe",
        /* aad= */ "1860f762ebfbd08284e421702de0de18baa9c9596291b08466f37de21c7f",
        /* plaintext= */ "6b3db4da3d57aa94842b9803a96e07fb6de7",
        /* ciphertext= */ "6298b296e24e8cc35dce0bed484b7f30d580",
        /* tag= */ "3e377094f04709f64d7b985310a4db84"),
    // RFC 8452, Appendix C.1, vector 24.
    new TestVector(
        /* key= */ "f901cfe8a69615a93fdf7a98cad48179",
        /* nonce= */ "6245709fb18853f68d833640",
        /* aad= */ "7576f7028ec6eb5ea7e298342a94d4b202b370ef9768ec6561c4fe6b7e7296fa859c21",
        /* plaintext= */ "e42a3c02c25b64869e146d7b233987bddfc240871d",
        /* ciphertext= */ "391cc328d484a4f46406181bcd62efd9b3ee197d05",
        /* tag= */ "2d15506c84a9edd65e13e9d24a2a6e70")
  };

  /**
   * One representative vector for tests that only need a single key, nonce, non-empty plaintext and
   * non-empty associated data: Appendix C.1, vector 18.
   */
  static TestVector nonEdgeCaseVector() {
    return aes128GcmSivVectors[18];
  }

  /** One vector with empty associated data (and a non-empty plaintext): Appendix C.1, vector 5. */
  static TestVector vectorWithEmptyAad() {
    return aes128GcmSivVectors[5];
  }

  static byte[] hex(String hex) {
    return BaseEncoding.base16().lowerCase().decode(hex);
  }

  private AesGcmSivRfcTestVectors() {}
}
