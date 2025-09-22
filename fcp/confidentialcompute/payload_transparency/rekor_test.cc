/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fcp/confidentialcompute/payload_transparency/rekor.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/fixed_array.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/confidentialcompute/payload_transparency/signatures.h"
#include "fcp/protos/confidentialcompute/key.pb.h"
#include "fcp/protos/confidentialcompute/payload_transparency.pb.h"
#include "fcp/testing/testing.h"

namespace fcp::confidential_compute::payload_transparency {
namespace {

using ::fcp::confidentialcompute::Key;
using ::fcp::confidentialcompute::RekorLogEntry;
using ::testing::HasSubstr;

// Many of these tests use a small Merkle tree containing 11 leaf nodes; this
// size is useful for testing a mix of left- and right-subtrees in a tree that
// is not perfect.
//
// Root hash (base 64): 8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=
//
// f3e997dd35a309d6503f20c75887e81765cae1c2ffb35d4e8d34c2b7ab556c25
//  ├ 2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe
//  │ ├ f625e204b44ef052ae52ec92695dd9648b83a9b601ba6cf76205124d522a8e15
//  │ │ ├ 1e303f9175115d62ccddb195586b1893fb074b00edd6ac1935ce666afdba7ced
//  │ │ │ ├ 35f81a9aaecb6df85b36b559eb660c2bbd5be628269c352b66f25c9b163dfa6e
//  │ │ │ │ ("leaf 0")
//  │ │ │ └ 92fb1f3d84bef84d8f14f7055c3c0a9d0a882811e51489f6161fb995160193f0
//  │ │ │   ("leaf 1")
//  │ │ └ dfa09bb23f87a6729fd0728a13b22141adb3da53c706b4fac0ac8edf19d1fafa
//  │ │   ├ 73c0841813b278270b17a5c16ec63f84fd1123af05169734031475d8a1a99d69
//  │ │   │ ("leaf 2")
//  │ │   └ 6ce162caa72042c992a84504891913b47eb4d830659a507adca8687bbb04b879
//  │ │     ("leaf 3")
//  │ └ c5ebc562f49755eccc26679d2aa22a700c76311a2e97b0be0c3c3e7a5c502786
//  │   ├ d5d85c38cf23702cc8015fc104abbfe6c1085a0b8ad031d4c679fb62492e9197
//  │   │ ├ 97fd43db9b236927e7921731ed75c06796047ab78b3e64b0ae0d50f96193d4e8
//  │   │ │ ("leaf 4")
//  │   │ └ 995681569fbf97aca7697ab645356b0834d0ce7498d8b822b060ede1d914f648
//  │   │   ("leaf 5")
//  │   └ e7f2620cfc19210f52498d84198bd3eff6dee1c8807710e3cae284d1d0f68a2f
//  │     ├ 5642602157cc4d5b6f09dfe6c3122c12a620d922dc1e8579dfa52c7a7fec0668
//  │     │ ("leaf 6")
//  │     └ 7c97b4c70177d4afca39410315b06776023fc3fbac987bb9b37775c6dc6078fb
//  │       ("leaf 7")
//  └ 23e18ca02de317968435d21756f6ff8189abbb44b864f5357264d100d4e44a97
//    ├ 3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476
//    │ ├ aedfc622b8d17454fa7318fcf4c65979484e6d8dbb7d07d175d1b2687e656ed8
//    │ │ ("leaf 8")
//    │ └ 8e70aedc03a3784c0fa0988dde490ab7b44dec3d4894c52d35d660e5e947d806
//    │   ("leaf 9")
//    └ c0e6be9a317ce82b482b91ac12c7a90555208bdfc6741f5f67207e2a7593ecc6
//      ("leaf 10")
constexpr std::array kLeaves = {
    // "leaf 0"
    "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\""
    "hash\":{\"algorithm\":\"sha256\",\"value\":\"20e325f06280f9d0d193fed01a0ed"
    "a5bef79063f2e602d93e3605cbe825d96ad\"}},\"signature\":{\"content\":\"MEYCI"
    "QCIBNw3Y/epRrQws36IqLbm2P37wjcrHOeMtngWJTjYwQIhAJudknwUm3+dGFpsaBW9ytHioRA"
    "RkE50vp37MKi3ODwN\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS"
    "0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFOC8rNTdmWWh1YUM"
    "4Vit1TUZPOWtkVEIzM05TWgo1cS9IYk1XY01FeS8wZGNYRnlXbmFqZi9Kdy8rQTlwa3RVc1h0K"
    "2k2Wit5TXhZOGp1YXRIaENPM3BBPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t\"}}}}",

    // "leaf 1"
    "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\""
    "hash\":{\"algorithm\":\"sha256\",\"value\":\"ccbf76d20974e563eb51d22ff1171"
    "a30472e0ae643b17863befd53614e7fefad\"}},\"signature\":{\"content\":\"MEQCI"
    "DhkrhjIjVl0hvxKObDHB+es2JOezJuWHQAwEoNAbewrAiALPupUcwb1LOrfzqFTL6rMu+T/NU1"
    "BJ2kWXcd3tL6n7A==\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS"
    "0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFWlhJaFNCeFEyalh"
    "SZzhHazlMUjliZjZGM28yYQo2L1VnYitOS1dWR3h3S3YxZU92UDYzTWlOYVNaQTY5VjM0eGtrY"
    "UpsVm1sMFJzMXVMZkwxWERtL0hnPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t\"}}}}",

    // "leaf 2"
    "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\""
    "hash\":{\"algorithm\":\"sha256\",\"value\":\"bac57df66fe6368188d1d4521bcff"
    "aecee76a03a50ff297a13439f7164de0a5f\"}},\"signature\":{\"content\":\"MEUCI"
    "QCglHPfDsyPz0g7qN9VuWl3YPaLBUJrT+DmUDV1wChfmQIgKJ2EQkZkiy3TwWtJ7JfJDJCIWet"
    "7Bk4l66nKIjSCueg=\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS"
    "0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFOC8rNTdmWWh1YUM"
    "4Vit1TUZPOWtkVEIzM05TWgo1cS9IYk1XY01FeS8wZGNYRnlXbmFqZi9Kdy8rQTlwa3RVc1h0K"
    "2k2Wit5TXhZOGp1YXRIaENPM3BBPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t\"}}}}",

    // "leaf 3"
    "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\""
    "hash\":{\"algorithm\":\"sha256\",\"value\":\"ca6e6588d55d58a70e0b4de60c2da"
    "b1e4574bb97d68fa88679852a5daaa9db02\"}},\"signature\":{\"content\":\"MEQCI"
    "EqY1KtqUduFNDIM5HnyVH/tKeWTPjkbA2Vxg5w6vyfWAiAQxnKAcON1gezZGLbHNlrdKjXC5gO"
    "9FPQqh18/aG5rVA==\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS"
    "0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFWlhJaFNCeFEyalh"
    "SZzhHazlMUjliZjZGM28yYQo2L1VnYitOS1dWR3h3S3YxZU92UDYzTWlOYVNaQTY5VjM0eGtrY"
    "UpsVm1sMFJzMXVMZkwxWERtL0hnPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t\"}}}}",

    // "leaf 4"
    "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\""
    "hash\":{\"algorithm\":\"sha256\",\"value\":\"5036d5415fd89f22c593f3a7a1034"
    "8af3c87b3f13d73373a42f8768e377da3e9\"}},\"signature\":{\"content\":\"MEQCI"
    "DFNHqL3m0zlDr8++45jvujQ0u4q6+dy74VB8jHIhL5qAiBqgbqu09mdSMZVI++XTr7xNuC8xin"
    "nOYIExstLuli6/g==\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS"
    "0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFOC8rNTdmWWh1YUM"
    "4Vit1TUZPOWtkVEIzM05TWgo1cS9IYk1XY01FeS8wZGNYRnlXbmFqZi9Kdy8rQTlwa3RVc1h0K"
    "2k2Wit5TXhZOGp1YXRIaENPM3BBPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t\"}}}}",

    // "leaf 5"
    "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\""
    "hash\":{\"algorithm\":\"sha256\",\"value\":\"5df183a7928a0b9a8a923c39009e8"
    "9847ab5f53d07547a362bd2b30e26ee2e16\"}},\"signature\":{\"content\":\"MEUCI"
    "QDxzsRvijGh5n4EykMBO6XH2je/0FxRrPNHnHg3Z33DlQIgcgwi5MXVSxf0zFOSV87Y3qekrz2"
    "10ac66IAR/KOfjZs=\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS"
    "0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFWlhJaFNCeFEyalh"
    "SZzhHazlMUjliZjZGM28yYQo2L1VnYitOS1dWR3h3S3YxZU92UDYzTWlOYVNaQTY5VjM0eGtrY"
    "UpsVm1sMFJzMXVMZkwxWERtL0hnPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t\"}}}}",

    // "leaf 6"
    "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\""
    "hash\":{\"algorithm\":\"sha256\",\"value\":\"e94f5850717db06bc4e0bd7444389"
    "dd11eb57caf8e60502cc633081a636510d0\"}},\"signature\":{\"content\":\"MEYCI"
    "QCGIajPjCyki9b2j+vOw3ZPE//Ifd0Istb8+UewNGuANgIhAIu2xWtkZz44p+0y4TIrde7L2OG"
    "cJqhbLKbvjRpA1K2S\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS"
    "0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFOC8rNTdmWWh1YUM"
    "4Vit1TUZPOWtkVEIzM05TWgo1cS9IYk1XY01FeS8wZGNYRnlXbmFqZi9Kdy8rQTlwa3RVc1h0K"
    "2k2Wit5TXhZOGp1YXRIaENPM3BBPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t\"}}}}",

    // "leaf 7"
    "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\""
    "hash\":{\"algorithm\":\"sha256\",\"value\":\"3fc6f659908e60026f20aeb6f1a90"
    "a14cacb8a27d870aec882a0529f2c829bb2\"}},\"signature\":{\"content\":\"MEUCI"
    "GTUxt6wDhOP1dQFJ30n5HbgQ6uv6bdEaHYvo+rAzFshAiEAmqA5j6ZnP40/guzZLcnZhvOUb3f"
    "ngxZk5FuXPF/kDdo=\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS"
    "0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFWlhJaFNCeFEyalh"
    "SZzhHazlMUjliZjZGM28yYQo2L1VnYitOS1dWR3h3S3YxZU92UDYzTWlOYVNaQTY5VjM0eGtrY"
    "UpsVm1sMFJzMXVMZkwxWERtL0hnPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t\"}}}}",

    // "leaf 8"
    "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\""
    "hash\":{\"algorithm\":\"sha256\",\"value\":\"3779790b00cf4dd355ba895adf368"
    "38158baa398f0e70fce7018503dbade253d\"}},\"signature\":{\"content\":\"MEUCI"
    "Fy+2n1e9Cp5SFxpR8etJa7i/NpUHZhA8Zk7yqCGgB1EAiEAl9HBuEum6BXLyXmb7x3mS1cHU3G"
    "I2FuVpwH6ZKItXcg=\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS"
    "0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFOC8rNTdmWWh1YUM"
    "4Vit1TUZPOWtkVEIzM05TWgo1cS9IYk1XY01FeS8wZGNYRnlXbmFqZi9Kdy8rQTlwa3RVc1h0K"
    "2k2Wit5TXhZOGp1YXRIaENPM3BBPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t\"}}}}",

    // "leaf 9"
    "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\""
    "hash\":{\"algorithm\":\"sha256\",\"value\":\"3327cf579337d5d2469ca9960f066"
    "abf8073645ce487d2a350f7bd7e44b7d52d\"}},\"signature\":{\"content\":\"MEQCI"
    "AtvdPzoXPggO3e3NCuJu7RD6nImciqJ4MRv1P0DA+MqAiBqvPJXMDNsyW54E9Py4/7sSU2BvLX"
    "nYNQHjK7s3Ofvvg==\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS"
    "0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFWlhJaFNCeFEyalh"
    "SZzhHazlMUjliZjZGM28yYQo2L1VnYitOS1dWR3h3S3YxZU92UDYzTWlOYVNaQTY5VjM0eGtrY"
    "UpsVm1sMFJzMXVMZkwxWERtL0hnPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t\"}}}}",

    // "leaf 10"
    "{\"apiVersion\":\"0.0.1\",\"kind\":\"hashedrekord\",\"spec\":{\"data\":{\""
    "hash\":{\"algorithm\":\"sha256\",\"value\":\"b789e63ed0a8dd152a5b020b53e0c"
    "9d7a2522b93702c255d2847575ebea98a92\"}},\"signature\":{\"content\":\"MEYCI"
    "QDyifHM2YsfHTnw+CZoEZeO9ELWadxfORWL0P6dCh/OPQIhAIklmtzK/g0+cwSjRviNp4LFn5k"
    "a2AvMjA4bvJJXMMau\",\"publicKey\":{\"content\":\"LS0tLS1CRUdJTiBQVUJMSUMgS"
    "0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFOC8rNTdmWWh1YUM"
    "4Vit1TUZPOWtkVEIzM05TWgo1cS9IYk1XY01FeS8wZGNYRnlXbmFqZi9Kdy8rQTlwa3RVc1h0K"
    "2k2Wit5TXhZOGp1YXRIaENPM3BBPT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t\"}}}}",
};

// Converts a hex string to bytes. This function is intended to ease the
// migration to the version of HexStringToBytes introduced in abseil 20240722.0.
bool HexStringToBytes(absl::string_view hex, std::string* bytes) {
  *bytes = absl::HexStringToBytes(hex);
  return true;
}

// The key used to sign even-index log entries.
Key GetEvenVerifyingKey() {
  Key key;
  key.set_algorithm(Key::ECDSA_P256);
  key.set_purpose(Key::VERIFY);
  CHECK(HexStringToBytes(
      "04f3ffb9edf621b9a0bc57eb8c14ef64753077dcd499e6afc76cc59c304cbfd1d7171725"
      "a76a37ff270ffe03da64b54b17b7e8ba67ec8cc58f23b9ab478423b7a4",
      key.mutable_key_material()));
  key.set_key_id("even");
  return key;
}

// The key used to sign odd-index log entries.
Key GetOddVerifyingKey() {
  Key key;
  key.set_algorithm(Key::ECDSA_P256);
  key.set_purpose(Key::VERIFY);
  CHECK(HexStringToBytes(
      "04657221481c50da35d183c1a4f4b47d6dfe85de8d9aebf5206fe34a5951b1c0abf578eb"
      "cfeb732235a49903af55df8c6491a26556697446cd6e2df2f55c39bf1e",
      key.mutable_key_material()));
  key.set_key_id("odd");
  return key;
}

// Signs a checkpoint string, returning the signature and verifying key.
std::tuple<std::string, Key> SignCheckpoint(absl::string_view checkpoint) {
  auto signer = EcdsaP256R1Signer::Create();
  std::string signature = signer.Sign(checkpoint);
  Key key;
  key.set_algorithm(Key::ECDSA_P256);
  key.set_purpose(Key::VERIFY);
  key.set_key_material(signer.GetPublicKey());
  key.set_key_id("key id");
  return {signature, key};
}

// Returns a SignedData function that emits the provided content.
std::function<void(absl::FunctionRef<void(absl::string_view)>)> BuildSignedData(
    absl::string_view content) {
  return [content](absl::FunctionRef<void(absl::string_view)> emitter) {
    emitter(content);
  };
}

// Leaf 0 is on the far left of the size 8 perfect subtree.
TEST(VerifyRekorLogEntryTest, Leaf0) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(kLeaves[0]);
  log_entry.set_log_index(0);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "92fb1f3d84bef84d8f14f7055c3c0a9d0a882811e51489f6161fb995160193f0",
           "dfa09bb23f87a6729fd0728a13b22141adb3da53c706b4fac0ac8edf19d1fafa",
           "c5ebc562f49755eccc26679d2aa22a700c76311a2e97b0be0c3c3e7a5c502786",
           "23e18ca02de317968435d21756f6ff8189abbb44b864f5357264d100d4e44a97",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  EXPECT_OK(VerifyRekorLogEntry(log_entry, {&verifying_key}, {&rekor_key},
                                BuildSignedData("leaf 0")));
}

// Leaf 5 is in the middle of the size 8 perfect subtree.
TEST(VerifyRekorLogEntryTest, Leaf5) {
  Key verifying_key = GetOddVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(kLeaves[5]);
  log_entry.set_log_index(5);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "97fd43db9b236927e7921731ed75c06796047ab78b3e64b0ae0d50f96193d4e8",
           "e7f2620cfc19210f52498d84198bd3eff6dee1c8807710e3cae284d1d0f68a2f",
           "f625e204b44ef052ae52ec92695dd9648b83a9b601ba6cf76205124d522a8e15",
           "23e18ca02de317968435d21756f6ff8189abbb44b864f5357264d100d4e44a97",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  EXPECT_OK(VerifyRekorLogEntry(log_entry, {&verifying_key}, {&rekor_key},
                                BuildSignedData("leaf 5")));
}

// Leaf 7 is on the right of the size 8 perfect subtree.
TEST(VerifyRekorLogEntryTest, Leaf7) {
  Key verifying_key = GetOddVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(kLeaves[7]);
  log_entry.set_log_index(7);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "5642602157cc4d5b6f09dfe6c3122c12a620d922dc1e8579dfa52c7a7fec0668",
           "d5d85c38cf23702cc8015fc104abbfe6c1085a0b8ad031d4c679fb62492e9197",
           "f625e204b44ef052ae52ec92695dd9648b83a9b601ba6cf76205124d522a8e15",
           "23e18ca02de317968435d21756f6ff8189abbb44b864f5357264d100d4e44a97",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  EXPECT_OK(VerifyRekorLogEntry(log_entry, {&verifying_key}, {&rekor_key},
                                BuildSignedData("leaf 7")));
}

// Leaf 8 is in the inside of the size 2 perfect subtree.
TEST(VerifyRekorLogEntryTest, Leaf8) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(kLeaves[8]);
  log_entry.set_log_index(8);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "8e70aedc03a3784c0fa0988dde490ab7b44dec3d4894c52d35d660e5e947d806",
           "c0e6be9a317ce82b482b91ac12c7a90555208bdfc6741f5f67207e2a7593ecc6",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  EXPECT_OK(VerifyRekorLogEntry(log_entry, {&verifying_key}, {&rekor_key},
                                BuildSignedData("leaf 8")));
}

// Leaf 10 is in the size 1 perfect subtree.
TEST(VerifyRekorLogEntryTest, Leaf10) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(kLeaves[10]);
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  EXPECT_OK(VerifyRekorLogEntry(log_entry, {&verifying_key}, {&rekor_key},
                                BuildSignedData("leaf 10")));
}

TEST(VerifyRekorLogEntryTest, SingleElementTree) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n1\nNfgamq7LbfhbNrVZ62YMK71b5igmnDUrZvJcmxY9+m4=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(kLeaves[0]);
  log_entry.set_log_index(0);
  log_entry.set_tree_size(1);
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  EXPECT_OK(VerifyRekorLogEntry(log_entry, {&verifying_key}, {&rekor_key},
                                BuildSignedData("leaf 0")));
}

TEST(VerifyRekorLogEntryTest, CheckpointOtherContents) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=\n"
      "extra 1\nextra 2\n");

  RekorLogEntry log_entry;
  log_entry.set_body(kLeaves[10]);
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.add_checkpoint_other_contents("extra 1");
  log_entry.add_checkpoint_other_contents("extra 2");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  EXPECT_OK(VerifyRekorLogEntry(log_entry, {&verifying_key}, {&rekor_key},
                                BuildSignedData("leaf 10")));
}

TEST(VerifyRekorLogEntryTest, UnsupportedLogEntryKind) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\nLCHRxFbKsvf4Jzr+px0Pqd4reWgwC+dRl4VcdjkLBQk=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(
      absl::StrReplaceAll(kLeaves[10], {{"hashedrekord", "dsse"}}));
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  absl::Status status = VerifyRekorLogEntry(
      log_entry, {&verifying_key}, {&rekor_key}, BuildSignedData("leaf 10"));
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("unsupported log entry kind"));
}

TEST(VerifyRekorLogEntryTest, MissingDataHash) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n/VSaEwzcUfM0Be3XuW7MM8QQmOOMc2/C6GFQ78unNpE=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(absl::StrReplaceAll(
      kLeaves[10],
      {{",\"value\":\"b789e63ed0a8dd152a5b020b53e0c9d7a2522b93702c255d2847575eb"
        "ea98a92\"",
        ""}}));
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  absl::Status status = VerifyRekorLogEntry(
      log_entry, {&verifying_key}, {&rekor_key}, BuildSignedData("leaf 10"));
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("log entry is missing data hash"));
}

TEST(VerifyRekorLogEntryTest, InvalidDataHashEncoding) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\nN84K5iEwOrqXSfGdSM/cJEjE4FnBrugXnDwzFpCXG/U=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(absl::StrReplaceAll(
      kLeaves[10],
      {{"b789e63ed0a8dd152a5b020b53e0c9d7a2522b93702c255d2847575ebea98a92",
        "invalid base16"}}));
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  absl::Status status = VerifyRekorLogEntry(
      log_entry, {&verifying_key}, {&rekor_key}, BuildSignedData("leaf 10"));
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("hash does not match payload"));
}

TEST(VerifyRekorLogEntryTest, MissmatchedDataHash) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n3BtuumfZR9RqgbP0NEvON8vVk5XcHeNkE+8FcmzPDhk=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(absl::StrReplaceAll(
      kLeaves[10],
      {{"b789e63ed0a8dd152a5b020b53e0c9d7a2522b93702c255d2847575ebea98a92",
        "0000000000000000000000000000000000000000000000000000000000000000"}}));
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  absl::Status status = VerifyRekorLogEntry(
      log_entry, {&verifying_key}, {&rekor_key}, BuildSignedData("leaf 10"));
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(),
              HasSubstr("log entry data hash does not match payload"));
}

TEST(VerifyRekorLogEntryTest, MissingSignature) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n9XUVBPhvc+eDGGyhi5wymRxas5oPyO2BRSgdsouEoSQ=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(absl::StrReplaceAll(
      kLeaves[10],
      {{"\"content\":\"MEYCIQDyifHM2YsfHTnw+CZoEZeO9ELWadxfORWL0P6dCh/OPQIhAIkl"
        "mtzK/g0+cwSjRviNp4LFn5ka2AvMjA4bvJJXMMau\",",
        ""}}));
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  absl::Status status = VerifyRekorLogEntry(
      log_entry, {&verifying_key}, {&rekor_key}, BuildSignedData("leaf 10"));
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("log entry is missing signature"));
}

TEST(VerifyRekorLogEntryTest, InvalidSignatureEncoding) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n2pObl5Tee5UwYch1FsUM1gHihTLC3hg94LdqSRqDdEA=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(absl::StrReplaceAll(
      kLeaves[10],
      {{"MEYCIQDyifHM2YsfHTnw+CZoEZeO9ELWadxfORWL0P6dCh/OPQIhAIklmtzK/g0+cwSjRv"
        "iNp4LFn5ka2AvMjA4bvJJXMMau",
        "invalid base64!"}}));
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  absl::Status status = VerifyRekorLogEntry(
      log_entry, {&verifying_key}, {&rekor_key}, BuildSignedData("leaf 10"));
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(),
              HasSubstr("signature is not a valid base64 string"));
}

TEST(VerifyRekorLogEntryTest, MissmatchedSignature) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\nQyAqzby0n2ekpgChOfnecBtbiBQKPO5qFKMMx9g4dZQ=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(absl::StrReplaceAll(
      kLeaves[10],
      {{"MEYCIQDyifHM2YsfHTnw+CZoEZeO9ELWadxfORWL0P6dCh/OPQIhAIklmtzK/g0+cwSjRv"
        "iNp4LFn5ka2AvMjA4bvJJXMMau",
        "MEYCIQCIBNw3Y/epRrQws36IqLbm2P37wjcrHOeMtngWJTjYwQIhAJudknwUm3+dGFpsaB"
        "W9ytHioRARkE50vp37MKi3ODwN"}}));
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  absl::Status status = VerifyRekorLogEntry(
      log_entry, {&verifying_key}, {&rekor_key}, BuildSignedData("leaf 10"));
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("log entry signature is invalid"));
}

TEST(VerifyRekorLogEntryTest, LeafIndexTooLarge) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(kLeaves[10]);
  log_entry.set_log_index(11);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  absl::Status status = VerifyRekorLogEntry(
      log_entry, {&verifying_key}, {&rekor_key}, BuildSignedData("leaf 10"));
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("leaf index larger than tree size"));

  // VerifyRekorLogEntry should also fail the log index is even larger.
  log_entry.set_log_index(12);
  EXPECT_THAT(VerifyRekorLogEntry(log_entry, {&verifying_key}, {&rekor_key},
                                  BuildSignedData("leaf 10"))
                  .message(),
              HasSubstr("leaf index larger than tree size"));
}

TEST(VerifyRekorLogEntryTest, WrongNumberOfHashes) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(kLeaves[10]);
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
           // Add an extra hash.
           "0000000000000000000000000000000000000000000000000000000000000000",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  absl::Status status = VerifyRekorLogEntry(
      log_entry, {&verifying_key}, {&rekor_key}, BuildSignedData("leaf 10"));
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("unexpected number of hashes"));

  // VerifyRekorLogEntry should also fail if there are too few hashes.
  log_entry.mutable_hashes()->RemoveLast();
  log_entry.mutable_hashes()->RemoveLast();
  EXPECT_THAT(VerifyRekorLogEntry(log_entry, {&verifying_key}, {&rekor_key},
                                  BuildSignedData("leaf 10"))
                  .message(),
              HasSubstr("unexpected number of hashes"));
}

TEST(VerifyRekorLogEntryTest, MissmatchedCheckpointSignature) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=\n");
  checkpoint_signature[0] ^= 0xff;

  RekorLogEntry log_entry;
  log_entry.set_body(kLeaves[10]);
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id());

  absl::Status status = VerifyRekorLogEntry(
      log_entry, {&verifying_key}, {&rekor_key}, BuildSignedData("leaf 10"));
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("checkpoint signature is invalid"));
}

TEST(VerifyRekorLogEntryTest, NoMatchingRekorKey) {
  Key verifying_key = GetEvenVerifyingKey();
  auto [checkpoint_signature, rekor_key] = SignCheckpoint(
      "origin\n11\n8+mX3TWjCdZQPyDHWIfoF2XK4cL/s11OjTTCt6tVbCU=\n");

  RekorLogEntry log_entry;
  log_entry.set_body(kLeaves[10]);
  log_entry.set_log_index(10);
  log_entry.set_tree_size(11);
  for (absl::string_view hash : {
           "3fe1a8f703995dc5db242e9040efabc666dbe615ebd5e3e4b751f610d4228476",
           "2044241076dd5d97acf82f027399fea9e2983655476e7ed581bbf0f86e256cbe",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("origin");
  log_entry.set_checkpoint_signature(checkpoint_signature);
  log_entry.set_checkpoint_signature_key_id(rekor_key.key_id() + "X");

  absl::Status status = VerifyRekorLogEntry(
      log_entry, {&verifying_key}, {&rekor_key}, BuildSignedData("leaf 10"));
  EXPECT_THAT(status, IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("checkpoint signature is invalid"));
}

// Verify an arbitrary real log entry:
// https://search.sigstore.dev/?uuid=108e9186e8c5677aa0d0ac75a6ef6647222cb925667b16943a2d30e321a8237ce62653e56d8eaf9a
// Note that since Rekor generates the inclusion proof on access, the tree size
// will be larger than what is specified below and some of the hashes will be
// different.
TEST(VerifyRekorLogEntryTest, RealLogEntry) {
  RekorLogEntry log_entry;
  ASSERT_TRUE(absl::Base64Unescape(
      "eyJhcGlWZXJzaW9uIjoiMC4wLjEiLCJraW5kIjoiaGFzaGVkcmVrb3JkIiwic3BlYyI6eyJk"
      "YXRhIjp7Imhhc2giOnsiYWxnb3JpdGhtIjoic2hhMjU2IiwidmFsdWUiOiJkM2NkZmExZGM4"
      "NmI5MWU4Y2E0MjkzMjdiMTQ0ZGE1MTRkYmMzMmIzODg4M2RmZjgzMDMzNmM5MTMyNmU3YTg3"
      "In19LCJzaWduYXR1cmUiOnsiY29udGVudCI6Ik1FWUNJUUNhM1hNWmpMN25TTmQ3TG42QnFG"
      "T1ZoN3E3aVAwWjNvUFZoQzEwZHFCaXBnSWhBUHNlTnI5aTl2Mjk1M3BvdXU2cnExWEdmMDNI"
      "VURWY1BKWkMvaXFjZWliYyIsInB1YmxpY0tleSI6eyJjb250ZW50IjoiTFMwdExTMUNSVWRK"
      "VGlCRFJWSlVTVVpKUTBGVVJTMHRMUzB0Q2sxSlNVTXJWRU5EUVc5RFowRjNTVUpCWjBsVlNW"
      "QXdWSFZITjFSRFNtRm9kVlJOYWtkVlEwOVZRbU5VYWtKSmQwTm5XVWxMYjFwSmVtb3dSVUYz"
      "VFhjS1RucEZWazFDVFVkQk1WVkZRMmhOVFdNeWJHNWpNMUoyWTIxVmRWcEhWakpOVWpSM1NF"
      "RlpSRlpSVVVSRmVGWjZZVmRrZW1SSE9YbGFVekZ3WW01U2JBcGpiVEZzV2tkc2FHUkhWWGRJ"
      "YUdOT1RXcFZkMDVFU1RWTmFrVjZUbFJWZVZkb1kwNU5hbFYzVGtSSk5VMXFSVEJPVkZWNVYy"
      "cEJRVTFHYTNkRmQxbElDa3R2V2tsNmFqQkRRVkZaU1V0dldrbDZhakJFUVZGalJGRm5RVVZx"
      "ZERaalpEbDRibWRRVld3M1NGWlpabEJMWTBoeWVtRkxlV3MwTmpCSGFVdDNOMllLVEV4MUwz"
      "WkNSalpJTlZsWVJtTTVSRkJHV0dKWE9EUTJNbEJGVFV4NUwwMXNiV3hQVkRoVWJ5dEJLMDUw"
      "VlRsNVMwdFBRMEZhT0hkblowZGlUVUUwUndwQk1WVmtSSGRGUWk5M1VVVkJkMGxJWjBSQlZF"
      "Sm5UbFpJVTFWRlJFUkJTMEpuWjNKQ1owVkdRbEZqUkVGNlFXUkNaMDVXU0ZFMFJVWm5VVlZP"
      "WlZSbkNrTXlRMWx0WjNBMmJFbERWazAyWkhwblVFRnpiVVp6ZDBoM1dVUldVakJxUWtKbmQw"
      "WnZRVlV6T1ZCd2VqRlphMFZhWWpWeFRtcHdTMFpYYVhocE5Ga0tXa1E0ZDFSM1dVUldVakJT"
      "UVZGSUwwSkZWWGRSTkVaQ1l6SnNibU16VW5aamJWVjBZMGhLZGxwRE1YZGpiVGwwV2xoU2Ix"
      "cFlWbnBNV0U1b1VVaENlUXBpTW5Cc1dUTlJkR050Vm5KaU0wbDFZVmRHZEV4dFpIcGFXRW95"
      "WVZkT2JGbFhUbXBpTTFaMVpFTTFhbUl5TUhkTFVWbExTM2RaUWtKQlIwUjJla0ZDQ2tGUlVX"
      "SmhTRkl3WTBoTk5reDVPV2haTWs1MlpGYzFNR041Tlc1aU1qbHVZa2RWZFZreU9YUk5RM05I"
      "UTJselIwRlJVVUpuTnpoM1FWRm5SVWhSZDJJS1lVaFNNR05JVFRaTWVUbG9XVEpPZG1SWE5U"
      "QmplVFZ1WWpJNWJtSkhWWFZaTWpsMFRVbEhTMEpuYjNKQ1owVkZRV1JhTlVGblVVTkNTSGRG"
      "WldkQ05BcEJTRmxCTTFRd2QyRnpZa2hGVkVwcVIxSTBZMjFYWXpOQmNVcExXSEpxWlZCTE15"
      "OW9OSEI1WjBNNGNEZHZORUZCUVVkWFp6TndURXRuUVVGQ1FVMUJDbEo2UWtaQmFVVkJOak5S"
      "VWpGemIxcHZiVlZXY21oaGVsVlNjemhSTDB4dk56TXpialZqYXpaSGVscEVZa3hPVXpZNVow"
      "TkpSelZtT1djM1dsUjBNVzhLWTJocFMxbElSRlZZZDJWVFRFWXdXVTUxZUVkM2FVNWFXR3hY"
      "U0dWMk5rTk5RVzlIUTBOeFIxTk5ORGxDUVUxRVFUSmpRVTFIVVVOTlJXRnVRV28yUVFwdVpU"
      "RlBOR1IwUldsRVJsaFVjaXQwV0hkVFdWbFVWMkV2TXpacmNGVTNZbkZ3ZG1FeFdXUm9RbmxK"
      "YjNwdlNHOVNPSGszYW1GUldWRlJTWGROYkhWV0NuaEVNbk4wUW5SRWFXdE1OSFJsTkZKSFYw"
      "dG1iRmwzYkdGUFpqSlphM0pTTjI5M1YwRkVXR3RRTWs1SlJYRjZhbXAzWW01U00xaGtkbloy"
      "ZUFvdExTMHRMVVZPUkNCRFJWSlVTVVpKUTBGVVJTMHRMUzB0Q2c9PSJ9fX19",
      log_entry.mutable_body()));
  log_entry.set_log_index(82490871);
  log_entry.set_tree_size(399434838);
  for (absl::string_view hash : {
           "51ebc29bd00b7b0f24e6d4e6853e05fc231616c41cdfd97eb24e3f2bfdcc5a2d",
           "0ecab23c92dc0bde40f675e1bc18550a8cc883c12c31db791ee53d1b1def246f",
           "35b0b6b9043090ff56e0f75777c85f2a86fd3889d08c288e6221e41c360fee51",
           "2586b74305c56519913c3ce242b6fcbf1b5ba5d419ac278067a6f99d97a15452",
           "13dc9bb6abeff21f7b8390bcb598a051c8781b3283a058a46749b4dea7fd12fd",
           "ba808cd3f35248dba7605e42e115a5caf81aa32918cb500d151af26c691ee66e",
           "a0a4d97d6435a5ce4d601c6b1a097375f6bacd05b4834b1448e6d4e2b13ac33a",
           "89345302123a5e712483f56b2cb5f5166d97c1d80d663fac0a55f51fecda7332",
           "deab4bcd4383d2687678e712eb5d3cd711f5ed38d1274198ba0ca85aa50f9570",
           "79aabc09c342862288e125b928f67d48f2d34557ae1918284f16a76ed1bdb14c",
           "68e99339405fd89514b4f94d6bb9dddbdfa36cff2fcd2a1382745829e118f7ff",
           "ac3a33834d71658ca51133d2c4b60e61ad5de2a27cbf365d89d54c8eecf77f45",
           "0b91a4b2e9d3de7f1a960b790dfe365264c6dbda3c712708988ffc7de981b177",
           "f4d809c66515e660c274db68114024d4d95c8b4411b9be391375f16c20919f13",
           "247edb1a9f562f448bab9b3875c62c2e6148e1508ef02d42323c72b5ba4f48e9",
           "6493fb2a66066f6f13bcde4ff948be5b633895a47d4177a77e0b915887b7dc47",
           "6f60449a7b0bd3e6c2e7d5a320e49d59c48107396349602a0817f9f753906c10",
           "cbeff6c02655ce6c176a676e7a0e242eb975bfbc7dcb78aaf4f5684d0820d02e",
           "7cb3943f1dc7163b764a5cad4410a992196cc3f095d6b50cbc07f9537f6c88c4",
           "c7ed1957bb002a839969e7015e2a012fb080a1290737612c89d865af6fad4289",
           "a12b57d38210c3ffb0cf30338833c51c7bf72328322efb4c431891c797616128",
           "261df2c53b76af587c65d293f980fb3f45a8f9362b3d49fd7111227ef9e6077b",
           "d0a9bc52b7d186852eabb1b838f4d74c5476d25ffa9e6c5ef15e447f33a1831e",
           "80636fa874a2c9aadb3e2106d259812cb22153617a76317fc2695c15e69074ff",
           "da5ba8605130ad6399cab1e1cd382e8416c7c716df0dac826e398eff1a6ee5f2",
           "7d1c38cc3e28dabef769950d3b131493b713706dfbb5bcb8f85c1d356cfe1ec2",
           "eeff2a3c73432deae976e68cc74e9e6ff3308284307334e7fdc606297ffdc19e",
           "906353f3bc653d8e5966373b0925f03ecdd0b0baf95039d510437789979b818c",
           "9c99f9a3422518e013f7682ef34dabbf5b6d3af762eac7892dde030e280ef023",
       }) {
    ASSERT_TRUE(HexStringToBytes(hash, log_entry.add_hashes()));
  }
  log_entry.set_checkpoint_origin("rekor.sigstore.dev - 1193050959916656506");
  ASSERT_TRUE(HexStringToBytes(
      "952e3f895a4f4619b9c522c830544533dc731c7d045731bb5af3921e7b8b31b6"
      "2c3b3ae0a8f7d15d6021409b817cfbed9388bbba096110ac8e2e5fa676fa37f2",
      log_entry.mutable_checkpoint_signature()));
  log_entry.set_checkpoint_signature_key_id("\xc0\xd2\x3d\x6a");

  Key verifying_key;
  verifying_key.set_algorithm(Key::ECDSA_P256);
  verifying_key.set_purpose(Key::VERIFY);
  ASSERT_TRUE(HexStringToBytes(
      "048ede9c77dc6780f525ec75587cf29c1ebcda2b2938eb41a22b0edf2cbbbfbc117a1f96"
      "1715cf433c55db5bce3ad8f10c2f2fcc96694e4fc4e8f80f8db54f7228",
      verifying_key.mutable_key_material()));
  Key rekor_key;
  rekor_key.set_algorithm(Key::ECDSA_P256);
  rekor_key.set_purpose(Key::VERIFY);
  rekor_key.set_key_id("\xc0\xd2\x3d\x6a");
  ASSERT_TRUE(HexStringToBytes(
      "04d86d98fb6b5a6dd4d5e41706881231d1af5f005c2b9016e62d21ad92ce0bdea5fac986"
      "34cee7c19e10bc52bfe2cb9e468563fff40fdb6362e10b7d0cf7e458b7",
      rekor_key.mutable_key_material()));
  absl::FixedArray<uint8_t> digest = {211, 205, 250, 29,  200, 107, 145, 232,
                                      202, 66,  147, 39,  177, 68,  218, 81,
                                      77,  188, 50,  179, 136, 131, 223, 248,
                                      48,  51,  108, 145, 50,  110, 122, 135};

  EXPECT_OK(
      VerifyRekorLogEntry(log_entry, {&verifying_key}, {&rekor_key}, digest));
}

}  // namespace
}  // namespace fcp::confidential_compute::payload_transparency
