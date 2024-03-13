// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fcp/confidentialcompute/cose.h"

#include <optional>
#include <string>

#include "google/protobuf/struct.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/testing/testing.h"

namespace fcp::confidential_compute {
namespace {

MATCHER_P(IsOkAndHolds, matcher, "") {
  return arg.ok() &&
         testing::ExplainMatchResult(matcher, arg.value(), result_listener);
}

TEST(OkpKeyTest, EncodeEmpty) {
  EXPECT_THAT(OkpKey().Encode(),
              IsOkAndHolds("\xa1"      // map with 1 item:
                           "\x01\x01"  // key type (1): OKP (1)
                           ));
}

TEST(OkpKeyTest, EncodeFull) {
  EXPECT_THAT((OkpKey{
                   .key_id = "key-id",
                   .algorithm = 7,
                   .curve = 45,
                   .x = "x-value",
               })
                  .Encode(),
              IsOkAndHolds("\xa5"             // map with 5 items:
                           "\x01\x01"         // key type (1): OKP (1)
                           "\x02\x46key-id"   // key id (2): b"key-id"
                           "\x03\x07"         // algorithm (3): 7
                           "\x20\x18\x2d"     // curve (-1): 45
                           "\x21\x47x-value"  // x (-2): b"x-value"
                           ));
}

TEST(OkpKeyTest, DecodeEmpty) {
  absl::StatusOr<OkpKey> key = OkpKey::Decode("\xa1\x01\x01");
  ASSERT_OK(key);
  EXPECT_EQ(key->key_id, "");
  EXPECT_EQ(key->algorithm, std::nullopt);
  EXPECT_EQ(key->curve, std::nullopt);
  EXPECT_EQ(key->x, "");
}

TEST(OkpKeyTest, DecodeFull) {
  absl::StatusOr<OkpKey> key = OkpKey::Decode(
      "\xa5\x01\x01\x02\x46key-id\x03\x07\x20\x18\x2d\x21\x47x-value");
  ASSERT_OK(key);
  EXPECT_EQ(key->key_id, "key-id");
  EXPECT_EQ(key->algorithm, 7);
  EXPECT_EQ(key->curve, 45);
  EXPECT_EQ(key->x, "x-value");
}

TEST(OkpKeyTest, DecodeInvalid) {
  EXPECT_THAT(OkpKey::Decode(""), IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OkpKey::Decode("\xa5"),  // map with 5 items
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OkpKey::Decode("\xa0 extra"),  // map with 0 items + " extra"
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(SymmetricKeyTest, EncodeEmpty) {
  EXPECT_THAT(SymmetricKey().Encode(),
              IsOkAndHolds("\xa1"      // map with 1 item:
                           "\x01\x04"  // key type (1): Symmetric (4)
                           ));
}

TEST(SymmetricKeyTest, EncodeFull) {
  EXPECT_THAT((SymmetricKey{
                   .algorithm = 7,
                   .k = "secret",
               })
                  .Encode(),
              IsOkAndHolds("\xa3"            // map with 3 items:
                           "\x01\x04"        // key type (1): Symmetric (4)
                           "\x03\x07"        // algorithm (3): 7
                           "\x20\x46secret"  // k (-1): b"secret"
                           ));
}

TEST(SymmetricKeyTest, DecodeEmpty) {
  absl::StatusOr<SymmetricKey> key = SymmetricKey::Decode("\xa1\x01\x04");
  ASSERT_OK(key);
  EXPECT_EQ(key->algorithm, std::nullopt);
  EXPECT_EQ(key->k, "");
}

TEST(SymmetricKeyTest, DecodeFull) {
  absl::StatusOr<SymmetricKey> key =
      SymmetricKey::Decode("\xa3\x01\x04\x03\x07\x20\x46secret");
  ASSERT_OK(key);
  EXPECT_EQ(key->algorithm, 7);
  EXPECT_EQ(key->k, "secret");
}

TEST(SymmetricKeyTest, DecodeInvalid) {
  EXPECT_THAT(SymmetricKey::Decode(""),
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SymmetricKey::Decode("\xa3"),  // map with 5 items
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SymmetricKey::Decode("\xa0 xtra"),  // map with 0 items + " xtra"
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(OkpCwtTest, BuildSigStructureEmpty) {
  EXPECT_THAT(OkpCwt().BuildSigStructure(""),
              IsOkAndHolds("\x84"            // array with 4 items:
                           "\x6aSignature1"  // "Signature1"
                           "\x41\xa0"        // protected headers
                           "\x40"            // aad: b""
                           "\x41\xa0"        // payload (empty claims map)
                           ));
}

TEST(OkpCwtTest, BuildSigStructureFull) {
  EXPECT_THAT(
      (OkpCwt{
           .issued_at = absl::FromUnixSeconds(1000),
           .expiration_time = absl::FromUnixSeconds(2000),
           .public_key = OkpKey(),
           .signature = "signature",
       })
          .BuildSigStructure("aad"),
      IsOkAndHolds(absl::string_view(
          "\x84"            // array with 4 items:
          "\x6aSignature1"  // "Signature1"
          "\x41\xa0"        // protected headers
          "\x43"
          "aad"               // b"aad"
          "\x52\xa3"          // payload: bstr map w/ 3 items (claims)
          "\x04\x19\x07\xd0"  // expiration time (4) = 2000
          "\x06\x19\x03\xe8"  // issued at (6) = 1000
          "\x3a\x00\x01\x00\x00\x43\xa1\x01\x01",  // public key (-65537)
                                                   // = empty OkpKey
          37)));
}

TEST(OkpCwtTest, EncodeEmpty) {
  EXPECT_THAT(
      OkpCwt().Encode(),
      IsOkAndHolds("\x84"      // array with 4 items:
                   "\x41\xa0"  // bstr containing empty map (protected headers)
                   "\xa0"      // empty map (unprotected headers)
                   "\x41\xa0"  // bstr containing empty map (claims)
                   "\x40"      // b"" (signature)
                   ));
}

TEST(OkpCwtTest, EncodeFull) {
  google::protobuf::Struct config_properties;
  (*config_properties.mutable_fields())["x"].set_bool_value(true);

  EXPECT_THAT(
      (OkpCwt{
           .issued_at = absl::FromUnixSeconds(1000),
           .expiration_time = absl::FromUnixSeconds(2000),
           .public_key = OkpKey(),
           .config_properties = config_properties,
           .signature = "signature",
       })
          .Encode(),
      IsOkAndHolds(absl::string_view(
          "\x84"              // array with 4 items:
          "\x41\xa0"          // bstr containing empty map (protected headers)
          "\xa0"              // empty map (unprotected headers)
          "\x58\x21\xa4"      // bstr containing a map with 4 items: (claims)
          "\x04\x19\x07\xd0"  // expiration time (4) = 2000
          "\x06\x19\x03\xe8"  // issued at (6) = 1000
          "\x3a\x00\x01\x00\x00\x43\xa1\x01\x01"  // public key (-65537)
                                                  // = empty OkpKey
          "\x3a\x00\x01\x00\x01\x49"              // config (-65538) = {
          "\x0a\x07"                              // fields (1) {
          "\x0a\x01x"                             //   key (1): "x"
          "\x12\x02"                              //   value (2): {
          "\x20\x01"                              //     bool_value (4): true
                                                  //   }
                                                  // }
          "\x49signature",
          49)));
}

TEST(OkpCwtTest, DecodeEmpty) {
  absl::StatusOr<OkpCwt> key = OkpCwt::Decode("\x84\x41\xa0\xa0\x41\xa0\x40");
  ASSERT_OK(key);
  EXPECT_EQ(key->issued_at, std::nullopt);
  EXPECT_EQ(key->expiration_time, std::nullopt);
  EXPECT_EQ(key->public_key, std::nullopt);
  EXPECT_THAT(key->config_properties, EqualsProto(""));
  EXPECT_EQ(key->signature, "");
}

TEST(OkpCwtTest, DecodeFull) {
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(absl::string_view(
      "\x84\x41\xa0\xa0\x58\x21\xa4\x04\x19\x07\xd0\x06\x19\x03\xe8\x3a\x00\x01"
      "\x00\x00\x43\xa1\x01\x01\x3a\x00\x01\x00\x01\x49\x0a\x07\x0a\x01x\x12"
      "\x02\x20\x01\x49signature",
      49));
  ASSERT_OK(cwt);
  EXPECT_EQ(cwt->issued_at, absl::FromUnixSeconds(1000));
  EXPECT_EQ(cwt->expiration_time, absl::FromUnixSeconds(2000));
  EXPECT_TRUE(cwt->public_key.has_value());
  EXPECT_THAT(cwt->config_properties, EqualsProto(R"pb(
                fields {
                  key: "x"
                  value { bool_value: true }
                }
              )pb"));
  EXPECT_EQ(cwt->signature, "signature");
}

TEST(OkpCwtTest, DecodeInvalid) {
  EXPECT_THAT(OkpCwt::Decode(""), IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OkpCwt::Decode("\xa3"),  // map with 5 items
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OkpCwt::Decode("\xa0 extra"),  // map with 0 items + " extra"
              IsCode(absl::StatusCode::kInvalidArgument));

  // Even if the CBOR is valid, the top-level structure must be a 4-element
  // array.
  EXPECT_THAT(OkpCwt::Decode("\xa0"),  // map with 0 items
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OkpCwt::Decode("\x80"),  // array with 0 items
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OkpCwt::Decode("\x83\x41\xa0\xa0\x41\xa0"),  // array with 3 items
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      OkpCwt::Decode("\x85\x41\xa0\xa0\x41\xa0\x40\x40"),  // array with 5 items
      IsCode(absl::StatusCode::kInvalidArgument));

  // The map entry types must be bstr, map, bstr, bstr.
  // "\x84\x41\xa0\xa0\x41\xa0\x40" is valid.
  EXPECT_THAT(OkpCwt::Decode("\x84\xa0\xa0\x41\xa0\x40"),  // 1st not bstr
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OkpCwt::Decode("\x84\x41\xa0\x40\x41\xa0\x40"),  // 2nd not map
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OkpCwt::Decode("\x84\x41\xa0\xa0\xa0\x40"),  // 3rd not bstr
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OkpCwt::Decode("\x84\x41\xa0\xa0\x41\xa0\xa0"),  // 4th not bstr
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(OkpCwtTest, DecodeReferenceExample) {
  // From RFC 8392 Section A.3, without the leading COSE_Sign1 tag "d2":
  std::string encoded = absl::HexStringToBytes(
      "8443a10126a104524173796d6d657472696345434453413235365850a701756"
      "36f61703a2f2f61732e6578616d706c652e636f6d02656572696b77037818636f"
      "61703a2f2f6c696768742e6578616d706c652e636f6d041a5612aeb0051a5610d"
      "9f0061a5610d9f007420b7158405427c1ff28d23fbad1f29c4c7c6a555e601d6f"
      "a29f9179bc3d7438bacaca5acd08c8d4d4f96131680c429a01f85951ecee743a5"
      "2b9b63632c57209120e1c9e30");
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(encoded);
  ASSERT_OK(cwt);
  EXPECT_EQ(cwt->issued_at, absl::FromUnixSeconds(1443944944));
  EXPECT_EQ(cwt->expiration_time, absl::FromUnixSeconds(1444064944));
  EXPECT_FALSE(cwt->public_key.has_value());
  EXPECT_EQ(
      cwt->signature,
      absl::HexStringToBytes(
          "5427c1ff28d23fbad1f29c4c7c6a555e601d6fa29f9179bc3d7438bacaca5acd08c8"
          "d4d4f96131680c429a01f85951ecee743a52b9b63632c57209120e1c9e30"));
}

}  // namespace
}  // namespace fcp::confidential_compute
