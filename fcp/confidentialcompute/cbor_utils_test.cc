// Copyright 2024 Google LLC
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

#include "fcp/confidentialcompute/cbor_utils.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <tuple>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/testing/testing.h"
#include "third_party/libcbor/src/cbor/bytestrings.h"
#include "third_party/libcbor/src/cbor/common.h"
#include "third_party/libcbor/src/cbor/data.h"
#include "third_party/libcbor/src/cbor/ints.h"

namespace fcp::confidential_compute {
namespace {

MATCHER_P(IsOkAndHolds, matcher, "") {
  return arg.ok() &&
         testing::ExplainMatchResult(matcher, arg.value(), result_listener);
}

TEST(CborUtilsTest, CborRef) {
  CborRef value(cbor_build_uint32(3));
  CborRef value2 = std::move(value);
  EXPECT_EQ(cbor_get_uint32(value2.get()), 3);
  EXPECT_EQ(value.get(), nullptr);
}

TEST(CborUtilsTest, BuildCborInt) {
  // Test uints of various sizes.
  for (auto [n, expected_width] :
       std::initializer_list<std::tuple<int64_t, int>>{
           {0, CBOR_INT_8},
           {1, CBOR_INT_8},
           {0xff, CBOR_INT_8},
           {0x100, CBOR_INT_16},
           {0xffff, CBOR_INT_16},
           {0x10000, CBOR_INT_32},
           {0xffffffff, CBOR_INT_32},
           {0x100000000, CBOR_INT_64},
           {0x7fffffffffffffff, CBOR_INT_64},
       }) {
    SCOPED_TRACE(testing::Message() << "n=" << n);
    auto item = BuildCborInt(n);
    ASSERT_TRUE(cbor_isa_uint(item.get())) << cbor_typeof(item.get());
    EXPECT_EQ(cbor_int_get_width(item.get()), expected_width);
    EXPECT_EQ(cbor_get_int(item.get()), n);
  }

  // Test negints of various sizes.
  for (auto [n, expected_width] :
       std::initializer_list<std::tuple<int64_t, int>>{
           {-1, CBOR_INT_8},
           {-0xff, CBOR_INT_8},
           {-0x100, CBOR_INT_8},
           {-0x101, CBOR_INT_16},
           {-0xffff, CBOR_INT_16},
           {-0x10000, CBOR_INT_16},
           {-0x10001, CBOR_INT_32},
           {-0xffffffffLL, CBOR_INT_32},
           {-0x100000000, CBOR_INT_32},
           {-0x100000001, CBOR_INT_64},
           {-0x7fffffffffffffff, CBOR_INT_64},
       }) {
    SCOPED_TRACE(testing::Message() << "n=" << n);
    auto item = BuildCborInt(n);
    ASSERT_TRUE(cbor_isa_negint(item.get())) << cbor_typeof(item.get());
    EXPECT_EQ(cbor_int_get_width(item.get()), expected_width);
    EXPECT_EQ(cbor_get_int(item.get()), -n - 1);
  }
}

TEST(CborUtilsTest, GetCborInt) {
  // Test various valid uints.
  EXPECT_THAT(GetCborInt(*CborRef(cbor_build_uint8(0))), IsOkAndHolds(0));
  EXPECT_THAT(GetCborInt(*CborRef(cbor_build_uint8(100))), IsOkAndHolds(100));
  EXPECT_THAT(GetCborInt(*CborRef(
                  cbor_build_uint64(std::numeric_limits<int64_t>::max()))),
              IsOkAndHolds(std::numeric_limits<int64_t>::max()));

  // Test various valid negints.
  EXPECT_THAT(GetCborInt(*CborRef(cbor_build_negint8(0))), IsOkAndHolds(-1));
  EXPECT_THAT(GetCborInt(*CborRef(cbor_build_negint8(100))),
              IsOkAndHolds(-101));
  EXPECT_THAT(GetCborInt(*CborRef(
                  cbor_build_negint64(std::numeric_limits<int64_t>::max()))),
              IsOkAndHolds(std::numeric_limits<int64_t>::min()));

  // Test values that are outside the range of int64_t.
  EXPECT_THAT(GetCborInt(*CborRef(cbor_build_uint64(0x8000000000000000))),
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetCborInt(*CborRef(
                  cbor_build_uint64(std::numeric_limits<uint64_t>::max()))),
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetCborInt(*CborRef(cbor_build_negint64(0x8000000000000000))),
              IsCode(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetCborInt(*CborRef(
                  cbor_build_negint64(std::numeric_limits<uint64_t>::max()))),
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(CborUtilsTest, SerializeCbor) {
  absl::StatusOr<CborRef> item =
      SerializeCbor(*CborRef(cbor_build_uint8(0x64)));
  ASSERT_OK(item);
  ASSERT_NE(*item, nullptr);
  ASSERT_TRUE(cbor_isa_bytestring(item->get())) << cbor_typeof(item->get());
  EXPECT_EQ(absl::string_view(
                reinterpret_cast<char*>(cbor_bytestring_handle(item->get())),
                cbor_bytestring_length(item->get())),
            "\x18\x64");
}

TEST(CborUtilsTest, SerializeCborToString) {
  // Just spot-check one value. The main benefit of this check is verifying that
  // there aren't memory leaks.
  EXPECT_THAT(SerializeCborToString(*CborRef(cbor_build_uint8(0x64))),
              IsOkAndHolds("\x18\x64"));
}

}  // namespace
}  // namespace fcp::confidential_compute
