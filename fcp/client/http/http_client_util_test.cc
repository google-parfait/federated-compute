/*
 * Copyright 2021 Google LLC
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
#include "fcp/client/http/http_client_util.h"

#include <optional>
#include <string>

#include "google/rpc/status.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"

namespace fcp::client::http {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::Lt;
using ::testing::Optional;
using ::testing::StrEq;

TEST(ConvertHttpCodeToStatusTest, ConvertsKnownCodesCorrectly) {
  ABSL_EXPECT_OK(ConvertHttpCodeToStatus(kHttpOk));
  EXPECT_THAT(ConvertHttpCodeToStatus(kHttpBadRequest),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ConvertHttpCodeToStatus(kHttpForbidden),
              StatusIs(absl::StatusCode::kPermissionDenied));
  EXPECT_THAT(ConvertHttpCodeToStatus(kHttpNotFound),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(ConvertHttpCodeToStatus(kHttpConflict),
              StatusIs(absl::StatusCode::kAborted));
  EXPECT_THAT(ConvertHttpCodeToStatus(kHttpTooManyRequests),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_THAT(ConvertHttpCodeToStatus(kHttpClientClosedRequest),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(ConvertHttpCodeToStatus(kHttpGatewayTimeout),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(ConvertHttpCodeToStatus(kHttpNotImplemented),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(ConvertHttpCodeToStatus(kHttpServiceUnavailable),
              StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_THAT(ConvertHttpCodeToStatus(kHttpUnauthorized),
              StatusIs(absl::StatusCode::kUnauthenticated));
}

TEST(ConvertHttpCodeToStatusTest, ConvertsUnknown200CodesToOk) {
  ABSL_EXPECT_OK(ConvertHttpCodeToStatus(201));
  ABSL_EXPECT_OK(ConvertHttpCodeToStatus(210));
  ABSL_EXPECT_OK(ConvertHttpCodeToStatus(299));
}

TEST(ConvertHttpCodeToStatusTest, ConvertsUnknown400CodesToFailedPrecondition) {
  // Note: 400, 401, and 499 are known errors codes that map to other values. We
  // hence test a few other values in the 400 range that aren't "known".
  EXPECT_THAT(ConvertHttpCodeToStatus(402),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(ConvertHttpCodeToStatus(405),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(ConvertHttpCodeToStatus(410),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(ConvertHttpCodeToStatus(498),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(ConvertHttpCodeToStatusTest, ConvertsUnknown500CodesToInternal) {
  // note: 501 is a known error code that maps to other values. We hence test
  // 502 instead.
  EXPECT_THAT(ConvertHttpCodeToStatus(500),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(ConvertHttpCodeToStatus(502),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(ConvertHttpCodeToStatus(510),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(ConvertHttpCodeToStatus(599),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(ConvertHttpCodeToStatusTest, ConvertsUnknownAllOtherCodesToUnknown) {
  EXPECT_THAT(ConvertHttpCodeToStatus(300),
              StatusIs(absl::StatusCode::kUnknown));
  EXPECT_THAT(ConvertHttpCodeToStatus(301),
              StatusIs(absl::StatusCode::kUnknown));
  EXPECT_THAT(ConvertHttpCodeToStatus(310),
              StatusIs(absl::StatusCode::kUnknown));
  EXPECT_THAT(ConvertHttpCodeToStatus(399),
              StatusIs(absl::StatusCode::kUnknown));
  EXPECT_THAT(ConvertHttpCodeToStatus(0), StatusIs(absl::StatusCode::kUnknown));
  EXPECT_THAT(ConvertHttpCodeToStatus(1), StatusIs(absl::StatusCode::kUnknown));
  EXPECT_THAT(ConvertHttpCodeToStatus(10),
              StatusIs(absl::StatusCode::kUnknown));
  EXPECT_THAT(ConvertHttpCodeToStatus(99),
              StatusIs(absl::StatusCode::kUnknown));
  EXPECT_THAT(ConvertHttpCodeToStatus(600),
              StatusIs(absl::StatusCode::kUnknown));
}

// Test to ensure that when we map an 'unknown' HTTP response to its fallback
// catch-all StatusCode, we keep the original error code in the error message
// (to aid in debugging).
TEST(ConvertHttpCodeToStatusTest, IncludesOriginalErrorCodeInMessage) {
  EXPECT_THAT(ConvertHttpCodeToStatus(400).message(), HasSubstr("code: 400"));
  EXPECT_THAT(ConvertHttpCodeToStatus(402).message(), HasSubstr("code: 402"));
  EXPECT_THAT(ConvertHttpCodeToStatus(502).message(), HasSubstr("code: 502"));
  EXPECT_THAT(ConvertHttpCodeToStatus(503).message(), HasSubstr("code: 503"));
  EXPECT_THAT(ConvertHttpCodeToStatus(300).message(), HasSubstr("code: 300"));
}

void ExpectRpcStatusMatchAbslStatus(absl::StatusCode code) {
  ::google::rpc::Status rpc_status;
  *rpc_status.mutable_message() = "the_message";
  rpc_status.set_code(static_cast<int>(code));
  absl::Status converted_status = ConvertRpcStatusToAbslStatus(rpc_status);
  EXPECT_THAT(converted_status, StatusIs(code));
  // OK Status objects always have empty messages.
  EXPECT_EQ(converted_status.message(),
            code == absl::StatusCode::kOk ? "" : "the_message");
}

TEST(ConvertRpcStatusToAbslStatusTest, ValidCodesShouldConvertSuccessfully) {
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kOk);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kCancelled);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kUnknown);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kInvalidArgument);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kDeadlineExceeded);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kNotFound);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kAlreadyExists);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kPermissionDenied);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kResourceExhausted);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kFailedPrecondition);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kAborted);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kOutOfRange);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kUnimplemented);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kInternal);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kUnavailable);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kDataLoss);
  ExpectRpcStatusMatchAbslStatus(absl::StatusCode::kUnauthenticated);
}

TEST(ConvertRpcStatusToAbslStatusTest, InvalidCodesShouldConvertToUnknown) {
  ::google::rpc::Status rpc_status;
  *rpc_status.mutable_message() = "the_message";
  rpc_status.set_code(100);  // 100 is not a valid status code.
  absl::Status converted_status = ConvertRpcStatusToAbslStatus(rpc_status);
  EXPECT_THAT(converted_status, StatusIs(absl::StatusCode::kUnknown));
  EXPECT_EQ(converted_status.message(), "the_message");
}

TEST(FindHeaderTest, DoesNotFindMissingHeader) {
  EXPECT_EQ(FindHeader({}, "Header-One"), std::nullopt);
  EXPECT_EQ(FindHeader({{"Header-Two", "bar"}}, "Header-One"), std::nullopt);
  EXPECT_EQ(FindHeader({{"Header-Two", "bar"}, {"Header-Three", "baz"}},
                       "Header-One"),
            std::nullopt);
}
TEST(FindHeaderTest, EmptyNeedleDoesNotFindAnyHeader) {
  EXPECT_EQ(FindHeader({}, ""), std::nullopt);
  EXPECT_EQ(FindHeader({{"Header-Two", "bar"}}, ""), std::nullopt);
}

TEST(FindHeaderTest, FindsHeaderInList) {
  EXPECT_THAT(FindHeader({{"Header-One", "foo"},
                          {"Header-Two", "bar"},
                          {"Header-Three", "baz"}},
                         "Header-Two"),
              Optional(StrEq("bar")));

  EXPECT_THAT(FindHeader({{"Header-Two", "BAR"},
                          {"Header-One", "foo"},
                          {"Header-Three", "baz"}},
                         "Header-Two"),
              Optional(StrEq("BAR")));

  EXPECT_THAT(FindHeader({{"Header-One", "foo"},
                          {"Header-Three", "baz"},
                          {"Header-Two", "BaR"}},
                         "Header-Two"),
              Optional(StrEq("BaR")));

  EXPECT_THAT(FindHeader({{"Header-Two", "bar"}}, "Header-Two"),
              Optional(StrEq("bar")));
}

TEST(FindHeaderTest, FindsHeaderInListDespiteMixedCase) {
  // In each of these scenarios, the fact that the list or needle are not the
  // same case should not affect the result.
  EXPECT_THAT(FindHeader({{"Header-One", "foo"},
                          {"header-two", "bar"},
                          {"Header-Three", "baz"}},
                         "Header-Two"),
              Optional(StrEq("bar")));

  EXPECT_THAT(FindHeader({{"Header-One", "foo"},
                          {"Header-Two", "bar"},
                          {"Header-Three", "baz"}},
                         "header-two"),
              Optional(StrEq("bar")));
}

TEST(FindHeaderTest, ReturnsFirstMatch) {
  // In each of these scenarios, the first matching header value should be
  // returned.
  EXPECT_THAT(
      FindHeader(
          {{"Header-One", "foo"}, {"Header-Two", "bar"}, {"Header-Two", "baz"}},
          "Header-Two"),
      Optional(StrEq("bar")));

  EXPECT_THAT(
      FindHeader(
          {{"Header-One", "foo"}, {"Header-Two", "bar"}, {"Header-Two", "baz"}},
          "Header-Two"),
      Optional(StrEq("bar")));

  EXPECT_THAT(
      FindHeader(
          {{"Header-One", "foo"}, {"header-two", "bar"}, {"HEADER-TWO", "baz"}},
          "HEADER-TWO"),
      Optional(StrEq("bar")));
}

TEST(JoinBaseUriWithSuffixTest, ReturnsJoinedUri) {
  // No trailing slash in base URI.
  auto result = JoinBaseUriWithSuffix("https://foo", "/bar");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq("https://foo/bar"));

  // Trailing slash in base URI.
  result = JoinBaseUriWithSuffix("https://foo/", "/bar");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq("https://foo/bar"));

  // Additional URI components are correctly merged.
  result = JoinBaseUriWithSuffix("https://foo:123", "/bar/baz");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq("https://foo:123/bar/baz"));

  result = JoinBaseUriWithSuffix("https://foo:123/", "/bar/baz");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq("https://foo:123/bar/baz"));

  // Empty suffixes should be allowed.
  result = JoinBaseUriWithSuffix("https://foo", "");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq("https://foo/"));

  // Trailing slash in base URI.
  result = JoinBaseUriWithSuffix("https://foo/", "");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq("https://foo/"));
}

TEST(JoinBaseUriWithSuffixTest, NoLeadingSlashInSuffixFails) {
  // No leading slash in the URI suffix, should result in error.
  auto result = JoinBaseUriWithSuffix("https://foo", "bar");
  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kInvalidArgument));
  result = JoinBaseUriWithSuffix("https://foo/", "bar");
  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EncodeUriTest, UnencodedCharsShouldRemainUnencoded) {
  std::string unencoded_single_path_segment_chars =
      "-_.~0123456789abcxyzABCXYZ";
  auto result = EncodeUriSinglePathSegment(unencoded_single_path_segment_chars);
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq(unencoded_single_path_segment_chars));

  std::string unencoded_multi_path_segment_chars =
      "-_.~/01234567899abcxyzABCXYZ";
  result = EncodeUriMultiplePathSegments(unencoded_multi_path_segment_chars);
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq(unencoded_multi_path_segment_chars));
}

TEST(EncodeUriTest, OtherCharsShouldBeEncoded) {
  auto result = EncodeUriSinglePathSegment("#?+%/");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq("%23%3F%2B%25%2F"));

  // For the "multiple path segments" version the slash should remain unencoded.
  result = EncodeUriMultiplePathSegments("#?+%/");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq("%23%3F%2B%25/"));

  // Non-encodable characters before/in between/after the encodable characters
  // should remain unencoded.
  result = EncodeUriSinglePathSegment("abc#?123+%/XYZ");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq("abc%23%3F123%2B%25%2FXYZ"));

  result = EncodeUriMultiplePathSegments("abc#?123+%/XYZ");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq("abc%23%3F123%2B%25/XYZ"));
}

TEST(EncodeUriTest, EmptyStringShouldReturnEmptyString) {
  auto result = EncodeUriSinglePathSegment("");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq(""));

  // For the "multiple path segments" version the slash should remain unencoded.
  result = EncodeUriMultiplePathSegments("");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, StrEq(""));
}

TEST(EncodeUriTest, NonAsciiStringShouldReturnError) {
  auto result = EncodeUriSinglePathSegment("€");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));

  // For the "multiple path segments" version the slash should remain unencoded.
  result = EncodeUriMultiplePathSegments("€");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CreateByteStreamUriTest, HappyCase) {
  auto result = CreateByteStreamUploadUriSuffix("my/resource");
  ABSL_ASSERT_OK(result);
  EXPECT_THAT(*result, "/upload/v1/media/my/resource");
}

TEST(CreateByteStreamUriTest, NonAsciiResourceNameShouldReturnError) {
  EXPECT_THAT(CreateByteStreamUploadUriSuffix("€"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(IsRetryableErrorTest, ReturnsTrueForRetryableErrors) {
  EXPECT_TRUE(IsRetryableError(absl::StatusCode::kUnavailable));
}

TEST(IsRetryableErrorTest, ReturnsFalseForNonRetryableErrors) {
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kOk));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kCancelled));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kUnknown));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kInvalidArgument));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kDeadlineExceeded));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kNotFound));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kAlreadyExists));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kPermissionDenied));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kResourceExhausted));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kFailedPrecondition));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kAborted));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kOutOfRange));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kUnimplemented));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kInternal));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kDataLoss));
  EXPECT_FALSE(IsRetryableError(absl::StatusCode::kUnauthenticated));
}

TEST(GetRetryDelayTest, ReturnsCorrectDelay) {
  absl::BitGen bit_gen;
  absl::Duration retry_delay = absl::Seconds(1);
  // Lower bound = (0.4 + 0.6 * 1.3^retries) * retry_delay
  // Upper bound = (0.4 + 1.3^retries) * retry_delay
  EXPECT_THAT(
      GetRetryDelay(bit_gen, retry_delay, 1),
      AllOf(Gt(absl::Milliseconds(1179)), Lt(absl::Milliseconds(1701))));
  EXPECT_THAT(
      GetRetryDelay(bit_gen, retry_delay, 2),
      AllOf(Gt(absl::Milliseconds(1413)), Lt(absl::Milliseconds(2100))));
  EXPECT_THAT(
      GetRetryDelay(bit_gen, retry_delay, 3),
      AllOf(Gt(absl::Milliseconds(1717)), Lt(absl::Milliseconds(2600))));
  EXPECT_THAT(
      GetRetryDelay(bit_gen, retry_delay, 4),
      AllOf(Gt(absl::Milliseconds(2112)), Lt(absl::Milliseconds(3257))));
  EXPECT_THAT(
      GetRetryDelay(bit_gen, retry_delay, 5),
      AllOf(Gt(absl::Milliseconds(2610)), Lt(absl::Milliseconds(4120))));
}

}  // namespace
}  // namespace fcp::client::http
