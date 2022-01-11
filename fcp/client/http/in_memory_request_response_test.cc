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
#include "fcp/client/http/in_memory_request_response.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/http_client_util.h"
#include "fcp/client/http/test_helpers.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/test_helpers.h"
#include "fcp/testing/testing.h"

namespace fcp::client::http {
namespace {

using ::fcp::IsCode;
using ::testing::_;
using ::testing::ContainerEq;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::MockFunction;
using ::testing::Ne;
using ::testing::NiceMock;
using ::testing::Pair;
using ::testing::Return;
using ::testing::StrEq;
using ::testing::StrictMock;

TEST(InMemoryHttpRequestTest, NonHttpsUriFails) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("http://invalid.com",
                                  HttpRequest::Method::kGet, {}, "");
  EXPECT_THAT(request.status(), IsCode(INVALID_ARGUMENT));
}

TEST(InMemoryHttpRequestTest, GetWithRequestBodyFails) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {},
                                  "non_empty_request_body");
  EXPECT_THAT(request.status(), IsCode(INVALID_ARGUMENT));
}

// Ensures that providing a Content-Length header results in an error (since it
// is automatically generated instead).
TEST(InMemoryHttpRequestTest, ContentLengthHeaderFails) {
  // Note that we purposely use a mixed-case "Content-length" header name, to
  // ensure that the check is correctly case-insensitive.
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create(
          "https://valid.com", HttpRequest::Method::kPost,
          {{"Content-length", "1234"}}, "non_empty_request_body");
  EXPECT_THAT(request.status(), IsCode(INVALID_ARGUMENT));
}

TEST(InMemoryHttpRequestTest, ValidGetRequest) {
  const std::string expected_uri = "https://valid.com";
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create(expected_uri, HttpRequest::Method::kGet, {},
                                  "");
  ASSERT_OK(request);
  EXPECT_THAT((*request)->uri(), StrEq(expected_uri));
  EXPECT_EQ((*request)->method(), HttpRequest::Method::kGet);
  // Because no request body is present, the Content-Length header shouldn't
  // have been added.
  EXPECT_THAT((*request)->extra_headers(), IsEmpty());
  EXPECT_FALSE((*request)->HasBody());
}

TEST(InMemoryHttpRequestTest, ValidGetRequestWithHeaders) {
  const HeaderList expected_headers{{"Foo", "Bar"}, {"Baz", "123"}};
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create(
          "https://valid.com", HttpRequest::Method::kGet, expected_headers, "");
  ASSERT_OK(request);
  EXPECT_THAT((*request)->extra_headers(), ContainerEq(expected_headers));
}

TEST(InMemoryHttpRequestTest, ValidPostRequestWithoutBody) {
  const std::string expected_uri = "https://valid.com";
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create(expected_uri, HttpRequest::Method::kPost, {},
                                  "");
  ASSERT_OK(request);
  EXPECT_THAT((*request)->uri(), StrEq(expected_uri));
  EXPECT_EQ((*request)->method(), HttpRequest::Method::kPost);
  // Because no request body is present, the Content-Length header shouldn't
  // have been added.
  EXPECT_THAT((*request)->extra_headers(), IsEmpty());
  EXPECT_FALSE((*request)->HasBody());
}

TEST(InMemoryHttpRequestTest, ValidPostRequestWithBody) {
  const std::string expected_uri = "https://valid.com";
  const std::string expected_body = "request_body";
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create(expected_uri, HttpRequest::Method::kPost, {},
                                  expected_body);
  ASSERT_OK(request);
  EXPECT_THAT((*request)->uri(), StrEq(expected_uri));
  EXPECT_EQ((*request)->method(), HttpRequest::Method::kPost);
  EXPECT_THAT((*request)->extra_headers(),
              ElementsAre(Pair("Content-Length",
                               std::to_string(expected_body.size()))));
  EXPECT_TRUE((*request)->HasBody());
}

// Checks that the automatically generated Content-Length header is
// appropriately added to any specified extra headers (rather than replacing
// them completely).
TEST(InMemoryHttpRequestTest, ValidPostRequestWithBodyAndHeaders) {
  const HeaderList original_headers{{"Foo", "Bar"}, {"Baz", "123"}};
  const std::string expected_body = "request_body";
  const HeaderList expected_headers{
      {"Foo", "Bar"},
      {"Baz", "123"},
      {"Content-Length", std::to_string(expected_body.size())}};
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kPost, original_headers,
                                  expected_body);
  ASSERT_OK(request);

  EXPECT_THAT((*request)->extra_headers(), ContainerEq(expected_headers));
}

TEST(InMemoryHttpRequestTest, ReadBodySimple) {
  const std::string expected_body = "request_body";
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create(
          "https://valid.com", HttpRequest::Method::kPost, {}, expected_body);
  ASSERT_OK(request);

  std::string actual_body;
  actual_body.resize(expected_body.size());
  // Read the body in one go (the "simple" case).
  auto read_result =
      (*request)->ReadBody(actual_body.data(), actual_body.size());
  ASSERT_OK(read_result);
  EXPECT_THAT(*read_result, actual_body.size());
  EXPECT_THAT(actual_body, StrEq(expected_body));

  // Expect the second read to indicate the end of the stream.
  EXPECT_THAT((*request)->ReadBody(nullptr, 1), IsCode(OUT_OF_RANGE));
}

TEST(InMemoryHttpRequestTest, ReadBodyChunked) {
  // This test reads the body in chunks of 3 bytes at a time (rather than all at
  // once, like previous test). To test some edge cases, we ensure that the
  // request body's length is not evenly dividable by 3.
  const std::string expected_body = "12345678";
  ASSERT_THAT(expected_body.size() % 3, Ne(0));
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create(
          "https://valid.com", HttpRequest::Method::kPost, {}, expected_body);
  ASSERT_OK(request);

  std::string actual_body;
  // Pre-size the buffer with 'X' characters. This will allow us to check
  // whether each read modifies the expected range of the buffer.
  actual_body.resize(expected_body.size(), 'X');

  // Read the body 3 bytes at a time.
  // The first read tests the case where less data is requested than is
  // available.
  absl::StatusOr<int64_t> read_result =
      (*request)->ReadBody(actual_body.data(), 3);
  ASSERT_OK(read_result);
  EXPECT_THAT(*read_result, 3);
  EXPECT_THAT(actual_body, StrEq("123XXXXX"));

  read_result = (*request)->ReadBody(actual_body.data() + 3, 3);
  ASSERT_OK(read_result);
  EXPECT_THAT(*read_result, 3);
  EXPECT_THAT(actual_body, StrEq("123456XX"));

  // The last read should only read 2 bytes. This tests the case where more data
  // is requested than is available. A correct implementation should not write
  // more bytes than it has available, which should ensure that no writes will
  // occur beyond actual_body.data()'s buffer size (which is only 8 bytes long).
  read_result = (*request)->ReadBody(actual_body.data() + 6, 3);
  ASSERT_OK(read_result);
  EXPECT_THAT(*read_result, 2);
  EXPECT_THAT(actual_body, StrEq(expected_body));

  // Expect the last read to indicate the end of the stream.
  EXPECT_THAT((*request)->ReadBody(nullptr, 1), IsCode(OUT_OF_RANGE));
}

TEST(InMemoryHttpRequestCallbackTest, ResponseFailsBeforeHeaders) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  InMemoryHttpRequestCallback callback;
  callback.OnResponseError(**request, absl::UnimplementedError("foobar"));

  absl::StatusOr<InMemoryHttpResponse> actual_response = callback.Response();
  EXPECT_THAT(actual_response, IsCode(UNIMPLEMENTED));
  EXPECT_THAT(actual_response.status().message(), HasSubstr("foobar"));
}

TEST(InMemoryHttpRequestCallbackTest, ResponseFailsAfterHeaders) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  auto fake_response = FakeHttpResponse(kHttpOk, {});

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  callback.OnResponseBodyError(**request, fake_response,
                               absl::UnimplementedError("foobar"));

  absl::StatusOr<InMemoryHttpResponse> actual_response = callback.Response();
  EXPECT_THAT(actual_response, IsCode(UNIMPLEMENTED));
  EXPECT_THAT(actual_response.status().message(), HasSubstr("foobar"));
}

TEST(InMemoryHttpRequestCallbackTest, ResponseFailsAfterPartialBody) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  auto fake_response = FakeHttpResponse(kHttpOk, {});

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, "response_"));
  callback.OnResponseBodyError(**request, fake_response,
                               absl::UnimplementedError("foobar"));

  absl::StatusOr<InMemoryHttpResponse> actual_response = callback.Response();
  EXPECT_THAT(actual_response, IsCode(UNIMPLEMENTED));
  EXPECT_THAT(actual_response.status().message(), HasSubstr("foobar"));
}

TEST(InMemoryHttpRequestCallbackTest,
     TestResponseWithContentLengthAndBodyTooShort) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  auto fake_response = FakeHttpResponse(kHttpOk, {{"Content-Length", "5"}});

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  // Return a partial response (only 4 characters instead of the expected 5
  // indicated by the Content-Length header).
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, "12"));
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, "34"));
  callback.OnResponseCompleted(**request, fake_response);

  absl::StatusOr<InMemoryHttpResponse> actual_response = callback.Response();
  EXPECT_THAT(actual_response, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(actual_response.status().message(),
              HasSubstr("Too little response body data received"));
}

TEST(InMemoryHttpRequestCallbackTest,
     TestResponseWithContentLengthAndBodyTooLong) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  auto fake_response = FakeHttpResponse(kHttpOk, {{"Content-Length", "5"}});

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  // Return a more response data than expected (6 characters instead of the
  // expected 5 indicated by the Content-Length header).
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, "12"));
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, "34"));
  absl::Status response_body_result =
      callback.OnResponseBody(**request, fake_response, "56");
  EXPECT_THAT(response_body_result, IsCode(OUT_OF_RANGE));
  EXPECT_THAT(response_body_result.message(),
              HasSubstr("Too much response body data received"));
}

TEST(InMemoryHttpRequestCallbackTest,
     TestResponseWithBodyTooLongHardcodedWithoutContentLength) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  auto fake_response = FakeHttpResponse(kHttpOk, {});

  // Construct a response body that is longer than our hardcoded limit.
  std::string response_body;
  response_body.resize(50 * 1024 * 1024 + 1, 'X');  // 50 MiB + 1 byte.

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  absl::Status response_body_result =
      callback.OnResponseBody(**request, fake_response, response_body);
  EXPECT_THAT(response_body_result, IsCode(OUT_OF_RANGE));
  EXPECT_THAT(response_body_result.message(),
              HasSubstr("Too much response body data received"));
}

TEST(InMemoryHttpRequestCallbackTest,
     TestResponseWithContentLengthTooLongHardcoded) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  // Set the Content-Length to a size larger than our hardcoded limit.
  auto fake_response =
      FakeHttpResponse(kHttpOk, {{"Content-Length",
                                  // 50 MiB + 1 byte
                                  std::to_string(50 * 1024 * 1024 + 1)}});

  InMemoryHttpRequestCallback callback;
  absl::Status response_started_result =
      callback.OnResponseStarted(**request, fake_response);
  EXPECT_THAT(response_started_result, IsCode(OUT_OF_RANGE));
  EXPECT_THAT(response_started_result.message(),
              HasSubstr("Content-Length response header too small/large"));
}

TEST(InMemoryHttpRequestCallbackTest, ResponseWithContentLengthNegative) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  // Set the Content-Length to an invalid negative value.
  auto fake_response = FakeHttpResponse(kHttpOk, {{"Content-Length", "-1"}});

  InMemoryHttpRequestCallback callback;
  absl::Status response_started_result =
      callback.OnResponseStarted(**request, fake_response);
  EXPECT_THAT(response_started_result, IsCode(OUT_OF_RANGE));
  EXPECT_THAT(response_started_result.message(),
              HasSubstr("Content-Length response header too small/large"));
}

TEST(InMemoryHttpRequestCallbackTest, ResponseWithContentLengthNonInteger) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  // Set the Content-Length to an invalid non-integer, non-ASCII value.
  auto fake_response =
      FakeHttpResponse(kHttpOk, {{"Content-Length", "\U0001F600"}});

  InMemoryHttpRequestCallback callback;
  absl::Status response_started_result =
      callback.OnResponseStarted(**request, fake_response);
  EXPECT_THAT(response_started_result, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(response_started_result.message(),
              HasSubstr("Could not parse Content-Length response header"));
}

TEST(InMemoryHttpRequestCallbackTest, ResponseWithContentEncodingHeader) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  // Add a Content-Encoding header, which implementations should never provide
  // to us, unless we advertised an Accept-Encoding header in the request.
  auto fake_response = FakeHttpResponse(kHttpOk, {{"Content-Encoding", "foo"}});

  InMemoryHttpRequestCallback callback;
  absl::Status response_started_result =
      callback.OnResponseStarted(**request, fake_response);
  EXPECT_THAT(response_started_result, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(response_started_result.message(),
              HasSubstr("Unexpected header: Content-Encoding"));
}

TEST(InMemoryHttpRequestCallbackTest, ResponseWithTransferEncodingHeader) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  // Add a Transfer-Encoding header, which implementations should never provide
  // to us.
  auto fake_response =
      FakeHttpResponse(kHttpOk, {{"Transfer-Encoding", "foo"}});

  InMemoryHttpRequestCallback callback;
  absl::Status response_started_result =
      callback.OnResponseStarted(**request, fake_response);
  EXPECT_THAT(response_started_result, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(response_started_result.message(),
              HasSubstr("Unexpected header: Transfer-Encoding"));
}

TEST(InMemoryHttpRequestCallbackTest, OkResponseWithoutContentLength) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  // Note that the fake response does not contain a Content-Length header. The
  // InMemoryHttpRequestCallback should be able to handle this (accumulating
  // response data until OnResponseCompleted is called).
  int expected_code = 201;  // "201 Created"
  auto fake_response = FakeHttpResponse(expected_code, {});
  const std::string expected_body = "response_body";

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  // We return the response body in one go (the "simple" case).
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, expected_body));
  callback.OnResponseCompleted(**request, fake_response);

  absl::StatusOr<InMemoryHttpResponse> actual_response = callback.Response();
  ASSERT_OK(actual_response);
  EXPECT_THAT(*actual_response,
              FieldsAre(expected_code, IsEmpty(), StrEq(expected_body)));
}

TEST(InMemoryHttpRequestCallbackTest, OkResponseWithContentLength) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  int expected_code = kHttpOk;
  const std::string expected_body = "response_body";
  auto fake_response = FakeHttpResponse(
      expected_code,
      {{"Content-Length", std::to_string(expected_body.size())}});

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, expected_body));
  callback.OnResponseCompleted(**request, fake_response);

  absl::StatusOr<InMemoryHttpResponse> actual_response = callback.Response();
  ASSERT_OK(actual_response);
  EXPECT_THAT(*actual_response,
              FieldsAre(expected_code, IsEmpty(), StrEq(expected_body)));
}

TEST(InMemoryHttpRequestCallbackTest, OkResponseChunkedBody) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  // Note that the fake response does not contain a Content-Length header. The
  // InMemoryHttpRequestCallback should be able to handle this (accumulating
  // response data until OnResponseCompleted is called).
  auto fake_response = FakeHttpResponse(kHttpOk, {});

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  // This test returns the body in chunks of 3 bytes at a time (rather than
  // all at once, like previous test). To test some edge cases, we ensure that
  // the request body's length is not evenly dividable by 3.
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, "123"));
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, "456"));
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, "78"));
  callback.OnResponseCompleted(**request, fake_response);

  absl::StatusOr<InMemoryHttpResponse> actual_response = callback.Response();
  ASSERT_OK(actual_response);
  EXPECT_THAT(actual_response->body, StrEq("12345678"));
}

TEST(InMemoryHttpRequestCallbackTest,
     TestOkResponseWithEmptyBodyWithoutContentLength) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  int expected_code = kHttpOk;
  auto fake_response = FakeHttpResponse(expected_code, {});

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, ""));
  callback.OnResponseCompleted(**request, fake_response);

  absl::StatusOr<InMemoryHttpResponse> actual_response = callback.Response();
  ASSERT_OK(actual_response);
  EXPECT_THAT(*actual_response, FieldsAre(expected_code, IsEmpty(), IsEmpty()));
}

TEST(InMemoryHttpRequestCallbackTest,
     TestOkResponseWithEmptyBodyWithContentLength) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  int expected_code = kHttpOk;
  auto fake_response =
      FakeHttpResponse(expected_code, {{"Content-Length", "0"}});

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, ""));
  callback.OnResponseCompleted(**request, fake_response);

  absl::StatusOr<InMemoryHttpResponse> actual_response = callback.Response();
  ASSERT_OK(actual_response);
  EXPECT_THAT(*actual_response, FieldsAre(expected_code, IsEmpty(), IsEmpty()));
}

TEST(InMemoryHttpRequestCallbackTest,
     TestOkResponsWithAcceptEncodingRequestHeader) {
  const std::string expected_content_encoding = "foo";
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create(
          "https://valid.com", HttpRequest::Method::kGet,
          {{"Accept-Encoding", expected_content_encoding}}, "");
  ASSERT_OK(request);

  // Note that the fake response contains a Content-Encoding header, which
  // should be allowed because the request contained an Accept-Encoding header.
  int expected_code = kHttpOk;
  auto fake_response = FakeHttpResponse(
      expected_code, {{"Some-Response-Header", "foo"},
                      {"Content-Encoding", expected_content_encoding}});
  const std::string expected_body = "response_body";

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  // We return the response body in one go (the "simple" case).
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, expected_body));
  callback.OnResponseCompleted(**request, fake_response);

  absl::StatusOr<InMemoryHttpResponse> actual_response = callback.Response();
  ASSERT_OK(actual_response);
  EXPECT_THAT(*actual_response,
              FieldsAre(expected_code, StrEq(expected_content_encoding),
                        StrEq(expected_body)));
}

TEST(InMemoryHttpRequestCallbackTest, NotFoundResponse) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  // Return an HTTP error response.
  auto fake_response = FakeHttpResponse(kHttpNotFound, {});
  const std::string expected_body = "response_body";

  InMemoryHttpRequestCallback callback;
  ASSERT_OK(callback.OnResponseStarted(**request, fake_response));
  ASSERT_OK(callback.OnResponseBody(**request, fake_response, expected_body));
  callback.OnResponseCompleted(**request, fake_response);

  absl::StatusOr<InMemoryHttpResponse> actual_response = callback.Response();
  EXPECT_THAT(actual_response, IsCode(NOT_FOUND));
  EXPECT_THAT(actual_response.status().message(), HasSubstr("404"));
}

class PerformRequestsTest : public testing::Test {
 protected:
  PerformRequestsTest()
      : interruptible_runner_(
            &mock_log_manager_, mock_should_abort_.AsStdFunction(),
            InterruptibleRunner::TimingConfig{
                .polling_period = absl::ZeroDuration(),
                .graceful_shutdown_period = absl::InfiniteDuration(),
                .extended_shutdown_period = absl::InfiniteDuration()},
            InterruptibleRunner::DiagnosticsConfig{
                .interrupted = ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP,
                .interrupt_timeout =
                    ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP_TIMED_OUT,
                .interrupted_extended = ProdDiagCode::
                    BACKGROUND_TRAINING_INTERRUPT_HTTP_EXTENDED_COMPLETED,
                .interrupt_timeout_extended = ProdDiagCode::
                    BACKGROUND_TRAINING_INTERRUPT_HTTP_EXTENDED_TIMED_OUT}) {}
  void SetUp() override {}

  NiceMock<MockLogManager> mock_log_manager_;
  NiceMock<MockFunction<bool()>> mock_should_abort_;
  InterruptibleRunner interruptible_runner_;
  StrictMock<MockHttpClient> mock_http_client_;
};

TEST_F(PerformRequestsTest, PerformRequestInMemoryOk) {
  std::string expected_request_body = "request_body";
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create(
          "https://valid.com", HttpRequest::Method::kPost,
          {{"Some-Request-Header", "foo"}}, expected_request_body);
  ASSERT_OK(request);

  int expected_response_code = kHttpOk;
  std::string expected_response_body = "response_body";
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(
                  FieldsAre((*request)->uri(), (*request)->method(),
                            Contains(Header{"Some-Request-Header", "foo"}),
                            StrEq(expected_request_body))))
      .WillOnce(Return(FakeHttpResponse(expected_response_code,
                                        {{"Some-Response-Header", "bar"}},
                                        expected_response_body)));

  int64_t bytes_received = 0;
  int64_t bytes_sent = 0;
  absl::StatusOr<InMemoryHttpResponse> result = PerformRequestInMemory(
      mock_http_client_, interruptible_runner_, std::move(*request),
      // We pass in non-null pointers for the network stats, to ensure they are
      // correctly updated.
      &bytes_received, &bytes_sent);
  ASSERT_OK(result);
  EXPECT_THAT(*result, FieldsAre(expected_response_code, IsEmpty(),
                                 StrEq(expected_response_body)));

  EXPECT_THAT(bytes_sent, Ne(bytes_received));
  EXPECT_THAT(bytes_sent, Ge(expected_request_body.size()));
  EXPECT_THAT(bytes_received, Ge(expected_response_body.size()));
}

TEST_F(PerformRequestsTest, PerformRequestInMemoryNotFound) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  int expected_response_code = kHttpNotFound;
  std::string expected_response_body = "response_body";
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(
                  FieldsAre((*request)->uri(), (*request)->method(), _, _)))
      .WillOnce(Return(FakeHttpResponse(expected_response_code, {},
                                        expected_response_body)));

  absl::StatusOr<InMemoryHttpResponse> result =
      PerformRequestInMemory(mock_http_client_, interruptible_runner_,
                             std::move(*request), nullptr, nullptr);
  EXPECT_THAT(result, IsCode(NOT_FOUND));
  EXPECT_THAT(result.status().message(), HasSubstr("404"));
}

TEST_F(PerformRequestsTest, PerformRequestInMemoryEarlyError) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kPost, {}, "");
  ASSERT_OK(request);

  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(
                  FieldsAre((*request)->uri(), (*request)->method(), _, _)))
      // Make the call to PerformSingleRequest (and therefore the call to
      // HttpClient::PerformRequests) fail as a whole (rather than having the
      // individual request return a failure).
      .WillOnce(Return(absl::InvalidArgumentError("PerformRequests failed")));

  int64_t bytes_received = 0;
  int64_t bytes_sent = 0;
  absl::StatusOr<InMemoryHttpResponse> result =
      PerformRequestInMemory(mock_http_client_, interruptible_runner_,
                             std::move(*request), &bytes_received, &bytes_sent);
  EXPECT_THAT(result, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(result.status().message(), HasSubstr("PerformRequests failed"));

  // We know that MockHttpClient will have updated the 'sent' network stat
  // before having called into PerformSingleRequest, so that should be
  // reflected.
  EXPECT_THAT(bytes_sent, Ge(0));
  // The 'received' network stat should be 0 OTOH, since issuing the request
  // failed.
  EXPECT_EQ(bytes_received, 0);
}

// Tests the case where the request gets interrupted.
TEST_F(PerformRequestsTest, PerformRequestInMemoryCancellation) {
  absl::StatusOr<std::unique_ptr<HttpRequest>> request =
      InMemoryHttpRequest::Create("https://valid.com",
                                  HttpRequest::Method::kGet, {}, "");
  ASSERT_OK(request);

  absl::Notification request_issued;
  // We expect one calls to the cancellation listener.
  absl::BlockingCounter counter_should_abort(1);
  // When the HttpClient receives a HttpRequestHandle::Cancel call, we decrement
  // the counter.
  mock_http_client_.SetCancellationListener(
      [&counter_should_abort]() { counter_should_abort.DecrementCount(); });

  // Make HttpClient::PerformRequests() block until the counter is decremented.
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(
                  FieldsAre((*request)->uri(), (*request)->method(), _, _)))
      .WillOnce([&request_issued, &counter_should_abort](
                    MockableHttpClient::SimpleHttpRequest ignored) {
        request_issued.Notify();
        counter_should_abort.Wait();
        return FakeHttpResponse(503, {}, "");
      });
  // Make should_abort return false until we know that the request was issued
  // (i.e. once InterruptibleRunner has actually started running the code it was
  // given), and then make it return true, triggering an abort sequence and
  // unblocking the PerformRequests() call we caused to block above.
  EXPECT_CALL(mock_should_abort_, Call()).WillRepeatedly([&request_issued] {
    return request_issued.HasBeenNotified();
  });

  EXPECT_CALL(mock_log_manager_,
              LogDiag(ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP));

  int64_t bytes_received = 0;
  int64_t bytes_sent = 0;
  // The request should result in a CANCELLED outcome.
  absl::StatusOr<InMemoryHttpResponse> result =
      PerformRequestInMemory(mock_http_client_, interruptible_runner_,
                             std::move(*request), &bytes_received, &bytes_sent);

  EXPECT_THAT(result, IsCode(CANCELLED));
  EXPECT_THAT(result.status().message(),
              HasSubstr("cancelled after graceful wait"));

  // The network stats should still have been updated though (to reflect the
  // data sent and received up until the point of interruption).
  EXPECT_THAT(bytes_sent, Ge(0));
  EXPECT_THAT(bytes_received, Ge(0));
}

// Tests the case where a zero-length vector of UriOrInlineData is passed in. It
// should result in a zero-length result vector (as opposed to an error or a
// crash).
TEST_F(PerformRequestsTest, FetchResourcesInMemoryEmptyInputVector) {
  auto result = FetchResourcesInMemory(mock_http_client_, interruptible_runner_,
                                       {}, nullptr, nullptr);
  ASSERT_OK(result);
  EXPECT_THAT(*result, IsEmpty());
}

// Tests the case where both fields of UriOrInlineData are empty. The empty
// inline_data field should be returned.
TEST_F(PerformRequestsTest, FetchResourcesInMemoryEmptyUriAndInline) {
  auto result = FetchResourcesInMemory(
      mock_http_client_, interruptible_runner_,
      {UriOrInlineData::CreateInlineData(absl::Cord())}, nullptr, nullptr);
  ASSERT_OK(result);

  ASSERT_OK((*result)[0]);
  EXPECT_THAT(*(*result)[0], FieldsAre(kHttpOk, IsEmpty(), IsEmpty()));
}

// Tests the case where one of the URIs is invalid. The whole request should
// result in an error in that case.
TEST_F(PerformRequestsTest, FetchResourcesInMemoryInvalidUri) {
  auto result =
      FetchResourcesInMemory(mock_http_client_, interruptible_runner_,
                             {UriOrInlineData::CreateUri("https://valid.com"),
                              UriOrInlineData::CreateUri("http://invalid.com")},
                             nullptr, nullptr);
  EXPECT_THAT(result, IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(result.status().message(), HasSubstr("Non-HTTPS"));
}

// Tests the case where all of the requested resources must be fetched via URI.
TEST_F(PerformRequestsTest, FetchResourcesInMemoryAllUris) {
  const std::string uri1 = "https://valid.com/1";
  const std::string uri2 = "https://valid.com/2";
  const std::string uri3 = "https://valid.com/3";
  const std::string uri4 = "https://valid.com/4";
  auto resource1 = UriOrInlineData::CreateUri(uri1);
  auto resource2 = UriOrInlineData::CreateUri(uri2);
  auto resource3 = UriOrInlineData::CreateUri(uri3);
  auto resource4 = UriOrInlineData::CreateUri(uri4);

  int expected_response_code1 = kHttpOk;
  int expected_response_code2 = kHttpNotFound;
  int expected_response_code3 = kHttpServiceUnavailable;
  int expected_response_code4 = 204;  // "204 No Content"
  std::string expected_response_body1 = "response_body1";
  std::string expected_response_body4 = "response_body4";
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(
                  FieldsAre(uri1, HttpRequest::Method::kGet, HeaderList{}, "")))
      .WillOnce(Return(FakeHttpResponse(expected_response_code1, {},
                                        expected_response_body1)));
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(
                  FieldsAre(uri2, HttpRequest::Method::kGet, HeaderList{}, "")))
      .WillOnce(Return(FakeHttpResponse(expected_response_code2, {}, "")));
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(
                  FieldsAre(uri3, HttpRequest::Method::kGet, HeaderList{}, "")))
      .WillOnce(Return(FakeHttpResponse(expected_response_code3, {}, "")));
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(
                  FieldsAre(uri4, HttpRequest::Method::kGet, HeaderList{}, "")))
      .WillOnce(Return(FakeHttpResponse(expected_response_code4, {},
                                        expected_response_body4)));

  int64_t bytes_received = 0;
  int64_t bytes_sent = 0;
  auto result =
      FetchResourcesInMemory(mock_http_client_, interruptible_runner_,
                             {resource1, resource2, resource3, resource4},
                             // We pass in non-null pointers for the network
                             // stats, to ensure they are correctly updated.
                             &bytes_received, &bytes_sent);
  ASSERT_OK(result);

  ASSERT_OK((*result)[0]);
  EXPECT_THAT(*(*result)[0], FieldsAre(expected_response_code1, IsEmpty(),
                                       StrEq(expected_response_body1)));
  EXPECT_THAT((*result)[1], IsCode(NOT_FOUND));
  EXPECT_THAT((*result)[1].status().message(), HasSubstr("404"));
  EXPECT_THAT((*result)[2], IsCode(UNAVAILABLE));
  EXPECT_THAT((*result)[2].status().message(), HasSubstr("503"));
  ASSERT_OK((*result)[3]);
  EXPECT_THAT(*(*result)[3], FieldsAre(expected_response_code4, IsEmpty(),
                                       StrEq(expected_response_body4)));

  EXPECT_THAT(bytes_sent, Ne(bytes_received));
  EXPECT_THAT(bytes_sent, Ge(0));
  EXPECT_THAT(bytes_received, Ge(0));
}

// Tests the case where some of the requested resources have inline data
// available.
TEST_F(PerformRequestsTest, FetchResourcesInMemorySomeInlineData) {
  const std::string uri1 = "https://valid.com/1";
  const std::string uri3 = "https://valid.com/3";
  std::string expected_response_body2 = "response_body2";
  std::string expected_response_body4 = "response_body4";
  auto resource1 = UriOrInlineData::CreateUri(uri1);
  auto resource2 =
      UriOrInlineData::CreateInlineData(absl::Cord(expected_response_body2));
  auto resource3 = UriOrInlineData::CreateUri(uri3);
  auto resource4 =
      UriOrInlineData::CreateInlineData(absl::Cord(expected_response_body4));

  int expected_response_code1 = kHttpServiceUnavailable;
  int expected_response_code3 = 204;  // "204 No Content"
  std::string expected_response_body3 = "response_body3";
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(
                  FieldsAre(uri1, HttpRequest::Method::kGet, HeaderList{}, "")))
      .WillOnce(Return(FakeHttpResponse(expected_response_code1, {}, "")));
  EXPECT_CALL(mock_http_client_,
              PerformSingleRequest(
                  FieldsAre(uri3, HttpRequest::Method::kGet, HeaderList{}, "")))
      .WillOnce(Return(FakeHttpResponse(expected_response_code3, {},
                                        expected_response_body3)));

  int64_t bytes_received = 0;
  int64_t bytes_sent = 0;
  auto result =
      FetchResourcesInMemory(mock_http_client_, interruptible_runner_,
                             {resource1, resource2, resource3, resource4},
                             &bytes_received, &bytes_sent);
  ASSERT_OK(result);

  EXPECT_THAT((*result)[0], IsCode(UNAVAILABLE));
  EXPECT_THAT((*result)[0].status().message(), HasSubstr("503"));
  ASSERT_OK((*result)[1]);
  EXPECT_THAT(*(*result)[1],
              FieldsAre(kHttpOk, IsEmpty(), StrEq(expected_response_body2)));
  ASSERT_OK((*result)[2]);
  EXPECT_THAT(*(*result)[2], FieldsAre(expected_response_code3, IsEmpty(),
                                       StrEq(expected_response_body3)));
  ASSERT_OK((*result)[3]);
  EXPECT_THAT(*(*result)[3],
              FieldsAre(kHttpOk, IsEmpty(), StrEq(expected_response_body4)));

  EXPECT_THAT(bytes_sent, Ne(bytes_received));
  EXPECT_THAT(bytes_sent, Ge(0));
  EXPECT_THAT(bytes_received, Ge(0));
}

// Tests the case where all of the requested resources have inline data
// available (and hence no HTTP requests are expected to be issued).
TEST_F(PerformRequestsTest, FetchResourcesInMemoryOnlyInlineData) {
  std::string expected_response_body1 = "response_body1";
  std::string expected_response_body2 = "response_body2";
  auto resource1 =
      UriOrInlineData::CreateInlineData(absl::Cord(expected_response_body1));
  auto resource2 =
      UriOrInlineData::CreateInlineData(absl::Cord(expected_response_body2));

  int64_t bytes_received = 0;
  int64_t bytes_sent = 0;
  auto result = FetchResourcesInMemory(mock_http_client_, interruptible_runner_,
                                       {resource1, resource2}, &bytes_received,
                                       &bytes_sent);
  ASSERT_OK(result);

  ASSERT_OK((*result)[0]);
  EXPECT_THAT(*(*result)[0],
              FieldsAre(kHttpOk, IsEmpty(), StrEq(expected_response_body1)));
  ASSERT_OK((*result)[1]);
  EXPECT_THAT(*(*result)[1],
              FieldsAre(kHttpOk, IsEmpty(), StrEq(expected_response_body2)));

  // The network stats should be untouched, since no network requests were
  // issued.
  EXPECT_EQ(bytes_sent, 0);
  EXPECT_EQ(bytes_received, 0);
}

// Tests the case where the fetches get interrupted.
TEST_F(PerformRequestsTest, FetchResourcesInMemoryCancellation) {
  const std::string uri1 = "https://valid.com/1";
  const std::string uri2 = "https://valid.com/2";
  auto resource1 = UriOrInlineData::CreateUri(uri1);
  auto resource2 = UriOrInlineData::CreateUri(uri2);

  EXPECT_CALL(mock_http_client_, PerformSingleRequest(FieldsAre(uri1, _, _, _)))
      .WillOnce(Return(FakeHttpResponse(kHttpOk, {}, "")));

  absl::Notification request_issued;
  // We expect two calls to the cancellation listener, one for each request.
  absl::BlockingCounter counter_should_abort(2);
  // When the HttpClient receives a HttpRequestHandle::Cancel call, we decrement
  // the counter.
  mock_http_client_.SetCancellationListener(
      [&counter_should_abort]() { counter_should_abort.DecrementCount(); });

  // Make HttpClient::PerformRequests() block until the counter is decremented.
  EXPECT_CALL(mock_http_client_, PerformSingleRequest(FieldsAre(uri2, _, _, _)))
      .WillOnce([&request_issued, &counter_should_abort](
                    MockableHttpClient::SimpleHttpRequest ignored) {
        request_issued.Notify();
        counter_should_abort.Wait();
        return FakeHttpResponse(503, {}, "");
      });
  // Make should_abort return false until we know that the 2nd request was
  // issued (i.e. once InterruptibleRunner has actually started running the code
  // it was given), and then make it return true, triggering an abort sequence
  // and unblocking the PerformRequests() call we caused to block above.
  EXPECT_CALL(mock_should_abort_, Call()).WillRepeatedly([&request_issued] {
    return request_issued.HasBeenNotified();
  });

  EXPECT_CALL(mock_log_manager_,
              LogDiag(ProdDiagCode::BACKGROUND_TRAINING_INTERRUPT_HTTP));

  int64_t bytes_received = 0;
  int64_t bytes_sent = 0;
  // The request should result in an overall CANCELLED outcome.
  auto result = FetchResourcesInMemory(mock_http_client_, interruptible_runner_,
                                       {resource1, resource2}, &bytes_received,
                                       &bytes_sent);
  EXPECT_THAT(result, IsCode(CANCELLED));
  EXPECT_THAT(result.status().message(),
              HasSubstr("cancelled after graceful wait"));

  // The network stats should still have been updated though (to reflect the
  // data sent and received up until the point of interruption).
  EXPECT_THAT(bytes_sent, Ge(0));
  EXPECT_THAT(bytes_received, Ge(0));
}

}  // namespace
}  // namespace fcp::client::http
