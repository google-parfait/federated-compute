/*
 * Copyright 2022 Google LLC
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

#include "fcp/client/http/curl/curl_http_client.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fcp/client/http/in_memory_request_response.h"
#include "fcp/client/http/testing/http_test_server.h"
#include "fcp/client/http/testing/test_helpers.h"

namespace fcp::client::http::curl {
namespace {
using ::testing::_;
using ::testing::StrictMock;

void SetUpGetRequestCallback(
    StrictMock<MockHttpRequestCallback>* request_callback, int port,
    const std::string& request_uri, const std::string& expected_response_body,
    size_t& total_bytes_downloaded) {
  EXPECT_CALL(*request_callback, OnResponseStarted(_, _))
      .WillOnce(::testing::Invoke([&request_uri](const HttpRequest& request,
                                                 const HttpResponse& response) {
        EXPECT_THAT(request.uri(), request_uri);
        EXPECT_THAT(request.method(), HttpRequest::Method::kGet);
        EXPECT_THAT(request.extra_headers().size(), 0);
        EXPECT_THAT(request.HasBody(), false);

        EXPECT_THAT(response.code(), 200);
        EXPECT_THAT(response.headers().size(), ::testing::Gt(0));
        return absl::OkStatus();
      }));

  EXPECT_CALL(*request_callback, OnResponseBody(_, _, _))
      .WillOnce(::testing::Invoke(
          [=, &total_bytes_downloaded](const HttpRequest& request,
                                       const HttpResponse& response,
                                       absl::string_view data) {
            EXPECT_THAT(request.uri(), request_uri);
            EXPECT_THAT(request.method(), HttpRequest::Method::kGet);
            EXPECT_THAT(request.extra_headers().size(), 0);
            EXPECT_THAT(request.HasBody(), false);

            total_bytes_downloaded += data.size();

            EXPECT_THAT(data, expected_response_body);
            return absl::OkStatus();
          }));

  EXPECT_CALL(*request_callback, OnResponseCompleted(_, _))
      .WillOnce(::testing::Invoke([&request_uri](const HttpRequest& request,
                                                 const HttpResponse& response) {
        EXPECT_THAT(request.uri(), request_uri);
        EXPECT_THAT(request.method(), HttpRequest::Method::kGet);
        EXPECT_THAT(request.extra_headers().size(), 0);
        EXPECT_THAT(request.HasBody(), false);

        EXPECT_THAT(response.code(), 200);
        EXPECT_THAT(response.headers().size(), ::testing::Gt(0));
      }));
}

void SetUpPostRequestCallback(
    StrictMock<MockHttpRequestCallback>* request_callback, int port,
    const std::string& request_uri, const std::string& expected_response_body,
    size_t& total_bytes_downloaded) {
  EXPECT_CALL(*request_callback, OnResponseStarted(_, _))
      .WillOnce(::testing::Invoke([&request_uri](const HttpRequest& request,
                                                 const HttpResponse& response) {
        EXPECT_THAT(request.uri(), request_uri);
        EXPECT_THAT(request.method(), HttpRequest::Method::kPost);
        EXPECT_THAT(request.extra_headers().size(), 1);
        EXPECT_THAT(request.HasBody(), true);

        EXPECT_THAT(response.code(), 200);
        EXPECT_THAT(response.headers().size(), ::testing::Gt(0));
        return absl::OkStatus();
      }));

  EXPECT_CALL(*request_callback, OnResponseBody(_, _, _))
      .WillOnce(::testing::Invoke(
          [=, &total_bytes_downloaded](const HttpRequest& request,
                                       const HttpResponse& response,
                                       absl::string_view data) {
            EXPECT_THAT(request.uri(), request_uri);
            EXPECT_THAT(request.method(), HttpRequest::Method::kPost);
            EXPECT_THAT(request.extra_headers().size(), 1);
            EXPECT_THAT(request.HasBody(), true);

            total_bytes_downloaded += data.size();

            EXPECT_THAT(data, expected_response_body);
            return absl::OkStatus();
          }));

  EXPECT_CALL(*request_callback, OnResponseCompleted(_, _))
      .WillOnce(::testing::Invoke([&request_uri](const HttpRequest& request,
                                                 const HttpResponse& response) {
        EXPECT_THAT(request.uri(), request_uri);
        EXPECT_THAT(request.method(), HttpRequest::Method::kPost);
        EXPECT_THAT(request.extra_headers().size(), 1);
        EXPECT_THAT(request.HasBody(), true);

        EXPECT_THAT(response.code(), 200);
        EXPECT_THAT(response.headers().size(), ::testing::Gt(0));
      }));
}

void PerformTwoRequests(CurlHttpClient* http_client, int port,
                        const std::string& request_uri1,
                        const std::string& request_uri2) {
  std::string request1_body;
  std::string request2_body = "test: 123-45";

  auto request1 = InMemoryHttpRequest::Create(
      request_uri1, HttpRequest::Method::kGet, HeaderList(), request1_body,
      /*use_compression*/ false);
  ASSERT_OK(request1);

  auto request2 = InMemoryHttpRequest::Create(
      request_uri2, HttpRequest::Method::kPost, HeaderList(), request2_body,
      /*use_compression*/ false);
  ASSERT_OK(request2);

  auto handle1 = http_client->EnqueueRequest(std::move(request1.value()));
  auto handle2 = http_client->EnqueueRequest(std::move(request2.value()));

  auto request_callback1 =
      std::make_unique<StrictMock<MockHttpRequestCallback>>();
  auto request_callback2 =
      std::make_unique<StrictMock<MockHttpRequestCallback>>();

  size_t total_bytes_downloaded_handle1 = 0;
  size_t total_bytes_downloaded_handle2 = 0;

  auto expected_response_body1 = absl::StrCat(
      "HTTP Method: GET\n", "Request Uri: /test\n", "Request Headers:\n",
      "Host: localhost:", port, "\nAccept: */*\n", "Accept-Encoding: gzip\n",
      "Request Body:\n");

  auto expected_response_body2 = absl::StrCat(
      "HTTP Method: POST\nRequest Uri: /test\n",
      "Request Headers:\nHost: localhost:", port,
      "\nAccept: */*\nAccept-Encoding: gzip\n", "Content-Length: 12\n",
      "Content-Type: application/x-www-form-urlencoded\n",
      "Request Body:\ntest: 123-45");

  SetUpGetRequestCallback(request_callback1.get(), port, request_uri1,
                          expected_response_body1,
                          total_bytes_downloaded_handle1);
  SetUpPostRequestCallback(request_callback2.get(), port, request_uri2,
                           expected_response_body2,
                           total_bytes_downloaded_handle2);

  std::vector<std::pair<HttpRequestHandle*, HttpRequestCallback*>> requests{
      std::make_pair(handle1.get(), request_callback1.get()),
      std::make_pair(handle2.get(), request_callback2.get())};

  EXPECT_THAT(handle1->TotalSentBytes(), 0);
  EXPECT_THAT(handle1->TotalReceivedBytes(), 0);
  EXPECT_THAT(handle2->TotalSentBytes(), 0);
  EXPECT_THAT(handle2->TotalReceivedBytes(), 0);

  absl::Status status = http_client->PerformRequests(requests);

  EXPECT_THAT(status, absl::OkStatus());
  EXPECT_THAT(handle1->TotalSentBytes(), request1_body.size());
  EXPECT_THAT(handle1->TotalReceivedBytes(), total_bytes_downloaded_handle1);
  EXPECT_THAT(handle2->TotalSentBytes(), request2_body.size());
  EXPECT_THAT(handle2->TotalReceivedBytes(), total_bytes_downloaded_handle2);
}

// Runs PerformRequests once in one thread.
TEST(CurlHttpClientTest, PerformTwoRequestsInParallelInOneThread) {
  const int port = 4568;
  const std::string request_uri =
      absl::StrCat("http://localhost:", port, "/test");

  auto curl_api = std::make_unique<CurlApi>();

  auto http_client = std::make_unique<CurlHttpClient>(curl_api.get());

  auto http_server = CreateHttpTestServer("/test", port, /*num_threads*/ 5);
  EXPECT_THAT(http_server.ok(), true);
  EXPECT_THAT(http_server.value()->StartAcceptingRequests(), true);

  PerformTwoRequests(http_client.get(), port, request_uri, request_uri);

  curl_api.reset();
  http_server.value()->Terminate();
  http_server.value()->WaitForTermination();
}

// Runs PerformRequests in five times in five threads.
TEST(CurlHttpClientTest, PerformTwoRequestsInParallelFiveTimesInFiveThreads) {
  const int port = 4568;
  const std::string request_uri =
      absl::StrCat("http://localhost:", port, "/test");

  auto curl_api = std::make_unique<CurlApi>();

  auto http_client = std::make_unique<CurlHttpClient>(curl_api.get());

  auto http_server = CreateHttpTestServer("/test", port, /*num_threads*/ 10);
  EXPECT_THAT(http_server.ok(), true);
  EXPECT_THAT(http_server.value()->StartAcceptingRequests(), true);

  auto thread_pool_scheduler = CreateThreadPoolScheduler(5);

  thread_pool_scheduler->Schedule([&http_client, request_uri]() {
    PerformTwoRequests(http_client.get(), port, request_uri, request_uri);
  });
  thread_pool_scheduler->Schedule([&http_client, request_uri]() {
    PerformTwoRequests(http_client.get(), port, request_uri, request_uri);
  });
  thread_pool_scheduler->Schedule([&http_client, request_uri]() {
    PerformTwoRequests(http_client.get(), port, request_uri, request_uri);
  });
  thread_pool_scheduler->Schedule([&http_client, request_uri]() {
    PerformTwoRequests(http_client.get(), port, request_uri, request_uri);
  });
  thread_pool_scheduler->Schedule([&http_client, request_uri]() {
    PerformTwoRequests(http_client.get(), port, request_uri, request_uri);
  });

  thread_pool_scheduler->WaitUntilIdle();

  curl_api.reset();
  http_server.value()->Terminate();
  http_server.value()->WaitForTermination();
}

// Runs PerformRequests with two requests and cancels the second after
// OnResponseStarted received.
TEST(CurlHttpClientTest, CancelRequest) {
  const int port = 4568;
  const std::string request_uri1 =
      absl::StrCat("http://localhost:", port, "/test");
  const std::string request_uri2 =
      absl::StrCat("http://localhost:", port, "/test");  //"/wait-one-min");

  auto curl_api = std::make_unique<CurlApi>();

  auto http_client = std::make_unique<CurlHttpClient>(curl_api.get());

  auto http_server = CreateHttpTestServer("/test", port, /*num_threads*/ 5);
  EXPECT_THAT(http_server.ok(), true);
  EXPECT_THAT(http_server.value()->StartAcceptingRequests(), true);

  std::string request1_body;
  std::string request2_body = "test: 123-45";

  auto request1 = InMemoryHttpRequest::Create(
      request_uri1, HttpRequest::Method::kGet, HeaderList(), request1_body,
      /*use_compression*/ false);
  ASSERT_OK(request1);

  auto request2 = InMemoryHttpRequest::Create(
      request_uri2, HttpRequest::Method::kPost, HeaderList(), request2_body,
      /*use_compression*/ false);
  ASSERT_OK(request2);

  auto handle1 = http_client->EnqueueRequest(std::move(request1.value()));
  auto handle2 = http_client->EnqueueRequest(std::move(request2.value()));

  auto request_callback1 =
      std::make_unique<StrictMock<MockHttpRequestCallback>>();
  auto request_callback2 =
      std::make_unique<StrictMock<MockHttpRequestCallback>>();

  size_t total_bytes_downloaded_handle1 = 0;

  auto expected_response_body = absl::StrCat(
      "HTTP Method: GET\n", "Request Uri: /test\n", "Request Headers:\n",
      "Host: localhost:", port, "\nAccept: */*\n", "Accept-Encoding: gzip\n",
      "Request Body:\n");

  SetUpGetRequestCallback(request_callback1.get(), port, request_uri1,
                          expected_response_body,
                          total_bytes_downloaded_handle1);

  EXPECT_CALL(*request_callback2, OnResponseStarted(_, _))
      .WillOnce(::testing::Invoke(
          [&handle2](const HttpRequest& request, const HttpResponse& response) {
            handle2->Cancel();
            return absl::OkStatus();
          }));

  EXPECT_CALL(*request_callback2, OnResponseBody(_, _, _))
      .WillOnce(::testing::Invoke([&request_uri2](const HttpRequest& request,
                                                  const HttpResponse& response,
                                                  absl::string_view data) {
        EXPECT_THAT(request.uri(), request_uri2);
        EXPECT_THAT(request.method(), HttpRequest::Method::kPost);
        EXPECT_THAT(request.extra_headers().size(), 1);
        EXPECT_THAT(request.HasBody(), true);

        auto expected_response_body = absl::StrCat(
            "HTTP Method: POST\nRequest Uri: /test\n",
            "Request Headers:\nHost: localhost:", port,
            "\nAccept: */*\nAccept-Encoding: gzip\n", "Content-Length: 12\n",
            "Content-Type: application/x-www-form-urlencoded\n",
            "Request Body:\ntest: 123-45");

        EXPECT_THAT(data, expected_response_body);
        return absl::OkStatus();
      }));

  EXPECT_CALL(*request_callback2, OnResponseBodyError(_, _, _))
      .WillOnce(::testing::Invoke([&request_uri2](const HttpRequest& request,
                                                  const HttpResponse& response,
                                                  const absl::Status& error) {
        EXPECT_THAT(request.uri(), request_uri2);
        EXPECT_THAT(request.method(), HttpRequest::Method::kPost);
        EXPECT_THAT(request.extra_headers().size(), 1);
        EXPECT_THAT(request.HasBody(), true);
      }));

  std::vector<std::pair<HttpRequestHandle*, HttpRequestCallback*>> requests{
      std::make_pair(handle1.get(), request_callback1.get()),
      std::make_pair(handle2.get(), request_callback2.get())};

  auto thread_pool_scheduler = CreateThreadPoolScheduler(2);
  thread_pool_scheduler->Schedule([&http_client, &requests]() {
    absl::Status status = http_client->PerformRequests(requests);

    EXPECT_THAT(status, absl::OkStatus());
  });

  thread_pool_scheduler->WaitUntilIdle();

  EXPECT_THAT(handle1->TotalSentBytes(), request1_body.size());
  EXPECT_THAT(handle1->TotalReceivedBytes(), total_bytes_downloaded_handle1);

  curl_api.reset();
  http_server.value()->Terminate();
  http_server.value()->WaitForTermination();
}

}  // namespace
}  // namespace fcp::client::http::curl
