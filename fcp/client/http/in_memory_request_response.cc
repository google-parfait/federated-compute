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
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/io/gzip_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/http/http_client_util.h"
#include "fcp/client/interruptible_runner.h"

namespace fcp {
namespace client {
namespace http {

namespace {
using ::google::protobuf::io::GzipOutputStream;
using ::google::protobuf::io::StringOutputStream;

absl::StatusOr<std::string> CompressWithGZip(
    const std::string& uncompressed_data) {
  int starting_pos = 0;
  size_t str_size = uncompressed_data.length();
  size_t in_size = str_size;

  std::string output;
  StringOutputStream string_output_stream(&output);
  GzipOutputStream::Options options;
  options.format = GzipOutputStream::GZIP;
  GzipOutputStream compressed_stream(&string_output_stream, options);
  void* out;
  int out_size;
  while (starting_pos < str_size) {
    if (!compressed_stream.Next(&out, &out_size) || out_size <= 0) {
      return absl::InternalError(
          absl::StrCat("An error has occurred during compression:",
                       compressed_stream.ZlibErrorMessage()));
    }

    if (in_size <= out_size) {
      uncompressed_data.copy(static_cast<char*>(out), in_size, starting_pos);
      // Ensure that the stream's output buffer is truncated to match the total
      // amount of data.
      compressed_stream.BackUp(out_size - static_cast<int>(in_size));
      break;
    }
    uncompressed_data.copy(static_cast<char*>(out), out_size, starting_pos);
    starting_pos += out_size;
    in_size -= out_size;
  }

  if (!compressed_stream.Close()) {
    return absl::InternalError(absl::StrCat(
        "Failed to close the stream: ", compressed_stream.ZlibErrorMessage()));
  }
  return output;
}

}  // namespace

absl::StatusOr<std::unique_ptr<HttpRequest>> InMemoryHttpRequest::Create(
    absl::string_view uri, HttpRequest::Method method, HeaderList extra_headers,
    std::string body, bool use_compression) {
  if (!absl::StartsWithIgnoreCase(uri, kHttpsScheme)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Non-HTTPS URIs are not supported: ", uri));
  }
  if (use_compression) {
    FCP_ASSIGN_OR_RETURN(body, CompressWithGZip(body));
    extra_headers.push_back({kContentEncodingHdr, kGzipEncodingHdrValue});
  }
  std::optional<std::string> content_length_hdr =
      FindHeader(extra_headers, kContentLengthHdr);
  if (content_length_hdr.has_value()) {
    return absl::InvalidArgumentError(
        "Content-Length header should not be provided!");
  }

  if (!body.empty()) {
    switch (method) {
      case HttpRequest::Method::kPost:
      case HttpRequest::Method::kPatch:
      case HttpRequest::Method::kPut:
      case HttpRequest::Method::kDelete:
        break;
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Request method does not allow request body: ", method));
    }
    // Add a Content-Length header, but only if there's a request body.
    extra_headers.push_back({kContentLengthHdr, std::to_string(body.size())});
  }

  return absl::WrapUnique(new InMemoryHttpRequest(
      uri, method, std::move(extra_headers), std::move(body)));
}

absl::StatusOr<int64_t> InMemoryHttpRequest::ReadBody(char* buffer,
                                                      int64_t requested) {
  // This method is called from the HttpClient's thread (we don't really care
  // which one). Hence, we use a mutex to ensure that subsequent calls to this
  // method see the modifications to cursor_.
  absl::WriterMutexLock _(&mutex_);

  // Check whether there's any bytes left to read, and indicate the end has been
  // reached if not.
  int64_t bytes_left = body_.size() - cursor_;
  if (bytes_left == 0) {
    return absl::OutOfRangeError("End of stream reached");
  }
  FCP_CHECK(buffer != nullptr);
  FCP_CHECK(requested > 0);
  // Calculate how much data we can return, based on the size of `buffer`.
  int64_t actual_read = bytes_left <= requested ? bytes_left : requested;
  std::memcpy(buffer, body_.data() + cursor_, actual_read);
  cursor_ += actual_read;
  return actual_read;
}

absl::Status InMemoryHttpRequestCallback::OnResponseStarted(
    const HttpRequest& request, const HttpResponse& response) {
  absl::WriterMutexLock _(&mutex_);
  response_code_ = response.code();

  std::optional<std::string> content_encoding_header =
      FindHeader(response.headers(), kContentEncodingHdr);
  if (content_encoding_header.has_value()) {
    // We don't expect the response body to be "Content-Encoding" encoded,
    // because the `HttpClient` is supposed to transparently handle the decoding
    // for us (unless we specified a "Accept-Encoding" header in the request,
    // which would indicate that we wanted to handle the response decoding).
    if (!FindHeader(request.extra_headers(), kAcceptEncodingHdr).has_value()) {
      // Note: technically, we should only receive Content-Encoding values that
      // match the Accept-Encoding values provided in the request headers. The
      // check above isn't quite that strict, but that's probably fine (since
      // such issues should be rare, and can be handled farther up the stack).
      status_ = absl::InvalidArgumentError(
          absl::StrCat("Unexpected header: ", kContentEncodingHdr));
      return status_;
    }
    content_encoding_ = *content_encoding_header;
  }

  // Similarly, we should under no circumstances receive a non-identity
  // Transfer-Encoding header, since the `HttpClient` is unconditionally
  // required to undo any such encoding for us.
  std::optional<std::string> transfer_encoding_header =
      FindHeader(response.headers(), kTransferEncodingHdr);
  if (transfer_encoding_header.has_value() &&
      absl::AsciiStrToLower(*transfer_encoding_header) !=
          kIdentityEncodingHdrValue) {
    status_ = absl::InvalidArgumentError(
        absl::StrCat("Unexpected header: ", kTransferEncodingHdr));
    return status_;
  }

  // If no Content-Length header is provided, this means that the server either
  // didn't provide one and is streaming the response, or that the HttpClient
  // implementation transparently decompressed the data for us and stripped the
  // Content-Length header (as per the HttpClient contract).
  std::optional<std::string> content_length_hdr =
      FindHeader(response.headers(), kContentLengthHdr);
  if (!content_length_hdr.has_value()) {
    return absl::OkStatus();
  }

  // A Content-Length header available. Let's parse it so that we know how much
  // data to expect.
  int64_t content_length;
  // Note that SimpleAtoi safely handles non-ASCII data.
  if (!absl::SimpleAtoi(*content_length_hdr, &content_length)) {
    status_ = absl::InvalidArgumentError(
        "Could not parse Content-Length response header");
    return status_;
  }
  if (content_length < 0) {
    status_ = absl::OutOfRangeError(absl::StrCat(
        "Invalid Content-Length response header: ", content_length));
    return status_;
  }
  expected_content_length_ = content_length;

  return absl::OkStatus();
}

void InMemoryHttpRequestCallback::OnResponseError(const HttpRequest& request,
                                                  const absl::Status& error) {
  absl::WriterMutexLock _(&mutex_);
  status_ = absl::Status(
      error.code(), absl::StrCat("Error receiving response headers (error: ",
                                 error.message(), ")"));
}

absl::Status InMemoryHttpRequestCallback::OnResponseBody(
    const HttpRequest& request, const HttpResponse& response,
    absl::string_view data) {
  // This runs on a thread chosen by the HttpClient implementation (i.e. it
  // could be our original thread, or a different one). Ensure that if
  // subsequent callbacks occur on different threads each thread sees the
  // previous threads' updates to response_buffer_.
  absl::WriterMutexLock _(&mutex_);

  // Ensure we're not receiving more data than expected.
  if (expected_content_length_.has_value() &&
      response_buffer_.size() + data.size() > *expected_content_length_) {
    status_ = absl::OutOfRangeError(absl::StrCat(
        "Too much response body data received (rcvd: ", response_buffer_.size(),
        ", new: ", data.size(), ", max: ", *expected_content_length_, ")"));
    return status_;
  }

  // Copy the data into the target buffer. Note that this means we'll always
  // store the response body as a number of memory fragments (rather than a
  // contiguous buffer). However, because HttpClient implementations are
  // encouraged to return response data in fairly large chunks, we don't expect
  // this too cause much overhead.
  response_buffer_.Append(data);

  return absl::OkStatus();
}

void InMemoryHttpRequestCallback::OnResponseBodyError(
    const HttpRequest& request, const HttpResponse& response,
    const absl::Status& error) {
  absl::WriterMutexLock _(&mutex_);
  status_ = absl::Status(
      error.code(),
      absl::StrCat("Error receiving response body (response code: ",
                   response.code(), ", error: ", error.message(), ")"));
}

void InMemoryHttpRequestCallback::OnResponseCompleted(
    const HttpRequest& request, const HttpResponse& response) {
  // Once the body has been received correctly, turn the response code into a
  // canonical code.
  absl::WriterMutexLock _(&mutex_);
  // Note: the case when too *much* response data is unexpectedly received is
  // handled in OnResponseBody (while this handles the case of too little data).
  if (expected_content_length_.has_value() &&
      response_buffer_.size() != *expected_content_length_) {
    status_ = absl::InvalidArgumentError(
        absl::StrCat("Too little response body data received (rcvd: ",
                     response_buffer_.size(),
                     ", expected: ", *expected_content_length_, ")"));
    return;
  }
  // If we received the expected amount of data, then assign the overall
  // response Status a value based on the HTTP code.
  status_ = ConvertHttpCodeToStatus(*response_code_);
}

absl::StatusOr<InMemoryHttpResponse> InMemoryHttpRequestCallback::Response()
    const {
  absl::ReaderMutexLock _(&mutex_);
  FCP_RETURN_IF_ERROR(status_);
  // If status_ is OK, then response_code_ and response_headers_ are guaranteed
  // to have values.
  return InMemoryHttpResponse{*response_code_, content_encoding_,
                              response_buffer_};
}

absl::StatusOr<InMemoryHttpResponse> PerformRequestInMemory(
    HttpClient& http_client, InterruptibleRunner& interruptible_runner,
    std::unique_ptr<http::HttpRequest> request, int64_t* bytes_received_acc,
    int64_t* bytes_sent_acc) {
  // Note: we must explicitly instantiate a vector here as opposed to passing an
  // initializer list to PerformRequestsInMemory, because initializer lists do
  // not support move-only values.
  std::vector<std::unique_ptr<http::HttpRequest>> requests;
  requests.push_back(std::move(request));
  FCP_ASSIGN_OR_RETURN(
      auto result, PerformMultipleRequestsInMemory(
                       http_client, interruptible_runner, std::move(requests),
                       bytes_received_acc, bytes_sent_acc));
  return std::move(result[0]);
}

absl::StatusOr<std::vector<absl::StatusOr<InMemoryHttpResponse>>>
PerformMultipleRequestsInMemory(
    HttpClient& http_client, InterruptibleRunner& interruptible_runner,
    std::vector<std::unique_ptr<http::HttpRequest>> requests,
    int64_t* bytes_received_acc, int64_t* bytes_sent_acc) {
  // A vector that will own the request handles and callbacks (and will
  // determine their lifetimes).
  std::vector<std::pair<std::unique_ptr<HttpRequestHandle>,
                        std::unique_ptr<InMemoryHttpRequestCallback>>>
      handles_and_callbacks;
  handles_and_callbacks.reserve(requests.size());

  // An accompanying vector that contains just the raw pointers, for passing to
  // `HttpClient::PerformRequests`.
  std::vector<std::pair<HttpRequestHandle*, HttpRequestCallback*>>
      handles_and_callbacks_ptrs;
  handles_and_callbacks_ptrs.reserve(requests.size());

  // Enqueue each request, and create a simple callback for each request which
  // will simply buffer the response body in-memory and allow us to consume that
  // buffer once all requests have finished.
  for (std::unique_ptr<HttpRequest>& request : requests) {
    std::unique_ptr<HttpRequestHandle> handle =
        http_client.EnqueueRequest(std::move(request));
    auto callback = std::make_unique<InMemoryHttpRequestCallback>();
    handles_and_callbacks_ptrs.push_back({handle.get(), callback.get()});
    handles_and_callbacks.push_back({std::move(handle), std::move(callback)});
  }

  // Issue the requests in one call (allowing the HttpClient to issue them
  // concurrently), in an interruptible fashion.
  absl::Status result = interruptible_runner.Run(
      [&http_client, &handles_and_callbacks_ptrs]() {
        return http_client.PerformRequests(handles_and_callbacks_ptrs);
      },
      [&handles_and_callbacks_ptrs] {
        // If we get aborted then call HttpRequestHandle::Cancel on all handles.
        // This should result in the PerformRequests call returning early and
        // InterruptibleRunner::Run returning CANCELLED.
        for (auto [handle, callback] : handles_and_callbacks_ptrs) {
          handle->Cancel();
        }
      });
  // Update the network stats *before* we return (just in case a failed
  // `PerformRequests` call caused some network traffic to have been sent
  // anyway).
  for (auto& [handle, callback] : handles_and_callbacks) {
    if (bytes_received_acc != nullptr) {
      *bytes_received_acc += handle->TotalReceivedBytes();
    }
    if (bytes_sent_acc != nullptr) {
      *bytes_sent_acc += handle->TotalSentBytes();
    }
  }

  FCP_RETURN_IF_ERROR(result);

  // Gather and return the results.
  std::vector<absl::StatusOr<InMemoryHttpResponse>> results;
  results.reserve(handles_and_callbacks.size());
  for (auto& [handle, callback] : handles_and_callbacks) {
    results.push_back(callback->Response());
  }
  return results;
}

absl::StatusOr<std::vector<absl::StatusOr<InMemoryHttpResponse>>>
FetchResourcesInMemory(HttpClient& http_client,
                       InterruptibleRunner& interruptible_runner,
                       const std::vector<UriOrInlineData>& resources,
                       int64_t* bytes_received_acc, int64_t* bytes_sent_acc) {
  // Each resource may have the data already available (by having been included
  // in a prior response inline), or may need to be fetched.

  // We'll create an 'accessor' for each resource, providing access to that
  // resource's data by the end of this function (either the fetched data, or
  // the inline data).
  std::vector<std::function<absl::StatusOr<InMemoryHttpResponse>()>>
      response_accessors;

  // We'll compile HttpRequest instances for those resources that do need to be
  // fetched, then we'll fire them off all at once, and then we'll gather their
  // responses once all requests have finished.
  std::vector<std::unique_ptr<http::HttpRequest>> http_requests;
  std::vector<absl::StatusOr<InMemoryHttpResponse>> http_responses;

  for (const UriOrInlineData& resource : resources) {
    if (!resource.uri().empty()) {
      // If the resource URI is set, then create a request to fetch the data for
      // it, and point the accessor at the slot in http_responses where that
      // request's response will eventually live.
      FCP_ASSIGN_OR_RETURN(
          std::unique_ptr<http::HttpRequest> request,
          InMemoryHttpRequest::Create(resource.uri(), HttpRequest::Method::kGet,
                                      {}, "", /*use_compression=*/false));
      http_requests.push_back(std::move(request));
      int64_t response_index = http_requests.end() - http_requests.begin() - 1;
      response_accessors.push_back([&http_responses, response_index]() {
        return std::move(http_responses.at(response_index));
      });
    } else {
      // The data is available inline. Make the accessor just return a "fake"
      // successful HTTP response (that way the caller can have unified error
      // handling logic and doesn't have to know whether a resource was truly
      // fetched via HTTP or not). Because the inline_data field is an
      // absl::Cord, making a copy of it should be very cheap.
      response_accessors.push_back([resource]() {
        return InMemoryHttpResponse{kHttpOk, "", resource.inline_data()};
      });
    }
  }

  // Perform the requests.
  auto resource_fetch_result = PerformMultipleRequestsInMemory(
      http_client, interruptible_runner, std::move(http_requests),
      bytes_received_acc, bytes_sent_acc);
  // Check whether issuing the requests failed as a whole (generally indicating
  // a programming error).
  FCP_RETURN_IF_ERROR(resource_fetch_result);
  http_responses = std::move(*resource_fetch_result);

  // Compile the result vector by getting each resource's response using the
  // corresponding accessor.
  // Note that the order of results returned corresponds to the order of
  // resources in the vector we originally received.
  std::vector<absl::StatusOr<InMemoryHttpResponse>> result;
  result.reserve(response_accessors.size());
  for (const auto& response_accessor : response_accessors) {
    result.push_back(response_accessor());
  }
  return result;
}

}  // namespace http
}  // namespace client
}  // namespace fcp
