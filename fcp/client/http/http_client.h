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
#ifndef FCP_CLIENT_HTTP_HTTP_CLIENT_H_
#define FCP_CLIENT_HTTP_HTTP_CLIENT_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace fcp {
namespace client {
namespace http {

using Header = std::pair<std::string, std::string>;
// This is a vector of pairs and not a map since multiple request headers with
// the same name are allowed (see RFC2616 section 4.2).
using HeaderList = std::vector<Header>;

class Request;          // forward declaration
class RequestCallback;  // forward declaration
class RequestHandle;    // forward declaration
class Response;         // forward declaration

// An interface that allows the callers to make HTTP requests and receive their
// responses.
//
// Platforms will be required to pass an instance of this class to
// `RunFederatedComputation(...)`, and such instances must remain alive for at
// least the duration of those calls.
//
// Instances of this class must support being called from any thread. In such
// cases implementations should ideally handle multiple requests in parallel, at
// least up to an implementation-defined max number of parallel requests (i.e.
// ideally two `PerformRequests` calls on different threads can be handled in
// parallel, rather than having the 2nd call block until the 1st call is
// completed).
//
// TODO(team): Submit additional documentation on the various
// requirements HttpClient implementations must adhere to, but that aren't
// listed here yet (details on protocol, TLS validation, compression, etc.).
class HttpClient {
 public:
  virtual ~HttpClient() = default;

  // Enqueues an HTTP request, without starting it yet. To start the request the
  // `RequestHandle` must be passed to `PerformRequests`. Each `RequestHandle`
  // must be passed to at most one `PerformRequests` call.
  //
  // The `HttpClient` implementation assumes ownership of the `Request` object,
  // and the implementation must delete the object when the `RequestHandle` is
  // deleted.
  virtual std::unique_ptr<RequestHandle> EnqueueRequest(
      std::unique_ptr<Request> request) = 0;

  // Performs the given requests. Results will be returned to each
  // corresponding`RequestCallback` while this method is blocked. This method
  // must block until all requests have finished or have been cancelled, and
  // until all corresponding request callbacks have returned.
  //
  // By decoupling the enqueueing and starting of (groups of) requests,
  // implementations may be able to handle concurrent requests more optimally
  // (e.g. by issuing them over a shared HTTP connection). Having separate
  // per-request `RequestHandle` objects also makes it easier to support
  // canceling specific requests, releasing resources for specific requests,
  // accessing stats for specific requests, etc.
  //
  // The `RequestHandle` and `RequestCallback` instances must outlive the call
  // to `PerformRequests`, but may be deleted any time after this call has
  // returned.
  //
  // Returns an `INVALID_ARGUMENT` error if a `RequestHandle` was previously
  // already passed to another `PerformRequests` call.
  virtual absl::Status PerformRequests(
      std::vector<std::pair<RequestHandle*, RequestCallback*>> requests) = 0;
};

// An HTTP request for a single resource. Implemented by the caller of
// `HttpClient`.
//
// Once instances are passed to `EnqueueRequest`, their lifetime is managed by
// the `HttpClient` implementation. Implementations must tie the `Request`
// instance lifetime to the lifetime of the `RequestHandle` they return (i.e.
// they should delete the `Request` from the `RequestHandle` destructor).
//
// Methods of this class may get called from any thread (and subsequent calls
// are not required to all happen on the same thread).
class Request {
 public:
  // Note: the request methods imply a set of standard request properties such
  // as cacheability, safety, and idempotency:
  // - GET: cacheable, idempotent, safe (readonly).
  // - POST: non-cacheable, non-idempotent, non-safe.
  // - PUT: non-cacheable, idempotent, non-safe.
  // See for example https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods.
  //
  // The caller of `HttpClient` may implement its own caching layer in the
  // future, so implementations are not expected to cache cacheable requests
  // (although they are technically allowed to).
  //
  // Implementations should not automatically retry requests, even if the
  // request method implies it is safe or idempotent. The caller of `HttpClient`
  // will own the responsibility for retrying requests.
  enum class Method { kGet, kPost, kPut, kPatch, kHead };

  // Must not be called until any corresponding `RequestHandle` has been
  // deleted.
  virtual ~Request() = default;

  // The URI to request. Will always have an "https://" scheme (but this may be
  // extended in the future).
  virtual absl::string_view uri() const = 0;

  // The HTTP method to use for this request.
  virtual Method method() const = 0;

  // Extra request headers to include with this request, in addition to any
  // headers specified by the `HttpClient` implementation.
  virtual const HeaderList& extra_request_headers() const = 0;

  // Returns true if the request has a request body (which can be read using
  // `ReadBody`). If the request body payload size is known ahead of time, then
  // the "Content-Length" header will be set in `extra_request_headers()`. If it
  // isn't known yet then the `HttpClient` implementation should use the
  // "Transfer-Encoding: chunked" encoding to transmit the request body to the
  // server in chunks.
  virtual bool HasRequestBody() const = 0;

  // Requests that up to `requested` bytes of the request body be read into
  // `buffer`, and that the actual amount of bytes read is returned. The caller
  // retains ownership of the buffer.
  //
  // Callees must return at least 1 byte, but may otherwise return less than the
  // requested amount of data, if more data isn't available yet. Callees should
  // return `OUT_OF_RANGE` when the end of data has been reached, in which case
  // `buffer` should not be modified.
  //
  // Callees should return data ASAP, as delaying this for too long may cause
  // the network stream to fall idle/run out of data to transmit.
  //
  // May also return other errors, in which case the request will be cancelled.
  virtual absl::StatusOr<int64_t> ReadBody(char* buffer, int64_t requested) = 0;
};

// A handle to a pending `Request`, allowing a caller of `HttpClient` to access
// stats for the request or to cancel ongoing requests. Implemented by the
// `HttpClient` implementer.
//
// The lifetimes of instances of this class are owned by the caller of
// `HttpClient`.
//
// Methods of this class may get called from any thread (and subsequent calls
// are not required to all happen on the same thread).
class RequestHandle {
 public:
  // When this is called, `HttpClient` implementations should delete all their
  // owned resources as well as the associated `Request`.
  virtual ~RequestHandle() = default;

  // The total amount of data sent/received over the network for this request up
  // to this point. These numbers should reflect as close as possible the amount
  // of bytes sent "over the wire". This means, for example, that if the data is
  // compressed or if a `Transfer-Encoding` is used, the numbers should reflect
  // the compressed and/or encoded size of the data. One exception to this rule
  // applies to the use of TLS: implementations are allowed to account for the
  // overhead of TLS encoding in these numbers, but are not required to (since
  // many HTTP libraries do not provide stats at that level of granularity).
  //
  // If the request was served from a cache then this should reflect only the
  // actual bytes sent over the network (e.g. 0 if returned from disk directly,
  // or if a cache validation request was sent only those bytes used by the
  // validation request/response).
  //
  // If the request involved redirects, the numbers returned here should include
  // the bytes sent/received for those redirects, if the implementation supports
  // this. Otherwise they are allowed to reflect only the final
  // request/response's bytes sent/received.
  //
  // Implementations should strive to return as up-to-date numbers are possible
  // from these methods (e.g. ideally the 'sent' number should reflect the
  // amount of request body data that has been uploaded so far, even if the
  // upload hasn't completed fully yet; similarly the 'received' number should
  // reflect the amount of response body data received so far, even if the
  // response hasn't been fully received yet).
  virtual int64_t TotalSentBytes() const = 0;
  virtual int64_t TotalReceivedBytes() const = 0;

  // Used to indicate that the request should be cancelled and that
  // implementations may release resources associated with this request (e.g.
  // the socket used by the request).
  //
  // Callers are still only allowed to delete this instance once after any
  // corresponding `PerformRequests()` call has completed, and not before.
  //
  // If a `PerformRequests` call is ongoing for this handle, then the
  // corresponding `RequestCallback` instance may still receive further method
  // invocations after this call returns (e.g. because an invocation may already
  // have been in flight).
  //
  // If a `PerformRequests` call is ongoing for this handle, and if the
  // `RequestCallback::OnResponseStarted` method was not called yet, then the
  // `RequestCallback::OnResponseError` method must be called with status
  // `CANCELLED`.
  //
  // Otherwise, if a `PerformRequests` call is ongoing for this handle, and if
  // the `RequestCallback::OnResponseCompleted` method was not called yet, then
  // the `RequestCallback::OnResponseBodyError` method must be called with
  // status `CANCELLED`.
  virtual void Cancel() = 0;
};

// The callback interface that `HttpClient` implementations must use to deliver
// the response to a `Request`. Implemented by the caller of `HttpClient`.
//
// The lifetimes of instances of this class are owned by the caller of
// `HttpClient`. Instances must remain alive for at least as long as their
// corresponding `PerformRequests` call.
//
// Methods of this class may get called from any thread (and subsequent calls
// are not required to all happen on the same thread), but only a single method
// should ever be called at the same time (i.e. implementations likely should
// use external synchronization).
class RequestCallback {
 public:
  virtual ~RequestCallback() = default;

  // Called when the final HTTP response headers have been received (i.e. after
  // any redirects have been followed but before the response body may have been
  // received fully) for the given `Request`. The response data can be accessed
  // via the given `Response`, which will remain alive for the lifetime of the
  // corresponding `RequestHandle`.
  //
  // Note that all the data in the `Response` object should reflect the
  // last/final response (i.e. it shouldn't reflect any already-followed
  // redirects).
  //
  // If the response has a body then after this method is called
  // `OnResponseBody` will be called one or more times to deliver the response
  // body (or `OnResponseBodyError` if an error occurs).
  //
  // Note that responses with an HTTP status code other than 200 ("OK") may
  // still have response bodies, and implementations must deliver these via the
  // `OnResponseBody` callback, just as they should for a successful response.
  virtual void OnResponseStarted(const Request& request,
                                 Response& response) = 0;

  // Called when the request encountered an error or timed out, before receiving
  // the response headers completely. No further methods must be called on this
  // `RequestCallback` instance for the given `Request` after this method is
  // called.
  //
  // If the implementation is able to discern that the error may have been
  // transient, they should return `UNAVAILABLE`.
  //
  // If more than the implementation's defined max number of redirects occurred
  // (without reaching the final response), then implementations should return
  // `OUT_OF_RANGE` here.
  //
  // If the implementation hit an implementation-specific timeout (even though
  // implementations are discouraged from imposing such timeouts), then this
  // should be `DEADLINE_EXCEEDED`.
  //
  // If the `RequestHandle::Cancel` method was called before
  // `OnResponseStarted` was called for the given `Request`, then this method
  // will be called with a `CANCELLED` status.
  virtual void OnResponseError(const Request& request,
                               const absl::Status& error) = 0;

  // Called (possibly multiple times per request) when a block of response data
  // is available in `data`. This method must only be called after
  // `OnResponseStarted` was called for the given `Request`.
  //
  // Callees must process the data ASAP, as delaying this for too long may
  // prevent additional data from arriving on the network stream.
  virtual absl::Status OnResponseBody(const Request& request,
                                      Response& response,
                                      absl::string_view data) = 0;

  // Called when the request encountered an error or timed out while receiving
  // the response body (i.e. after `OnResponseStarted` was called). No further
  // methods must be called on this `RequestCallback` instance for the given
  // `Request` after this method is called.
  //
  // If the implementation is able to discern that the error may have been
  // transient, they should return `UNAVAILABLE`.
  //
  // If the implementation hit an implementation-specific timeout (even though
  // implementations are discouraged from imposing such timeouts), then this
  // should be `DEADLINE_EXCEEDED`.
  //
  // If the `RequestHandle::Cancel` method was called before
  // `OnResponseCompleted` was called for the given `Request`, then this method
  // will be called with a `CANCELLED` status.
  virtual void OnResponseBodyError(const Request& request, Response& response,
                                   const absl::Status& error) = 0;

  // Called when the request has completed successfully (i.e. the response
  // headers were delivered, and if there was a response body then it was also
  // delivered successfully). Must not be called if one of the error callbacks
  // was already called for the given `Request`, and no further methods must be
  // called on this `RequestCallback` instance for the given `Request` after
  // this method is called.
  virtual void OnResponseCompleted(const Request& request,
                                   Response& response) = 0;
};

// A response to a given `Request`. Implemented by the `HttpClient` implementer.
//
// The lifetimes of instances of this class are managed by the `HttpClient`
// implementer. Instances of this class must remain alive for at least long as
// the corresponding `RequestHandle` is alive.
//
// Note that all the data in this object should be for the last/final response.
// I.e. any responses corresponding to redirects should not be reflected here.
class Response {
 public:
  virtual ~Response() = default;

  // The response code returned by the server (e.g. 200).
  virtual int code() const = 0;

  // The response headers. Implementations are allowed to either coalesce
  // repeated headers using commas (as per RFC2616 section 4.2), or to return
  // them as separate entries.
  virtual const HeaderList& headers() const = 0;
};
};  // namespace http
};  // namespace client
};  // namespace fcp

#endif  // FCP_CLIENT_HTTP_HTTP_CLIENT_H_
