// Copyright 2021 Google LLC
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

#include "fcp/client/fake_server.h"

#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/status_converters.h"
#include "fcp/client/grpc_bidi_stream.h"
#include "fcp/protocol/grpc_chunked_bidi_stream.h"
#include "fcp/protos/federated_api.pb.h"

namespace fcp {
namespace client {
namespace test {

using fcp::base::ToGrpcStatus;
using fcp::client::GrpcChunkedBidiStream;
using google::internal::federatedml::v2::ClientStreamMessage;
using google::internal::federatedml::v2::RetryWindow;
using google::internal::federatedml::v2::ServerStreamMessage;

static RetryWindow GetRetryWindow(const std::string& token, int64_t min,
                                  int64_t max) {
  RetryWindow retry_window;
  retry_window.mutable_delay_min()->set_seconds(min);
  retry_window.mutable_delay_max()->set_seconds(max);
  *retry_window.mutable_retry_token() = token;
  return retry_window;
}

grpc::Status FakeServer::Session(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<ServerStreamMessage, ClientStreamMessage>*
        stream) {
  GrpcChunkedBidiStream<ServerStreamMessage, ClientStreamMessage>
      chunked_bidi_stream(
          stream, stream,
          {chunk_size_for_upload_, max_pending_chunks_, compression_level_});
  ClientStreamMessage request;
  ServerStreamMessage response;
  FCP_LOG(INFO) << "Server session started";
  absl::Status status;
  while ((status = chunked_bidi_stream.Receive(&request)).ok()) {
    FCP_LOG(INFO) << "Request is: " << request.DebugString();
    for (const auto& [key, value] : context->client_metadata()) {
      client_metadata_.insert(
          std::make_pair(std::string(key.data(), key.size()),
                         std::string(value.data(), value.size())));
    }
    if (request.eligibility_eval_checkin_request()
            .protocol_options_request()
            .should_ack_checkin() ||
        request.checkin_request()
            .protocol_options_request()
            .should_ack_checkin()) {
      ServerStreamMessage checkin_request_ack_msg;
      auto checkin_request_ack =
          checkin_request_ack_msg.mutable_checkin_request_ack();
      *checkin_request_ack->mutable_retry_window_if_accepted() =
          GetRetryWindow("A", 111L, 222L);
      *checkin_request_ack->mutable_retry_window_if_rejected() =
          GetRetryWindow("R", 333L, 444L);
      if (!chunked_bidi_stream.Send(&checkin_request_ack_msg).ok()) {
        FCP_LOG(INFO) << "Server returning status " << status;
        return ToGrpcStatus(status);
      }
    }
    if (request.has_eligibility_eval_checkin_request() ||
        request.has_checkin_request()) {
      auto protocol_options_response =
          request.has_eligibility_eval_checkin_request()
              ? response.mutable_eligibility_eval_checkin_response()
                    ->mutable_protocol_options_response()
              : response.mutable_checkin_response()
                    ->mutable_protocol_options_response();
      protocol_options_response->set_compression_level(compression_level_);
      protocol_options_response->set_chunk_size_for_upload(
          chunk_size_for_upload_);
      protocol_options_response->set_max_pending_chunks(max_pending_chunks_);
    }
    if (!(status = Handle(request, &response, &chunked_bidi_stream)).ok()) {
      FCP_LOG(INFO) << "Server returning status " << status;
      return ToGrpcStatus(status);
    }
  }
  session_done_.Notify();
  FCP_LOG(INFO) << "Server returning status " << status;
  return ToGrpcStatus(status);
}

std::multimap<std::string, std::string> FakeServer::GetClientMetadata() const {
  return client_metadata_;
}

void FakeServer::WaitForSessionDone() { session_done_.WaitForNotification(); }

absl::Status FakeServer::Handle(
    const ClientStreamMessage& request, ServerStreamMessage* first_reply,
    GrpcChunkedBidiStream<ServerStreamMessage, ClientStreamMessage>* stream) {
  return stream->Send(first_reply);
}

}  // namespace test
}  // namespace client
}  // namespace fcp
