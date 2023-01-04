/*
 * Copyright 2019 Google LLC
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

#include "fcp/client/grpc_bidi_stream.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "fcp/base/status_converters.h"
#include "fcp/client/grpc_bidi_channel.h"
#include "grpcpp/support/time.h"

namespace fcp {
namespace client {

using fcp::base::FromGrpcStatus;
using google::internal::federatedml::v2::ClientStreamMessage;
using google::internal::federatedml::v2::FederatedTrainingApi;
using google::internal::federatedml::v2::ServerStreamMessage;
using grpc::ChannelInterface;

GrpcBidiStream::GrpcBidiStream(const std::string& target,
                               const std::string& api_key,
                               const std::string& population_name,
                               int64_t grpc_channel_deadline_seconds,
                               std::string cert_path)
    : GrpcBidiStream(GrpcBidiChannel::Create(target, std::move(cert_path)),
                     api_key, population_name, grpc_channel_deadline_seconds) {}

GrpcBidiStream::GrpcBidiStream(
    const std::shared_ptr<grpc::ChannelInterface>& channel,
    const std::string& api_key, const std::string& population_name,
    int64_t grpc_channel_deadline_seconds)
    : mu_(), stub_(FederatedTrainingApi::NewStub(channel)) {
  FCP_LOG(INFO) << "Connecting to stub: " << stub_.get();
  gpr_timespec deadline = gpr_time_add(
      gpr_now(GPR_CLOCK_REALTIME),
      gpr_time_from_seconds(grpc_channel_deadline_seconds, GPR_TIMESPAN));
  client_context_.set_deadline(deadline);
  client_context_.AddMetadata(kApiKeyHeader, api_key);
  client_context_.AddMetadata(kPopulationNameHeader, population_name);
  client_reader_writer_ = stub_->Session(&client_context_);
  GrpcChunkedBidiStream<ClientStreamMessage,
                        ServerStreamMessage>::GrpcChunkedBidiStreamOptions
      options;
  chunked_bidi_stream_ = std::make_unique<
      GrpcChunkedBidiStream<ClientStreamMessage, ServerStreamMessage>>(
      client_reader_writer_.get(), client_reader_writer_.get(), options);
  if (!channel) Close();
}

absl::Status GrpcBidiStream::Send(ClientStreamMessage* message) {
  absl::Status status;
  {
    absl::MutexLock _(&mu_);
    if (client_reader_writer_ == nullptr) {
      return absl::CancelledError(
          "Send failed because GrpcBidiStream was closed.");
    }
    status = chunked_bidi_stream_->Send(message);
    if (status.code() == absl::StatusCode::kAborted) {
      FCP_LOG(INFO) << "Send aborted: " << status.code();
      auto finish_status = FromGrpcStatus(client_reader_writer_->Finish());
      // If the connection aborts early or harshly enough, there will be no
      // error status from Finish().
      if (!finish_status.ok()) status = finish_status;
    }
  }
  if (!status.ok()) {
    FCP_LOG(INFO) << "Closing; error on send: " << status.message();
    Close();
  }
  return status;
}

absl::Status GrpcBidiStream::Receive(ServerStreamMessage* message) {
  absl::Status status;
  {
    absl::MutexLock _(&mu_);
    if (client_reader_writer_ == nullptr) {
      return absl::CancelledError(
          "Receive failed because GrpcBidiStream was closed.");
    }
    status = chunked_bidi_stream_->Receive(message);
    if (status.code() == absl::StatusCode::kAborted) {
      FCP_LOG(INFO) << "Receive aborted: " << status.code();
      auto finish_status = FromGrpcStatus(client_reader_writer_->Finish());
      // If the connection aborts early or harshly enough, there will be no
      // error status from Finish().
      if (!finish_status.ok()) status = finish_status;
    }
  }
  if (!status.ok()) {
    FCP_LOG(INFO) << "Closing; error on receive: " << status.message();
    Close();
  }
  return status;
}

void GrpcBidiStream::Close() {
  if (!mu_.TryLock()) {
    client_context_.TryCancel();
    mu_.Lock();
  }
  chunked_bidi_stream_->Close();
  if (client_reader_writer_) client_reader_writer_->WritesDone();
  client_reader_writer_.reset();
  FCP_LOG(INFO) << "Closing stub: " << stub_.get();
  stub_.reset();
  mu_.Unlock();
}

int64_t GrpcBidiStream::ChunkingLayerBytesReceived() {
  absl::MutexLock _(&mu_);
  return chunked_bidi_stream_->ChunkingLayerBytesReceived();
}

int64_t GrpcBidiStream::ChunkingLayerBytesSent() {
  absl::MutexLock _(&mu_);
  return chunked_bidi_stream_->ChunkingLayerBytesSent();
}

}  // namespace client
}  // namespace fcp
