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

#ifndef FCP_CLIENT_FAKE_SERVER_H_
#define FCP_CLIENT_FAKE_SERVER_H_

#include <cstddef>
#include <string>
#include <tuple>

#include "grpcpp/impl/codegen/status.h"
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/grpc_bidi_stream.h"
#include "fcp/protocol/grpc_chunked_bidi_stream.h"
#include "fcp/protos/federated_api.grpc.pb.h"
#include "fcp/protos/federated_api.pb.h"
#include "grpcpp/impl/codegen/server_context.h"

namespace fcp {
namespace client {
namespace test {

class FakeServer
    : public google::internal::federatedml::v2::FederatedTrainingApi::Service {
 public:
  FakeServer()
      : chunk_size_for_upload_(8192),
        max_pending_chunks_(2),
        compression_level_(google::internal::federatedml::v2::CompressionLevel::
                               ZLIB_BEST_COMPRESSION) {}
  FakeServer(
      int32_t chunk_size_for_upload, int32_t max_pending_chunks,
      google::internal::federatedml::v2::CompressionLevel compression_level)
      : chunk_size_for_upload_(chunk_size_for_upload),
        max_pending_chunks_(max_pending_chunks),
        compression_level_(compression_level) {}

  // FakeServer is neither copyable nor movable.
  FakeServer(const FakeServer&) = delete;
  FakeServer& operator=(const FakeServer&) = delete;

  grpc::Status Session(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<
          google::internal::federatedml::v2::ServerStreamMessage,
          google::internal::federatedml::v2::ClientStreamMessage>* stream)
      override;
  void WaitForSessionDone();

  virtual absl::Status Handle(
      const google::internal::federatedml::v2::ClientStreamMessage& request,
      google::internal::federatedml::v2::ServerStreamMessage* first_reply,
      ::fcp::client::GrpcChunkedBidiStream<
          google::internal::federatedml::v2::ServerStreamMessage,
          google::internal::federatedml::v2::ClientStreamMessage>* stream);

  // Returns the client metadata from the most recent session call.
  std::multimap<std::string, std::string> GetClientMetadata() const;

 protected:
  int32_t chunk_size_for_upload_;
  int32_t max_pending_chunks_;
  google::internal::federatedml::v2::CompressionLevel compression_level_;
  absl::Notification session_done_;

 private:
  std::multimap<std::string, std::string> client_metadata_;
};

}  // namespace test
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_FAKE_SERVER_H_
