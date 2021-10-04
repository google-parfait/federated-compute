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

#ifndef FCP_CLIENT_GRPC_BIDI_STREAM_H_
#define FCP_CLIENT_GRPC_BIDI_STREAM_H_

#include <memory>
#include <string>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "fcp/base/monitoring.h"
#include "fcp/protocol/grpc_chunked_bidi_stream.h"
#include "fcp/protos/federated_api.grpc.pb.h"
#include "grpcpp/impl/codegen/channel_interface.h"
#include "grpcpp/impl/codegen/client_context.h"

namespace fcp {
namespace client {

/**
 * Interface to support dependency injection and hence testing
 */
class GrpcBidiStreamInterface {
 public:
  virtual ~GrpcBidiStreamInterface() = default;

  virtual ABSL_MUST_USE_RESULT absl::Status Send(
      google::internal::federatedml::v2::ClientStreamMessage* message) = 0;

  virtual ABSL_MUST_USE_RESULT absl::Status Receive(
      google::internal::federatedml::v2::ServerStreamMessage* message) = 0;

  virtual void Close() = 0;

  virtual int64_t ChunkingLayerBytesSent() = 0;

  virtual int64_t ChunkingLayerBytesReceived() = 0;
};

/**
 * A class which encapsulates a chunking gRPC endpoint for the federated
 * learning API.
 *
 * This class is thread-safe, but note that calls to Send() and Receive() are
 * serialized *and* blocking.
 */
class GrpcBidiStream : public GrpcBidiStreamInterface {
 public:
  /**
   * Create a chunking gRPC endpoint for the federated learning API.
   * @param target The URI of the target endpoint.
   * @param api_key The API key of the target endpoint.
   * @param population_name The population this connection is associated with.
   * This param will not be empty if the include_population_in_header flag is
   * False.
   * @param grpc_channel_deadline_seconds The deadline (in seconds) for the gRPC
   * channel.
   * @param cert_path Test-only path to a CA certificate root, to be used in
   * combination with an "https+test://" URI scheme.
   */
  GrpcBidiStream(const std::string& target, const std::string& api_key,
                 const std::string& population_name,
                 int64_t grpc_channel_deadline_seconds,
                 std::string cert_path = "");

  /**
   * @param channel A preexisting channel to the target endpoint.
   * @param api_key The API of the target endpoint.
   * @param population_name The population this connection is associated with.
   * This param will not be empty if the include_population_in_header flag is
   * False.
   * @param grpc_channel_deadline_seconds The deadline (in seconds) for the gRPC
   * channel.
   */
  GrpcBidiStream(const std::shared_ptr<grpc::ChannelInterface>& channel,
                 const std::string& api_key, const std::string& population_name,
                 int64_t grpc_channel_deadline_seconds);
  ~GrpcBidiStream() override = default;

  // GrpcBidiStream is neither copyable nor movable.
  GrpcBidiStream(const GrpcBidiStream&) = delete;
  GrpcBidiStream& operator=(const GrpcBidiStream&) = delete;

  /**
   * Send a ClientStreamMessage to the remote endpoint.
   * @param message The message to send.
   * @return absl::Status, which will have code OK if the message was sent
   *   successfully.
   */
  ABSL_MUST_USE_RESULT absl::Status Send(
      google::internal::federatedml::v2::ClientStreamMessage* message) override
      ABSL_LOCKS_EXCLUDED(mu_);

  /**
   * Receive a ServerStreamMessage from the remote endpoint.  Blocking.
   * @param message The message to receive.
   * @return absl::Status. This may be a translation of the status returned by
   * the server, or a status generated during execution of the chunking
   * protocol.
   */
  ABSL_MUST_USE_RESULT absl::Status Receive(
      google::internal::federatedml::v2::ServerStreamMessage* message) override
      ABSL_LOCKS_EXCLUDED(mu_);

  /**
   * Close this stream.
   * Releases any blocked readers. Thread safe.
   */
  void Close() override ABSL_LOCKS_EXCLUDED(mu_);

  /**
   * Returns the number of bytes sent from the chunking layer.
   * Flow control means this value may not increment until Receive() is called.
   */
  int64_t ChunkingLayerBytesSent() override;

  /**
   * Returns the number of bytes received by the chunking layer.
   */
  int64_t ChunkingLayerBytesReceived() override;

  // Note: Must be lowercase:
  static constexpr char kApiKeyHeader[] = "x-goog-api-key";
  static constexpr char kPopulationNameHeader[] = "x-goog-population";

 private:
  absl::Mutex mu_;
  std::unique_ptr<google::internal::federatedml::v2::FederatedTrainingApi::Stub>
      stub_;
  grpc::ClientContext client_context_;
  std::unique_ptr<grpc::ClientReaderWriter<
      google::internal::federatedml::v2::ClientStreamMessage,
      google::internal::federatedml::v2::ServerStreamMessage>>
      client_reader_writer_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<GrpcChunkedBidiStream<
      google::internal::federatedml::v2::ClientStreamMessage,
      google::internal::federatedml::v2::ServerStreamMessage>>
      chunked_bidi_stream_ ABSL_GUARDED_BY(mu_);
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_GRPC_BIDI_STREAM_H_
