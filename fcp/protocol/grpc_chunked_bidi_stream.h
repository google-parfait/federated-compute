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

#ifndef FCP_PROTOCOL_GRPC_CHUNKED_BIDI_STREAM_H_
#define FCP_PROTOCOL_GRPC_CHUNKED_BIDI_STREAM_H_

#include <stddef.h>

#include <algorithm>
#include <deque>
#include <memory>
#include <string>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/federated_api.grpc.pb.h"
#include "grpcpp/impl/codegen/call_op_set.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "google/protobuf/io/gzip_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace fcp {
namespace client {

/**
 * A class which implements the chunking protocol for the federated learning
 * API.
 *
 * Can be used by both client and server.
 *
 * @tparam Outgoing The type of the outgoing protocol buffer message.
 * @tparam Incoming The type of the incoming protocol buffer message.
 */
template <typename Outgoing, typename Incoming>
class GrpcChunkedBidiStream {
 public:
  struct GrpcChunkedBidiStreamOptions {
    int32_t chunk_size_for_upload = -1;
    int32_t max_pending_chunks = -1;
    google::internal::federatedml::v2::CompressionLevel compression_level{};
  };
  GrpcChunkedBidiStream(
      grpc::internal::WriterInterface<Outgoing>* writer_interface,
      grpc::internal::ReaderInterface<Incoming>* reader_interface);
  GrpcChunkedBidiStream(
      grpc::internal::WriterInterface<Outgoing>* writer_interface,
      grpc::internal::ReaderInterface<Incoming>* reader_interface,
      GrpcChunkedBidiStreamOptions options);
  virtual ~GrpcChunkedBidiStream() = default;

  // GrpcChunkedBidiStream is neither copyable nor movable.
  GrpcChunkedBidiStream(const GrpcChunkedBidiStream&) = delete;
  GrpcChunkedBidiStream& operator=(const GrpcChunkedBidiStream&) = delete;

  ABSL_MUST_USE_RESULT absl::Status Send(Outgoing* message);
  ABSL_MUST_USE_RESULT absl::Status Receive(Incoming* message);
  void Close();
  int64_t ChunkingLayerBytesSent();
  int64_t ChunkingLayerBytesReceived();

 private:
  ABSL_MUST_USE_RESULT absl::Status TryDecorateCheckinRequest(
      Outgoing* message);
  ABSL_MUST_USE_RESULT absl::Status ChunkMessage(const Outgoing& message);
  ABSL_MUST_USE_RESULT absl::Status TrySendPending();
  ABSL_MUST_USE_RESULT absl::Status TrySend(const Outgoing& message);
  ABSL_MUST_USE_RESULT absl::Status SendAck(int32_t chunk_index);
  ABSL_MUST_USE_RESULT absl::Status SendRaw(const Outgoing& message,
                                            bool disable_compression = false);
  ABSL_MUST_USE_RESULT absl::Status TrySnoopCheckinResponse(Incoming* message);
  ABSL_MUST_USE_RESULT absl::Status TryAssemblePending(Incoming* message,
                                                       bool* message_assembled);
  ABSL_MUST_USE_RESULT absl::Status AssemblePending(Incoming* message,
                                                    bool* message_assembled);
  ABSL_MUST_USE_RESULT absl::Status ReceiveRaw(Incoming* message);

  grpc::internal::WriterInterface<Outgoing>* writer_interface_;
  grpc::internal::ReaderInterface<Incoming>* reader_interface_;

  struct {
    int32_t uncompressed_size = -1;
    google::internal::federatedml::v2::CompressionLevel compression_level{};
    int32_t blob_size_bytes = -1;
    std::deque<std::string> deque;
    std::string composite;
    int64_t total_bytes_downloaded = 0;
  } incoming_;

  struct {
    int32_t chunk_size_for_upload = 0;
    int32_t max_pending_chunks = 0;
    int32_t pending_chunks = 0;
    google::internal::federatedml::v2::CompressionLevel compression_level{};
    std::deque<std::unique_ptr<Outgoing>> deque;
    int64_t total_bytes_uploaded = 0;

    google::internal::federatedml::v2::ChunkedTransferMessage* Add() {
      deque.push_back(std::make_unique<Outgoing>());
      return deque.back()->mutable_chunked_transfer();
    }
  } outgoing_;
};

#define COMMON_USING_DIRECTIVES                                    \
  using google::internal::federatedml::v2::ChunkedTransferMessage; \
  using google::internal::federatedml::v2::ClientStreamMessage;    \
  using google::internal::federatedml::v2::CompressionLevel;       \
  using google::internal::federatedml::v2::ServerStreamMessage;    \
  using google::protobuf::io::ArrayInputStream;                              \
  using google::protobuf::io::StringOutputStream;                            \
  using google::protobuf::io::GzipInputStream;                               \
  using google::protobuf::io::GzipOutputStream;                              \
  using google::protobuf::io::ZeroCopyOutputStream;

template <typename Outgoing, typename Incoming>
GrpcChunkedBidiStream<Outgoing, Incoming>::GrpcChunkedBidiStream(
    grpc::internal::WriterInterface<Outgoing>* writer_interface,
    grpc::internal::ReaderInterface<Incoming>* reader_interface)
    : GrpcChunkedBidiStream(writer_interface, reader_interface,
                            GrpcChunkedBidiStreamOptions()) {}

template <typename Outgoing, typename Incoming>
GrpcChunkedBidiStream<Outgoing, Incoming>::GrpcChunkedBidiStream(
    grpc::internal::WriterInterface<Outgoing>* writer_interface,
    grpc::internal::ReaderInterface<Incoming>* reader_interface,
    GrpcChunkedBidiStreamOptions options)
    : writer_interface_(writer_interface), reader_interface_(reader_interface) {
  outgoing_.chunk_size_for_upload = options.chunk_size_for_upload;
  outgoing_.max_pending_chunks = options.max_pending_chunks;
  outgoing_.compression_level = options.compression_level;
}

template <typename Outgoing, typename Incoming>
absl::Status GrpcChunkedBidiStream<Outgoing, Incoming>::Send(
    Outgoing* message) {
  COMMON_USING_DIRECTIVES;
  FCP_RETURN_IF_ERROR(TryDecorateCheckinRequest(message));
  switch (message->kind_case()) {
    case Outgoing::KindCase::kChunkedTransfer:
      Close();
      return absl::InvalidArgumentError(
          absl::StrCat("Message is pre-chunked: ", message->DebugString()));
    default:
      break;
  }

  return TrySend(*message);
}

template <typename Outgoing, typename Incoming>
absl::Status GrpcChunkedBidiStream<Outgoing, Incoming>::Receive(
    Incoming* message) {
  COMMON_USING_DIRECTIVES;
  Status status;
  bool message_assembled = false;

  do {
    FCP_RETURN_IF_ERROR(status = ReceiveRaw(message));
    switch (message->kind_case()) {
      case Incoming::KindCase::kChunkedTransfer:
        if (message->chunked_transfer().kind_case() ==
            ChunkedTransferMessage::kAck) {
          --outgoing_.pending_chunks;
          FCP_RETURN_IF_ERROR(status = TrySendPending());
        } else {
          FCP_RETURN_IF_ERROR(
              status = TryAssemblePending(message, &message_assembled));
        }
        break;
      default:
        if (incoming_.uncompressed_size != -1)
          return absl::InvalidArgumentError("Chunk reassembly in progress.");
        message_assembled = true;
        break;
    }
  } while (!message_assembled);

  FCP_RETURN_IF_ERROR(status = TrySnoopCheckinResponse(message));
  return status;
}

template <>
inline absl::Status
GrpcChunkedBidiStream<google::internal::federatedml::v2::ClientStreamMessage,
                      google::internal::federatedml::v2::ServerStreamMessage>::
    TryDecorateCheckinRequest(
        google::internal::federatedml::v2::ClientStreamMessage* message) {
  COMMON_USING_DIRECTIVES;
  if (message->kind_case() !=
          ClientStreamMessage::kEligibilityEvalCheckinRequest &&
      message->kind_case() != ClientStreamMessage::kCheckinRequest)
    return absl::OkStatus();
  // Both an EligibilityEvalCheckinRequest or a CheckinRequest message need to
  // specify a ProtocolOptionsRequest message.
  auto options = (message->has_eligibility_eval_checkin_request()
                      ? message->mutable_eligibility_eval_checkin_request()
                            ->mutable_protocol_options_request()
                      : message->mutable_checkin_request()
                            ->mutable_protocol_options_request());
  options->set_supports_chunked_blob_transfer(true);
  options->add_supported_compression_levels(CompressionLevel::UNCOMPRESSED);
  options->add_supported_compression_levels(CompressionLevel::ZLIB_DEFAULT);
  options->add_supported_compression_levels(
      CompressionLevel::ZLIB_BEST_COMPRESSION);
  options->add_supported_compression_levels(CompressionLevel::ZLIB_BEST_SPEED);
  return absl::OkStatus();
}

template <typename Outgoing, typename Incoming>
absl::Status
GrpcChunkedBidiStream<Outgoing, Incoming>::TryDecorateCheckinRequest(
    Outgoing*) {
  return absl::OkStatus();
}

template <typename Outgoing, typename Incoming>
absl::Status GrpcChunkedBidiStream<Outgoing, Incoming>::ChunkMessage(
    const Outgoing& message) {
  COMMON_USING_DIRECTIVES;

  auto start = outgoing_.Add()->mutable_start();
  start->set_compression_level(outgoing_.compression_level);

  // TODO(team): Replace with a more efficient serialization mechanism.
  std::string output;
  if (outgoing_.compression_level == CompressionLevel::UNCOMPRESSED) {
    if (!message.AppendToString(&output))
      return absl::InternalError("Could not append to string.");
  } else {
    StringOutputStream string_output_stream(&output);
    GzipOutputStream::Options options;
    options.format = GzipOutputStream::ZLIB;
    switch (outgoing_.compression_level) {
      case CompressionLevel::ZLIB_DEFAULT:
        options.compression_level = Z_DEFAULT_COMPRESSION;
        break;
      case CompressionLevel::ZLIB_BEST_COMPRESSION:
        options.compression_level = Z_BEST_COMPRESSION;
        break;
      case CompressionLevel::ZLIB_BEST_SPEED:
        options.compression_level = Z_BEST_SPEED;
        break;
      default:
        Close();
        return absl::InternalError("Unsupported compression level.");
    }
    GzipOutputStream compressed_stream(&string_output_stream, options);
    if (!message.SerializeToZeroCopyStream(&compressed_stream) ||
        !compressed_stream.Close())
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to serialize message: ",
                       compressed_stream.ZlibErrorMessage()));
  }

  auto blob_size_bytes = static_cast<int32_t>(output.size());
  int32_t chunk_index = 0;
  if (!blob_size_bytes) blob_size_bytes = 1;  // Force one empty packet.
  for (size_t offset = 0; offset < blob_size_bytes;
       offset += std::min(blob_size_bytes, outgoing_.chunk_size_for_upload),
              ++chunk_index) {
    auto data = outgoing_.Add()->mutable_data();
    data->set_chunk_index(chunk_index);
    data->set_chunk_bytes(output.substr(
        offset, static_cast<size_t>(outgoing_.chunk_size_for_upload)));
  }

  start->set_uncompressed_size(static_cast<int32_t>(message.ByteSizeLong()));
  start->set_blob_size_bytes(blob_size_bytes);

  auto end = outgoing_.Add()->mutable_end();
  end->set_chunk_count(chunk_index);
  return absl::OkStatus();
}

template <typename Outgoing, typename Incoming>
absl::Status GrpcChunkedBidiStream<Outgoing, Incoming>::TrySendPending() {
  COMMON_USING_DIRECTIVES;
  auto status = absl::OkStatus();
  while (!outgoing_.deque.empty() &&
         outgoing_.pending_chunks < outgoing_.max_pending_chunks) {
    auto& front = outgoing_.deque.front();
    FCP_RETURN_IF_ERROR(status =
                            SendRaw(*front, outgoing_.compression_level > 0));
    if (front->chunked_transfer().kind_case() == ChunkedTransferMessage::kData)
      ++outgoing_.pending_chunks;
    outgoing_.deque.pop_front();
  }
  return status;
}

template <typename Outgoing, typename Incoming>
absl::Status GrpcChunkedBidiStream<Outgoing, Incoming>::TrySend(
    const Outgoing& message) {
  COMMON_USING_DIRECTIVES;
  if (outgoing_.chunk_size_for_upload <= 0 || outgoing_.max_pending_chunks <= 0)
    return SendRaw(message);  // No chunking.
  absl::Status status;
  if (!(status = ChunkMessage(message)).ok()) {
    Close();
    return status;
  }
  return TrySendPending();
}

template <typename Outgoing, typename Incoming>
absl::Status GrpcChunkedBidiStream<Outgoing, Incoming>::SendAck(
    int32_t chunk_index) {
  Outgoing ack;
  ack.mutable_chunked_transfer()->mutable_ack()->set_chunk_index(chunk_index);
  return SendRaw(ack);
}

template <typename Outgoing, typename Incoming>
absl::Status GrpcChunkedBidiStream<Outgoing, Incoming>::SendRaw(
    const Outgoing& message, bool disable_compression) {
  if (!writer_interface_)
    return absl::FailedPreconditionError("Send on closed stream.");
  grpc::WriteOptions write_options;
  if (disable_compression) write_options.set_no_compression();
  outgoing_.total_bytes_uploaded += message.ByteSizeLong();
  if (!writer_interface_->Write(message, write_options)) {
    Close();
    return absl::AbortedError("End of stream.");
  }
  return absl::OkStatus();
}

// If this class is used on the client side, we need to break the abstraction
// that messages are opaque in order to read the chunking parameters sent by the
// server to determine how to carry out the remainder of the protocol.
// Inspect the checkin response to record these chunking options.
template <>
inline absl::Status
GrpcChunkedBidiStream<google::internal::federatedml::v2::ClientStreamMessage,
                      google::internal::federatedml::v2::ServerStreamMessage>::
    TrySnoopCheckinResponse(
        google::internal::federatedml::v2::ServerStreamMessage* message) {
  COMMON_USING_DIRECTIVES;
  if (message->kind_case() !=
          ServerStreamMessage::kEligibilityEvalCheckinResponse &&
      message->kind_case() != ServerStreamMessage::kCheckinResponse)
    return absl::OkStatus();
  if (incoming_.uncompressed_size != -1)
    return absl::InvalidArgumentError("Chunk reassembly in progress.");
  // We adopt any new protocol options we may receive, even if we previously
  // received some options already. I.e. a ProtocolOptionsResponse received in a
  // CheckinResponse will overwrite any ProtocolOptionsResponse that was
  // previously received in a EligibilityEvalCheckinResponse.
  // OTOH, we also don't require that every EligibilityEvalCheckinResponse or
  // CheckinResponse message actually has a ProtocolOptionsResponse message set
  // (e.g. CheckinResponse may not have a ProtocolOptionsResponse if one was
  // already returned inside a prior EligibilityEvalCheckinResponse).
  if (message->eligibility_eval_checkin_response()
          .has_protocol_options_response() ||
      message->checkin_response().has_protocol_options_response()) {
    auto options =
        (message->has_eligibility_eval_checkin_response()
             ? message->eligibility_eval_checkin_response()
                   .protocol_options_response()
             : message->checkin_response().protocol_options_response());
    outgoing_.chunk_size_for_upload = options.chunk_size_for_upload();
    outgoing_.max_pending_chunks = options.max_pending_chunks();
    outgoing_.compression_level = options.compression_level();
  }
  return absl::OkStatus();
}

// If this class is being used by the server, this is a no-op as the server
// determines the chunking options.
template <typename Outgoing, typename Incoming>
absl::Status GrpcChunkedBidiStream<Outgoing, Incoming>::TrySnoopCheckinResponse(
    Incoming*) {
  return absl::OkStatus();
}

template <typename Outgoing, typename Incoming>
absl::Status GrpcChunkedBidiStream<Outgoing, Incoming>::TryAssemblePending(
    Incoming* message, bool* message_assembled) {
  COMMON_USING_DIRECTIVES;
  *message_assembled = false;
  auto chunk = message->chunked_transfer();
  switch (chunk.kind_case()) {
    case ChunkedTransferMessage::kStart:
      if (!incoming_.deque.empty() || incoming_.uncompressed_size != -1)
        return absl::InternalError("Unexpected Start.");
      incoming_.uncompressed_size = chunk.start().uncompressed_size();
      incoming_.compression_level = chunk.start().compression_level();
      incoming_.blob_size_bytes = chunk.start().blob_size_bytes();
      break;
    case ChunkedTransferMessage::kData:
      if (chunk.data().chunk_index() != incoming_.deque.size())
        return absl::InternalError("Unexpected Data.");
      incoming_.deque.emplace_back(chunk.data().chunk_bytes());
      incoming_.composite.append(incoming_.deque.back());
      return SendAck(static_cast<int32_t>(incoming_.deque.size() - 1));
    case ChunkedTransferMessage::kEnd:
      if (incoming_.deque.empty() ||
          chunk.end().chunk_count() != incoming_.deque.size())
        return absl::InternalError("Unexpected End.");
      return AssemblePending(message, message_assembled);
    case ChunkedTransferMessage::kAck:
      return absl::InternalError("Unexpected Ack.");
    default:
      return absl::InternalError(
          absl::StrCat("Unexpected message subtype: ",
                       message->chunked_transfer().kind_case()));
  }

  return absl::OkStatus();
}

template <typename Outgoing, typename Incoming>
absl::Status GrpcChunkedBidiStream<Outgoing, Incoming>::AssemblePending(
    Incoming* message, bool* message_assembled) {
  COMMON_USING_DIRECTIVES;
  // TODO(team): Replace with a more efficient deserialization mechanism.
  if (incoming_.compression_level == CompressionLevel::UNCOMPRESSED) {
    if (!message->ParseFromString(incoming_.composite))
      return absl::InternalError(absl::StrCat("Could not parse from string. ",
                                              incoming_.composite.size()));
  } else {
    ArrayInputStream string_input_stream(
        incoming_.composite.c_str(),
        static_cast<int>(incoming_.composite.size()));
    GzipInputStream compressed_stream(&string_input_stream);
    if (!message->ParseFromZeroCopyStream(&compressed_stream))
      return absl::InternalError("Could not parse proto from input stream.");
  }
  *message_assembled = true;
  incoming_.uncompressed_size = -1;
  incoming_.blob_size_bytes = -1;
  incoming_.deque.clear();
  incoming_.composite.clear();
  return absl::OkStatus();
}

template <typename Outgoing, typename Incoming>
absl::Status GrpcChunkedBidiStream<Outgoing, Incoming>::ReceiveRaw(
    Incoming* message) {
  if (!reader_interface_)
    return absl::FailedPreconditionError("Receive on closed stream.");
  if (!reader_interface_->Read(message)) {
    Close();
    return absl::AbortedError("End of stream.");
  }
  incoming_.total_bytes_downloaded += message->ByteSizeLong();
  return absl::OkStatus();
}

template <typename Outgoing, typename Incoming>
void GrpcChunkedBidiStream<Outgoing, Incoming>::Close() {
  writer_interface_ = nullptr;
  reader_interface_ = nullptr;
}

template <typename Outgoing, typename Incoming>
int64_t
GrpcChunkedBidiStream<Outgoing, Incoming>::ChunkingLayerBytesReceived() {
  return incoming_.total_bytes_downloaded;
}

template <typename Outgoing, typename Incoming>
int64_t GrpcChunkedBidiStream<Outgoing, Incoming>::ChunkingLayerBytesSent() {
  return outgoing_.total_bytes_uploaded;
}

}  // namespace client
}  // namespace fcp

#endif  // FCP_PROTOCOL_GRPC_CHUNKED_BIDI_STREAM_H_
