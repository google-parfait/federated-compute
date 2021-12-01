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

#include "fcp/protocol/grpc_chunked_bidi_stream.h"

#include <cctype>
#include <string>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/fake_server.h"
#include "fcp/client/grpc_bidi_stream.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/testing/testing.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"

namespace fcp {
namespace client {
namespace test {
namespace {

using google::internal::federatedml::v2::ClientStreamMessage;
using google::internal::federatedml::v2::CompressionLevel;
using google::internal::federatedml::v2::ServerStreamMessage;
using ::testing::Gt;
using ::testing::Le;
using ::testing::Not;

std::string SimpleSelfVerifyingString(size_t n) {
  std::string str;
  str.reserve(n);
  for (auto i = 0; i < n; ++i) str.push_back(static_cast<char>(i % 128));
  return str;
}

bool VerifyString(const std::string str) {
  auto n = str.length();
  for (auto i = 0; i < n; ++i) {
    if (str[i] != static_cast<char>(i % 128)) return false;
  }
  return true;
}

Status PerformInitialCheckin(GrpcBidiStream* stream) {
  ClientStreamMessage request_;
  ServerStreamMessage reply;
  Status status;

  auto options =
      request_.mutable_checkin_request()->mutable_protocol_options_request();
  options->set_supports_chunked_blob_transfer(true);
  options->add_supported_compression_levels(CompressionLevel::UNCOMPRESSED);
  options->add_supported_compression_levels(CompressionLevel::ZLIB_DEFAULT);
  options->add_supported_compression_levels(
      CompressionLevel::ZLIB_BEST_COMPRESSION);
  options->add_supported_compression_levels(CompressionLevel::ZLIB_BEST_SPEED);

  EXPECT_THAT((status = stream->Send(&request_)), IsOk());
  EXPECT_THAT((status = stream->Receive(&reply)), IsOk());
  EXPECT_TRUE(reply.has_checkin_response()) << reply.DebugString();

  return status;
}

using ChunkingParameters =
    std::tuple<int32_t, /* chunk_size_for_upload */
               int32_t, /* max_pending_chunks */
               google::internal::federatedml::v2::CompressionLevel,
               int32_t, /* request size */
               size_t,  /* request count */
               int32_t, /* reply size */
               size_t   /* replies per request */
               >;

class ByteVerifyingFakeServer : public FakeServer {
 public:
  explicit ByteVerifyingFakeServer(const ChunkingParameters& params)
      : FakeServer(std::get<0>(params), std::get<1>(params),
                   std::get<2>(params)),
        reply_size_(std::get<5>(params)),
        replies_per_request_(std::get<6>(params)) {}

  Status Handle(const ClientStreamMessage& request,
                ServerStreamMessage* first_reply,
                GrpcChunkedBidiStream<ServerStreamMessage, ClientStreamMessage>*
                    stream) override {
    Status status;
    if (request.has_checkin_request()) {
      EXPECT_THAT((status = stream->Send(first_reply)), IsOk());
      return status;
    }
    EXPECT_TRUE(
        VerifyString(request.report_request().report().update_checkpoint()));
    ServerStreamMessage reply;
    reply.mutable_report_response()->mutable_retry_window()->set_retry_token(
        SimpleSelfVerifyingString(reply_size_));
    for (auto i = 0; i < replies_per_request_; ++i)
      EXPECT_THAT((status = stream->Send(&reply)), IsOk());
    return status;
  }

 private:
  int32_t reply_size_;
  size_t replies_per_request_;
};

class GrpcChunkedMessageStreamTest
    : public ::testing::TestWithParam<ChunkingParameters> {
 public:
  GrpcChunkedMessageStreamTest() : server_impl_(GetParam()) {
    auto params = GetParam();
    request_size_ = std::get<3>(params);
    request_count_ = std::get<4>(params);
    reply_size_ = std::get<5>(params);
    replies_per_request_ = std::get<6>(params);

    grpc::ServerBuilder builder;
    builder.AddListeningPort("dns:///localhost:0",
                             grpc::InsecureServerCredentials(), &port_);
    builder.RegisterService(&server_impl_);
    grpc_server_ = builder.BuildAndStart();
    client_stream_ =
        std::make_unique<GrpcBidiStream>(addr_uri(), "none", "",
                                         /*grpc_channel_deadline_seconds=*/600);
    EXPECT_THAT(PerformInitialCheckin(client_stream_.get()), IsOk());

    request_.mutable_report_request()->mutable_report()->set_update_checkpoint(
        SimpleSelfVerifyingString(request_size_));
  }

  std::string addr_uri() { return absl::StrCat(kAddrUri, ":", port_); }

  int32_t request_size_;
  size_t request_count_;
  int32_t reply_size_;
  size_t replies_per_request_;

  static constexpr char kAddrUri[] = "dns:///localhost";
  ByteVerifyingFakeServer server_impl_;
  int port_ = -1;
  std::unique_ptr<grpc::Server> grpc_server_;
  std::unique_ptr<GrpcBidiStream> client_stream_;

  ClientStreamMessage request_;
  ServerStreamMessage reply_;
};

TEST_P(GrpcChunkedMessageStreamTest, RequestReply) {
  for (size_t i = 0; i < request_count_; ++i) {
    EXPECT_THAT(client_stream_->Send(&request_), IsOk());
    for (size_t i = 0; i < replies_per_request_; ++i) {
      EXPECT_THAT(client_stream_->Receive(&reply_), IsOk());
      EXPECT_TRUE(
          VerifyString(reply_.report_response().retry_window().retry_token()));
    }
  }
  client_stream_->Close();
  EXPECT_THAT(client_stream_->Receive(&reply_), Not(IsOk()));
}

TEST_P(GrpcChunkedMessageStreamTest, RequestReplyChunkingLayerBandwidth) {
  int64_t bytes_sent_so_far = client_stream_->ChunkingLayerBytesSent();
  int64_t bytes_received_so_far = client_stream_->ChunkingLayerBytesReceived();
  for (size_t i = 0; i < request_count_; ++i) {
    EXPECT_THAT(client_stream_->Send(&request_), IsOk());
    int64_t request_message_size = request_.ByteSizeLong();
    // Sends may be deferred if flow control has paused the stream; in this
    // case, they will not be recorded in statistics until they are sent as part
    // of the next Receive(). Therefore, we assert sizes after the receives.

    for (size_t i = 0; i < replies_per_request_; ++i) {
      EXPECT_THAT(client_stream_->Receive(&reply_), IsOk());
      int64_t bytes_received_delta =
          client_stream_->ChunkingLayerBytesReceived() - bytes_received_so_far;
      EXPECT_THAT(bytes_received_delta, Gt(0));
      int64_t receive_message_size = reply_.ByteSizeLong();
      // Small messages may actually be expanded due to compression overhead.
      if (receive_message_size > 64) {
        EXPECT_THAT(bytes_received_delta, Le(receive_message_size));
      }
      bytes_received_so_far += bytes_received_delta;
    }

    int64_t bytes_sent_delta =
        client_stream_->ChunkingLayerBytesSent() - bytes_sent_so_far;
    EXPECT_THAT(client_stream_->ChunkingLayerBytesSent(), Gt(0));
    EXPECT_THAT(bytes_sent_delta, Gt(0));
    // Small messages may actually be expanded due to compression overhead.
    if (request_message_size > 64) {
      EXPECT_THAT(bytes_sent_delta, Le(request_message_size));
    }
    bytes_sent_so_far += bytes_sent_delta;
  }
  client_stream_->Close();
  EXPECT_THAT(client_stream_->Receive(&reply_), Not(IsOk()));
}

// #define GRPC_CHUNKED_EXPENSIVE_COMPRESSED_TESTS
#if defined(GRPC_CHUNKED_EXPENSIVE_COMPRESSED_TESTS)
// Ideally we would generate a covering array rather than a Cartesian product.
INSTANTIATE_TEST_SUITE_P(
    CartesianProductExpensive, GrpcChunkedMessageStreamTest,
    testing::Combine(
        /* chunk_size_for_upload */
        testing::ValuesIn({0, 1, 129}),
        /* max_pending_chunks */
        testing::ValuesIn({0, 1, 129}),
        /* compression_level */
        testing::ValuesIn({CompressionLevel::ZLIB_DEFAULT}),
        /* request size */
        testing::ValuesIn({0, 1, 129}),
        /* request count */
        testing::ValuesIn({1ul, 129ul}),
        /* reply size */
        testing::ValuesIn({0, 1, 129}),
        /* replies per request */
        testing::ValuesIn({1ul, 129ul})),
    [](const testing::TestParamInfo<GrpcChunkedMessageStreamTest::ParamType>&
           info) {
      // clang-format off
      std::string name = absl::StrCat(
          std::get<0>(info.param), "csfu" "_",
          std::get<1>(info.param), "mpc", "_",
          std::get<2>(info.param), "cl", "_",
          std::get<3>(info.param), "rqs", "_",
          std::get<4>(info.param), "rqc", "_",
          std::get<5>(info.param), "rps", "_",
          std::get<6>(info.param), "rppr");
      absl::c_replace_if(
          name, [](char c) { return !std::isalnum(c); }, '_');
      // clang-format on
      return name;
    });
#endif

// #define GRPC_CHUNKED_EXPENSIVE_UNCOMPRESSED_TESTS
#if defined(GRPC_CHUNKED_EXPENSIVE_UNCOMPRESSED_TESTS)
// Ideally we would generate a covering array rather than a Cartesian product.
INSTANTIATE_TEST_SUITE_P(
    CartesianProductUncompressed, GrpcChunkedMessageStreamTest,
    testing::Combine(
        /* chunk_size_for_upload */
        testing::ValuesIn({0, 1, 129}),
        /* max_pending_chunks */
        testing::ValuesIn({0, 1, 129}),
        /* compression_level */
        testing::ValuesIn({CompressionLevel::UNCOMPRESSED}),
        /* request size */
        testing::ValuesIn({0, 1, 129}),
        /* request count */
        testing::ValuesIn({1ul, 129ul}),
        /* reply size */
        testing::ValuesIn({0, 1, 129}),
        /* replies per request */
        testing::ValuesIn({1ul, 129ul})),
    [](const testing::TestParamInfo<GrpcChunkedMessageStreamTest::ParamType>&
           info) {
      // clang-format off
      std::string name = absl::StrCat(
          std::get<0>(info.param), "csfu" "_",
          std::get<1>(info.param), "mpc", "_",
          std::get<2>(info.param), "cl", "_",
          std::get<3>(info.param), "rqs", "_",
          std::get<4>(info.param), "rqc", "_",
          std::get<5>(info.param), "rps", "_",
          std::get<6>(info.param), "rppr");
      absl::c_replace_if(
          name, [](char c) { return !std::isalnum(c); }, '_');
      // clang-format on
      return name;
    });
#endif

INSTANTIATE_TEST_SUITE_P(
    CartesianProductLargeChunks, GrpcChunkedMessageStreamTest,
    testing::Combine(
        /* chunk_size_for_upload */
        testing::ValuesIn({8192}),
        /* max_pending_chunks */
        testing::ValuesIn({2}),
        /* compression_level */
        testing::ValuesIn({CompressionLevel::ZLIB_BEST_SPEED}),
        /* request size */
        testing::ValuesIn({1024 * 1024 * 10}),
        /* request count */
        testing::ValuesIn({2ul}),
        /* reply size */
        testing::ValuesIn({1024 * 1024 * 10}),
        /* replies per request */
        testing::ValuesIn({2ul})),
    [](const testing::TestParamInfo<GrpcChunkedMessageStreamTest::ParamType>&
           info) {
      // clang-format off
      std::string name = absl::StrCat(
          std::get<0>(info.param), "csfu" "_",
          std::get<1>(info.param), "mpc", "_",
          std::get<2>(info.param), "cl", "_",
          std::get<3>(info.param), "rqs", "_",
          std::get<4>(info.param), "rqc", "_",
          std::get<5>(info.param), "rps", "_",
          std::get<6>(info.param), "rppr");
      absl::c_replace_if(
          name, [](char c) { return !std::isalnum(c); }, '_');
      // clang-format on
      return name;
    });

}  // namespace
}  // namespace test
}  // namespace client
}  // namespace fcp
