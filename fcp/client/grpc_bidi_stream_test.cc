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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/scheduler.h"
#include "fcp/client/fake_server.h"
#include "fcp/client/test_helpers.h"
#include "fcp/testing/testing.h"
#include "grpcpp/server_builder.h"

namespace fcp {
namespace client {
namespace test {
namespace {

using google::internal::federatedml::v2::ClientStreamMessage;
using google::internal::federatedml::v2::ServerStreamMessage;
using ::testing::Contains;
using ::testing::Not;
using ::testing::Pair;

class GrpcBidiStreamTest : public testing::Test {
 protected:
  void SetUp() override { BuildAndStartServer(); }

  void TearDown() override {
    server_->Shutdown();
    server_->Wait();
  }

  void CreateClient(const std::string& population_name = "") {
    client_stream_ = std::make_unique<GrpcBidiStream>(
        absl::StrCat("dns:///localhost", ":", port_), "none", population_name,
        /* grpc_channel_deadline_seconds=*/600);
    FCP_LOG(INFO) << "Client created." << std::endl;
  }

  std::unique_ptr<GrpcBidiStream> client_stream_;
  FakeServer server_impl_;

 private:
  void BuildAndStartServer() {
    grpc::ServerBuilder builder;
    builder.AddListeningPort("dns:///localhost:0",
                             grpc::InsecureServerCredentials(), &port_);
    builder.RegisterService(&server_impl_);
    server_ = builder.BuildAndStart();
  }
  // Variables that must be in scope for the lifetime of a test but are not
  // used by test code.
  int port_ = 0;
  std::unique_ptr<grpc::Server> server_;
};

TEST_F(GrpcBidiStreamTest, ClientContainsPopulationMetadata) {
  CreateClient("population_name");
  ClientStreamMessage request;
  request.mutable_checkin_request();
  EXPECT_THAT(client_stream_->Send(&request), IsOk());
  ServerStreamMessage reply;
  EXPECT_THAT(client_stream_->Receive(&reply), IsOk());
  EXPECT_TRUE(reply.has_checkin_response()) << reply.DebugString();
  EXPECT_THAT(server_impl_.GetClientMetadata(),
              Contains(Pair(GrpcBidiStream::kApiKeyHeader, "none")));
  EXPECT_THAT(
      server_impl_.GetClientMetadata(),
      Contains(Pair(GrpcBidiStream::kPopulationNameHeader, "population_name")));
  client_stream_->Close();
  server_impl_.WaitForSessionDone();
}

TEST_F(GrpcBidiStreamTest, CancellationDuringBlockingOp) {
  CreateClient();
  auto pool = CreateThreadPoolScheduler(1);
  pool->Schedule([this]() {
    sleep(1);
    client_stream_->Close();
  });
  ServerStreamMessage reply;
  auto start = absl::Now();
  // Will block indefinitely, as the default FakeServer requires a request
  // before sending a response.
  EXPECT_THAT(client_stream_->Receive(&reply),
              IsCode(absl::StatusCode::kCancelled));
  EXPECT_GE(absl::Now() - start, absl::Seconds(1));

  server_impl_.WaitForSessionDone();

  // Idempotency check:
  client_stream_->Close();
  EXPECT_THAT(client_stream_->Receive(&reply), Not(IsOk()));
  pool->WaitUntilIdle();
}

TEST_F(GrpcBidiStreamTest, CancellationBeforeSend) {
  CreateClient();
  absl::Status status;
  client_stream_->Close();
  server_impl_.WaitForSessionDone();
  ClientStreamMessage request;
  request.mutable_checkin_request();
  EXPECT_THAT(client_stream_->Send(&request),
              IsCode(absl::StatusCode::kCancelled));
}

TEST_F(GrpcBidiStreamTest, CancellationBeforeReceive) {
  CreateClient();
  ClientStreamMessage request;
  request.mutable_checkin_request();
  EXPECT_THAT(client_stream_->Send(&request), IsOk());
  client_stream_->Close();
  server_impl_.WaitForSessionDone();
  ServerStreamMessage reply;
  EXPECT_THAT(client_stream_->Receive(&reply),
              IsCode(absl::StatusCode::kCancelled));
  // Idempotency check:
  EXPECT_THAT(client_stream_->Receive(&reply),
              IsCode(absl::StatusCode::kCancelled));
}

TEST_F(GrpcBidiStreamTest, CancellationWithoutBlockingOp) {
  CreateClient();
  ClientStreamMessage request;
  request.mutable_checkin_request();
  EXPECT_THAT(client_stream_->Send(&request), IsOk());
  ServerStreamMessage reply;
  EXPECT_THAT(client_stream_->Receive(&reply), IsOk());
  EXPECT_TRUE(reply.has_checkin_response()) << reply.DebugString();
  EXPECT_THAT(server_impl_.GetClientMetadata(),
              Contains(Pair(GrpcBidiStream::kApiKeyHeader, "none")));
  EXPECT_THAT(server_impl_.GetClientMetadata(),
              Contains(Pair(GrpcBidiStream::kPopulationNameHeader, "")));

  client_stream_->Close();
  server_impl_.WaitForSessionDone();
}

}  // namespace
}  // namespace test
}  // namespace client
}  // namespace fcp
