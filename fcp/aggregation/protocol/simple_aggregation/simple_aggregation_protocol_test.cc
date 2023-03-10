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

#include "fcp/aggregation/protocol/simple_aggregation/simple_aggregation_protocol.h"

#include <atomic>
#include <functional>
#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/notification.h"
#include "fcp/aggregation/core/agg_vector_aggregator.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/aggregation/protocol/testing/test_callback.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/scheduler.h"
#include "fcp/testing/testing.h"

namespace fcp::aggregation {
namespace {

using ::testing::_;
using ::testing::ByMove;
using ::testing::Eq;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::StrEq;

// TODO(team): Consider moving mock classes into a separate test library.
class MockCheckpointParser : public CheckpointParser {
 public:
  MOCK_METHOD(absl::StatusOr<Tensor>, GetTensor, (const std::string& name),
              (const override));
};

class MockCheckpointParserFactory : public CheckpointParserFactory {
 public:
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<CheckpointParser>>, Create,
              (const absl::Cord& serialized_checkpoint), (const override));
};

class MockCheckpointBuilder : public CheckpointBuilder {
 public:
  MOCK_METHOD(absl::Status, Add,
              (const std::string& name, const Tensor& tensor), (override));
  MOCK_METHOD(absl::StatusOr<absl::Cord>, Build, (), (override));
};

class MockCheckpointBuilderFactory : public CheckpointBuilderFactory {
 public:
  MOCK_METHOD(std::unique_ptr<CheckpointBuilder>, Create, (), (const override));
};

class MockResourceResolver : public ResourceResolver {
 public:
  MOCK_METHOD(absl::StatusOr<absl::Cord>, RetrieveResource,
              (int64_t client_id, const std::string& uri), (override));
};

class SimpleAggregationProtocolTest : public ::testing::Test {
 protected:
  // Returns default configuration.
  Configuration default_configuration() const;

  // Returns the default instance of checkpoint bilder;
  MockCheckpointBuilder& ExpectCheckpointBuilder() {
    MockCheckpointBuilder& checkpoint_builder = *wrapped_checkpoint_builder_;
    EXPECT_CALL(checkpoint_builder_factory_, Create())
        .WillOnce(Return(ByMove(std::move(wrapped_checkpoint_builder_))));
    EXPECT_CALL(checkpoint_builder, Build()).WillOnce(Return(absl::Cord{}));
    return checkpoint_builder;
  }

  // Creates an instance of SimpleAggregationProtocol with the specified config.
  std::unique_ptr<SimpleAggregationProtocol> CreateProtocol(
      Configuration config);

  // Creates an instance of SimpleAggregationProtocol with the default config.
  std::unique_ptr<SimpleAggregationProtocol> CreateProtocolWithDefaultConfig() {
    return CreateProtocol(default_configuration());
  }

  MockAggregationProtocolCallback callback_;

  MockCheckpointParserFactory checkpoint_parser_factory_;
  MockCheckpointBuilderFactory checkpoint_builder_factory_;
  MockResourceResolver resource_resolver_;

 private:
  std::unique_ptr<MockCheckpointBuilder> wrapped_checkpoint_builder_ =
      std::make_unique<MockCheckpointBuilder>();
};

Configuration SimpleAggregationProtocolTest::default_configuration() const {
  // One "federated_sum" intrinsic with a single scalar int32 tensor.
  return PARSE_TEXT_PROTO(R"pb(
    aggregation_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
}

std::unique_ptr<SimpleAggregationProtocol>
SimpleAggregationProtocolTest::CreateProtocol(Configuration config) {
  // Verify that the protocol can be created successfully.
  absl::StatusOr<std::unique_ptr<SimpleAggregationProtocol>>
      protocol_or_status = SimpleAggregationProtocol::Create(
          config, &callback_, &checkpoint_parser_factory_,
          &checkpoint_builder_factory_, &resource_resolver_);
  EXPECT_THAT(protocol_or_status, IsOk());
  return std::move(protocol_or_status).value();
}

ClientMessage MakeClientMessage() {
  ClientMessage message;
  message.mutable_simple_aggregation()->mutable_input()->set_inline_bytes("");
  return message;
}

TEST_F(SimpleAggregationProtocolTest, Create_UnsupportedNumberOfInputs) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    aggregation_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      intrinsic_args {
        input_tensor {
          name: "bar"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(SimpleAggregationProtocol::Create(
                  config_message, &callback_, &checkpoint_parser_factory_,
                  &checkpoint_builder_factory_, &resource_resolver_),
              IsCode(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, Create_UnsupportedNumberOfOutputs) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    aggregation_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
      output_tensors {
        name: "bar_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(SimpleAggregationProtocol::Create(
                  config_message, &callback_, &checkpoint_parser_factory_,
                  &checkpoint_builder_factory_, &resource_resolver_),
              IsCode(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, Create_UnsupportedInputType) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    aggregation_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args { parameter {} }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(SimpleAggregationProtocol::Create(
                  config_message, &callback_, &checkpoint_parser_factory_,
                  &checkpoint_builder_factory_, &resource_resolver_),
              IsCode(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, Create_UnsupportedIntrinsicUri) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    aggregation_configs {
      intrinsic_uri: "unsupported_xyz"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(SimpleAggregationProtocol::Create(
                  config_message, &callback_, &checkpoint_parser_factory_,
                  &checkpoint_builder_factory_, &resource_resolver_),
              IsCode(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, Create_UnsupportedInputSpec) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    aggregation_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim { size: -1 } }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(SimpleAggregationProtocol::Create(
                  config_message, &callback_, &checkpoint_parser_factory_,
                  &checkpoint_builder_factory_, &resource_resolver_),
              IsCode(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, Create_UnmatchingInputAndOutputDataType) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    aggregation_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_FLOAT
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(SimpleAggregationProtocol::Create(
                  config_message, &callback_, &checkpoint_parser_factory_,
                  &checkpoint_builder_factory_, &resource_resolver_),
              IsCode(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, Create_UnmatchingInputAndOutputShape) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    aggregation_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim { size: 1 } }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape { dim { size: 2 } }
      }
    }
  )pb");
  EXPECT_THAT(SimpleAggregationProtocol::Create(
                  config_message, &callback_, &checkpoint_parser_factory_,
                  &checkpoint_builder_factory_, &resource_resolver_),
              IsCode(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, StartProtocol_Success) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Start(3), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_pending: 3")));
}

// TODO(team): Add similar tests for other callbacks.
TEST_F(SimpleAggregationProtocolTest,
       StartProtocol_AcceptClientsProtocolReentrace) {
  // This verifies that the protocol can be re-entered from the callback.
  auto protocol = CreateProtocolWithDefaultConfig();

  EXPECT_CALL(callback_, OnAcceptClients(0, 1, _)).WillOnce(Invoke([&]() {
    EXPECT_THAT(
        protocol->GetStatus(),
        EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_pending: 1")));
  }));

  EXPECT_THAT(protocol->Start(1), IsOk());
}

TEST_F(SimpleAggregationProtocolTest, StartProtocol_MultipleCalls) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients).Times(1);
  EXPECT_THAT(protocol->Start(1), IsOk());
  // The second Start call must fail.
  EXPECT_THAT(protocol->Start(1), IsCode(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, StartProtocol_ZeroClients) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients).Times(0);
  EXPECT_THAT(protocol->Start(0), IsOk());
}

TEST_F(SimpleAggregationProtocolTest, StartProtocol_NegativeClients) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients).Times(0);
  EXPECT_THAT(protocol->Start(-1), IsCode(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, AddClients_Success) {
  auto protocol = CreateProtocolWithDefaultConfig();

  EXPECT_CALL(callback_, OnAcceptClients(0, 1, _));
  EXPECT_THAT(protocol->Start(1), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_pending: 1")));

  EXPECT_CALL(callback_, OnAcceptClients(1, 3, _));
  EXPECT_THAT(protocol->AddClients(3), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_pending: 4")));
}

TEST_F(SimpleAggregationProtocolTest, AddClients_ProtocolNotStarted) {
  auto protocol = CreateProtocolWithDefaultConfig();
  // Must fail because the protocol isn't started.
  EXPECT_CALL(callback_, OnAcceptClients).Times(0);
  EXPECT_THAT(protocol->AddClients(1), IsCode(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_ProtocolNotStarted) {
  auto protocol = CreateProtocolWithDefaultConfig();
  // Must fail because the protocol isn't started.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()),
              IsCode(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_InvalidMessage) {
  auto protocol = CreateProtocolWithDefaultConfig();
  ClientMessage message;
  // Empty message without SimpleAggregation.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, message),
              IsCode(INVALID_ARGUMENT));
  // Message with SimpleAggregation but without the input.
  message.mutable_simple_aggregation();
  EXPECT_THAT(protocol->ReceiveClientMessage(0, message),
              IsCode(INVALID_ARGUMENT));
  // Message with empty input.
  message.mutable_simple_aggregation()->mutable_input();
  EXPECT_THAT(protocol->ReceiveClientMessage(0, message),
              IsCode(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_InvalidClientId) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());
  // Must fail for the client_id -1 and 2.
  EXPECT_THAT(protocol->ReceiveClientMessage(-1, MakeClientMessage()),
              IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(protocol->ReceiveClientMessage(2, MakeClientMessage()),
              IsCode(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest,
       ReceiveClientMessage_DuplicateClientIdInputs) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients);
  EXPECT_THAT(protocol->Start(2), IsOk());

  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  // The second input for the same client must succeed to without changing the
  // aggregated state.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
                  "num_clients_pending: 1 num_clients_completed: 1 "
                  "num_inputs_aggregated_and_included: 1")));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_AfterClosingClient) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());

  EXPECT_THAT(protocol->CloseClient(0, absl::OkStatus()), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
                  "num_clients_completed: 1 num_inputs_discarded: 1")));
  // This must succeed to without changing the aggregated state.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
                  "num_clients_completed: 1 num_inputs_discarded: 1")));
}

TEST_F(SimpleAggregationProtocolTest,
       ReceiveClientMessage_FailureToParseInput) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(
          Return(ByMove(absl::InvalidArgumentError("Invalid checkpoint"))));

  EXPECT_CALL(callback_, OnCloseClient(Eq(0), IsCode(INVALID_ARGUMENT)));

  // Receiving the client input should still succeed.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_failed: 1")));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_MissingTensor) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());

  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo")))
      .WillOnce(Return(ByMove(absl::NotFoundError("Missing tensor foo"))));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  EXPECT_CALL(callback_, OnCloseClient(Eq(0), IsCode(NOT_FOUND)));

  // Receiving the client input should still succeed.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_failed: 1")));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_MismatchingTensor) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());

  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {}, CreateTestData({2.f}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  EXPECT_CALL(callback_, OnCloseClient(Eq(0), IsCode(INVALID_ARGUMENT)));

  // Receiving the client input should still succeed.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_failed: 1")));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientMessage_UriType_Success) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());
  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
  }));
  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  // Receive input for the client #0
  EXPECT_CALL(resource_resolver_, RetrieveResource(0, StrEq("foo_uri")))
      .WillOnce(Return(absl::Cord{}));
  ClientMessage message;
  message.mutable_simple_aggregation()->mutable_input()->set_uri("foo_uri");
  EXPECT_CALL(callback_, OnCloseClient(Eq(0), IsCode(OK)));
  EXPECT_THAT(protocol->ReceiveClientMessage(0, message), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
          "num_clients_completed: 1 num_inputs_aggregated_and_included: 1")));
}

TEST_F(SimpleAggregationProtocolTest,
       ReceiveClientMessage_UriType_FailToParse) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());

  // Receive invalid input for the client #0
  EXPECT_CALL(resource_resolver_, RetrieveResource(0, _))
      .WillOnce(Return(absl::InvalidArgumentError("Invalid uri")));
  ClientMessage message;
  message.mutable_simple_aggregation()->mutable_input()->set_uri("foo_uri");
  EXPECT_CALL(callback_, OnCloseClient(Eq(0), IsCode(INVALID_ARGUMENT)));

  // Receiving the client input should still succeed.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, message), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_failed: 1")));
}

TEST_F(SimpleAggregationProtocolTest, Complete_NoInputsReceived) {
  // Two intrinsics:
  // 1) federated_sum "foo" that takes int32 {2,3} tensors.
  // 2) federated_sum "bar" that takes scalar float tensors.
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    aggregation_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {
            dim { size: 2 }
            dim { size: 3 }
          }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {
          dim { size: 2 }
          dim { size: 3 }
        }
      }
    }
    aggregation_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "bar"
          dtype: DT_FLOAT
          shape {}
        }
      }
      output_tensors {
        name: "bar_out"
        dtype: DT_FLOAT
        shape {}
      }
    }
  )pb");
  auto protocol = CreateProtocol(config_message);

  EXPECT_CALL(callback_, OnAcceptClients(0, 1, _));
  EXPECT_THAT(protocol->Start(1), IsOk());

  // Verify that the checkpoint builder is created.
  auto& checkpoint_builder = ExpectCheckpointBuilder();

  // Verify that foo_out and bar_out tensors are added to the result checkpoint
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("foo_out"), IsTensor({2, 3}, {0, 0, 0, 0, 0, 0})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder, Add(StrEq("bar_out"), IsTensor({}, {0.f})))
      .WillOnce(Return(absl::OkStatus()));

  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_pending: 1")));

  // Verify that the pending client is closed.
  EXPECT_CALL(callback_, OnCloseClient(0, IsCode(ABORTED)));
  // Verify that the Complete callback method is called.
  EXPECT_CALL(callback_, OnComplete);

  EXPECT_THAT(protocol->Complete(), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_aborted: 1")));
}

TEST_F(SimpleAggregationProtocolTest, Complete_TwoInputsReceived) {
  // Two intrinsics:
  // 1) federated_sum "foo" that takes int32 {2,3} tensors.
  // 2) federated_sum "bar" that takes scalar float tensors.
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    aggregation_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {
            dim { size: 2 }
            dim { size: 3 }
          }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {
          dim { size: 2 }
          dim { size: 3 }
        }
      }
    }
    aggregation_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "bar"
          dtype: DT_FLOAT
          shape {}
        }
      }
      output_tensors {
        name: "bar_out"
        dtype: DT_FLOAT
        shape {}
      }
    }
  )pb");
  auto protocol = CreateProtocol(config_message);
  EXPECT_CALL(callback_, OnAcceptClients);
  EXPECT_THAT(protocol->Start(2), IsOk());

  // Expect two inputs.
  auto parser1 = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser1, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {2, 3},
                          CreateTestData({4, 3, 11, 7, 1, 6}));
  }));
  EXPECT_CALL(*parser1, GetTensor(StrEq("bar"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {}, CreateTestData({1.f}));
  }));

  auto parser2 = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser2, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {2, 3},
                          CreateTestData({1, 8, 2, 10, 13, 2}));
  }));
  EXPECT_CALL(*parser2, GetTensor(StrEq("bar"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {}, CreateTestData({2.f}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser1))))
      .WillOnce(Return(ByMove(std::move(parser2))));

  EXPECT_CALL(callback_, OnCloseClient(Eq(0), IsCode(OK)));
  EXPECT_CALL(callback_, OnCloseClient(Eq(1), IsCode(OK)));

  // Handle the inputs.
  EXPECT_THAT(protocol->ReceiveClientMessage(0, MakeClientMessage()), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
                  "num_clients_pending: 1 num_clients_completed: 1 "
                  "num_inputs_aggregated_and_included: 1")));

  EXPECT_THAT(protocol->ReceiveClientMessage(1, MakeClientMessage()), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
          "num_clients_completed: 2 num_inputs_aggregated_and_included: 2")));

  // Complete the protocol.
  // Verify that the checkpoint builder is created.
  auto& checkpoint_builder = ExpectCheckpointBuilder();

  // Verify that foo_out and bar_out tensors are added to the result checkpoint
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("foo_out"), IsTensor({2, 3}, {5, 11, 13, 17, 14, 8})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder, Add(StrEq("bar_out"), IsTensor({}, {3.f})))
      .WillOnce(Return(absl::OkStatus()));

  // Verify that the OnComplete callback method is called.
  EXPECT_CALL(callback_, OnComplete);

  EXPECT_THAT(protocol->Complete(), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
          "num_clients_completed: 2 num_inputs_aggregated_and_included: 2")));
}

TEST_F(SimpleAggregationProtocolTest, Complete_ProtocolNotStarted) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Complete(), IsCode(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, Abort_NoInputsReceived) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients(0, 2, _));
  EXPECT_THAT(protocol->Start(2), IsOk());

  EXPECT_CALL(callback_, OnCloseClient(Eq(0), IsCode(ABORTED)));
  EXPECT_CALL(callback_, OnCloseClient(Eq(1), IsCode(ABORTED)));
  EXPECT_THAT(protocol->Abort(), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_aborted: 2")));
}

TEST_F(SimpleAggregationProtocolTest, Abort_OneInputReceived) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients(0, 2, _));
  EXPECT_THAT(protocol->Start(2), IsOk());

  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  // Receive input for the client #1
  EXPECT_CALL(callback_, OnCloseClient(Eq(1), IsCode(OK)));
  EXPECT_THAT(protocol->ReceiveClientMessage(1, MakeClientMessage()), IsOk());

  // The client #0 should be aborted on Abort().
  EXPECT_CALL(callback_, OnCloseClient(Eq(0), IsCode(ABORTED)));
  EXPECT_THAT(protocol->Abort(), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
                  "num_clients_aborted: 1 num_clients_completed:1 "
                  "num_inputs_discarded: 1")));
}

TEST_F(SimpleAggregationProtocolTest, Abort_ProtocolNotStarted) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_THAT(protocol->Abort(), IsCode(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, ConcurrentAggregation_Success) {
  const int64_t kNumClients = 10;
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, OnAcceptClients(0, kNumClients, _));
  EXPECT_THAT(protocol->Start(kNumClients), IsOk());

  // The following block will repeatedly create CheckpointParser instances
  // which will be creating scalar int tensors with repeatedly incrementing
  // values.
  std::atomic<int> tensor_value = 0;
  EXPECT_CALL(checkpoint_parser_factory_, Create(_)).WillRepeatedly(Invoke([&] {
    auto parser = std::make_unique<MockCheckpointParser>();
    EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([&] {
      return Tensor::Create(DT_INT32, {}, CreateTestData({++tensor_value}));
    }));
    return parser;
  }));

  // Schedule receiving inputs on 4 concurrent threads.
  auto scheduler = CreateThreadPoolScheduler(4);
  for (int64_t i = 0; i < kNumClients; ++i) {
    scheduler->Schedule([&, i]() {
      EXPECT_THAT(protocol->ReceiveClientMessage(i, MakeClientMessage()),
                  IsOk());
    });
  }
  scheduler->WaitUntilIdle();

  // Complete the protocol.
  // Verify that the checkpoint builder is created.
  auto& checkpoint_builder = ExpectCheckpointBuilder();
  // Verify that foo_out tensor is added to the result checkpoint
  EXPECT_CALL(checkpoint_builder, Add(StrEq("foo_out"), IsTensor({}, {55})))
      .WillOnce(Return(absl::OkStatus()));

  // Verify that the OnComplete callback method is called.
  EXPECT_CALL(callback_, OnComplete);
  EXPECT_THAT(protocol->Complete(), IsOk());
}

// A trivial test aggregator that delegates aggregation to a function.
class FunctionAggregator final : public AggVectorAggregator<int> {
 public:
  using Func = std::function<int(int, int)>;

  FunctionAggregator(DataType dtype, TensorShape shape, Func agg_function)
      : AggVectorAggregator<int>(dtype, shape), agg_function_(agg_function) {}

 private:
  void AggregateVector(const AggVector<int>& agg_vector) override {
    for (auto [i, v] : agg_vector) {
      data()[i] = agg_function_(data()[i], v);
    }
  }

  const Func agg_function_;
};

// Factory for the FunctionAggregator.
class FunctionAggregatorFactory final : public TensorAggregatorFactory {
 public:
  explicit FunctionAggregatorFactory(FunctionAggregator::Func agg_function)
      : agg_function_(agg_function) {}

 private:
  absl::StatusOr<std::unique_ptr<TensorAggregator>> Create(
      DataType dtype, TensorShape shape) const override {
    if (dtype != DT_INT32) {
      return absl::InvalidArgumentError("Unsupported dtype: expected DT_INT32");
    }
    return std::make_unique<FunctionAggregator>(dtype, shape, agg_function_);
  }

  const FunctionAggregator::Func agg_function_;
};

TEST_F(SimpleAggregationProtocolTest, ConcurrentAggregation_AbortWhileQueued) {
  const int64_t kNumClients = 10;
  const int64_t kNumClientBeforeBlocking = 3;

  // Notifies the aggregation to unblock;
  absl::Notification resume_aggregation_notification;
  absl::Notification aggregation_blocked_notification;
  std::atomic<int> agg_counter = 0;
  FunctionAggregatorFactory agg_factory([&](int a, int b) {
    if (++agg_counter > kNumClientBeforeBlocking &&
        !aggregation_blocked_notification.HasBeenNotified()) {
      aggregation_blocked_notification.Notify();
      resume_aggregation_notification.WaitForNotification();
    }
    return a + b;
  });
  RegisterAggregatorFactory("foo1_aggregation", &agg_factory);

  // The configuration below refers to the custom aggregation registered
  // above.
  auto protocol = CreateProtocol(PARSE_TEXT_PROTO(R"pb(
    aggregation_configs {
      intrinsic_uri: "foo1_aggregation"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb"));
  EXPECT_CALL(callback_, OnAcceptClients(0, kNumClients, _));
  EXPECT_THAT(protocol->Start(kNumClients), IsOk());

  EXPECT_CALL(checkpoint_parser_factory_, Create(_)).WillRepeatedly(Invoke([&] {
    auto parser = std::make_unique<MockCheckpointParser>();
    EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([&] {
      return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
    }));
    return parser;
  }));

  // Schedule receiving inputs on 10 concurrent threads.
  auto scheduler = CreateThreadPoolScheduler(10);
  for (int64_t i = 0; i < kNumClients; ++i) {
    scheduler->Schedule([&, i]() {
      EXPECT_THAT(protocol->ReceiveClientMessage(i, MakeClientMessage()),
                  IsOk());
    });
  }

  aggregation_blocked_notification.WaitForNotification();

  StatusMessage status_message;
  do {
    status_message = protocol->GetStatus();
  } while (status_message.num_clients_pending() > 0);

  // At this point one input must be blocked inside the aggregation waiting for
  // the notification, 3 inputs should already be gone through the aggregation,
  // and the remaining 6 inputs should be blocked waiting to enter the
  // aggregation.

  // TODO(team): Need to revise the status implementation because it
  // treats received and pending (queued) inputs "as aggregated and pending".
  EXPECT_THAT(protocol->GetStatus(),
              EqualsProto<StatusMessage>(
                  PARSE_TEXT_PROTO("num_clients_completed: 10 "
                                   "num_inputs_aggregated_and_pending: 7 "
                                   "num_inputs_aggregated_and_included: 3")));

  resume_aggregation_notification.Notify();

  // Abort and let all blocked aggregations continue.
  EXPECT_THAT(protocol->Abort(), IsOk());
  scheduler->WaitUntilIdle();

  // All 10 inputs should now be discarded.
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_completed: 10 "
                                                  "num_inputs_discarded: 10")));
}

}  // namespace
}  // namespace fcp::aggregation
