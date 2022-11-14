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

#include "fcp/aggregation/protocol/simple_aggregation_protocol.h"

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/base/monitoring.h"
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

class MockAggregationProtocolCallback : public AggregationProtocol::Callback {
 public:
  MOCK_METHOD(void, AcceptClients,
              (int64_t start_client_id, int64_t num_clients,
               const AcceptanceMessage& message),
              (override));
  MOCK_METHOD(void, SendServerMessage,
              (int64_t client_id, const ServerMessage& message), (override));
  MOCK_METHOD(void, CloseClient,
              (int64_t client_id, absl::Status diagnostic_status), (override));
  MOCK_METHOD(void, Complete, (absl::Cord result), (override));
  MOCK_METHOD(void, Abort, (absl::Status diagnostic_status), (override));
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
          &checkpoint_builder_factory_);
  EXPECT_THAT(protocol_or_status, IsOk());
  return std::move(protocol_or_status).value();
}

TEST_F(SimpleAggregationProtocolTest, CreateAndGetResult_Success) {
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

  EXPECT_CALL(callback_, AcceptClients(0, 1, _));
  EXPECT_THAT(protocol->Start(1), IsOk());

  // Verify that the checkpoint builder is created.
  auto& checkpoint_builder = ExpectCheckpointBuilder();

  // Verify that foo_out and bar_out tensors are added to the result checkpoint
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("foo_out"), IsTensor({2, 3}, {0, 0, 0, 0, 0, 0})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder, Add(StrEq("bar_out"), IsTensor({}, {0.f})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder, Build()).WillOnce(Return(absl::Cord{}));

  // Verify that the Complete callback method is called.
  // TODO(team): Expect more callback methods to be called.
  EXPECT_CALL(callback_, Complete);

  EXPECT_THAT(protocol->Complete(), IsOk());

  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_pending: 1")));
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
  EXPECT_THAT(SimpleAggregationProtocol::Create(config_message, &callback_,
                                                &checkpoint_parser_factory_,
                                                &checkpoint_builder_factory_),
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
  EXPECT_THAT(SimpleAggregationProtocol::Create(config_message, &callback_,
                                                &checkpoint_parser_factory_,
                                                &checkpoint_builder_factory_),
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
  EXPECT_THAT(SimpleAggregationProtocol::Create(config_message, &callback_,
                                                &checkpoint_parser_factory_,
                                                &checkpoint_builder_factory_),
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
  EXPECT_THAT(SimpleAggregationProtocol::Create(config_message, &callback_,
                                                &checkpoint_parser_factory_,
                                                &checkpoint_builder_factory_),
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
  EXPECT_THAT(SimpleAggregationProtocol::Create(config_message, &callback_,
                                                &checkpoint_parser_factory_,
                                                &checkpoint_builder_factory_),
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
  EXPECT_THAT(SimpleAggregationProtocol::Create(config_message, &callback_,
                                                &checkpoint_parser_factory_,
                                                &checkpoint_builder_factory_),
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
  EXPECT_THAT(SimpleAggregationProtocol::Create(config_message, &callback_,
                                                &checkpoint_parser_factory_,
                                                &checkpoint_builder_factory_),
              IsCode(INVALID_ARGUMENT));
}

// TODO(team): Add similar tests for other callbacks.
TEST_F(SimpleAggregationProtocolTest,
       StartProtocol_AcceptClientsProtocolReentrace) {
  // This verifies that the protocol can be re-entered from the callback.
  auto protocol = CreateProtocolWithDefaultConfig();

  EXPECT_CALL(callback_, AcceptClients(0, 1, _)).WillOnce(Invoke([&]() {
    EXPECT_THAT(
        protocol->GetStatus(),
        EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_pending: 1")));
  }));

  EXPECT_THAT(protocol->Start(1), IsOk());
}

TEST_F(SimpleAggregationProtocolTest, StartProtocol_MultipleCalls) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, AcceptClients).Times(1);
  EXPECT_THAT(protocol->Start(1), IsOk());
  // The second Start call must fail.
  EXPECT_THAT(protocol->Start(1), IsCode(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, AddClients_Success) {
  auto protocol = CreateProtocolWithDefaultConfig();

  EXPECT_CALL(callback_, AcceptClients(0, 1, _));
  EXPECT_THAT(protocol->Start(1), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_pending: 1")));

  EXPECT_CALL(callback_, AcceptClients(1, 3, _));
  EXPECT_THAT(protocol->AddClients(3), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_pending: 4")));
}

TEST_F(SimpleAggregationProtocolTest, AddClients_ProtocolNotStarted) {
  auto protocol = CreateProtocolWithDefaultConfig();
  // Must fail because the protocol isn't started.
  EXPECT_CALL(callback_, AcceptClients).Times(0);
  EXPECT_THAT(protocol->AddClients(1), IsCode(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientInput_Success) {
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
  EXPECT_CALL(callback_, AcceptClients);
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

  EXPECT_CALL(callback_, CloseClient(Eq(0), IsCode(OK)));
  EXPECT_CALL(callback_, CloseClient(Eq(1), IsCode(OK)));

  // Handle the inputs.
  EXPECT_THAT(protocol->ReceiveClientInput(0, absl::Cord{}), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
                  "num_clients_pending: 1 num_clients_completed: 1 "
                  "num_inputs_aggregated_and_included: 1")));

  EXPECT_THAT(protocol->ReceiveClientInput(1, absl::Cord{}), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
          "num_clients_completed: 2 num_inputs_aggregated_and_included: 2")));

  // TODO(team): Verify the number of inputs processed.

  // Complete the protocol.
  // Verify that the checkpoint builder is created.
  auto& checkpoint_builder = ExpectCheckpointBuilder();

  // Verify that foo_out and bar_out tensors are added to the result checkpoint
  EXPECT_CALL(checkpoint_builder,
              Add(StrEq("foo_out"), IsTensor({2, 3}, {5, 11, 13, 17, 14, 8})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder, Add(StrEq("bar_out"), IsTensor({}, {3.f})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder, Build()).WillOnce(Return(absl::Cord{}));

  // Verify that the Complete callback method is called.
  // TODO(team): Expect more callback methods to be called.
  EXPECT_CALL(callback_, Complete);

  EXPECT_THAT(protocol->Complete(), IsOk());
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientInput_ProtocolNotStarted) {
  auto protocol = CreateProtocolWithDefaultConfig();
  // Must fail because the protocol isn't started.
  EXPECT_THAT(protocol->ReceiveClientInput(0, absl::Cord{}),
              IsCode(FAILED_PRECONDITION));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientInput_InvalidClientId) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, AcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());
  // Must fail for the client_id -1 and 2.
  EXPECT_THAT(protocol->ReceiveClientInput(-1, absl::Cord{}),
              IsCode(INVALID_ARGUMENT));
  EXPECT_THAT(protocol->ReceiveClientInput(2, absl::Cord{}),
              IsCode(INVALID_ARGUMENT));
}

TEST_F(SimpleAggregationProtocolTest,
       ReceiveClientInput_DuplicateClientIdInputs) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, AcceptClients);
  EXPECT_THAT(protocol->Start(2), IsOk());

  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  EXPECT_THAT(protocol->ReceiveClientInput(0, absl::Cord{}), IsOk());
  // The second input for the same client must succeed to without changing the
  // aggregated state.
  EXPECT_THAT(protocol->ReceiveClientInput(0, absl::Cord{}), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
                  "num_clients_pending: 1 num_clients_completed: 1 "
                  "num_inputs_aggregated_and_included: 1")));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientInput_AfterClosingClient) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, AcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());

  EXPECT_THAT(protocol->CloseClient(0, absl::OkStatus()), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
                  "num_clients_completed: 1 num_inputs_discarded: 1")));
  // This must succeed to without changing the aggregated state.
  EXPECT_THAT(protocol->ReceiveClientInput(0, absl::Cord{}), IsOk());
  EXPECT_THAT(protocol->GetStatus(),
              EqualsProto<StatusMessage>(PARSE_TEXT_PROTO(
                  "num_clients_completed: 1 num_inputs_discarded: 1")));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientInput_FailureToParseInput) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, AcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(
          Return(ByMove(absl::InvalidArgumentError("Invalid checkpoint"))));

  EXPECT_CALL(callback_, CloseClient(Eq(0), IsCode(INVALID_ARGUMENT)));

  // Receiving the client input should still succeed.
  EXPECT_THAT(protocol->ReceiveClientInput(0, absl::Cord{}), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_failed: 1")));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientInput_MissingTensor) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, AcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());

  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo")))
      .WillOnce(Return(ByMove(absl::NotFoundError("Missing tensor foo"))));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  EXPECT_CALL(callback_, CloseClient(Eq(0), IsCode(NOT_FOUND)));

  // Receiving the client input should still succeed.
  EXPECT_THAT(protocol->ReceiveClientInput(0, absl::Cord{}), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_failed: 1")));
}

TEST_F(SimpleAggregationProtocolTest, ReceiveClientInput_MismatchingTensor) {
  auto protocol = CreateProtocolWithDefaultConfig();
  EXPECT_CALL(callback_, AcceptClients);
  EXPECT_THAT(protocol->Start(1), IsOk());

  auto parser = std::make_unique<MockCheckpointParser>();
  EXPECT_CALL(*parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {}, CreateTestData({2.f}));
  }));

  EXPECT_CALL(checkpoint_parser_factory_, Create(_))
      .WillOnce(Return(ByMove(std::move(parser))));

  EXPECT_CALL(callback_, CloseClient(Eq(0), IsCode(INVALID_ARGUMENT)));

  // Receiving the client input should still succeed.
  EXPECT_THAT(protocol->ReceiveClientInput(0, absl::Cord{}), IsOk());
  EXPECT_THAT(
      protocol->GetStatus(),
      EqualsProto<StatusMessage>(PARSE_TEXT_PROTO("num_clients_failed: 1")));
}

}  // namespace
}  // namespace fcp::aggregation
