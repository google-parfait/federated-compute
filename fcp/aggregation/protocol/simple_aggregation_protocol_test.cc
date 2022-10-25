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
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/testing/testing.h"

namespace fcp::aggregation {
namespace {

using ::testing::Invoke;
using ::testing::Return;
using ::testing::StrEq;

// TODO(team): Consider moving mock classes into a separate test library.
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
  MOCK_METHOD(void, AcceptClients, (const AcceptanceMessage& message),
              (override));
  MOCK_METHOD(void, SendServerMessage,
              (int64_t client_id, const ServerMessage& message), (override));
  MOCK_METHOD(void, CloseClient,
              (int64_t client_id, absl::Status diagnostic_status), (override));
  MOCK_METHOD(void, Complete, (absl::Cord result), (override));
  MOCK_METHOD(void, Abort, (absl::Status diagnostic_status), (override));
};

class SimpleAggregationProtocolTest : public ::testing::Test {
 public:
  SimpleAggregationProtocolTest()
      : wrapped_checkpoint_builder_(std::make_unique<MockCheckpointBuilder>()),
        checkpoint_builder_(*wrapped_checkpoint_builder_) {}

 protected:
  MockAggregationProtocolCallback callback_;
  MockCheckpointBuilderFactory checkpoint_builder_factory_;
  std::unique_ptr<MockCheckpointBuilder> wrapped_checkpoint_builder_;
  MockCheckpointBuilder& checkpoint_builder_;
};

TEST_F(SimpleAggregationProtocolTest, CreateAndGetResult_Success) {
  // Configuration with two intrinsics.
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

  // Verify that the protocol can be created successfully.
  absl::StatusOr<std::unique_ptr<SimpleAggregationProtocol>>
      protocol_or_status = SimpleAggregationProtocol::Create(
          config_message, &callback_, &checkpoint_builder_factory_);
  EXPECT_THAT(protocol_or_status, IsOk());
  std::unique_ptr<SimpleAggregationProtocol> protocol =
      std::move(protocol_or_status).value();

  // Verify that the checkpoint builder is created.
  EXPECT_CALL(checkpoint_builder_factory_, Create()).WillOnce(Invoke([&] {
    return std::move(wrapped_checkpoint_builder_);
  }));

  // Verify that foo_out and bar_out tensors are added to the result checkpoint
  EXPECT_CALL(
      checkpoint_builder_,
      Add(StrEq("foo_out"), IsTensor<int32_t>({2, 3}, {0, 0, 0, 0, 0, 0})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder_,
              Add(StrEq("bar_out"), IsTensor<float>({}, {0})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(checkpoint_builder_, Build()).WillOnce(Return(absl::Cord{}));

  // Verify that the Complete callback method is called.
  // TODO(team): Expect more callback methods to be called.
  EXPECT_CALL(callback_, Complete);

  EXPECT_THAT(protocol->Complete(), IsOk());
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
                                                &checkpoint_builder_factory_),
              IsCode(NOT_FOUND));
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
                                                &checkpoint_builder_factory_),
              IsCode(INVALID_ARGUMENT));
}

}  // namespace
}  // namespace fcp::aggregation
