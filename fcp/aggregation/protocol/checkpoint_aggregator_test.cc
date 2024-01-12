/*
 * Copyright 2024 Google LLC
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

#include "fcp/aggregation/protocol/checkpoint_aggregator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/base/monitoring.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

Configuration default_configuration() {
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

TEST(CheckpointAggregatorTest, CreateSuccess) {
  EXPECT_OK(CheckpointAggregator::Create(default_configuration()));
}

TEST(CheckpointAggregatorTest, CreateUnsupportedNumberOfInputs) {
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
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              IsCode(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateUnsupportedNumberOfOutputs) {
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
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              IsCode(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateUnsupportedInputType) {
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
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              IsCode(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateUnsupportedIntrinsicUri) {
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
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              IsCode(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateUnsupportedInputSpec) {
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
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              IsCode(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateMismatchingInputAndOutputDataType) {
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
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              IsCode(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateMismatchingInputAndOutputShape) {
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
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              IsCode(INVALID_ARGUMENT));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
