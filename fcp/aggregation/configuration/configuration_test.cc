// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fcp/aggregation/configuration/configuration.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "fcp/aggregation/configuration/configuration.pb.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_aggregator_factory.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::SizeIs;

class MockFactory : public TensorAggregatorFactory {
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<TensorAggregator>>, Create,
              (const Intrinsic&), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<TensorAggregator>>, Deserialize,
              (const Intrinsic&, std::string), (const, override));
};

class ConfigurationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // The registry keeps a pointer to the factory, so it must outlive the
    // tests.
    static MockFactory* mock_factory = new MockFactory();
    RegisterAggregatorFactory("my_intrinsic", mock_factory);
    RegisterAggregatorFactory("other_intrinsic", mock_factory);
    RegisterAggregatorFactory("GoogleSQL:sum", mock_factory);
  }
};

TEST_F(ConfigurationTest, ParseEmpty) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: { intrinsic_uri: "my_intrinsic" }
  )pb");
  FCP_ASSERT_OK_AND_ASSIGN(std::vector<Intrinsic> intrinsics,
                           ParseConfiguration(config));
  ASSERT_THAT(intrinsics, SizeIs(1));
  EXPECT_THAT(intrinsics[0],
              EqIntrinsic(Intrinsic{"my_intrinsic", {}, {}, {}, {}}));
}

TEST_F(ConfigurationTest, ParseNoIntrinsics) {
  Configuration config;
  FCP_ASSERT_OK_AND_ASSIGN(std::vector<Intrinsic> intrinsics,
                           ParseConfiguration(config));
  EXPECT_THAT(intrinsics, SizeIs(0));
}

TEST_F(ConfigurationTest, ParseInputs) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim_sizes: 8 }
        }
      }
      intrinsic_args {
        input_tensor {
          name: "bar"
          dtype: DT_FLOAT
          shape { dim_sizes: 2 dim_sizes: 3 }
        }
      }
    }
  )pb");
  FCP_ASSERT_OK_AND_ASSIGN(std::vector<Intrinsic> intrinsics,
                           ParseConfiguration(config));
  ASSERT_THAT(intrinsics, SizeIs(1));
  EXPECT_THAT(intrinsics[0],
              EqIntrinsic(Intrinsic{"my_intrinsic",
                                    {TensorSpec{"foo", DT_INT32, {8}},
                                     TensorSpec{"bar", DT_FLOAT, {2, 3}}},
                                    {},
                                    {},
                                    {}}));
}

TEST_F(ConfigurationTest, ParseOutputs) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape { dim_sizes: 16 }
      }
      output_tensors {
        name: "bar_out"
        dtype: DT_FLOAT
        shape { dim_sizes: 3 dim_sizes: 4 }
      }
    }
  )pb");
  FCP_ASSERT_OK_AND_ASSIGN(std::vector<Intrinsic> intrinsics,
                           ParseConfiguration(config));
  ASSERT_THAT(intrinsics, SizeIs(1));
  EXPECT_THAT(intrinsics[0],
              EqIntrinsic(Intrinsic{"my_intrinsic",
                                    {},
                                    {TensorSpec{"foo_out", DT_INT32, {16}},
                                     TensorSpec{"bar_out", DT_FLOAT, {3, 4}}},
                                    {},
                                    {}}));
}

TEST_F(ConfigurationTest, ParseParams) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      intrinsic_args {
        parameter {
          dtype: DT_FLOAT
          shape { dim_sizes: 2 dim_sizes: 3 }
        }
      }
    }
  )pb");
  std::initializer_list<float> values{1, 2, 3, 4, 5, 6};
  std::string data =
      std::string(reinterpret_cast<char*>(std::vector(values).data()),
                  values.size() * sizeof(float));
  *config.mutable_intrinsic_configs(0)
       ->mutable_intrinsic_args(0)
       ->mutable_parameter()
       ->mutable_content() = data;
  FCP_ASSERT_OK_AND_ASSIGN(std::vector<Intrinsic> intrinsics,
                           ParseConfiguration(config));
  ASSERT_THAT(intrinsics, SizeIs(1));
  FCP_ASSERT_OK_AND_ASSIGN(
      Tensor expected_tensor,
      Tensor::Create(DT_FLOAT, {2, 3}, CreateTestData<float>(values)));
  Intrinsic expected{"my_intrinsic", {}, {}, {}, {}};
  expected.parameters.push_back(std::move(expected_tensor));
  EXPECT_THAT(intrinsics[0], EqIntrinsic(std::move(expected)));
}

TEST_F(ConfigurationTest, ParseInnerIntrinsics) {
  // Inner intrinsics don't need to be registered; they are validated by the
  // factory of the parent intrinsic.
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      inner_intrinsics {
        intrinsic_uri: "unregistered_inner_intrinsic"
        intrinsic_args {
          input_tensor {
            name: "foo"
            dtype: DT_INT32
            shape { dim_sizes: 8 }
          }
        }
        output_tensors {
          name: "foo_out"
          dtype: DT_INT32
          shape { dim_sizes: 16 }
        }
      }
    }
  )pb");
  FCP_ASSERT_OK_AND_ASSIGN(std::vector<Intrinsic> intrinsics,
                           ParseConfiguration(config));
  ASSERT_THAT(intrinsics, SizeIs(1));
  Intrinsic expected_inner{"unregistered_inner_intrinsic",
                           {TensorSpec{"foo", DT_INT32, {8}}},
                           {TensorSpec{"foo_out", DT_INT32, {16}}},
                           {},
                           {}};
  Intrinsic expected{"my_intrinsic", {}, {}, {}, {}};
  expected.nested_intrinsics.push_back(std::move(expected_inner));
  EXPECT_THAT(intrinsics[0], EqIntrinsic(std::move(expected)));
}

TEST_F(ConfigurationTest, ParseMultipleIntrinsicsPreservesOrder) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: { intrinsic_uri: "other_intrinsic" }
    intrinsic_configs: { intrinsic_uri: "my_intrinsic" }
  )pb");
  FCP_ASSERT_OK_AND_ASSIGN(std::vector<Intrinsic> intrinsics,
                           ParseConfiguration(config));
  ASSERT_THAT(intrinsics, SizeIs(2));
  EXPECT_THAT(intrinsics[0],
              EqIntrinsic(Intrinsic{"other_intrinsic", {}, {}, {}, {}}));
  EXPECT_THAT(intrinsics[1],
              EqIntrinsic(Intrinsic{"my_intrinsic", {}, {}, {}, {}}));
}

TEST_F(ConfigurationTest, ParseDoesNotTransformIntrinsics) {
  // Top-level intrinsics are not wrapped and scalar shapes are not modified,
  // regardless of the intrinsic URI.
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "GoogleSQL:sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT64
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT64
        shape {}
      }
    }
  )pb");
  FCP_ASSERT_OK_AND_ASSIGN(std::vector<Intrinsic> intrinsics,
                           ParseConfiguration(config));
  ASSERT_THAT(intrinsics, SizeIs(1));
  EXPECT_THAT(intrinsics[0],
              EqIntrinsic(Intrinsic{"GoogleSQL:sum",
                                    {TensorSpec{"foo", DT_INT64, {}}},
                                    {TensorSpec{"foo_out", DT_INT64, {}}},
                                    {},
                                    {}}));
}

TEST_F(ConfigurationTest, UnsupportedIntrinsicUri) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: { intrinsic_uri: "my_intrinsic" }
    intrinsic_configs: { intrinsic_uri: "unsupported_intrinsic" }
  )pb");
  EXPECT_THAT(ParseConfiguration(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported_intrinsic is not a supported "
                                 "intrinsic_uri")));
}

TEST_F(ConfigurationTest, IntrinsicArgNotSet) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      intrinsic_args {}
    }
  )pb");
  EXPECT_THAT(ParseConfiguration(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("All intrinsic args must have an arg value")));
}

TEST_F(ConfigurationTest, InnerIntrinsicArgNotSet) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      inner_intrinsics {
        intrinsic_uri: "inner_intrinsic"
        intrinsic_args {}
      }
    }
  )pb");
  EXPECT_THAT(ParseConfiguration(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("All intrinsic args must have an arg value")));
}

TEST_F(ConfigurationTest, InvalidParameter) {
  // The content size doesn't match the shape.
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      intrinsic_args {
        parameter {
          dtype: DT_FLOAT
          shape { dim_sizes: 2 }
          content: "abc"
        }
      }
    }
  )pb");
  // Errors from parsing the tensor are passed through unchanged.
  EXPECT_THAT(ParseConfiguration(config),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("byte_size() must be a multiple")));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
