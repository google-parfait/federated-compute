#include "fcp/aggregation/core/config_converter.h"

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

TEST(ConfigConverterTest, ConvertEmpty) {
  Configuration::ServerAggregationConfig config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_uri: "my_intrinsic"
  )pb");
  StatusOr<Intrinsic> parsed_intrinsic = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsic, IsOk());
  ASSERT_THAT(parsed_intrinsic.value(),
              EqIntrinsic(Intrinsic{"my_intrinsic", {}, {}, {}, {}}));
}

TEST(ConfigConverterTest, ConvertInputs) {
  Configuration::ServerAggregationConfig config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_uri: "my_intrinsic"
    intrinsic_args {
      input_tensor {
        name: "foo"
        dtype: DT_INT32
        shape { dim { size: 8 } }
      }
    }
    intrinsic_args {
      input_tensor {
        name: "bar"
        dtype: DT_FLOAT
        shape {
          dim { size: 2 }
          dim { size: 3 }
        }
      }
    }
  )pb");

  StatusOr<Intrinsic> parsed_intrinsic = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsic, IsOk());
  ASSERT_THAT(parsed_intrinsic.value(),
              EqIntrinsic(Intrinsic{"my_intrinsic",
                                    {TensorSpec{"foo", DT_INT32, {8}},
                                     TensorSpec{"bar", DT_FLOAT, {2, 3}}},
                                    {},
                                    {},
                                    {}}));
}

TEST(ConfigConverterTest, ConvertOutputs) {
  Configuration::ServerAggregationConfig config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_uri: "my_intrinsic"
    output_tensors {
      name: "foo_out"
      dtype: DT_INT32
      shape { dim { size: 16 } }
    }
    output_tensors {
      name: "bar_out"
      dtype: DT_FLOAT
      shape {
        dim { size: 3 }
        dim { size: 4 }
      }
    }
  )pb");
  StatusOr<Intrinsic> parsed_intrinsic = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsic, IsOk());
  ASSERT_THAT(parsed_intrinsic.value(),
              EqIntrinsic(Intrinsic{"my_intrinsic",
                                    {},
                                    {TensorSpec{"foo_out", DT_INT32, {16}},
                                     TensorSpec{"bar_out", DT_FLOAT, {3, 4}}},
                                    {},
                                    {}}));
}

TEST(ConfigConverterTest, ConvertParams) {
  Configuration::ServerAggregationConfig config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_uri: "my_intrinsic"
    intrinsic_args {
      parameter {
        dtype: DT_FLOAT
        tensor_shape {
          dim { size: 2 }
          dim { size: 3 }
        }
        float_val: 1
        float_val: 2
        float_val: 3
        float_val: 4
        float_val: 5
        float_val: 6
      }
    }
  )pb");
  StatusOr<Intrinsic> parsed_intrinsic = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsic, IsOk());
  Tensor expected_tensor =
      Tensor::Create(DT_FLOAT, {2, 3},
                     CreateTestData<float>({1, 2, 3, 4, 5, 6}))
          .value();
  Intrinsic expected{"my_intrinsic", {}, {}, {}, {}};
  expected.parameters.push_back(std::move(expected_tensor));
  ASSERT_THAT(parsed_intrinsic.value(), EqIntrinsic(std::move(expected)));
}

TEST(ConfigConverterTest, ConvertInnerAggregations) {
  Configuration::ServerAggregationConfig config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_uri: "my_intrinsic"
    inner_aggregations {
      intrinsic_uri: "inner_intrinsic"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim { size: 8 } }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape { dim { size: 16 } }
      }
    }
  )pb");
  StatusOr<Intrinsic> parsed_intrinsic = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsic, IsOk());
  Intrinsic expected_inner = Intrinsic{"inner_intrinsic",
                                       {TensorSpec{"foo", DT_INT32, {8}}},
                                       {TensorSpec{"foo_out", DT_INT32, {16}}},
                                       {},
                                       {}};
  Intrinsic expected{"my_intrinsic", {}, {}, {}, {}};
  expected.nested_intrinsics.push_back(std::move(expected_inner));
  ASSERT_THAT(parsed_intrinsic.value(), EqIntrinsic(std::move(expected)));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
