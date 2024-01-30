#include "fcp/aggregation/protocol/config_converter.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fcp/testing/parse_text_proto.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_aggregator_registry.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/core/tensor_spec.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/base/monitoring.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

using testing::SizeIs;

class MockFactory : public TensorAggregatorFactory {
  MOCK_METHOD(StatusOr<std::unique_ptr<TensorAggregator>>, Create,
              (const Intrinsic&), (const override));
};

class ConfigConverterTest : public ::testing::Test {
 protected:
  ConfigConverterTest() {
    if (!is_registered_) {
      MockFactory mock_factory;
      RegisterAggregatorFactory("my_intrinsic", &mock_factory);
      RegisterAggregatorFactory("inner_intrinsic", &mock_factory);
      RegisterAggregatorFactory("outer_intrinsic", &mock_factory);
      RegisterAggregatorFactory("other_intrinsic", &mock_factory);
      RegisterAggregatorFactory("fedsql_group_by", &mock_factory);
      RegisterAggregatorFactory("GoogleSQL:sum", &mock_factory);
      RegisterAggregatorFactory("GoogleSQL:max", &mock_factory);
      is_registered_ = true;
    }
  }

 private:
  static bool is_registered_;
};

bool ConfigConverterTest::is_registered_ = false;

TEST_F(ConfigConverterTest, ConvertEmpty) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: { intrinsic_uri: "my_intrinsic" }
  )pb");
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  ASSERT_THAT(parsed_intrinsics.value()[0],
              EqIntrinsic(Intrinsic{"my_intrinsic", {}, {}, {}, {}}));
}

TEST_F(ConfigConverterTest, ConvertInputs) {
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

  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  ASSERT_THAT(parsed_intrinsics.value()[0],
              EqIntrinsic(Intrinsic{"my_intrinsic",
                                    {TensorSpec{"foo", DT_INT32, {8}},
                                     TensorSpec{"bar", DT_FLOAT, {2, 3}}},
                                    {},
                                    {},
                                    {}}));
}

TEST_F(ConfigConverterTest, ConvertOutputs) {
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
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  ASSERT_THAT(parsed_intrinsics.value()[0],
              EqIntrinsic(Intrinsic{"my_intrinsic",
                                    {},
                                    {TensorSpec{"foo_out", DT_INT32, {16}},
                                     TensorSpec{"bar_out", DT_FLOAT, {3, 4}}},
                                    {},
                                    {}}));
}

TEST_F(ConfigConverterTest, ConvertParams) {
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
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  Tensor expected_tensor =
      Tensor::Create(DT_FLOAT, {2, 3}, CreateTestData<float>(values)).value();
  Intrinsic expected{"my_intrinsic", {}, {}, {}, {}};
  expected.parameters.push_back(std::move(expected_tensor));
  ASSERT_THAT(parsed_intrinsics.value()[0], EqIntrinsic(std::move(expected)));
}

TEST_F(ConfigConverterTest, ConvertInnerAggregations) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      inner_intrinsics {
        intrinsic_uri: "inner_intrinsic"
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
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  Intrinsic expected_inner = Intrinsic{"inner_intrinsic",
                                       {TensorSpec{"foo", DT_INT32, {8}}},
                                       {TensorSpec{"foo_out", DT_INT32, {16}}},
                                       {},
                                       {}};
  Intrinsic expected{"my_intrinsic", {}, {}, {}, {}};
  expected.nested_intrinsics.push_back(std::move(expected_inner));
  ASSERT_THAT(parsed_intrinsics.value()[0], EqIntrinsic(std::move(expected)));
}

TEST_F(ConfigConverterTest, ConvertFedSql_GroupByAlreadyPresent) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_group_by"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim_sizes: -1 }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape { dim_sizes: -1 }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "bar"
            dtype: DT_INT32
            shape {}
          }
        }
        output_tensors {
          name: "bar_out"
          dtype: DT_INT32
          shape {}
        }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:max"
        intrinsic_args {
          input_tensor {
            name: "baz"
            dtype: DT_INT32
            shape {}
          }
        }
        output_tensors {
          name: "baz_out"
          dtype: DT_INT32
          shape {}
        }
      }
    }
  )pb");
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  Intrinsic expected_sum = Intrinsic{"GoogleSQL:sum",
                                     {TensorSpec{"bar", DT_INT32, {-1}}},
                                     {TensorSpec{"bar_out", DT_INT32, {-1}}},
                                     {},
                                     {}};
  Intrinsic expected_max = Intrinsic{"GoogleSQL:max",
                                     {TensorSpec{"baz", DT_INT32, {-1}}},
                                     {TensorSpec{"baz_out", DT_INT32, {-1}}},
                                     {},
                                     {}};

  Intrinsic expected{"fedsql_group_by",
                     {TensorSpec{"foo", DT_INT32, {-1}}},
                     {TensorSpec{"foo_out", DT_INT32, {-1}}},
                     {},
                     {}};
  expected.nested_intrinsics.push_back(std::move(expected_sum));
  expected.nested_intrinsics.push_back(std::move(expected_max));
  ASSERT_THAT(parsed_intrinsics.value()[0], EqIntrinsic(std::move(expected)));
}

TEST_F(ConfigConverterTest, ConvertFedSql_WrapWhenGroupByNotPresent) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "GoogleSQL:sum"
      intrinsic_args {
        input_tensor {
          name: "bar"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "bar_out"
        dtype: DT_INT32
        shape {}
      }
    }
    intrinsic_configs: {
      intrinsic_uri: "other_intrinsic"
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
    intrinsic_configs: {
      intrinsic_uri: "GoogleSQL:max"
      intrinsic_args {
        input_tensor {
          name: "baz"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "baz_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  // Even though there are three top level intrinsics in the configuration, the
  // two fedsql intrinsics should be wrapped by a group by intrinsic so only two
  // toplevel intrinsics will be present in the output.
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(2));
  Intrinsic expected_sum = Intrinsic{"GoogleSQL:sum",
                                     {TensorSpec{"bar", DT_INT32, {-1}}},
                                     {TensorSpec{"bar_out", DT_INT32, {-1}}},
                                     {},
                                     {}};
  Intrinsic expected_max = Intrinsic{"GoogleSQL:max",
                                     {TensorSpec{"baz", DT_INT32, {-1}}},
                                     {TensorSpec{"baz_out", DT_INT32, {-1}}},
                                     {},
                                     {}};

  Intrinsic expected_other{"other_intrinsic",
                           {TensorSpec{"foo", DT_INT32, {}}},
                           {TensorSpec{"foo_out", DT_INT32, {}}},
                           {},
                           {}};
  Intrinsic expected_groupby{"fedsql_group_by", {}, {}, {}, {}};
  expected_groupby.nested_intrinsics.push_back(std::move(expected_sum));
  expected_groupby.nested_intrinsics.push_back(std::move(expected_max));
  ASSERT_THAT(parsed_intrinsics.value()[0],
              EqIntrinsic(std::move(expected_other)));
  ASSERT_THAT(parsed_intrinsics.value()[1],
              EqIntrinsic(std::move(expected_groupby)));
}

TEST_F(ConfigConverterTest, ConvertFedSql_WrapWhenGroupByNotPresent_Nested) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "outer_intrinsic"
      inner_intrinsics: {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "bar"
            dtype: DT_INT32
            shape {}
          }
        }
        output_tensors {
          name: "bar_out"
          dtype: DT_INT32
          shape {}
        }
      }
      inner_intrinsics: {
        intrinsic_uri: "other_intrinsic"
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
      inner_intrinsics: {
        intrinsic_uri: "GoogleSQL:max"
        intrinsic_args {
          input_tensor {
            name: "baz"
            dtype: DT_INT32
            shape {}
          }
        }
        output_tensors {
          name: "baz_out"
          dtype: DT_INT32
          shape {}
        }
      }
    }
  )pb");
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  Intrinsic expected_sum = Intrinsic{"GoogleSQL:sum",
                                     {TensorSpec{"bar", DT_INT32, {-1}}},
                                     {TensorSpec{"bar_out", DT_INT32, {-1}}},
                                     {},
                                     {}};
  Intrinsic expected_max = Intrinsic{"GoogleSQL:max",
                                     {TensorSpec{"baz", DT_INT32, {-1}}},
                                     {TensorSpec{"baz_out", DT_INT32, {-1}}},
                                     {},
                                     {}};
  Intrinsic expected_other{"other_intrinsic",
                           {TensorSpec{"foo", DT_INT32, {}}},
                           {TensorSpec{"foo_out", DT_INT32, {}}},
                           {},
                           {}};
  Intrinsic expected_groupby{"fedsql_group_by", {}, {}, {}, {}};
  expected_groupby.nested_intrinsics.push_back(std::move(expected_sum));
  expected_groupby.nested_intrinsics.push_back(std::move(expected_max));
  Intrinsic expected_outer{"outer_intrinsic", {}, {}, {}, {}};
  expected_outer.nested_intrinsics.push_back(std::move(expected_other));
  expected_outer.nested_intrinsics.push_back(std::move(expected_groupby));
  ASSERT_THAT(parsed_intrinsics.value()[0],
              EqIntrinsic(std::move(expected_outer)));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
