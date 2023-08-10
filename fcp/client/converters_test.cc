#include "fcp/client/converters.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/testing/testing.h"
#include "fcp/client/example_query_result.pb.h"

namespace fcp::client {
namespace {

TEST(ConvertersTest, ConvertNumericTensor_Success) {
  ExampleQueryResult::VectorData::Values float_values;
  float_values.mutable_float_values()->add_value(0.24f);
  float_values.mutable_float_values()->add_value(0.42f);
  float_values.mutable_float_values()->add_value(0.33f);

  EXPECT_THAT(*ConvertNumericTensor<float>(aggregation::DT_FLOAT,
                                           aggregation::TensorShape({3}),
                                           float_values.float_values().value()),
              aggregation::IsTensor<float>({3}, {0.24f, 0.42f, 0.33f}));
}

TEST(ConvertersTest, ConvertStringTensor_Success) {
  ExampleQueryResult::VectorData::Values string_values;
  string_values.mutable_string_values()->add_value("string_value1");
  string_values.mutable_string_values()->add_value("string_value2");

  EXPECT_THAT(*ConvertStringTensor(aggregation::TensorShape({2}),
                                   string_values.string_values().value()),
              aggregation::IsTensor<absl::string_view>(
                  {2}, {"string_value1", "string_value2"}));
}
}  // namespace
}  // namespace fcp::client
