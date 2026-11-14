/*
 * Copyright 2023 Google LLC
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

#include "fcp/aggregation/group_by/composite_key_combiner.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "fcp/aggregation/core/agg_vector.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/input_tensor_list.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;

TEST(CompositeKeyCombinerTest, EmptyInput_Invalid) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT});
  absl::StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({}));
  ASSERT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeKeyCombinerTest, InputWithWrongShapeTensor_Invalid) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_INT32});
  Tensor t1({1.1f, 1.2f, 1.3f});
  Tensor t2({1, 2, 3, 4});
  absl::StatusOr<Tensor> result =
      combiner.Accumulate(InputTensorList({&t1, &t2}));
  ASSERT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeKeyCombinerTest, InputWithTooFewTensorsInvalid) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_INT32});
  Tensor t1({1.1f, 1.2f, 1.3f});
  absl::StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({&t1}));
  ASSERT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeKeyCombinerTest, InputWithTooManyTensors_Invalid) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_INT32});
  Tensor t1({1.1f, 1.2f, 1.3f});
  Tensor t2({1, 2, 3});
  Tensor t3({4, 5, 6});
  absl::StatusOr<Tensor> result =
      combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeKeyCombinerTest, InputWithWrongTypes_Invalid) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_STRING});
  Tensor t1({1.1f, 1.2f, 1.3f});
  Tensor t2({1, 2, 3});
  absl::StatusOr<Tensor> result =
      combiner.Accumulate(InputTensorList({&t1, &t2}));
  ASSERT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeKeyCombinerTest, OutputBeforeAccumulateOutputsEmptyTensor) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT});
  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(1));
  EXPECT_THAT(output[0], IsTensor<float>({}));
}

TEST(CompositeKeyCombinerTest, AccumulateAndOutput_SingleElement) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT});
  Tensor t1({1.3f});
  EXPECT_THAT(combiner.Accumulate(InputTensorList({&t1})),
              IsOkAndHolds(IsTensor<int64_t>({0})));
  EXPECT_THAT(combiner.GetOutputKeys(), ElementsAre(IsTensor<float>({1.3})));
}

TEST(CompositeKeyCombinerTest, AccumulateAndOutput_NumericTypes) {
  CompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_INT32, DT_INT64});
  Tensor t1({1.1f, 1.2f, 1.3f});
  Tensor t2({1, 2, 3});
  Tensor t3({INT64_C(4), INT64_C(5), INT64_C(6)});
  EXPECT_THAT(combiner.Accumulate(InputTensorList({&t1, &t2, &t3})),
              IsOkAndHolds(IsTensor<int64_t>({0, 1, 2})));
  EXPECT_THAT(
      combiner.GetOutputKeys(),
      ElementsAre(IsTensor<float>({1.1, 1.2, 1.3}),
                  IsTensor<int32_t>({1, 2, 3}), IsTensor<int64_t>({4, 5, 6})));
}

TEST(CompositeKeyCombinerTest,
     NumericTypes_SameKeysResultInSameOrdinalsAcrossAccumulateCalls) {
  CompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_INT32, DT_INT64});
  Tensor t1({1.1f, 1.2f, 1.1f, 1.2f});
  Tensor t2({1, 2, 3, 2});
  Tensor t3({INT64_C(4), INT64_C(5), INT64_C(6), INT64_C(5)});
  ASSERT_THAT(combiner.Accumulate(InputTensorList({&t1, &t2, &t3})),
              IsOkAndHolds(IsTensor<int64_t>({0, 1, 2, 1})));

  // Across different calls to Accumulate, tensors can have different shape.
  Tensor t4({1.2f, 1.1f, 1.1f, 1.1f, 1.2f});
  Tensor t5({2, 3, 2, 3, 2});
  Tensor t6({INT64_C(5), INT64_C(6), INT64_C(5), INT64_C(6), INT64_C(5)});
  ASSERT_THAT(combiner.Accumulate(InputTensorList({&t4, &t5, &t6})),
              IsOkAndHolds(IsTensor<int64_t>({1, 2, 3, 2, 1})));

  EXPECT_THAT(combiner.GetOutputKeys(),
              ElementsAre(IsTensor<float>({1.1, 1.2, 1.1, 1.1}),
                          IsTensor<int32_t>({1, 2, 3, 2}),
                          IsTensor<int64_t>({4, 5, 6, 5})));
}

TEST(CompositeKeyCombinerTest, AccumulateAndOutput_StringTypes) {
  CompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_STRING, DT_STRING});
  Tensor t1({1.1f, 1.2f, 1.3f});
  Tensor t2({"abc", "de", ""});
  Tensor t3({"fghi", "jklmn", "o"});
  EXPECT_THAT(combiner.Accumulate(InputTensorList({&t1, &t2, &t3})),
              IsOkAndHolds(IsTensor<int64_t>({0, 1, 2})));

  EXPECT_THAT(combiner.GetOutputKeys(),
              ElementsAre(IsTensor<float>({1.1, 1.2, 1.3}),
                          IsTensor<string_view>({"abc", "de", ""}),
                          IsTensor<string_view>({"fghi", "jklmn", "o"})));
}

TEST(CompositeKeyCombinerTest,
     StringTypes_SameCompositeKeysResultInSameOrdinalsAcrossAccumulateCalls) {
  CompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_STRING, DT_STRING});
  Tensor t1({1.1f, 1.2f, 1.2f, 1.3f});
  Tensor t2({"abc", "de", "de", ""});
  Tensor t3({"fghi", "jklmn", "jklmn", "o"});
  ASSERT_THAT(combiner.Accumulate(InputTensorList({&t1, &t2, &t3})),
              IsOkAndHolds(IsTensor<int64_t>({0, 1, 1, 2})));

  // Across different calls to Accumulate, tensors can have different shape.
  Tensor t4({1.3f, 1.4f, 1.1f, 1.2f, 1.1f});
  Tensor t5({"", "abc", "abc", "de", "abc"});
  Tensor t6({"o", "pqrs", "fghi", "jklmn", "fghi"});
  ASSERT_THAT(combiner.Accumulate(InputTensorList({&t4, &t5, &t6})),
              IsOkAndHolds(IsTensor<int64_t>({2, 3, 0, 1, 0})));

  EXPECT_THAT(
      combiner.GetOutputKeys(),
      ElementsAre(IsTensor<float>({1.1, 1.2, 1.3, 1.4}),
                  IsTensor<string_view>({"abc", "de", "", "abc"}),
                  IsTensor<string_view>({"fghi", "jklmn", "o", "pqrs"})));
}

TEST(CompositeKeyCombinerTest, LargeScale) {
  CompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_INT64, DT_STRING});
  constexpr size_t kNumElements = 1000 * 1000;
  std::vector<float> float_data;
  std::vector<int64_t> int_data;
  std::vector<std::string> string_data;
  std::vector<string_view> string_view_data;

  std::vector<int64_t> expected_ordinals;

  for (int i = 0; i < kNumElements; ++i) {
    expected_ordinals.push_back(i);
    float_data.push_back(static_cast<float>(i));
    int_data.push_back(i % 2);
    string_data.push_back(absl::StrCat("abcdefghiklmnop", i % 100));
  }

  for (int i = 0; i < kNumElements; ++i) {
    string_view_data.push_back(string_data[i]);
  }

  Tensor t1(std::vector<float>(float_data.begin(), float_data.end()));
  Tensor t2(std::vector<int64_t>(int_data.begin(), int_data.end()));
  Tensor t3(std::vector<std::string>(string_data.begin(), string_data.end()));

  // Accumulate the same tensors twice.
  absl::StatusOr<Tensor> result =
      combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_THAT(result, IsOk());
  ASSERT_THAT(result.value(), IsTensor({kNumElements}, expected_ordinals));
  result = combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_THAT(result, IsOk());
  ASSERT_THAT(result.value(), IsTensor({kNumElements}, expected_ordinals));

  ASSERT_THAT(
      combiner.GetOutputKeys(),
      ElementsAre(IsTensor<float>({kNumElements}, float_data),
                  IsTensor<int64_t>({kNumElements}, int_data),
                  IsTensor<string_view>({kNumElements}, string_view_data)));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
