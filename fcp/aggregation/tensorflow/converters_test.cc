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

#include "fcp/aggregation/tensorflow/converters.h"

#include <initializer_list>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/base/monitoring.h"
#include "fcp/testing/testing.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace fcp::aggregation::tensorflow {
namespace {

namespace tf = ::tensorflow;

tf::TensorShape CreateTfShape(std::initializer_list<int64_t> dim_sizes) {
  tf::TensorShape shape;
  EXPECT_TRUE(tf::TensorShape::BuildTensorShape(dim_sizes, &shape).ok());
  return shape;
}

TEST(ConvertersTest, ConvertDataType_Success) {
  EXPECT_EQ(*ConvertDataType(tf::DT_FLOAT), DT_FLOAT);
  EXPECT_EQ(*ConvertDataType(tf::DT_DOUBLE), DT_DOUBLE);
  EXPECT_EQ(*ConvertDataType(tf::DT_INT32), DT_INT32);
  EXPECT_EQ(*ConvertDataType(tf::DT_INT64), DT_INT64);
}

TEST(ConvertersTest, ConvertDataType_Unsupported) {
  EXPECT_THAT(ConvertDataType(tf::DT_STRING), IsCode(INVALID_ARGUMENT));
}

TEST(ConvertersTest, ConvertShape_Success) {
  EXPECT_EQ(ConvertShape(CreateTfShape({})), TensorShape({}));
  EXPECT_EQ(ConvertShape(CreateTfShape({1})), TensorShape({1}));
  EXPECT_EQ(ConvertShape(CreateTfShape({2, 3})), TensorShape({2, 3}));
}

}  // namespace
}  // namespace fcp::aggregation::tensorflow
