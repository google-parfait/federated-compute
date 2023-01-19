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

#include "fcp/aggregation/core/tensor_shape.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace fcp {
namespace aggregation {
namespace {

using testing::ElementsAre;

TEST(TensorShapeTest, CreateFromInitializerList) {
  TensorShape shape({2, 3, 5});
  EXPECT_THAT(shape.dim_sizes(), ElementsAre(2, 3, 5));
}

TEST(TensorShapeTest, CreateFromIterator) {
  std::vector<size_t> dim_sizes = {4, 8, 3, 2};
  TensorShape shape(dim_sizes.begin(), dim_sizes.end());
  EXPECT_THAT(shape.dim_sizes(), ElementsAre(4, 8, 3, 2));
}

TEST(TensorShapeTest, NumElements) {
  TensorShape shape({2, 3, 5});
  EXPECT_EQ(shape.NumElements(), 30);
}

TEST(TensorShapeTest, ScalarShape) {
  TensorShape shape({});
  EXPECT_EQ(shape.dim_sizes().size(), 0);
  EXPECT_EQ(shape.NumElements(), 1);
}

TEST(TensorShapeTest, EqualityOperators) {
  TensorShape shape({3, 5});
  EXPECT_EQ(shape, TensorShape({3, 5}));
  EXPECT_NE(shape, TensorShape({}));
  EXPECT_NE(shape, TensorShape({1}));
  EXPECT_NE(shape, TensorShape({3, 4}));
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
