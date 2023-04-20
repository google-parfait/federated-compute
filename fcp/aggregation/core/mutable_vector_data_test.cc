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
#include "fcp/aggregation/core/mutable_vector_data.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

TEST(MutableVectorDataTest, MutableVectorDataValid) {
  MutableVectorData<int64_t> vector_data;
  vector_data.push_back(1);
  vector_data.push_back(2);
  vector_data.push_back(3);
  EXPECT_THAT(vector_data.CheckValid(sizeof(int64_t)), IsOk());
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
