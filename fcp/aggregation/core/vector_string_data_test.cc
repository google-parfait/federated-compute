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
#include "fcp/aggregation/core/vector_string_data.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

TEST(VectorDataTest, VectorStringDataValid) {
  VectorStringData vector_data(std::vector<std::string>(
      {"string1", "another-string", "one_more_string"}));
  EXPECT_THAT(vector_data.CheckValid(sizeof(string_view)), IsOk());
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
