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
#include "fcp/aggregation/core/mutable_string_data.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace aggregation {
namespace {

TEST(MutableStringDataTest, MutableStringDataValid) {
  MutableStringData string_data(0);
  EXPECT_THAT(string_data.CheckValid<string_view>(), IsOk());
}

TEST(MutableStringDataTest, ValidAfterAddingValue) {
  MutableStringData string_data(1);
  string_data.Add("added-string");
  EXPECT_THAT(string_data.CheckValid<string_view>(), IsOk());
}

}  // namespace
}  // namespace aggregation
}  // namespace fcp
