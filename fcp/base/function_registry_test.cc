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

#include "fcp/base/function_registry.h"

#include <functional>
#include <memory>

#include "gtest/gtest.h"
#include "absl/strings/string_view.h"

namespace fcp {
namespace {

const absl::string_view kKey = "key";

TEST(FunctionRegistryTest, RegisterSucceeds) {
  FunctionRegistry<absl::string_view, int()> registry;
  EXPECT_TRUE(registry.Register("foo", [] { return 1; }));
  RegisterOrDie(registry, "bar", [] { return 2; });
}

TEST(FunctionRegistryTest, GetSucceeds) {
  // Use a move-only function argument to test forwarding.
  FunctionRegistry<absl::string_view, int(std::unique_ptr<int>)>  // NOLINT
      registry;
  ASSERT_TRUE(
      registry.Register(kKey, [](std::unique_ptr<int> x) { return *x; }));

  auto function = registry.Get(kKey);
  ASSERT_TRUE(function);
  EXPECT_EQ(1, function(std::make_unique<int>(1)));
}

TEST(FunctionRegistryTest, GetFails) {
  FunctionRegistry<absl::string_view, int()> registry;
  auto function = registry.Get(kKey);
  EXPECT_FALSE(function);
}

}  // namespace
}  // namespace fcp
