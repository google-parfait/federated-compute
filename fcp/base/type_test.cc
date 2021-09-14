/*
 * Copyright 2018 Google LLC
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

#include "fcp/base/type.h"

#include "absl/hash/hash_testing.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace fcp {

using ::testing::Eq;
using ::testing::Not;

struct UserType {
  struct TypeRepInfo {
    static constexpr char const* name() { return "UserType"; }
  };
};

TEST(TypeTest, StringifyPrim) {
  EXPECT_THAT(TypeOf<void>().ToString(), Eq("void"));
  EXPECT_THAT(TypeOf<int32_t>().ToString(), Eq("int32_t"));
  EXPECT_THAT(TypeOf<int64_t>().ToString(), Eq("int64_t"));
  EXPECT_THAT(TypeOf<float32_t>().ToString(), Eq("float32_t"));
  EXPECT_THAT(TypeOf<float64_t>().ToString(), Eq("float64_t"));
  EXPECT_THAT(TypeOf<bool>().ToString(), Eq("bool"));
}

TEST(TypeTest, StringifyFunction) {
  EXPECT_THAT(TypeOf<void(int32_t)>().ToString(), Eq("<unnamed type>"));
}

TEST(TypeTest, StringifyUserType) {
  EXPECT_THAT(TypeOf<UserType>().ToString(), "UserType");
}

TEST(TypeTest, Equality) {
  EXPECT_THAT(TypeOf<bool>(), Eq(TypeOf<bool>()));
  EXPECT_THAT(TypeOf<void()>(), Eq(TypeOf<void()>()));
  EXPECT_THAT(TypeOf<int32_t(bool)>(), Eq(TypeOf<int32_t(bool)>()));
  EXPECT_THAT(TypeOf<UserType>(), Eq(TypeOf<UserType>()));
}

template <typename T>
::testing::Matcher<T> NotEq(T val) {
  return Not(Eq(val));
}

TEST(TypeTest, Inequality) {
  EXPECT_THAT(TypeOf<bool>(), NotEq(TypeOf<void>()));
  EXPECT_THAT(TypeOf<bool>(), NotEq(TypeOf<bool()>()));
  EXPECT_THAT(TypeOf<void(int32_t)>(), NotEq(TypeOf<void(bool)>()));
}

constexpr TypeRep kTypeOfInt32 = TypeOf<int32_t>();

TEST(TypeTest, Constexpr) {
  EXPECT_THAT(kTypeOfInt32.ToString(), Eq("int32_t"));
  EXPECT_THAT(kTypeOfInt32, Eq(TypeOf<int32_t>()));
}

TEST(TypeTest, Hashing) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
    TypeOf<int32_t>(),
    TypeOf<std::vector<int32_t>>(),
    TypeOf<void(int32_t)>(),
    TypeOf<int32_t()>(),
    TypeOf<UserType>()
  }));
}

}  // namespace fcp
