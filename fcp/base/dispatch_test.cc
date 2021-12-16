/*
 * Copyright 2019 Google LLC
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

#include "fcp/base/dispatch.h"

#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace fcp {

using ::testing::Eq;
using ::testing::VariantWith;

using ExampleGeneric = std::variant<Unit, int, bool>;

struct ExampleGenericRepr {
  using GenericType = ExampleGeneric;

  template <typename T>
  static constexpr T Unwrap(GenericType val) {
    return absl::get<T>(val);
  }

  template <typename T>
  static constexpr GenericType Wrap(T val) {
    return ExampleGeneric{val};
  }
};

class Q {
 public:
  virtual ~Q() = default;
  BEGIN_PROXY_DECLARATIONS(Q);
  virtual bool Check(int i) = 0;
  DECLARE_PROXIED_FN(0, Check);
  virtual int Subtract(int a, int b) = 0;
  DECLARE_PROXIED_FN(1, Subtract);
  virtual void Set(int new_value) = 0;
  DECLARE_PROXIED_FN(2, Set);
};

DECLARE_PROXY_META(Q,
  PROXIED_FN(Q, Check),
  PROXIED_FN(Q, Subtract),
  PROXIED_FN(Q, Set)
);

constexpr int kCheck = PROXIED_FN(Q, Check)::Ordinal();
constexpr int kSubtract = PROXIED_FN(Q, Subtract)::Ordinal();
constexpr int kSet = PROXIED_FN(Q, Set)::Ordinal();

class Q_Impl : public Q {
 public:
  explicit Q_Impl(int value) : value_(value) {}

  bool Check(int i) override { return i == value_; };
  int Subtract(int a, int b) override { return a - b; }
  void Set(int new_value) override { value_ = new_value; }

  int value_;
};

std::shared_ptr<Q_Impl> MakeImpl(int val) {
  return std::make_shared<Q_Impl>(val);
}

// Note that we take a shared_ptr<Q>, not shared_ptr<Q_Impl>. Currently, we must
// instantiate DispatchPtr with an interface type.
DispatchPtr<ExampleGenericRepr> MakeDispatchPtr(std::shared_ptr<Q> interface) {
  return MakeDispatchPtr<ExampleGenericRepr>(std::move(interface));
}

TEST(DispatchTest, OneArg) {
  std::shared_ptr<Q_Impl> impl = MakeImpl(42);
  DispatchPtr<ExampleGenericRepr> d = MakeDispatchPtr(impl);

  EXPECT_THAT(d.Dispatch(kCheck, {ExampleGeneric{42}}),
              VariantWith<bool>(true));
  EXPECT_THAT(d.Dispatch(kCheck, {ExampleGeneric{123}}),
              VariantWith<bool>(false));
  // Does the 'this' pointer get set correctly?
  impl->value_ = 123;
  EXPECT_THAT(d.Dispatch(kCheck, {ExampleGeneric{123}}),
              VariantWith<bool>(true));
}

TEST(DispatchTest, TwoArgs) {
  std::shared_ptr<Q_Impl> impl = MakeImpl(0);
  DispatchPtr<ExampleGenericRepr> d = MakeDispatchPtr(impl);

  EXPECT_THAT(d.Dispatch(kSubtract, {ExampleGeneric{42}, ExampleGeneric{1}}),
              VariantWith<int>(41));
  EXPECT_THAT(d.Dispatch(kSubtract, {ExampleGeneric{1}, ExampleGeneric{100}}),
              VariantWith<int>(-99));
}

TEST(DispatchTest, ReturnVoid) {
  std::shared_ptr<Q_Impl> impl = MakeImpl(0);
  DispatchPtr<ExampleGenericRepr> d = MakeDispatchPtr(impl);
  EXPECT_THAT(impl->value_, Eq(0));

  EXPECT_THAT(d.Dispatch(kSet, {ExampleGeneric{42}}),
              VariantWith<Unit>(Unit{}));
  EXPECT_THAT(impl->value_, Eq(42));
  EXPECT_THAT(d.Dispatch(kSet, {ExampleGeneric{123}}),
              VariantWith<Unit>(Unit{}));
  EXPECT_THAT(impl->value_, Eq(123));
}

}  // namespace fcp
