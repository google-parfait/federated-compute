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

#include "fcp/base/proxy.h"

#include <vector>

#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

// Let's make sure proxy declarations work in namespaces other than ::fcp.
namespace some_ns {

class Q {
 public:
  virtual ~Q() = default;
  BEGIN_PROXY_DECLARATIONS(Q);
  virtual void Foo() = 0;
  DECLARE_PROXIED_FN(0, Foo);
  virtual bool Bar(int) = 0;
  DECLARE_PROXIED_FN(1, Bar);
};

DECLARE_PROXY_META(Q,
  PROXIED_FN(Q, Foo),
  PROXIED_FN(Q, Bar)
);

}  // namespace some_ns

namespace fcp {

using ::some_ns::Q;
using ::testing::Eq;
using ::testing::SizeIs;
using ::testing::ElementsAre;

static_assert(PROXIED_FN(Q, Foo)::Ordinal() == 0, "Declaration is broken");
static_assert(PROXIED_FN(Q, Foo)::InfoType::Ordinal() == 0,
              "Declaration is broken");
static_assert(PROXIED_FN(Q, Bar)::Ordinal() == 1, "Declaration is broken");
static_assert(PROXIED_FN(Q, Bar)::InfoType::Ordinal() == 1,
              "Declaration is broken");

static_assert(
    proxy_internal::CheckOrdinals<PROXIED_FN(Q, Foo), PROXIED_FN(Q, Bar)>(),
    "Ordinals are good");
static_assert(
    !proxy_internal::CheckOrdinals<PROXIED_FN(Q, Bar), PROXIED_FN(Q, Foo)>(),
    "Wrong relative order");
static_assert(!proxy_internal::CheckOrdinals<PROXIED_FN(Q, Bar)>(),
              "Indexed from 0, but Bar has ordinal 1");

class Q_Impl : public Q {
 public:
  void Foo() override { foo_count_++; };
  bool Bar(int i) override { return i == 42; };
  int foo_count() const { return foo_count_; }

 private:
  int foo_count_ = 0;
};

class TracingProxy : public ProxyMeta<Q>::ProxyBase<TracingProxy> {
 public:
  explicit TracingProxy(Q* target) : target_(target) {}

  std::vector<int> const& calls() const { return calls_; }

  template <typename Fn, typename R, typename... A>
  static R Call(TracingProxy* self, A... args) {
    auto ptr = Fn::Member();
    self->calls_.push_back(Fn::Ordinal());
    return (self->target_->*ptr)(args...);
  }
 private:
  Q* target_;
  std::vector<int> calls_;
};

TEST(ProxyTest, TracingProxy) {
  constexpr int kFoo = PROXIED_FN(Q, Foo)::Ordinal();
  constexpr int kBar = PROXIED_FN(Q, Bar)::Ordinal();

  Q_Impl impl{};
  TracingProxy proxy{&impl};

  EXPECT_THAT(proxy.calls(), SizeIs(0));
  EXPECT_THAT(impl.foo_count(), Eq(0));

  proxy.Foo();
  EXPECT_THAT(proxy.Bar(1), Eq(false));
  EXPECT_THAT(proxy.Bar(42), Eq(true));
  EXPECT_THAT(proxy.Bar(2), Eq(false));
  proxy.Foo();

  EXPECT_THAT(impl.foo_count(), Eq(2));
  EXPECT_THAT(proxy.calls(), ElementsAre(kFoo, kBar, kBar, kBar, kFoo));
}

using ReflectionEntry = std::pair<absl::string_view, int>;

class ReflectionCollector {
 public:
  using ResultType = ReflectionEntry;

  template <typename Fn>
  ResultType ReflectProxyFunction() {
    return {Fn::Name(), Fn::Ordinal()};
  }
};

TEST(ProxyTest, Reflection) {
  auto entries = ProxyMeta<Q>::Reflect(ReflectionCollector{});
  // Note that the returned array is supposed to be indexed by ordinal.
  EXPECT_THAT(entries, ElementsAre(ReflectionEntry{"Foo", 0},
                                   ReflectionEntry{"Bar", 1}));
}

TEST(ProxyTest, Invoker) {
  using FooInvoker = PROXIED_FN(Q, Foo)::Invoker;
  using BarInvoker = PROXIED_FN(Q, Bar)::Invoker;

  Q_Impl impl{};
  auto foo = FooInvoker{&impl};
  auto bar = BarInvoker{&impl};

  EXPECT_THAT(impl.foo_count(), Eq(0));
  foo();
  EXPECT_THAT(impl.foo_count(), Eq(1));

  EXPECT_THAT(bar(1), Eq(false));
  EXPECT_THAT(bar(42), Eq(true));
}

}  // namespace fcp
