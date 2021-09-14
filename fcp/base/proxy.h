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

/**
 * Macros and magic for generating 'proxy' types.
 *
 * Consider some struct/class-type 'S' with member functions (both virtual and
 * non-virtual are fine). We define a _proxy type_ P for S to be one such that:
 *
 *  1) All member functions of S are also present on P, with the same type (but
 *     note the one on P is never virtual).
 *  2) Calls to those 'proxied' functions on an instance of P are all
 *     implemented by a single static _template_ function on P:
 *
 *       template <typename Fn, typename R, typename... A>
 *       static R Call(P* self, A... args)
 *
 * Example:
 *
 *    class S {
 *     public:
 *      virtual ~S() = default;
 *      BEGIN_PROXY_DECLARATIONS(S);
 *      virtual void Foo() = 0;
 *      DECLARE_PROXIED_FN(0, Foo);
 *      virtual bool Bar(int) = 0;
 *      DECLARE_PROXIED_FN(1, Bar);
 *    };
 *
 *    DECLARE_PROXY_META(S,
 *      PROXIED_FN(S, Foo),
 *      PROXIED_FN(S, Bar)
 *    );
 *
 * Adding proxy support to a type requires just the following boilerplate:
 *
 *   1) Precede the first proxied function with BEGIN_PROXY_DECLARATIONS(type)
 *   2) After each function to be proxied, write DECLARE_PROXIED_FN(ordinal,
 *      name). The ordinal *must* be unique across all proxied functions on the
 *      same type, and should be dense starting at 0.
 *   3) Call DECLARE_PROXY_META(type, fns...) where each of 'fns' is of the form
 *      PROXIED_FN(type, name).
 *
 * Each DECLARE_PROXIED_FN instance produces a distinct type (in the class
 * body), then accessible as PROXIED_FN(type, name). Each such type has the
 * following static members:
 *
 *   - const char* Name(): The function name
 *
 *   - int Ordinal(): The ordinal as provided to DECLARE_PROXIED_FN
 *
 *   - R (type::*)(A..) Member(): The member function pointer (usable to call
 *     the function given an instance of 'type').
 *
 *   - ProxyFunction<Self>: A class type with a single method - 'name' -
 *     implemented as so:
 *         return Self::template Call<Fn, R>(
 *             static_cast<Self*>(this), args...);
 *     (where R == PROXIED_FN(type, name), and 'Self' assumes CRTP)
 *
 *   - Invoker: Callable type which forwards to this function. Initialize it
 *     with a pointer to the target type. This is functionally redundant with
 *     Member(), but both gcc and clang have shown an inability to optimize
 *     calls via member function pointers (whereas this optimizes as expected).
 *
 * DECLARE_PROXY_META(type, fns...) defines ProxyMeta<type>, which aggregate the
 * PROXIED_FN types. In particular, ProxyMeta provides a type ProxyBase<Self>
 * (CRTP again), which simply inherits from each ProxyFunction<Self>.
 *
 * The following example defines a 'trivial' proxy type for S; that is, it
 * forwards 'Foo' and 'Bar' calls to some target S*, without doing anything
 * else:
 *
 *   class TrivialProxy : public ProxyMeta<S>::ProxyBase<TrivialProxy> {
 *    public:
 *     TrivialProxy(S* target) : target(target) {}
 *
 *     template <typename Fn, typename R, typename... A>
 *     static R Call(NoOpProxy* self, A... args) {
 *       auto ptr = Fn::Member();
 *       return (self->target->*ptr)(args...);
 *     }
 *
 *     S* target;
 *   };
 *
 * That definition is equivalent to the following:
 *
 *   class TrivialProxy {
 *    public:
 *     TrivialProxy(S* target) : target(target) {}
 *
 *     void Foo() {
 *       target->Foo();
 *     }
 *
 *     bool Bar(int i) {
 *       return target->Bar(i);
 *     }
 *
 *     S* target;
 *   };
 *
 * ProxyMeta also provides a function Reflect<R>(R r), which calls
 * r.ReflectProxyFunction<Fn>() for each PROXIED_FN  type. This can be used to
 * build reflection structures at runtime, such as e.g. a map from ordinal ->
 * member-pointer. ReflectProxyFunction must have the same return-type for each
 * instantiation, indicated as R::ResultType. It returns an array of
 * R::ResultType, indexed by function ordinal.
 */

#ifndef FCP_BASE_PROXY_H_
#define FCP_BASE_PROXY_H_

#include <array>
#include "fcp/base/meta.h"

namespace fcp {

namespace proxy_internal {

/**
 * Consider a member function pointer &S::F, where F has (function-)type T. To
 * proxy it, we need to define a type with one function F of type T. Naively
 * we'd like to write something like:
 *    Function<T>::Result F(Function<T>::Args... args)
 * That's unforunately not possible. Parameter-packs are not first-class; they
 * can only be brought into scope with template<>.
 *
 * We use ProxyFunctionTypeBinder to work around that. The function-type
 * specialization brings a parameter-pack into scope, which we can then apply to
 * a template for F.
 */
template<
    typename Self,
    typename FnType,
    template<typename Self_, typename R, typename... A> class Impl>
struct ProxyFunctionTypeBinder {};
template<
    typename Self,
    template<typename Self_, typename R, typename... A> class Impl,
    typename R, typename... A>
struct ProxyFunctionTypeBinder<Self, R(A...), Impl> {
    using Type = Impl<Self, R, A...>;
};

/** Like ProxyFunctionTypeBinder */
template<
    typename FnType,
    template<typename R, typename... A> class Impl>
struct InvokerTypeBinder {};
template<
    template<typename R, typename... A> class Impl,
    typename R, typename... A>
struct InvokerTypeBinder<R(A...), Impl> {
    using Type = Impl<R, A...>;
};

template <typename... FnInfo>
constexpr bool CheckOrdinalsImpl(
    size_t expected,
    typename absl::enable_if_t<sizeof...(FnInfo) == 0, int> = 0) {
  return true;
}

template <typename FnInfoHead, typename... FnInfo>
constexpr bool CheckOrdinalsImpl(size_t expected) {
  return FnInfoHead::Ordinal() == expected &&
         CheckOrdinalsImpl<FnInfo...>(expected + 1);
}

template <typename... FnInfo>
constexpr bool CheckOrdinals() {
  return CheckOrdinalsImpl<FnInfo...>(0);
}

/**
 * The implementation of ProxyMeta. ProxyMeta has one type parameter (the
 * proxied type), whereas this one is parameterized on the function-info types.
 * We only want to mention the FnInfo... pack once as part of
 * DECLARE_PROXY_META.
 */
template <typename... FnInfo>
struct ProxyMetaImpl {
  static_assert(CheckOrdinals<FnInfo...>(),
                "Function ordinals must match their indices in this list (i.e. "
                "0, 1, ...)");

  static constexpr size_t Size() { return sizeof...(FnInfo); }

  // This parameter pack expansion across the ProxyFunction<> types is
  // just multiple inheritance. The resulting type is efficient to use, on our
  // assumption that the base classes do not have vtables. If they did, we'd get
  // a vtable pointer *per base class*.
  //
  // See DECLARE_PROXIED_FN below for how the ProxyFunction types are declared.
  template <typename Self>
  struct ProxyBase : public FnInfo::template ProxyFunction<Self>... {};

  template <typename Reflector>
  static constexpr std::array<typename Reflector::ResultType, Size()> Reflect(
      Reflector r) {
    return {r.template ReflectProxyFunction<FnInfo>()...};
  }
};

}  //  namespace proxy_internal

/**
 * Must appear before the first DECLARE_PROXIED_FN in a class body.
 */
#define BEGIN_PROXY_DECLARATIONS(type) \
  using ProxyTarget = type;            \
  template <typename T>                \
  struct ProxyFunctionInfo_ {};

#define DECLARE_PROXIED_FN(ordinal, name)                                      \
  template <>                                                                  \
  struct ProxyFunctionInfo_<LIFT_MEMBER_TO_TYPE(ProxyTarget, name)> {          \
    using PtrType = decltype(&ProxyTarget::name);                              \
    using FnType = ::fcp::MemberPointerTraits<PtrType>::TargetType;            \
    using InfoType =                                                           \
        ProxyFunctionInfo_<LIFT_MEMBER_TO_TYPE(ProxyTarget, name)>;            \
    static constexpr int Ordinal() { return ordinal; }                         \
    static constexpr PtrType Member() { return &ProxyTarget::name; }           \
    static constexpr char const* Name() { return #name; }                      \
    template <typename Self, typename R, typename... A>                        \
    struct ProxyFunctionTmpl_ {                                                \
      R name(A... args) {                                                      \
        return Self::template Call<InfoType, R>(static_cast<Self*>(this),      \
                                                args...);                      \
      }                                                                        \
    };                                                                         \
    template <typename Self>                                                   \
    using ProxyFunction =                                                      \
        typename ::fcp::proxy_internal::ProxyFunctionTypeBinder<               \
            Self, FnType, ProxyFunctionTmpl_>::Type;                           \
    template <typename R, typename... A>                                       \
    struct InvokerTmpl_ {                                                      \
      ProxyTarget* target;                                                     \
      R operator()(A... args) const { return target->name(args...); }          \
    };                                                                         \
    using Invoker =                                                            \
        typename ::fcp::proxy_internal::InvokerTypeBinder<FnType,              \
                                                          InvokerTmpl_>::Type; \
  }

/**
 * Gives the proxy-function metadata type for a member previously annotated with
 * DECLARE_PROXIED_FN.
 *
 * Example:
 *   DECLARE_PROXIED_FN(1, X);
 *   ...
 *   PROXIED_FN(S, X)::Ordinal() => 1
 */
#define PROXIED_FN(type, name) \
  type::template ProxyFunctionInfo_<LIFT_MEMBER_TO_TYPE(type, name)>

/**
 * See file remarks for the meaning of ProxyMeta.
 *
 * We use MAKE_LINK / LinkedType (vs. template specialization) so that
 * DECLARE_PROXY_META can appear in the same namespace as the proxied type.
 */
template<typename T>
using ProxyMeta = LinkedType<T>;

/**
 * Declares ProxyMeta<type>. Each of the variable arguments must be of the form
 * PROXIED_FN(type, name).
 */
#define DECLARE_PROXY_META(type, ...)                                          \
  struct ProxyMeta_##type##_ {                                                 \
    using Impl = ::fcp::proxy_internal::ProxyMetaImpl<__VA_ARGS__>;            \
    static constexpr size_t Size() { return Impl::Size(); }                    \
    template <typename Self>                                                   \
    using ProxyBase = Impl::ProxyBase<Self>;                                   \
    template <typename Reflector>                                              \
    static constexpr auto Reflect(Reflector r) -> decltype(Impl::Reflect(r)) { \
      return Impl::Reflect(r);                                                 \
    }                                                                          \
  };                                                                           \
  MAKE_LINK(type, ProxyMeta_##type##_)

}  // namespace fcp

#endif  // FCP_BASE_PROXY_H_
