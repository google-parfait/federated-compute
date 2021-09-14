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
 * Method dispatch tables for proxy-annotated interfaces (as in proxy.h).
 *
 * The functionality in proxy.h allows one to define generic proxy types for
 * interfaces. A proxy type provides one function, instantiated for each of the
 * variously-typed interface methods (specific => generic):
 *
 *       template <typename Fn, typename R, typename... A>
 *       static R Call(P* self, A... args)
 *
 * Dual to that (generic => specific) is dispatching method calls: given a
 * method ordinal (visible as Fn::Ordinal() in Call() above) and generic
 * representation of arguments, make a (virtual-)call to the interface
 * implementation. That is provided here.
 *
 * We approach this in two parts:
 *
 *   - Construction of 'dtables' (dispatch tables - so named, in relation to
 *     vtables), for various proxied types. We do so using the reflection
 *     facility on ProxyMeta<>, as a constexpr. Just like the compiler-owned
 *     (and hidden) vtables, a 'dtable pointer' refers to an array of function
 *     pointers in a read-only data section. Unlike vtables, a dtable's function
 *     pointers share a _single type_, corresponding to some chosen generic
 *     representation: they each point to a wrapper function that converts the
 *     arguments and return values.
 *
 *   - Dispatch using a dtable: Whereas vtable indices are kept secret by the
 *     compiler (one can only name virtual member functions), we index dtables
 *     by ordinal - i.e. PROXIED_FN(I, F)::Ordinal() for F on interface I. Since
 *     there's a single generic representation and function-pointer type,
 *     dispatch is independent of the target function's type (PROXIED_FN(I,
 *     F)::FnType).
 *
 * This all amounts to DispatchPtr<G>, for some 'generic representation' G. One
 * may be constructed given an std::shared_ptr to an interface; a dtable pointer
 * is captured based on the interface type, and the interface type is then
 * forgotten. It offers a Dispatch() method, taking an ordinal and
 * generically-represented arguments.
 */

#ifndef FCP_BASE_DISPATCH_H_
#define FCP_BASE_DISPATCH_H_

#include <memory>
#include "fcp/base/meta.h"
#include "fcp/base/proxy.h"

#include "absl/types/span.h"
#include "absl/utility/utility.h"

namespace fcp {

namespace dispatch_internal {

// See DispatchPtr for the contract on the GenericRepr parameters.

template<typename GenericRepr>
struct DispatchContext {
  using GenericType = typename GenericRepr::GenericType;
  using DispatchThunkType = GenericType(void*,
                                        absl::Span<GenericType const>);
};

/**
 * This adapts GenericRepr to the callable that CastContainerElements expects;
 * see DispatchThunkReflector.
 */
template<typename GenericRepr>
struct UnwrapCastOp {
  using GenericType = typename GenericRepr::GenericType;

  template<typename T>
  using TargetType = T;

  template <typename T>
  TargetType<T> Cast(GenericType val) const {
    return GenericRepr::template Unwrap<T>(val);
  }
};

/**
 * Used with ProxyMeta<>::Reflect. Reflects each proxied function to a dispatch
 * thunk that converts args (generic => specific), forwards the call, and
 * converts the return value (specific => generic).
 */
template<typename Target, typename GenericRepr>
struct DispatchThunkReflector {
  using ContextType = DispatchContext<GenericRepr>;
  using GenericType = typename ContextType::GenericType;
  using ResultType = typename ContextType::DispatchThunkType*;

  template <typename FnInfo>
  static GenericType DispatchThunk(
      void* target, absl::Span<typename ContextType::GenericType const> args) {
    using FnType = typename FnInfo::FnType;
    auto typed_target = static_cast<Target*>(target);
    auto typed_args =
        CastContainerElements(typename FunctionTraits<FnType>::ArgPackType{},
                              args, UnwrapCastOp<GenericRepr>{});
    auto invoker = LiftVoidReturn(typename FnInfo::Invoker{typed_target});
    auto r = absl::apply(std::move(invoker), std::move(typed_args));
    return GenericRepr::Wrap(std::move(r));
  }

  template <typename FnInfo>
  constexpr ResultType ReflectProxyFunction() const {
    return DispatchThunk<FnInfo>;
  }
};

/**
 * We want a single definition / symbol per dtable. Each is fully determined by
 * a Target (interface) type and a GenericRepr.
 */
template<typename Target, typename GenericRepr>
struct DTableHolder {
  using ContextType = DispatchContext<GenericRepr>;
  using DispatchThunkType = typename ContextType::DispatchThunkType;

  static constexpr size_t Size() { return ProxyMeta<Target>::Size(); }

  using DTableType = std::array<DispatchThunkType*, Size()>;
  static constexpr DTableType kDTable =
      ProxyMeta<Target>::Reflect(DispatchThunkReflector<Target, GenericRepr>{});
};

// Definition
template <typename Target, typename GenericRepr>
constexpr typename DTableHolder<Target, GenericRepr>::DTableType
    DTableHolder<Target, GenericRepr>::kDTable;

}  // namespace dispatch_internal

/**
 * A pointer to an object to which we can generically Dispatch() method calls by
 * ordinal; its interface type is erased. Wraps an std::shared_ptr. Use
 * MakeDispatchPtr() to create one.
 *
 * The GenericRepr parameter indicates a bi-directional conversion to be used
 * for arguments and return values. It should be shaped like so:
 *
 *    struct ExampleGenericRepr {
 *      using GenericType = ExampleGeneric;
 *
 *      template <typename T>
 *      static constexpr T Unwrap(GenericType val) {
 *        ...
 *      }
 *
 *      template <typename T>
 *      static constexpr GenericType Wrap(T val) {
 *        ...
 *      }
 *    };
 *
 * As a special case, note that a 'void' return type results in a call to
 * Wrap(Unit).
 *
 * We currently represent this as a pair of the shared_ptr and a Span for the
 * dtable - so this is rather large for a 'pointer'. If there's usually just one
 * DispatchPtr per object (like in the current use-case), that doesn't matter
 * much.
 */
template<typename GenericRepr>
class DispatchPtr {
 public:
  using ContextType = dispatch_internal::DispatchContext<GenericRepr>;
  using GenericType = typename ContextType::GenericType;

  GenericType Dispatch(uint32_t ordinal,
                       absl::Span<GenericType const> args) const {
    FCP_CHECK(ordinal < dtable_.size());
    auto thunk = dtable_[ordinal];
    return thunk(ptr_.get(), args);
  }

 private:
  template <typename GenericRepr_, typename Target_>
  friend DispatchPtr<GenericRepr_> MakeDispatchPtr(std::shared_ptr<Target_>);

  template <typename Target>
  explicit DispatchPtr(std::shared_ptr<Target> ptr)
      : ptr_(std::move(ptr)),
        dtable_(dispatch_internal::DTableHolder<Target, GenericRepr>::kDTable) {
  }

  std::shared_ptr<void> ptr_;
  absl::Span<typename ContextType::DispatchThunkType* const> dtable_;
};

/**
 * Creates a DispatchPtr. The given std::shared_ptr must specify an interface
 * type (*not* a type implementing it); the Target type can be specified
 * explicitly to induce a conversion.
 */
template<typename GenericRepr, typename Target>
DispatchPtr<GenericRepr> MakeDispatchPtr(std::shared_ptr<Target> ptr) {
  return DispatchPtr<GenericRepr>(std::move(ptr));
}

}  // namespace fcp

#endif  // FCP_BASE_DISPATCH_H_
