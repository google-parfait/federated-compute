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

// Overview
// ========

/**
 * For a statically known type T, one can obtain a representation via
 * 'TypeOf<T>()'. This returns a 'TypeRep' (note that T has been _erased_). The
 * primary utility of a runtime type representation stems from well-behaved
 * equality: given some data, related to a type with representation 'r',
 * learning that 'r == TypeOf<T>()' means that it is valid to 'cast' that data
 * to whatever is related to T; T is then _un-erased_.
 *
 * TypeRep Traits
 * --------------
 *
 * It can be convenient for TypeOf<T>() to provide data beyond just a unique
 * identity - for example, a human-readable name (available as TypeRep::name()).
 * User-defined types can provide a human-readable name like so:
 *
 *   struct X {
 *     struct TypeRepInfo {
 *       static constexpr char const* name() { return "X"; }
 *     };
 *   };
 *
 * Or, alternatively:
 *
 *   struct X {
 *     TYPE_REP_INFO(X);
 *   };
 *
 *
 * Equality and linking assumptions
 * --------------------------------
 *
 * Checking equality of TypeReps is just a pointer comparison (and so an obvious
 * absl hashing implementation is provided too). It's quite reasonable to use
 * TypeReps as collection keys.
 *
 * This requires that - for each distinct T - TypeRepHolder<T>::kTypeRepData
 * has a single address across all translation units. That's not an exotic
 * requirement; it is a consequence of the 'one definition rule'. That rule
 * still applies even with templates, even despite repeated instantiations in
 * different translation units. Note that this same approach is used by the
 * absl::any implementation (see FastTypeId()).
 *
 * In terms of ELF objects, template-related symbols are marked 'weak' and so
 * simply merge together at link-time rather than conflict; this applies for
 * both static and dynamic linking.
 *
 * However, complications arise with MSVC on Windows. Symbols supposed to be
 * export or imported from a DLL must be marked as such with __declspec. This is
 * not easily compatible with C++'s rules around templates, and templated types
 * crossing DLL boundaries is known to be a tricky business.
 *
 * **Limitation**:
 *
 * Consequently, passing TypeReps between DLLs will *not* behaves as desired
 * (absl::any has the same problem). We currently suppose that it is unlikely
 * for that to be a problem.
 */

#ifndef FCP_BASE_TYPE_H_
#define FCP_BASE_TYPE_H_

#include <stddef.h>
#include <stdint.h>
#include <array>
#include <limits>
#include <string>
#include <vector>
#include "absl/meta/type_traits.h"

namespace fcp {

using float32_t = float;
static_assert(sizeof(float32_t) == 4, "Expected to use 'float' for float32_t");
using float64_t = double;
static_assert(sizeof(float64_t) == 8, "Expected to use 'double' for float64_t");

namespace type_internal {

/**
 * Internal representation of a type, which can be a constexpr.
 *
 * Implementation note: observe that the set of 'T's for TypeOf<T>() are known
 * at compile time. So, we can make constexpr TypeReps, that points to
 * TypeRepData in the .rodata section (like a std::string literal).
 */
struct TypeRepData {
  constexpr explicit TypeRepData(char const* name) : name(name) {}

  // Optional pointer to a c-std::string containing a name. May be nullptr.
  char const* name;
};

}  // namespace type_internal

/**
 * Runtime representation of a type. Defines equality, and is hashable.
 */
class TypeRep {
 public:
  explicit constexpr TypeRep(type_internal::TypeRepData const* data)
      : data_(data) {}

  /**
   * Returns a human-readable name of the type if available; nullptr otherwise.
   */
  constexpr char const* name() const {
    return data_->name;
  }

  std::string ToString() const;

  bool operator==(TypeRep const& other) const {
    return data_ == other.data_;
  }

 private:
  template <typename H>
  friend H AbslHashValue(H, TypeRep);
  type_internal::TypeRepData const* data_;
};

template <typename H>
H AbslHashValue(H h, TypeRep t) {
  return H::combine(std::move(h), t.data_);
}

/**
 * Function returning a TypeRep of a type T.
 *
 * It holds: TypeOf<T>() == TypeOf<U>() iff T == U.
 */
template <typename T>
constexpr TypeRep TypeOf();

namespace type_internal {

template <typename T>
struct PrimitiveTypeRepTraits {};

#define DECLARE_PRIM_TYPE(cpp)                            \
  template <>                                             \
  struct PrimitiveTypeRepTraits<cpp> {                    \
    static constexpr bool present() { return true; }      \
    static constexpr char const* name() { return #cpp; } \
  };

DECLARE_PRIM_TYPE(void);
DECLARE_PRIM_TYPE(bool);
DECLARE_PRIM_TYPE(int32_t);
DECLARE_PRIM_TYPE(int64_t);
DECLARE_PRIM_TYPE(float32_t);
DECLARE_PRIM_TYPE(float64_t);
#undef DECLARE_PRIM_TYPE

// Default traits
template <typename T, typename Enable = void>
struct TypeRepTraits {
  static constexpr char const* name() { return nullptr; }
};

// User-defined types can have 'struct TypeRepInfo { .. }' inside to specify a
// name, etc.
template <typename T>
struct TypeRepTraits<T, absl::void_t<typename T::TypeRepInfo>> {
  static constexpr char const* name() { return T::TypeRepInfo::name(); }
};

// Traits for types we don't control, such as primitives.
template <typename T>
struct TypeRepTraits<T,
                     std::enable_if_t<PrimitiveTypeRepTraits<T>::present()>> {
  static constexpr char const* name() {
    return PrimitiveTypeRepTraits<T>::name();
  }
};

template <typename T>
struct TypeRepHolder {
  static constexpr TypeRepData kTypeRepData{TypeRepTraits<T>::name()};
};

// Definition (above is a declaration).
template <typename T>
constexpr TypeRepData TypeRepHolder<T>::kTypeRepData;

}  // namespace type_internal

template <typename T>
constexpr TypeRep TypeOf() {
  return TypeRep{&type_internal::TypeRepHolder<T>::kTypeRepData};
}

#define TYPE_REP_INFO(name_)                               \
  struct TypeRepInfo {                                     \
    static constexpr char const* name() { return #name_; } \
  }

}  // namespace fcp

#endif  // FCP_BASE_TYPE_H_
