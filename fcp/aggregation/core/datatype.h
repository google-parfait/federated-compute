/*
 * Copyright 2022 Google LLC
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

#ifndef FCP_AGGREGATION_CORE_DATATYPE_H_
#define FCP_AGGREGATION_CORE_DATATYPE_H_

#include <cstdint>

#include "fcp/base/monitoring.h"

#ifndef FCP_NANOLIBC
#include "absl/strings/string_view.h"
#include "fcp/aggregation/core/tensor.pb.h"
#endif

namespace fcp {
namespace aggregation {

#ifndef FCP_NANOLIBC
// Unless when building with Nanolibc, we can use absl::string_view directly.
using string_view = absl::string_view;
#else
// TODO(team): Minimal implementation of string_view for bare-metal
// environment.
struct string_view {};
#endif

#ifdef FCP_NANOLIBC
// TODO(team): Derive these values from tensor.proto built with Nanopb
enum DataType {
  // The constants below should be kept in sync with tensorflow::Datatype:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
  // While not strictly required, that has a number of benefits.
  DT_INVALID = 0,
  DT_FLOAT = 1,
  DT_DOUBLE = 2,
  DT_INT32 = 3,
  DT_STRING = 7,
  DT_INT64 = 9,

  // TODO(team): Add other types.
  // This should be a small subset of tensorflow::DataType types and include
  // only simple numeric types and floating point types.
  //
  // When a tensor DT_ type is added here, it must also be added to the list of
  // MATCH_TYPE_AND_DTYPE macros below and to the CASES macro.
};
#endif  // FCP_NANOLIBC

namespace internal {

// This struct is used to map typename T to DataType and specify other traits
// of typename T.
template <typename T>
struct TypeTraits {
  constexpr static DataType kDataType = DT_INVALID;
};

#define MATCH_TYPE_AND_DTYPE(TYPE, DTYPE)        \
  template <>                                    \
  struct TypeTraits<TYPE> {                      \
    constexpr static DataType kDataType = DTYPE; \
  }

// Mapping of native types to DT_ types.
// TODO(team): Add other types.
MATCH_TYPE_AND_DTYPE(float, DT_FLOAT);
MATCH_TYPE_AND_DTYPE(double, DT_DOUBLE);
MATCH_TYPE_AND_DTYPE(int32_t, DT_INT32);
MATCH_TYPE_AND_DTYPE(int64_t, DT_INT64);
MATCH_TYPE_AND_DTYPE(string_view, DT_STRING);

// The macros DTYPE_CASE and DTYPE_CASES are used to translate Tensor DataType
// to strongly typed calls of code parameterized with the template typename
// TYPE_ARG.
//
// For example, let's say there is a function that takes an AggVector<T>:
// template <typename T>
// void DoSomething(AggVector<T> agg_vector) { ... }
//
// Given a Tensor, the following code can be used to make a DoSomething call:
// DTYPE_CASES(tensor.dtype(), T, DoSomething(tensor.AsAggVector<T>()));
//
// The second parameter specifies the type argument to be used as the template
// parameter in the statement in the third argument.

#define SINGLE_ARG(...) __VA_ARGS__
#define DTYPE_CASE(TYPE, TYPE_ARG, STMTS)       \
  case internal::TypeTraits<TYPE>::kDataType: { \
    typedef TYPE TYPE_ARG;                      \
    STMTS;                                      \
    break;                                      \
  }

// TODO(team): Add other types.
#define DTYPE_CASES(TYPE_ENUM, TYPE_ARG, STMTS)          \
  switch (TYPE_ENUM) {                                   \
    DTYPE_CASE(float, TYPE_ARG, SINGLE_ARG(STMTS))       \
    DTYPE_CASE(double, TYPE_ARG, SINGLE_ARG(STMTS))      \
    DTYPE_CASE(int32_t, TYPE_ARG, SINGLE_ARG(STMTS))     \
    DTYPE_CASE(int64_t, TYPE_ARG, SINGLE_ARG(STMTS))     \
    DTYPE_CASE(string_view, TYPE_ARG, SINGLE_ARG(STMTS)) \
    case DT_INVALID:                                     \
      FCP_LOG(FATAL) << "Invalid type";                  \
      break;                                             \
    default:                                             \
      FCP_LOG(FATAL) << "Unknown type";                  \
  }

}  // namespace internal

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_DATATYPE_H_
