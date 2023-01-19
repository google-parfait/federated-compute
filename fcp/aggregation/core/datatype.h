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

namespace fcp {
namespace aggregation {

// A list of supported tensor value types.
enum DataType {
  // The constants below should be kept in sync with tensorflow::Datatype:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
  // While not strictly required, that has a number of benefits.
  DT_INVALID = 0,
  DT_FLOAT = 1,
  DT_DOUBLE = 2,
  DT_INT32 = 3,
  DT_INT64 = 9,

  // TODO(team): Add other types.
  // This should be a small subset of tensorflow::DataType types and include
  // only simple numeric types and floating point types.
  //
  // When a tensor DT_ type is added here, it must also be added to the list of
  // MATCH_TYPE_AND_DTYPE macros below and to the CASES macro.
};

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

// The macros CASE and CASES are used to translate Tensor DataType to strongly
// typed calls of code parameterized with the template typename T.
//
// For example, let's say there is a function that takes an AggVector<T>:
// template <typename T>
// void DoSomething(AggVector<T> agg_vector) { ... }
//
// Given a Tensor, the following code can be used to make a DoSomething call:
// CASES(tensor.dtype(), DoSomething(tensor.AsAggVector<T>()));

#define SINGLE_ARG(...) __VA_ARGS__
#define CASE(TYPE, STMTS)                       \
  case internal::TypeTraits<TYPE>::kDataType: { \
    typedef TYPE T;                             \
    STMTS;                                      \
    break;                                      \
  }

// TODO(team): Add other types.
#define CASES(TYPE_ENUM, STMTS)         \
  switch (TYPE_ENUM) {                  \
    CASE(float, SINGLE_ARG(STMTS))      \
    CASE(double, SINGLE_ARG(STMTS))     \
    CASE(int32_t, SINGLE_ARG(STMTS))    \
    CASE(int64_t, SINGLE_ARG(STMTS))    \
    case DT_INVALID:                    \
      FCP_LOG(FATAL) << "Invalid type"; \
      break;                            \
    default:                            \
      FCP_LOG(FATAL) << "Unknown type"; \
  }

}  // namespace internal

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_DATATYPE_H_
