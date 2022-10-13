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

#ifndef FCP_AGGREGATION_TESTING_TESTING_H_
#define FCP_AGGREGATION_TESTING_TESTING_H_

#include <initializer_list>
#include <vector>

#include "gmock/gmock.h"
#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor_shape.h"

namespace fcp::aggregation {

// Converts a potentially sparse tensor to a flat vector of tensor values.
template <typename T>
std::vector<T> TensorValuesToVector(const Tensor& arg) {
  std::vector<T> vec(arg.shape().NumElements());
  AggVector<T> agg_vector = arg.AsAggVector<T>();
  for (int i = 0; i < agg_vector.num_slices(); i++) {
    auto slice = agg_vector.get_slice(i);
    size_t index = slice.start_index();
    for (auto v : slice) {
      vec[index++] = v;
    }
  }
  return vec;
}

// Writes description of a tensor to the ostream.
template <typename T>
void DescribeTensor(::std::ostream* os, DataType dtype, TensorShape shape,
                    std::vector<T> values) {
  // Max number of tensor values to be printed.
  constexpr int kMaxValues = 100;
  // TODO(team): Print dtype name istead of number.
  *os << "{dtype: " << dtype;
  *os << ", shape: {";
  bool insert_comma = false;
  for (auto dim_size : shape.dim_sizes()) {
    if (insert_comma) {
      *os << ", ";
    }
    *os << dim_size;
    insert_comma = true;
  }
  *os << "}, values: {";
  int num_values = 0;
  insert_comma = false;
  for (auto v : values) {
    if (++num_values > kMaxValues) {
      *os << "...";
      break;
    }
    if (insert_comma) {
      *os << ", ";
    }
    *os << v;
    insert_comma = true;
  }
  *os << "}}";
}

// Writes description of a tensor to the ostream.
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

// TensorMatcher implementation.
template <typename T>
class TensorMatcherImpl : public ::testing::MatcherInterface<const Tensor&> {
 public:
  TensorMatcherImpl(DataType expected_dtype, TensorShape expected_shape,
                    std::vector<T> expected_values)
      : expected_dtype_(expected_dtype),
        expected_shape_(expected_shape),
        expected_values_(expected_values) {}

  void DescribeTo(std::ostream* os) const override {
    DescribeTensor<T>(os, expected_dtype_, expected_shape_, expected_values_);
  }

  bool MatchAndExplain(
      const Tensor& arg,
      ::testing::MatchResultListener* listener) const override {
    return arg.dtype() == expected_dtype_ && arg.shape() == expected_shape_ &&
           TensorValuesToVector<T>(arg) == expected_values_;
  }

 private:
  DataType expected_dtype_;
  TensorShape expected_shape_;
  std::vector<T> expected_values_;
};

// TensorMatcher can be used to compare a tensor against an expected
// value type, shape, and the list of values.
template <typename T>
class TensorMatcher {
 public:
  explicit TensorMatcher(DataType expected_dtype, TensorShape expected_shape,
                         std::initializer_list<T> expected_values)
      : expected_dtype_(expected_dtype),
        expected_shape_(expected_shape),
        expected_values_(expected_values.begin(), expected_values.end()) {}
  // Intentionally allowed to be implicit.
  operator ::testing::Matcher<const Tensor&>() const {  // NOLINT
    return ::testing::MakeMatcher(new TensorMatcherImpl<T>(
        expected_dtype_, expected_shape_, expected_values_));
  }

 private:
  DataType expected_dtype_;
  TensorShape expected_shape_;
  std::vector<T> expected_values_;
};

template <typename T>
TensorMatcher<T> IsTensor(TensorShape expected_shape,
                          std::initializer_list<T> expected_values) {
  return TensorMatcher<T>(internal::TypeTraits<T>::kDataType, expected_shape,
                          expected_values);
}

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_TESTING_TESTING_H_
