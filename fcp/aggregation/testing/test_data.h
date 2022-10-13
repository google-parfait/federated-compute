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

#ifndef FCP_AGGREGATION_TESTING_TEST_DATA_H_
#define FCP_AGGREGATION_TESTING_TEST_DATA_H_

#include <initializer_list>
#include <memory>
#include <tuple>
#include <vector>

#include "fcp/aggregation/core/tensor_data.h"

namespace fcp::aggregation {

template <typename T>
using TestSlice = std::tuple<size_t, std::vector<T>>;

// A trivial test implementation of TensorData.
template <typename T>
class TestData : public TensorData {
 public:
  explicit TestData(size_t size) : byte_size_(size * sizeof(T)) {}

  void AddSlice(size_t index, std::vector<T> values) {
    slices_.emplace_back(index, values);
  }

  void AddSlice(TestSlice<T> slice) { slices_.emplace_back(slice); }

  int num_slices() const override { return static_cast<int>(slices_.size()); }

  Slice get_slice(int n) const override {
    const auto& t = slices_[n];
    size_t byte_offset = std::get<0>(t) * sizeof(T);
    const auto& vec = std::get<1>(t);
    return Slice{byte_offset, vec.size() * sizeof(T), vec.data()};
  }

  size_t byte_size() const override { return byte_size_; }

 private:
  size_t byte_size_;
  std::vector<TestSlice<T>> slices_;
};

// Creates test data with a given number of elements and zero slices.
// Slices can be added later.
template <typename T>
std::unique_ptr<TestData<T>> CreateTestData(size_t size) {
  return std::make_unique<TestData<T>>(size);
}

// Creates test data with a given number of elements and initializes it with the
// provided slices.
template <typename T>
std::unique_ptr<TestData<T>> CreateTestData(
    size_t size, std::initializer_list<TestSlice<T>> slices) {
  auto test_data = CreateTestData<T>(size);
  for (auto& slice : slices) {
    test_data->AddSlice(slice);
  }
  return test_data;
}

// Creates test data with a single slice initialized with the specified values.
template <typename T>
std::unique_ptr<TestData<T>> CreateTestData(std::initializer_list<T> values) {
  auto test_data = CreateTestData<T>(values.size());
  test_data->AddSlice(0, values);
  return test_data;
}

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_TESTING_TEST_DATA_H_
