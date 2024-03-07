/*
 * Copyright 2024 Google LLC
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
#ifndef FCP_AGGREGATION_CORE_MUTABLE_STRING_DATA_H_
#define FCP_AGGREGATION_CORE_MUTABLE_STRING_DATA_H_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "fcp/aggregation/core/datatype.h"
#include "fcp/aggregation/core/tensor_data.h"

namespace fcp {
namespace aggregation {

// MutableStringData owns its string values and allows string values to be added
// one by one.
class MutableStringData : public TensorData {
 public:
  explicit MutableStringData(size_t expected_size) {
    strings_.reserve(expected_size);
    string_views_.reserve(expected_size);
  }
  ~MutableStringData() override = default;

  // Implementation of TensorData methods.
  size_t byte_size() const override {
    return string_views_.size() * sizeof(string_view);
  }
  const void* data() const override { return string_views_.data(); }

  void Add(std::string&& string) {
    strings_.emplace_back(std::move(string));
    string_views_.emplace_back(strings_.back());
  }

 private:
  std::vector<std::string> strings_;
  std::vector<string_view> string_views_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_MUTABLE_STRING_DATA_H_
