// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FCP_AGGREGATION_CHECKPOINT_IN_MEMORY_CHECKPOINT_PARSER_H_
#define FCP_AGGREGATION_CHECKPOINT_IN_MEMORY_CHECKPOINT_PARSER_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "fcp/aggregation/checkpoint/checkpoint_parser.h"
#include "fcp/aggregation/core/tensor.h"

namespace fcp::aggregation {

// A simple pass-through CheckpointParser.
class InMemoryCheckpointParser : public CheckpointParser {
 public:
  explicit InMemoryCheckpointParser(std::vector<Tensor> columns) {
    for (auto& column : columns) {
      tensors_[column.name()] = std::move(column);
    }
  }

  // Returns the tensor with the given name.
  //
  // This method is destructive, i.e. subsequent calls for the same tensor
  // name will result in a NotFoundError.
  //
  // Returns a NotFoundError if the tensor is not found.
  absl::StatusOr<Tensor> GetTensor(const std::string& name) override {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      return absl::NotFoundError(absl::StrCat("Tensor not found: ", name));
    }
    // Move the tensor out of the map. Subsequent calls for the same tensor
    // name will result in a NotFoundError.
    Tensor tensor = std::move(it->second);
    tensors_.erase(it);
    return tensor;
  }

  // Returns all tensors from the checkpoint.
  //
  // This method is destructive, i.e. subsequent calls will result in an empty
  // map.
  absl::StatusOr<absl::flat_hash_map<std::string, Tensor>> LoadAllTensors()
      override {
    return std::move(tensors_);
  }

 private:
  absl::flat_hash_map<std::string, Tensor> tensors_;
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_CHECKPOINT_IN_MEMORY_CHECKPOINT_PARSER_H_
