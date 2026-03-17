/*
 * Copyright 2021 Google LLC
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
#include "fcp/client/opstats/opstats_example_store.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "fcp/client/diag_codes.pb.h"

namespace fcp {
namespace client {
namespace opstats {

using ::google::internal::federated::plan::ExampleSelector;

namespace {

class OpStatsExampleIterator : public fcp::client::ExampleIterator {
 public:
  OpStatsExampleIterator() : finished_(false) {}
  absl::StatusOr<std::string> Next() override {
    if (finished_) {
      return absl::OutOfRangeError("The iterator is out of range.");
    }
    finished_ = true;
    return "";
  }

  void Close() override { finished_ = true; }

 private:
  bool finished_;
};

}  // anonymous namespace

bool OpStatsExampleIteratorFactory::CanHandle(
    const ExampleSelector& example_selector) {
  return example_selector.collection_uri() == opstats::kOpStatsCollectionUri;
}

absl::StatusOr<std::unique_ptr<fcp::client::ExampleIterator>>
OpStatsExampleIteratorFactory::CreateExampleIterator(
    const ExampleSelector& example_selector) {
  if (example_selector.collection_uri() != kOpStatsCollectionUri) {
    log_manager_->LogDiag(ProdDiagCode::OPSTATS_INCORRECT_COLLECTION_URI);
    return absl::InvalidArgumentError(absl::StrCat(
        "The collection uri is ", example_selector.collection_uri(),
        ", which is not the expected uri: ", kOpStatsCollectionUri));
  }

  return std::make_unique<OpStatsExampleIterator>();
}

}  // namespace opstats
}  // namespace client
}  // namespace fcp
