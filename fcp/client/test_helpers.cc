/*
 * Copyright 2020 Google LLC
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
#include "fcp/client/test_helpers.h"

#include <fcntl.h>

#include <fstream>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace fcp {
namespace client {

using ::google::internal::federated::plan::Dataset;

char const* const kFlAssetsPath = "fcp/client/assets";
char const* const kLcExamplesPath =
    "fcp/client/lc_assets/"
    "emnist_cnn_personalization_artifacts_test_examples.pb";

namespace {
bool LoadMessageFromFileAsString_(std::string path, std::string* msg) {
  std::ifstream checkpoint_istream(path);
  if (!checkpoint_istream) {
    return false;
  }
  std::stringstream checkpoint_stream;
  checkpoint_stream << checkpoint_istream.rdbuf();
  *msg = checkpoint_stream.str();
  return true;
}

bool LoadMessageLiteFromFile_(std::string path, google::protobuf::MessageLite* msg) {
  std::string data;
  if (!LoadMessageFromFileAsString_(path, &data)) {
    return false;
  }
  if (!msg->ParseFromString(data)) {
    return false;
  }
  return true;
}
}  // namespace

SimpleExampleIterator::SimpleExampleIterator(Dataset dataset) {
  for (const Dataset::ClientDataset& client_dataset : dataset.client_data()) {
    FCP_CHECK(client_dataset.selected_example_size() == 0)
        << "This constructor can only be used for Dataset protos with unnamed "
           "example data.";
    for (const auto& example : client_dataset.example()) {
      examples_.push_back(example);
    }
  }
  FCP_CHECK(!examples_.empty()) << "No data was loaded";
}

SimpleExampleIterator::SimpleExampleIterator(Dataset dataset,
                                             absl::string_view collection_uri) {
  for (const Dataset::ClientDataset& client_dataset : dataset.client_data()) {
    FCP_CHECK(client_dataset.selected_example_size() > 0)
        << "This constructor can only be used for Dataset protos with named "
           "example data.";
    for (const Dataset::ClientDataset::SelectedExample& selected_example :
         client_dataset.selected_example()) {
      // Only use those examples whose `ExampleSelector` matches the
      // `collection_uri` argument. Note that the `ExampleSelector`'s selection
      // criteria is ignored/not taken into account here.
      if (selected_example.selector().collection_uri() != collection_uri) {
        continue;
      }
      for (const auto& example : selected_example.example()) {
        examples_.push_back(example);
      }
    }
  }
  FCP_CHECK(!examples_.empty()) << "No data was loaded for " << collection_uri;
}

absl::StatusOr<std::string> SimpleExampleIterator::Next() {
  if (index_ < examples_.size()) {
    return examples_[index_++];
  }
  return absl::OutOfRangeError("");
}

absl::StatusOr<FlArtifacts> LoadFlArtifactResources(
    const std::string& artifact_prefix) {
  FlArtifacts result;
  // Load plan
  auto plan_filepath =
      absl::StrCat(std::string(kFlAssetsPath), "/", artifact_prefix,
                   "_federation_client_only_plan.pb");
  if (!LoadMessageLiteFromFile_(plan_filepath, &result.plan)) {
    return absl::InternalError("Failed to load ClientOnlyPlan");
  }

  // Load dataset
  auto dataset_filepath =
      absl::StrCat(std::string(kFlAssetsPath), "/", artifact_prefix,
                   "_federation_proxy_train_examples.pb");
  if (!LoadMessageLiteFromFile_(dataset_filepath, &result.dataset)) {
    return absl::InternalError("Failed to load example Dataset");
  }

  // Load checkpoint
  auto checkpoint_filepath =
      absl::StrCat(std::string(kFlAssetsPath), "/", artifact_prefix,
                   "_federation_test_checkpoint.client.ckp");
  if (!LoadMessageFromFileAsString_(checkpoint_filepath, &result.checkpoint)) {
    return absl::InternalError("Failed to load checkpoint");
  }
  return result;
}

bool LoadLcArtifactResources(Dataset* dataset) {
  auto dataset_filepath = std::string(kLcExamplesPath);
  return LoadMessageLiteFromFile_(dataset_filepath, dataset);
}

std::string ExtractSingleString(const tensorflow::Example& example,
                                const char key[]) {
  return example.features().feature().at(key).bytes_list().value().at(0);
}

google::protobuf::RepeatedPtrField<std::string> ExtractRepeatedString(
    const tensorflow::Example& example, const char key[]) {
  return example.features().feature().at(key).bytes_list().value();
}

int64_t ExtractSingleInt64(const tensorflow::Example& example,
                           const char key[]) {
  return example.features().feature().at(key).int64_list().value().at(0);
}

google::protobuf::RepeatedField<int64_t> ExtractRepeatedInt64(
    const tensorflow::Example& example, const char key[]) {
  return example.features().feature().at(key).int64_list().value();
}

}  // namespace client
}  // namespace fcp
