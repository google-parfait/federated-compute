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

#ifndef FCP_CLIENT_CLIENT_RUNNER_H_
#define FCP_CLIENT_CLIENT_RUNNER_H_

#include <cxxabi.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <typeinfo>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/fake_event_publisher.h"
#include "fcp/client/files.h"
#include "fcp/client/flags.h"
#include "fcp/client/histogram_counters.pb.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/protos/plan.pb.h"

namespace fcp::client {

// A stub implementation of the SimpleTaskEnvironment interface that logs calls
// to stderr and returns empty example iterators.
class FederatedTaskEnvDepsImpl : public SimpleTaskEnvironment {
 public:
  explicit FederatedTaskEnvDepsImpl(int num_examples)
      : num_examples_(num_examples) {}

  std::string GetBaseDir() override {
    return std::filesystem::path(testing::TempDir());
  }

  std::string GetCacheDir() override {
    return std::filesystem::path(testing::TempDir());
  }

  absl::StatusOr<std::unique_ptr<ExampleIterator>> CreateExampleIterator(
      const google::internal::federated::plan::ExampleSelector&
          example_selector) override {
    SelectorContext unused;
    return CreateExampleIterator(example_selector, unused);
  }

  absl::StatusOr<std::unique_ptr<ExampleIterator>> CreateExampleIterator(
      const google::internal::federated::plan::ExampleSelector&
          example_selector,
      const SelectorContext& selector_context) override {
    class EmptyExampleIterator : public ExampleIterator {
     public:
      explicit EmptyExampleIterator(int num_examples)
          : num_examples_(num_examples), num_examples_served_(0) {}
      absl::StatusOr<std::string> Next() override {
        if (num_examples_served_ >= num_examples_) {
          return absl::OutOfRangeError("");
        }
        num_examples_served_++;
        return std::string("");
      }
      void Close() override {}

     private:
      const int num_examples_;
      int num_examples_served_;
    };
    FCP_CLIENT_LOG_FUNCTION_NAME
        << ":\n\turi: " << example_selector.collection_uri()
        << "\n\ttype: " << example_selector.criteria().type_url();
    return absl::StatusOr<std::unique_ptr<ExampleIterator>>(
        std::make_unique<EmptyExampleIterator>(num_examples_));
  }

 private:
  bool TrainingConditionsSatisfied() override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
    return true;
  }

  const int num_examples_;
};

// An implementation of the Files interface that attempts to create a temporary
// file with the given prefix and suffix in a directory suitable for temporary
// files.
// NB this is a proof-of-concept implementation that does not use existing infra
// such as mkstemps() or std::tmpfile due to the requirements of the existing
// Files API: include prefix, suffix strings in filename; return file path
// instead of file descriptor.
class FilesImpl : public Files {
 public:
  FilesImpl() { std::srand(static_cast<int32_t>(std::time(nullptr))); }

  absl::StatusOr<std::string> CreateTempFile(
      const std::string& prefix, const std::string& suffix) override {
    const auto tmp_dir = std::filesystem::path(testing::TempDir());
    std::filesystem::path candidate_path;
    int fd;
    do {
      candidate_path =
          tmp_dir / absl::StrCat(prefix, std::to_string(std::rand()), suffix);
    } while ((fd = open(candidate_path.c_str(), O_CREAT | O_EXCL | O_RDWR,
                        S_IRWXU)) == -1 &&
             errno == EEXIST);
    close(fd);
    std::ofstream tmp_file(candidate_path);
    if (!tmp_file) {
      return absl::InvalidArgumentError(
          absl::StrCat("could not create file ", candidate_path.string()));
    }
    FCP_CLIENT_LOG_FUNCTION_NAME << ": " << candidate_path;
    return candidate_path.string();
  }
};

// A stub implementation of the LogManager interface that logs invocations to
// stderr.
class LogManagerImpl : public LogManager {
 public:
  void LogDiag(ProdDiagCode diag_code) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << ": " << ProdDiagCode_Name(diag_code);
  }
  void LogDiag(DebugDiagCode diag_code) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << ": " << DebugDiagCode_Name(diag_code);
  }
  void LogToLongHistogram(HistogramCounters histogram_counter, int, int,
                          engine::DataSourceType data_source_type,
                          int64_t value) override {
    FCP_CLIENT_LOG_FUNCTION_NAME
        << ": " << HistogramCounters_Name(histogram_counter) << " <- " << value;
  }

  void SetModelIdentifier(const std::string& model_identifier) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << ":\n\t" << model_identifier;
  }
};

class FlagsImpl : public Flags {
 public:
  int64_t condition_polling_period_millis() const override { return 1000; }
  int64_t tf_execution_teardown_grace_period_millis() const override {
    return 1000;
  }
  int64_t tf_execution_teardown_extended_period_millis() const override {
    return 2000;
  }
  int64_t grpc_channel_deadline_seconds() const override { return 0; }
  bool log_tensorflow_error_messages() const override { return true; }
};

}  // namespace fcp::client

#endif  // FCP_CLIENT_CLIENT_RUNNER_H_
