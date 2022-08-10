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
#include "fcp/client/event_publisher.h"
#include "fcp/client/files.h"
#include "fcp/client/flags.h"
#include "fcp/client/histogram_counters.pb.h"
#include "fcp/client/http/http_client.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/secagg_event_publisher.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/protos/plan.pb.h"

namespace fcp::client {

// Macro to print log messages prefixed by ClassName::FunctionName, stripping
// namespaces before ClassName, if any.
#define FCP_CLIENT_LOG_FUNCTION_NAME                              \
  std::string _demangle_buf(1024, '\0');                          \
  size_t _demangle_buf_len = _demangle_buf.length();              \
  abi::__cxa_demangle(typeid(*this).name(), _demangle_buf.data(), \
                      &_demangle_buf_len, nullptr);               \
  FCP_LOG(INFO) << static_cast<std::vector<std::string>>(         \
                       absl::StrSplit(_demangle_buf, "::"))       \
                       .back()                                    \
                       .c_str()                                   \
                << "::" << __func__

// An implementation of the SecAggEventPublisher interface that logs calls to
// stderr.
class SecAggLoggingEventPublisher : public SecAggEventPublisher {
 public:
  void PublishStateTransition(::fcp::secagg::ClientState state,
                              size_t last_sent_message_size,
                              size_t last_received_message_size) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }
  void PublishError() override { FCP_CLIENT_LOG_FUNCTION_NAME; }
  void PublishAbort(bool client_initiated,
                    const std::string& error_message) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }
  void set_execution_session_id(int64_t execution_session_id) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }
};

// An implementation of the EventPublisher interface that logs calls to stderr.
class LoggingEventPublisher : public EventPublisher {
 public:
  void PublishEligibilityEvalCheckin() override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishEligibilityEvalPlanReceived(int64_t, int64_t,
                                          absl::Duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishEligibilityEvalNotConfigured(int64_t, int64_t,
                                           absl::Duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishEligibilityEvalRejected(int64_t, int64_t,
                                      absl::Duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishCheckin() override { FCP_CLIENT_LOG_FUNCTION_NAME; }

  void PublishCheckinFinished(int64_t, int64_t, absl::Duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishRejected() override { FCP_CLIENT_LOG_FUNCTION_NAME; }

  void PublishReportStarted(int64_t) override { FCP_CLIENT_LOG_FUNCTION_NAME; }

  void PublishReportFinished(int64_t, int64_t, absl::Duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishPlanExecutionStarted() override { FCP_CLIENT_LOG_FUNCTION_NAME; }

  void PublishEpochStarted(int, int) override { FCP_CLIENT_LOG_FUNCTION_NAME; }

  void PublishTensorFlowError(int execution_index, int epoch_index, int,
                              absl::string_view error_message) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << "exec " << execution_index << ", epoch "
                                 << epoch_index << ": " << error_message;
  }

  void PublishIoError(int execution_index,
                      absl::string_view error_message) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << "exec " << execution_index << ": "
                                 << error_message;
  }

  void PublishExampleSelectorError(int execution_index, int epoch_index, int,
                                   absl::string_view error_message) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << "exec " << execution_index << ", epoch "
                                 << epoch_index << ": " << error_message;
  }

  void PublishInterruption(int, int, int, int64_t, absl::Time) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishEpochCompleted(int, int, int, int64_t, absl::Time) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishPlanCompleted(int total_example_count, int64_t,
                            absl::Time) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << ":\ttotal_example_count="
                                 << total_example_count;
  }

  void PublishTaskNotStarted(absl::string_view error_message) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalCheckInIoError(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalCheckInClientInterrupted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalCheckInServerAborted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalCheckInErrorInvalidPayload(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalComputationStarted() override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishEligibilityEvalComputationInvalidArgument(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalComputationExampleIteratorError(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalComputationTensorflowError(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalComputationInterrupted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishEligibilityEvalComputationCompleted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishCheckinIoError(int64_t bytes_downloaded,
                             int64_t chunking_layer_bytes_received,
                             absl::string_view error_message,
                             absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishCheckinClientInterrupted(
      int64_t bytes_downloaded, int64_t chunking_layer_bytes_received,
      absl::string_view error_message,
      absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishCheckinServerAborted(int64_t bytes_downloaded,
                                   int64_t chunking_layer_bytes_received,
                                   absl::string_view error_message,
                                   absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishCheckinInvalidPayload(int64_t bytes_downloaded,
                                    int64_t chunking_layer_bytes_received,
                                    absl::string_view error_message,
                                    absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishRejected(int64_t bytes_downloaded,
                       int64_t chunking_layer_bytes_downloaded,
                       absl::Duration download_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishCheckinFinishedV2(int64_t, int64_t, absl::Duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishComputationStarted() override { FCP_CLIENT_LOG_FUNCTION_NAME; }

  void PublishComputationInvalidArgument(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishComputationIOError(absl::string_view error_message,
                                 int total_example_count,
                                 int64_t total_example_size_bytes,
                                 absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishComputationExampleIteratorError(
      absl::string_view error_message, int total_example_count,
      int64_t total_example_size_bytes,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishComputationTensorflowError(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishComputationInterrupted(
      int total_example_count, int64_t total_example_size_bytes,
      absl::string_view error_message,
      absl::Duration computation_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishComputationCompleted(int total_example_count, int64_t,
                                   absl::Time) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << ":\ttotal_example_count="
                                 << total_example_count;
  }

  void PublishResultUploadStarted() override { FCP_CLIENT_LOG_FUNCTION_NAME; }

  void PublishResultUploadIOError(int64_t report_size_bytes,
                                  int64_t chunking_layer_bytes_sent,
                                  absl::string_view error_message,
                                  absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishResultUploadClientInterrupted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishResultUploadServerAborted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishResultUploadCompleted(int64_t report_size_bytes,
                                    int64_t chunking_layer_bytes_sent,
                                    absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void PublishFailureUploadStarted() override { FCP_CLIENT_LOG_FUNCTION_NAME; }

  void PublishFailureUploadIOError(int64_t report_size_bytes,
                                   int64_t chunking_layer_bytes_sent,
                                   absl::string_view error_message,
                                   absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishFailureUploadClientInterrupted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishFailureUploadServerAborted(
      int64_t report_size_bytes, int64_t chunking_layer_bytes_sent,
      absl::string_view error_message,
      absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << error_message;
  }

  void PublishFailureUploadCompleted(int64_t report_size_bytes,
                                     int64_t chunking_layer_bytes_sent,
                                     absl::Duration upload_duration) override {
    FCP_CLIENT_LOG_FUNCTION_NAME;
  }

  void SetModelIdentifier(const std::string& model_identifier) override {
    FCP_CLIENT_LOG_FUNCTION_NAME << ":\n\t" << model_identifier;
  }

  SecAggEventPublisher* secagg_event_publisher() override {
    return &secagg_event_publisher_;
  }

 private:
  SecAggLoggingEventPublisher secagg_event_publisher_;
};

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
