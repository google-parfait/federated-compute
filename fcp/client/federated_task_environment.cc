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
#include "fcp/client/federated_task_environment.h"

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/base/platform.h"

namespace fcp {
namespace client {

using ::fcp::client::engine::PhaseOutcome;
using ::google::internal::federated::plan::ExampleSelector;
using ::google::internal::federatedml::v2::Checkpoint;

FederatedTaskEnvironment::FederatedTaskEnvironment(
    SimpleTaskEnvironment* env_deps, FederatedProtocol* federated_protocol,
    LogManager* log_manager, EventPublisher* event_publisher,
    opstats::OpStatsLogger* opstats_logger, const Flags* flags,
    absl::Time reference_time, absl::Duration condition_polling_period)
    : FederatedTaskEnvironment(
          env_deps, federated_protocol, log_manager, event_publisher,
          opstats_logger, flags, reference_time, []() { return absl::Now(); },
          condition_polling_period) {}

FederatedTaskEnvironment::FederatedTaskEnvironment(
    SimpleTaskEnvironment* env_deps, FederatedProtocol* federated_protocol,
    LogManager* log_manager, EventPublisher* event_publisher,
    opstats::OpStatsLogger* opstats_logger, const Flags* flags,
    absl::Time reference_time, std::function<absl::Time()> get_time_fn,
    absl::Duration condition_polling_period)
    : env_deps_(env_deps),
      federated_protocol_(federated_protocol),
      log_manager_(log_manager),
      event_publisher_(event_publisher),
      opstats_logger_(opstats_logger),
      flags_(flags),
      reference_time_(reference_time),
      get_time_fn_(get_time_fn),
      condition_polling_period_(condition_polling_period) {}

FederatedTaskEnvironment::~FederatedTaskEnvironment() {
  absl::Status status = DeleteStagedFiles();
  if (!status.ok()) {
    FCP_LOG(ERROR) << "Error deleting staged files: " << status.message();
  }
}

absl::Status FederatedTaskEnvironment::Finish(
    PhaseOutcome phase_outcome, absl::Duration plan_duration,
    const std::vector<std::pair<std::string, double>>& stats) {
  if (phase_outcome == engine::INTERRUPTED) {
    // If plan execution got interrupted, we do not report to the server, and do
    // not wait for a reply, but bail fast to get out of the way.
    return absl::OkStatus();
  }
  absl::Status report_result = absl::InternalError("");
  const absl::Time before_report = get_time_fn_();
  if (flags_->per_phase_logs()) {
    FCP_RETURN_IF_ERROR(LogReportStart());
  }
  if (phase_outcome == engine::COMPLETED) {
    FCP_ASSIGN_OR_RETURN(auto results, CreateComputationResults());
    report_result = federated_protocol_->ReportCompleted(std::move(results),
                                                         stats, plan_duration);
  } else {
    report_result =
        federated_protocol_->ReportNotCompleted(phase_outcome, plan_duration);
  }

  const absl::Time after_report = get_time_fn_();
  if (flags_->per_phase_logs() && report_result.ok()) {
    LogReportFinish(federated_protocol_->report_request_size_bytes(),
                    federated_protocol_->chunking_layer_bytes_sent(),
                    after_report - before_report);
  }
  log_manager_->LogToLongHistogram(
      HistogramCounters::TRAINING_FL_REPORT_RESULTS_END_TIME,
      0 /*execution_index*/, 0 /*epoch_index*/,
      engine::DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
      absl::ToInt64Milliseconds(after_report - reference_time_));
  log_manager_->LogToLongHistogram(
      HistogramCounters::TRAINING_FL_REPORT_RESULTS_LATENCY,
      0 /*execution_index*/, 0 /*epoch_index*/,
      engine::DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
      absl::ToInt64Milliseconds(after_report - before_report));
  return report_result;
}

bool FederatedTaskEnvironment::ShouldAbort() {
  return env_deps_->ShouldAbort(get_time_fn_(), condition_polling_period_);
}

absl::StatusOr<std::unique_ptr<ExampleIterator>>
FederatedTaskEnvironment::CreateExampleIterator(
    const ExampleSelector& example_selector) {
  // CreateExampleIterator with SelectorContext not supported in legacy plans.
  return env_deps_->CreateExampleIterator(example_selector);
}

absl::StatusOr<ComputationResults>
FederatedTaskEnvironment::CreateComputationResults() const {
  ComputationResults results;
  // Key of the TF checkpoint inside the ComputationResults map. Ignored by
  // downstream code.
  const std::string kTensorflowCheckpointAggregand = "tensorflow_checkpoint";
  if (tf_checkpoint_file_.empty() || secagg_checkpoint_file_.empty()) {
    // Engine never called publish - create Checkpoint with empty TF checkpoint.
    results[kTensorflowCheckpointAggregand] = "";
  } else {
    // 1. Parse SecAgg Checkpoint proto, and move into results.
    std::ifstream secagg_checkpoint(secagg_checkpoint_file_);
    Checkpoint checkpoint;
    if (checkpoint.ParseFromIstream(&secagg_checkpoint) == false) {
      return absl::InvalidArgumentError("Cannot parse SecAgg checkpoint");
    }
    for (const auto& [k, v] : checkpoint.aggregands()) {
      if (!v.has_quantized()) {
        return absl::InvalidArgumentError(
            "SecAgg Checkpoint must not contain TF checkpoint.");
      }
      QuantizedTensor quantized;
      quantized.values.reserve(v.quantized().values_size());
      for (auto e : v.quantized().values()) {
        quantized.values.push_back(e);
      }
      quantized.bitwidth = v.quantized().bitwidth();
      quantized.dimensions.reserve(v.quantized().dimensions_size());
      for (auto e : v.quantized().dimensions()) {
        quantized.dimensions.push_back(e);
      }
      results.emplace(k, std::move(quantized));
    }
    // 2. Append TF checkpoint.
    FCP_ASSIGN_OR_RETURN(std::string tf_checkpoint,
                         fcp::ReadFileToString(tf_checkpoint_file_));
    results[kTensorflowCheckpointAggregand] = std::move(tf_checkpoint);
  }

  return results;
}

absl::Status FederatedTaskEnvironment::PublishParameters(
    const std::string& tf_checkpoint_file,
    const std::string& secagg_checkpoint_file) {
  FCP_CHECK(!tf_checkpoint_file.empty());
  FCP_CHECK(!secagg_checkpoint_file.empty());
  // First clean up previously published files.
  FCP_RETURN_IF_ERROR(DeleteStagedFiles());

  // Now stage new files, if they exist.
  if (!fcp::FileExists(tf_checkpoint_file) ||
      !fcp::FileExists(secagg_checkpoint_file)) {
    return absl::InvalidArgumentError("published file does not exist");
  }

  tf_checkpoint_file_ = tf_checkpoint_file;
  secagg_checkpoint_file_ = secagg_checkpoint_file;
  return absl::OkStatus();
}

absl::Status FederatedTaskEnvironment::DeleteStagedFiles() {
  for (const std::string& path :
       {tf_checkpoint_file_, secagg_checkpoint_file_}) {
    if (!path.empty() && std::remove(path.c_str()) != 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Could not delete previously published file ", path));
    }
  }
  tf_checkpoint_file_.clear();
  secagg_checkpoint_file_.clear();
  return absl::OkStatus();
}

bool FederatedTaskEnvironment::ShouldPublishStats() { return false; }

absl::Status FederatedTaskEnvironment::LogReportStart() {
  event_publisher_->PublishReportStarted(0);
  opstats_logger_->AddEvent(
      opstats::OperationalStats::Event::EVENT_KIND_UPLOAD_STARTED);
  return opstats_logger_->CommitToStorage();
}

void FederatedTaskEnvironment::LogReportFinish(
    int64_t report_request_size, int64_t chunking_layers_sent_bytes,
    absl::Duration upload_time) {
  event_publisher_->PublishReportFinished(
      report_request_size, chunking_layers_sent_bytes, upload_time);
  opstats_logger_->AddEvent(
      opstats::OperationalStats::Event::EVENT_KIND_UPLOAD_FINISHED);
}

}  // namespace client
}  // namespace fcp
