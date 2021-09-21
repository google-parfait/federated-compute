/*
 * Copyright 2019 Google LLC
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
#include "fcp/client/engine/plan_engine.h"

#include <fstream>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/engine/plan_engine_helpers.h"
#include "fcp/client/histogram_counters.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/tensorflow/external_dataset.h"
#include "tensorflow/core/protobuf/saver.pb.h"

namespace fcp {
namespace client {
namespace engine {

using ::fcp::client::opstats::OperationalStats;
using ::fcp::client::opstats::OpStatsLogger;
using ::google::internal::federated::plan::CheckpointOp;
using ::google::internal::federated::plan::ClientExecution;
using ::google::internal::federated::plan::ClientOnlyPlan;
using ::google::internal::federated::plan::ExampleSelector;
using ::google::internal::federatedml::v2::Checkpoint;

static constexpr char kResultCheckpointPrefix[] = "result";
static constexpr char kResultCheckpointSuffix[] = ".ckp";
static constexpr char kSecAggCheckpointPrefix[] = "secagg";
static constexpr char kSecAggCheckpointSuffix[] = ".pb";
bool PlanEngine::RunPlan(TaskEnvironment* task_environment, Files* files,
                         LogManager* log_manager,
                         EventPublisher* event_publisher,
                         OpStatsLogger* opstats_logger,
                         const ClientOnlyPlan& client_plan,
                         const std::string& initial_checkpoint_uri,
                         const InterruptibleRunner::TimingConfig& timing_config,
                         absl::Time reference_time,
                         bool log_tensorflow_error_messages) {
  absl::Time run_plan_start_time = absl::Now();
  log_tensorflow_error_messages_ = log_tensorflow_error_messages;
  if (!PlanIntegrityChecks(client_plan)) {
    log_manager->LogDiag(
        ProdDiagCode::BACKGROUND_TRAINING_FAILED_PLAN_FAILS_SANITY_CHECK);
    std::string message = "invalid ClientOnlyPlan";
    event_publisher->PublishIoError(0, message);
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
    (void)CallFinish(task_environment, event_publisher, opstats_logger,
                     PhaseOutcome::ERROR, run_plan_start_time);
    return false;
  }

  auto tf_wrapper_or = TensorFlowWrapper::Create(
      client_plan.graph(),
      // Plan-provided Tensorflow ConfigProtos aren't supported for legacy
      // plans. We just provide an empty value, so that hardcoded defaults will
      // be used.
      ::google::protobuf::Any(),
      [&task_environment]() { return task_environment->ShouldAbort(); },
      timing_config, log_manager);
  if (!tf_wrapper_or.ok()) {
    event_publisher->PublishTensorFlowError(
        /*execution_index=*/0,
        /*epoch_index=*/0,
        /*epoch_example_index=*/0,
        log_tensorflow_error_messages_ ? tf_wrapper_or.status().message() : "");
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW,
        std::string(tf_wrapper_or.status().message()));
    (void)CallFinish(task_environment, event_publisher, opstats_logger,
                     PhaseOutcome::ERROR, run_plan_start_time);
    return false;
  }
  tf_wrapper_ = std::move(tf_wrapper_or).value();

  absl::Status status =
      RunPlanInternal(task_environment, files, log_manager, event_publisher,
                      opstats_logger, client_plan, initial_checkpoint_uri);
  FCP_CHECK(tf_wrapper_->CloseAndRelease().ok());

  // Log timing info.
  LogTimeSince(log_manager, HistogramCounters::TRAINING_RUN_PHASE_LATENCY, 0, 0,
               DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
               run_plan_start_time);
  LogTimeSince(log_manager, HistogramCounters::TRAINING_RUN_PHASE_END_TIME, 0,
               0, DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
               reference_time);

  switch (status.code()) {
    case absl::StatusCode::kOk:
      return CallFinish(task_environment, event_publisher, opstats_logger,
                        PhaseOutcome::COMPLETED, run_plan_start_time);
    case absl::StatusCode::kCancelled:
      (void)CallFinish(task_environment, event_publisher, opstats_logger,
                       PhaseOutcome::INTERRUPTED, run_plan_start_time);
      return false;
    case absl::StatusCode::kInvalidArgument:
      (void)CallFinish(task_environment, event_publisher, opstats_logger,
                       PhaseOutcome::ERROR, run_plan_start_time);
      return false;
    default:
      FCP_LOG(FATAL) << "unexpected status code: " << status.code();
  }
  // Unreachable, but clang doesn't get it.
  return false;
}

absl::Status PlanEngine::RunPlanInternal(
    TaskEnvironment* task_environment, Files* files, LogManager* log_manager,
    EventPublisher* event_publisher, OpStatsLogger* opstats_logger,
    const ClientOnlyPlan& client_plan,
    const std::string& initial_checkpoint_uri) {
  absl::Time start_time = absl::Now();
  // Start running the plan.
  event_publisher->PublishPlanExecutionStarted();
  opstats_logger->AddEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED);

  // 1. Init for the first ClientExecution. Must be run before checkpoint is
  // loaded.
  const ClientExecution& first_execution = client_plan.phase().execution(0);
  FCP_ENGINE_RETURN_IF_ERROR(
      RunTensorFlow(first_execution.init_op(), event_publisher, opstats_logger,
                    /*execution_index=*/0,
                    /*epoch_index=*/0,
                    /*example_index=*/0,
                    /*example_size=*/0, start_time, log_manager,
                    HistogramCounters::TRAINING_INIT_OP_LATENCY,
                    DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED));

  // 2. Load initial checkpoint.
  FCP_ENGINE_RETURN_IF_ERROR(
      LoadState(first_execution.read_client_init(), initial_checkpoint_uri,
                event_publisher, opstats_logger,
                /*execution_index=*/0,
                /*epoch_index=*/0,
                /*example_index=*/0,
                /*example_size=*/0, start_time, log_manager));

  // 3. Loop over ClientExecutions.
  std::atomic<int> total_example_count = 0;
  std::atomic<int64_t> total_example_size_bytes = 0;
  for (int execution_index = 0;
       execution_index < client_plan.phase().execution_size();
       execution_index++) {
    FCP_ENGINE_RETURN_IF_ERROR(
        RunExecution(task_environment, files, log_manager, event_publisher,
                     opstats_logger, client_plan, execution_index, start_time,
                     &total_example_count, &total_example_size_bytes));
  }

  event_publisher->PublishPlanCompleted(total_example_count,
                                        total_example_size_bytes, start_time);
  opstats_logger->AddEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED);
  log_manager->LogToLongHistogram(
      HistogramCounters::TRAINING_OVERALL_EXAMPLE_COUNT, 0, 0,
      DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED, total_example_count);
  log_manager->LogToLongHistogram(
      HistogramCounters::TRAINING_OVERALL_EXAMPLE_SIZE, 0, 0,
      DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED, total_example_size_bytes);

  return absl::OkStatus();
}

absl::Status PlanEngine::RunExecution(
    TaskEnvironment* task_environment, Files* files, LogManager* log_manager,
    EventPublisher* event_publisher, OpStatsLogger* opstats_logger,
    const ClientOnlyPlan& client_plan, int execution_index,
    absl::Time start_time, std::atomic<int>* total_example_count,
    std::atomic<int64_t>* total_example_size_bytes) {
  absl::Time execution_start_time = absl::Now();
  const ClientExecution& execution =
      client_plan.phase().execution(execution_index);
  DataSourceType data_source_type =
      DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED;
  if (execution.input_kind_case() == ClientExecution::kInputFeed) {
    data_source_type = DataSourceType::FEED;
  } else if (execution.input_kind_case() ==
             ClientExecution::kExternalDatasetTokenFeed) {
    data_source_type = DataSourceType::DATASET;
  }

  // Do not run the first execution's init op - it was run already before the
  // initial checkpoint was loaded.
  if (execution_index != 0) {
    FCP_ENGINE_RETURN_IF_ERROR(RunTensorFlow(
        execution.init_op(), event_publisher, opstats_logger, execution_index,
        0, 0, *total_example_size_bytes, start_time, log_manager,
        HistogramCounters::TRAINING_INIT_OP_LATENCY, data_source_type));
  }
  int epoch_index = 0;
  if (execution.input_kind_case() ==
      ClientExecution::kExternalDatasetTokenFeed) {
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    // AddDatasetTokenToInputs first registers a DatasetProvider with the
    // global ExternalDatasetProviderRegistry and then returns a
    // HostObjectRegistration object. Hold onto the HostObjectRegistration
    // object since it de-registers upon destruction.
    HostObjectRegistration host_registration = AddDatasetTokenToInputs(
        [&task_environment](
            const google::internal::federated::plan::ExampleSelector&
                selector) {
          return task_environment->CreateExampleIterator(selector);
        },
        event_publisher, log_manager, opstats_logger, &inputs,
        execution.external_dataset_token_feed(), total_example_count,
        total_example_size_bytes);

    // unused
    std::vector<std::string> output_names(0);
    std::vector<tensorflow::Tensor> output_values;

    std::vector<std::string> target_names;
    target_names.emplace_back(execution.before_op());

    absl::Time call_start_time = absl::Now();
    bool should_abort = false;
    FCP_ENGINE_RETURN_IF_ERROR(RunTensorFlowInternal(
        inputs, output_names, target_names, &output_values, event_publisher,
        opstats_logger, 0, 0, 0, 0, call_start_time, &should_abort));

    LogTimeSince(log_manager, HistogramCounters::TRAINING_BEFORE_OP_LATENCY,
                 execution_index, 0, data_source_type, call_start_time);
  } else {
    FCP_ENGINE_RETURN_IF_ERROR(RunTensorFlow(
        execution.before_op(), event_publisher, opstats_logger, execution_index,
        /*epoch_index=*/0,
        /*example_index=*/0,
        /*example_size=*/0, start_time, log_manager,
        HistogramCounters::TRAINING_BEFORE_OP_LATENCY, data_source_type));

    ExecPerfStats exec_perf_stats = {
        .example_count = 0,
        .example_size_bytes = 0l,
        .gather_minibatch_latency = absl::ZeroDuration(),
        .run_minibatch_latency = absl::ZeroDuration(),
        .num_minibatches = 0};
    // Set by TensorFlow when we should abort the ongoing ClientExecution.
    // NB this is a plan-induced abort, which is expected to happen, and is
    // handled differently from an external abort (e.g. OS interruption).
    bool execution_aborted = false;
    for (epoch_index = 0;
         epoch_index < execution.epochs_to_run() && !execution_aborted;
         epoch_index++) {
      absl::Time epoch_start_time = absl::Now();
      FCP_ENGINE_RETURN_IF_ERROR(RunEpoch(
          task_environment, files, log_manager, event_publisher, opstats_logger,
          execution, execution_index, start_time, epoch_index,
          &execution_aborted, *total_example_size_bytes, &exec_perf_stats));
      LogTimeSince(log_manager, HistogramCounters::TRAINING_RUN_EPOCH_LATENCY,
                   execution_index, epoch_index, data_source_type,
                   epoch_start_time);
    }
    *total_example_count += exec_perf_stats.example_count;
    *total_example_size_bytes += exec_perf_stats.example_size_bytes;

    // Log latencies and counters about this execution.
    // 1. Log number and size of examples processed.
    log_manager->LogToLongHistogram(
        HistogramCounters::TRAINING_CLIENT_EXECUTION_EXAMPLE_SIZE,
        execution_index, 0, DataSourceType::FEED,
        exec_perf_stats.example_size_bytes);
    log_manager->LogToLongHistogram(
        HistogramCounters::TRAINING_CLIENT_EXECUTION_EXAMPLE_COUNT,
        execution_index, 0, DataSourceType::FEED,
        exec_perf_stats.example_count);
    // 2. Log batch-level statistics (number + size of examples, latencies for
    //    gathering batches + processing them).
    if (exec_perf_stats.num_minibatches > 0) {
      log_manager->LogToLongHistogram(
          HistogramCounters::TRAINING_GATHER_MINI_BATCH_LATENCY,
          execution_index, 0, DataSourceType::FEED,
          absl::ToInt64Milliseconds(exec_perf_stats.gather_minibatch_latency) /
              exec_perf_stats.num_minibatches);
      log_manager->LogToLongHistogram(
          HistogramCounters::TRAINING_RUN_MINI_BATCH_LATENCY, execution_index,
          0, DataSourceType::FEED,
          absl::ToInt64Milliseconds(exec_perf_stats.run_minibatch_latency) /
              exec_perf_stats.num_minibatches);
      log_manager->LogToLongHistogram(
          HistogramCounters::TRAINING_MINI_BATCH_EXAMPLE_SIZE, execution_index,
          0, DataSourceType::FEED,
          exec_perf_stats.example_size_bytes / exec_perf_stats.num_minibatches);
      log_manager->LogToLongHistogram(
          HistogramCounters::TRAINING_MINI_BATCH_EXAMPLE_COUNT, execution_index,
          0, DataSourceType::FEED,
          exec_perf_stats.example_count / exec_perf_stats.num_minibatches);
    }
    // 3. Log number and size of examples processed per epoch.
    if (epoch_index != 0) {
      log_manager->LogToLongHistogram(
          HistogramCounters::TRAINING_EPOCH_EXAMPLE_SIZE, execution_index, 0,
          DataSourceType::FEED,
          exec_perf_stats.example_size_bytes / epoch_index);
      log_manager->LogToLongHistogram(
          HistogramCounters::TRAINING_EPOCH_EXAMPLE_COUNT, execution_index, 0,
          DataSourceType::FEED, exec_perf_stats.example_count / epoch_index);
    }
  }

  FCP_ENGINE_RETURN_IF_ERROR(RunTensorFlow(
      execution.after_op(), event_publisher, opstats_logger, execution_index,
      /*epoch_index=*/0,
      /*example_index=*/0, *total_example_size_bytes, start_time, log_manager,
      HistogramCounters::TRAINING_AFTER_OP_LATENCY,
      DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED));

  FCP_ENGINE_RETURN_IF_ERROR(FinishExecution(
      task_environment, files, log_manager, event_publisher, opstats_logger,
      data_source_type, execution, execution_index, execution_start_time,
      epoch_index, *total_example_size_bytes));

  return absl::OkStatus();
}

absl::Status PlanEngine::RunEpoch(
    TaskEnvironment* task_environment, Files* files, LogManager* log_manager,
    EventPublisher* event_publisher, OpStatsLogger* opstats_logger,
    const ClientExecution& execution, int execution_index,
    absl::Time start_time, int epoch_index, bool* execution_aborted,
    int64_t total_example_size_bytes, ExecPerfStats* exec_perf_stats) {
  absl::Time epoch_start_time = absl::Now();
  int epoch_example_index = 0;
  int64_t epoch_example_size_bytes = 0;
  event_publisher->PublishEpochStarted(execution_index, epoch_index);
  if (!execution.input_feed().empty() && !execution.loop_op().empty()) {
    auto example_iterator_or = GetExampleIterator(
        execution.example_selector(), log_manager, opstats_logger,
        [&task_environment](
            const google::internal::federated::plan::ExampleSelector&
                selector) {
          return task_environment->CreateExampleIterator(selector);
        });
    if (example_iterator_or.status().code() ==
        absl::StatusCode::kInvalidArgument) {
      event_publisher->PublishExampleSelectorError(
          execution_index, epoch_index, epoch_example_index,
          example_iterator_or.status().message());
      opstats_logger->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_EXAMPLE_SELECTOR,
          std::string(example_iterator_or.status().message()));
      return absl::InvalidArgumentError("");
    }
    FCP_CHECK(example_iterator_or.ok());
    std::unique_ptr<ExampleIterator> example_iterator =
        std::move(example_iterator_or.value());

    // Read examples until done.
    absl::Time gather_batch_start_time = absl::Now();
    bool end_of_iterator = false;
    std::vector<std::string> batch;
    int batch_size = (execution.batch_size() == 0) ? 1 : execution.batch_size();
    batch.reserve(execution.batch_size());
    int64_t batch_size_bytes = 0;
    while (!end_of_iterator && !(*execution_aborted)) {
      if (task_environment->ShouldAbort()) {
        opstats_logger->AddEventWithErrorMessage(
            OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
            "Task environment decided to abort");
        return absl::CancelledError("");
      }
      absl::StatusOr<std::string> example_or = example_iterator->Next();
      absl::StatusCode error_code = example_or.status().code();
      if (error_code == absl::StatusCode::kCancelled) {
        event_publisher->PublishInterruption(
            execution_index, epoch_index, epoch_example_index,
            total_example_size_bytes + exec_perf_stats->example_size_bytes,
            start_time);
        opstats_logger->AddEventWithErrorMessage(
            OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
            std::string(example_or.status().message()));
        return example_or.status();
      } else if (error_code == absl::StatusCode::kInvalidArgument) {
        std::string message = absl::StrCat("Error reading example: ",
                                           example_or.status().message());
        event_publisher->PublishIoError(execution_index, message);
        opstats_logger->AddEventWithErrorMessage(
            OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
        return example_or.status();
      } else if (error_code == OUT_OF_RANGE) {
        end_of_iterator = true;
      } else {
        FCP_CHECK(error_code == absl::StatusCode::kOk);
        batch.emplace_back(example_or.value());
        batch_size_bytes += example_or.value().size();
      }
      if (batch.size() >= batch_size || (end_of_iterator && !batch.empty())) {
        if (task_environment->ShouldAbort()) {
          return absl::CancelledError("");
        }
        absl::Time run_batch_start_time = absl::Now();
        exec_perf_stats->gather_minibatch_latency +=
            run_batch_start_time - gather_batch_start_time;

        log_manager->LogDiag(DebugDiagCode::TRAINING_BEFORE_LOOP_OP);

        FCP_ENGINE_RETURN_IF_ERROR(RunTensorFlow(
            execution.input_feed(), batch, execution.loop_op(),
            execution_aborted, event_publisher, opstats_logger, execution_index,
            epoch_index, epoch_example_index,
            total_example_size_bytes + exec_perf_stats->example_size_bytes,
            start_time));

        gather_batch_start_time = absl::Now();
        exec_perf_stats->run_minibatch_latency +=
            gather_batch_start_time - run_batch_start_time;

        log_manager->LogDiag(DebugDiagCode::TRAINING_AFTER_LOOP_OP);

        exec_perf_stats->example_size_bytes += batch_size_bytes;
        exec_perf_stats->example_count += batch.size();
        exec_perf_stats->num_minibatches += 1;
        epoch_example_size_bytes += batch_size_bytes;
        epoch_example_index += batch.size();
        batch.clear();
        batch_size_bytes = 0;
        gather_batch_start_time = absl::Now();
      }
    }
    example_iterator->Close();
    opstats_logger->UpdateDatasetStats(
        execution.example_selector().collection_uri(), epoch_example_index,
        epoch_example_size_bytes);
  }
  event_publisher->PublishEpochCompleted(
      execution_index, epoch_index, epoch_example_index,
      epoch_example_size_bytes, epoch_start_time);

  return absl::OkStatus();
}

absl::Status PlanEngine::FinishExecution(
    TaskEnvironment* task_environment, Files* files, LogManager* log_manager,
    EventPublisher* event_publisher, OpStatsLogger* opstats_logger,
    DataSourceType data_source_type, const ClientExecution& execution,
    int execution_index, absl::Time start_time, int epoch_index,
    int64_t total_example_size_bytes) {
  if (execution.stats_size() > 0) {
    FCP_ENGINE_RETURN_IF_ERROR(RunStatsAndConditionallyPublish(
        task_environment, log_manager, event_publisher, opstats_logger,
        execution, execution_index, epoch_index, start_time,
        total_example_size_bytes));
  }

  if (execution.has_write_update()) {
    FCP_ENGINE_RETURN_IF_ERROR(StageParameters(
        task_environment, files, event_publisher, opstats_logger, execution,
        execution_index, start_time, total_example_size_bytes));
  }
  LogTimeSince(log_manager,
               HistogramCounters::TRAINING_RUN_CLIENT_EXECUTION_LATENCY,
               execution_index, 0, data_source_type, start_time);
  return absl::OkStatus();
}

absl::Status PlanEngine::RunStatsAndConditionallyPublish(
    TaskEnvironment* task_environment, LogManager* log_manager,
    EventPublisher* event_publisher, OpStatsLogger* opstats_logger,
    const ClientExecution& execution, int execution_index, int epoch_index,
    absl::Time start_time, int64_t total_example_size_bytes) {
  std::vector<std::string> stat_variable_names;
  absl::flat_hash_map<std::string, double> stat_values;
  stat_variable_names.reserve(execution.stats_size());
  for (const auto& it : execution.stats()) {
    stat_variable_names.push_back(it.variable_name());
  }

  // A map from variable names to statistic values.
  absl::flat_hash_map<std::string, double> stat_variables_to_values;
  FCP_ENGINE_RETURN_IF_ERROR(
      RunStats(stat_variable_names, &stat_variables_to_values, event_publisher,
               opstats_logger, execution_index, 0, 0, total_example_size_bytes,
               start_time));
  // Convert to a map from human readable names to statistic values.
  for (const auto& it : execution.stats()) {
    auto stat_it = stat_variables_to_values.find(it.variable_name());
    if (stat_it != stat_variables_to_values.end()) {
      stat_values[it.stat_name()] = (*stat_it).second;
      // Append to list of all published stats, for later use in CallFinish().
      all_published_stats_.emplace_back(it.stat_name(), (*stat_it).second);
    }
  }

  if (task_environment->ShouldPublishStats()) {
    event_publisher->PublishStats(execution_index, epoch_index, stat_values);
  }

  // If there's a staged result, check whether we just observed the required
  // quality metric, and if yes, if the threshold is met.
  if (staged_result_.has_value()) {
    const auto& it = stat_values.find(staged_result_->quality_metric_name_);
    if (it != stat_values.end() &&
        it->second > staged_result_->quality_metric_threshold_) {
      absl::Status status = task_environment->PublishParameters(
          staged_result_->checkpoint_path_,
          staged_result_->secagg_checkpoint_path_);
      if (ABSL_PREDICT_FALSE(status.code() ==
                             absl::StatusCode::kInvalidArgument)) {
        std::string message =
            absl::StrCat("Error publishing parameters: ", status.message());
        event_publisher->PublishIoError(execution_index, message);
        opstats_logger->AddEventWithErrorMessage(
            OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
        return status;
      }
      FCP_CHECK(status.code() == absl::StatusCode::kOk);
      // NB Deletion of the files is left to the environment. This keeps the
      // ownership clearer (the environment created them in the first place and
      // can keep track of them).
      staged_result_.reset();
    }
  }
  return absl::OkStatus();
}

absl::Status PlanEngine::StageParameters(
    TaskEnvironment* task_environment, Files* files,
    EventPublisher* event_publisher, OpStatsLogger* opstats_logger,
    const ClientExecution& execution, int execution_index,
    absl::Time start_time, int64_t total_example_size_bytes) {
  if (staged_result_.has_value()) {
    staged_result_.reset();
  }
  // Create standard checkpoint file.
  absl::StatusOr<std::string> checkpoint_file =
      files->CreateTempFile(kResultCheckpointPrefix, kResultCheckpointSuffix);
  if (!checkpoint_file.ok()) {
    std::string message = "could not create temporary checkpoint file";
    event_publisher->PublishIoError(execution_index, message);
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
    return absl::InvalidArgumentError("");
  }
  // Create SecAgg checkpoint file.
  absl::StatusOr<std::string> secagg_checkpoint_file =
      files->CreateTempFile(kSecAggCheckpointPrefix, kSecAggCheckpointSuffix);
  if (!secagg_checkpoint_file.ok()) {
    std::string message = "could not create temporary secagg checkpoint file";
    event_publisher->PublishIoError(execution_index, message);
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
    return absl::InvalidArgumentError("");
  }
  // Store variables to checkpoint.
  FCP_ENGINE_RETURN_IF_ERROR(
      SaveState(execution.write_update(), checkpoint_file.value(),
                secagg_checkpoint_file.value(), event_publisher, opstats_logger,
                execution_index,
                /*epoch_index=*/0,
                /*example_index=*/0, total_example_size_bytes, start_time));

  if (execution.write_update_quality_metric().empty()) {
    // No quality threshold -> publish now.
    absl::Status status = task_environment->PublishParameters(
        checkpoint_file.value(), secagg_checkpoint_file.value());
    if (ABSL_PREDICT_FALSE(status.code() ==
                           absl::StatusCode::kInvalidArgument)) {
      event_publisher->PublishIoError(
          execution_index,
          absl::StrCat("Error publishing parameters: ", status.message()));
      return status;
    }
    FCP_CHECK(status.code() == absl::StatusCode::kOk);
  } else {
    // Stage for possible later publishing.
    staged_result_.emplace(execution.write_update_quality_metric(),
                           execution.write_update_quality_threshold(),
                           checkpoint_file.value(),
                           secagg_checkpoint_file.value());
  }
  return absl::OkStatus();
}

bool PlanEngine::CallFinish(TaskEnvironment* task_environment,
                            EventPublisher* event_publisher,
                            OpStatsLogger* opstats_logger,
                            PhaseOutcome phase_outcome,
                            absl::Time plan_start_time) {
  auto finish_result = task_environment->Finish(
      phase_outcome, absl::Now() - plan_start_time, all_published_stats_);
  if (!finish_result.ok()) {
    event_publisher->PublishIoError(
        0, absl::StrCat("Error finishing: ", finish_result.message()));
    LogOpStatsNetworkErrors(opstats_logger, finish_result,
                            std::string(finish_result.message()));
    FCP_LOG(INFO) << "Error finishing: " << finish_result.message();
    return false;
  }
  return true;
}

bool PlanEngine::PlanIntegrityChecks(const ClientOnlyPlan& plan) {
  bool plan_ok = true;
  if (!plan.has_phase() || plan.phase().execution_size() == 0) {
    FCP_LOG(ERROR) << "Plan has no phase or no executions";
    plan_ok = false;
  }
  for (int i = 0; i < plan.phase().execution_size(); i++) {
    const ClientExecution& execution = plan.phase().execution(i);
    if (execution.batch_size() < 0) {
      FCP_LOG(ERROR) << "Batch size is < 0";
      plan_ok = false;
    }
    if (execution.epochs_to_run() < 0) {
      FCP_LOG(ERROR) << "epochs_to_run is < 0";
      plan_ok = false;
    }
    for (const auto& it : execution.stats()) {
      if (it.variable_name().empty()) {
        FCP_LOG(ERROR) << "Execution " << i << " has empty variable_name";
        plan_ok = false;
      }
      if (it.stat_name().empty()) {
        FCP_LOG(ERROR) << "Execution " << i << " has empty stat_name";
        plan_ok = false;
      }
    }
  }
  return plan_ok;
}

absl::Status PlanEngine::RunTensorFlowInternal(
    const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
    const std::vector<std::string>& output_tensor_names,
    const std::vector<std::string>& target_node_names,
    std::vector<tensorflow::Tensor>* outputs, EventPublisher* event_publisher,
    OpStatsLogger* opstats_logger, int execution_index, int epoch_index,
    int example_index, int64_t example_size, absl::Time start, bool* aborted) {
  if (aborted != nullptr) {
    *aborted = false;
  }
  absl::Status status =
      tf_wrapper_->Run(inputs, output_tensor_names, target_node_names, outputs);
  switch (status.code()) {
    case absl::StatusCode::kCancelled:
      event_publisher->PublishInterruption(execution_index, epoch_index,
                                           example_index, example_size, start);
      opstats_logger->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
          std::string(status.message()));
      return status;
    case absl::StatusCode::kInvalidArgument:
      event_publisher->PublishTensorFlowError(
          execution_index, epoch_index, example_index,
          log_tensorflow_error_messages_ ? status.message() : "");
      opstats_logger->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW,
          std::string(status.message()));
      return status;
    case absl::StatusCode::kOutOfRange:
      if (aborted != nullptr) {
        *aborted = true;
        break;
      } else {
        event_publisher->PublishTensorFlowError(
            execution_index, epoch_index, example_index,
            log_tensorflow_error_messages_ ? "unexpected abort" : "");
        opstats_logger->AddEventWithErrorMessage(
            OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW,
            "unexpected abort");
        return absl::InvalidArgumentError("unexpected abort");
      }
    case absl::StatusCode::kOk:
      break;
    default:
      FCP_CHECK_STATUS(status);
  }

  return absl::OkStatus();
}

absl::Status PlanEngine::RunTensorFlow(const std::string& target_node_name,
                                       EventPublisher* event_publisher,
                                       OpStatsLogger* opstats_logger,
                                       int execution_index, int epoch_index,
                                       int example_index, int64_t example_size,
                                       absl::Time start) {
  if (target_node_name.empty()) return absl::OkStatus();
  return RunTensorFlowInternal({}, {}, {SanitizeTargetName(target_node_name)},
                               nullptr, event_publisher, opstats_logger,
                               execution_index, epoch_index, example_index,
                               example_size, start);
}

absl::Status PlanEngine::RunTensorFlow(
    const std::string& target_node_name, EventPublisher* event_publisher,
    OpStatsLogger* opstats_logger, int execution_index, int epoch_index,
    int example_index, int64_t example_size, absl::Time start,
    LogManager* log_manager, HistogramCounters histogram_counter,
    DataSourceType data_source_type) {
  if (target_node_name.empty()) return absl::OkStatus();
  absl::Time call_start_time = absl::Now();
  absl::Status status = RunTensorFlowInternal(
      {}, {}, {SanitizeTargetName(target_node_name)}, nullptr, event_publisher,
      opstats_logger, execution_index, epoch_index, example_index, example_size,
      start);
  LogTimeSince(log_manager, histogram_counter, execution_index, epoch_index,
               data_source_type, call_start_time);
  return status;
}

absl::Status PlanEngine::RunTensorFlow(
    const std::string& feed_name, const std::vector<std::string>& batch,
    const std::string& target_node_name, bool* execution_aborted,
    EventPublisher* event_publisher, OpStatsLogger* opstats_logger,
    int execution_index, int epoch_index, int example_index, int64_t example_size,
    absl::Time start) {
  tensorflow::Tensor string_batch_tensor(tensorflow::DT_STRING,
                                         {static_cast<int64_t>(batch.size())});
  for (int i = 0; i < batch.size(); i++) {
    string_batch_tensor.flat<tensorflow::tstring>()(i) = batch[i];
  }
  return RunTensorFlowInternal({{feed_name, string_batch_tensor}}, {},
                               {SanitizeTargetName(target_node_name)}, nullptr,
                               event_publisher, opstats_logger, execution_index,
                               epoch_index, example_index, example_size, start,
                               execution_aborted);
}

absl::Status PlanEngine::RunStats(
    const std::vector<std::string>& fetch_names,
    absl::flat_hash_map<std::string, double>* output_values,
    EventPublisher* event_publisher, OpStatsLogger* opstats_logger,
    int execution_index, int epoch_index, int example_index, int64_t example_size,
    absl::Time start) {
  std::vector<tensorflow::Tensor> output_tensors;
  std::vector<std::string> sanitized_fetch_names;
  sanitized_fetch_names.reserve(fetch_names.size());
  for (const auto& it : fetch_names) {
    sanitized_fetch_names.emplace_back(SanitizeTargetName(it));
  }
  FCP_ENGINE_RETURN_IF_ERROR(
      RunTensorFlowInternal({}, sanitized_fetch_names, {}, &output_tensors,
                            event_publisher, opstats_logger, execution_index,
                            epoch_index, example_index, example_size, start));
  output_values->clear();
  for (int i = 0; i < fetch_names.size(); i++) {
    const tensorflow::Tensor& tensor = output_tensors[i];
    const std::string& name = fetch_names[i];
    if (tensor.NumElements() == 0) {
      std::string message =
          absl::StrCat("stat '", name, "' returned empty tensor");
      event_publisher->PublishIoError(execution_index, message);
      opstats_logger->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
      return absl::InvalidArgumentError("");
    }
    switch (tensor.dtype()) {
      case tensorflow::DT_FLOAT:
        (*output_values)[name] = static_cast<double>(tensor.flat<float>()(0));
        break;
      case tensorflow::DT_DOUBLE:
        (*output_values)[name] = static_cast<double>(tensor.flat<double>()(0));
        break;
      case tensorflow::DT_INT32:
        (*output_values)[name] = static_cast<double>(tensor.flat<int32_t>()(0));
        break;
      case tensorflow::DT_INT64:
        (*output_values)[name] =
            static_cast<double>(tensor.flat<tensorflow::int64>()(0));
        break;
      case tensorflow::DT_BOOL:
        (*output_values)[name] = tensor.flat<bool>()(0) ? 1.0 : 0.0;
        break;
      default:
        FCP_LOG(ERROR) << "Unsupported stat value " << tensor.dtype();
    }
  }
  return absl::OkStatus();
}

absl::Status PlanEngine::LoadState(const CheckpointOp& checkpoint_op,
                                   const std::string& checkpoint_path,
                                   EventPublisher* event_publisher,
                                   OpStatsLogger* opstats_logger,
                                   int execution_index, int epoch_index,
                                   int example_index, int64_t example_size,
                                   absl::Time start, LogManager* log_manager) {
  absl::Time load_state_start_time = absl::Now();
  absl::Status status = LoadOrSaveState(
      checkpoint_op.before_restore_op(), checkpoint_op.has_saver_def(),
      checkpoint_op.saver_def().filename_tensor_name(), checkpoint_path,
      checkpoint_op.saver_def().restore_op_name(),
      checkpoint_op.after_restore_op(), event_publisher, opstats_logger,
      execution_index, epoch_index, example_index, example_size, start);

  LogTimeSince(log_manager, HistogramCounters::TRAINING_RESTORE_STATE_LATENCY,
               execution_index, epoch_index,
               DataSourceType::TRAINING_DATA_SOURCE_UNDEFINED,
               load_state_start_time);
  return status;
}

absl::Status PlanEngine::SaveState(
    const CheckpointOp& checkpoint_op, const std::string& checkpoint_path,
    const std::string& secagg_path, EventPublisher* event_publisher,
    OpStatsLogger* opstats_logger, int execution_index, int epoch_index,
    int example_index, int64_t example_size, absl::Time start) {
  // Save state to a TensorFlow v1 checkpoint. Don't run after_save_op yet -
  // save SecAgg checkpoint first.
  FCP_ENGINE_RETURN_IF_ERROR(LoadOrSaveState(
      checkpoint_op.before_save_op(), checkpoint_op.has_saver_def(),
      checkpoint_op.saver_def().filename_tensor_name(), checkpoint_path,
      checkpoint_op.saver_def().save_tensor_name(), /*after_op=*/"",
      event_publisher, opstats_logger, execution_index, epoch_index,
      example_index, example_size, start));

  // Save SecAgg tensors to a separate file.
  std::vector<std::string> fetch_names;
  std::vector<std::vector<int64_t>> dimensions;
  std::vector<tensorflow::Tensor> output_tensors;
  if (checkpoint_op.side_channel_tensors_size() > 0) {
    // 1. Get output tensor names.
    for (const auto& it : checkpoint_op.side_channel_tensors()) {
      fetch_names.push_back(it.first);
      auto& tensor_dimensions = dimensions.emplace_back();
      for (const auto& dimension : it.second.secure_aggregand().dimension())
        tensor_dimensions.emplace_back(dimension.size());
    }

    // 2. Get tensors.
    FCP_ENGINE_RETURN_IF_ERROR(RunTensorFlowInternal(
        {}, fetch_names, {}, &output_tensors, event_publisher, opstats_logger,
        execution_index, epoch_index, example_index, example_size, start));
    CHECK_EQ(fetch_names.size(), output_tensors.size());
  }

  // 3. Save to SecAgg proto.
  Checkpoint checkpoint;
  auto aggregands = checkpoint.mutable_aggregands();
  for (int i = 0; i < fetch_names.size(); i++) {
    absl::StatusOr<Checkpoint::Aggregand> aggregand_or =
        TensorProtoToQuantizedAggregand(output_tensors[i], dimensions[i]);
    if (!aggregand_or.ok()) {
      event_publisher->PublishIoError(execution_index,
                                      aggregand_or.status().message());
      opstats_logger->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_IO,
          std::string(aggregand_or.status().message()));
      return aggregand_or.status();
    }
    (*aggregands)[fetch_names[i]] = aggregand_or.value();
  }

  // 4. Serialize SecAgg proto to file.
  std::fstream checkpoint_stream(secagg_path, std::ios_base::out);
  if (checkpoint_stream.fail() ||
      !checkpoint.SerializeToOstream(&checkpoint_stream)) {
    std::string message = "Error creating SecAgg checkpoint";
    event_publisher->PublishIoError(execution_index, message);
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
    return absl::InvalidArgumentError("");
  }

  return RunTensorFlow(checkpoint_op.after_save_op(), event_publisher,
                       opstats_logger, execution_index, epoch_index,
                       example_index, example_size, start);
}

absl::Status PlanEngine::LoadOrSaveState(
    const std::string& before_op, bool has_saver_def,
    const std::string& filename_tensor_name, const std::string& checkpoint_path,
    const std::string& save_or_restore_op, const std::string& after_op,
    EventPublisher* event_publisher, OpStatsLogger* opstats_logger,
    int execution_index, int epoch_index, int example_index, int64_t example_size,
    absl::Time start) {
  FCP_ENGINE_RETURN_IF_ERROR(
      RunTensorFlow(before_op, event_publisher, opstats_logger, execution_index,
                    epoch_index, example_index, example_size, start));
  if (has_saver_def) {
    auto filename_tensor = tensorflow::Tensor(tensorflow::DT_STRING, {});
    filename_tensor.scalar<tensorflow::tstring>()() = checkpoint_path;
    FCP_ENGINE_RETURN_IF_ERROR(
        RunTensorFlowInternal({{filename_tensor_name, filename_tensor}}, {},
                              {SanitizeTargetName(save_or_restore_op)}, {},
                              event_publisher, opstats_logger, execution_index,
                              epoch_index, example_index, example_size, start));
  }
  return RunTensorFlow(after_op, event_publisher, opstats_logger,
                       execution_index, epoch_index, example_index,
                       example_size, start);
}

void PlanEngine::LogTimeSince(LogManager* log_manager,
                              HistogramCounters histogram_counter,
                              int execution_index, int epoch_index,
                              DataSourceType data_source_type,
                              absl::Time reference_time) {
  absl::Duration duration = absl::Now() - reference_time;
  log_manager->LogToLongHistogram(histogram_counter, execution_index,
                                  epoch_index, data_source_type,
                                  absl::ToInt64Milliseconds(duration));
}

const std::string PlanEngine::SanitizeTargetName(const std::string& node_name) {
  if (absl::EndsWith(node_name, ":0")) {
    return node_name.substr(0, node_name.size() - 2);
  }
  return node_name;
}

absl::StatusOr<Checkpoint::Aggregand>
PlanEngine::TensorProtoToQuantizedAggregand(
    const tensorflow::Tensor& tensor, const std::vector<int64_t>& dimensions) {
  switch (tensor.dtype()) {
    case tensorflow::DT_INT8:
      return AddValuesToQuantized<int8_t>(tensor, dimensions);
    case tensorflow::DT_UINT8:
      return AddValuesToQuantized<uint8_t>(tensor, dimensions);
    case tensorflow::DT_INT16:
      return AddValuesToQuantized<int16_t>(tensor, dimensions);
    case tensorflow::DT_UINT16:
      return AddValuesToQuantized<uint16_t>(tensor, dimensions);
    case tensorflow::DT_INT32:
      return AddValuesToQuantized<int32_t>(tensor, dimensions);
    case tensorflow::DT_INT64:
      return AddValuesToQuantized<tensorflow::int64>(tensor, dimensions);
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Tensor of type ", tensorflow::DataType_Name(tensor.dtype()),
          " could not be converted to Aggregand.Quantized"));
  }
}

template <typename T>
absl::StatusOr<Checkpoint::Aggregand> PlanEngine::AddValuesToQuantized(
    const tensorflow::Tensor& tensor, const std::vector<int64_t>& dimensions) {
  FCP_ASSIGN_OR_RETURN(int bitwidth, TensorTypeToBitWidth(tensor.dtype()));
  Checkpoint::Aggregand aggregand;
  auto mutable_quantized = aggregand.mutable_quantized();
  mutable_quantized->set_bitwidth(bitwidth);
  auto flat_tensor = tensor.flat<T>();
  for (int i = 0; i < flat_tensor.size(); i++) {
    mutable_quantized->add_values(flat_tensor(i));
  }
  for (auto i : dimensions) mutable_quantized->add_dimensions(i);

  return aggregand;
}

absl::StatusOr<int> PlanEngine::TensorTypeToBitWidth(
    tensorflow::DataType datatype) {
  switch (datatype) {
    case tensorflow::DT_INT8:
      return 7;
    case tensorflow::DT_UINT8:
      return 8;
    case tensorflow::DT_INT32:
      return 31;
    case tensorflow::DT_INT16:
      return 15;
    case tensorflow::DT_UINT16:
      return 16;
    case tensorflow::DT_INT64:
      return 62;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Tensor of type ", tensorflow::DataType_Name(datatype),
                       " could not be converted to Aggregand.Quantized"));
  }
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
