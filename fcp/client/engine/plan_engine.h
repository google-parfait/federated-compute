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
#ifndef FCP_CLIENT_ENGINE_PLAN_ENGINE_H_
#define FCP_CLIENT_ENGINE_PLAN_ENGINE_H_

#include <cstddef>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/engine/plan_engine_helpers.h"
#include "fcp/client/engine/tf_wrapper.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/files.h"
#include "fcp/client/histogram_counters.pb.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/task_environment.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace engine {

// A class used to "run" (interpret) the provided plan.
// Dependencies such as access to files, logging, and TensorFlow are provided
// at runtime.
class PlanEngine {
 public:
  // Runs the plan and reports results to the server via task_environment.
  // Returns true if both running the plan and reporting to the server were
  // successful.
  bool RunPlan(
      TaskEnvironment* task_environment, Files* files, LogManager* log_manager,
      EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      const google::internal::federated::plan::ClientOnlyPlan& client_plan,
      const std::string& initial_checkpoint_uri,
      const InterruptibleRunner::TimingConfig& timing_config,
      absl::Time reference_time, bool log_tensorflow_error_messages);

 private:
  // Runs the plan. Returns one of three error codes:
  // OK, INVALID_ARGUMENT, CANCELLED.
  absl::Status RunPlanInternal(
      TaskEnvironment* task_environment, Files* files, LogManager* log_manager,
      EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      const google::internal::federated::plan::ClientOnlyPlan& client_plan,
      const std::string& initial_checkpoint_uri);

  // Runs the ClientExecution specified by execution_index.
  absl::Status RunExecution(
      TaskEnvironment* task_environment, Files* files, LogManager* log_manager,
      EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      const google::internal::federated::plan::ClientOnlyPlan& client_plan,
      int execution_index, absl::Time start_time,
      std::atomic<int>* total_example_count,
      std::atomic<int64_t>* total_example_size_bytes,
      ExampleIteratorStatus* example_iterator_status);

  // Struct used to aggregate latency + counter statistics across multiple
  // epochs for telemetry purposes.
  struct ExecPerfStats {
    int example_count;
    int64_t example_size_bytes;
    absl::Duration gather_minibatch_latency;
    absl::Duration run_minibatch_latency;
    int num_minibatches;
  };

  // Runs an epoch (pass over data) during a ClientExecution.
  // execution_aborted and exec_perf_stats must be non-NULL and are modified by
  // this function to indicate if TF execution was aborted by the model, and to
  // aggregate lstency + counter stats for this epoch, respectively.
  absl::Status RunEpoch(
      TaskEnvironment* task_environment, Files* files, LogManager* log_manager,
      EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      const google::internal::federated::plan::ClientExecution& execution,
      int execution_index, absl::Time start_time, int epoch_index,
      bool* execution_aborted, int64_t total_example_size_bytes,
      ExecPerfStats* exec_perf_stats);

  // After training has finished, stage parameters and publish results.
  absl::Status FinishExecution(
      TaskEnvironment* task_environment, Files* files, LogManager* log_manager,
      EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      DataSourceType data_source_type,
      const google::internal::federated::plan::ClientExecution& execution,
      int execution_index, absl::Time start_time, int epoch_index,
      int64_t total_example_size_bytes);

  // Invokes TensorFlowWrapper, and takes care of logging TensorFlow errors and
  // external interruptions via event_publisher.
  // If the TF call fails because it got aborted externally, returns CANCELLED.
  // If the TF call reports an OUT_OF_RANGE error ("internal" abortion):
  //   If aborted is not a nullptr, sets *aborted to true, returns OK.
  //   If aborted is nullptr (no abort expected), logs error + returns
  //   INVALID_ARGUMENT.
  // If the TF call fails with an INVALID argument, indicating a TF error,
  //   publishes an event, then returns INVALID_ARGUMENT
  // If the TF call is successful, returns OK.
  absl::Status RunTensorFlowInternal(
      const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
      const std::vector<std::string>& output_tensor_names,
      const std::vector<std::string>& target_node_names,
      std::vector<tensorflow::Tensor>* outputs, EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      int execution_index, int epoch_index, int example_index,
      int64_t example_size, absl::Time start, bool* aborted = nullptr);

  // Stages parameters from the model by writing to temporary files, for
  // possible later publishing.
  absl::Status StageParameters(
      TaskEnvironment* task_environment, Files* files,
      EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      const google::internal::federated::plan::ClientExecution& execution,
      int execution_index, absl::Time start_time,
      int64_t total_example_size_bytes);

  // Retrieve + log previously computed stats from the plan. These stats may
  // trigger publishing of previously staged parameters.
  absl::Status RunStatsAndConditionallyPublish(
      TaskEnvironment* task_environment, LogManager* log_manager,
      EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      const google::internal::federated::plan::ClientExecution& execution,
      int execution_index, int epoch_index, absl::Time start_time,
      int64_t total_example_size_bytes);

  // Calls the environment's Finish function with the provided
  // phase_outcome.
  // Returns true on success, false otherwise.
  bool CallFinish(TaskEnvironment* task_environment,
                  EventPublisher* event_publisher,
                  ::fcp::client::opstats::OpStatsLogger* opstats_logger,
                  PhaseOutcome phase_outcome, absl::Time plan_start_time);

  // Performs some integrity checks on the provided plan, returns true if plan
  // looks good.
  static bool PlanIntegrityChecks(
      const google::internal::federated::plan::ClientOnlyPlan& plan);

  // Call TensorFlow with provided target node name, but do not feed any data
  // or fetch any outputs. No-op if target_node_name is empty.
  absl::Status RunTensorFlow(
      const std::string& target_node_name, EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      int execution_index, int epoch_index, int example_index,
      int64_t example_size, absl::Time start);

  // Call TensorFlow with provided target node name, but do not feed any data
  // or fetch any outputs; and measure time taken.
  // No-op if target_node_name is empty.
  absl::Status RunTensorFlow(
      const std::string& target_node_name, EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      int execution_index, int epoch_index, int example_index,
      int64_t example_size, absl::Time start, LogManager* log_manager,
      HistogramCounters histogram_counter, DataSourceType data_source_type);

  // Call TensorFlow with provided target node name, feeding the provided batch
  // of data as DT_STRING tensor to the provided feed op.
  absl::Status RunTensorFlow(
      const std::string& feed_name, const std::vector<std::string>& batch,
      const std::string& target_node_name, bool* execution_aborted,
      EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      int execution_index, int epoch_index, int example_index,
      int64_t example_size, absl::Time start);

  // Fetches stat values (indicated by fetch_names) from the graph, returning
  // them as a map<std::string, double> in output_values. The tensors being fetched
  // should be scalars - if not, only their first element is considered - and
  // will be cast to double.
  absl::Status RunStats(const std::vector<std::string>& fetch_names,
                        absl::flat_hash_map<std::string, double>* output_values,
                        EventPublisher* event_publisher,
                        ::fcp::client::opstats::OpStatsLogger* opstats_logger,
                        int execution_index, int epoch_index, int example_index,
                        int64_t example_size, absl::Time start);

  // Load TensorFlow variables from the provided checkpoint.
  absl::Status LoadState(
      const google::internal::federated::plan::CheckpointOp& checkpoint_op,
      const std::string& checkpoint_path, EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      int execution_index, int epoch_index, int example_index,
      int64_t example_size, absl::Time start, LogManager* log_manager);

  // Save TensorFlow variables to the specified files.
  absl::Status SaveState(
      const google::internal::federated::plan::CheckpointOp& checkpoint_op,
      const std::string& checkpoint_path, const std::string& secagg_path,
      EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      int execution_index, int epoch_index, int example_index,
      int64_t example_size, absl::Time start);

  // Utility function to restore variables from or save variables to a
  // checkpoint.
  absl::Status LoadOrSaveState(
      const std::string& before_op, bool has_saver_def,
      const std::string& filename_tensor_name,
      const std::string& checkpoint_path, const std::string& save_or_restore_op,
      const std::string& after_op, EventPublisher* event_publisher,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger,
      int execution_index, int epoch_index, int example_index,
      int64_t example_size, absl::Time start);

  // Logs duration between reference_time and call to this function to the
  // specified HistogramCounter.
  static void LogTimeSince(LogManager* log_manager,
                           HistogramCounters histogram_counter,
                           int execution_index, int epoch_index,
                           DataSourceType data_source_type,
                           absl::Time reference_time);

  // Py wrappers often produce tensor names instead of a node name, which
  // the session doesn't accept as target names. Remove the trailing ':0' to
  // work around this.
  static const std::string SanitizeTargetName(const std::string& node_name);

  // Converts a Tensor with integer values to a Checkpoint::Aggregand
  // for use in Secure Aggregation.
  static absl::StatusOr<
      google::internal::federatedml::v2::Checkpoint::Aggregand>
  TensorProtoToQuantizedAggregand(const tensorflow::Tensor& tensor,
                                  const std::vector<int64_t>& dimensions);

  // Helper functions for producing Secure Aggregation checkpoints.
  template <typename T>
  static absl::StatusOr<
      google::internal::federatedml::v2::Checkpoint::Aggregand>
  AddValuesToQuantized(const tensorflow::Tensor& tensor,
                       const std::vector<int64_t>& dimensions);
  static absl::StatusOr<int> TensorTypeToBitWidth(
      tensorflow::DataType datatype);

  // A class representing a staged, but not yet published result.
  // This state is necessary to allow for postponed publishing of previously
  // generated results.
  class StagedResult {
   public:
    StagedResult(std::string quality_metric_name,
                 double quality_metric_threshold, std::string checkpoint_path,
                 std::string secagg_checkpoint_path)
        : quality_metric_name_(std::move(quality_metric_name)),
          quality_metric_threshold_(quality_metric_threshold),
          checkpoint_path_(std::move(checkpoint_path)),
          secagg_checkpoint_path_(std::move(secagg_checkpoint_path)) {}

    std::string quality_metric_name_;
    double quality_metric_threshold_;
    std::string checkpoint_path_;
    std::string secagg_checkpoint_path_;
  };

  const google::internal::federatedml::v2::RetryWindow default_retry_window_;
  std::unique_ptr<TensorFlowWrapper> tf_wrapper_;
  std::optional<StagedResult> staged_result_;
  std::vector<std::pair<std::string, double>> all_published_stats_;
  bool log_tensorflow_error_messages_;
};
}  // namespace engine
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_ENGINE_PLAN_ENGINE_H_
