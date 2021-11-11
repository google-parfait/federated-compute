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
#ifndef FCP_CLIENT_FEDERATED_TASK_ENVIRONMENT_H_
#define FCP_CLIENT_FEDERATED_TASK_ENVIRONMENT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/federated_protocol.h"
#include "fcp/client/flags.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/client/task_environment.h"

namespace fcp {
namespace client {

class FederatedTaskEnvironment : public TaskEnvironment {
 public:
  ~FederatedTaskEnvironment() override;

  // Prod constructor.
  FederatedTaskEnvironment(SimpleTaskEnvironment* env_deps,
                           FederatedProtocol* federated_protocol,
                           LogManager* log_manager,
                           EventPublisher* event_publisher,
                           opstats::OpStatsLogger* opstats_logger,
                           const Flags* flags, absl::Time reference_time,
                           absl::Duration condition_polling_period);

  // Test constructor to allow injection of a mock time source.
  FederatedTaskEnvironment(SimpleTaskEnvironment* env_deps,
                           FederatedProtocol* federated_protocol,
                           LogManager* log_manager,
                           EventPublisher* event_publisher,
                           opstats::OpStatsLogger* opstats_logger,
                           const Flags* flags, absl::Time reference_time,
                           std::function<absl::Time()> get_time_fn,
                           absl::Duration condition_polling_period);

  absl::Status Finish(
      engine::PhaseOutcome phase_outcome, absl::Duration plan_duration,
      const std::vector<std::pair<std::string, double>>& stats) override;

  bool ShouldAbort() override;

  absl::StatusOr<std::unique_ptr<ExampleIterator>> CreateExampleIterator(
      const google::internal::federated::plan::ExampleSelector&
          example_selector) override;

  absl::Status PublishParameters(
      const std::string& tf_checkpoint_file,
      const std::string& secagg_checkpoint_file) override;

  bool ShouldPublishStats() override;

 private:
  // Combines the published SecAgg and TensorFlow checkpoints, if any, into one
  // ComputationResults object for reporting to the server.
  // @return A ComputationResults object containing either the combined SecAgg
  //         and TF checkpoint, a default object with an empty TF checkpoint if
  //         no SecAgg/TF checkpoints have been published, or INVALID_ARGUMENT
  //         on parser and I/O errors.
  absl::StatusOr<ComputationResults> CreateComputationResults() const;

  // Delete staged files from previous calls to PublishParameters(), if any.
  absl::Status DeleteStagedFiles();

  absl::Status LogReportStart();

  void LogReportFinish(int64_t report_request_size,
                       int64_t chunking_layers_sent_bytes,
                       absl::Duration upload_time);

  SimpleTaskEnvironment* const env_deps_;
  FederatedProtocol* const federated_protocol_;
  LogManager* const log_manager_;
  EventPublisher* const event_publisher_;
  opstats::OpStatsLogger* const opstats_logger_;
  const Flags* const flags_;
  const absl::Time reference_time_;
  std::string tf_checkpoint_file_;
  std::string secagg_checkpoint_file_;
  std::function<absl::Time()> get_time_fn_;
  absl::Duration condition_polling_period_;
};

}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_FEDERATED_TASK_ENVIRONMENT_H_
