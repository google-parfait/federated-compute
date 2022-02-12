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
#ifndef FCP_CLIENT_ENGINE_TFLITE_PLAN_ENGINE_H_
#define FCP_CLIENT_ENGINE_TFLITE_PLAN_ENGINE_H_

#include <functional>
#include <string>

#include "fcp/client/engine/common.h"
#include "fcp/client/event_publisher.h"
#include "fcp/client/flags.h"
#include "fcp/client/interruptible_runner.h"
#include "fcp/client/log_manager.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/simple_task_environment.h"

namespace fcp {
namespace client {
namespace engine {

// A class used to "run" (interpret) a TensorflowSpec-based plan with TfLite.
// Each instance should generally only be used once to run a plan.
class TfLitePlanEngine {
 public:
  TfLitePlanEngine(SimpleTaskEnvironment* task_env, LogManager* log_manager,
                   EventPublisher* event_publisher,
                   ::fcp::client::opstats::OpStatsLogger* opstats_logger,
                   const InterruptibleRunner::TimingConfig* timing_config,
                   const Flags* flags)
      : task_env_(task_env),
        log_manager_(log_manager),
        event_publisher_(event_publisher),
        opstats_logger_(opstats_logger),
        timing_config_(timing_config),
        flags_(flags) {}

  // Runs the plan, and takes care of logging TfLite errors and external
  // interruptions via event_publisher. If the TfLite call fails because it got
  // aborted externally, returns CANCELLED. If the TfLite call fails because of
  // other reasons, publishes an event, then returns INVALID_ARGUMENT. If the
  // TfLite call is successful, returns OK, and the output tensors.
  PlanResult RunPlan(
      const google::internal::federated::plan::TensorflowSpec& tensorflow_spec,
      const std::string& model,
      std::unique_ptr<absl::flat_hash_map<std::string, std::string>> inputs,
      const std::vector<std::string>& output_names,
      absl::Time run_plan_start_time, absl::Time reference_time,
      std::function<void()> log_computation_started,
      std::function<void()> log_computation_finished,
      const SelectorContext& selector_context);

 private:
  // Logs duration between reference_time and call to this function to the
  // specified HistogramCounter.
  void LogTimeSince(HistogramCounters histogram_counter,
                    absl::Time reference_time);

  SimpleTaskEnvironment* task_env_;
  LogManager* log_manager_;
  EventPublisher* event_publisher_;
  ::fcp::client::opstats::OpStatsLogger* opstats_logger_;
  const InterruptibleRunner::TimingConfig* timing_config_;
  const Flags* flags_;
};

}  // namespace engine
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_ENGINE_TFLITE_PLAN_ENGINE_H_
