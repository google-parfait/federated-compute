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
#ifndef FCP_CLIENT_TASK_ENVIRONMENT_H_
#define FCP_CLIENT_TASK_ENVIRONMENT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "fcp/client/engine/engine.pb.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {

// An interface used by the plan engine to interact with its "environment".
// This interface is used to abstract away platform and plan-specific details
// such as how to publish training results and getting access to training data.
class TaskEnvironment {
 public:
  virtual ~TaskEnvironment() = default;

  // Finishes this computation by reporting previously published parameters
  // along with the provided PhaseOutcome, duration of the compute, and stats
  // reported by the plan.
  virtual absl::Status Finish(
      engine::PhaseOutcome phase_outcome, absl::Duration plan_duration,
      const std::vector<std::pair<std::string, double>>& stats) = 0;

  // Returns true if the caller (the engine) should abort computation.
  virtual bool ShouldAbort() = 0;

  // Returns an ExampleIterator, or, on error, one of the following error codes:
  // - INVALID_ARGUMENT - if the provided ExampleSelector was invalid.
  virtual absl::StatusOr<std::unique_ptr<ExampleIterator>>
  CreateExampleIterator(
      const google::internal::federated::plan::ExampleSelector&
          example_selector) = 0;

  // Publishes parameters resulting from plan execution to the environment.
  // checkpoint_file must be a TF v1 checkpoint, secagg_checkpoint_file must
  // contain a serialized Checkpoint proto (see federated_api_client.proto).
  // File ownership is transferred by the caller to the environment, and the
  // caller should not make any assumptions about the file lifetime after this
  // call.
  // On error, returns
  //  - INVALID_ARGUMENT - on "expected" errors such as I/O issues.
  // TODO(team): Once local computation is available in native code,
  // change this method and its callers to use ComputationResults instead.
  virtual absl::Status PublishParameters(
      const std::string& checkpoint_file,
      const std::string& secagg_checkpoint_file) = 0;

  // Whether stats produced by the model should be published via
  // EventPublisher::PublishStats() or not.
  // TODO(team): Remove once we have gotten rid of publishing stats
  // via EventPublisher entirely.
  virtual bool ShouldPublishStats() = 0;
};
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_TASK_ENVIRONMENT_H_
