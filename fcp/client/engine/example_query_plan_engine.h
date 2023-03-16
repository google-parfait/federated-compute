/*
 * Copyright 2023 Google LLC
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
#ifndef FCP_CLIENT_ENGINE_EXAMPLE_QUERY_PLAN_ENGINE_H_
#define FCP_CLIENT_ENGINE_EXAMPLE_QUERY_PLAN_ENGINE_H_

#include <string>
#include <vector>

#include "fcp/client/engine/common.h"
#include "fcp/client/engine/example_iterator_factory.h"
#include "fcp/client/opstats/opstats_logger.h"

namespace fcp {
namespace client {
namespace engine {

// A class used to "run" (interpret) an ExampleQuerySpec-based plan. Each
// instance should generally only be used once to run a plan.
class ExampleQueryPlanEngine {
 public:
  ExampleQueryPlanEngine(
      std::vector<ExampleIteratorFactory*> example_iterator_factories,
      ::fcp::client::opstats::OpStatsLogger* opstats_logger);

  // Runs a plan and writes an output into a checkpoint at the given path.
  ::fcp::client::engine::PlanResult RunPlan(
      const google::internal::federated::plan::ExampleQuerySpec&
          example_query_spec,
      const std::string& output_checkpoint_filename);

 private:
  std::vector<ExampleIteratorFactory*> example_iterator_factories_;
  ::fcp::client::opstats::OpStatsLogger* opstats_logger_;
};
}  // namespace engine
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_ENGINE_EXAMPLE_QUERY_PLAN_ENGINE_H_
