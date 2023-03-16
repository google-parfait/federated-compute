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

#include "fcp/client/engine/example_query_plan_engine.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/client/engine/common.h"
#include "fcp/client/engine/plan_engine_helpers.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace engine {

using ::fcp::client::ExampleQueryResult;
using ::fcp::client::engine::PlanResult;
using ::fcp::client::opstats::OpStatsLogger;
using ::google::internal::federated::plan::ExampleQuerySpec;
using ::google::internal::federated::plan::ExampleSelector;

ExampleQueryPlanEngine::ExampleQueryPlanEngine(
    std::vector<ExampleIteratorFactory*> example_iterator_factories,
    OpStatsLogger* opstats_logger)
    : example_iterator_factories_(example_iterator_factories),
      opstats_logger_(opstats_logger) {}

PlanResult ExampleQueryPlanEngine::RunPlan(
    const ExampleQuerySpec& example_query_spec,
    const std::string& output_checkpoint_filename) {
  // TODO(team): Add the same logging as in simple_plan_engine.
  if (example_query_spec.example_queries_size() != 1) {
    // TODO(team): Add support of multiple example queries.
    return PlanResult(
        PlanOutcome::kInvalidArgument,
        absl::UnimplementedError("Multiple example queries not supported yet"));
  }
  ExampleSelector selector =
      example_query_spec.example_queries(0).example_selector();
  ExampleIteratorFactory* example_iterator_factory =
      FindExampleIteratorFactory(selector, example_iterator_factories_);
  if (example_iterator_factory == nullptr) {
    return PlanResult(
        PlanOutcome::kExampleIteratorError,
        absl::InternalError("Could not find suitable ExampleIteratorFactory"));
  }
  absl::StatusOr<std::unique_ptr<ExampleIterator>> example_iterator =
      example_iterator_factory->CreateExampleIterator(selector);
  if (!example_iterator.ok()) {
    return PlanResult(PlanOutcome::kExampleIteratorError,
                      example_iterator.status());
  }

  std::atomic<int> total_example_count = 0;
  std::atomic<int64_t> total_example_size_bytes = 0;
  ExampleIteratorStatus example_iterator_status;

  auto dataset_iterator = std::make_unique<DatasetIterator>(
      std::move(*example_iterator), opstats_logger_, &total_example_count,
      &total_example_size_bytes, &example_iterator_status,
      selector.collection_uri(),
      /*collect_stats=*/example_iterator_factory->ShouldCollectStats());

  absl::StatusOr<std::string> example_query_result_str =
      dataset_iterator->GetNext();
  if (!example_query_result_str.ok()) {
    return PlanResult(PlanOutcome::kExampleIteratorError,
                      example_query_result_str.status());
  }

  ExampleQueryResult example_query_result;
  if (!example_query_result.ParseFromString(*example_query_result_str)) {
    return PlanResult(
        PlanOutcome::kExampleIteratorError,
        absl::DataLossError("Unexpected example query result format"));
  }
  // TODO(team): Write example query result into the checkpoint file.
  return PlanResult(PlanOutcome::kSuccess, absl::OkStatus());
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
