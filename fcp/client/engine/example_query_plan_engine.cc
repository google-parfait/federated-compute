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
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/engine/common.h"
#include "fcp/client/engine/plan_engine_helpers.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/tensorflow/status.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace fcp {
namespace client {
namespace engine {

namespace tf = ::tensorflow;

using ::fcp::client::ExampleQueryResult;
using ::fcp::client::engine::PlanResult;
using ::fcp::client::opstats::OpStatsLogger;
using ::google::internal::federated::plan::ExampleQuerySpec;
using ::google::internal::federated::plan::ExampleSelector;

namespace {

// Writes an one-dimensional tensor using the slice writer.
template <typename T>
absl::Status WriteSlice(tf::checkpoint::TensorSliceWriter& slice_writer,
                        const std::string& name, const int64_t size,
                        const T* data) {
  tf::TensorShape shape;
  shape.AddDim(size);
  tf::TensorSlice slice(shape.dims());
  tf::Status tf_status = slice_writer.Add(name, shape, slice, data);
  return ConvertFromTensorFlowStatus(tf_status);
}

// Returns a map of (vector name) -> tuple(output name, vector spec).
absl::flat_hash_map<std::string,
                    std::tuple<std::string, ExampleQuerySpec::OutputVectorSpec>>
GetOutputVectorSpecs(const ExampleQuerySpec::ExampleQuery& example_query) {
  absl::flat_hash_map<
      std::string, std::tuple<std::string, ExampleQuerySpec::OutputVectorSpec>>
      map;
  for (auto const& [output_name, output_vector_spec] :
       example_query.output_vector_specs()) {
    map[output_vector_spec.vector_name()] =
        std::make_tuple(output_name, output_vector_spec);
  }
  return map;
}

absl::Status CheckOutputVectorDataType(
    const ExampleQuerySpec::OutputVectorSpec& output_vector_spec,
    const ExampleQuerySpec::OutputVectorSpec::DataType& expected_data_type) {
  if (output_vector_spec.data_type() != expected_data_type) {
    return absl::FailedPreconditionError(
        "Unexpected data type in the example query");
  }
  return absl::OkStatus();
}

// Writes example query results into a checkpoint. Example query results order
// must be the same as example_query_spec.example_queries.
absl::Status WriteCheckpoint(
    const std::string& output_checkpoint_filename,
    const std::vector<ExampleQueryResult>& example_query_results,
    const ExampleQuerySpec& example_query_spec) {
  tf::checkpoint::TensorSliceWriter slice_writer(
      output_checkpoint_filename,
      tf::checkpoint::CreateTableTensorSliceBuilder);
  for (int i = 0; i < example_query_results.size(); ++i) {
    const ExampleQueryResult& example_query_result = example_query_results[i];
    const ExampleQuerySpec::ExampleQuery& example_query =
        example_query_spec.example_queries()[i];
    for (auto const& [vector_name, vector_tuple] :
         GetOutputVectorSpecs(example_query)) {
      std::string output_name = std::get<0>(vector_tuple);
      ExampleQuerySpec::OutputVectorSpec output_vector_spec =
          std::get<1>(vector_tuple);
      auto it = example_query_result.vector_data().vectors().find(vector_name);
      if (it == example_query_result.vector_data().vectors().end()) {
        return absl::DataLossError(
            "Expected value not found in the example query result");
      }
      const ExampleQueryResult::VectorData::Values values = it->second;
      absl::Status status;
      if (values.has_int32_values()) {
        FCP_RETURN_IF_ERROR(CheckOutputVectorDataType(
            output_vector_spec, ExampleQuerySpec::OutputVectorSpec::INT32));
        int64_t size = values.int32_values().value_size();
        auto data =
            static_cast<const int32_t*>(values.int32_values().value().data());
        FCP_RETURN_IF_ERROR(WriteSlice(slice_writer, output_name, size, data));
      } else if (values.has_int64_values()) {
        FCP_RETURN_IF_ERROR(CheckOutputVectorDataType(
            output_vector_spec, ExampleQuerySpec::OutputVectorSpec::INT64));
        int64_t size = values.int64_values().value_size();
        auto data =
            static_cast<const int64_t*>(values.int64_values().value().data());
        FCP_RETURN_IF_ERROR(WriteSlice(slice_writer, output_name, size, data));
      } else if (values.has_string_values()) {
        FCP_RETURN_IF_ERROR(CheckOutputVectorDataType(
            output_vector_spec, ExampleQuerySpec::OutputVectorSpec::STRING));
        int64_t size = values.string_values().value_size();
        std::vector<tf::tstring> tf_string_vector;
        for (const auto& value : values.string_values().value()) {
          tf_string_vector.emplace_back(value);
        }
        FCP_RETURN_IF_ERROR(WriteSlice(slice_writer, output_name, size,
                                       tf_string_vector.data()));
      } else if (values.has_bool_values()) {
        FCP_RETURN_IF_ERROR(CheckOutputVectorDataType(
            output_vector_spec, ExampleQuerySpec::OutputVectorSpec::BOOL));
        int64_t size = values.bool_values().value_size();
        auto data =
            static_cast<const bool*>(values.bool_values().value().data());
        FCP_RETURN_IF_ERROR(WriteSlice(slice_writer, output_name, size, data));
      } else if (values.has_float_values()) {
        FCP_RETURN_IF_ERROR(CheckOutputVectorDataType(
            output_vector_spec, ExampleQuerySpec::OutputVectorSpec::FLOAT));
        int64_t size = values.float_values().value_size();
        auto data =
            static_cast<const float*>(values.float_values().value().data());
        FCP_RETURN_IF_ERROR(WriteSlice(slice_writer, output_name, size, data));
      } else if (values.has_double_values()) {
        FCP_RETURN_IF_ERROR(CheckOutputVectorDataType(
            output_vector_spec, ExampleQuerySpec::OutputVectorSpec::DOUBLE));
        int64_t size = values.double_values().value_size();
        auto data =
            static_cast<const double*>(values.double_values().value().data());
        FCP_RETURN_IF_ERROR(WriteSlice(slice_writer, output_name, size, data));
      } else if (values.has_bytes_values()) {
        FCP_RETURN_IF_ERROR(CheckOutputVectorDataType(
            output_vector_spec, ExampleQuerySpec::OutputVectorSpec::BYTES));
        int64_t size = values.bytes_values().value_size();
        std::vector<tf::tstring> tf_string_vector;
        for (const auto& value : values.string_values().value()) {
          tf_string_vector.emplace_back(value);
        }
        FCP_RETURN_IF_ERROR(WriteSlice(slice_writer, output_name, size,
                                       tf_string_vector.data()));
      } else {
        return absl::DataLossError(
            "Unexpected data type in the example query result");
      }
    }
  }
  return ConvertFromTensorFlowStatus(slice_writer.Finish());
}

}  // anonymous namespace

ExampleQueryPlanEngine::ExampleQueryPlanEngine(
    std::vector<ExampleIteratorFactory*> example_iterator_factories,
    OpStatsLogger* opstats_logger)
    : example_iterator_factories_(example_iterator_factories),
      opstats_logger_(opstats_logger) {}

PlanResult ExampleQueryPlanEngine::RunPlan(
    const ExampleQuerySpec& example_query_spec,
    const std::string& output_checkpoint_filename) {
  // TODO(team): Add the same logging as in simple_plan_engine.
  std::vector<ExampleQueryResult> example_query_results;

  for (const auto& example_query : example_query_spec.example_queries()) {
    ExampleSelector selector = example_query.example_selector();
    ExampleIteratorFactory* example_iterator_factory =
        FindExampleIteratorFactory(selector, example_iterator_factories_);
    if (example_iterator_factory == nullptr) {
      return PlanResult(PlanOutcome::kExampleIteratorError,
                        absl::InternalError(
                            "Could not find suitable ExampleIteratorFactory"));
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
    example_query_results.push_back(std::move(example_query_result));
  }
  absl::Status status = WriteCheckpoint(
      output_checkpoint_filename, example_query_results, example_query_spec);
  if (!status.ok()) {
    return PlanResult(PlanOutcome::kExampleIteratorError, status);
  }
  return PlanResult(PlanOutcome::kSuccess, absl::OkStatus());
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
