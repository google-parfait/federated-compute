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

#include <fcntl.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "fcp/client/client_runner.h"
#include "fcp/client/engine/common.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/plan.pb.h"

namespace fcp {
namespace client {
namespace engine {
namespace {

using ::fcp::client::ExampleQueryResult;
using ::google::internal::federated::plan::AggregationConfig;
using ::google::internal::federated::plan::ClientOnlyPlan;
using ::google::internal::federated::plan::Dataset;
using ::google::internal::federated::plan::ExampleQuerySpec;
using ::google::internal::federated::plan::ExampleSelector;
using ::testing::StrictMock;

const char* const kCollectionUri = "app:/test_collection";
const char* const kOutputStringVectorName = "vector1";
const char* const kOutputIntVectorName = "vector2";

class InvalidExampleIteratorFactory : public ExampleIteratorFactory {
 public:
  InvalidExampleIteratorFactory() = default;

  bool CanHandle(const google::internal::federated::plan::ExampleSelector&
                     example_selector) override {
    return false;
  }

  absl::StatusOr<std::unique_ptr<ExampleIterator>> CreateExampleIterator(
      const ExampleSelector& example_selector) override {
    absl::Status error(absl::StatusCode::kInternal, "");
    return error;
  }

  bool ShouldCollectStats() override { return false; }
};

class NoIteratorExampleIteratorFactory : public ExampleIteratorFactory {
 public:
  NoIteratorExampleIteratorFactory() = default;

  bool CanHandle(const google::internal::federated::plan::ExampleSelector&
                     example_selector) override {
    return true;
  }

  absl::StatusOr<std::unique_ptr<ExampleIterator>> CreateExampleIterator(
      const ExampleSelector& example_selector) override {
    absl::Status error(absl::StatusCode::kInternal, "");
    return error;
  }

  bool ShouldCollectStats() override { return false; }
};

class ExampleQueryPlanEngineTest : public testing::Test {
 protected:
  void Initialize() {
    ExampleQuerySpec::OutputVectorSpec string_vector_spec;
    string_vector_spec.set_vector_name(kOutputStringVectorName);
    string_vector_spec.set_data_type(
        ExampleQuerySpec::OutputVectorSpec::STRING);
    ExampleQuerySpec::OutputVectorSpec int_vector_spec;
    int_vector_spec.set_vector_name(kOutputIntVectorName);
    int_vector_spec.set_data_type(ExampleQuerySpec::OutputVectorSpec::INT64);

    ExampleQuerySpec::ExampleQuery example_query;
    example_query.mutable_example_selector()->set_collection_uri(
        kCollectionUri);
    (*example_query.mutable_output_vector_specs())[kOutputStringVectorName] =
        string_vector_spec;
    (*example_query.mutable_output_vector_specs())[kOutputIntVectorName] =
        int_vector_spec;
    client_only_plan_.mutable_phase()
        ->mutable_example_query_spec()
        ->mutable_example_queries()
        ->Add(std::move(example_query));

    AggregationConfig aggregation_config;
    aggregation_config.mutable_tf_v1_checkpoint_aggregation();
    (*client_only_plan_.mutable_phase()
          ->mutable_federated_example_query()
          ->mutable_aggregations())[kOutputStringVectorName] =
        aggregation_config;
    (*client_only_plan_.mutable_phase()
          ->mutable_federated_example_query()
          ->mutable_aggregations())[kOutputIntVectorName] = aggregation_config;

    ExampleQueryResult example_query_result;
    ExampleQueryResult::VectorData::Values int_values;
    int_values.mutable_int64_values()->add_value(42);
    int_values.mutable_int64_values()->add_value(24);
    (*example_query_result.mutable_vector_data()
          ->mutable_vectors())[kOutputIntVectorName] = int_values;
    ExampleQueryResult::VectorData::Values string_values;
    string_values.mutable_string_values()->add_value("value1");
    string_values.mutable_string_values()->add_value("value2");
    (*example_query_result.mutable_vector_data()
          ->mutable_vectors())[kOutputStringVectorName] = string_values;
    std::string example = example_query_result.SerializeAsString();

    Dataset::ClientDataset client_dataset;
    client_dataset.set_client_id("client_id");
    client_dataset.add_example(example);
    dataset_.mutable_client_data()->Add(std::move(client_dataset));

    num_examples_ = 1;
    example_bytes_ = example.size();

    example_iterator_factory_ =
        std::make_unique<FunctionalExampleIteratorFactory>(
            [&dataset = dataset_](
                const google::internal::federated::plan::ExampleSelector&
                    selector) {
              return std::make_unique<SimpleExampleIterator>(dataset);
            });
  }

  fcp::client::FilesImpl files_impl_;
  StrictMock<MockOpStatsLogger> mock_opstats_logger_;
  std::unique_ptr<ExampleIteratorFactory> example_iterator_factory_;

  ClientOnlyPlan client_only_plan_;
  Dataset dataset_;
  std::string checkpoint_output_filename_;

  int num_examples_ = 0;
  int64_t example_bytes_ = 0;
};

TEST_F(ExampleQueryPlanEngineTest, PlanSucceeds) {
  Initialize();

  EXPECT_CALL(
      mock_opstats_logger_,
      UpdateDatasetStats(kCollectionUri, num_examples_, example_bytes_));

  ExampleQueryPlanEngine plan_engine({example_iterator_factory_.get()},
                                     &mock_opstats_logger_);
  engine::PlanResult result =
      plan_engine.RunPlan(client_only_plan_.phase().example_query_spec(),
                          checkpoint_output_filename_);

  EXPECT_THAT(result.outcome, PlanOutcome::kSuccess);
}

TEST_F(ExampleQueryPlanEngineTest, MultipleQueries) {
  Initialize();

  ExampleQuerySpec::ExampleQuery example_query1;
  example_query1.mutable_example_selector()->set_collection_uri("collection1");
  ExampleQuerySpec::ExampleQuery example_query2;
  example_query2.mutable_example_selector()->set_collection_uri("collection2");
  client_only_plan_.mutable_phase()
      ->mutable_example_query_spec()
      ->mutable_example_queries()
      ->Add(std::move(example_query1));
  client_only_plan_.mutable_phase()
      ->mutable_example_query_spec()
      ->mutable_example_queries()
      ->Add(std::move(example_query2));

  ExampleQueryPlanEngine plan_engine({example_iterator_factory_.get()},
                                     &mock_opstats_logger_);
  engine::PlanResult result =
      plan_engine.RunPlan(client_only_plan_.phase().example_query_spec(),
                          checkpoint_output_filename_);

  EXPECT_THAT(result.outcome, PlanOutcome::kInvalidArgument);
}

TEST_F(ExampleQueryPlanEngineTest, FactoryNotFound) {
  Initialize();
  auto invalid_example_factory =
      std::make_unique<InvalidExampleIteratorFactory>();

  ExampleQueryPlanEngine plan_engine({invalid_example_factory.get()},
                                     &mock_opstats_logger_);
  engine::PlanResult result =
      plan_engine.RunPlan(client_only_plan_.phase().example_query_spec(),
                          checkpoint_output_filename_);

  EXPECT_THAT(result.outcome, PlanOutcome::kExampleIteratorError);
}

TEST_F(ExampleQueryPlanEngineTest, NoIteratorCreated) {
  Initialize();
  auto invalid_example_factory =
      std::make_unique<NoIteratorExampleIteratorFactory>();

  ExampleQueryPlanEngine plan_engine({invalid_example_factory.get()},
                                     &mock_opstats_logger_);
  engine::PlanResult result =
      plan_engine.RunPlan(client_only_plan_.phase().example_query_spec(),
                          checkpoint_output_filename_);

  EXPECT_THAT(result.outcome, PlanOutcome::kExampleIteratorError);
}

TEST_F(ExampleQueryPlanEngineTest, InvalidExampleQueryResultFormat) {
  Initialize();
  std::string invalid_example = "invalid_example";
  Dataset::ClientDataset client_dataset;
  client_dataset.add_example(invalid_example);
  dataset_.clear_client_data();
  dataset_.mutable_client_data()->Add(std::move(client_dataset));
  example_iterator_factory_ =
      std::make_unique<FunctionalExampleIteratorFactory>(
          [&dataset = dataset_](
              const google::internal::federated::plan::ExampleSelector&
                  selector) {
            return std::make_unique<SimpleExampleIterator>(dataset);
          });
  EXPECT_CALL(mock_opstats_logger_,
              UpdateDatasetStats(kCollectionUri, 1, invalid_example.size()));

  ExampleQueryPlanEngine plan_engine({example_iterator_factory_.get()},
                                     &mock_opstats_logger_);
  engine::PlanResult result =
      plan_engine.RunPlan(client_only_plan_.phase().example_query_spec(),
                          checkpoint_output_filename_);

  EXPECT_THAT(result.outcome, PlanOutcome::kExampleIteratorError);
}

}  // anonymous namespace
}  // namespace engine
}  // namespace client
}  // namespace fcp
