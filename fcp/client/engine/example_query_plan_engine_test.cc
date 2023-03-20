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

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "fcp/client/client_runner.h"
#include "fcp/client/engine/common.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/testing/testing.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace fcp {
namespace client {
namespace engine {
namespace {

namespace tf = ::tensorflow;

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
const char* const kOutputStringTensorName = "tensor1";
const char* const kOutputIntTensorName = "tensor2";

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

class TwoExampleIteratorsFactory : public ExampleIteratorFactory {
 public:
  explicit TwoExampleIteratorsFactory(
      std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
          const google::internal::federated::plan::ExampleSelector&

          )>
          create_first_iterator_func,
      std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
          const google::internal::federated::plan::ExampleSelector&

          )>
          create_second_iterator_func,
      const std::string& first_collection_uri,
      const std::string& second_collection_uri)
      : create_first_iterator_func_(create_first_iterator_func),
        create_second_iterator_func_(create_second_iterator_func),
        first_collection_uri_(first_collection_uri),
        second_collection_uri_(second_collection_uri) {}

  bool CanHandle(const google::internal::federated::plan::ExampleSelector&
                     example_selector) override {
    return true;
  }

  absl::StatusOr<std::unique_ptr<ExampleIterator>> CreateExampleIterator(
      const google::internal::federated::plan::ExampleSelector&
          example_selector) override {
    if (example_selector.collection_uri() == first_collection_uri_) {
      return create_first_iterator_func_(example_selector);
    } else if (example_selector.collection_uri() == second_collection_uri_) {
      return create_second_iterator_func_(example_selector);
    }
    return absl::InvalidArgumentError("Unknown collection URI");
  }

  bool ShouldCollectStats() override { return false; }

 private:
  std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
      const google::internal::federated::plan::ExampleSelector&)>
      create_first_iterator_func_;
  std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
      const google::internal::federated::plan::ExampleSelector&)>
      create_second_iterator_func_;
  std::string first_collection_uri_;
  std::string second_collection_uri_;
};

absl::StatusOr<absl::flat_hash_map<std::string, tf::Tensor>> ReadTensors(
    std::string checkpoint_path) {
  absl::flat_hash_map<std::string, tf::Tensor> tensors;
  tf::TF_StatusPtr tf_status(TF_NewStatus());
  tf::checkpoint::CheckpointReader tf_checkpoint_reader(checkpoint_path,
                                                        tf_status.get());
  if (TF_GetCode(tf_status.get()) != TF_OK) {
    return absl::NotFoundError("Couldn't read an input checkpoint");
  }
  for (const auto& [name, tf_dtype] :
       tf_checkpoint_reader.GetVariableToDataTypeMap()) {
    std::unique_ptr<tf::Tensor> tensor;
    tf_checkpoint_reader.GetTensor(name, &tensor, tf_status.get());
    if (TF_GetCode(tf_status.get()) != TF_OK) {
      return absl::NotFoundError(
          absl::StrFormat("Checkpoint doesn't have tensor %s", name));
    }
    tensors[name] = *tensor;
  }

  return tensors;
}

class ExampleQueryPlanEngineTest : public testing::Test {
 protected:
  void Initialize() {
    std::filesystem::path root_dir(testing::TempDir());
    std::filesystem::path output_path = root_dir / std::string("output.ckpt");
    output_checkpoint_filename_ = output_path.string();

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
    (*example_query.mutable_output_vector_specs())[kOutputStringTensorName] =
        string_vector_spec;
    (*example_query.mutable_output_vector_specs())[kOutputIntTensorName] =
        int_vector_spec;
    client_only_plan_.mutable_phase()
        ->mutable_example_query_spec()
        ->mutable_example_queries()
        ->Add(std::move(example_query));

    AggregationConfig aggregation_config;
    aggregation_config.mutable_tf_v1_checkpoint_aggregation();
    (*client_only_plan_.mutable_phase()
          ->mutable_federated_example_query()
          ->mutable_aggregations())[kOutputStringTensorName] =
        aggregation_config;
    (*client_only_plan_.mutable_phase()
          ->mutable_federated_example_query()
          ->mutable_aggregations())[kOutputIntTensorName] = aggregation_config;

    ExampleQueryResult::VectorData::Values int_values;
    int_values.mutable_int64_values()->add_value(42);
    int_values.mutable_int64_values()->add_value(24);
    (*example_query_result_.mutable_vector_data()
          ->mutable_vectors())[kOutputIntVectorName] = int_values;
    ExampleQueryResult::VectorData::Values string_values;
    string_values.mutable_string_values()->add_value("value1");
    string_values.mutable_string_values()->add_value("value2");
    (*example_query_result_.mutable_vector_data()
          ->mutable_vectors())[kOutputStringVectorName] = string_values;
    std::string example = example_query_result_.SerializeAsString();

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

  ExampleQueryResult example_query_result_;
  ClientOnlyPlan client_only_plan_;
  Dataset dataset_;
  std::string output_checkpoint_filename_;

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
                          output_checkpoint_filename_);

  EXPECT_THAT(result.outcome, PlanOutcome::kSuccess);

  auto tensors = ReadTensors(output_checkpoint_filename_);
  ASSERT_OK(tensors);
  tf::Tensor int_tensor = tensors.value()[kOutputIntTensorName];
  ASSERT_EQ(int_tensor.shape(), tf::TensorShape({2}));
  ASSERT_EQ(int_tensor.dtype(), tf::DT_INT64);
  auto int_data = static_cast<int64_t*>(int_tensor.data());
  std::vector<int64_t> expected_int_data({42, 24});
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(int_data[i], expected_int_data[i]);
  }

  tf::Tensor string_tensor = tensors.value()[kOutputStringTensorName];
  ASSERT_EQ(string_tensor.shape(), tf::TensorShape({2}));
  ASSERT_EQ(string_tensor.dtype(), tf::DT_STRING);
  auto string_data = static_cast<tf::tstring*>(string_tensor.data());
  std::vector<std::string> expected_string_data({"value1", "value2"});
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(static_cast<std::string>(string_data[i]),
              expected_string_data[i]);
  }
}

TEST_F(ExampleQueryPlanEngineTest, MultipleQueries) {
  Initialize();

  ExampleQuerySpec::OutputVectorSpec float_vector_spec;
  float_vector_spec.set_vector_name("float_vector");
  float_vector_spec.set_data_type(ExampleQuerySpec::OutputVectorSpec::FLOAT);
  ExampleQuerySpec::OutputVectorSpec string_vector_spec;
  // Same vector name as in the other ExampleQuery, but with a different output
  // one to make sure these vectors are distinguished in
  // example_query_plan_engine.
  string_vector_spec.set_vector_name(kOutputStringVectorName);
  string_vector_spec.set_data_type(ExampleQuerySpec::OutputVectorSpec::STRING);

  ExampleQuerySpec::ExampleQuery second_example_query;
  second_example_query.mutable_example_selector()->set_collection_uri(
      "app:/second_collection");
  (*second_example_query.mutable_output_vector_specs())["float_tensor"] =
      float_vector_spec;
  (*second_example_query
        .mutable_output_vector_specs())["another_string_tensor"] =
      string_vector_spec;
  client_only_plan_.mutable_phase()
      ->mutable_example_query_spec()
      ->mutable_example_queries()
      ->Add(std::move(second_example_query));

  AggregationConfig aggregation_config;
  aggregation_config.mutable_tf_v1_checkpoint_aggregation();
  (*client_only_plan_.mutable_phase()
        ->mutable_federated_example_query()
        ->mutable_aggregations())["float_tensor"] = aggregation_config;

  ExampleQueryResult second_example_query_result;
  ExampleQueryResult::VectorData::Values float_values;
  float_values.mutable_float_values()->add_value(0.24f);
  float_values.mutable_float_values()->add_value(0.42f);
  float_values.mutable_float_values()->add_value(0.33f);
  ExampleQueryResult::VectorData::Values string_values;
  string_values.mutable_string_values()->add_value("another_string_value");
  (*second_example_query_result.mutable_vector_data()
        ->mutable_vectors())["float_vector"] = float_values;
  (*second_example_query_result.mutable_vector_data()
        ->mutable_vectors())[kOutputStringVectorName] = string_values;
  std::string example = second_example_query_result.SerializeAsString();

  Dataset::ClientDataset dataset;
  dataset.set_client_id("second_client_id");
  dataset.add_example(example);
  Dataset second_dataset;
  second_dataset.mutable_client_data()->Add(std::move(dataset));

  example_iterator_factory_ = std::make_unique<TwoExampleIteratorsFactory>(
      [&dataset = dataset_](
          const google::internal::federated::plan::ExampleSelector& selector) {
        return std::make_unique<SimpleExampleIterator>(dataset);
      },
      [&dataset = second_dataset](
          const google::internal::federated::plan::ExampleSelector& selector) {
        return std::make_unique<SimpleExampleIterator>(dataset);
      },
      kCollectionUri, "app:/second_collection");

  ExampleQueryPlanEngine plan_engine({example_iterator_factory_.get()},
                                     &mock_opstats_logger_);
  engine::PlanResult result =
      plan_engine.RunPlan(client_only_plan_.phase().example_query_spec(),
                          output_checkpoint_filename_);

  EXPECT_THAT(result.outcome, PlanOutcome::kSuccess);

  auto tensors = ReadTensors(output_checkpoint_filename_);
  ASSERT_OK(tensors);
  tf::Tensor int_tensor = tensors.value()[kOutputIntTensorName];
  ASSERT_EQ(int_tensor.shape(), tf::TensorShape({2}));
  ASSERT_EQ(int_tensor.dtype(), tf::DT_INT64);
  auto int_data = static_cast<int64_t*>(int_tensor.data());
  std::vector<int64_t> expected_int_data({42, 24});
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(int_data[i], expected_int_data[i]);
  }

  tf::Tensor string_tensor = tensors.value()[kOutputStringTensorName];
  ASSERT_EQ(string_tensor.shape(), tf::TensorShape({2}));
  ASSERT_EQ(string_tensor.dtype(), tf::DT_STRING);
  auto string_data = static_cast<tf::tstring*>(string_tensor.data());
  std::vector<std::string> expected_string_data({"value1", "value2"});
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(static_cast<std::string>(string_data[i]),
              expected_string_data[i]);
  }

  tf::Tensor float_tensor = tensors.value()["float_tensor"];
  ASSERT_EQ(float_tensor.shape(), tf::TensorShape({3}));
  ASSERT_EQ(float_tensor.dtype(), tf::DT_FLOAT);
  auto float_data = static_cast<float*>(float_tensor.data());
  std::vector<float> expected_float_data({0.24f, 0.42f, 0.33f});
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(float_data[i], expected_float_data[i]);
  }

  tf::Tensor second_query_string_tensor =
      tensors.value()["another_string_tensor"];
  ASSERT_EQ(second_query_string_tensor.shape(), tf::TensorShape({1}));
  ASSERT_EQ(second_query_string_tensor.dtype(), tf::DT_STRING);
  auto second_query_string_data =
      static_cast<tf::tstring*>(second_query_string_tensor.data());
  ASSERT_EQ(static_cast<std::string>(*second_query_string_data),
            "another_string_value");
}

TEST_F(ExampleQueryPlanEngineTest, OutputVectorSpecMissingInResult) {
  Initialize();

  ExampleQuerySpec::OutputVectorSpec new_vector_spec;
  new_vector_spec.set_vector_name("new_vector");
  new_vector_spec.set_data_type(ExampleQuerySpec::OutputVectorSpec::DOUBLE);

  ExampleQuerySpec::ExampleQuery example_query =
      client_only_plan_.phase().example_query_spec().example_queries().at(0);
  (*example_query.mutable_output_vector_specs())["new_tensor"] =
      new_vector_spec;
  client_only_plan_.mutable_phase()
      ->mutable_example_query_spec()
      ->clear_example_queries();
  client_only_plan_.mutable_phase()
      ->mutable_example_query_spec()
      ->mutable_example_queries()
      ->Add(std::move(example_query));

  ExampleQueryResult example_query_result;
  ExampleQueryResult::VectorData::Values bool_values;
  bool_values.mutable_bool_values()->add_value(true);
  (*example_query_result_.mutable_vector_data()
        ->mutable_vectors())["new_vector"] = bool_values;
  std::string example = example_query_result_.SerializeAsString();

  Dataset::ClientDataset client_dataset;
  client_dataset.set_client_id("client_id");
  client_dataset.add_example(example);
  dataset_.clear_client_data();
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

  EXPECT_CALL(
      mock_opstats_logger_,
      UpdateDatasetStats(kCollectionUri, num_examples_, example_bytes_));

  ExampleQueryPlanEngine plan_engine({example_iterator_factory_.get()},
                                     &mock_opstats_logger_);
  engine::PlanResult result =
      plan_engine.RunPlan(client_only_plan_.phase().example_query_spec(),
                          output_checkpoint_filename_);

  EXPECT_THAT(result.outcome, PlanOutcome::kExampleIteratorError);
}

TEST_F(ExampleQueryPlanEngineTest, OutputVectorSpecTypeMismatch) {
  Initialize();

  ExampleQuerySpec::OutputVectorSpec new_vector_spec;
  new_vector_spec.set_vector_name("new_vector");
  new_vector_spec.set_data_type(ExampleQuerySpec::OutputVectorSpec::DOUBLE);

  ExampleQuerySpec::ExampleQuery example_query =
      client_only_plan_.phase().example_query_spec().example_queries().at(0);
  (*example_query.mutable_output_vector_specs())["new_tensor"] =
      new_vector_spec;
  client_only_plan_.mutable_phase()
      ->mutable_example_query_spec()
      ->clear_example_queries();
  client_only_plan_.mutable_phase()
      ->mutable_example_query_spec()
      ->mutable_example_queries()
      ->Add(std::move(example_query));

  EXPECT_CALL(
      mock_opstats_logger_,
      UpdateDatasetStats(kCollectionUri, num_examples_, example_bytes_));

  ExampleQueryPlanEngine plan_engine({example_iterator_factory_.get()},
                                     &mock_opstats_logger_);
  engine::PlanResult result =
      plan_engine.RunPlan(client_only_plan_.phase().example_query_spec(),
                          output_checkpoint_filename_);

  EXPECT_THAT(result.outcome, PlanOutcome::kExampleIteratorError);
}

TEST_F(ExampleQueryPlanEngineTest, FactoryNotFound) {
  Initialize();
  auto invalid_example_factory =
      std::make_unique<InvalidExampleIteratorFactory>();

  ExampleQueryPlanEngine plan_engine({invalid_example_factory.get()},
                                     &mock_opstats_logger_);
  engine::PlanResult result =
      plan_engine.RunPlan(client_only_plan_.phase().example_query_spec(),
                          output_checkpoint_filename_);

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
                          output_checkpoint_filename_);

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
                          output_checkpoint_filename_);

  EXPECT_THAT(result.outcome, PlanOutcome::kExampleIteratorError);
}

}  // anonymous namespace
}  // namespace engine
}  // namespace client
}  // namespace fcp
