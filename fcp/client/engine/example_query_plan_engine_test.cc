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
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/type/datetime.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "fcp/client/client_runner.h"
#include "fcp/client/engine/common.h"
#include "fcp/client/engine/example_iterator_factory.h"
#include "fcp/client/event_time_range.pb.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/client/simple_task_environment.h"
#include "fcp/client/tensorflow/tensorflow_runner.h"
#include "fcp/client/tensorflow/tensorflow_runner_impl.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/testing/testing.h"  // IWYU pragma: keep
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_header.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"

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
using tensorflow_federated::aggregation::Tensor;
using tensorflow_federated::aggregation::TensorShape;
using ::testing::StrictMock;
using ::testing::UnorderedElementsAreArray;

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

absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
ReadFCCheckpointTensors(absl::string_view checkpoint) {
  google::protobuf::io::ArrayInputStream input(checkpoint.data(),
                                     static_cast<int>(checkpoint.size()));
  google::protobuf::io::CodedInputStream stream(&input);

  std::string header;
  if (!stream.ReadString(&header, 4)) {
    return absl::InvalidArgumentError("Failed to read header");
  }
  if (header !=
      tensorflow_federated::aggregation::kFederatedComputeCheckpointHeader) {
    return absl::InvalidArgumentError("Invalid header");
  }

  absl::flat_hash_map<std::string, std::string> tensors;
  while (true) {
    uint32_t name_size;
    if (!stream.ReadVarint32(&name_size)) {
      return absl::InvalidArgumentError("Failed to read name size");
    }
    if (name_size == 0) {
      break;
    }
    std::string name;
    if (!stream.ReadString(&name, name_size)) {
      return absl::InvalidArgumentError("Failed to read name");
    }
    uint32_t tensor_size;
    if (!stream.ReadVarint32(&tensor_size)) {
      return absl::InvalidArgumentError("Failed to read tensor size");
    }
    std::string tensor;
    if (!stream.ReadString(&tensor, tensor_size)) {
      return absl::InvalidArgumentError("Failed to read tensor");
    }
    tensors[name] = tensor;
  }
  return tensors;
}

ExampleQuerySpec::ExampleQuery CreateDirectDataUploadExampleQuery(
    absl::string_view tensor_name, absl::string_view collection_uri) {
  ExampleQuerySpec::ExampleQuery query;
  query.set_direct_output_tensor_name(std::string(tensor_name));
  query.mutable_example_selector()->set_collection_uri(
      std::string(collection_uri));
  return query;
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
    tensorflow_runner_factory_ = []() {
      return std::make_unique<TensorflowRunnerImpl>();
    };
  }

  fcp::client::FilesImpl files_impl_;
  StrictMock<MockOpStatsLogger> mock_opstats_logger_;
  std::unique_ptr<ExampleIteratorFactory> example_iterator_factory_;

  ExampleQueryResult example_query_result_;
  ClientOnlyPlan client_only_plan_;
  Dataset dataset_;
  std::string output_checkpoint_filename_;
  absl::AnyInvocable<std::unique_ptr<TensorflowRunner>() const>
      tensorflow_runner_factory_;

  int num_examples_ = 0;
  int64_t example_bytes_ = 0;
};

TEST_F(ExampleQueryPlanEngineTest, PlanSucceeds) {
  Initialize();

  EXPECT_CALL(
      mock_opstats_logger_,
      UpdateDatasetStats(kCollectionUri, num_examples_, example_bytes_));
  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/false);

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

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/false);

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

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/false);

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

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/false);

  EXPECT_THAT(result.outcome, PlanOutcome::kExampleIteratorError);
}

TEST_F(ExampleQueryPlanEngineTest, FactoryNotFound) {
  Initialize();
  auto invalid_example_factory =
      std::make_unique<InvalidExampleIteratorFactory>();

  ExampleQueryPlanEngine plan_engine(
      {invalid_example_factory.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/false);

  EXPECT_THAT(result.outcome, PlanOutcome::kExampleIteratorError);
}

TEST_F(ExampleQueryPlanEngineTest, NoIteratorCreated) {
  Initialize();
  auto invalid_example_factory =
      std::make_unique<NoIteratorExampleIteratorFactory>();

  ExampleQueryPlanEngine plan_engine(
      {invalid_example_factory.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/false);

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

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/false);

  EXPECT_THAT(result.outcome, PlanOutcome::kExampleIteratorError);
}

TEST_F(ExampleQueryPlanEngineTest,
       PlanSucceedsWithFederatedComputeWireFormatEnabled) {
  Initialize();

  EXPECT_CALL(
      mock_opstats_logger_,
      UpdateDatasetStats(kCollectionUri, num_examples_, example_bytes_));

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/true);

  EXPECT_THAT(result.outcome, PlanOutcome::kSuccess);

  absl::string_view str = result.federated_compute_checkpoint.Flatten();
  absl::StatusOr<absl::flat_hash_map<std::string, std::string>> tensors =
      ReadFCCheckpointTensors(str);
  ASSERT_OK(tensors);

  absl::StatusOr<Tensor> int_tensor = Tensor::Create(
      tensorflow_federated::aggregation::DT_INT64, TensorShape({2}),
      tensorflow_federated::aggregation::CreateTestData<uint64_t>({42, 24}));
  ASSERT_OK(int_tensor.status());
  absl::StatusOr<Tensor> string_tensor = Tensor::Create(
      tensorflow_federated::aggregation::DT_STRING, TensorShape({2}),
      tensorflow_federated::aggregation::CreateTestData<absl::string_view>(
          {"value1", "value2"}));
  ASSERT_OK(string_tensor.status());
  absl::flat_hash_map<std::string, std::string> expected_tensors = {
      {kOutputIntTensorName, int_tensor->ToProto().SerializeAsString()},
      {kOutputStringTensorName, string_tensor->ToProto().SerializeAsString()}};

  ASSERT_THAT(*tensors, UnorderedElementsAreArray(expected_tensors));
}

TEST_F(ExampleQueryPlanEngineTest, PlanSucceedsWithEventTimeRange) {
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
  EventTimeRange event_time_range;
  event_time_range.mutable_start_event_time()->set_year(2024);
  event_time_range.mutable_start_event_time()->set_month(1);
  event_time_range.mutable_start_event_time()->set_day(1);
  event_time_range.mutable_end_event_time()->set_year(2024);
  event_time_range.mutable_end_event_time()->set_month(1);
  event_time_range.mutable_end_event_time()->set_day(7);
  second_example_query_result.mutable_stats()
      ->mutable_event_time_range()
      ->insert({"query_name", event_time_range});
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

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/true);

  EXPECT_THAT(result.outcome, PlanOutcome::kSuccess);

  ASSERT_THAT(result.event_time_range, EqualsProto(event_time_range));
}

TEST_F(ExampleQueryPlanEngineTest, PlanSucceedsWithOverriddenEventTimeRange) {
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
  EventTimeRange event_time_range;
  event_time_range.mutable_start_event_time()->set_year(2024);
  event_time_range.mutable_start_event_time()->set_month(1);
  event_time_range.mutable_start_event_time()->set_day(1);
  event_time_range.mutable_start_event_time()->set_hours(2);
  event_time_range.mutable_end_event_time()->set_year(2024);
  event_time_range.mutable_end_event_time()->set_month(1);
  event_time_range.mutable_end_event_time()->set_day(1);
  event_time_range.mutable_end_event_time()->set_hours(2);
  example_query_result.mutable_stats()->mutable_event_time_range()->insert(
      {"query_name", event_time_range});

  Dataset::ClientDataset client_dataset;
  client_dataset.set_client_id("client_id");
  client_dataset.add_example(example_query_result.SerializeAsString());
  Dataset dataset;
  dataset.mutable_client_data()->Add(std::move(client_dataset));

  ExampleQueryResult second_example_query_result;
  ExampleQueryResult::VectorData::Values float_values;
  float_values.mutable_float_values()->add_value(0.24f);
  float_values.mutable_float_values()->add_value(0.42f);
  float_values.mutable_float_values()->add_value(0.33f);
  ExampleQueryResult::VectorData::Values second_string_values;
  second_string_values.mutable_string_values()->add_value(
      "another_string_value");
  (*second_example_query_result.mutable_vector_data()
        ->mutable_vectors())["float_vector"] = float_values;
  (*second_example_query_result.mutable_vector_data()
        ->mutable_vectors())[kOutputStringVectorName] = second_string_values;
  EventTimeRange second_event_time_range;
  second_event_time_range.mutable_start_event_time()->set_year(2024);
  second_event_time_range.mutable_start_event_time()->set_month(1);
  second_event_time_range.mutable_start_event_time()->set_day(1);
  second_event_time_range.mutable_start_event_time()->set_hours(1);
  second_event_time_range.mutable_start_event_time()->set_minutes(1);
  second_event_time_range.mutable_end_event_time()->set_year(2024);
  second_event_time_range.mutable_end_event_time()->set_month(1);
  second_event_time_range.mutable_end_event_time()->set_day(2);
  second_event_time_range.mutable_end_event_time()->set_hours(1);
  second_event_time_range.mutable_end_event_time()->set_minutes(1);
  second_example_query_result.mutable_stats()
      ->mutable_event_time_range()
      ->insert({"query_name", second_event_time_range});

  Dataset::ClientDataset second_client_dataset;
  second_client_dataset.set_client_id("second_client_id");
  second_client_dataset.add_example(
      second_example_query_result.SerializeAsString());
  Dataset second_dataset;
  second_dataset.mutable_client_data()->Add(std::move(second_client_dataset));

  example_iterator_factory_ = std::make_unique<TwoExampleIteratorsFactory>(
      [&dataset = dataset](
          const google::internal::federated::plan::ExampleSelector& selector) {
        return std::make_unique<SimpleExampleIterator>(dataset);
      },
      [&dataset = second_dataset](
          const google::internal::federated::plan::ExampleSelector& selector) {
        return std::make_unique<SimpleExampleIterator>(dataset);
      },
      kCollectionUri, "app:/second_collection");

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/true);

  EXPECT_THAT(result.outcome, PlanOutcome::kSuccess);

  ASSERT_THAT(result.event_time_range, EqualsProto(second_event_time_range));
}

TEST_F(ExampleQueryPlanEngineTest, PlanSucceedsWithMergedEventTimeRange) {
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
  EventTimeRange event_time_range;
  event_time_range.mutable_start_event_time()->set_year(2024);
  event_time_range.mutable_start_event_time()->set_month(1);
  event_time_range.mutable_start_event_time()->set_day(2);
  event_time_range.mutable_start_event_time()->set_hours(1);
  event_time_range.mutable_start_event_time()->set_minutes(1);
  event_time_range.mutable_end_event_time()->set_year(2024);
  event_time_range.mutable_end_event_time()->set_month(1);
  event_time_range.mutable_end_event_time()->set_day(3);
  event_time_range.mutable_end_event_time()->set_hours(1);
  event_time_range.mutable_end_event_time()->set_minutes(1);
  example_query_result.mutable_stats()->mutable_event_time_range()->insert(
      {"query_name", event_time_range});

  Dataset::ClientDataset client_dataset;
  client_dataset.set_client_id("client_id");
  client_dataset.add_example(example_query_result.SerializeAsString());
  Dataset dataset;
  dataset.mutable_client_data()->Add(std::move(client_dataset));

  ExampleQueryResult second_example_query_result;
  ExampleQueryResult::VectorData::Values float_values;
  float_values.mutable_float_values()->add_value(0.24f);
  float_values.mutable_float_values()->add_value(0.42f);
  float_values.mutable_float_values()->add_value(0.33f);
  ExampleQueryResult::VectorData::Values second_string_values;
  second_string_values.mutable_string_values()->add_value(
      "another_string_value");
  (*second_example_query_result.mutable_vector_data()
        ->mutable_vectors())["float_vector"] = float_values;
  (*second_example_query_result.mutable_vector_data()
        ->mutable_vectors())[kOutputStringVectorName] = second_string_values;
  EventTimeRange second_event_time_range;
  second_event_time_range.mutable_start_event_time()->set_year(2024);
  second_event_time_range.mutable_start_event_time()->set_month(1);
  second_event_time_range.mutable_start_event_time()->set_day(2);
  second_event_time_range.mutable_end_event_time()->set_year(2024);
  second_event_time_range.mutable_end_event_time()->set_month(1);
  second_event_time_range.mutable_end_event_time()->set_day(3);
  second_example_query_result.mutable_stats()
      ->mutable_event_time_range()
      ->insert({"query_name", second_event_time_range});

  Dataset::ClientDataset second_client_dataset;
  second_client_dataset.set_client_id("second_client_id");
  second_client_dataset.add_example(
      second_example_query_result.SerializeAsString());
  Dataset second_dataset;
  second_dataset.mutable_client_data()->Add(std::move(second_client_dataset));

  example_iterator_factory_ = std::make_unique<TwoExampleIteratorsFactory>(
      [&dataset = dataset](
          const google::internal::federated::plan::ExampleSelector& selector) {
        return std::make_unique<SimpleExampleIterator>(dataset);
      },
      [&dataset = second_dataset](
          const google::internal::federated::plan::ExampleSelector& selector) {
        return std::make_unique<SimpleExampleIterator>(dataset);
      },
      kCollectionUri, "app:/second_collection");

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/true);

  EXPECT_THAT(result.outcome, PlanOutcome::kSuccess);

  EventTimeRange expected_event_time_range;
  expected_event_time_range.mutable_start_event_time()->set_year(2024);
  expected_event_time_range.mutable_start_event_time()->set_month(1);
  expected_event_time_range.mutable_start_event_time()->set_day(2);
  expected_event_time_range.mutable_end_event_time()->set_year(2024);
  expected_event_time_range.mutable_end_event_time()->set_month(1);
  expected_event_time_range.mutable_end_event_time()->set_day(3);
  expected_event_time_range.mutable_end_event_time()->set_hours(1);
  expected_event_time_range.mutable_end_event_time()->set_minutes(1);
  ASSERT_THAT(result.event_time_range, EqualsProto(expected_event_time_range));
}

TEST_F(ExampleQueryPlanEngineTest, MissingEndEventTimeFails) {
  Initialize();
  EventTimeRange event_time_range;
  event_time_range.mutable_start_event_time()->set_year(2024);
  event_time_range.mutable_start_event_time()->set_month(1);
  event_time_range.mutable_start_event_time()->set_day(1);
  event_time_range.mutable_start_event_time()->set_hours(2);

  example_query_result_.mutable_stats()->mutable_event_time_range()->insert(
      {"query_name", event_time_range});

  Dataset::ClientDataset client_dataset;
  client_dataset.set_client_id("client_id");
  auto example = example_query_result_.SerializeAsString();
  client_dataset.add_example(example);
  Dataset dataset;
  dataset.mutable_client_data()->Add(std::move(client_dataset));

  example_iterator_factory_ =
      std::make_unique<FunctionalExampleIteratorFactory>(
          [&dataset = dataset](
              const google::internal::federated::plan::ExampleSelector&
                  selector) {
            return std::make_unique<SimpleExampleIterator>(dataset);
          });
  EXPECT_CALL(mock_opstats_logger_,
              UpdateDatasetStats(kCollectionUri, 1, example.size()));

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/true);

  EXPECT_THAT(result.outcome, PlanOutcome::kExampleIteratorError);
}

TEST_F(ExampleQueryPlanEngineTest, SingleQueryDirectDataUploadTaskSucceeds) {
  const std::string kTensorName = "data";
  client_only_plan_.mutable_phase()
      ->mutable_example_query_spec()
      ->mutable_example_queries()
      ->Add(CreateDirectDataUploadExampleQuery(kTensorName, kCollectionUri));

  auto* aggregations = client_only_plan_.mutable_phase()
                           ->mutable_federated_example_query()
                           ->mutable_aggregations();
  AggregationConfig aggregation_config;
  aggregation_config.mutable_federated_compute_checkpoint_aggregation();
  (*aggregations)[kTensorName] = aggregation_config;

  tensorflow::Example example_1;
  (*example_1.mutable_features()->mutable_feature())["col1"]
      .mutable_int64_list()
      ->add_value(1);
  tensorflow::Example example_2;
  (*example_2.mutable_features()->mutable_feature())["col1"]
      .mutable_int64_list()
      ->add_value(2);
  std::string example_1_str = example_1.SerializeAsString();
  std::string example_2_str = example_2.SerializeAsString();

  Dataset::ClientDataset client_dataset;
  client_dataset.set_client_id("client_id");
  client_dataset.add_example(example_1_str);
  client_dataset.add_example(example_2_str);
  dataset_.mutable_client_data()->Add(std::move(client_dataset));

  num_examples_ = 2;
  example_bytes_ = example_1.ByteSizeLong() + example_2.ByteSizeLong();

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

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/true);

  EXPECT_THAT(result.outcome, PlanOutcome::kSuccess);

  absl::string_view str = result.federated_compute_checkpoint.Flatten();
  absl::StatusOr<absl::flat_hash_map<std::string, std::string>> tensors =
      ReadFCCheckpointTensors(str);
  ASSERT_OK(tensors);

  absl::StatusOr<Tensor> string_tensor = Tensor::Create(
      tensorflow_federated::aggregation::DT_STRING, TensorShape({2}),
      tensorflow_federated::aggregation::CreateTestData<absl::string_view>(
          {example_1_str, example_2_str}));
  ASSERT_OK(string_tensor.status());
  absl::flat_hash_map<std::string, std::string> expected_tensors = {
      {kTensorName, string_tensor->ToProto().SerializeAsString()}};

  ASSERT_THAT(*tensors, UnorderedElementsAreArray(expected_tensors));
}

TEST_F(ExampleQueryPlanEngineTest, TwoQueryDirectDataUploadTaskSucceeds) {
  const std::string kTensorName1 = "data_1";
  const std::string kTensorName2 = "data_2";
  const std::string kCollectionUri2 = "app:/collection_uri_2";
  auto* example_queries = client_only_plan_.mutable_phase()
                              ->mutable_example_query_spec()
                              ->mutable_example_queries();
  example_queries->Add(
      CreateDirectDataUploadExampleQuery(kTensorName1, kCollectionUri));
  example_queries->Add(
      CreateDirectDataUploadExampleQuery(kTensorName2, kCollectionUri2));

  auto* aggregations = client_only_plan_.mutable_phase()
                           ->mutable_federated_example_query()
                           ->mutable_aggregations();
  AggregationConfig aggregation_config;
  aggregation_config.mutable_federated_compute_checkpoint_aggregation();
  (*aggregations)[kTensorName1] = aggregation_config;
  (*aggregations)[kTensorName2] = aggregation_config;

  tensorflow::Example example_1;
  (*example_1.mutable_features()->mutable_feature())["col1"]
      .mutable_int64_list()
      ->add_value(1);
  tensorflow::Example example_2;
  (*example_2.mutable_features()->mutable_feature())["col1"]
      .mutable_int64_list()
      ->add_value(2);
  tensorflow::Example example_3;
  (*example_3.mutable_features()->mutable_feature())["col1"]
      .mutable_int64_list()
      ->add_value(3);
  tensorflow::Example example_4;
  (*example_4.mutable_features()->mutable_feature())["col1"]
      .mutable_int64_list()
      ->add_value(4);
  std::string example_1_str = example_1.SerializeAsString();
  std::string example_2_str = example_2.SerializeAsString();
  std::string example_3_str = example_3.SerializeAsString();
  std::string example_4_str = example_4.SerializeAsString();

  Dataset::ClientDataset client_dataset;
  client_dataset.set_client_id("client_id_1");
  client_dataset.add_example(example_1_str);
  client_dataset.add_example(example_2_str);
  dataset_.mutable_client_data()->Add(std::move(client_dataset));

  Dataset second_dataset;
  Dataset::ClientDataset second_client_dataset;
  second_client_dataset.set_client_id("client_id_2");
  second_client_dataset.add_example(example_3_str);
  second_client_dataset.add_example(example_4_str);
  second_dataset.mutable_client_data()->Add(std::move(second_client_dataset));

  example_iterator_factory_ = std::make_unique<TwoExampleIteratorsFactory>(
      [&dataset = dataset_](
          const google::internal::federated::plan::ExampleSelector& selector) {
        return std::make_unique<SimpleExampleIterator>(dataset);
      },
      [&dataset = second_dataset](
          const google::internal::federated::plan::ExampleSelector& selector) {
        return std::make_unique<SimpleExampleIterator>(dataset);
      },
      kCollectionUri, kCollectionUri2);

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/true);

  EXPECT_THAT(result.outcome, PlanOutcome::kSuccess);

  absl::string_view str = result.federated_compute_checkpoint.Flatten();
  absl::StatusOr<absl::flat_hash_map<std::string, std::string>> tensors =
      ReadFCCheckpointTensors(str);
  ASSERT_OK(tensors);

  absl::StatusOr<Tensor> string_tensor = Tensor::Create(
      tensorflow_federated::aggregation::DT_STRING, TensorShape({2}),
      tensorflow_federated::aggregation::CreateTestData<absl::string_view>(
          {example_1_str, example_2_str}));
  ASSERT_OK(string_tensor.status());
  absl::StatusOr<Tensor> second_string_tensor = Tensor::Create(
      tensorflow_federated::aggregation::DT_STRING, TensorShape({2}),
      tensorflow_federated::aggregation::CreateTestData<absl::string_view>(
          {example_3_str, example_4_str}));
  ASSERT_OK(second_string_tensor.status());
  absl::flat_hash_map<std::string, std::string> expected_tensors = {
      {kTensorName1, string_tensor->ToProto().SerializeAsString()},
      {kTensorName2, second_string_tensor->ToProto().SerializeAsString()}};

  ASSERT_THAT(*tensors, UnorderedElementsAreArray(expected_tensors));
}

TEST_F(ExampleQueryPlanEngineTest, MixedQueryTaskSucceeds) {
  const std::string kTensorName1 = "data_1";
  const std::string kTensorName2 = "data_2";
  const std::string kTensorName3 = "data_3";
  const std::string kCollectionUri2 = "app:/collection_uri_2";
  auto* example_queries = client_only_plan_.mutable_phase()
                              ->mutable_example_query_spec()
                              ->mutable_example_queries();
  example_queries->Add(
      CreateDirectDataUploadExampleQuery(kTensorName1, kCollectionUri));
  ExampleQuerySpec::ExampleQuery sql_example_query;
  sql_example_query.mutable_example_selector()->set_collection_uri(
      kCollectionUri2);
  ExampleQuerySpec::OutputVectorSpec output_vector_spec_1;
  output_vector_spec_1.set_vector_name("vector_1");
  output_vector_spec_1.set_data_type(ExampleQuerySpec::OutputVectorSpec::INT64);
  ExampleQuerySpec::OutputVectorSpec output_vector_spec_2;
  output_vector_spec_2.set_vector_name("vector_2");
  output_vector_spec_2.set_data_type(
      ExampleQuerySpec::OutputVectorSpec::STRING);
  (*sql_example_query.mutable_output_vector_specs())[kTensorName2] =
      output_vector_spec_1;
  (*sql_example_query.mutable_output_vector_specs())[kTensorName3] =
      output_vector_spec_2;
  example_queries->Add(std::move(sql_example_query));

  auto* aggregations = client_only_plan_.mutable_phase()
                           ->mutable_federated_example_query()
                           ->mutable_aggregations();
  AggregationConfig fc_checkpoint_aggregation_config;
  fc_checkpoint_aggregation_config
      .mutable_federated_compute_checkpoint_aggregation();
  AggregationConfig tf_v1_checkpoint_aggregation_config;
  tf_v1_checkpoint_aggregation_config.mutable_tf_v1_checkpoint_aggregation();
  (*aggregations)[kTensorName1] = fc_checkpoint_aggregation_config;
  (*aggregations)[kTensorName2] = tf_v1_checkpoint_aggregation_config;
  (*aggregations)[kTensorName3] = tf_v1_checkpoint_aggregation_config;

  tensorflow::Example example_1;
  (*example_1.mutable_features()->mutable_feature())["col1"]
      .mutable_int64_list()
      ->add_value(1);
  tensorflow::Example example_2;
  (*example_2.mutable_features()->mutable_feature())["col1"]
      .mutable_int64_list()
      ->add_value(2);
  std::string example_1_str = example_1.SerializeAsString();
  std::string example_2_str = example_2.SerializeAsString();

  Dataset::ClientDataset client_dataset;
  client_dataset.set_client_id("client_id_1");
  client_dataset.add_example(example_1_str);
  client_dataset.add_example(example_2_str);
  dataset_.mutable_client_data()->Add(std::move(client_dataset));

  ExampleQueryResult sql_query_result;
  auto* vector = sql_query_result.mutable_vector_data()->mutable_vectors();
  (*vector)["vector_1"].mutable_int64_values()->add_value(1);
  (*vector)["vector_2"].mutable_string_values()->add_value("string_value1");

  Dataset second_dataset;
  Dataset::ClientDataset second_client_dataset;
  second_client_dataset.set_client_id("client_id_2");
  second_client_dataset.add_example(sql_query_result.SerializeAsString());
  second_dataset.mutable_client_data()->Add(std::move(second_client_dataset));

  example_iterator_factory_ = std::make_unique<TwoExampleIteratorsFactory>(
      [&dataset = dataset_](
          const google::internal::federated::plan::ExampleSelector& selector) {
        return std::make_unique<SimpleExampleIterator>(dataset);
      },
      [&dataset = second_dataset](
          const google::internal::federated::plan::ExampleSelector& selector) {
        return std::make_unique<SimpleExampleIterator>(dataset);
      },
      kCollectionUri, kCollectionUri2);

  ExampleQueryPlanEngine plan_engine(
      {example_iterator_factory_.get()}, &mock_opstats_logger_,
      /*example_iterator_query_recorder=*/nullptr, tensorflow_runner_factory_);
  engine::PlanResult result = plan_engine.RunPlan(
      client_only_plan_.phase().example_query_spec(),
      output_checkpoint_filename_, /*use_client_report_wire_format=*/true);

  EXPECT_THAT(result.outcome, PlanOutcome::kSuccess);

  absl::string_view str = result.federated_compute_checkpoint.Flatten();
  absl::StatusOr<absl::flat_hash_map<std::string, std::string>> tensors =
      ReadFCCheckpointTensors(str);
  ASSERT_OK(tensors);

  absl::StatusOr<Tensor> first_tensor = Tensor::Create(
      tensorflow_federated::aggregation::DT_STRING, TensorShape({2}),
      tensorflow_federated::aggregation::CreateTestData<absl::string_view>(
          {example_1_str, example_2_str}));
  ASSERT_OK(first_tensor.status());
  absl::StatusOr<Tensor> second_tensor = Tensor::Create(
      tensorflow_federated::aggregation::DT_INT64, TensorShape({1}),
      tensorflow_federated::aggregation::CreateTestData<int64_t>({1}));
  ASSERT_OK(second_tensor.status());
  absl::StatusOr<Tensor> third_tensor = Tensor::Create(
      tensorflow_federated::aggregation::DT_STRING, TensorShape({1}),
      tensorflow_federated::aggregation::CreateTestData<absl::string_view>(
          {"string_value1"}));
  absl::flat_hash_map<std::string, std::string> expected_tensors = {
      {kTensorName1, first_tensor->ToProto().SerializeAsString()},
      {kTensorName2, second_tensor->ToProto().SerializeAsString()},
      {kTensorName3, third_tensor->ToProto().SerializeAsString()}};

  ASSERT_THAT(*tensors, UnorderedElementsAreArray(expected_tensors));
}

}  // anonymous namespace
}  // namespace engine
}  // namespace client
}  // namespace fcp
