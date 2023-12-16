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
#include "fcp/client/engine/plan_engine_helpers.h"

#include <atomic>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/example_iterator_query_recorder.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/opstats/opstats_logger_impl.h"
#include "fcp/client/opstats/pds_backed_opstats_db.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/tensorflow/external_dataset.h"

namespace fcp {
namespace client {
namespace engine {
namespace {

using ::fcp::client::opstats::OpStatsLogger;
using ::fcp::client::opstats::OpStatsLoggerImpl;
using ::fcp::client::opstats::PdsBackedOpStatsDb;
using ::google::internal::federated::plan::ExampleSelector;

/** An iterator that forwards the failing status from the external dataset to
 * TensorFlow. */
class FailingDatasetIterator : public ExternalDatasetIterator {
 public:
  explicit FailingDatasetIterator(absl::Status status) : status_(status) {}

  absl::StatusOr<std::string> GetNext() final { return status_; }

 private:
  const absl::Status status_;
};

class TrainingDatasetProvider
    : public ExternalDatasetProvider::UsingProtoSelector<ExampleSelector> {
 public:
  TrainingDatasetProvider(
      std::vector<ExampleIteratorFactory*> example_iterator_factories,
      OpStatsLogger* opstats_logger,
      ExampleIteratorQueryRecorder* example_iterator_query_recorder,
      std::atomic<int>* total_example_count,
      std::atomic<int64_t>* total_example_size_bytes,
      ExampleIteratorStatus* example_iterator_status)
      : example_iterator_factories_(example_iterator_factories),
        opstats_logger_(opstats_logger),
        example_iterator_query_recorder_(example_iterator_query_recorder),
        total_example_count_(total_example_count),
        total_example_size_bytes_(total_example_size_bytes),
        example_iterator_status_(example_iterator_status) {}

  absl::StatusOr<std::unique_ptr<ExternalDataset>> MakeDataset(
      ExampleSelector selector) final {
    return ExternalDataset::FromFunction(
        [example_iterator_factories = example_iterator_factories_,
         opstats_logger = opstats_logger_,
         example_iterator_query_recorder = example_iterator_query_recorder_,
         selector, total_example_count = total_example_count_,
         total_example_size_bytes = total_example_size_bytes_,
         example_iterator_status = example_iterator_status_]()
            -> std::unique_ptr<ExternalDatasetIterator> {
          ExampleIteratorFactory* example_iterator_factory =
              FindExampleIteratorFactory(selector, example_iterator_factories);
          // The DatasetOp requires a valid iterator at this stage so return an
          // empty iterator if there was an error.
          if (example_iterator_factory == nullptr) {
            absl::Status error(
                absl::StatusCode::kInternal,
                "Could not find suitable ExampleIteratorFactory");
            example_iterator_status->SetStatus(error);
            return std::make_unique<FailingDatasetIterator>(error);
          }
          absl::StatusOr<std::unique_ptr<ExampleIterator>> example_iterator =
              example_iterator_factory->CreateExampleIterator(selector);
          if (!example_iterator.ok()) {
            example_iterator_status->SetStatus(example_iterator.status());
            return std::make_unique<FailingDatasetIterator>(
                example_iterator.status());
          }
          SingleExampleIteratorQueryRecorder* single_query_recorder = nullptr;
          if (example_iterator_query_recorder) {
            single_query_recorder =
                example_iterator_query_recorder->RecordQuery(selector);
          }
          return std::make_unique<DatasetIterator>(
              std::move(*example_iterator), opstats_logger,
              single_query_recorder, total_example_count,
              total_example_size_bytes, example_iterator_status,
              selector.collection_uri(),
              /*collect_stats=*/
              example_iterator_factory->ShouldCollectStats());
        });
  }

 private:
  std::vector<ExampleIteratorFactory*> example_iterator_factories_;
  OpStatsLogger* opstats_logger_;
  ExampleIteratorQueryRecorder* example_iterator_query_recorder_;
  std::atomic<int>* total_example_count_;
  std::atomic<int64_t>* total_example_size_bytes_;
  ExampleIteratorStatus* example_iterator_status_;
};

}  // namespace

DatasetIterator::DatasetIterator(
    std::unique_ptr<ExampleIterator> example_iterator,
    opstats::OpStatsLogger* opstats_logger,
    SingleExampleIteratorQueryRecorder* single_query_recorder,
    std::atomic<int>* total_example_count,
    std::atomic<int64_t>* total_example_size_bytes,
    ExampleIteratorStatus* example_iterator_status,
    const std::string& collection_uri, bool collect_stats)
    : example_iterator_(std::move(example_iterator)),
      opstats_logger_(opstats_logger),
      single_query_recorder_(single_query_recorder),
      iterator_start_time_(absl::Now()),
      total_example_count_(total_example_count),
      total_example_size_bytes_(total_example_size_bytes),
      example_iterator_status_(example_iterator_status),
      example_count_(0),
      example_size_bytes_(0),
      collection_uri_(collection_uri),
      iterator_finished_(false),
      collect_stats_(collect_stats) {}

DatasetIterator::~DatasetIterator() {
  if (collect_stats_) {
    opstats_logger_->UpdateDatasetStats(collection_uri_, example_count_,
                                        example_size_bytes_);
  }
}

// Returns the next entry from the dataset.
absl::StatusOr<std::string> DatasetIterator::GetNext() {
  absl::MutexLock locked(&iterator_lock_);
  if (iterator_finished_) {
    // If we've reached the end of the iterator, always return OUT_OF_RANGE.
    return absl::OutOfRangeError("End of iterator reached");
  }
  absl::StatusOr<std::string> example = example_iterator_->Next();
  absl::StatusCode error_code = example.status().code();
  example_iterator_status_->SetStatus(example.status());
  if (error_code == absl::StatusCode::kOutOfRange) {
    example_iterator_->Close();
    iterator_finished_ = true;
  }
  if (example.ok()) {
    if (single_query_recorder_) {
      single_query_recorder_->Increment();
    }
    // If we're not forwarding an OUT_OF_RANGE to the caller, record example
    // stats for metrics logging.
    if (collect_stats_) {
      // TODO: b/184863488 - Consider reducing logic duplication in
      // cross-dataset and single-dataset example stat variables.
      *total_example_count_ += 1;
      *total_example_size_bytes_ += example->size();
      example_count_ += 1;
      example_size_bytes_ += example->size();
    }
  }
  return example;
}

void ExampleIteratorStatus::SetStatus(absl::Status status) {
  absl::MutexLock lock(&mu_);
  // We ignores normal status such as ok and outOfRange to avoid running into a
  // race condition when an error happened, then an outofRange or ok status
  // returned in a different thread which overrides the error status.
  if (status.code() != absl::StatusCode::kOk &&
      status.code() != absl::StatusCode::kOutOfRange) {
    status_ = status;
  }
}

absl::Status ExampleIteratorStatus::GetStatus() {
  absl::MutexLock lock(&mu_);
  return status_;
}

HostObjectRegistration AddDatasetTokenToInputs(
    std::vector<ExampleIteratorFactory*> example_iterator_factories,
    OpStatsLogger* opstats_logger,
    ExampleIteratorQueryRecorder* example_iterator_query_recorder,
    std::vector<std::pair<std::string, tensorflow::Tensor>>* inputs,
    const std::string& dataset_token_tensor_name,
    std::atomic<int>* total_example_count,
    std::atomic<int64_t>* total_example_size_bytes,
    ExampleIteratorStatus* example_iterator_status) {
  // Register the TrainingDatasetProvider with the global
  // ExternalDatasetProviderRegistry.
  auto host_registration = fcp::ExternalDatasetProviderRegistry::Register(
      std::make_shared<TrainingDatasetProvider>(
          example_iterator_factories, opstats_logger,
          example_iterator_query_recorder, total_example_count,
          total_example_size_bytes, example_iterator_status));
  // Pack the token returned from registering the provider into a string
  // tensor. TensorFlow will use that token via the ExternalDatasetOp to create
  // datasets and iterators.
  tensorflow::Tensor token_scalar(std::string{});
  token_scalar.scalar<tensorflow::tstring>()() =
      host_registration.token().ToString();
  std::pair<std::string, tensorflow::Tensor> token_pair(
      dataset_token_tensor_name, token_scalar);
  inputs->emplace_back(token_pair);
  return host_registration;
}

HostObjectRegistration AddDatasetTokenToInputsForTfLite(
    std::vector<ExampleIteratorFactory*> example_iterator_factories,
    OpStatsLogger* opstats_logger,
    ExampleIteratorQueryRecorder* example_iterator_query_recorder,
    absl::flat_hash_map<std::string, std::string>* inputs,
    const std::string& dataset_token_tensor_name,
    std::atomic<int>* total_example_count,
    std::atomic<int64_t>* total_example_size_bytes,
    ExampleIteratorStatus* example_iterator_status) {
  // Registers the TrainingDatasetProvider with the global
  // ExternalDatasetProviderRegistry.
  auto host_registration = fcp::ExternalDatasetProviderRegistry::Register(
      std::make_shared<TrainingDatasetProvider>(
          example_iterator_factories, opstats_logger,
          example_iterator_query_recorder, total_example_count,
          total_example_size_bytes, example_iterator_status));
  // Adds the token returned from registering the provider to the map of inputs.
  // TfLite will use that token via the ExternalDatasetOp to create
  // datasets and iterators.
  (*inputs)[dataset_token_tensor_name] = host_registration.token().ToString();
  return host_registration;
}

std::unique_ptr<::fcp::client::opstats::OpStatsLogger> CreateOpStatsLogger(
    const std::string& base_dir, const Flags* flags, LogManager* log_manager,
    const std::string& session_name, const std::string& population_name) {
  if (flags->enable_opstats()) {
    auto db_or = PdsBackedOpStatsDb::Create(
        base_dir, flags->opstats_ttl_days() * absl::Hours(24), *log_manager,
        flags->opstats_db_size_limit_bytes());
    if (db_or.ok()) {
      return std::make_unique<OpStatsLoggerImpl>(std::move(db_or).value(),
                                                 log_manager, flags,
                                                 session_name, population_name);
    } else {
      return std::make_unique<OpStatsLogger>(
          /*opstats_enabled=*/flags->enable_opstats(),
          /*init_status=*/db_or.status());
    }
  }
  return std::make_unique<OpStatsLogger>(
      /*opstats_enabled=*/flags->enable_opstats());
}

PlanResult CreateComputationErrorPlanResult(
    absl::Status example_iterator_status,
    absl::Status computation_error_status) {
  switch (example_iterator_status.code()) {
    case absl::StatusCode::kOk:
    case absl::StatusCode::kOutOfRange:
      // Either example iterators are working fine or we don't know the status
      // of the example iterators. In this case, we'll use the error status
      // returned from TensorFlow.
      return PlanResult(PlanOutcome::kTensorflowError,
                        computation_error_status);
    case absl::StatusCode::kCancelled:
      // Example iterator got interrupted.
      return PlanResult(PlanOutcome::kInterrupted, example_iterator_status);
    default:
      // All other Example iterator errors.
      return PlanResult(PlanOutcome::kExampleIteratorError,
                        example_iterator_status);
  }
}

ExampleIteratorFactory* FindExampleIteratorFactory(
    const ExampleSelector& selector,
    std::vector<ExampleIteratorFactory*> example_iterator_factories) {
  for (ExampleIteratorFactory* factory : example_iterator_factories) {
    if (factory->CanHandle(selector)) {
      return factory;
    }
  }
  return nullptr;
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
