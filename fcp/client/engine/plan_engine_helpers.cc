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

#include <functional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/opstats/opstats_example_store.h"
#include "fcp/client/opstats/opstats_logger.h"
#include "fcp/client/opstats/opstats_logger_impl.h"
#include "fcp/client/opstats/pds_backed_opstats_db.h"
#include "fcp/tensorflow/external_dataset.h"

namespace fcp {
namespace client {
namespace engine {
namespace {

using ::fcp::client::opstats::OperationalStats;
using ::fcp::client::opstats::OpStatsLogger;
using ::fcp::client::opstats::OpStatsLoggerImpl;
using ::fcp::client::opstats::PdsBackedOpStatsDb;
using ::google::internal::federated::plan::ExampleSelector;

class DatasetIterator : public ExternalDatasetIterator {
 public:
  DatasetIterator(std::unique_ptr<ExampleIterator> example_iterator,
                  EventPublisher* event_publisher,
                  OpStatsLogger* opstats_logger,
                  std::atomic<int>* total_example_count,
                  std::atomic<int64_t>* total_example_size_bytes,
                  const std::string& collection_uri)
      : example_iterator_(std::move(example_iterator)),
        event_publisher_(event_publisher),
        opstats_logger_(opstats_logger),
        iterator_start_time_(absl::Now()),
        total_example_count_(total_example_count),
        total_example_size_bytes_(total_example_size_bytes),
        example_count_(0),
        example_size_bytes_(0),
        collection_uri_(collection_uri),
        iterator_finished_(false) {}

  ~DatasetIterator() override {
    opstats_logger_->UpdateDatasetStats(collection_uri_, example_count_,
                                        example_size_bytes_);
  }

  absl::StatusOr<std::string> GetNext() final {
    absl::MutexLock locked(&iterator_lock_);
    if (iterator_finished_) {
      // If we've reached the end of the iterator, always return OUT_OF_RANGE.
      return absl::OutOfRangeError("End of iterator reached");
    }
    absl::StatusOr<std::string> example_or = example_iterator_->Next();
    absl::StatusCode error_code = example_or.status().code();
    if (error_code == absl::StatusCode::kCancelled) {
      event_publisher_->PublishInterruption(0, 0, 0, 0, iterator_start_time_);
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED,
          std::string(example_or.status().message()));
    } else if (error_code == absl::StatusCode::kInvalidArgument) {
      std::string message = absl::StrCat("Error reading example: ",
                                         example_or.status().message());
      event_publisher_->PublishIoError(0, message);
      opstats_logger_->AddEventWithErrorMessage(
          OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
    } else if (error_code == absl::StatusCode::kOutOfRange) {
      example_iterator_->Close();
      iterator_finished_ = true;
    }
    // If we're not forwarding an OUT_OF_RANGE to the caller, record example
    // stats for metrics logging.
    if (example_or.ok()) {
      // TODO(team): Consider reducing logic duplication in cross-dataset
      // and single-dataset example stat variables.
      *total_example_count_ += 1;
      *total_example_size_bytes_ += example_or.value().size();
      example_count_ += 1;
      example_size_bytes_ += example_or.value().size();
    }
    return example_or;
  }

 private:
  std::unique_ptr<ExampleIterator> example_iterator_
      ABSL_GUARDED_BY(iterator_lock_);
  EventPublisher* event_publisher_;
  OpStatsLogger* opstats_logger_;
  absl::Time iterator_start_time_;
  // Example stats across all datasets.
  std::atomic<int>* total_example_count_;
  std::atomic<int64_t>* total_example_size_bytes_;
  // Example stats only for this dataset.
  std::atomic<int> example_count_;
  std::atomic<int64_t> example_size_bytes_;
  const std::string collection_uri_;
  bool iterator_finished_ ABSL_GUARDED_BY(iterator_lock_);
  absl::Mutex iterator_lock_;
};

class EmptyDatasetIterator : public ExternalDatasetIterator {
 public:
  EmptyDatasetIterator() {}
  absl::StatusOr<std::string> GetNext() final {
    return absl::OutOfRangeError("");
  }
};

class TrainingDatasetProvider
    : public ExternalDatasetProvider::UsingProtoSelector<ExampleSelector> {
 public:
  TrainingDatasetProvider(
      std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
          const google::internal::federated::plan::ExampleSelector&)>
          create_example_iterator,
      EventPublisher* event_publisher, LogManager* log_manager,
      OpStatsLogger* opstats_logger, std::atomic<int>* total_example_count,
      std::atomic<int64_t>* total_example_size_bytes)
      : create_example_iterator_(create_example_iterator),
        event_publisher_(event_publisher),
        log_manager_(log_manager),
        opstats_logger_(opstats_logger),
        total_example_count_(total_example_count),
        total_example_size_bytes_(total_example_size_bytes) {}

  absl::StatusOr<std::unique_ptr<ExternalDataset>> MakeDataset(
      ExampleSelector selector) final {
    return ExternalDataset::FromFunction(
        [create_example_iterator = create_example_iterator_,
         event_publisher = event_publisher_, log_manager = log_manager_,
         opstats_logger = opstats_logger_, selector,
         total_example_count = total_example_count_,
         total_example_size_bytes = total_example_size_bytes_]()
            -> std::unique_ptr<ExternalDatasetIterator> {
          auto example_iterator_or = GetExampleIterator(
              selector, log_manager, opstats_logger, create_example_iterator);
          // The DatasetOp requires a valid iterator at
          // this stage so return an empty iterator if there was an error.
          if (!example_iterator_or.ok()) {
            if (example_iterator_or.status().code() ==
                absl::StatusCode::kInvalidArgument) {
              event_publisher->PublishExampleSelectorError(
                  0, 0, 0, example_iterator_or.status().message());
              opstats_logger->AddEventWithErrorMessage(
                  OperationalStats::Event::EVENT_KIND_ERROR_EXAMPLE_SELECTOR,
                  std::string(example_iterator_or.status().message()));
            }
            return std::make_unique<EmptyDatasetIterator>();
          }
          return std::make_unique<DatasetIterator>(
              std::move(example_iterator_or.value()), event_publisher,
              opstats_logger, total_example_count, total_example_size_bytes,
              selector.collection_uri());
        });
  }

 private:
  std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
      const google::internal::federated::plan::ExampleSelector&)>
      create_example_iterator_;
  EventPublisher* event_publisher_;
  LogManager* log_manager_;
  OpStatsLogger* opstats_logger_;
  std::atomic<int>* total_example_count_;
  std::atomic<int64_t>* total_example_size_bytes_;
};

}  // namespace

HostObjectRegistration AddDatasetTokenToInputs(
    std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
        const google::internal::federated::plan::ExampleSelector&)>
        create_example_iterator,
    EventPublisher* event_publisher, LogManager* log_manager,
    OpStatsLogger* opstats_logger,
    std::vector<std::pair<std::string, tensorflow::Tensor>>* inputs,
    const std::string& dataset_token_tensor_name,
    std::atomic<int>* total_example_count,
    std::atomic<int64_t>* total_example_size_bytes) {
  // Register the TrainingDatasetProvider with the global
  // ExternalDatasetProviderRegistry.
  auto host_registration = fcp::ExternalDatasetProviderRegistry::Register(
      std::make_shared<TrainingDatasetProvider>(
          create_example_iterator, event_publisher, log_manager, opstats_logger,
          total_example_count, total_example_size_bytes));
  // Pack the token returned from registering the provider into a std::string
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
    std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
        const google::internal::federated::plan::ExampleSelector&)>
        create_example_iterator,
    EventPublisher* event_publisher, LogManager* log_manager,
    OpStatsLogger* opstats_logger,
    absl::flat_hash_map<std::string, std::string>* inputs,
    const std::string& dataset_token_tensor_name,
    std::atomic<int>* total_example_count,
    std::atomic<int64_t>* total_example_size_bytes) {
  // Registers the TrainingDatasetProvider with the global
  // ExternalDatasetProviderRegistry.
  auto host_registration = fcp::ExternalDatasetProviderRegistry::Register(
      std::make_shared<TrainingDatasetProvider>(
          create_example_iterator, event_publisher, log_manager, opstats_logger,
          total_example_count, total_example_size_bytes));
  // Adds the token returned from registering the provider to the map of inputs.
  // TfLite will use that token via the ExternalDatasetOp to create
  // datasets and iterators.
  (*inputs)[dataset_token_tensor_name] = host_registration.token().ToString();
  return host_registration;
}

absl::StatusOr<std::unique_ptr<ExampleIterator>> GetExampleIterator(
    const ExampleSelector& selector, LogManager* log_manager,
    OpStatsLogger* opstats_logger,
    std::function<absl::StatusOr<std::unique_ptr<ExampleIterator>>(
        const google::internal::federated::plan::ExampleSelector&)>
        create_example_iterator) {
  if (selector.collection_uri() == opstats::kOpStatsCollectionUri) {
    if (!opstats_logger->IsOpStatsEnabled()) {
      log_manager->LogDiag(
          ProdDiagCode::OPSTATS_EXAMPLE_STORE_REQUESTED_NOT_ENABLED);
      return absl::InvalidArgumentError(
          "OpStats example store is not enabled.");
    } else {
      return opstats::CreateExampleIterator(
          selector, *opstats_logger->GetOpStatsDb(), *log_manager);
    }
  } else {
    return create_example_iterator(selector);
  }
}

void LogOpStatsNetworkErrors(OpStatsLogger* opstats_logger, Status status,
                             const std::string& message) {
  if (status.code() == absl::StatusCode::kAborted) {
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_SERVER_ABORTED, message);
  } else if (status.code() == absl::StatusCode::kCancelled) {
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_CLIENT_INTERRUPTED, message);
  } else if (!status.ok()) {
    opstats_logger->AddEventWithErrorMessage(
        OperationalStats::Event::EVENT_KIND_ERROR_IO, message);
  }
}

std::unique_ptr<::fcp::client::opstats::OpStatsLogger> CreateOpStatsLogger(
    const std::string& base_dir, const Flags* flags, LogManager* log_manager,
    const std::string& session_name, const std::string& population_name) {
  if (flags->enable_opstats()) {
    auto db_or = PdsBackedOpStatsDb::Create(
        base_dir, flags->opstats_ttl_days() * absl::Hours(24), *log_manager,
        flags->opstats_db_size_limit_bytes(),
        flags->opstats_enforce_singleton(),
        flags->record_earliest_trustworthy_time_for_opstats());
    if (db_or.ok()) {
      return std::make_unique<OpStatsLoggerImpl>(
          std::move(db_or).value(), log_manager, session_name, population_name);
    }
  }
  return std::make_unique<OpStatsLogger>(
      /*opstats_enabled=*/flags->enable_opstats());
}

}  // namespace engine
}  // namespace client
}  // namespace fcp
