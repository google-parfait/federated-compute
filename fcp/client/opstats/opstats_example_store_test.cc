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
#include "fcp/client/opstats/opstats_example_store.h"

#include <string>
#include <utility>

#include "google/protobuf/any.pb.h"
#include "google/protobuf/util/time_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/federated_api.pb.h"
#include "fcp/protos/plan.pb.h"
#include "fcp/testing/testing.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

namespace fcp {
namespace client {
namespace opstats {
namespace {

using ::google::internal::federated::plan::ExampleSelector;
using ::google::internal::federatedml::v2::RetryWindow;
using ::google::protobuf::util::TimeUtil;
using ::testing::Return;

constexpr char kTestTaskName[] = "stefans_really_cool_task";

class OpStatsExampleStoreTest : public testing::Test {
 public:
  OpStatsExampleStoreTest() {
    EXPECT_CALL(mock_opstats_logger_, IsOpStatsEnabled())
        .WillRepeatedly(Return(true));
    EXPECT_CALL(mock_opstats_logger_, GetOpStatsDb())
        .WillRepeatedly(Return(&mock_db_));
    EXPECT_CALL(mock_opstats_logger_, GetCurrentTaskName())
        .WillRepeatedly(Return(kTestTaskName));
  }

 protected:
  static OperationalStats::Event CreateEvent(
      OperationalStats::Event::EventKind event_kind, int64_t event_time_ms) {
    OperationalStats::Event event;
    event.set_event_type(event_kind);
    *event.mutable_timestamp() =
        TimeUtil::MillisecondsToTimestamp(event_time_ms);
    return event;
  }

  static OperationalStats::DatasetStats CreateDatasetStats(
      int64_t num_examples_read, int64_t num_bytes_read) {
    OperationalStats::DatasetStats stats;
    stats.set_num_bytes_read(num_bytes_read);
    stats.set_num_examples_read(num_examples_read);
    return stats;
  }

  testing::StrictMock<MockOpStatsLogger> mock_opstats_logger_;
  testing::StrictMock<MockOpStatsDb> mock_db_;
  testing::StrictMock<MockLogManager> mock_log_manager_;
  OpStatsExampleIteratorFactory iterator_factory_ =
      OpStatsExampleIteratorFactory(
          &mock_opstats_logger_, &mock_log_manager_,
          /*opstats_last_successful_contribution_criteria=*/false);
};

TEST_F(OpStatsExampleStoreTest, TestInvalidCollectionUrl) {
  ExampleSelector selector;
  selector.set_collection_uri("INVALID");
  EXPECT_CALL(mock_log_manager_,
              LogDiag(ProdDiagCode::OPSTATS_INCORRECT_COLLECTION_URI));

  EXPECT_FALSE(iterator_factory_.CanHandle(selector));

  absl::StatusOr<std::unique_ptr<ExampleIterator>> status_or =
      iterator_factory_.CreateExampleIterator(selector);
  EXPECT_THAT(status_or.status(), IsCode(absl::StatusCode::kInvalidArgument));
}

TEST_F(OpStatsExampleStoreTest, TestMalformedCriteria) {
  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  selector.mutable_criteria()->set_value("NOT_A_PROTO");
  EXPECT_CALL(mock_log_manager_,
              LogDiag(ProdDiagCode::OPSTATS_INVALID_SELECTION_CRITERIA));
  absl::StatusOr<std::unique_ptr<ExampleIterator>> status_or =
      iterator_factory_.CreateExampleIterator(selector);
  EXPECT_THAT(status_or.status(), IsCode(absl::StatusCode::kInvalidArgument));
}

TEST_F(OpStatsExampleStoreTest, TestInvalidCriteria) {
  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  OpStatsSelectionCriteria criteria;
  *criteria.mutable_start_time() = TimeUtil::MillisecondsToTimestamp(2000L);
  *criteria.mutable_end_time() = TimeUtil::MillisecondsToTimestamp(1000L);
  selector.mutable_criteria()->PackFrom(criteria);
  EXPECT_CALL(mock_log_manager_,
              LogDiag(ProdDiagCode::OPSTATS_INVALID_SELECTION_CRITERIA));
  absl::StatusOr<std::unique_ptr<ExampleIterator>> status_or =
      iterator_factory_.CreateExampleIterator(selector);
  EXPECT_THAT(status_or.status(), IsCode(absl::StatusCode::kInvalidArgument));
}

TEST_F(OpStatsExampleStoreTest, TestReadFromDbFailed) {
  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  EXPECT_CALL(mock_db_, Read())
      .WillOnce(Return(absl::InternalError("Something's wrong.")));
  absl::StatusOr<std::unique_ptr<ExampleIterator>> status_or =
      iterator_factory_.CreateExampleIterator(selector);
  EXPECT_THAT(status_or.status(), IsCode(absl::StatusCode::kInternal));
}

TEST_F(OpStatsExampleStoreTest, Success) {
  // Prepare some data
  OpStatsSequence opstats_sequence;

  OperationalStats* stats_first = opstats_sequence.add_opstats();
  std::string session_first = "session_first";
  std::string population_first = "population_first";
  stats_first->set_session_name(session_first);
  stats_first->set_population_name(population_first);

  OperationalStats* stats_last = opstats_sequence.add_opstats();
  std::string session_last = "session_last";
  std::string population_last = "population_last";
  stats_last->set_session_name(session_last);
  stats_last->set_population_name(population_last);

  EXPECT_CALL(mock_db_, Read()).WillOnce(Return(opstats_sequence));

  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  absl::StatusOr<std::unique_ptr<ExampleIterator>> iterator_or =
      iterator_factory_.CreateExampleIterator(selector);
  ASSERT_TRUE(iterator_or.ok());
  std::unique_ptr<ExampleIterator> iterator = std::move(iterator_or.value());
  absl::StatusOr<std::string> example_or = iterator->Next();
  ASSERT_TRUE(example_or.ok());
  tensorflow::Example example_last;
  ASSERT_TRUE(example_last.ParseFromString(example_or.value()));
  example_or = iterator->Next();
  ASSERT_TRUE(example_or.ok());
  tensorflow::Example example_first;
  ASSERT_TRUE(example_first.ParseFromString(example_or.value()));

  // Check if the examples contain the expected data. Opstats examples are
  // returned in last in, first out order.
  std::set<std::string> actual_session_names;
  actual_session_names.insert(ExtractSingleString(example_last, kSessionName));
  actual_session_names.insert(ExtractSingleString(example_first, kSessionName));
  std::set<std::string> expected_session_names = {session_last, session_first};
  EXPECT_EQ(actual_session_names, expected_session_names);

  std::set<std::string> actual_population_names;
  actual_population_names.insert(
      ExtractSingleString(example_last, kPopulationName));
  actual_population_names.insert(
      ExtractSingleString(example_first, kPopulationName));
  std::set<std::string> expected_population_names = {population_last,
                                                     population_first};
  EXPECT_EQ(actual_population_names, expected_population_names);

  // We should have arrived at the end of the iterator.
  example_or = iterator->Next();
  EXPECT_THAT(example_or.status(), IsCode(absl::StatusCode::kOutOfRange));

  // Subsequent Next() calls should all return OUT_OF_RANGE.
  example_or = iterator->Next();
  EXPECT_THAT(example_or.status(), IsCode(absl::StatusCode::kOutOfRange));

  // Close() should work without exceptions.
  iterator->Close();
}

TEST_F(OpStatsExampleStoreTest, EmptyData) {
  EXPECT_CALL(mock_db_, Read())
      .WillOnce(Return(OpStatsSequence::default_instance()));

  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  absl::StatusOr<std::unique_ptr<ExampleIterator>> iterator_or =
      iterator_factory_.CreateExampleIterator(selector);
  ASSERT_TRUE(iterator_or.ok());
  std::unique_ptr<ExampleIterator> iterator = std::move(iterator_or.value());
  absl::StatusOr<std::string> status_or = iterator->Next();
  EXPECT_THAT(status_or.status(), IsCode(absl::StatusCode::kOutOfRange));
}

TEST_F(OpStatsExampleStoreTest, DataIsFilteredBySelectionCriteria) {
  OperationalStats included;
  included.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 900L));
  included.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED, 1000L));

  OperationalStats excluded_early;
  excluded_early.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 500L));
  excluded_early.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED, 700L));

  OperationalStats excluded_late;
  excluded_late.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 800L));
  excluded_late.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED, 2001L));

  OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(excluded_early);
  *opstats_sequence.add_opstats() = std::move(included);
  *opstats_sequence.add_opstats() = std::move(excluded_late);
  EXPECT_CALL(mock_db_, Read()).WillOnce(Return(opstats_sequence));

  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  OpStatsSelectionCriteria criteria;
  *criteria.mutable_start_time() = TimeUtil::MillisecondsToTimestamp(1000L);
  *criteria.mutable_end_time() = TimeUtil::MillisecondsToTimestamp(2000L);
  selector.mutable_criteria()->PackFrom(criteria);
  absl::StatusOr<std::unique_ptr<ExampleIterator>> iterator_or =
      iterator_factory_.CreateExampleIterator(selector);

  ASSERT_TRUE(iterator_or.ok());
  std::unique_ptr<ExampleIterator> iterator = std::move(iterator_or.value());
  absl::StatusOr<std::string> example_or = iterator->Next();
  ASSERT_TRUE(example_or.ok());
  tensorflow::Example example;
  example.ParseFromString(example_or.value());
  auto event_type_list = ExtractRepeatedInt64(example, kEventsEventType);
  ASSERT_EQ(event_type_list.at(0),
            OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED);
  ASSERT_EQ(event_type_list.at(1),
            OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED);
  auto event_time_ms_list =
      ExtractRepeatedInt64(example, kEventsTimestampMillis);
  ASSERT_EQ(event_time_ms_list.at(0), 900);
  ASSERT_EQ(event_time_ms_list.at(1), 1000);

  // We expect the iterator reaches the end because there's only 1 example.
  example_or = iterator->Next();
  EXPECT_THAT(example_or.status(), IsCode(absl::StatusCode::kOutOfRange));
}

TEST_F(OpStatsExampleStoreTest, SelectionCriteriaOnlyContainsBeginTime) {
  OperationalStats included;
  included.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 900L));
  included.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED, 1000L));

  OperationalStats excluded_early;
  excluded_early.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 500L));
  excluded_early.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED, 700L));

  OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(excluded_early);
  *opstats_sequence.add_opstats() = std::move(included);
  EXPECT_CALL(mock_db_, Read()).WillOnce(Return(opstats_sequence));

  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  OpStatsSelectionCriteria criteria;
  *criteria.mutable_start_time() = TimeUtil::MillisecondsToTimestamp(1000L);
  selector.mutable_criteria()->PackFrom(criteria);
  absl::StatusOr<std::unique_ptr<ExampleIterator>> iterator_or =
      iterator_factory_.CreateExampleIterator(selector);

  ASSERT_TRUE(iterator_or.ok());
  std::unique_ptr<ExampleIterator> iterator = std::move(iterator_or.value());
  absl::StatusOr<std::string> example_or = iterator->Next();
  ASSERT_TRUE(example_or.ok());
  tensorflow::Example example;
  example.ParseFromString(example_or.value());
  auto event_type_list = ExtractRepeatedInt64(example, kEventsEventType);
  ASSERT_EQ(event_type_list.at(0),
            OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED);
  ASSERT_EQ(event_type_list.at(1),
            OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED);
  auto event_time_ms_list =
      ExtractRepeatedInt64(example, kEventsTimestampMillis);
  ASSERT_EQ(event_time_ms_list.at(0), 900);
  ASSERT_EQ(event_time_ms_list.at(1), 1000);

  // We expect the iterator reaches the end because there's only 1 example.
  example_or = iterator->Next();
  EXPECT_THAT(example_or.status(), IsCode(absl::StatusCode::kOutOfRange));
}

TEST_F(OpStatsExampleStoreTest, SelectionCriteriaOnlyContainsEndTime) {
  OperationalStats included;
  included.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 900L));
  included.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED, 1000L));

  OperationalStats excluded_late;
  excluded_late.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 800L));
  excluded_late.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED, 2001L));

  OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(included);
  *opstats_sequence.add_opstats() = std::move(excluded_late);
  EXPECT_CALL(mock_db_, Read()).WillOnce(Return(opstats_sequence));

  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  OpStatsSelectionCriteria criteria;
  *criteria.mutable_start_time() = TimeUtil::MillisecondsToTimestamp(1000L);
  *criteria.mutable_end_time() = TimeUtil::MillisecondsToTimestamp(2000L);
  selector.mutable_criteria()->PackFrom(criteria);
  absl::StatusOr<std::unique_ptr<ExampleIterator>> iterator_or =
      iterator_factory_.CreateExampleIterator(selector);

  ASSERT_TRUE(iterator_or.ok());
  std::unique_ptr<ExampleIterator> iterator = std::move(iterator_or.value());
  absl::StatusOr<std::string> example_or = iterator->Next();
  ASSERT_TRUE(example_or.ok());
  tensorflow::Example example;
  example.ParseFromString(example_or.value());
  auto event_type_list = ExtractRepeatedInt64(example, kEventsEventType);
  ASSERT_EQ(event_type_list.at(0),
            OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED);
  ASSERT_EQ(event_type_list.at(1),
            OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED);
  auto event_time_ms_list =
      ExtractRepeatedInt64(example, kEventsTimestampMillis);
  ASSERT_EQ(event_time_ms_list.at(0), 900);
  ASSERT_EQ(event_time_ms_list.at(1), 1000);

  // We expect the iterator reaches the end because there's only 1 example.
  example_or = iterator->Next();
  EXPECT_THAT(example_or.status(), IsCode(absl::StatusCode::kOutOfRange));
}

TEST_F(OpStatsExampleStoreTest,
       SelectionCriteriaLastSuccessfulContributionEnabledAndExists) {
  OpStatsExampleIteratorFactory iterator_factory =
      OpStatsExampleIteratorFactory(
          &mock_opstats_logger_, &mock_log_manager_,
          /*opstats_last_successful_contribution_criteria=*/true);
  OperationalStats included;
  included.set_task_name(kTestTaskName);
  included.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 900L));
  included.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED, 1000L));

  OperationalStats last_successful_contribution;
  last_successful_contribution.set_task_name(kTestTaskName);
  last_successful_contribution.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 1200L));
  last_successful_contribution.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED, 2001L));

  OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(included);
  *opstats_sequence.add_opstats() = std::move(last_successful_contribution);
  EXPECT_CALL(mock_db_, Read()).WillOnce(Return(opstats_sequence));

  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  OpStatsSelectionCriteria criteria;
  criteria.set_last_successful_contribution(true);
  selector.mutable_criteria()->PackFrom(criteria);
  absl::StatusOr<std::unique_ptr<ExampleIterator>> iterator_or =
      iterator_factory.CreateExampleIterator(selector);

  EXPECT_OK(iterator_or);
  std::unique_ptr<ExampleIterator> iterator = std::move(iterator_or.value());
  absl::StatusOr<std::string> example_or = iterator->Next();
  EXPECT_OK(example_or);
  tensorflow::Example example;
  example.ParseFromString(example_or.value());
  auto event_type_list = ExtractRepeatedInt64(example, kEventsEventType);
  ASSERT_EQ(event_type_list.at(0),
            OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED);
  ASSERT_EQ(event_type_list.at(1),
            OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED);
  auto event_time_ms_list =
      ExtractRepeatedInt64(example, kEventsTimestampMillis);
  ASSERT_EQ(event_time_ms_list.at(0), 1200L);
  ASSERT_EQ(event_time_ms_list.at(1), 2001L);

  // We expect the iterator reaches the end because there's only 1 example.
  example_or = iterator->Next();
  EXPECT_THAT(example_or.status(), IsCode(absl::StatusCode::kOutOfRange));
}

TEST_F(OpStatsExampleStoreTest,
       SelectionCriteriaLastSuccessfulContributionEnabledAndDoesNotExist) {
  OpStatsExampleIteratorFactory iterator_factory =
      OpStatsExampleIteratorFactory(
          &mock_opstats_logger_, &mock_log_manager_,
          /*opstats_last_successful_contribution_criteria=*/true);
  OperationalStats non_matching;
  non_matching.set_task_name("non_matching_task_name");
  non_matching.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 900L));
  non_matching.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED, 1000L));

  OperationalStats matching_but_no_upload;
  matching_but_no_upload.set_task_name(kTestTaskName);
  matching_but_no_upload.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 1200L));

  OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(non_matching);
  *opstats_sequence.add_opstats() = std::move(matching_but_no_upload);
  EXPECT_CALL(mock_db_, Read()).WillOnce(Return(opstats_sequence));

  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  OpStatsSelectionCriteria criteria;
  criteria.set_last_successful_contribution(true);
  selector.mutable_criteria()->PackFrom(criteria);
  absl::StatusOr<std::unique_ptr<ExampleIterator>> iterator_or =
      iterator_factory.CreateExampleIterator(selector);

  EXPECT_OK(iterator_or);
  std::unique_ptr<ExampleIterator> iterator = std::move(iterator_or.value());
  absl::StatusOr<std::string> example_or = iterator->Next();
  EXPECT_THAT(example_or.status(), IsCode(absl::StatusCode::kOutOfRange));
}

TEST_F(OpStatsExampleStoreTest,
       SelectionCriteriaLastSuccessfulContributionDisabled) {
  // disable the feature but put in some matching entries.
  OpStatsExampleIteratorFactory iterator_factory =
      OpStatsExampleIteratorFactory(
          &mock_opstats_logger_, &mock_log_manager_,
          /*opstats_last_successful_contribution_criteria=*/false);

  OperationalStats included;
  included.set_task_name(kTestTaskName);
  included.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 900L));
  included.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED, 1000L));

  OperationalStats last_successful_contribution;
  last_successful_contribution.set_task_name(kTestTaskName);
  last_successful_contribution.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED, 1200L));
  last_successful_contribution.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED, 2001L));

  OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(included);
  *opstats_sequence.add_opstats() = std::move(last_successful_contribution);
  EXPECT_CALL(mock_db_, Read()).WillOnce(Return(opstats_sequence));

  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  OpStatsSelectionCriteria criteria;
  criteria.set_last_successful_contribution(true);
  selector.mutable_criteria()->PackFrom(criteria);
  absl::StatusOr<std::unique_ptr<ExampleIterator>> iterator_or =
      iterator_factory.CreateExampleIterator(selector);
  // Enabling last successful contribution in the criteria when it's not enabled
  // in the client returns INVALID_ARGUMENT.
  EXPECT_THAT(iterator_or.status(), IsCode(absl::StatusCode::kInvalidArgument));
}

TEST_F(OpStatsExampleStoreTest, FullSerialization) {
  OperationalStats stats;
  // Set singular fields
  std::string session = "session";
  std::string population = "population";
  std::string task_name = "task";
  std::string error = "error";
  int64_t chunking_layer_bytes_downloaded = 200;
  int64_t chunking_layer_bytes_uploaded = 600;
  int64_t network_duration_ms = 700;
  stats.set_session_name(session);
  stats.set_population_name(population);
  stats.set_task_name(task_name);
  stats.set_error_message(error);
  stats.set_chunking_layer_bytes_downloaded(chunking_layer_bytes_downloaded);
  stats.set_chunking_layer_bytes_uploaded(chunking_layer_bytes_uploaded);
  *stats.mutable_network_duration() =
      TimeUtil::MillisecondsToDuration(network_duration_ms);

  // Set two events
  OperationalStats::Event::EventKind event_kind_a =
      OperationalStats::Event::EVENT_KIND_COMPUTATION_STARTED;
  int64_t event_time_ms_a = 1000;
  OperationalStats::Event::EventKind event_kind_b =
      OperationalStats::Event::EVENT_KIND_COMPUTATION_FINISHED;
  int64_t event_time_ms_b = 1500;
  stats.mutable_events()->Add(CreateEvent(event_kind_a, event_time_ms_a));
  stats.mutable_events()->Add(CreateEvent(event_kind_b, event_time_ms_b));

  // Set two dataset stats
  std::string uri_a = "app:/train";
  int64_t num_examples_a = 10;
  int64_t example_bytes_a = 1000;
  std::string uri_b = "app:/test";
  int64_t num_examples_b = 5;
  int64_t example_bytes_b = 500;
  (*stats.mutable_dataset_stats())[uri_a] =
      CreateDatasetStats(num_examples_a, example_bytes_a);
  (*stats.mutable_dataset_stats())[uri_b] =
      CreateDatasetStats(num_examples_b, example_bytes_b);

  // Set retry window
  int64_t min_delay_ms = 5000;
  int64_t max_delay_ms = 9000;
  RetryWindow retry;
  retry.set_retry_token("token");
  *retry.mutable_delay_min() = TimeUtil::MillisecondsToDuration(min_delay_ms);
  *retry.mutable_delay_max() = TimeUtil::MillisecondsToDuration(max_delay_ms);
  *stats.mutable_retry_window() = retry;

  OpStatsSequence opstats_sequence;
  ::google::protobuf::Timestamp currentTime = TimeUtil::GetCurrentTime();
  *opstats_sequence.mutable_earliest_trustworthy_time() = currentTime;
  *opstats_sequence.add_opstats() = std::move(stats);
  EXPECT_CALL(mock_db_, Read()).WillOnce(Return(opstats_sequence));

  ExampleSelector selector;
  selector.set_collection_uri(kOpStatsCollectionUri);
  absl::StatusOr<std::unique_ptr<ExampleIterator>> iterator_or =
      iterator_factory_.CreateExampleIterator(selector);

  ASSERT_TRUE(iterator_or.ok());
  std::unique_ptr<ExampleIterator> iterator = std::move(iterator_or.value());
  absl::StatusOr<std::string> example_or = iterator->Next();
  ASSERT_TRUE(example_or.ok());
  tensorflow::Example example;
  example.ParseFromString(example_or.value());

  // Verify the example contains all the correct information.
  // Singular fields
  ASSERT_EQ(ExtractSingleString(example, kSessionName), session);
  ASSERT_EQ(ExtractSingleString(example, kPopulationName), population);
  ASSERT_EQ(ExtractSingleString(example, kTaskName), task_name);
  ASSERT_EQ(ExtractSingleString(example, kErrorMessage), error);
  ASSERT_EQ(ExtractSingleInt64(example, kChunkingLayerBytesDownloaded),
            chunking_layer_bytes_downloaded);
  ASSERT_EQ(ExtractSingleInt64(example, kChunkingLayerBytesUploaded),
            chunking_layer_bytes_uploaded);
  ASSERT_EQ(ExtractSingleInt64(example, kNetworkDuration), network_duration_ms);
  ASSERT_EQ(ExtractSingleInt64(example, kEarliestTrustWorthyTimeMillis),
            TimeUtil::TimestampToMilliseconds(currentTime));

  // Events
  auto event_types = ExtractRepeatedInt64(example, kEventsEventType);
  ASSERT_EQ(event_types.at(0), event_kind_a);
  ASSERT_EQ(event_types.at(1), event_kind_b);
  auto event_times = ExtractRepeatedInt64(example, kEventsTimestampMillis);
  ASSERT_EQ(event_times.at(0), event_time_ms_a);
  ASSERT_EQ(event_times.at(1), event_time_ms_b);

  // Dataset stats
  auto dataset_urls = ExtractRepeatedString(example, kDatasetStatsUri);
  // The order of the dataset stats doesn't matter, but should be consistent
  // across the individual features.
  int index_a = dataset_urls.at(1) == uri_a;
  ASSERT_EQ(dataset_urls.at(index_a), uri_a);
  ASSERT_EQ(dataset_urls.at(1 - index_a), uri_b);
  auto example_counts =
      ExtractRepeatedInt64(example, kDatasetStatsNumExamplesRead);
  ASSERT_EQ(example_counts.at(index_a), num_examples_a);
  ASSERT_EQ(example_counts.at(1 - index_a), num_examples_b);
  auto example_bytes = ExtractRepeatedInt64(example, kDatasetStatsNumBytesRead);
  ASSERT_EQ(example_bytes.at(index_a), example_bytes_a);
  ASSERT_EQ(example_bytes.at(1 - index_a), example_bytes_b);

  // RetryWindow
  ASSERT_EQ(ExtractSingleInt64(example, kRetryWindowDelayMinMillis),
            min_delay_ms);
  ASSERT_EQ(ExtractSingleInt64(example, kRetryWindowDelayMaxMillis),
            max_delay_ms);
}

}  // anonymous namespace
}  // namespace opstats
}  // namespace client
}  // namespace fcp
