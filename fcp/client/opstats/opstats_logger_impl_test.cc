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

#include "fcp/client/opstats/opstats_logger_impl.h"

#include <filesystem>
#include <string>
#include <utility>

#include "google/protobuf/util/time_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/diag_codes.pb.h"
#include "fcp/client/histogram_counters.pb.h"
#include "fcp/client/opstats/pds_backed_opstats_db.h"
#include "fcp/client/test_helpers.h"
#include "fcp/protos/opstats.pb.h"
#include "fcp/testing/testing.h"

namespace fcp {
namespace client {
namespace opstats {
namespace {

using google::internal::federatedml::v2::RetryWindow;
using google::protobuf::Timestamp;
using google::protobuf::util::TimeUtil;
using testing::Ge;
using testing::Return;
using testing::StrictMock;

constexpr char kSessionName[] = "SESSION";
constexpr char kPopulationName[] = "POPULATION";
constexpr char kTaskName[] = "TASK";

class OpStatsLoggerImplTest : public testing::Test {
 protected:
  void SetUp() override {
    ON_CALL(mock_flags_, enable_opstats()).WillByDefault(Return(true));
    ON_CALL(mock_flags_, opstats_ttl_days()).WillByDefault(Return(1));
    ON_CALL(mock_flags_, opstats_db_size_limit_bytes())
        .WillByDefault(Return(1 * 1024 * 1024));
    base_dir_ = testing::TempDir();
  }

  void TearDown() override {
    auto db = PdsBackedOpStatsDb::Create(
        base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
        mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
    ASSERT_OK(db);
    EXPECT_CALL(
        mock_log_manager_,
        LogToLongHistogram(OPSTATS_DB_SIZE_BYTES, /*execution_index=*/0,
                           /*epoch_index=*/0, engine::DataSourceType::DATASET,
                           /*value=*/0));
    EXPECT_CALL(
        mock_log_manager_,
        LogToLongHistogram(HistogramCounters::OPSTATS_DB_NUM_ENTRIES,
                           /*execution_index=*/0, /*epoch_index=*/0,
                           engine::DataSourceType::DATASET, /*value=*/0));
    EXPECT_THAT((*db)->Transform([](OpStatsSequence& data) { data.Clear(); }),
                IsOk());
  }

  std::unique_ptr<OpStatsLogger> CreateOpStatsLoggerImpl(
      const std::string& session_name, const std::string& population_name) {
    auto db = PdsBackedOpStatsDb::Create(
        base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
        mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
    FCP_CHECK(db.ok());
    return std::make_unique<OpStatsLoggerImpl>(std::move(*db),
                                               &mock_log_manager_, &mock_flags_,
                                               session_name, population_name);
  }

  // Checks that the expected and actual protos are equivalent, ignoring the
  // timestamps in the actual proto, which must also be increasing.
  void CheckEqualProtosAndIncreasingTimestamps(const Timestamp& start_time,
                                               const OpStatsSequence& expected,
                                               OpStatsSequence actual) {
    auto previous_timestamp = start_time;
    for (auto& opstats : *actual.mutable_opstats()) {
      for (auto& event : *opstats.mutable_events()) {
        EXPECT_GE(event.timestamp(), previous_timestamp);
        previous_timestamp = event.timestamp();
        // Remove the timestamp
        event.clear_timestamp();
      }
    }
    actual.clear_earliest_trustworthy_time();
    EXPECT_THAT(actual, EqualsProto(expected));
  }

  void ExpectOpstatsEnabledEvents(int num_opstats_loggers) {
    ExpectOpstatsEnabledEvents(num_opstats_loggers, num_opstats_loggers);
  }

  void ExpectOpstatsEnabledEvents(int num_opstats_loggers,
                                  int num_opstats_commits) {
    EXPECT_CALL(mock_log_manager_,
                LogDiag(DebugDiagCode::TRAINING_OPSTATS_ENABLED))
        .Times(num_opstats_loggers);
    // Logged when the class is initialized.
    EXPECT_CALL(mock_log_manager_,
                LogDiag(ProdDiagCode::OPSTATS_DB_COMMIT_EXPECTED))
        .Times(num_opstats_loggers);
    EXPECT_CALL(mock_log_manager_,
                LogDiag(ProdDiagCode::OPSTATS_DB_COMMIT_ATTEMPTED))
        .Times(num_opstats_commits);
    EXPECT_CALL(
        mock_log_manager_,
        LogToLongHistogram(TRAINING_OPSTATS_COMMIT_LATENCY,
                           /*execution_index=*/0, /*epoch_index=*/0,
                           engine::DataSourceType::DATASET, /*value=*/Ge(0)))
        .Times(num_opstats_commits);
    EXPECT_CALL(
        mock_log_manager_,
        LogToLongHistogram(OPSTATS_DB_SIZE_BYTES, /*execution_index=*/0,
                           /*epoch_index=*/0, engine::DataSourceType::DATASET,
                           /*value=*/Ge(0)))
        .Times(num_opstats_commits);
    EXPECT_CALL(
        mock_log_manager_,
        LogToLongHistogram(OPSTATS_DB_NUM_ENTRIES, /*execution_index=*/0,
                           /*epoch_index=*/0, engine::DataSourceType::DATASET,
                           /*value=*/Ge(0)))
        .Times(num_opstats_commits);
  }

  RetryWindow CreateRetryWindow(const std::string& retry_token,
                                int64_t delay_min_seconds,
                                int64_t delay_max_seconds) {
    RetryWindow retry_window;
    retry_window.set_retry_token(retry_token);
    retry_window.mutable_delay_min()->set_seconds(delay_min_seconds);
    retry_window.mutable_delay_max()->set_seconds(delay_max_seconds);
    return retry_window;
  }

  std::string base_dir_;
  MockFlags mock_flags_;
  StrictMock<MockLogManager> mock_log_manager_;
};

TEST_F(OpStatsLoggerImplTest, SetTaskName) {
  auto start_time = TimeUtil::GetCurrentTime();
  ExpectOpstatsEnabledEvents(/*num_opstats_loggers=*/3);

  auto opstats_logger = CreateOpStatsLoggerImpl(kSessionName, kPopulationName);
  opstats_logger->AddEventAndSetTaskName(
      kTaskName, OperationalStats::Event::EVENT_KIND_CHECKIN_ACCEPTED);

  opstats_logger.reset();

  auto opstats_logger_no_population =
      CreateOpStatsLoggerImpl(kSessionName,
                              /*population_name=*/"");
  opstats_logger_no_population->AddEventAndSetTaskName(
      kTaskName, OperationalStats::Event::EVENT_KIND_CHECKIN_ACCEPTED);

  opstats_logger_no_population.reset();

  auto opstats_logger_no_session =
      CreateOpStatsLoggerImpl(/*session_name=*/"", kPopulationName);
  opstats_logger_no_session->AddEventAndSetTaskName(
      kTaskName, OperationalStats::Event::EVENT_KIND_CHECKIN_ACCEPTED);

  opstats_logger_no_session.reset();

  auto db = PdsBackedOpStatsDb::Create(
      base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
      mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
  ASSERT_OK(db);
  auto data = (*db)->Read();
  ASSERT_OK(data);

  OpStatsSequence expected;
  // Add the first run
  auto new_opstats = expected.add_opstats();
  new_opstats->set_session_name(kSessionName);
  new_opstats->set_population_name(kPopulationName);
  new_opstats->set_task_name(kTaskName);
  new_opstats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_CHECKIN_ACCEPTED);
  // Add the second run
  new_opstats = expected.add_opstats();
  new_opstats->set_session_name(kSessionName);
  new_opstats->set_task_name(kTaskName);
  new_opstats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_CHECKIN_ACCEPTED);
  // Add the third run
  new_opstats = expected.add_opstats();
  new_opstats->set_population_name(kPopulationName);
  new_opstats->set_task_name(kTaskName);
  new_opstats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_CHECKIN_ACCEPTED);

  CheckEqualProtosAndIncreasingTimestamps(start_time, expected, *data);
}

TEST_F(OpStatsLoggerImplTest, NewRunAfterCorruption) {
  auto start_time = TimeUtil::GetCurrentTime();
  ExpectOpstatsEnabledEvents(/*num_opstats_loggers=*/2);

  auto opstats_logger = CreateOpStatsLoggerImpl(kSessionName, kPopulationName);
  opstats_logger->AddEventAndSetTaskName(
      kTaskName, OperationalStats::Event::EVENT_KIND_CHECKIN_ACCEPTED);
  opstats_logger.reset();

  // Make the db file corrupt
  {
    std::filesystem::path db_path(base_dir_);
    db_path /= PdsBackedOpStatsDb::kParentDir;
    db_path /= PdsBackedOpStatsDb::kDbFileName;
    protostore::FileStorage file_storage;
    std::unique_ptr<protostore::OutputStream> ostream =
        file_storage.OpenForWrite(db_path).value();
    EXPECT_THAT(ostream->Append("not a proto"), IsOk());
    EXPECT_THAT(ostream->Close(), IsOk());
  }

  EXPECT_CALL(mock_log_manager_, LogDiag(ProdDiagCode::OPSTATS_READ_FAILED));
  auto opstats_logger_no_population =
      CreateOpStatsLoggerImpl(kSessionName,
                              /*population_name=*/"");
  opstats_logger_no_population->AddEventAndSetTaskName(
      kTaskName, OperationalStats::Event::EVENT_KIND_CHECKIN_ACCEPTED);

  opstats_logger_no_population.reset();

  auto db = PdsBackedOpStatsDb::Create(
      base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
      mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
  ASSERT_OK(db);
  auto data = (*db)->Read();
  ASSERT_OK(data);

  // Expect only the second run to be represented in the db.
  OpStatsSequence expected;
  auto new_opstats = expected.add_opstats();
  new_opstats->set_session_name(kSessionName);
  new_opstats->set_task_name(kTaskName);
  new_opstats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_CHECKIN_ACCEPTED);
  CheckEqualProtosAndIncreasingTimestamps(start_time, expected, *data);
}

TEST_F(OpStatsLoggerImplTest, AddEvent) {
  auto start_time = TimeUtil::GetCurrentTime();
  ExpectOpstatsEnabledEvents(/*num_opstats_loggers=*/2);

  auto opstats_logger = CreateOpStatsLoggerImpl(kSessionName, kPopulationName);
  opstats_logger->AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED);
  opstats_logger.reset();

  auto opstats_logger_no_population =
      CreateOpStatsLoggerImpl(kSessionName,
                              /*population_name=*/"");
  opstats_logger_no_population->AddEvent(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED);
  opstats_logger_no_population->AddEvent(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED);
  opstats_logger_no_population.reset();

  auto db = PdsBackedOpStatsDb::Create(
      base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
      mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
  ASSERT_OK(db);
  auto data = (*db)->Read();
  ASSERT_OK(data);

  OpStatsSequence expected;
  // Add the first run
  auto new_opstats = expected.add_opstats();
  new_opstats->set_session_name(kSessionName);
  new_opstats->set_population_name(kPopulationName);
  new_opstats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED);
  // Add the second run
  new_opstats = expected.add_opstats();
  new_opstats->set_session_name(kSessionName);
  new_opstats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED);
  new_opstats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED);

  CheckEqualProtosAndIncreasingTimestamps(start_time, expected, *data);
}

TEST_F(OpStatsLoggerImplTest, AddEventAfterTtl) {
  auto start_time = TimeUtil::GetCurrentTime();
  ExpectOpstatsEnabledEvents(/*num_opstats_loggers=*/2);

  // Set the ttl to 0 so that previous data will be wiped out each time the
  // logger tries to commit new data.
  EXPECT_CALL(mock_flags_, opstats_ttl_days()).WillRepeatedly(Return(0));
  auto opstats_logger = CreateOpStatsLoggerImpl(kSessionName, kPopulationName);
  opstats_logger->AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED);
  opstats_logger.reset();

  auto opstats_logger_no_population =
      CreateOpStatsLoggerImpl(kSessionName,
                              /*population_name=*/"");
  opstats_logger_no_population->AddEvent(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED);
  opstats_logger_no_population->AddEvent(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED);
  opstats_logger_no_population.reset();

  auto db = PdsBackedOpStatsDb::Create(
      base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
      mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
  auto data = (*db)->Read();
  ASSERT_OK(data);

  // Expect the db to contain only data associated with the second run. The
  // second run should be complete, however.
  OpStatsSequence expected;
  auto new_opstats = expected.add_opstats();
  new_opstats->set_session_name(kSessionName);
  new_opstats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED);
  new_opstats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED);

  CheckEqualProtosAndIncreasingTimestamps(start_time, expected, *data);
}

TEST_F(OpStatsLoggerImplTest, AddEventWithErrorMessage) {
  auto start_time = TimeUtil::GetCurrentTime();
  ExpectOpstatsEnabledEvents(/*num_opstats_loggers=*/1);

  auto opstats_logger = CreateOpStatsLoggerImpl(kSessionName, kPopulationName);
  opstats_logger->AddEventWithErrorMessage(
      OperationalStats::Event::EVENT_KIND_ERROR_IO, "first error");
  opstats_logger->AddEventWithErrorMessage(
      OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW, "second error");
  opstats_logger.reset();

  auto db = PdsBackedOpStatsDb::Create(
      base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
      mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
  auto data = (*db)->Read();
  ASSERT_OK(data);

  OpStatsSequence expected;
  auto new_opstats = expected.add_opstats();
  new_opstats->set_session_name(kSessionName);
  new_opstats->set_population_name(kPopulationName);
  new_opstats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_ERROR_IO);
  new_opstats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_ERROR_TENSORFLOW);
  new_opstats->set_error_message("first error");

  CheckEqualProtosAndIncreasingTimestamps(start_time, expected, *data);
}

TEST_F(OpStatsLoggerImplTest, UpdateDatasetStats) {
  ExpectOpstatsEnabledEvents(/*num_opstats_loggers=*/1);

  auto opstats_logger = CreateOpStatsLoggerImpl(kSessionName, kPopulationName);
  const std::string kCollectionUri = "app:/collection_uri";
  const std::string kCollectionUriOther = "app:/collection_uri_other";
  opstats_logger->UpdateDatasetStats(kCollectionUri,
                                     /*additional_example_count=*/100,
                                     /*additional_example_size_bytes=*/1000);
  opstats_logger->UpdateDatasetStats(kCollectionUriOther,
                                     /*additional_example_count=*/200,
                                     /*additional_example_size_bytes=*/2000);
  opstats_logger->UpdateDatasetStats(kCollectionUri,
                                     /*additional_example_count=*/300,
                                     /*additional_example_size_bytes=*/3000);
  opstats_logger.reset();

  auto db = PdsBackedOpStatsDb::Create(
      base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
      mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
  auto data = (*db)->Read();
  ASSERT_OK(data);

  OpStatsSequence expected;
  auto new_opstats = expected.add_opstats();
  new_opstats->set_session_name(kSessionName);
  new_opstats->set_population_name(kPopulationName);
  OperationalStats::DatasetStats dataset_stats;
  dataset_stats.set_num_examples_read(400);
  dataset_stats.set_num_bytes_read(4000);
  (*new_opstats->mutable_dataset_stats())[kCollectionUri] =
      std::move(dataset_stats);
  OperationalStats::DatasetStats dataset_stats_other;
  dataset_stats_other.set_num_examples_read(200);
  dataset_stats_other.set_num_bytes_read(2000);
  (*new_opstats->mutable_dataset_stats())[kCollectionUriOther] =
      std::move(dataset_stats_other);

  (*data).clear_earliest_trustworthy_time();
  EXPECT_THAT(*data, EqualsProto(expected));
}

TEST_F(OpStatsLoggerImplTest, SetNetworkStats) {
  ExpectOpstatsEnabledEvents(/*num_opstats_loggers=*/1);

  auto opstats_logger = CreateOpStatsLoggerImpl(kSessionName, kPopulationName);
  opstats_logger->SetNetworkStats(
      {.bytes_downloaded = 102,
       .bytes_uploaded = 103,
       .network_duration = absl::Milliseconds(104)});
  opstats_logger->SetNetworkStats(
      {.bytes_downloaded = 202,
       .bytes_uploaded = 203,
       .network_duration = absl::Milliseconds(204)});
  opstats_logger.reset();

  auto db = PdsBackedOpStatsDb::Create(
      base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
      mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
  ASSERT_OK(db);
  auto data = (*db)->Read();
  ASSERT_OK(data);

  OpStatsSequence expected;
  auto new_opstats = expected.add_opstats();
  new_opstats->set_session_name(kSessionName);
  new_opstats->set_population_name(kPopulationName);
  // The bytes_downloaded/bytes_uploaded fields should not be set anymore
  new_opstats->set_chunking_layer_bytes_downloaded(202);
  new_opstats->set_chunking_layer_bytes_uploaded(203);
  // The new network_duration field should be set now.
  new_opstats->mutable_network_duration()->set_nanos(
      static_cast<int32_t>(absl::ToInt64Nanoseconds(absl::Milliseconds(204))));

  (*data).clear_earliest_trustworthy_time();
  EXPECT_THAT(*data, EqualsProto(expected));
}

TEST_F(OpStatsLoggerImplTest, SetRetryWindow) {
  ExpectOpstatsEnabledEvents(/*num_opstats_loggers=*/1);

  auto opstats_logger = CreateOpStatsLoggerImpl(kSessionName, kPopulationName);
  opstats_logger->SetRetryWindow(CreateRetryWindow("retry_token", 100, 200));
  opstats_logger->SetRetryWindow(CreateRetryWindow("retry_token", 300, 400));
  opstats_logger.reset();

  auto db = PdsBackedOpStatsDb::Create(
      base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
      mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
  ASSERT_OK(db);
  auto data = (*db)->Read();
  ASSERT_OK(data);

  OpStatsSequence expected;
  auto new_opstats = expected.add_opstats();
  new_opstats->set_session_name(kSessionName);
  new_opstats->set_population_name(kPopulationName);
  *new_opstats->mutable_retry_window() =
      CreateRetryWindow(/*retry_token=*/"", 300, 400);

  (*data).clear_earliest_trustworthy_time();
  EXPECT_THAT(*data, EqualsProto(expected));
}

TEST_F(OpStatsLoggerImplTest, AddEventCommitAddMoreEvents) {
  auto start_time = TimeUtil::GetCurrentTime();
  ExpectOpstatsEnabledEvents(
      /*num_opstats_loggers=*/2, /*num_opstats_commits=*/4);

  auto opstats_logger = CreateOpStatsLoggerImpl(kSessionName, kPopulationName);
  opstats_logger->AddEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED);
  opstats_logger.reset();

  auto opstats_logger_no_population =
      CreateOpStatsLoggerImpl(kSessionName,
                              /*population_name=*/"");
  opstats_logger_no_population->AddEvent(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED);
  ASSERT_OK(opstats_logger_no_population->CommitToStorage());
  opstats_logger_no_population->AddEvent(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED);
  ASSERT_OK(opstats_logger_no_population->CommitToStorage());
  opstats_logger_no_population->AddEvent(
      OperationalStats::Event::EVENT_KIND_TRAIN_NOT_STARTED);
  opstats_logger_no_population.reset();

  auto db = PdsBackedOpStatsDb::Create(
      base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
      mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
  ASSERT_OK(db);
  auto data = (*db)->Read();
  ASSERT_OK(data);

  OpStatsSequence expected;
  // Add the first run
  auto second_run = expected.add_opstats();
  second_run->set_session_name(kSessionName);
  second_run->set_population_name(kPopulationName);
  second_run->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED);
  // Add the second run
  second_run = expected.add_opstats();
  second_run->set_session_name(kSessionName);
  second_run->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED);
  second_run->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED);
  second_run->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_TRAIN_NOT_STARTED);

  CheckEqualProtosAndIncreasingTimestamps(start_time, expected, *data);
}

TEST_F(OpStatsLoggerImplTest, MisconfiguredTtlMultipleCommit) {
  auto start_time = TimeUtil::GetCurrentTime();
  ExpectOpstatsEnabledEvents(/*num_opstats_loggers=*/1,
                             /*num_opstats_commits*/ 3);
  auto db_zero_ttl = PdsBackedOpStatsDb::Create(
                         base_dir_, absl::ZeroDuration(), mock_log_manager_,
                         mock_flags_.opstats_db_size_limit_bytes())
                         .value();
  auto opstats_logger = std::make_unique<OpStatsLoggerImpl>(
      std::move(db_zero_ttl), &mock_log_manager_, &mock_flags_, kSessionName,
      kPopulationName);

  opstats_logger->AddEvent(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED);
  ASSERT_OK(opstats_logger->CommitToStorage());
  opstats_logger->AddEvent(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED);
  ASSERT_OK(opstats_logger->CommitToStorage());
  opstats_logger->AddEvent(
      OperationalStats::Event::EVENT_KIND_TRAIN_NOT_STARTED);
  opstats_logger.reset();

  auto db = PdsBackedOpStatsDb::Create(
      base_dir_, mock_flags_.opstats_ttl_days() * absl::Hours(24),
      mock_log_manager_, mock_flags_.opstats_db_size_limit_bytes());
  ASSERT_OK(db);
  auto data = (*db)->Read();
  ASSERT_OK(data);

  // Even though we had corruption in the middle of the run, it should be ok
  // because we committed the entire history successfully at the end.
  OpStatsSequence expected;
  auto expected_stats = expected.add_opstats();
  expected_stats->set_population_name(kPopulationName);
  expected_stats->set_session_name(kSessionName);
  expected_stats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_CHECKIN_STARTED);
  expected_stats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_ELIGIBILITY_REJECTED);
  expected_stats->add_events()->set_event_type(
      OperationalStats::Event::EVENT_KIND_TRAIN_NOT_STARTED);

  CheckEqualProtosAndIncreasingTimestamps(start_time, expected, *data);
}

}  // anonymous namespace
}  // namespace opstats
}  // namespace client
}  // namespace fcp
