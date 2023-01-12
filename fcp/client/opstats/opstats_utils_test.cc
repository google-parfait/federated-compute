/*
 * Copyright 2022 Google LLC
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

#include "fcp/client/opstats/opstats_utils.h"

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fcp/client/test_helpers.h"

namespace fcp {
namespace client {
namespace opstats {
namespace {

constexpr char kTaskName[] = "task";
OperationalStats::Event::EventKind kUploadStartedEvent =
    OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED;
OperationalStats::Event::EventKind kUploadServerAbortedEvent =
    OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_SERVER_ABORTED;

OperationalStats::Event CreateEvent(
    OperationalStats::Event::EventKind event_kind, int64_t event_time_seconds) {
  OperationalStats::Event event;
  event.set_event_type(event_kind);
  google::protobuf::Timestamp t;
  t.set_seconds(event_time_seconds);
  *event.mutable_timestamp() = t;
  return event;
}

TEST(OpStatsUtils,
     GetLastSuccessfulContributionTimeReturnsUploadStartedTimestamp) {
  OperationalStats stats;
  stats.set_task_name(kTaskName);

  int64_t upload_started_time_sec = 1000;
  stats.mutable_events()->Add(
      CreateEvent(kUploadStartedEvent, upload_started_time_sec));

  OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(stats);

  auto last_time =
      GetLastSuccessfulContributionTime(opstats_sequence, kTaskName);
  EXPECT_EQ(last_time->seconds(), upload_started_time_sec);
}

TEST(OpStatsUtils,
     GetLastSuccessfulContributionTimeReturnNotFoundForUnknownTask) {
  OperationalStats stats;
  stats.set_task_name(kTaskName);

  int64_t upload_started_time_sec = 1000;
  stats.mutable_events()->Add(
      CreateEvent(kUploadStartedEvent, upload_started_time_sec));

  OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(stats);
  EXPECT_FALSE(
      GetLastSuccessfulContributionTime(opstats_sequence, "task_name_not_found")
          .has_value());
}

TEST(OpStatsUtils,
     GetLastSuccessfulContributionTimeReturnsMostRecentUploadStartedTimestamp) {
  OpStatsSequence opstats_sequence;

  OperationalStats old_stats;
  old_stats.set_task_name(kTaskName);
  old_stats.mutable_events()->Add(CreateEvent(kUploadStartedEvent, 1000));
  *opstats_sequence.add_opstats() = std::move(old_stats);

  OperationalStats new_stats;
  new_stats.set_task_name(kTaskName);
  int64_t new_upload_started_sec = 2000;
  new_stats.mutable_events()->Add(
      CreateEvent(kUploadStartedEvent, new_upload_started_sec));
  *opstats_sequence.add_opstats() = std::move(new_stats);

  auto last_time =
      GetLastSuccessfulContributionTime(opstats_sequence, kTaskName);
  EXPECT_EQ(last_time->seconds(), new_upload_started_sec);
}

TEST(OpStatsUtils,
     GetLastSuccessfulContributionTimeReturnNotFoundIfAbortedByServer) {
  OperationalStats stats;
  stats.set_task_name(kTaskName);
  stats.mutable_events()->Add(CreateEvent(kUploadStartedEvent, 1000));
  stats.mutable_events()->Add(CreateEvent(kUploadServerAbortedEvent, 1001));

  OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(stats);

  EXPECT_FALSE(GetLastSuccessfulContributionTime(opstats_sequence, kTaskName)
                   .has_value());
}

TEST(OpStatsUtils,
     GetLastSuccessfulContributionTimeReturnOlderIfNewerAbortedByServer) {
  OpStatsSequence opstats_sequence;

  OperationalStats old_stats;
  old_stats.set_task_name(kTaskName);
  int64_t expected_time_sec = 1000;
  old_stats.mutable_events()->Add(
      CreateEvent(kUploadStartedEvent, expected_time_sec));
  *opstats_sequence.add_opstats() = std::move(old_stats);

  OperationalStats new_stats;
  new_stats.set_task_name(kTaskName);
  new_stats.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_STARTED, 2000));
  new_stats.mutable_events()->Add(CreateEvent(
      OperationalStats::Event::EVENT_KIND_RESULT_UPLOAD_SERVER_ABORTED, 2001));
  *opstats_sequence.add_opstats() = std::move(new_stats);

  auto last_time =
      GetLastSuccessfulContributionTime(opstats_sequence, kTaskName);
  EXPECT_EQ(last_time->seconds(), expected_time_sec);
}

TEST(OpStatsUtils,
     GetLastSuccessfulContributionTimeReturnNotFoundIfNoUploadStarted) {
  OperationalStats stats;
  stats.set_task_name(kTaskName);
  stats.mutable_events()->Add(
      CreateEvent(OperationalStats::Event::EVENT_KIND_CHECKIN_STARTED, 1000));

  OpStatsSequence opstats_sequence;
  *opstats_sequence.add_opstats() = std::move(stats);
  EXPECT_FALSE(GetLastSuccessfulContributionTime(opstats_sequence, kTaskName)
                   .has_value());
}

}  // namespace
}  // namespace opstats
}  // namespace client
}  // namespace fcp
