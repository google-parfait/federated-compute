// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fcp/aggregation/protocol/simple_aggregation/cancelable_callback.h"

#include "gtest/gtest.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "fcp/base/clock.h"
#include "fcp/base/simulated_clock.h"

namespace fcp::aggregation {
namespace {

TEST(CancelableCallbackTest, SimulatedClockCallback) {
  SimulatedClock clock;
  bool called_back = false;
  auto token = ScheduleCallback(&clock, absl::Seconds(2),
                                [&called_back]() { called_back = true; });
  clock.AdvanceTime(absl::Seconds(1));
  EXPECT_FALSE(called_back);
  clock.AdvanceTime(absl::Seconds(1));
  EXPECT_TRUE(called_back);
}

TEST(CancelableCallbackTest, SimulatedClockCancelation) {
  SimulatedClock clock;
  bool called_back = false;
  auto token = ScheduleCallback(&clock, absl::Seconds(2),
                                [&called_back]() { called_back = true; });
  clock.AdvanceTime(absl::Seconds(1));
  token->Cancel();
  clock.AdvanceTime(absl::Seconds(1));
  EXPECT_FALSE(called_back);
}

TEST(CancelableCallbackTest, RealClockCallback) {
  absl::Notification notification;
  auto token = ScheduleCallback(Clock::RealClock(), absl::Milliseconds(1),
                                [&notification]() { notification.Notify(); });
  notification.WaitForNotification();
  EXPECT_TRUE(notification.HasBeenNotified());
}

TEST(CancelableCallbackTest, RealClockCancelation) {
  absl::Notification notification;
  auto token = ScheduleCallback(Clock::RealClock(), absl::Seconds(1),
                                [&notification]() { notification.Notify(); });
  token->Cancel();
  EXPECT_FALSE(notification.WaitForNotificationWithTimeout(absl::Seconds(2)));
}

}  // namespace
}  // namespace fcp::aggregation
