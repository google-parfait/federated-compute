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
