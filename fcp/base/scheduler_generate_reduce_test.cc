/*
 * Copyright 2019 Google LLC
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

#include "fcp/base/scheduler_generate_reduce.h"

#include <atomic>
#include <functional>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/scheduler.h"

namespace fcp {
namespace {

using ::testing::_;
using ::testing::Eq;
using ::testing::IsFalse;
using ::testing::Lt;
using ::testing::StrictMock;
using ::testing::Test;

class MockScheduler : public Scheduler {
 public:
  MOCK_METHOD(void, Schedule, (std::function<void()>), (override));
  MOCK_METHOD(void, WaitUntilIdle, ());
};

// Wrap int in a struct to keep Clang-tidy happy.
struct Integer {
  Integer() : value(0) {}
  explicit Integer(int v) : value(v) {}
  int value;
};

std::vector<std::function<std::unique_ptr<Integer>()>> IntGenerators(int n) {
  std::vector<std::function<std::unique_ptr<Integer>()>> generators;
  for (int i = 1; i <= n; ++i) {
    generators.emplace_back([i]() { return std::make_unique<Integer>(i); });
  }
  return generators;
}

constexpr auto multiply_accumulator = [](const Integer& l, const Integer& r) {
  return std::make_unique<Integer>(l.value * r.value);
};
constexpr auto call_fn = [](const std::function<void()>& f) { f(); };

TEST(SchedulerGenerateReduceTest, SingleCall) {
  auto parallel_scheduler = std::make_unique<StrictMock<MockScheduler>>();
  auto sequential_scheduler = std::make_unique<StrictMock<MockScheduler>>();

  EXPECT_CALL(*parallel_scheduler, Schedule(_))
      .Times(6)
      .WillRepeatedly(call_fn);
  EXPECT_CALL(*sequential_scheduler, Schedule(_))
      .Times(6)
      .WillRepeatedly(call_fn);

  // Technically unsafe, but we know the pointers will be valid as long as
  // runner is alive.
  SchedulerGenerateReduce<Integer> runner(std::move(parallel_scheduler),
                                          std::move(sequential_scheduler));

  std::vector<std::function<std::unique_ptr<Integer>()>> generators =
      IntGenerators(6);

  Integer result;
  runner.Run(generators, std::make_unique<Integer>(1), multiply_accumulator,
             [&result](std::unique_ptr<Integer> r) { result = *r; });

  EXPECT_THAT(result.value, Eq(720));  // 6! = 720
}

TEST(SchedulerGenerateReduceTest, TwoCalls) {
  auto parallel_scheduler = std::make_unique<MockScheduler>();
  auto sequential_scheduler = std::make_unique<MockScheduler>();

  EXPECT_CALL(*parallel_scheduler, Schedule(_)).WillRepeatedly(call_fn);
  EXPECT_CALL(*sequential_scheduler, Schedule(_)).WillRepeatedly(call_fn);

  // Technically unsafe, but we know the pointers will be valid as long as
  // runner is alive.
  SchedulerGenerateReduce<Integer> runner(std::move(parallel_scheduler),
                                          std::move(sequential_scheduler));

  // First call
  std::vector<std::function<std::unique_ptr<Integer>()>> generators =
      IntGenerators(6);

  Integer result;
  runner.Run(generators, std::make_unique<Integer>(1), multiply_accumulator,
             [&result](std::unique_ptr<Integer> r) { result = *r; });

  EXPECT_THAT(result.value, Eq(720));  // 6! = 720

  // Second call
  std::vector<std::function<std::unique_ptr<Integer>()>> generators2 =
      IntGenerators(4);

  runner.Run(generators2, std::make_unique<Integer>(1), multiply_accumulator,
             [&result](std::unique_ptr<Integer> r) { result = *r; });

  EXPECT_THAT(result.value, Eq(24));  // 4! = 24
}

TEST(SchedulerGenerateReduceAbortTest, Abort) {
  auto parallel_scheduler = fcp::CreateThreadPoolScheduler(4);
  auto sequential_scheduler = fcp::CreateThreadPoolScheduler(1);

  absl::Notification signal_abort;
  std::atomic<int> callback_counter = 0;

  std::vector<std::function<std::unique_ptr<Integer>()>> generators;
  for (int i = 1; i <= 100; ++i) {
    generators.emplace_back([&, i]() {
      callback_counter++;
      // Signal abort when running 10th parallel task
      if (i == 10) {
        signal_abort.Notify();
      }
      absl::SleepFor(absl::Milliseconds(1));
      return std::make_unique<Integer>(i);
    });
  }

  auto accumulator = [&](const Integer& l, const Integer& r) {
    callback_counter++;
    return std::make_unique<Integer>(l.value * r.value);
  };

  SchedulerGenerateReduce<Integer> runner(std::move(parallel_scheduler),
                                          std::move(sequential_scheduler));
  bool final_callback_called = false;
  CancellationToken cancellation_token = runner.Run(
      generators, std::make_unique<Integer>(1), accumulator,
      [&](std::unique_ptr<Integer> r) { final_callback_called = true; });

  signal_abort.WaitForNotification();
  cancellation_token->Cancel();

  int count_after_abort = callback_counter.load();
  FCP_LOG(INFO) << "count_after_abort = " << count_after_abort;

  // Wait for all scheduled tasks to finish
  runner.WaitUntilIdle();

  // The final number of callbacks should not change since returning from
  // Abort.
  int final_count = callback_counter.load();
  EXPECT_THAT(final_count, Eq(count_after_abort));
  EXPECT_THAT(final_count, Lt(generators.size()));
  EXPECT_THAT(final_callback_called, IsFalse());
}

}  // namespace
}  // namespace fcp
