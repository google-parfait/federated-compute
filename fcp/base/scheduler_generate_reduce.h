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

#ifndef FCP_BASE_SCHEDULER_GENERATE_REDUCE_H_
#define FCP_BASE_SCHEDULER_GENERATE_REDUCE_H_

#include <atomic>
#include <memory>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/parallel_generate_sequential_reduce.h"
#include "fcp/base/reentrancy_guard.h"
#include "fcp/base/scheduler.h"

namespace fcp {

// Implementation of ParallelGenerateSequentialReduce based on fcp::Scheduler.
// Takes two Schedulers, one which is responsible for parallel execution and
// another for serial execution.
template <typename T>
class SchedulerGenerateReduce : public ParallelGenerateSequentialReduce<T> {
 public:
  SchedulerGenerateReduce(std::unique_ptr<Scheduler> parallel_scheduler,
                          std::unique_ptr<Scheduler> sequential_scheduler)
      : parallel_scheduler_(std::move(parallel_scheduler)),
        sequential_scheduler_(std::move(sequential_scheduler)) {}

  // SchedulerGenerateReduce is neither copyable nor movable.
  SchedulerGenerateReduce(const SchedulerGenerateReduce&) = delete;
  SchedulerGenerateReduce& operator=(const SchedulerGenerateReduce&) = delete;

  inline void WaitUntilIdle() {
    parallel_scheduler_->WaitUntilIdle();
    sequential_scheduler_->WaitUntilIdle();
  }

  CancellationToken Run(
      const std::vector<std::function<std::unique_ptr<T>()>>& generators,
      std::unique_ptr<T> initial_value,
      std::function<std::unique_ptr<T>(const T&, const T&)> accumulator,
      std::function<void(std::unique_ptr<T>)> on_complete) override;

 protected:
  // Virtual for testing
  virtual void RunParallel(std::function<void()> function);
  virtual void RunSequential(std::function<void()> function);

 private:
  class State : public CancellationImpl {
   public:
    State(size_t count, std::unique_ptr<T> initial_value)
        : remaining_count_(count),
          accumulated_value_(std::move(initial_value)) {}

    // CancellationImpl implementation
    void Cancel() override {
      mutex_.Lock();
      is_cancelled_ = true;
      while (active_count_ > 0) {
        inactive_cv_.Wait(&mutex_);
      }
      mutex_.Unlock();
    }

    bool IsCancelled() {
      absl::MutexLock lock(&mutex_);
      return is_cancelled_;
    }

    bool MaybeIncrementActiveCount() {
      absl::MutexLock lock(&mutex_);
      if (is_cancelled_) {
        return false;
      }
      active_count_++;
      return true;
    }

    void DescrementActiveCount() {
      absl::MutexLock lock(&mutex_);
      if (--active_count_ == 0 && is_cancelled_) {
        inactive_cv_.SignalAll();
      }
    }

    size_t DecrementRemainingCount() { return --remaining_count_; }

    std::unique_ptr<T>& accumulated_value() { return accumulated_value_; }
    void set_accumulated_value(std::unique_ptr<T> new_value) {
      accumulated_value_ = std::move(new_value);
    }

    std::atomic<bool>* in_sequential_call() { return &in_sequential_call_; }

   private:
    // Remaining number of tasks - accessed by sequential tasks only.
    size_t remaining_count_;
    // Accumulated value - accessed by sequential tasks only.
    std::unique_ptr<T> accumulated_value_;

    // Number of active calls to either callback function.
    size_t active_count_ ABSL_GUARDED_BY(mutex_) = 0;
    // This is set to true when the run is aborted.
    bool is_cancelled_ ABSL_GUARDED_BY(mutex_) = false;
    // Protectes active_count_ and cancelled_.
    absl::Mutex mutex_;
    // Used to notify cancellation about reaching inactive state;
    absl::CondVar inactive_cv_;
    // This is used by ReentrancyGuard to ensure that Sequential tasks are
    // indeed sequential.
    std::atomic<bool> in_sequential_call_ = false;
  };

  std::unique_ptr<Scheduler> parallel_scheduler_;
  std::unique_ptr<Scheduler> sequential_scheduler_;
};

// Run can be called multiple times, but only after a previous Run has completed
template <typename T>
CancellationToken SchedulerGenerateReduce<T>::Run(
    const std::vector<std::function<std::unique_ptr<T>()>>& generators,
    std::unique_ptr<T> initial_value,
    std::function<std::unique_ptr<T>(const T&, const T&)> accumulator,
    std::function<void(std::unique_ptr<T>)> on_complete) {
  FCP_CHECK(!generators.empty());
  FCP_CHECK(initial_value);

  auto state =
      std::make_shared<State>(generators.size(), std::move(initial_value));

  for (const auto& generator : generators) {
    RunParallel([=] {
      if (!state->MaybeIncrementActiveCount()) {
        return;
      }
      auto partial = generator();
      FCP_CHECK(partial);
      state->DescrementActiveCount();
      if (state->IsCancelled()) {
        return;
      }

      RunSequential([=, partial = std::shared_ptr<T>(partial.release())] {
        ReentrancyGuard guard;
        FCP_CHECK_STATUS(guard.Check(state->in_sequential_call()));

        if (!state->MaybeIncrementActiveCount()) {
          return;
        }
        auto new_value = accumulator(*state->accumulated_value(), *partial);
        FCP_CHECK(new_value);
        state->set_accumulated_value(std::move(new_value));

        if (state->DecrementRemainingCount() == 0) {
          on_complete(std::move(state->accumulated_value()));
        }
        state->DescrementActiveCount();
      });
    });
  }

  return state;
}

template <typename T>
void SchedulerGenerateReduce<T>::RunParallel(std::function<void()> function) {
  parallel_scheduler_->Schedule(function);
}

template <typename T>
void SchedulerGenerateReduce<T>::RunSequential(std::function<void()> function) {
  sequential_scheduler_->Schedule(function);
}

}  // namespace fcp

#endif  // FCP_BASE_SCHEDULER_GENERATE_REDUCE_H_
