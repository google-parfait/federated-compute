#include "fcp/aggregation/protocol/simple_aggregation/cancelable_callback.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/time/time.h"
#include "fcp/base/clock.h"

namespace fcp::aggregation {

namespace {

class CancelableWaiter : public Cancelable, public Clock::Waiter {
 public:
  explicit CancelableWaiter(std::function<void()> callback)
      : callback_(std::move(callback)) {}
  ~CancelableWaiter() override = default;

 private:
  void Cancel() override {
    absl::MutexLock lock(&mu_);
    callback_ = nullptr;
  }

  void WakeUp() override {
    absl::MutexLock lock(&mu_);
    if (callback_ != nullptr) {
      callback_();
      callback_ = nullptr;
    }
  }

  absl::Mutex mu_;
  std::function<void()> callback_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

CancelationToken ScheduleCallback(Clock* clock, absl::Duration delay,
                                  std::function<void()> callback) {
  auto waiter = std::make_shared<CancelableWaiter>(std::move(callback));
  clock->WakeupWithDeadline(clock->Now() + delay, waiter);
  return waiter;
}

}  // namespace fcp::aggregation
