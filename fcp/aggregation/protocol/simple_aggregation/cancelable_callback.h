#ifndef FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_CANCELABLE_CALLBACK_H_
#define FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_CANCELABLE_CALLBACK_H_

#include <functional>
#include <memory>

#include "absl/time/time.h"
#include "fcp/base/clock.h"

namespace fcp::aggregation {

// Abstract interface for an object that can be canceled.
class Cancelable {
 public:
  virtual ~Cancelable() = default;
  virtual void Cancel() = 0;
};

using CancelationToken = std::shared_ptr<Cancelable>;

// Schedules a delayed callback that can be canceled by calling Cancel method on
// the CancelationToken.
CancelationToken ScheduleCallback(Clock* clock, absl::Duration delay,
                                  std::function<void()> callback);

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_CANCELABLE_CALLBACK_H_
