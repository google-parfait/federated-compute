#ifndef FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_LATENCY_AGGREGATOR_H_
#define FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_LATENCY_AGGREGATOR_H_

#include <cmath>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"

namespace fcp::aggregation {

// Utility class used to calculate mean and standard deviation of client
// participation in the Aggregation Protocol.
//
// This class implements Welford's online algorithm:
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
//
// This class isn't thread-safe. The caller must ensure that the
// calls are strictly sequential.
class LatencyAggregator final {
 public:
  void Add(absl::Duration latency) {
    double latency_sec = absl::ToDoubleSeconds(latency);
    count_++;
    double delta = latency_sec - mean_;
    mean_ += delta / count_;
    double delta2 = latency_sec - mean_;
    sum_of_squares_ += delta * delta2;
  }

  size_t GetCount() const { return count_; }

  absl::Duration GetMean() const { return absl::Seconds(mean_); }

  absl::StatusOr<absl::Duration> GetStandardDeviation() const {
    if (count_ < 2) {
      return absl::FailedPreconditionError(
          "At least 2 latency samples required");
    }
    return absl::Seconds(std::sqrt(sum_of_squares_ / (count_ - 1)));
  }

 private:
  size_t count_ = 0;
  double mean_ = 0.0;
  double sum_of_squares_ = 0.0;
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_LATENCY_AGGREGATOR_H_
