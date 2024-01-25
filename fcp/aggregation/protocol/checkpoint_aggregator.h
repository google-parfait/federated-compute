/*
 * Copyright 2024 Google LLC
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

#ifndef FCP_AGGREGATION_PROTOCOL_CHECKPOINT_AGGREGATOR_H_
#define FCP_AGGREGATION_PROTOCOL_CHECKPOINT_AGGREGATOR_H_

#include <stdbool.h>

#include <atomic>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor_aggregator.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"
#include "fcp/aggregation/protocol/configuration.pb.h"

namespace fcp {
namespace aggregation {

class CheckpointAggregator {
 public:
  ~CheckpointAggregator();

  // Not copyable or moveable.
  CheckpointAggregator(const CheckpointAggregator&) = delete;
  CheckpointAggregator& operator=(const CheckpointAggregator&) = delete;

  // Validates the Configuration that will subsequently be used to create an
  // instance of CheckpointAggregator.
  // Returns INVALID_ARGUMENT if the configuration is invalid.
  static absl::Status ValidateConfig(const Configuration& configuration);

  // Creates an instance of CheckpointAggregator.
  static absl::StatusOr<std::unique_ptr<CheckpointAggregator>> Create(
      const Configuration& configuration);

  // Accumulates a checkpoint via nested tensor aggregators. The tensors are
  // provided by the CheckpointParser instance.
  absl::Status Accumulate(CheckpointParser& checkpoint_parser);
  // Merges with another compatible instance of CheckpointAggregator consuming
  // it in the process.
  absl::Status MergeWith(CheckpointAggregator&& other);
  // Returns true if the report can be processed.
  bool CanReport() const;
  // Builds the report using the supplied CheckpointBuilder instance.
  absl::Status Report(CheckpointBuilder& checkpoint_builder);
  // Signal that the aggregation must be aborted and the report can't be
  // produced.
  void Abort();

 private:
  CheckpointAggregator(
      std::vector<Intrinsic> intrinsics,
      std::vector<std::unique_ptr<TensorAggregator>> aggregators);

  // Creates an aggregation intrinsic based on the intrinsic configuration.
  static absl::StatusOr<std::unique_ptr<TensorAggregator>> CreateAggregator(
      const Intrinsic& intrinsic);

  // Used by the implementation of Merge.
  std::vector<std::unique_ptr<TensorAggregator>> TakeAggregators() &&;

  // Protects calls into the aggregators.
  mutable absl::Mutex aggregation_mu_;

  // The intrinsics vector need not be guarded by the mutex, as accessing
  // immutable state can happen concurrently.
  std::vector<Intrinsic> const intrinsics_;
  // TensorAggregators are not thread safe and must be protected by a mutex.
  std::vector<std::unique_ptr<TensorAggregator>> aggregators_
      ABSL_GUARDED_BY(aggregation_mu_);
  // This indicates that the aggregation has finished either by producing the
  // report or by destroying this instance.
  // This field is atomic is to allow the Abort() method to work promptly
  // without having to lock on aggregation_mu_ and potentially waiting on all
  // concurrent Accumulate() calls.
  std::atomic<bool> aggregation_finished_ = false;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_PROTOCOL_CHECKPOINT_AGGREGATOR_H_
