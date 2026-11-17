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

#ifndef FCP_AGGREGATION_CHECKPOINT_CHECKPOINT_AGGREGATOR_H_
#define FCP_AGGREGATION_CHECKPOINT_CHECKPOINT_AGGREGATOR_H_

#include <stdbool.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "fcp/aggregation/checkpoint/checkpoint_aggregator.pb.h"
#include "fcp/aggregation/checkpoint/checkpoint_builder.h"
#include "fcp/aggregation/checkpoint/checkpoint_parser.h"
#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/core/tensor_aggregator.h"

namespace fcp {
namespace aggregation {

class CheckpointAggregator {
 public:
  ~CheckpointAggregator() = default;

  // Not copyable or moveable.
  CheckpointAggregator(const CheckpointAggregator&) = delete;
  CheckpointAggregator& operator=(const CheckpointAggregator&) = delete;

  // Creates an instance of CheckpointAggregator for the given intrinsics.
  // CheckpointAggregator does not take ownership of `intrinsics`, which must
  // outlive it.
  // Returns an error if a TensorAggregator can't be created for any of the
  // intrinsics, e.g. because no factory is registered for its URI or because
  // the intrinsic is invalid.
  static absl::StatusOr<std::unique_ptr<CheckpointAggregator>> Create(
      const std::vector<Intrinsic>* intrinsics ABSL_ATTRIBUTE_LIFETIME_BOUND);

  // Creates an instance of CheckpointAggregator for the given intrinsics.
  // CheckpointAggregator takes ownership of `intrinsics`.
  static absl::StatusOr<std::unique_ptr<CheckpointAggregator>> Create(
      std::vector<Intrinsic> intrinsics);

  // Creates an instance of CheckpointAggregator based on the given intrinsics
  // and serialized state.
  // CheckpointAggregator does not take ownership of `intrinsics`, which must
  // outlive it.
  static absl::StatusOr<std::unique_ptr<CheckpointAggregator>> Deserialize(
      const std::vector<Intrinsic>* intrinsics ABSL_ATTRIBUTE_LIFETIME_BOUND,
      std::string serialized_state);

  // Creates an instance of CheckpointAggregator based on the given intrinsics
  // and serialized state.
  // CheckpointAggregator takes ownership of `intrinsics`.
  static absl::StatusOr<std::unique_ptr<CheckpointAggregator>> Deserialize(
      std::vector<Intrinsic> intrinsics, std::string serialized_state);

  // Accumulates a checkpoint via nested tensor aggregators. The tensors are
  // provided by the CheckpointParser instance.
  absl::Status Accumulate(CheckpointParser& checkpoint_parser);

  // Merges with another compatible instance of CheckpointAggregator consuming
  // it in the process.
  absl::Status MergeWith(CheckpointAggregator&& other);
  // Returns the number of inputs that have been aggregated into this
  // CheckpointAggregator, including those that have been merged in from a
  // separate CheckpointAggregator and/or those that were aggregated before the
  // aggregator was serialized and deserialized.
  //
  // Returns an error if the aggregation has already been finished, or if the
  // number of checkpoints aggregated is undefined because the inner
  // aggregators have aggregated different numbers of tensors.
  absl::StatusOr<int> GetNumCheckpointsAggregated() const;
  // Returns true if the report can be processed.
  bool CanReport() const;
  // Builds the report using the supplied CheckpointBuilder instance.
  absl::Status Report(CheckpointBuilder& checkpoint_builder);
  // Signal that the aggregation must be aborted and the report can't be
  // produced.
  void Abort();
  // Serialize the internal state of the checkpoint aggregator as a string.
  absl::StatusOr<std::string> Serialize() &&;

  // Partitions the internal state of the CheckpointAggregator and serializes
  // them as a vector of strings.
  absl::StatusOr<std::vector<std::string>> Partition(int num_partitions) &&;

 private:
  // Constructs a CheckpointAggregator that refers to unowned `intrinsics`.
  explicit CheckpointAggregator(
      const std::vector<Intrinsic>* intrinsics ABSL_ATTRIBUTE_LIFETIME_BOUND);

  // Constructs a CheckpointAggregator that owns `intrinsics`.
  explicit CheckpointAggregator(std::vector<Intrinsic> intrinsics);

  // Creates an aggregation intrinsic based on the intrinsic configuration and
  // optional serialized state.
  static absl::StatusOr<std::unique_ptr<TensorAggregator>> CreateAggregator(
      const Intrinsic& intrinsic, const std::string* serialized_aggregator);

  // Creates the TensorAggregators for `checkpoint_aggregator`'s intrinsics,
  // optionally from the serialized `aggregator_state`. The TensorAggregators
  // are created from the intrinsics in their final location so that they may
  // safely hold references to them.
  static absl::StatusOr<std::unique_ptr<CheckpointAggregator>> CreateInternal(
      std::unique_ptr<CheckpointAggregator> checkpoint_aggregator,
      const CheckpointAggregatorState* aggregator_state);

  // Used by the implementation of Merge.
  std::vector<std::unique_ptr<TensorAggregator>> TakeAggregators() &&;

  // Protects calls into the aggregators.
  mutable absl::Mutex aggregation_mu_;

  // Intrinsics owned by the CheckpointAggregator. These should not be used
  // directly, and instead should be accessed through `intrinsics_` which will
  // point to `owned_intrinsics_` if it is present.
  std::optional<std::vector<Intrinsic>> const owned_intrinsics_;
  // The intrinsics vector need not be guarded by the mutex, as accessing
  // immutable state can happen concurrently.
  const std::vector<Intrinsic>& intrinsics_;
  // Precalculated number of inputs for each intrinsic.
  const std::vector<size_t> input_counts_;
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

#endif  // FCP_AGGREGATION_CHECKPOINT_CHECKPOINT_AGGREGATOR_H_
