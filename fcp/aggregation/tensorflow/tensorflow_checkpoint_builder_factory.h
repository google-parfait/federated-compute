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

#ifndef FCP_AGGREGATION_TENSORFLOW_TENSORFLOW_CHECKPOINT_BUILDER_FACTORY_H_
#define FCP_AGGREGATION_TENSORFLOW_TENSORFLOW_CHECKPOINT_BUILDER_FACTORY_H_

#include <memory>

#include "fcp/aggregation/protocol/simple_aggregation/checkpoint_builder.h"

namespace fcp::aggregation::tensorflow {

// A CheckpointBuilderFactory implementation that writes TensorFlow checkpoints.
class TensorflowCheckpointBuilderFactory
    : public fcp::aggregation::CheckpointBuilderFactory {
 public:
  std::unique_ptr<fcp::aggregation::CheckpointBuilder> Create() const override;
};

}  // namespace fcp::aggregation::tensorflow

#endif  // FCP_AGGREGATION_TENSORFLOW_TENSORFLOW_CHECKPOINT_BUILDER_FACTORY_H_
