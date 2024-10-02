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
#ifndef FCP_CLIENT_ENGINE_TENSORFLOW_UTILS_H_
#define FCP_CLIENT_ENGINE_TENSORFLOW_UTILS_H_

#include <vector>

#include "absl/status/statusor.h"
#include "fcp/protos/federated_api.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace fcp::client::engine {

// Parses the output of an eligibility eval plan.
//
// The output of an eligibility eval plan is a single scalar tensor of dtype
// string. The value of the tensor is a serialized
// google.internal.federatedml.v2.TaskEligibilityInfo proto.
absl::StatusOr<google::internal::federatedml::v2::TaskEligibilityInfo>
ParseEligibilityEvalPlanOutput(
    const std::vector<tensorflow::Tensor>& output_tensors);

}  // namespace fcp::client::engine

#endif  // FCP_CLIENT_ENGINE_TENSORFLOW_UTILS_H_
