/*
 * Copyright 2023 Google LLC
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
#ifndef FCP_AGGREGATION_CORE_CONFIG_CONVERTER_H_
#define FCP_AGGREGATION_CORE_CONFIG_CONVERTER_H_

#include "fcp/aggregation/core/intrinsic.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/base/monitoring.h"

namespace fcp {
namespace aggregation {

// Creates an INVALID_ARGUMENT error with the provided error message.
Status ServerAggregationConfigArgumentError(
    const Configuration::ServerAggregationConfig& aggregation_config,
    string_view error_message);

// Parses a ServerAggregationConfig proto into an Intrinsic struct to represent
// the aggregation intrinsic independently from the proto.
StatusOr<Intrinsic> ParseFromConfig(
    const Configuration::ServerAggregationConfig& aggregation_config);

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_CONFIG_CONVERTER_H_
