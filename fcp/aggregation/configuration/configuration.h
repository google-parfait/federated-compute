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

#ifndef FCP_AGGREGATION_CONFIGURATION_CONFIGURATION_H_
#define FCP_AGGREGATION_CONFIGURATION_CONFIGURATION_H_

#include <vector>

#include "absl/status/statusor.h"
#include "fcp/aggregation/configuration/configuration.pb.h"
#include "fcp/aggregation/core/intrinsic.h"

namespace fcp {
namespace aggregation {

// Validates a Configuration proto and parses it into a vector of Intrinsic
// structs that represent the aggregation intrinsics independently from the
// proto.
//
// The conversion is a direct structural mapping: input tensor specs, output
// tensor specs, parameters and inner intrinsics are copied as they are. No
// aggregator specific transformations are applied.
//
// Validation checks that the intrinsic URI of every top-level intrinsic has an
// aggregator factory registered. Inner intrinsics are validated by the factory
// of their parent intrinsic when the aggregator is created. Only factories
// linked into the binary are registered, so callers must link the aggregator
// libraries they intend to use.
//
// Returns INVALID_ARGUMENT if the configuration is invalid.
absl::StatusOr<std::vector<Intrinsic>> ParseConfiguration(
    const Configuration& configuration);

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CONFIGURATION_CONFIGURATION_H_
