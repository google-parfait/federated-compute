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

#ifndef FCP_AGGREGATION_PROTOCOL_AGGREGATION_PROTOCOL_FACTORY_H_
#define FCP_AGGREGATION_PROTOCOL_AGGREGATION_PROTOCOL_FACTORY_H_

#include <memory>

#include "absl/status/statusor.h"
#include "fcp/aggregation/protocol/aggregation_protocol.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"

namespace fcp::aggregation {

// A factory interface for creating instances of AggregationProtocol.
class AggregationProtocolFactory {
 public:
  AggregationProtocolFactory() = default;
  virtual ~AggregationProtocolFactory() = default;

  // Creates an instance of the AggregationProtocol based on the passed in
  // Configuration message.
  virtual absl::StatusOr<std::unique_ptr<AggregationProtocol>> Create(
      const Configuration& configuration,
      AggregationProtocol::Callback* const callback) const = 0;
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_AGGREGATION_PROTOCOL_FACTORY_H_
