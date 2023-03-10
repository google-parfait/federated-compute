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

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "fcp/aggregation/protocol/aggregation_protocol.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/aggregation/protocol/simple_aggregation/simple_aggregation_protocol.h"
#include "fcp/aggregation/tensorflow/tensorflow_checkpoint_builder_factory.h"
#include "fcp/aggregation/tensorflow/tensorflow_checkpoint_parser_factory.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace py = ::pybind11;

using ::fcp::aggregation::AggregationProtocol;
using ::fcp::aggregation::Configuration;
using ::fcp::aggregation::ResourceResolver;
using ::fcp::aggregation::tensorflow::TensorflowCheckpointBuilderFactory;
using ::fcp::aggregation::tensorflow::TensorflowCheckpointParserFactory;

PYBIND11_MODULE(aggregation_protocols, m) {
  class DefaultResourceResolver : public ResourceResolver {
    absl::StatusOr<absl::Cord> RetrieveResource(
        int64_t client_id, const std::string& uri) override {
      return absl::UnimplementedError("RetrieveResource() is not supported.");
    }
  };

  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  static const TensorflowCheckpointBuilderFactory* const
      kCheckpointBuilderFactory = new TensorflowCheckpointBuilderFactory();
  static const TensorflowCheckpointParserFactory* const
      kCheckpointParserFactory = new TensorflowCheckpointParserFactory();
  static ResourceResolver* kResourceResolver = new DefaultResourceResolver();

  m.def(
      "create_simple_aggregation_protocol",
      [](const Configuration& configuration,
         AggregationProtocol::Callback* callback)
          -> absl::StatusOr<std::unique_ptr<AggregationProtocol>> {
        return fcp::aggregation::SimpleAggregationProtocol::Create(
            configuration, callback, kCheckpointParserFactory,
            kCheckpointBuilderFactory, kResourceResolver);
      },
      // Ensure the Callback object outlives the AggregationProtocol.
      py::keep_alive<0, 2>());
}
