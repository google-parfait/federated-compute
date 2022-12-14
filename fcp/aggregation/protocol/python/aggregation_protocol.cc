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

#include "fcp/aggregation/protocol/aggregation_protocol.h"

#include <pybind11/pybind11.h>

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "fcp/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace {

namespace py = ::pybind11;

using ::fcp::aggregation::AcceptanceMessage;
using ::fcp::aggregation::AggregationProtocol;
using ::fcp::aggregation::ServerMessage;

// Allow AggregationProtocol::Callback to be subclassed in Python. See
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
class PyAggregationProtocolCallback : public AggregationProtocol::Callback {
 public:
  void OnAcceptClients(int64_t start_client_id, int64_t num_clients,
                       const AcceptanceMessage& message) override {
    PYBIND11_OVERRIDE_PURE(void, AggregationProtocol::Callback, OnAcceptClients,
                           start_client_id, num_clients, message);
  }

  void OnSendServerMessage(int64_t client_id,
                           const ServerMessage& message) override {
    PYBIND11_OVERRIDE_PURE(void, AggregationProtocol::Callback,
                           OnSendServerMessage, client_id, message);
  }

  void OnCloseClient(int64_t client_id,
                     absl::Status diagnostic_status) override {
    PYBIND11_OVERRIDE_PURE(void, AggregationProtocol::Callback, OnCloseClient,
                           client_id,
                           py::google::DoNotThrowStatus(diagnostic_status));
  }

  void OnComplete(absl::Cord result) override {
    PYBIND11_OVERRIDE_PURE(void, AggregationProtocol::Callback, OnComplete,
                           result);
  }

  void OnAbort(absl::Status diagnostic_status) override {
    PYBIND11_OVERRIDE_PURE(void, AggregationProtocol::Callback, OnAbort,
                           py::google::DoNotThrowStatus(diagnostic_status));
  }
};

}  // namespace

PYBIND11_MODULE(aggregation_protocol, m) {
  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  auto py_aggregation_protocol =
      py::class_<AggregationProtocol>(m, "AggregationProtocol")
          .def("Start", &AggregationProtocol::Start)
          .def("AddClients", &AggregationProtocol::AddClients)
          .def("ReceiveClientMessage",
               &AggregationProtocol::ReceiveClientMessage)
          .def("CloseClient", &AggregationProtocol::CloseClient)
          .def("Complete", &AggregationProtocol::Complete)
          .def("Abort", &AggregationProtocol::Abort)
          .def("GetStatus", &AggregationProtocol::GetStatus);

  pybind11::class_<AggregationProtocol::Callback,
                   PyAggregationProtocolCallback>(py_aggregation_protocol,
                                                  "Callback")
      .def(py::init<>())
      .def("OnAcceptClients", &AggregationProtocol::Callback::OnAcceptClients)
      .def("OnSendServerMessage",
           &AggregationProtocol::Callback::OnSendServerMessage)
      .def("OnCloseClient", &AggregationProtocol::Callback::OnCloseClient)
      .def("OnComplete", &AggregationProtocol::Callback::OnComplete)
      .def("OnAbort", &AggregationProtocol::Callback::OnAbort);
}
