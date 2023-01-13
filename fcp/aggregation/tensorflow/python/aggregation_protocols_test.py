# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for aggregation_protocols."""

import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
import tensorflow as tf

from fcp.aggregation.protocol import aggregation_protocol_messages_pb2 as apm_pb2
from fcp.aggregation.protocol import configuration_pb2
from fcp.aggregation.protocol.python import aggregation_protocol
from fcp.aggregation.tensorflow.python import aggregation_protocols
from pybind11_abseil import status


def create_client_input(tensors: dict[str, Any]) -> apm_pb2.ClientMessage:
  with tempfile.NamedTemporaryFile() as tmpfile:
    tf.raw_ops.Save(
        filename=tmpfile.name,
        tensor_names=list(tensors.keys()),
        data=list(tensors.values()))
    with open(tmpfile.name, 'rb') as f:
      return apm_pb2.ClientMessage(
          simple_aggregation=apm_pb2.ClientMessage.SimpleAggregation(
              input=apm_pb2.ClientResource(inline_bytes=f.read())))


class CallbackProxy(aggregation_protocol.AggregationProtocol.Callback):
  """A pass-through Callback that delegates to another Callback.

  This works around the issue that mock.Mock objects aren't recognized as
  Callback subclasses by pybind11.
  """

  def __init__(self,
               callback: aggregation_protocol.AggregationProtocol.Callback):
    super().__init__()
    self._callback = callback

  def OnAcceptClients(self, start_client_id: int, num_clients: int,
                      message: apm_pb2.AcceptanceMessage):
    self._callback.OnAcceptClients(start_client_id, num_clients, message)

  def OnSendServerMessage(self, client_id: int, message: apm_pb2.ServerMessage):
    self._callback.OnSendServerMessage(client_id, message)

  def OnCloseClient(self, client_id: int, diagnostic_status: status.Status):
    self._callback.OnCloseClient(client_id, diagnostic_status)

  def OnComplete(self, result: bytes):
    self._callback.OnComplete(result)

  def OnAbort(self, diagnostic_status: status.Status):
    self._callback.OnAbort(diagnostic_status)


class AggregationProtocolsTest(absltest.TestCase):

  def test_simple_aggregation_protocol(self):
    input_tensor = tf.TensorSpec((), tf.int32, 'in')
    output_tensor = tf.TensorSpec((), tf.int32, 'out')
    config = configuration_pb2.Configuration(aggregation_configs=[
        configuration_pb2.Configuration.ServerAggregationConfig(
            intrinsic_uri='federated_sum',
            intrinsic_args=[
                configuration_pb2.Configuration.ServerAggregationConfig.
                IntrinsicArg(input_tensor=input_tensor.experimental_as_proto()),
            ],
            output_tensors=[output_tensor.experimental_as_proto()],
        ),
    ])
    callback = mock.create_autospec(
        aggregation_protocol.AggregationProtocol.Callback, instance=True)

    agg_protocol = aggregation_protocols.create_simple_aggregation_protocol(
        config, CallbackProxy(callback))
    self.assertIsNotNone(agg_protocol)

    agg_protocol.Start(2)
    callback.OnAcceptClients.assert_called_once_with(mock.ANY, 2, mock.ANY)
    start_client_id = callback.OnAcceptClients.call_args.args[0]

    agg_protocol.ReceiveClientMessage(
        start_client_id, create_client_input({input_tensor.name: 3}))
    agg_protocol.ReceiveClientMessage(
        start_client_id + 1, create_client_input({input_tensor.name: 5}))
    callback.OnCloseClient.assert_has_calls([
        mock.call(start_client_id, status.Status.OkStatus()),
        mock.call(start_client_id + 1, status.Status.OkStatus()),
    ])

    agg_protocol.Complete()
    callback.OnComplete.assert_called_once()
    with tempfile.NamedTemporaryFile('wb') as tmpfile:
      tmpfile.write(callback.OnComplete.call_args.args[0])
      tmpfile.flush()
      self.assertEqual(
          tf.raw_ops.Restore(
              file_pattern=tmpfile.name,
              tensor_name=output_tensor.name,
              dt=output_tensor.dtype), 8)


if __name__ == '__main__':
  absltest.main()
