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
"""Tests for server."""

import asyncio
import http
import http.client
import os
import threading
import unittest
import urllib.parse

from absl import flags
from absl import logging
from absl.testing import absltest
import tensorflow as tf

from google.longrunning import operations_pb2
from fcp.demo import server
from fcp.demo import test_utils
from fcp.protos import plan_pb2
from fcp.protos.federatedcompute import task_assignments_pb2
from fcp.tensorflow import external_dataset

POPULATION_NAME = 'test/population'
CAP_TENSOR_NAME = 'cap'
COUNT_TENSOR_NAME = 'count'


def create_plan() -> plan_pb2.Plan:
  """Creates a test plan that counts examples, with a per-client cap."""

  with tf.compat.v1.Graph().as_default() as client_graph:
    dataset_token = tf.compat.v1.placeholder(tf.string, shape=())
    input_filepath = tf.compat.v1.placeholder(tf.string, shape=())
    output_filepath = tf.compat.v1.placeholder(tf.string, shape=())
    ds = external_dataset.ExternalDataset(token=dataset_token, selector=b'')
    cap = tf.raw_ops.Restore(
        file_pattern=input_filepath, tensor_name=CAP_TENSOR_NAME, dt=tf.int32)
    count = ds.take(tf.cast(cap, dtype=tf.int64)).reduce(0, lambda x, _: x + 1)
    target_node = tf.raw_ops.Save(
        filename=output_filepath,
        tensor_names=[COUNT_TENSOR_NAME],
        data=[count])

  with tf.compat.v1.Graph().as_default() as server_graph:
    filename = tf.compat.v1.placeholder(tf.string, shape=())
    contribution_cap = tf.Variable(0, dtype=tf.int32)
    count = tf.Variable(0, dtype=tf.int32)
    load_initial_count = count.assign(
        tf.raw_ops.Restore(
            file_pattern=filename, tensor_name=COUNT_TENSOR_NAME, dt=tf.int32),
        read_value=False)
    load_contribution_cap = contribution_cap.assign(
        tf.raw_ops.Restore(
            file_pattern=filename, tensor_name=CAP_TENSOR_NAME, dt=tf.int32),
        read_value=False)
    with tf.control_dependencies([load_initial_count, load_contribution_cap]):
      restore_server_savepoint = tf.no_op()
    write_client_init = tf.raw_ops.Save(
        filename=filename,
        tensor_names=[CAP_TENSOR_NAME],
        data=[contribution_cap])

    read_intermediate_update = count.assign_add(
        tf.raw_ops.Restore(
            file_pattern=filename, tensor_name=COUNT_TENSOR_NAME, dt=tf.int32))
    save_count = tf.raw_ops.Save(
        filename=filename, tensor_names=[COUNT_TENSOR_NAME], data=[count])

  plan = plan_pb2.Plan(
      phase=[
          plan_pb2.Plan.Phase(
              client_phase=plan_pb2.ClientPhase(
                  tensorflow_spec=plan_pb2.TensorflowSpec(
                      dataset_token_tensor_name=dataset_token.op.name,
                      input_tensor_specs=[
                          tf.TensorSpec.from_tensor(
                              input_filepath).experimental_as_proto(),
                          tf.TensorSpec.from_tensor(
                              output_filepath).experimental_as_proto(),
                      ],
                      target_node_names=[target_node.name]),
                  federated_compute=plan_pb2.FederatedComputeIORouter(
                      input_filepath_tensor_name=input_filepath.op.name,
                      output_filepath_tensor_name=output_filepath.op.name)),
              server_phase=plan_pb2.ServerPhase(
                  write_client_init=plan_pb2.CheckpointOp(
                      saver_def=tf.compat.v1.train.SaverDef(
                          filename_tensor_name=filename.name,
                          save_tensor_name=write_client_init.name)),
                  read_intermediate_update=plan_pb2.CheckpointOp(
                      saver_def=tf.compat.v1.train.SaverDef(
                          filename_tensor_name=filename.name,
                          restore_op_name=read_intermediate_update.name))),
              server_phase_v2=plan_pb2.ServerPhaseV2(aggregations=[
                  plan_pb2.ServerAggregationConfig(
                      intrinsic_uri='federated_sum',
                      intrinsic_args=[
                          plan_pb2.ServerAggregationConfig.IntrinsicArg(
                              input_tensor=tf.TensorSpec(
                                  (), tf.int32,
                                  COUNT_TENSOR_NAME).experimental_as_proto())
                      ],
                      output_tensors=[
                          tf.TensorSpec((), tf.int32, COUNT_TENSOR_NAME)
                          .experimental_as_proto()
                      ])
              ]))
      ],
      server_savepoint=plan_pb2.CheckpointOp(
          saver_def=tf.compat.v1.train.SaverDef(
              filename_tensor_name=filename.name,
              save_tensor_name=save_count.name,
              restore_op_name=restore_server_savepoint.name)),
      version=1)
  plan.client_graph_bytes.Pack(client_graph.as_graph_def())
  plan.server_graph_bytes.Pack(server_graph.as_graph_def())
  return plan


class ServerTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.server = server.InProcessServer(  # pytype: disable=wrong-arg-types
        population_name=POPULATION_NAME,
        host='localhost',
        port=0)
    self._server_thread = threading.Thread(target=self.server.serve_forever)
    self._server_thread.start()
    self.conn = http.client.HTTPConnection(
        self.server.server_name, port=self.server.server_port)

  def tearDown(self):
    self.server.shutdown()
    self._server_thread.join()
    self.server.server_close()
    super().tearDown()

  async def wait_for_task(self):
    """Polls the server until a task is being served."""
    pop = urllib.parse.quote(POPULATION_NAME, safe='')
    url = f'/v1/populations/{pop}/taskassignments/test:start?%24alt=proto'
    request = task_assignments_pb2.StartTaskAssignmentRequest()
    while True:
      self.conn.request('POST', url, request.SerializeToString())
      http_response = self.conn.getresponse()
      if http_response.status == http.HTTPStatus.OK:
        op = operations_pb2.Operation.FromString(http_response.read())
        response = task_assignments_pb2.StartTaskAssignmentResponse()
        op.response.Unpack(response)
        if response.HasField('task_assignment'):
          logging.info('wait_for_task received assignment to %s',
                       response.task_assignment.task_name)
          break
      await asyncio.sleep(0.5)

  async def test_run_computation(self):
    initial_count = 100
    cap = 10
    examples_per_client = [1, 5, 15]
    checkpoint = test_utils.create_checkpoint({
        CAP_TENSOR_NAME: cap,
        COUNT_TENSOR_NAME: initial_count,
    })
    run_computation_task = asyncio.create_task(
        self.server.run_computation('task/name', create_plan(), checkpoint,
                                    len(examples_per_client)))

    # Wait for task assignment to return a task.
    wait_task = asyncio.create_task(self.wait_for_task())
    await asyncio.wait([run_computation_task, wait_task],
                       timeout=10,
                       return_when=asyncio.FIRST_COMPLETED)
    self.assertTrue(wait_task.done())
    # `run_computation` should not be done since no clients have reported.
    self.assertFalse(run_computation_task.done())

    client_runner = os.path.join(
        flags.FLAGS.test_srcdir,
        'com_google_fcp',
        'fcp',
        'client',
        'client_runner_main')
    server_url = f'http://{self.server.server_name}:{self.server.server_port}/'
    clients = []
    for num_examples in examples_per_client:
      subprocess = asyncio.create_subprocess_exec(
          client_runner, f'--server={server_url}',
          f'--population={POPULATION_NAME}',
          f'--num_empty_examples={num_examples}', '--sleep_after_round_secs=0',
          '--use_http_federated_compute_protocol')
      clients.append(asyncio.create_task((await subprocess).wait()))

    # Wait for the computation to complete.
    await asyncio.wait([run_computation_task] + clients, timeout=10)
    self.assertTrue(run_computation_task.done())
    for client in clients:
      self.assertTrue(client.done())
      self.assertEqual(client.result(), 0)

    # Verify the sum in the checkpoint.
    result = test_utils.read_tensor_from_checkpoint(
        run_computation_task.result(), COUNT_TENSOR_NAME, tf.int32)
    self.assertEqual(
        result, initial_count + sum([min(n, cap) for n in examples_per_client]))


if __name__ == '__main__':
  absltest.main()
