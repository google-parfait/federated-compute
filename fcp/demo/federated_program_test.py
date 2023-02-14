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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expresus or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end test running a simple Federated Program."""

import asyncio
import os
import tempfile
import unittest

from absl import flags
from absl.testing import absltest
import tensorflow as tf
import tensorflow_federated as tff

from fcp import demo
from fcp.client import client_runner_example_data_pb2
from fcp.protos import plan_pb2

POPULATION_NAME = 'test/population'
COLLECTION_URI = 'app:/example'


@tff.federated_computation()
def initialize() -> tff.Value:
  """Returns the initial state."""
  return tff.federated_value(0, tff.SERVER)


@tff.federated_computation(
    tff.type_at_server(tf.int32),
    tff.type_at_clients(tff.SequenceType(tf.string)))
def sum_counts(state, client_data):
  """Sums the value of all 'count' features across all clients."""

  @tf.function
  def reduce_counts(s: tf.int32, example: tf.string) -> tf.int32:
    features = {'count': tf.io.FixedLenFeature((), tf.int64)}
    count = tf.io.parse_example(example, features=features)['count']
    return s + tf.cast(count, tf.int32)

  @tff.tf_computation
  def client_work(client_data):
    return client_data.reduce(0, reduce_counts)

  client_counts = tff.federated_map(client_work, client_data)
  aggregated_count = tff.federated_sum(client_counts)

  num_clients = tff.federated_sum(tff.federated_value(1, tff.CLIENTS))
  metrics = tff.federated_zip((num_clients,))
  return state + aggregated_count, metrics


async def program_logic(init: tff.Computation, comp: tff.Computation,
                        data_source: tff.program.FederatedDataSource,
                        total_rounds: int, number_of_clients: int,
                        release_manager: tff.program.ReleaseManager) -> None:
  """Initializes and runs a computation, releasing metrics and final state."""
  tff.program.check_in_federated_context()
  data_iterator = data_source.iterator()
  state = init()
  for i in range(total_rounds):
    cohort_config = data_iterator.select(number_of_clients)
    state, metrics = comp(state, cohort_config)
    await release_manager.release(
        metrics, comp.type_signature.result[1], key=f'metrics/{i}')
  await release_manager.release(
      state, comp.type_signature.result[0], key='result')


async def run_client(population_name: str, server_url: str, num_rounds: int,
                     collection_uri: str,
                     examples: list[tf.train.Example]) -> int:
  """Runs a client and returns its return code."""
  client_runner = os.path.join(
      flags.FLAGS.test_srcdir,
      'com_google_fcp',
      'fcp',
      'client',
      'client_runner_main')

  example_data = client_runner_example_data_pb2.ClientRunnerExampleData(
      examples_by_collection_uri={
          collection_uri:
              client_runner_example_data_pb2.ClientRunnerExampleData
              .ExampleList(examples=[e.SerializeToString() for e in examples])
      })

  # Unfortunately, since there's no convenient way to tell when the server has
  # actually started serving the computation, we cannot delay starting the
  # client until the server's ready to assign it a task. This isn't an issue in
  # a production setting, where there's a steady stream of clients connecting,
  # but it is a problem in this unit test, where each client only connects to
  # the server a fixed number of times. To work around this, we give the server
  # a little extra time to become ready; this delay doesn't significantly slow
  # down the test since there are many other time-consuming steps.
  await asyncio.sleep(1)

  with tempfile.NamedTemporaryFile() as tmpfile:
    tmpfile.write(example_data.SerializeToString())
    tmpfile.flush()
    subprocess = await asyncio.create_subprocess_exec(
        client_runner, f'--population={population_name}',
        f'--server={server_url}', f'--example_data_path={tmpfile.name}',
        f'--num_rounds={num_rounds}', '--sleep_after_round_secs=1',
        '--use_http_federated_compute_protocol', '--use_tflite_training')
    return await subprocess.wait()


def create_examples(counts: list[int]) -> list[tf.train.Example]:
  """Creates a list of tf.train.Example with the provided 'count' features."""
  examples = []
  for count in counts:
    example = tf.train.Example()
    example.features.feature['count'].int64_list.value.append(count)
    examples.append(example)
  return examples


class FederatedProgramTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  async def test_multiple_rounds(self):
    data_source = demo.FederatedDataSource(
        POPULATION_NAME,
        plan_pb2.ExampleSelector(collection_uri=COLLECTION_URI))
    comp = demo.FederatedComputation(sum_counts, name='sum_counts')
    release_manager = tff.program.MemoryReleaseManager()
    num_rounds = 2
    client_counts = [
        [0, 3, 5, 1],
        [2, 4],
    ]

    base_context = tff.backends.native.create_sync_local_cpp_execution_context()

    with demo.FederatedContext(
        POPULATION_NAME,
        base_context=base_context) as ctx:
      clients = [
          run_client(POPULATION_NAME, f'http://localhost:{ctx.server_port}',
                     num_rounds, COLLECTION_URI, create_examples(counts))
          for counts in client_counts
      ]
      with tff.framework.get_context_stack().install(ctx):
        program = program_logic(initialize, comp, data_source, num_rounds,
                                len(client_counts), release_manager)
        return_codes = (await asyncio.gather(program, *clients))[1:]
        # All clients should complete successfully.
        self.assertListEqual(return_codes, [0] * len(client_counts))

    self.assertSequenceEqual(release_manager.values()['result'],
                             (num_rounds * sum([sum(l) for l in client_counts]),
                              tff.type_at_server(tf.int32)))
    for i in range(num_rounds):
      self.assertSequenceEqual(
          release_manager.values()[f'metrics/{i}'],
          ((len(client_counts),),
           tff.type_at_server(tff.StructWithPythonType([tf.int32], tuple))))


if __name__ == '__main__':
  absltest.main()
