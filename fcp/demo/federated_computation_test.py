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
"""Tests for federated_computation."""

from unittest import mock

from absl.testing import absltest
import tensorflow as tf
import tensorflow_federated as tff

from fcp.demo import federated_computation as fc


@tff.tf_computation(tf.int32, tf.int32)
def add_values(x, y):
  return x + y


@tff.federated_computation(
    tff.type_at_server(tf.int32),
    tff.type_at_clients(tff.SequenceType(tf.string)))
def count_clients(state, client_data):
  """Example TFF computation that counts clients."""
  del client_data
  client_value = tff.federated_value(1, tff.CLIENTS)
  aggregated_count = tff.federated_sum(client_value)
  metrics = tff.federated_value(tff.structure.Struct(()), tff.SERVER)
  return tff.federated_map(add_values, (state, aggregated_count)), metrics


@tff.federated_computation(
    tff.type_at_server(tf.int32),
    tff.type_at_clients(tff.SequenceType(tf.string)))
def count_examples(state, client_data):
  """Example TFF computation that counts client examples."""

  @tff.tf_computation
  def client_work(client_data):
    return client_data.reduce(0, lambda x, _: x + 1)

  client_counts = tff.federated_map(client_work, client_data)
  aggregated_count = tff.federated_sum(client_counts)
  metrics = tff.federated_value(tff.structure.Struct(()), tff.SERVER)
  return tff.federated_map(add_values, (state, aggregated_count)), metrics


class FederatedComputationTest(absltest.TestCase):

  def test_invalid_name(self):
    with self.assertRaisesRegex(ValueError, r'name must match ".+"'):
      fc.FederatedComputation(count_clients, name='^invalid^')

  def test_incompatible_computation(self):
    # This function doesn't have the return value structure required for MRF.
    @tff.federated_computation(tff.type_at_server(tf.int32))
    def add_one(value):
      return value + tff.federated_value(1, tff.SERVER)

    with self.assertRaises(TypeError):
      fc.FederatedComputation(add_one, name='comp')

  @tff.test.with_context(
      tff.backends.test.create_sync_test_cpp_execution_context
  )
  def test_map_reduce_form(self):
    comp1 = fc.FederatedComputation(count_clients, name='comp1')
    comp2 = fc.FederatedComputation(count_examples, name='comp2')
    self.assertNotEqual(comp1.map_reduce_form, comp2.map_reduce_form)

    # While we treat the MRF contents as an implementation detail, we can verify
    # the invocation results of the corresponding computation.
    # comp1 should return the number of clients.
    self.assertEqual(
        tff.backends.mapreduce.get_computation_for_map_reduce_form(
            comp1.map_reduce_form
        )(0, [['', '']] * 3),
        (3, ()),
    )
    # comp2 should return the number of examples across all clients.
    self.assertEqual(
        tff.backends.mapreduce.get_computation_for_map_reduce_form(
            comp2.map_reduce_form)(0, [['', '']] * 3), (6, ()))

  @tff.test.with_context(
      tff.backends.native.create_sync_local_cpp_execution_context
  )
  def test_distribute_aggregate_form(self):
    comp1 = fc.FederatedComputation(count_clients, name='comp1')
    comp2 = fc.FederatedComputation(count_examples, name='comp2')
    self.assertNotEqual(
        comp1.distribute_aggregate_form, comp2.distribute_aggregate_form
    )

    # While we treat the DAF contents as an implementation detail, we can verify
    # the invocation results of the corresponding computation.
    # comp1 should return the number of clients.
    self.assertEqual(
        tff.backends.mapreduce.get_computation_for_distribute_aggregate_form(
            comp1.distribute_aggregate_form
        )(0, [['', '']] * 3),
        (3, ()),
    )
    # comp2 should return the number of examples across all clients.
    self.assertEqual(
        tff.backends.mapreduce.get_computation_for_distribute_aggregate_form(
            comp2.distribute_aggregate_form
        )(0, [['', '']] * 3),
        (6, ()),
    )

  def test_wrapped_computation(self):
    comp = fc.FederatedComputation(count_clients, name='comp')
    self.assertEqual(comp.wrapped_computation, count_clients)

  def test_name(self):
    comp = fc.FederatedComputation(count_clients, name='comp')
    self.assertEqual(comp.name, 'comp')

  def test_type_signature(self):
    comp = fc.FederatedComputation(count_clients, name='comp')
    self.assertEqual(comp.type_signature, count_clients.type_signature)

  def test_call(self):
    comp = fc.FederatedComputation(count_clients, name='comp')
    ctx = mock.create_autospec(tff.program.FederatedContext, instance=True)
    ctx.invoke.return_value = 1234
    with tff.framework.get_context_stack().install(ctx):
      self.assertEqual(comp(1, 2, 3, kw1='a', kw2='b'), 1234)
    ctx.invoke.assert_called_once_with(
        comp,
        tff.structure.Struct([(None, 1), (None, 2), (None, 3), ('kw1', 'a'),
                              ('kw2', 'b')]))

  def test_hash(self):
    comp = fc.FederatedComputation(count_clients, name='comp')
    # Equivalent objects should have equal hashes.
    self.assertEqual(
        hash(comp), hash(fc.FederatedComputation(count_clients, name='comp')))
    # Different computations or names should produce different hashes.
    self.assertNotEqual(
        hash(comp), hash(fc.FederatedComputation(count_clients, name='other')))
    self.assertNotEqual(
        hash(comp), hash(fc.FederatedComputation(count_examples, name='comp')))


if __name__ == '__main__':
  absltest.main()
