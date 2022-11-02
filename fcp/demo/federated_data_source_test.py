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
"""Tests for federated_data_source."""

from absl.testing import absltest
import tensorflow as tf
import tensorflow_federated as tff

from fcp.demo import federated_data_source as fds
from fcp.protos import plan_pb2

POPULATION_NAME = 'test/name'
EXAMPLE_SELECTOR = plan_pb2.ExampleSelector(collection_uri='app://test')


class FederatedDataSourceTest(absltest.TestCase):

  def test_invalid_population_name(self):
    with self.assertRaisesRegex(ValueError, r'population_name must match ".+"'):
      fds.FederatedDataSource('^invalid^', EXAMPLE_SELECTOR)

  def test_population_name(self):
    ds = fds.FederatedDataSource(POPULATION_NAME, EXAMPLE_SELECTOR)
    self.assertEqual(ds.population_name, POPULATION_NAME)

  def test_example_selector(self):
    ds = fds.FederatedDataSource(POPULATION_NAME, EXAMPLE_SELECTOR)
    self.assertEqual(ds.example_selector, EXAMPLE_SELECTOR)

  def test_federated_type(self):
    ds = fds.FederatedDataSource(POPULATION_NAME, EXAMPLE_SELECTOR)
    self.assertEqual(
        ds.federated_type,
        tff.FederatedType(tff.SequenceType(tf.string), tff.CLIENTS))

  def test_federated_type_nested(self):
    nested_example_selector = {
        'a': EXAMPLE_SELECTOR,
        'b': EXAMPLE_SELECTOR,
        'c': {
            '1': EXAMPLE_SELECTOR,
            '2': EXAMPLE_SELECTOR
        },
    }
    ds = fds.FederatedDataSource(POPULATION_NAME, nested_example_selector)
    self.assertEqual(
        ds.federated_type,
        tff.FederatedType(
            tff.StructType([
                ('a', tff.SequenceType(tf.string)),
                ('b', tff.SequenceType(tf.string)),
                ('c',
                 tff.StructType([
                     ('1', tff.SequenceType(tf.string)),
                     ('2', tff.SequenceType(tf.string)),
                 ])),
            ]), tff.CLIENTS))

  def test_capabilities(self):
    ds = fds.FederatedDataSource(POPULATION_NAME, EXAMPLE_SELECTOR)
    self.assertListEqual(ds.capabilities,
                         [tff.program.Capability.SUPPORTS_REUSE])

  def test_iterator_federated_type(self):
    ds = fds.FederatedDataSource(POPULATION_NAME, EXAMPLE_SELECTOR)
    self.assertEqual(ds.iterator().federated_type, ds.federated_type)

  def test_iterator_select(self):
    ds = fds.FederatedDataSource(POPULATION_NAME, EXAMPLE_SELECTOR)
    self.assertEqual(
        ds.iterator().select(10),
        fds.DataSelectionConfig(POPULATION_NAME, EXAMPLE_SELECTOR, 10))

  def test_iterator_select_with_invalid_num_clients(self):
    ds = fds.FederatedDataSource(POPULATION_NAME, EXAMPLE_SELECTOR)
    with self.assertRaisesRegex(ValueError, 'num_clients must be positive'):
      ds.iterator().select(num_clients=None)
    with self.assertRaisesRegex(ValueError, 'num_clients must be positive'):
      ds.iterator().select(num_clients=-5)
    with self.assertRaisesRegex(ValueError, 'num_clients must be positive'):
      ds.iterator().select(num_clients=0)


if __name__ == '__main__':
  absltest.main()
