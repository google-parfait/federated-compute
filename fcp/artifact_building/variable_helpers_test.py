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
"""Tests for variable_helpers.py."""

from absl.testing import absltest

import tensorflow as tf
import tensorflow_federated as tff

from fcp.artifact_building import artifact_constants
from fcp.artifact_building import variable_helpers


@tff.federated_computation(
    tff.type_at_server(tf.int32), tff.type_at_clients(tf.float32)
)
def sample_comp(x, y):
  a = tff.federated_broadcast(x)
  output1 = tff.federated_secure_sum_bitwidth(a, 5)
  output2 = tff.federated_mean([y, y], y)
  return output1, output2


class VariableHelpersTest(absltest.TestCase):

  def test_create_vars_for_tff_type(self):
    with tf.Graph().as_default():
      vl = variable_helpers.create_vars_for_tff_type(
          tff.to_type(
              [('a', tf.int32), ('b', [('c', tf.bool), ('d', tf.float32)])]
          ),
          'x',
      )
    self.assertLen(vl, 3)
    for v in vl:
      self.assertTrue(type(v).__name__.endswith('Variable'))
      self.assertEqual(v.shape.ndims, 0)
    self.assertEqual([v.dtype for v in vl], [tf.int32, tf.bool, tf.float32])
    self.assertEqual([v.name for v in vl], ['x/a:0', 'x/b/c:0', 'x/b/d:0'])

  def test_create_vars_for_tff_type_with_none_and_zero_shape(self):
    with tf.Graph().as_default():
      vl = variable_helpers.create_vars_for_tff_type(
          tff.TensorType(dtype=tf.int32, shape=[5, None, 0])
      )
      self.assertLen(vl, 1)
      test_variable = vl[0]
      self.assertEqual(test_variable.initial_value.shape.as_list(), [5, 0, 0])
      self.assertEqual(test_variable.shape.as_list(), [5, None, None])

  def test_create_vars_for_tff_federated_type(self):
    tff_type = tff.FederatedType(tff.TensorType(tf.int32), tff.SERVER)
    with tf.Graph().as_default():
      vl = variable_helpers.create_vars_for_tff_type(tff_type)

    self.assertLen(vl, 1)
    v = vl[0]
    self.assertTrue(type(v).__name__.endswith('Variable'))
    self.assertEqual(v.shape.ndims, 0)
    self.assertEqual(v.dtype, tf.int32)
    self.assertEqual(v.name, 'v:0')

  def test_create_vars_for_struct_of_tff_federated_types(self):
    tff_type = tff.StructType([
        (
            'num_examples_secagg',
            tff.FederatedType(tff.TensorType(tf.int32), tff.SERVER),
        ),
        (
            'num_examples_simpleagg',
            tff.FederatedType(tff.TensorType(tf.int32), tff.SERVER),
        ),
    ])
    with tf.Graph().as_default():
      vl = variable_helpers.create_vars_for_tff_type(tff_type)

    self.assertLen(vl, 2)
    for v in vl:
      self.assertTrue(type(v).__name__.endswith('Variable'))
      self.assertEqual(v.shape.ndims, 0)
      self.assertEqual([v.dtype for v in vl], [tf.int32, tf.int32])
      self.assertEqual(
          [v.name for v in vl],
          ['v/num_examples_secagg:0', 'v/num_examples_simpleagg:0'],
      )

  def test_create_vars_fails_for_client_placed_type(self):
    tff_type = tff.FederatedType(tff.TensorType(tf.int32), tff.CLIENTS)
    with self.assertRaisesRegex(TypeError, 'Can only create vars'):
      with tf.Graph().as_default():
        _ = variable_helpers.create_vars_for_tff_type(tff_type)

  def test_create_vars_fails_for_struct_with_client_placed_type(self):
    tff_type = tff.StructType([
        (
            'num_examples_secagg',
            tff.FederatedType(tff.TensorType(tf.int32), tff.SERVER),
        ),
        (
            'num_examples_simpleagg',
            tff.FederatedType(tff.TensorType(tf.int32), tff.CLIENTS),
        ),
    ])
    with self.assertRaisesRegex(TypeError, 'Can only create vars'):
      with tf.Graph().as_default():
        _ = variable_helpers.create_vars_for_tff_type(tff_type)

  def test_variable_names_from_type_with_tensor_type_and_no_name(self):
    names = variable_helpers.variable_names_from_type(
        tff.TensorType(dtype=tf.int32)
    )
    self.assertEqual(names, ['v'])

  def test_variable_names_from_type_with_tensor_type(self):
    names = variable_helpers.variable_names_from_type(
        tff.TensorType(dtype=tf.int32), 'test_name'
    )
    self.assertEqual(names, ['test_name'])

  def test_variable_names_from_type_with_federated_type(self):
    names = variable_helpers.variable_names_from_type(
        tff.FederatedType(tff.TensorType(dtype=tf.int32), tff.SERVER),
        'test_name',
    )
    self.assertEqual(names, ['test_name'])

  def test_variable_names_from_type_with_named_tuple_type_and_no_name(self):
    names = variable_helpers.variable_names_from_type(
        tff.to_type(
            [('a', tf.int32), ('b', [('c', tf.bool), ('d', tf.float32)])]
        )
    )
    self.assertEqual(names, ['v/a', 'v/b/c', 'v/b/d'])

  def test_variable_names_from_type_with_named_tuple_type(self):
    names = variable_helpers.variable_names_from_type(
        tff.to_type(
            [('a', tf.int32), ('b', [('c', tf.bool), ('d', tf.float32)])]
        ),
        'test_name',
    )
    self.assertEqual(names, ['test_name/a', 'test_name/b/c', 'test_name/b/d'])

  def test_variable_names_from_type_with_named_tuple_type_no_name_field(self):
    names = variable_helpers.variable_names_from_type(
        tff.to_type([(tf.int32), ('b', [(tf.bool), ('d', tf.float32)])]),
        'test_name',
    )
    self.assertEqual(names, ['test_name/0', 'test_name/b/0', 'test_name/b/d'])

  def test_get_flattened_tensor_specs_with_tensor_type(self):
    specs = variable_helpers.get_flattened_tensor_specs(
        tff.TensorType(dtype=tf.int32, shape=tf.TensorShape([3, 5])),
        'test_name',
    )
    self.assertEqual(
        specs,
        [
            tf.TensorSpec(
                name='test_name',
                shape=tf.TensorShape([3, 5]),
                dtype=tf.int32,
            )
        ],
    )

  def test_get_flattened_tensor_specs_with_federated_type(self):
    specs = variable_helpers.get_flattened_tensor_specs(
        tff.FederatedType(
            tff.TensorType(dtype=tf.int32, shape=tf.TensorShape([3, 5])),
            tff.SERVER,
        ),
        'test_name',
    )
    self.assertEqual(
        specs,
        [
            tf.TensorSpec(
                name='test_name',
                shape=tf.TensorShape([3, 5]),
                dtype=tf.int32,
            )
        ],
    )

  def test_get_flattened_tensor_specs_with_tuple_type(self):
    specs = variable_helpers.get_flattened_tensor_specs(
        tff.StructType([
            (
                'a',
                tff.TensorType(dtype=tf.int32, shape=tf.TensorShape([3, 5])),
            ),
            (
                'b',
                tff.StructType([
                    (tff.TensorType(dtype=tf.bool, shape=tf.TensorShape([4]))),
                    (
                        'd',
                        tff.TensorType(
                            dtype=tf.float32,
                            shape=tf.TensorShape([1, 3, 5]),
                        ),
                    ),
                ]),
            ),
        ]),
        'test_name',
    )
    self.assertEqual(
        specs,
        [
            tf.TensorSpec(
                name='test_name/a',
                shape=tf.TensorShape([3, 5]),
                dtype=tf.int32,
            ),
            tf.TensorSpec(
                name='test_name/b/0',
                shape=tf.TensorShape([4]),
                dtype=tf.bool,
            ),
            tf.TensorSpec(
                name='test_name/b/d',
                shape=tf.TensorShape([1, 3, 5]),
                dtype=tf.float32,
            ),
        ],
    )

  def test_get_grouped_input_tensor_specs_for_aggregations(self):
    daf = tff.backends.mapreduce.get_distribute_aggregate_form_for_computation(
        sample_comp
    )
    grouped_input_tensor_specs = variable_helpers.get_grouped_input_tensor_specs_for_aggregations(
        daf.client_to_server_aggregation.to_building_block(),
        artifact_constants.AGGREGATION_INTRINSIC_ARG_SELECTION_INDEX_TO_NAME_DICT,
    )
    self.assertEqual(
        grouped_input_tensor_specs,
        [
            [  # federated_weighted_mean intrinsic args
                [  # federated_weighted_mean value arg
                    tf.TensorSpec(
                        name='update/0/0',
                        shape=tf.TensorShape([]),
                        dtype=tf.float32,
                    ),
                    tf.TensorSpec(
                        name='update/0/1',
                        shape=tf.TensorShape([]),
                        dtype=tf.float32,
                    ),
                ],
                [  # federated_weighted_mean weight arg
                    tf.TensorSpec(
                        name='update/1',
                        shape=tf.TensorShape([]),
                        dtype=tf.float32,
                    )
                ],
            ],
            [  # federated_secure_sum_bitwidth intrinsic args
                [  # federated_secure_sum_bitwidth value arg
                    tf.TensorSpec(
                        name='update/2',
                        shape=tf.TensorShape([]),
                        dtype=tf.int32,
                    )
                ],
                [  # federated_secure_sum_bitwidth bitwidth arg
                    tf.TensorSpec(
                        name='intermediate_state/0',
                        shape=tf.TensorShape([]),
                        dtype=tf.int32,
                    )
                ],
            ],
        ],
    )

  def test_get_grouped_output_tensor_specs_for_aggregations(self):
    daf = tff.backends.mapreduce.get_distribute_aggregate_form_for_computation(
        sample_comp
    )
    grouped_output_tensor_specs = (
        variable_helpers.get_grouped_output_tensor_specs_for_aggregations(
            daf.client_to_server_aggregation.to_building_block()
        )
    )
    self.assertEqual(
        grouped_output_tensor_specs,
        [
            [  # federated_weighted_mean intrinsic output
                tf.TensorSpec(
                    name='intermediate_update/0/0/0',
                    shape=tf.TensorShape([]),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    name='intermediate_update/0/0/1',
                    shape=tf.TensorShape([]),
                    dtype=tf.float32,
                ),
            ],
            [  # federated_secure_sum_bitwidth intrinsic output
                tf.TensorSpec(
                    name='intermediate_update/0/1',
                    shape=tf.TensorShape([]),
                    dtype=tf.int32,
                )
            ],
        ],
    )


if __name__ == '__main__':
  absltest.main()
