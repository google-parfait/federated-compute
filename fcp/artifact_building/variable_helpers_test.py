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

from fcp.artifact_building import variable_helpers


class VariableHelpersTest(absltest.TestCase):

  def test_create_vars_for_tff_type(self):
    with tf.Graph().as_default():
      vl = variable_helpers.create_vars_for_tff_type(
          tff.to_type([('a', tf.int32),
                       ('b', [('c', tf.bool), ('d', tf.float32)])]), 'x')
    self.assertLen(vl, 3)
    for v in vl:
      self.assertTrue(type(v).__name__.endswith('Variable'))
      self.assertEqual(v.shape.ndims, 0)
    self.assertEqual([v.dtype for v in vl], [tf.int32, tf.bool, tf.float32])
    self.assertEqual([v.name for v in vl], ['x/a:0', 'x/b/c:0', 'x/b/d:0'])

  def test_create_vars_for_tff_type_with_none_and_zero_shape(self):
    with tf.Graph().as_default():
      vl = variable_helpers.create_vars_for_tff_type(
          tff.TensorType(dtype=tf.int32, shape=[5, None, 0]))
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
        ('num_examples_secagg',
         tff.FederatedType(tff.TensorType(tf.int32), tff.SERVER)),
        ('num_examples_simpleagg',
         tff.FederatedType(tff.TensorType(tf.int32), tff.SERVER))
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
          ['v/num_examples_secagg:0', 'v/num_examples_simpleagg:0'])

  def test_create_vars_fails_for_client_placed_type(self):
    tff_type = tff.FederatedType(tff.TensorType(tf.int32), tff.CLIENTS)
    with self.assertRaisesRegex(TypeError, 'Can only create vars'):
      with tf.Graph().as_default():
        _ = variable_helpers.create_vars_for_tff_type(tff_type)

  def test_create_vars_fails_for_struct_with_client_placed_type(self):
    tff_type = tff.StructType([
        ('num_examples_secagg',
         tff.FederatedType(tff.TensorType(tf.int32), tff.SERVER)),
        ('num_examples_simpleagg',
         tff.FederatedType(tff.TensorType(tf.int32), tff.CLIENTS))
    ])
    with self.assertRaisesRegex(TypeError, 'Can only create vars'):
      with tf.Graph().as_default():
        _ = variable_helpers.create_vars_for_tff_type(tff_type)

  def test_variable_names_from_type_with_tensor_type_and_no_name(self):
    names = variable_helpers.variable_names_from_type(
        tff.TensorType(dtype=tf.int32))
    self.assertEqual(names, ['v'])

  def test_variable_names_from_type_with_tensor_type(self):
    names = variable_helpers.variable_names_from_type(
        tff.TensorType(dtype=tf.int32), 'test_name')
    self.assertEqual(names, ['test_name'])

  def test_variable_names_from_type_with_federated_type(self):
    names = variable_helpers.variable_names_from_type(
        tff.FederatedType(tff.TensorType(dtype=tf.int32), tff.SERVER),
        'test_name')
    self.assertEqual(names, ['test_name'])

  def test_variable_names_from_type_with_named_tuple_type_and_no_name(self):
    names = variable_helpers.variable_names_from_type(
        tff.to_type([('a', tf.int32), ('b', [('c', tf.bool),
                                             ('d', tf.float32)])]))
    self.assertEqual(names, ['v/a', 'v/b/c', 'v/b/d'])

  def test_variable_names_from_type_with_named_tuple_type(self):
    names = variable_helpers.variable_names_from_type(
        tff.to_type([('a', tf.int32), ('b', [('c', tf.bool),
                                             ('d', tf.float32)])]), 'test_name')
    self.assertEqual(names, ['test_name/a', 'test_name/b/c', 'test_name/b/d'])

  def test_variable_names_from_type_with_named_tuple_type_no_name_field(self):
    names = variable_helpers.variable_names_from_type(
        tff.to_type([(tf.int32), ('b', [(tf.bool), ('d', tf.float32)])]),
        'test_name')
    self.assertEqual(names, ['test_name/0', 'test_name/b/0', 'test_name/b/d'])


if __name__ == '__main__':
  absltest.main()
