"""Tests for checkpoint_utils."""

import collections
import os

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from google.protobuf import any_pb2
from fcp.artifact_building import checkpoint_utils


class CheckpointUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_tff_type_to_dtype_list_as_expected(self):
    tff_type = tff.FederatedType(
        tff.StructType([('foo', tf.int32), ('bar', tf.string)]), tff.SERVER)
    expected_dtype_list = [tf.int32, tf.string]
    self.assertEqual(
        checkpoint_utils.tff_type_to_dtype_list(tff_type), expected_dtype_list)

  def test_tff_type_to_dtype_list_type_error(self):
    list_type = [tf.int32, tf.string]
    with self.assertRaisesRegex(TypeError, 'to be an instance of type'):
      checkpoint_utils.tff_type_to_dtype_list(list_type)

  def test_tff_type_to_tensor_spec_list_as_expected(self):
    tff_type = tff.FederatedType(
        tff.StructType([('foo', tf.int32),
                        ('bar', tff.TensorType(tf.string, shape=[1]))]),
        tff.SERVER)
    expected_tensor_spec_list = [
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([1], tf.string)
    ]
    self.assertEqual(
        checkpoint_utils.tff_type_to_tensor_spec_list(tff_type),
        expected_tensor_spec_list)

  def test_tff_type_to_tensor_spec_list_type_error(self):
    list_type = [tf.int32, tf.string]
    with self.assertRaisesRegex(TypeError, 'to be an instance of type'):
      checkpoint_utils.tff_type_to_tensor_spec_list(list_type)

  def test_pack_tff_value_with_tensors_as_expected(self):
    tff_type = tff.StructType([('foo', tf.int32), ('bar', tf.string)])
    value_list = [
        tf.constant(1, dtype=tf.int32),
        tf.constant('bla', dtype=tf.string)
    ]
    expected_packed_structure = tff.structure.Struct([
        ('foo', tf.constant(1, dtype=tf.int32)),
        ('bar', tf.constant('bla', dtype=tf.string))
    ])
    self.assertEqual(
        checkpoint_utils.pack_tff_value(tff_type, value_list),
        expected_packed_structure)

  def test_pack_tff_value_with_federated_server_tensors_as_expected(self):
    # This test must create a type that has `StructType`s nested under the
    # `FederatedType` to cover testing that tff.structure.pack_sequence_as
    # package correctly descends through the entire type tree.
    tff_type = tff.to_type(
        collections.OrderedDict(
            foo=tff.FederatedType(tf.int32, tff.SERVER),
            # Some arbitrarily deep nesting to ensure full traversals.
            bar=tff.FederatedType([(), ([tf.int32], tf.int32)], tff.SERVER)))
    value_list = [tf.constant(1), tf.constant(2), tf.constant(3)]
    expected_packed_structure = tff.structure.from_container(
        collections.OrderedDict(
            foo=tf.constant(1), bar=[(), ([tf.constant(2)], tf.constant(3))]),
        recursive=True)
    self.assertEqual(
        checkpoint_utils.pack_tff_value(tff_type, value_list),
        expected_packed_structure)

  def test_pack_tff_value_with_unmatched_input_sizes(self):
    tff_type = tff.StructType([('foo', tf.int32), ('bar', tf.string)])
    value_list = [tf.constant(1, dtype=tf.int32)]
    with self.assertRaises(ValueError):
      checkpoint_utils.pack_tff_value(tff_type, value_list)

  def test_pack_tff_value_with_tff_type_error(self):

    @tff.federated_computation
    def fed_comp():
      return tff.federated_value(0, tff.SERVER)

    tff_function_type = fed_comp.type_signature
    value_list = [tf.constant(1, dtype=tf.int32)]
    with self.assertRaisesRegex(TypeError, 'to be an instance of type'):
      checkpoint_utils.pack_tff_value(tff_function_type, value_list)

  def test_variable_names_from_structure_with_tensor_and_no_name(self):
    names = checkpoint_utils.variable_names_from_structure(tf.constant(1.0))
    self.assertEqual(names, ['v'])

  def test_variable_names_from_structure_with_tensor(self):
    names = checkpoint_utils.variable_names_from_structure(
        tf.constant(1.0), 'test_name')
    self.assertEqual(names, ['test_name'])

  def test_variable_names_from_structure_with_named_tuple_type_and_no_name(
      self):
    names = checkpoint_utils.variable_names_from_structure(
        tff.structure.Struct([('a', tf.constant(1.0)),
                              ('b',
                               tff.structure.Struct([('c', tf.constant(True)),
                                                     ('d', tf.constant(0.0))]))
                             ]))
    self.assertEqual(names, ['v/a', 'v/b/c', 'v/b/d'])

  def test_variable_names_from_structure_with_named_struct(self):
    names = checkpoint_utils.variable_names_from_structure(
        tff.structure.Struct([('a', tf.constant(1.0)),
                              ('b',
                               tff.structure.Struct([('c', tf.constant(True)),
                                                     ('d', tf.constant(0.0))]))
                             ]), 'test_name')
    self.assertEqual(names, ['test_name/a', 'test_name/b/c', 'test_name/b/d'])

  def test_variable_names_from_structure_with_named_tuple_type_no_name_field(
      self):
    names = checkpoint_utils.variable_names_from_structure(
        tff.structure.Struct([(None, tf.constant(1.0)),
                              ('b',
                               tff.structure.Struct([(None, tf.constant(False)),
                                                     ('d', tf.constant(0.0))]))
                             ]), 'test_name')
    self.assertEqual(names, ['test_name/0', 'test_name/b/0', 'test_name/b/d'])

  def test_save_tf_tensor_to_checkpoint_as_expected(self):
    temp_dir = self.create_tempdir()
    output_checkpoint_path = os.path.join(temp_dir, 'output_checkpoint.ckpt')

    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])

    checkpoint_utils.save_tff_structure_to_checkpoint(
        tensor, ['v'], output_checkpoint_path=output_checkpoint_path)

    reader = tf.compat.v1.train.NewCheckpointReader(output_checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    self.assertLen(var_to_shape_map, 1)
    self.assertIn('v', var_to_shape_map)
    np.testing.assert_almost_equal([[1.0, 2.0], [3.0, 4.0]],
                                   reader.get_tensor('v'))

  def test_save_tff_struct_to_checkpoint_as_expected(self):
    temp_dir = self.create_tempdir()
    output_checkpoint_path = os.path.join(temp_dir, 'output_checkpoint.ckpt')

    struct = tff.structure.Struct([('foo', tf.constant(1, dtype=tf.int32)),
                                   ('bar', tf.constant('bla',
                                                       dtype=tf.string))])

    checkpoint_utils.save_tff_structure_to_checkpoint(
        struct,
        ordered_var_names=['v/foo', 'v/bar'],
        output_checkpoint_path=output_checkpoint_path)

    reader = tf.compat.v1.train.NewCheckpointReader(output_checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    self.assertLen(var_to_shape_map, 2)
    self.assertIn('v/foo', var_to_shape_map)
    self.assertIn('v/bar', var_to_shape_map)
    self.assertEqual(1, reader.get_tensor('v/foo'))
    self.assertEqual(b'bla', reader.get_tensor('v/bar'))

  def test_save_tff_struct_to_checkpoint_fails_if_wrong_num_var_names(self):
    temp_dir = self.create_tempdir()
    output_checkpoint_path = os.path.join(temp_dir, 'output_checkpoint.ckpt')

    struct = tff.structure.Struct([('foo', tf.constant(1, dtype=tf.int32)),
                                   ('bar', tf.constant('bla',
                                                       dtype=tf.string))])

    with self.assertRaisesRegex(ValueError, 'does not match the number'):
      checkpoint_utils.save_tff_structure_to_checkpoint(
          struct,
          ordered_var_names=['v/foo'],
          output_checkpoint_path=output_checkpoint_path)

  @parameterized.named_parameters(
      ('tf.tensor', tf.constant(1.0)),
      ('ndarray', np.asarray([1.0, 2.0, 3.0])),
      ('npnumber', np.float64(1.0)),
      ('int', 1),
      ('float', 1.0),
      ('str', 'test'),
      ('bytes', b'test'),
  )
  def test_is_allowed(self, structure):
    self.assertTrue(checkpoint_utils.is_structure_of_allowed_types(structure))

  @parameterized.named_parameters(
      ('function', lambda x: x),
      ('any_proto', any_pb2.Any()),
  )
  def test_is_not_allowed(self, structure):
    self.assertFalse(checkpoint_utils.is_structure_of_allowed_types(structure))


if __name__ == '__main__':
  absltest.main()
