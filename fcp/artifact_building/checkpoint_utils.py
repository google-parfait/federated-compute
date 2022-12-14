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
"""Helper methods for working with demo server checkpoints."""

import collections
from collections.abc import Iterable, Mapping
from typing import Any, Union

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from fcp.artifact_building import tensor_utils
from fcp.artifact_building import type_checks
from fcp.artifact_building import variable_helpers


def create_deterministic_saver(var_list: Union[Iterable[tf.Variable],
                                               Mapping[str, tf.Variable]],
                               *args, **kwargs) -> tf.compat.v1.train.Saver:
  """Creates a `tf.compat.v1.Saver` that is deterministic.

  This method sorts the `var_list` to ensure a deterministic ordering which
  in turn ensures a deterministic checkpoint.

  Uses `tf.compat.v1.train.SaverDef.V1` version for writing checkpoints.

  Args:
    var_list: An `Iterable` or `str` keyed `Mapping` of `tf.Variables`. In the
      case of a `dict`, the keys become the names of the checkpoint variables
      (rather than reading the names off the `tf.Variable` values).
    *args: Positional arguments forwarded to the `tf.compat.v1.train.Saver`
      constructor.
    **kwargs: Keyword arguments forwarded to the `tf.compat.v1.train.Saver`
      constructor.

  Returns:
    A `tf.compat.v1.train.Saver` instance.
  """
  if isinstance(var_list, collections.abc.Mapping):
    determinisic_names = collections.OrderedDict(sorted(var_list.items()))
  elif isinstance(var_list, collections.abc.Iterable):
    determinisic_names = sorted(var_list, key=lambda v: v.name)
  else:
    raise ValueError(
        'Do not know how to make a deterministic saver for '
        '`var_list` of type [{t}]. Must be a Mapping or Sequence'.format(
            t=type(var_list)))
  return tf.compat.v1.train.Saver(
      determinisic_names,
      write_version=tf.compat.v1.train.SaverDef.V1,
      *args,
      **kwargs)


def tff_type_to_dtype_list(
    tff_type: variable_helpers.AllowedTffTypes) -> list[tf.DType]:
  """Creates a flat list of `tf.DType`s for tensors in a `tff.Type`.

  Args:
    tff_type: Either a `tff.StructType`, `tff.FederatedType`, or a
      `tff.TensorType` object.

  Returns:
    A flat list of `tf.DType`s.
  """
  type_checks.check_type(tff_type,
                         (tff.TensorType, tff.FederatedType, tff.StructType))
  if isinstance(tff_type, tff.TensorType):
    return [tff_type.dtype]
  elif isinstance(tff_type, tff.FederatedType):
    return tff_type_to_dtype_list(tff_type.member)
  else:  # tff.StructType
    elem_list = []
    for elem_type in tff_type:
      elem_list.extend(tff_type_to_dtype_list(elem_type))
    return elem_list


def tff_type_to_tensor_spec_list(
    tff_type: variable_helpers.AllowedTffTypes) -> list[tf.TensorSpec]:
  """Creates a flat list of tensor specs for tensors in a `tff.Type`.

  Args:
    tff_type: Either a `tff.StructType`, `tff.FederatedType` or a
      `tff.TensorType` object.

  Returns:
    A flat list of `tf.TensorSpec`s.
  """
  type_checks.check_type(tff_type,
                         (tff.TensorType, tff.FederatedType, tff.StructType))
  if isinstance(tff_type, tff.TensorType):
    return [tf.TensorSpec(tff_type.shape, dtype=tff_type.dtype)]
  elif isinstance(tff_type, tff.FederatedType):
    return tff_type_to_tensor_spec_list(tff_type.member)
  else:  # tff.StructType
    elem_list = []
    for elem_type in tff_type:
      elem_list.extend(tff_type_to_tensor_spec_list(elem_type))
    return elem_list


def pack_tff_value(tff_type: variable_helpers.AllowedTffTypes,
                   value_list: Any) -> Any:
  """Packs a list of values into a shape specified by a `tff.Type`.

  Args:
    tff_type: Either a `tff.StructType`, `tff.FederatedType`, or a
      `tff.TensorType` object.
    value_list: A flat list of `tf.Tensor` or `CheckpointTensorReference`.

  Returns:
    A Python container with a structure consistent with a `tff.Type`.

  Raises:
    ValueError: If the number of leaves in `tff_type` does not match the length
    of `value_list`, or `tff_type` is of a disallowed type.
  """
  type_checks.check_type(tff_type,
                         (tff.TensorType, tff.FederatedType, tff.StructType))

  # We must "unwrap" any FederatedTypes because the
  # `tff.structure.pack_sequence_as` call below will fail to recurse into them.
  # Instead, we remove all the FederatedTypes, because we're only trying to
  # build up a Python tree structure that matches the struct/tensor types from a
  # list of values.
  def remove_federated_types(
      type_spec: tff.Type) -> Union[tff.StructType, tff.TensorType]:
    """Removes `FederatedType` from a type tree, returning a new tree."""
    if type_spec.is_tensor():
      return type_spec
    elif type_spec.is_federated():
      return type_spec.member
    elif type_spec.is_struct():
      return tff.StructType(
          (elem_name, remove_federated_types(elem_type))
          for elem_name, elem_type in tff.structure.iter_elements(type_spec))
    else:
      raise ValueError(
          'Must be either tff.TensorType, tff.FederatedType, or tff.StructType.'
          f' Got a {type(type_spec)}')

  try:
    tff_type = remove_federated_types(tff_type)
  except ValueError as e:
    raise ValueError('`tff_type` is not packable, see earlier error. '
                     f'Attempted to pack type: {tff_type}') from e

  ordered_dtypes = tff_type_to_dtype_list(tff_type)
  if len(ordered_dtypes) != len(value_list):
    raise ValueError('The number of leaves in `tff_type` must equals the length'
                     ' of `value_list`. Found `tff_type` with'
                     f' {len(ordered_dtypes)} leaves and `value_list` of length'
                     f' {len(value_list)}.')

  if tff_type.is_tensor():
    return value_list[0]
  elif tff_type.is_struct():
    return tff.structure.pack_sequence_as(tff_type, value_list)
  else:
    raise ValueError('`tff_type` must be either tff.TensorType or '
                     'tff.StructType, reaching here is an internal coding '
                     'error, please file a bug.')


def variable_names_from_structure(tff_structure: Union[tff.structure.Struct,
                                                       tf.Tensor],
                                  name: str = 'v') -> list[str]:
  """Creates a flattened list of variable names for the given structure.

  If the `tff_structure` is a `tf.Tensor`, the name is the `name` parameter if
  specified, otheriwse a default name: `v`. If `tff_structure` is a
  `tff.structure.Struct` then '/' is used between inner and outer fields
  together with the tuple name or index of the element in the tuple.

  Some examples:
  1. If the `tff_structure` is `<'a'=tf.constant(1.0), 'b'=tf.constant(0.0)>`
     and name is not specified, the returned variable name list is
     ['v/a', 'v/b'].
  2. If the `tff_structure` is `<None=tf.constant(1.0), None=tf.constant(0.0)>`
     and `name` is `update`, the returned variable name list is
     ['update/0', 'update/1'].
  3. If the `tff_structure` is
     `<'a'=<'b'=tf.constant(1.0), 'c'=tf.constant(0.0)>>` and `name` is
     `update`, the returned variable name list is ['update/a/b', 'update/a/c'].
  4. If the `tff_structure` is
     `<'a'=<'b'=tf.constant(1.0), 'c'=tf.constant(1.0), tf.constant(0.0)>>` and
     `name` is `update`, the returned variable name list is ['update/a/b',
    'update/a/c', 'update/a/2'].

  Args:
    tff_structure: Either a `tff.structure.Struct` or a `tf.Tensor` object.
    name: The preferred name to use at the top-most level (if not None, must be
      a string). If `tff_structure` is a `tff.structure.Struct`, the names of
      the inner fields will be scoped under `name`, e.g. `some_name/field_name`.

  Returns:
    A flat Python `list` of `str` names.

  Raises:
    TypeError: If either argument is of the wrong type.
  """
  type_checks.check_type(
      tff_structure, (tff.structure.Struct, tf.Tensor), name='structure_type')
  type_checks.check_type(name, str, name='name')
  if isinstance(tff_structure, tf.Tensor):
    return [name]
  elif isinstance(tff_structure, tff.structure.Struct):
    result = []
    fields = tff.structure.iter_elements(tff_structure)
    for (index, (field_name, field_type)) in enumerate(fields):
      # Default the name of the element to its index so that we don't wind up
      # with multiple child fields listed under `/v/`
      field_name = field_name or str(index)
      result.extend(
          variable_names_from_structure(
              field_type, name=name + '/' + field_name))
    return result
  else:
    raise TypeError('Cannot create variable names from [{t}] type. '
                    'Short-hand: {s}'.format(
                        t=type(tff_structure), s=tff_structure))


def is_structure_of_allowed_types(
    structure: Union[tff.structure.Struct, tf.Tensor, np.ndarray, np.number,
                     int, float, str, bytes]
) -> bool:
  """Checks if each node in `structure` is an allowed type for serialization."""
  flattened_structure = tff.structure.flatten(structure)
  for item in flattened_structure:
    if not (tf.is_tensor(item) or
            isinstance(item, (np.ndarray, np.number, int, float, str, bytes))):
      return False
  return True


def save_tff_structure_to_checkpoint(tff_structure: Union[tff.structure.Struct,
                                                          tf.Tensor],
                                     ordered_var_names: list[str],
                                     output_checkpoint_path: str) -> None:
  """Saves a TFF structure to a checkpoint file.

  The input `tff_structure` is a either `tff.structure.Struct` or a single
  `tf.Tensor`. This function saves `tff_structure` to a checkpoint file using
  variable names supplied via the `ordered_var_names` argument.

  Args:
    tff_structure: A `tff.structure.Struct` of values or a single value. Each
      leaf in the structure must be a value serializable to a TensorFlow
      checkpoint.
    ordered_var_names: The list of variable names for the values that appear in
      `tff_structure` after calling `tff.structure.flatten()`.
    output_checkpoint_path: A string specifying the path to the output
      checkpoint file.

  Raises:
    TypeError: If not all leaves in `tff_structure` are of allowed types.
    ValueError: If the number of `tf.Tensor`s in `tff_structure` does not match
      the size of `ordered_var_names`.
  """
  if not is_structure_of_allowed_types(tff_structure):
    raise TypeError('Not all leaves in `tff_structure` are `tf.Tensor`s, '
                    '`np.ndarray`s, `np.number`s, or Python scalars. Got: '
                    f'{tf.nest.map_structure(type, tff_structure)!r})')

  tensors = tff.structure.flatten(tff_structure)
  if len(tensors) != len(ordered_var_names):
    raise ValueError('The length of `ordered_var_names` does not match the '
                     'number of tensors in `tff_structure`:'
                     f'{len(ordered_var_names)} != {len(tensors)}')

  tensor_utils.save(
      output_checkpoint_path, tensor_names=ordered_var_names, tensors=tensors)
