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
"""Helper methods for TensorFlow variables."""

from typing import Optional, Union

import tensorflow as tf
import tensorflow_federated as tff

from fcp.artifact_building import tensor_utils
from fcp.artifact_building import type_checks

# TFF types allowed for variables created at input/output serialization
# boundaries.
AllowedTffTypes = Union[tff.TensorType, tff.StructType, tff.FederatedType]


# The prefix for the name of the sidechannel for a securely-summed variable.
#
# This transformed name is used as the name of the Op which *reads* from the
# variable, rather than identifies the variable itself. Names with this prefix
# are used as the keys in the `side_channel_tensors` map entries corresponding
# with the variable of the unprefixed name.
SIDECHANNEL_NAME_PREFIX = 'sidechannel_'

# `variable_names_from_type` returns the `name` argument of `tf.Variable()`.
# However when the variable is created, the name of its tensor is actually
# `<name>:0`. This macro is created to match this behavior.
_TF_TENSOR_NAME_SUFFIX = ':0'


def _create_var_for_tff_tensor(tff_type: tff.TensorType, name: str,
                               **kwargs) -> tf.Variable:
  """Creates a TensorFlow variable to hold a value of the `tff.TensorType`."""
  type_checks.check_type(tff_type, tff.TensorType)
  type_checks.check_type(name, str)
  # `tff_type` can have shapes that contain `None` or `0`:
  # * `None` shape cannot be used in `tf.zeros` to create the initial value
  #   of a `tf.Variable`. Hence, we replace it with a `0` in `tf.zeros`.
  # * The dimension that has `0` shape may change its shape at run time. To
  #   support this, we use `None` for that dimension when creating the
  #   `tf.Variable`.
  initial_value_shape = []
  variable_shape = []
  for shape in tff_type.shape.as_list():
    if shape is None or shape == 0:
      initial_value_shape.append(0)
      variable_shape.append(None)
    else:
      initial_value_shape.append(shape)
      variable_shape.append(shape)
  return tf.Variable(
      initial_value=tf.zeros(shape=initial_value_shape, dtype=tff_type.dtype),
      name=name,
      dtype=tff_type.dtype,
      shape=variable_shape,
      **kwargs)


# Build the TensorSpec for the values we will send to the client so that the
# client graph will know how to read the incoming values.
def tensorspec_from_var(var: tf.Variable) -> tf.TensorSpec:
  """Builds `tf.TensorSpec` from `tf.Variables`.

  Args:
    var: An instance of `tf.Variable`.

  Returns:
    An instance of `tf.TensorSpec` corresponding to the input `tf.Variable`.
  """
  return tf.TensorSpec(
      shape=var.shape, dtype=var.dtype, name=tensor_utils.bare_name(var.name))


def create_vars_for_tff_type(tff_type: AllowedTffTypes,
                             name: Optional[str] = None,
                             **kwargs) -> list[tf.Variable]:
  """Creates TensorFlow variables to hold a value of the given `tff_type`.

  The variables are created in the default graph and scope. The variables are
  automatically given `tf.zeros` initializers.

  Args:
    tff_type: Either a `tff.StructType`, SERVER-placed `tff.FederatedType` or a
      `tff.TensorType` object.
    name: The preferred name to use at the top-most level (if not None, must be
      a string). If `tff_type` is a `tff.StructType`, the names of the inner
      fields will be scoped under `name`, e.g. `some_name/field_name`.
    **kwargs: Optional arguments, if any, to pass to the `tf.Variable()` calls.

  Returns:
    A flat Python `list` of TensorFlow variable instances.

  Raises:
    TypeError: If the argument is of the wrong type or has the wrong placement.
  """
  type_checks.check_type(
      tff_type, (tff.TensorType, tff.StructType, tff.FederatedType),
      name='tff_type')
  if name is not None:
    type_checks.check_type(name, str)
  else:
    name = 'v'
  if isinstance(tff_type, tff.TensorType):
    return [_create_var_for_tff_tensor(tff_type, name, **kwargs)]
  elif isinstance(tff_type, tff.FederatedType):
    if tff_type.placement != tff.SERVER:
      raise TypeError('Can only create vars for unplaced types or types placed '
                      'on the SERVER.')
    return create_vars_for_tff_type(tff_type.member, name, **kwargs)
  else:  # tff.StructType
    result = []
    with tf.compat.v1.variable_scope(name):
      fields = tff.structure.to_elements(tff_type)
      for (index, (field_name, field_type)) in enumerate(fields):
        # Default the name of the element to its index so that we don't wind up
        # with multiple child fields listed under `/v/`
        if field_name is None:
          field_name = str(index)
        result.extend(
            create_vars_for_tff_type(field_type, name=field_name, **kwargs))
    return result


def variable_names_from_type(tff_type: AllowedTffTypes,
                             name: str = 'v') -> list[str]:
  """Creates a flattened list of variables names for the given `tff_type`.

  If `tff_type` is a `tff.TensorType`, the name is the `name` parameter if
  specified, otherwise a default name: `v`. If `tff_type` is a
  `tff.StructType` then '/' is used between inner and outer fields together
  with the tuple name or index of the element in the tuple.

  Some examples:
  1. If the tff_type is `<'a'=tf.int32, 'b'=tf.int32>` and `name` is not
    specified, the returned variable name list is ['v/a', 'v/b'].
  2. If the tff_type is `<tf.int32, tf.int32>` and `name` is `update`, the
    returned variable name list is ['update/0', 'update/1'].
  3. If the tff_type is `<'a'=<'b'=tf.int32, 'c'=tf.int32>>` and `name` is
    `update`, the returned variable name list is ['update/a/b', 'update/a/c'].
  4. If the tff_type is `<'a'=<'b'=tf.int32, 'c'=tf.int32, tf.int32>>` and
    `name` is `update`, the returned variable name list is ['update/a/b',
    'update/a/c', 'update/a/2'].

  Args:
    tff_type: Either a `tff.StructType`, a `tff.FederatedType` or a
      `tff.TensorType` object.
    name: The preferred name to use at the top-most level (if not None, must be
      a string). If `tff_type` is a `tff.StructType`, the names of the inner
      fields will be scoped under `name`, e.g. `some_name/field_name`.

  Returns:
    A flat Python `list` of `str` names.

  Raises:
    TypeError: If the argument is of the wrong type.
  """
  type_checks.check_type(
      tff_type, (tff.TensorType, tff.FederatedType, tff.StructType),
      name='tff_type')
  type_checks.check_type(name, str, name='name')
  if isinstance(tff_type, tff.TensorType):
    return [name]
  elif isinstance(tff_type, tff.FederatedType):
    return variable_names_from_type(tff_type.member, name)
  elif isinstance(tff_type, tff.StructType):
    result = []
    fields = tff.structure.iter_elements(tff_type)
    for (index, (field_name, field_type)) in enumerate(fields):
      # Default the name of the element to its index so that we don't wind up
      # with multiple child fields listed under `/v/`
      field_name = field_name or str(index)
      result.extend(
          variable_names_from_type(field_type, name=name + '/' + field_name))
    return result
  else:
    raise TypeError('Cannot create variable names from [{t}] TFF type. '
                    'Short-hand: {s}'.format(t=type(tff_type), s=tff_type))


def get_shared_secagg_tensor_names(intrinsic_name: str,
                                   tff_type: AllowedTffTypes) -> list[str]:
  """Creates the shared name of secagg tensors in client and server graph.

  This is the canonical function for ensuring the secagg tensor names in the
  client and server graph are the same. The server uses secagg tensor
  names as the keys to retrieve values from secagg server which are originally
  from client graph, so if the secagg tensor names in the client and server
  graph are not the same, the server could not find secagg tensors. This
  function is created to ensure this implicit dependency.

  Args:
    intrinsic_name: The name of the secure aggregation intrinsic being used.
    tff_type: Either a `tff.StructType`, `tff.FederatedType` or a
      `tff.TensorType` object.

  Returns:
    A list of variable names created from the input TFF type.
  """
  tensor_names = variable_names_from_type(tff_type,
                                          f'secagg_{intrinsic_name}_update')
  return [
      SIDECHANNEL_NAME_PREFIX + name + _TF_TENSOR_NAME_SUFFIX
      for name in tensor_names
  ]
