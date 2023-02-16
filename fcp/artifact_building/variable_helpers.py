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


def _create_var_for_tff_tensor(
    tff_type: tff.TensorType, name: str, **kwargs
) -> tf.Variable:
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
      **kwargs,
  )


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
      shape=var.shape, dtype=var.dtype, name=tensor_utils.bare_name(var.name)
  )


def create_vars_for_tff_type(
    tff_type: AllowedTffTypes, name: Optional[str] = None, **kwargs
) -> list[tf.Variable]:
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
      tff_type,
      (tff.TensorType, tff.StructType, tff.FederatedType),
      name='tff_type',
  )
  if name is not None:
    type_checks.check_type(name, str)
  else:
    name = 'v'
  if isinstance(tff_type, tff.TensorType):
    return [_create_var_for_tff_tensor(tff_type, name, **kwargs)]
  elif isinstance(tff_type, tff.FederatedType):
    if tff_type.placement != tff.SERVER:
      raise TypeError(
          'Can only create vars for unplaced types or types placed '
          'on the SERVER.'
      )
    return create_vars_for_tff_type(tff_type.member, name, **kwargs)
  else:  # tff.StructType
    result = []
    with tf.compat.v1.variable_scope(name):
      fields = tff.structure.to_elements(tff_type)
      for index, (field_name, field_type) in enumerate(fields):
        # Default the name of the element to its index so that we don't wind up
        # with multiple child fields listed under `/v/`
        if field_name is None:
          field_name = str(index)
        result.extend(
            create_vars_for_tff_type(field_type, name=field_name, **kwargs)
        )
    return result


def variable_names_from_type(
    tff_type: AllowedTffTypes, name: str = 'v'
) -> list[str]:
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
      tff_type,
      (tff.TensorType, tff.FederatedType, tff.StructType),
      name='tff_type',
  )
  type_checks.check_type(name, str, name='name')
  if isinstance(tff_type, tff.TensorType):
    return [name]
  elif isinstance(tff_type, tff.FederatedType):
    return variable_names_from_type(tff_type.member, name)
  elif isinstance(tff_type, tff.StructType):
    result = []
    fields = tff.structure.iter_elements(tff_type)
    for index, (field_name, field_type) in enumerate(fields):
      # Default the name of the element to its index so that we don't wind up
      # with multiple child fields listed under `/v/`
      field_name = field_name or str(index)
      result.extend(
          variable_names_from_type(field_type, name=name + '/' + field_name)
      )
    return result
  else:
    raise TypeError(
        'Cannot create variable names from [{t}] TFF type. '
        'Short-hand: {s}'.format(t=type(tff_type), s=tff_type)
    )


def get_shared_secagg_tensor_names(
    intrinsic_name: str, tff_type: AllowedTffTypes
) -> list[str]:
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
  tensor_names = variable_names_from_type(
      tff_type, f'secagg_{intrinsic_name}_update'
  )
  return [
      SIDECHANNEL_NAME_PREFIX + name + _TF_TENSOR_NAME_SUFFIX
      for name in tensor_names
  ]


def get_flattened_tensor_specs(
    tff_type: AllowedTffTypes, name: str
) -> list[tf.TensorSpec]:
  """Generates TensorSpecs for a flattened version of the given `tff_type`.

  This function uses the same naming logic as `variable_names_from_type`. Please
  see that function's docstring.

  Args:
    tff_type: Either a `tff.StructType`, a `tff.FederatedType` or a
      `tff.TensorType` object.
    name: The preferred name to use at the top-most level (if not None, must be
      a string). If `tff_type` is a `tff.StructType`, the names of the inner
      fields will be scoped under `name`, e.g. `some_name/field_name`.

  Returns:
    A flat Python `list` of `TensorSpec`s.

  Raises:
    TypeError: If the argument is of the wrong type.
  """
  type_checks.check_type(
      tff_type,
      (tff.TensorType, tff.FederatedType, tff.StructType),
      name='tff_type',
  )
  type_checks.check_type(name, str, name='name')
  if isinstance(tff_type, tff.TensorType):
    return [tf.TensorSpec(tff_type.shape, tff_type.dtype, name=name)]
  elif isinstance(tff_type, tff.FederatedType):
    return get_flattened_tensor_specs(tff_type.member, name)
  elif isinstance(tff_type, tff.StructType):
    result = []
    fields = tff.structure.iter_elements(tff_type)
    for index, (field_name, field_type) in enumerate(fields):
      # Default the name of the element to its index so that we don't wind up
      # with multiple child fields listed under `/v/`
      field_name = field_name or str(index)
      result.extend(
          get_flattened_tensor_specs(field_type, name=name + '/' + field_name)
      )
    return result
  else:
    raise TypeError(
        'Cannot create TensorSpecs from [{t}] TFF type. Short-hand: {s}'.format(
            t=type(tff_type), s=tff_type
        )
    )


def get_grouped_input_tensor_specs_for_aggregations(
    aggregation_comp: tff.framework.ComputationBuildingBlock,
    names: dict[int, str],
) -> list[list[list[tf.TensorSpec]]]:
  """Gets the input TensorSpecs for an aggregation computation.

  This function can be used to generate the TensorSpecs that are assigned to
  ServerAggregationConfig.IntrinsicArg messages to represent the aggregation
  intrinsic calls in DistributeAggregateForm.client_to_server_aggregation.

  It derives the tensor name(s) for each intrinsic input argument by following
  naming logic similar to `variable_names_from_type`. DistributeAggregateForm
  does guarantee that each intrinsic input argument will be a
  `building_block.Selection` or a (potentially nested) struct of
  `building_block.Selection`s. The first element of the path is used to
  determine the top-level name, which must match the top-level name that was
  used to construct the tensor that will be getting consumed by this argument.

  Args:
    aggregation_comp: The aggregation computation.
    names: A dictionary describing how to map the first element of the path to a
      top-level name.

  Returns:
    A `list` where the ith entry represents the input tensor specs for the
    ith intrinsic in the aggregation computation. The ith entry is itself a list
    where the jth entry represents the input tensor specs for the jth argument
    of the ith intrinsic in the aggregation computation.

  Raises:
    TypeError: If the argument is of the wrong type.
    ValueError: If the argument contains an unexpected
      `building_block.Selection` index.
  """

  def _get_selection_path(
      selection: tff.framework.ComputationBuildingBlock,
  ) -> list[int]:
    """Gets the list of selection indices for a building_blocks.Selection."""

    path = []
    while selection.is_selection():
      path.append(selection.index)  # pytype: disable=attribute-error
      selection = selection.source  # pytype: disable=attribute-error
    # In ASTs like x[0][1], we'll see the last (outermost) selection first.
    path.reverse()
    return path

  def _get_input_tensor_specs_for_aggregation_arg(
      value: tff.framework.ComputationBuildingBlock, names: dict[int, str]
  ) -> list[tf.TensorSpec]:
    """Gets the input TensorSpecs for a single intrinsic argument."""

    # An intrinsic arg may be a `building_block.Selection` or a (potentially
    # nested) struct of `building_block.Selection`s. Start by creating a
    # flattened list of the `building_block.Selection`s.
    inner_values = []
    if value.is_struct():
      inner_values = tff.structure.flatten(value)
    else:
      inner_values = [value]

    # For each `building_block.Selection`, reconstruct the tensor name that
    # will be used to supply that value. The first index of the selection path
    # indicates whether the tensor will be coming from the intermediate state
    # checkpoint (0) or from the client checkpoint (1), since TFF condenses
    # daf.client_to_server_aggregation(temp_server_state, client_update)
    # into a 1-arg function. Since the tensors within the checkpoints
    # corresponding to temp_server_state and work_at_clients will be named using
    # variable_names_from_type, which uses a simple filepath-like naming pattern
    # to refer to the tensors within a struct, we can reconstruct the relevant
    # tensor name by concatenating together the remaining indices of each
    # selection path.
    tensor_specs = []
    for inner_value in inner_values:
      inner_value.check_selection()
      path = _get_selection_path(inner_value)
      arg_index = path[0]
      if arg_index in names:
        prefix = names[arg_index]
      else:
        raise ValueError('Unexpected arg index for aggregation selection')
      prefix += '/' + '/'.join([str(x) for x in path[1:]])
      tensor_specs.extend(
          get_flattened_tensor_specs(inner_value.type_signature, name=prefix)
      )

    return tensor_specs

  grouped_input_tensor_specs = []

  for _, local_value in aggregation_comp.result.locals:  # pytype: disable=attribute-error
    local_value.check_call()
    local_value.function.check_intrinsic()
    assert local_value.function.intrinsic_def().aggregation_kind

    # Collect the input TensorFlowSpecs for each argument for this intrinsic.
    input_tensor_specs_for_intrinsic = []
    if (
        local_value.function.intrinsic_def().type_signature.parameter.is_struct()
    ):
      for element in local_value.argument.children():
        input_tensor_specs_for_intrinsic.append(
            _get_input_tensor_specs_for_aggregation_arg(element, names)
        )
    else:
      input_tensor_specs_for_intrinsic.append(
          _get_input_tensor_specs_for_aggregation_arg(
              local_value.argument, names
          )
      )

    grouped_input_tensor_specs.append(input_tensor_specs_for_intrinsic)

  return grouped_input_tensor_specs


def get_grouped_output_tensor_specs_for_aggregations(
    aggregation_comp: tff.framework.ComputationBuildingBlock,
) -> list[list[tf.TensorSpec]]:
  """Gets the output TensorSpecs for an aggregation computation.

  This function can be used to generate the TensorSpecs that are assigned
  to the output_tensors value in ServerAggregationConfig messages to represent
  the aggregation intrinsic calls in
  DistributeAggregateForm.client_to_server_aggregation.

  It derives the tensor name(s) for each intrinsic output argument by following
  naming logic similar to `variable_names_from_type`. It must produce tensor
  names that match the tensor names that are expected by the post-aggregation
  computation.

  Args:
    aggregation_comp: The aggregation computation.

  Returns:
    A list where the ith entry represents the output tensor specs for the ith
    intrinsic in the aggregation computation.

  Raises:
    TypeError: If the argument is of the wrong type.
  """
  # TensorflowSpecs for all the intrinsic results. These TensorflowSpecs must
  # have names that mirror the result of calling variable_names_from_type on
  # the output type of DistributeAggregateForm.client_to_server_aggregation
  # (which is the same as the type of the aggregation result input arg in
  # DistributeAggregateForm.server_result).
  output_tensor_specs = get_flattened_tensor_specs(
      tff.StructType([aggregation_comp.type_signature.result]),
      name='intermediate_update',
  )
  output_tensor_spec_index = 0

  grouped_output_tensor_specs = []

  for _, local_value in aggregation_comp.result.locals:  # pytype: disable=attribute-error
    local_value.check_call()
    local_value.function.check_intrinsic()
    local_value.type_signature.check_federated()
    assert local_value.function.intrinsic_def().aggregation_kind

    tensor_specs = []
    # If the output is a struct, select the appropriate number of
    # TensorflowSpecs.
    if local_value.type_signature.member.is_struct():
      num_specs = len(tff.structure.flatten(local_value.type_signature.member))
      tensor_specs = output_tensor_specs[
          output_tensor_spec_index : output_tensor_spec_index + num_specs
      ]
      output_tensor_spec_index += num_specs
    else:
      tensor_specs.append(output_tensor_specs[output_tensor_spec_index])
      output_tensor_spec_index += 1
    grouped_output_tensor_specs.append(tensor_specs)

  return grouped_output_tensor_specs
