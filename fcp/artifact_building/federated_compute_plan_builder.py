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
"""A library responsible for building Federated Compute plans.

This library builds TFF-backed plans, using the `MapReduceForm` object
output by the TFF compiler pipeline.
"""

import collections
from collections.abc import Callable, Iterable, Mapping, Sequence
import enum
from typing import Optional, TypeVar, Union

import attr
import tensorflow as tf
import tensorflow_federated as tff

from fcp.artifact_building import artifact_constants
from fcp.artifact_building import checkpoint_type
from fcp.artifact_building import checkpoint_utils
from fcp.artifact_building import data_spec
from fcp.artifact_building import graph_helpers
from fcp.artifact_building import proto_helpers
from fcp.artifact_building import tensor_utils
from fcp.artifact_building import type_checks
from fcp.artifact_building import variable_helpers
from fcp.protos import plan_pb2
from fcp.tensorflow import append_slices
from fcp.tensorflow import delete_file

SECURE_SUM_BITWIDTH_URI = 'federated_secure_sum_bitwidth'
SECURE_SUM_URI = 'federated_secure_sum'
SECURE_MODULAR_SUM_URI = 'federated_secure_modular_sum'


class SecureAggregationTensorShapeError(Exception):
  """Error raised when secagg tensors do not have fully defined shape."""


@enum.unique
class ClientPlanType(enum.Enum):
  """Option adjusting client plan type during plan building.

  Values:
    TENSORFLOW: The default value. Uses a TF client graph for client
      computation.
    EXAMPLE_QUERY: Uses an example query containing client computation logic in
      the provided example selector(s).
  """

  TENSORFLOW = 'tensorflow'
  EXAMPLE_QUERY = 'example_query'


# A type representing a potentially nested struct of integers.
IntStruct = Union[
    int,
    Mapping[str, Union['IntStruct', int]],
    Sequence[Union['IntStruct', int]],
]


def _compute_secagg_parameters(
    mrf: tff.backends.mapreduce.MapReduceForm,
) -> tuple[IntStruct, IntStruct, IntStruct]:
  """Executes the TensorFlow logic that computes the SecAgg parameters.

  This function makes use of `mrf.secure_sum_bitwidth`,
  `mrf.secure_sum_max_input`, and `mrf.secure_modular_sum_modulus` to derive
  the parameters needed for the SecAgg protocol variants.

  Args:
    mrf: An instance of `tff.backends.mapreduce.MapReduceForm`.

  Returns:
    A 3-tuple of `bitwidth`, `max_input` and `moduli` structures of parameters
    for the associated SecAgg variants.
  """
  type_checks.check_type(mrf, tff.backends.mapreduce.MapReduceForm, name='mrf')
  secagg_parameters = []
  with tf.Graph().as_default() as g:
    for name, computation in [
        ('bitwidth', mrf.secure_sum_bitwidth),
        ('max_input', mrf.secure_sum_max_input),
        ('modulus', mrf.secure_modular_sum_modulus),
    ]:
      secagg_parameters.append(
          graph_helpers.import_tensorflow(name, computation)
      )
  with tf.compat.v1.Session(graph=g) as sess:
    flat_output = sess.run(fetches=tf.nest.flatten(secagg_parameters))
    return tf.nest.pack_sequence_as(secagg_parameters, flat_output)


# A side-channel through which one tensor is securely aggregated.
@attr.s
class SecAggSideChannel:
  # The name of the tensor being aggregated in the client and server graphs.
  tensor_name: str = attr.ib()
  # A proto describing how the side-channel is to be aggregated.
  side_channel_proto: plan_pb2.SideChannel = attr.ib()
  # A placeholder tensor into which the sidechannel aggregation is filled.
  placeholder: tf.Tensor = attr.ib()
  # The variable to feed into the server graph.
  update_var: tf.Variable = attr.ib()


SecAggParam = TypeVar('SecAggParam')


def _create_secagg_sidechannels(
    intrinsic_name: str,
    update_type: variable_helpers.AllowedTffTypes,
    get_modulus_scheme: Callable[
        [SecAggParam], plan_pb2.SideChannel.SecureAggregand
    ],
    params: list[SecAggParam],
) -> list[SecAggSideChannel]:
  """Returns `SecAggSideChannel`s for tensors aggregated with `intrinsic_name`.

  This method also creates variables for the securely-aggregated tensors within
  the current default graph using `create_vars_for_tff_type`.

  Args:
    intrinsic_name: The name of the intrinsic (e.g.
      `federated_secure_sum_bitwidth`) with which the tensors in `update_type`
      are being aggregated.
    update_type: The TFF type representing a structure of all tensors being
      aggregated with `intrinsic_name`.
    get_modulus_scheme: A function which will get the modulus scheme being used.
      This typically requires some additional per-tensor parameters which must
      be supplied using `params`.
    params: A list of arguments to pass to `set_modulus_scheme`. There must be
      exactly one element in this list per tensor in `update_type`.

  Returns:
    A list of `SecAggSideChannel`s describing how to aggregate each tensor.
  """
  # For secure aggregation, we don't use a saver (but still store metadata in a
  # CheckpointOp). Instead we create sidechannel tensors that get fed into the
  # server graph.
  update_vars = variable_helpers.create_vars_for_tff_type(
      update_type, f'{intrinsic_name}_update'
  )

  # For tensors aggregated by secagg, we make sure the tensor names are aligned
  # in both client and sever graph by getting the names from the same method.
  tensor_names = variable_helpers.get_shared_secagg_tensor_names(
      intrinsic_name, update_type
  )
  assert len(update_vars) == len(params) == len(tensor_names), (
      'The length of update_vars, params and tensor_names for'
      f' {{intrinsic_name}} should be all equal, but found: {len(update_vars)},'
      f' {len(params)}, and {len(tensor_names)}.'
  )

  results = []
  for param, update_var, tensor_name in zip(params, update_vars, tensor_names):
    secure_aggregand = get_modulus_scheme(param)
    secure_aggregand.dimension.extend(
        plan_pb2.SideChannel.SecureAggregand.Dimension(size=d.value)
        for d in update_var.shape.dims
    )
    secure_aggregand.dtype = update_var.dtype.base_dtype.as_datatype_enum
    placeholder = tf.compat.v1.placeholder(
        update_var.dtype, update_var.get_shape()
    )
    side_channel_proto = plan_pb2.SideChannel(
        secure_aggregand=secure_aggregand, restore_name=placeholder.name
    )
    results.append(
        SecAggSideChannel(
            tensor_name=tensor_name,
            side_channel_proto=side_channel_proto,
            placeholder=placeholder,
            update_var=update_var,
        )
    )
  return results


def _read_secagg_update_from_sidechannel_into_vars(
    *,  # Require parameters to be named.
    secagg_intermediate_update_vars: list[tf.Variable],
    secure_sum_bitwidth_update_type: (variable_helpers.AllowedTffTypes),
    bitwidths: list[int],
    secure_sum_update_type: (variable_helpers.AllowedTffTypes),
    max_inputs: list[int],
    secure_modular_sum_update_type: (variable_helpers.AllowedTffTypes),
    moduli: list[int],
) -> plan_pb2.CheckpointOp:
  """Creates the `read_secagg_update` op.

  `read_secagg_update` is a `plan_pb2.CheckpointOp` and used to restore the
  secagg tensors in server graph.

  Args:
    secagg_intermediate_update_vars: A list of variables to assign the
      secagg_update_data in the `after_restore_op`.
    secure_sum_bitwidth_update_type: The type of the tensors aggregated using
      `bitwidth`-based secure sum.
    bitwidths: The `bitwidth`s for the tensors that will be aggregated using
      `bitwidth`-based secure summation.
    secure_sum_update_type: The type of the tensors aggregated using
      `max_input`-based secure sum.
    max_inputs: The max_input`s for the tensors that will be aggregated using
      `max_input`-based secure summation.
    secure_modular_sum_update_type: The type of the tensors aggregated using
      modular secure summation.
    moduli: The `modulus`s for the tensors that will be aggregated using modular
      secure summation.

  Returns:
    A `plan_pb2.CheckpointOp` which performs the `read_secagg_update`.
  """
  side_channels: list[SecAggSideChannel] = []

  def _aggregand_for_bitwidth(bitwidth):
    return plan_pb2.SideChannel.SecureAggregand(
        quantized_input_bitwidth=bitwidth
    )

  side_channels += _create_secagg_sidechannels(
      SECURE_SUM_BITWIDTH_URI,
      secure_sum_bitwidth_update_type,
      _aggregand_for_bitwidth,
      bitwidths,
  )

  def _aggregand_for_max_input(max_input):
    # Note the +1-- `max_input` is inclusive, so `base_modulus == max_input`
    # would overflow maximum-valued inputs to zero.
    base_modulus = max_input + 1
    modulus_times_shard_size = (
        plan_pb2.SideChannel.SecureAggregand.ModulusTimesShardSize(
            base_modulus=base_modulus
        )
    )
    return plan_pb2.SideChannel.SecureAggregand(
        modulus_times_shard_size=modulus_times_shard_size
    )

  side_channels += _create_secagg_sidechannels(
      SECURE_SUM_URI,
      secure_sum_update_type,
      _aggregand_for_max_input,
      max_inputs,
  )

  def _aggregand_for_modulus(modulus):
    fixed_modulus = plan_pb2.SideChannel.SecureAggregand.FixedModulus(
        modulus=modulus
    )
    return plan_pb2.SideChannel.SecureAggregand(fixed_modulus=fixed_modulus)

  side_channels += _create_secagg_sidechannels(
      SECURE_MODULAR_SUM_URI,
      secure_modular_sum_update_type,
      _aggregand_for_modulus,
      moduli,
  )

  # Operations assigning from sidechannel placeholders to update variables.
  assign_placeholders_to_updates = []
  # Operations assigning from update variables to the result variables.
  assign_updates_to_intermediate = []
  read_secagg_update = plan_pb2.CheckpointOp()
  for intermediate_update_var, side_channel in zip(
      secagg_intermediate_update_vars, side_channels
  ):
    assign_placeholders_to_updates.append(
        side_channel.update_var.assign(side_channel.placeholder)
    )
    assign_updates_to_intermediate.append(
        intermediate_update_var.assign(side_channel.update_var)
    )
    read_secagg_update.side_channel_tensors[side_channel.tensor_name].CopyFrom(
        side_channel.side_channel_proto
    )

  read_secagg_update.before_restore_op = tf.group(
      *(assign_placeholders_to_updates)
  ).name
  read_secagg_update.after_restore_op = tf.group(
      *(assign_updates_to_intermediate)
  ).name

  return read_secagg_update


def _merge_secagg_vars(
    secure_sum_bitwidth_update_type: tff.Type,
    secure_sum_update_type: tff.Type,
    secure_modular_sum_update_type: tff.Type,
    flattened_moduli: list[int],
    variables: list[tf.Variable],
    tensors: list[tf.Variable],
) -> list[tf.Operation]:
  """Generates a set of ops to `merge` secagg `tensors` into `variables`."""
  if len(variables) != len(tensors):
    raise ValueError(
        'Expected an equal number of variables and tensors, but found '
        f'{len(variables)} variables and {len(tensors)} tensors.'
    )
  num_simple_add_vars = len(
      tff.structure.flatten(
          tff.to_type([
              secure_sum_bitwidth_update_type,
              secure_sum_update_type,
          ])
      )
  )
  num_modular_add_vars = len(
      tff.structure.flatten(secure_modular_sum_update_type)
  )
  # There must be one variable and tensor for each tensor in the secure update
  # types.
  num_vars_from_types = num_simple_add_vars + num_modular_add_vars
  if num_vars_from_types != len(variables):
    raise ValueError(
        'Expected one variable for each leaf element of the secagg update, but '
        f'found {len(variables)} variables and {num_vars_from_types} leaf '
        'elements in the following types:\n'
        f'secure_sum_bitwidth_update_type: {secure_sum_bitwidth_update_type}\n'
        f'secure_sum_update_type: {secure_sum_update_type}\n'
        f'secure_modular_sum_update_type: {secure_modular_sum_update_type}\n'
    )
  if num_modular_add_vars != len(flattened_moduli):
    raise ValueError(
        'Expected one modulus for each leaf element of the secure modular sum '
        f'update type. Found {len(flattened_moduli)} moduli and '
        f'{num_modular_add_vars} leaf elements in the secure modular sum '
        f'update type:\n{secure_modular_sum_update_type}'
    )
  # Add `tensors` to `vars`, using simple addition for the first
  # `num_secagg_simple_add_vars` variables and modular addition for the rest
  # (those coming from `secure_modular_sum`).
  ops = []
  simple_add_vars = variables[:num_simple_add_vars]
  simple_add_tensors = tensors[:num_simple_add_vars]
  for variable, tensor in zip(simple_add_vars, simple_add_tensors):
    ops.append(variable.assign_add(tensor))
  modular_add_vars = variables[num_simple_add_vars:]
  modular_add_tensors = tensors[num_simple_add_vars:]
  for modulus, (variable, tensor) in zip(
      flattened_moduli, zip(modular_add_vars, modular_add_tensors)
  ):
    new_sum = tf.math.add(variable.read_value(), tensor)
    modular_sum = tf.math.floormod(new_sum, modulus)
    ops.append(variable.assign(tf.reshape(modular_sum, tf.shape(variable))))
  return ops


def _build_server_graphs_from_distribute_aggregate_form(
    daf: tff.backends.mapreduce.DistributeAggregateForm,
    is_broadcast_empty: bool,
    grappler_config: tf.compat.v1.ConfigProto,
) -> tuple[
    tf.compat.v1.GraphDef, tf.compat.v1.GraphDef, plan_pb2.ServerPhaseV2
]:
  """Generates the server plan components based on DistributeAggregateForm.

  Derives the pre-broadcast, aggregation, and post-aggregation logical
  components in the ServerPhaseV2 message that will be executed on the server.
  The pre-broadcast and post-aggregation components are to be executed with a
  single TF sess.run call using the corresponding GraphDef. The aggregation
  component is to be executed natively (i.e. not using TensorFlow) according to
  the aggregation messages contained in the ServerPhaseV2 message.

  Args:
    daf: An instance of `tff.backends.mapreduce.DistributeAggregateForm`.
    is_broadcast_empty: A boolean indicating whether the broadcasted value from
      the server is expected to be empty based on the DistributeAggregateForm,
      in which case the server should broadcast a placeholder tf.int32 tensor as
      empty checkpoints are not supported.
    grappler_config: The config specifying Grappler optimizations for TFF-
      generated graphs.

  Returns:
    A tuple containing the server_prepare GraphDef, the server_result GraphDef,
    and the ServerPhaseV2 message.
  """
  # Generate the TensorFlow graph needed to execute the server_prepare step,
  # including reading input checkpoints and writing output checkpoints.
  server_prepare_input_tensors = []
  server_prepare_target_nodes = []
  with tf.Graph().as_default() as server_prepare_graph:
    # Create the placeholders for the input and output filenames needed by
    # the server_prepare step.
    server_prepare_server_state_input_filepath_placeholder = (
        tf.compat.v1.placeholder(
            name='server_state_input_filepath', shape=(), dtype=tf.string
        )
    )
    server_prepare_output_filepath_placeholder = tf.compat.v1.placeholder(
        name='server_prepare_output_filepath', shape=(), dtype=tf.string
    )
    server_prepare_intermediate_state_output_filepath_placeholder = (
        tf.compat.v1.placeholder(
            name='server_intermediate_state_output_filepath',
            shape=(),
            dtype=tf.string,
        )
    )
    server_prepare_input_tensors.extend([
        server_prepare_server_state_input_filepath_placeholder,
        server_prepare_output_filepath_placeholder,
        server_prepare_intermediate_state_output_filepath_placeholder,
    ])

    # Restore the server state.
    server_state_type = daf.server_prepare.type_signature.parameter
    server_state_vars = variable_helpers.create_vars_for_tff_type(
        server_state_type, name='server'
    )
    server_state_tensor_specs = tf.nest.map_structure(
        variable_helpers.tensorspec_from_var, server_state_vars
    )
    server_state = checkpoint_utils.restore_tensors_from_savepoint(
        server_state_tensor_specs,
        server_prepare_server_state_input_filepath_placeholder,
    )

    # TODO(team): Add support for federated select slice generation.

    # Perform the server_prepare step.
    prepared_values, intermediate_state_values = (
        graph_helpers.import_tensorflow(
            'server_prepare',
            tff.framework.ConcreteComputation.from_building_block(
                tff.backends.mapreduce.consolidate_and_extract_local_processing(
                    daf.server_prepare.to_building_block(), grappler_config
                )
            ),
            server_state,
            split_outputs=True,
        )
    )

    # Create checkpoints storing the broadcast values and intermediate server
    # state. If there is no broadcast value, create a checkpoint containing a
    # placeholder tf.int32 constant since empty broadcasts are not supported.
    # If there is no intermediate server state, don't create an intermediate
    # server state checkpoint.
    save_tensor_names = variable_helpers.variable_names_from_type(
        daf.server_prepare.type_signature.result[0], name='client'
    )
    save_tensors = prepared_values
    if is_broadcast_empty:
      save_tensor_names = variable_helpers.variable_names_from_type(
          tff.StructType([tf.int32]), name='client'
      )
      save_tensors = [tf.constant(0, tf.int32)]
    prepared_values_save_op = tensor_utils.save(
        filename=server_prepare_output_filepath_placeholder,
        tensor_names=save_tensor_names,
        tensors=save_tensors,
        name='save_prepared_values_tensors',
    )
    server_prepare_target_nodes.append(prepared_values_save_op.name)

    intermediate_state_empty = (
        isinstance(daf.server_prepare.type_signature.result[1], tff.StructType)
        and not daf.server_prepare.type_signature.result[1]
    )
    if not intermediate_state_empty:
      intermediate_state_values_save_op = tensor_utils.save(
          filename=server_prepare_intermediate_state_output_filepath_placeholder,
          tensor_names=variable_helpers.variable_names_from_type(
              daf.server_prepare.type_signature.result[1], 'intermediate_state'
          ),
          tensors=intermediate_state_values,
          name='save_intermediate_state_values_tensors',
      )
      server_prepare_target_nodes.append(intermediate_state_values_save_op.name)

  # Build aggregations.
  # The client_to_server_aggregation computation is guaranteed to conform to
  # a specific structure. It is a lambda computation whose result block contains
  # locals that are exclusively aggregation-type intrinsics.
  aggregations_bb = daf.client_to_server_aggregation.to_building_block()
  aggregations_bb.check_lambda()
  aggregations_bb.result.check_block()  # pytype: disable=attribute-error

  # Get lists of the TensorSpecProtos for the inputs and outputs of all
  # intrinsic calls. These lists are formatted such that the ith entry
  # represents the TensorSpecProtos for the ith intrinsic in the aggregation
  # computation. Since intrinsics may have one or more args, the ith entry in
  # the input TensorSpecProto list is itself a list, where the jth entry
  # represents the TensorSpecProtos corresponding to the jth argument of the
  # ith intrinsic.
  grouped_input_tensor_specs = variable_helpers.get_grouped_input_tensor_specs_for_aggregations(
      aggregations_bb,
      artifact_constants.AGGREGATION_INTRINSIC_ARG_SELECTION_INDEX_TO_NAME_DICT,
  )
  grouped_output_tensor_specs = (
      variable_helpers.get_grouped_output_tensor_specs_for_aggregations(
          aggregations_bb
      )
  )
  assert len(grouped_input_tensor_specs) == len(grouped_output_tensor_specs)

  intrinsic_uris = [
      local_value.function.intrinsic_def().uri
      for _, local_value in aggregations_bb.result.locals  # pytype: disable=attribute-error
  ]
  assert len(intrinsic_uris) == len(grouped_output_tensor_specs)

  # Each intrinsic input arg can be a struct or even a nested struct, which
  # requires the intrinsic to be applied independently to each element (e.g. a
  # tff.federated_sum call applied to a struct will result in a federated_sum
  # aggregation message for each element of the struct). Note that elements of
  # structs can themselves be multi-dimensional tensors. When an intrinsic call
  # has multiple args with mismatching structure (e.g. a federated_weighted_mean
  # intrinsic applied to a 2D struct value arg and scalar weight arg), some args
  # will need to be "scaled up" via repetition to match the args with the
  # "largest" structure.
  aggregations = []
  for intrinsic_index, (input_tensor_specs, output_tensor_specs) in enumerate(
      zip(grouped_input_tensor_specs, grouped_output_tensor_specs)
  ):
    # Generate the aggregation messages for this intrinsic call.
    max_input_struct_length = max([len(x) for x in input_tensor_specs])
    max_struct_length = max(max_input_struct_length, len(output_tensor_specs))
    for i in range(max_struct_length):
      intrinsic_args = []
      for j, _ in enumerate(input_tensor_specs):
        # Scale up any "smaller" structure args by reusing their last element.
        tensor_spec = input_tensor_specs[j][
            min(i, len(input_tensor_specs[j]) - 1)
        ]
        if tensor_spec.name.startswith('update'):
          intrinsic_args.append(
              plan_pb2.ServerAggregationConfig.IntrinsicArg(
                  input_tensor=tensor_spec.experimental_as_proto()
              )
          )
        else:
          intrinsic_args.append(
              plan_pb2.ServerAggregationConfig.IntrinsicArg(
                  state_tensor=tensor_spec.experimental_as_proto()
              )
          )
      aggregations.append(
          plan_pb2.ServerAggregationConfig(
              intrinsic_uri=intrinsic_uris[intrinsic_index],
              intrinsic_args=intrinsic_args,
              # Scale up the output structure by reusing the last element if
              # needed.
              output_tensors=[
                  output_tensor_specs[
                      min(i, len(output_tensor_specs) - 1)
                  ].experimental_as_proto()
              ],
          )
      )

  # Generate the TensorFlow graph needed to execute the server_result step,
  # including reading input checkpoints, writing output checkpoints, and
  # generating output tensors.
  server_result_input_tensors = []
  server_result_output_tensors = []
  server_result_target_nodes = []
  with tf.Graph().as_default() as server_result_graph:
    # Create the placeholders for the input and output filenames needed by
    # the server_result step.
    server_result_intermediate_state_input_filepath_placeholder = (
        tf.compat.v1.placeholder(
            name='server_intermediate_state_input_filepath',
            shape=(),
            dtype=tf.string,
        )
    )
    server_result_aggregate_result_input_filepath_placeholder = (
        tf.compat.v1.placeholder(
            name='aggregate_result_input_filepath', shape=(), dtype=tf.string
        )
    )
    server_result_server_state_output_filepath_placeholder = (
        tf.compat.v1.placeholder(
            name='server_state_output_filepath', shape=(), dtype=tf.string
        )
    )
    server_result_input_tensors.extend([
        server_result_intermediate_state_input_filepath_placeholder,
        server_result_aggregate_result_input_filepath_placeholder,
        server_result_server_state_output_filepath_placeholder,
    ])

    # Restore the intermediate server state.
    intermediate_state = []
    if not intermediate_state_empty:
      intermediate_state_type = daf.server_result.type_signature.parameter[0]
      intermediate_state_vars = variable_helpers.create_vars_for_tff_type(
          intermediate_state_type, 'intermediate_state'
      )
      intermediate_state_tensor_specs = tf.nest.map_structure(
          variable_helpers.tensorspec_from_var, intermediate_state_vars
      )
      intermediate_state = checkpoint_utils.restore_tensors_from_savepoint(
          intermediate_state_tensor_specs,
          server_result_intermediate_state_input_filepath_placeholder,
      )

    # Restore the aggregation results.
    aggregate_result_type = tff.StructType(
        [daf.server_result.type_signature.parameter[1]]
    )
    aggregate_result_vars = variable_helpers.create_vars_for_tff_type(
        aggregate_result_type, 'intermediate_update'
    )
    aggregate_result_tensor_specs = tf.nest.map_structure(
        variable_helpers.tensorspec_from_var, aggregate_result_vars
    )
    aggregate_result = checkpoint_utils.restore_tensors_from_savepoint(
        aggregate_result_tensor_specs,
        server_result_aggregate_result_input_filepath_placeholder,
    )

    # Perform the server_result step.
    server_state_values, server_output_values = graph_helpers.import_tensorflow(
        'server_result',
        tff.framework.ConcreteComputation.from_building_block(
            tff.backends.mapreduce.consolidate_and_extract_local_processing(
                daf.server_result.to_building_block(), grappler_config
            )
        ),
        (intermediate_state, aggregate_result),
        split_outputs=True,
    )

    # Create checkpoints storing the updated server state.
    server_state_save_op = tensor_utils.save(
        filename=server_result_server_state_output_filepath_placeholder,
        tensor_names=variable_helpers.variable_names_from_type(
            daf.server_result.type_signature.result[0], 'server'
        ),
        tensors=server_state_values,
        name='save_server_state_tensors',
    )
    server_result_target_nodes.append(server_state_save_op.name)

    # Generate the output TensorSpecProtos for the server metrics if some exist.
    server_output_empty = (
        isinstance(daf.server_result.type_signature.result[1], tff.StructType)
        and not daf.server_result.type_signature.result[1]
    )
    if not server_output_empty:
      metric_names = variable_helpers.variable_names_from_type(
          daf.server_result.type_signature.result[1], 'server'
      )
      metric_tensors = [
          tf.identity(tensor, name)
          for tensor, name in zip(server_output_values, metric_names)
      ]
      for metric in metric_tensors:
        server_result_output_tensors.append(
            proto_helpers.make_tensor_spec_from_tensor(
                metric
            ).experimental_as_proto()
        )

  # Create the TensorflowSpec messages for the pre-broadcast (server_prepare)
  # and post-aggregation (server_result) steps.
  tensorflow_spec_prepare = plan_pb2.TensorflowSpec(
      input_tensor_specs=[
          proto_helpers.make_tensor_spec_from_tensor(t).experimental_as_proto()
          for t in server_prepare_input_tensors
      ],
      target_node_names=server_prepare_target_nodes,
  )
  tensorflow_spec_result = plan_pb2.TensorflowSpec(
      input_tensor_specs=[
          proto_helpers.make_tensor_spec_from_tensor(t).experimental_as_proto()
          for t in server_result_input_tensors
      ],
      output_tensor_specs=server_result_output_tensors,
      target_node_names=server_result_target_nodes,
  )

  # Create the IORouter messages for the pre-broadcast (server_prepare) and
  # post-aggregation (server_result) steps.
  server_prepare_io_router = plan_pb2.ServerPrepareIORouter(
      prepare_server_state_input_filepath_tensor_name=server_prepare_server_state_input_filepath_placeholder.name,
      prepare_output_filepath_tensor_name=server_prepare_output_filepath_placeholder.name,
      prepare_intermediate_state_output_filepath_tensor_name=server_prepare_intermediate_state_output_filepath_placeholder.name,
  )
  server_result_io_router = plan_pb2.ServerResultIORouter(
      result_intermediate_state_input_filepath_tensor_name=server_result_intermediate_state_input_filepath_placeholder.name,
      result_aggregate_result_input_filepath_tensor_name=server_result_aggregate_result_input_filepath_placeholder.name,
      result_server_state_output_filepath_tensor_name=server_result_server_state_output_filepath_placeholder.name,
  )

  server_phase_v2 = plan_pb2.ServerPhaseV2(
      tensorflow_spec_prepare=tensorflow_spec_prepare,
      prepare_router=server_prepare_io_router,
      aggregations=aggregations,
      tensorflow_spec_result=tensorflow_spec_result,
      result_router=server_result_io_router,
  )

  return (
      server_prepare_graph.as_graph_def(),
      server_result_graph.as_graph_def(),
      server_phase_v2,
  )


def _build_server_graph(
    mrf: tff.backends.mapreduce.MapReduceForm,
    broadcast_tff_type: variable_helpers.AllowedTffTypes,
    is_broadcast_empty: bool,
    flattened_bitwidths: list[int],
    flattened_max_inputs: list[int],
    flattened_moduli: list[int],
    write_metrics_to_checkpoint: bool = True,
    additional_checkpoint_metadata_var_fn: Optional[
        Callable[[tff.StructType, tff.StructType, bool], list[tf.Variable]]
    ] = None,
    experimental_client_update_format: checkpoint_type.CheckpointFormatType = checkpoint_type.CheckpointFormatType.TF1_SAVE_SLICES,
) -> tuple[
    tf.compat.v1.GraphDef,
    plan_pb2.CheckpointOp,
    plan_pb2.ServerPhase,
    list[tf.TensorSpec],
]:
  """Builds the `tf.Graph` that will run on the server.

  Args:
    mrf: A `MapReduceForm` object containing the different computations to
      combine into a single server graph.
    broadcast_tff_type: A `tff.Type` object that specifies the tensors in the
      model that are broadcasted and aggregated.
    is_broadcast_empty: boolean indicating whether the broadcasted value from
      the server was initially empty.
    flattened_bitwidths: The `bitwidth`s for the tensors that will be aggregated
      using `bitwidth`-based secure summation.
    flattened_max_inputs: The max_input`s for the tensors that will be
      aggregated using `max_input`-based secure summation.
    flattened_moduli: The `modulus`s for the tensors that will be aggregated
      using modular secure summation.
    write_metrics_to_checkpoint: If False, revert to legacy behavior where
      metrics values were handled by post-processing separate from the outputted
      checkpoint. Regardless, they will additionally continue to be written to
      recordio and accumulator checkpoints as defined by the Plan proto.
    additional_checkpoint_metadata_var_fn: An optional method that takes in a
      server state type, a server metrics type, and a boolean determining
      whether to revert to legacy metrics behavior to produce additional
      metadata variables.
    experimental_client_update_format: Determines how the client update will be
      interpreted. The value has to match experimental_checkpoint_write argument
      of the  _build_client_graph_with_tensorflow_spec call.

  Returns:
    A `tuple` containing the following (in order):
      - A server `tf.GraphDef`,
      - A server checkpoint,
      - A server phase proto message, and
      - A list of `tf.TensorSpec`s for the broadcasted values.
  """
  (
      simpleagg_update_type,
      secure_sum_bitwidth_update_type,
      secure_sum_update_type,
      secure_modular_sum_update_type,
  ) = mrf.work.type_signature.result
  with tf.Graph().as_default() as server_graph:
    # Creates all server-side variables and savepoints for both the coordinator
    # and the intermediate aggregators.
    # server_state_type will be a SERVER-placed federated type.
    server_state_type, server_metrics_type = mrf.type_signature.result
    assert server_state_type.is_federated(), server_state_type
    assert server_state_type.placement == tff.SERVER, server_state_type
    # server_metrics_type can be a tff.FederatedType or a structure containing
    # tff.FederatedTypes.
    if isinstance(server_metrics_type, tff.FederatedType):
      # We need to check for server metrics without the placement so
      # tff.structure.flatten works correctly.
      has_server_metrics = bool(
          tff.structure.flatten(server_metrics_type.member)
      )
    else:
      has_server_metrics = bool(tff.structure.flatten(server_metrics_type))
    if isinstance(server_metrics_type, tff.TensorType) or (
        isinstance(server_metrics_type, tff.FederatedType)
        and isinstance(server_metrics_type.member, tff.TensorType)
    ):
      # Single tensor; must be wrapped inside of a NamedTuple for proper
      # variable initialization.
      server_metrics_type = tff.StructType([server_metrics_type])
    (
        server_state_vars,
        server_metrics_vars,
        metadata_vars,
        server_savepoint,
    ) = checkpoint_utils.create_server_checkpoint_vars_and_savepoint(
        server_state_type=server_state_type,
        server_metrics_type=server_metrics_type,
        write_metrics_to_checkpoint=write_metrics_to_checkpoint,
        additional_checkpoint_metadata_var_fn=(
            additional_checkpoint_metadata_var_fn
        ),
    )

    # TODO(team): Switch to `tf.save()` in lieu of savers to avoid the
    # need to create client variables on the server.
    client_vars_on_server, write_client = (
        checkpoint_utils.create_state_vars_and_savepoint(
            broadcast_tff_type, 'client'
        )
    )

    secure_sum_update_types = [
        secure_sum_bitwidth_update_type,
        secure_sum_update_type,
        secure_modular_sum_update_type,
    ]
    combined_intermediate_update_type = tff.StructType(
        [mrf.zero.type_signature.result] + secure_sum_update_types
    )

    combined_intermediate_update_vars, write_intermediate_update = (
        checkpoint_utils.create_state_vars_and_savepoint(
            combined_intermediate_update_type, 'intermediate_update'
        )
    )
    num_simpleagg_vars = len(combined_intermediate_update_vars) - len(
        tff.structure.flatten(tff.to_type(secure_sum_update_types))
    )
    intermediate_update_vars = combined_intermediate_update_vars[
        :num_simpleagg_vars
    ]
    secagg_intermediate_update_vars = combined_intermediate_update_vars[
        num_simpleagg_vars:
    ]

    read_secagg_update = _read_secagg_update_from_sidechannel_into_vars(
        secagg_intermediate_update_vars=secagg_intermediate_update_vars,
        secure_sum_bitwidth_update_type=secure_sum_bitwidth_update_type,
        bitwidths=flattened_bitwidths,
        secure_sum_update_type=secure_sum_update_type,
        max_inputs=flattened_max_inputs,
        secure_modular_sum_update_type=secure_modular_sum_update_type,
        moduli=flattened_moduli,
    )

    combined_aggregated_update_vars, write_accumulators = (
        checkpoint_utils.create_state_vars_and_savepoint(
            combined_intermediate_update_type, 'aggregated_update'
        )
    )
    aggregated_update_vars = combined_aggregated_update_vars[
        :num_simpleagg_vars
    ]
    secagg_aggregated_update_vars = combined_aggregated_update_vars[
        num_simpleagg_vars:
    ]

    # Throws in the initializer for all state variables, to be executed prior
    # to restoring the savepoint. Run this variable initializer prior to
    # restoring from the savepoint to allow the vars to be overwritten by the
    # savepoint in this case, and so they do not get re-executed after being
    # overwritten. Also include the metrics vars here in case the execution
    # environment wants to read those in.
    server_vars_initializer = tf.compat.v1.variables_initializer(
        server_state_vars + metadata_vars + server_metrics_vars,
        'initialize_server_state_and_non_state_vars',
    )
    server_savepoint.before_restore_op = server_vars_initializer.name

    # In graph mode, TensorFlow does not allow creating a
    # `tf.compat.v1.train.Saver` when there are no variables. As a result,
    # calling `create_state_vars_and_savepoint` below will fail when there are
    # no SimpleAgg variables (e.g., all results are aggregated via SecAgg). In
    # this case, there are no client checkpoints, and hence, no need to populate
    # the `read_update` field.
    if num_simpleagg_vars > 0:
      # Run the initializer for update vars prior to restoring the client update
      update_vars, read_update = (
          checkpoint_utils.create_state_vars_and_savepoint(
              simpleagg_update_type, artifact_constants.UPDATE
          )
      )
      update_vars_initializer = tf.compat.v1.variables_initializer(
          update_vars, 'initialize_update_vars'
      )
      if (
          experimental_client_update_format
          == checkpoint_type.CheckpointFormatType.APPEND_SLICES_MERGE_READ
      ):
        graph = tf.compat.v1.get_default_graph()
        checkpoint_pl = graph.get_tensor_by_name(
            read_update.saver_def.filename_tensor_name
        )
        merge_checkpoint_slices = append_slices.merge_appended_slices(
            checkpoint_pl, 'merge_checkpoint_slices'
        )
        init_merge = tf.group(update_vars_initializer, merge_checkpoint_slices)
        read_update.before_restore_op = init_merge.name
      else:
        read_update.before_restore_op = update_vars_initializer.name
    else:
      # Create a empty list for `update_vars` when there are no SimpleAgg
      # variables, to be compatible with the `accumulated_values` defined below.
      update_vars = []

    # Copy the intermediate aggregator's update saver for use on coordinator.
    read_intermediate_update = plan_pb2.CheckpointOp()
    read_intermediate_update.CopyFrom(write_intermediate_update)

    # Condition all the remaining logic on the variable initializers, since
    # intermediate aggregators are supposed to be stateless (no savepoint, and
    # therefore no `before_restore_op`, either).
    with tf.control_dependencies(
        [
            tf.compat.v1.variables_initializer(
                (intermediate_update_vars + aggregated_update_vars),
                'initialize_accumulator_vars',
            )
        ]
    ):
      # Embeds the `zero` logic and hooks it up to `after_restore_op` of
      # server's checkpointed state (shared between the coordinator and the
      # intermediate aggregators). The zeros get assigned to
      # `intermediate_update_vars` and to the `aggregated_update_vars` at the
      # very beginning, right after restoring from `server_savepoint`.
      zero_values = graph_helpers.import_tensorflow('zero', mrf.zero)
      assign_zero_ops = tf.nest.map_structure(
          lambda variable, value: variable.assign(value),
          intermediate_update_vars,
          zero_values,
      ) + tf.nest.map_structure(
          lambda variable, value: variable.assign(value),
          aggregated_update_vars,
          zero_values,
      )

    # Embeds the `prepare` logic, and hooks it up to `before_save_op` of
    # client state (to be checkpointed and sent to the clients at the
    # beginning of the round by the central coordinator).
    with tf.control_dependencies(
        [
            tf.compat.v1.variables_initializer(
                client_vars_on_server, 'initialize_client_vars_on_server'
            )
        ]
    ):
      # Configure the session token for `write_client` so that the `prepare`
      # operation may be fed the callback ID for the `SaveSlices` op
      # (necessary for plans containing `federated_select`).
      write_client_session_token = tf.compat.v1.placeholder_with_default(
          input='', shape=(), name='write_client_session_token'
      )
      prepared_values = graph_helpers.import_tensorflow(
          'prepare',
          mrf.prepare,
          server_state_vars,
          session_token_tensor=write_client_session_token,
      )
      if is_broadcast_empty:
        # If the broadcast was empty, don't assigning the sample incoming
        # tf.int32 to anything.
        client_state_assign_ops = [tf.no_op()]
      else:
        client_state_assign_ops = tf.nest.map_structure(
            lambda variable, tensor: variable.assign(tensor),
            client_vars_on_server,
            prepared_values,
        )
    write_client.before_save_op = tf.group(*client_state_assign_ops).name
    write_client.session_token_tensor_name = write_client_session_token.name

    # Embeds the `accumulate` logic, and hooks up the assignment of a client
    # update to the intermediate update to `aggregate_into_accumulators_op`.
    accumulated_values = graph_helpers.import_tensorflow(
        'accumulate', mrf.accumulate, (intermediate_update_vars, update_vars)
    )
    intermediate_update_assign_ops = tf.nest.map_structure(
        lambda variable, tensor: variable.assign(tensor),
        intermediate_update_vars,
        accumulated_values,
    )
    aggregate_into_accumulators_op = tf.group(
        *intermediate_update_assign_ops
    ).name

    secagg_aggregated_update_init = tf.compat.v1.variables_initializer(
        secagg_aggregated_update_vars
    )

    # Reset the accumulators in `phase_init_op`, after variable initializers
    # and after restoring from the savepoint.
    phase_init_op = tf.group(
        *(assign_zero_ops + [secagg_aggregated_update_init])
    ).name

    # Embeds the `merge` logic, and hooks up the assignment of an intermediate
    # update to the top-level aggregate update at the coordinator to
    # `intermediate_aggregate_into_accumulators_op`.
    merged_values = graph_helpers.import_tensorflow(
        'merge', mrf.merge, (aggregated_update_vars, intermediate_update_vars)
    )
    aggregated_update_assign_ops = tf.nest.map_structure(
        lambda variable, tensor: variable.assign(tensor),
        aggregated_update_vars,
        merged_values,
    )

    secagg_aggregated_update_ops = _merge_secagg_vars(
        secure_sum_bitwidth_update_type,
        secure_sum_update_type,
        secure_modular_sum_update_type,
        flattened_moduli,
        secagg_aggregated_update_vars,
        secagg_intermediate_update_vars,
    )

    intermediate_aggregate_into_accumulators_op = tf.group(
        *(aggregated_update_assign_ops + secagg_aggregated_update_ops)
    ).name

    # Embeds the `report` and `update` logic, and hooks up the assignments of
    # the results of the final update to the server state and metric vars, to
    # be triggered by `apply_aggregrated_updates_op`.
    simpleagg_reported_values = graph_helpers.import_tensorflow(
        'report', mrf.report, aggregated_update_vars
    )

    # NOTE: In MapReduceForm, the `update` method takes in the simpleagg vars
    # and SecAgg vars as a tuple of two separate lists. However, here, as
    # above, we concatenate the simpleagg values and the secure values into a
    # single list. This mismatch is not a problem because the variables are all
    # flattened either way when traveling in and out of the tensorflow graph.
    combined_update_vars = (
        simpleagg_reported_values + secagg_aggregated_update_vars
    )
    new_server_state_values, server_metrics_values = (
        graph_helpers.import_tensorflow(
            artifact_constants.UPDATE,
            mrf.update,
            (server_state_vars, combined_update_vars),
            split_outputs=True,
        )
    )

    assign_server_state_ops = tf.nest.map_structure(
        lambda variable, tensor: variable.assign(tensor),
        server_state_vars,
        new_server_state_values,
    )
    assign_non_state_ops = tf.nest.map_structure(
        lambda variable, value: variable.assign(value),
        server_metrics_vars,
        server_metrics_values,
    )
    all_assign_ops = assign_server_state_ops + assign_non_state_ops
    apply_aggregrated_updates_op = tf.group(*all_assign_ops).name

    # Constructs the metadata for server metrics to be included in the plan.
    server_metrics = [
        proto_helpers.make_metric(v, artifact_constants.SERVER_STATE_VAR_PREFIX)
        for v in server_metrics_vars
    ]

  server_phase_kwargs = collections.OrderedDict(
      phase_init_op=phase_init_op,
      write_client_init=write_client,
      read_aggregated_update=read_secagg_update,
      write_intermediate_update=write_intermediate_update,
      read_intermediate_update=read_intermediate_update,
      intermediate_aggregate_into_accumulators_op=(
          intermediate_aggregate_into_accumulators_op
      ),
      write_accumulators=write_accumulators,
      apply_aggregrated_updates_op=apply_aggregrated_updates_op,
      metrics=server_metrics,
  )

  if num_simpleagg_vars > 0:
    # The `read_update` loads SimpleAgg updates from client checkpoints. The
    # `aggregate_into_accumulators_op` aggregates SimpleAgg data after loading
    # the client updates. No need to populate the two fields if there are no
    # SimpleAgg variables (e.g., if all results are aggregated via SecAgg).
    server_phase_kwargs['read_update'] = read_update
    server_phase_kwargs['aggregate_into_accumulators_op'] = (
        aggregate_into_accumulators_op
    )

  server_phase = plan_pb2.ServerPhase(**server_phase_kwargs)

  broadcasted_tensor_specs = tf.nest.map_structure(
      variable_helpers.tensorspec_from_var, client_vars_on_server
  )
  server_graph_def = server_graph.as_graph_def()

  if write_metrics_to_checkpoint:
    server_graph_def = _redirect_save_saver_to_restore_saver_placeholder(
        server_graph_def
    )

  return (
      server_graph_def,
      server_savepoint,
      server_phase,
      broadcasted_tensor_specs,
  )


def _redirect_save_saver_to_restore_saver_placeholder(
    graph_def: tf.compat.v1.GraphDef,
) -> tf.compat.v1.GraphDef:
  """Updates save Saver's savepoint to point to restore Saver's placeholder.

  NOTE: mutates the GraphDef passed in and returns the mutated GraphDef.

  When we created the server_savepoint Saver when we are outputting all of
  the metrics to the output checkpoint as well, we set different nodes for
  saving and restoring so that we could save state + metrics and restore
  just state. However, the only way to do so was to make two Savers and
  splice them together. This meant that the save and restore operations
  depend on two different placeholders for the checkpoint filename. To
  avoid server changes that pass the same checkpoint name in twice to both
  placeholders, we make a few changes to the server GraphDef so that the
  saving op connects back to the placeholder for the restore operation.
  Once this is done, the original save placeholder node will still exist in
  the graph, but it won't be used by any part of the graph that connects to
  an operation we care about.

  Args:
    graph_def: A `tf.compat.v1.GraphDef` to mutate.

  Returns:
    The mutated `tf.compat.v1.GraphDef` that was passed in as graph_def.
  """
  old_const_node = f'{checkpoint_utils.SAVE_SERVER_SAVEPOINT_NAME}/Const'
  new_const_node = (
      f'{artifact_constants.SERVER_STATE_VAR_PREFIX}_savepoint/Const'
  )
  nodes_to_change = [
      f'{checkpoint_utils.SAVE_SERVER_SAVEPOINT_NAME}/save',
      f'{checkpoint_utils.SAVE_SERVER_SAVEPOINT_NAME}/control_dependency',
      f'{checkpoint_utils.SAVE_SERVER_SAVEPOINT_NAME}/RestoreV2',
  ]
  num_changed_nodes = 0
  for node in graph_def.node:
    if node.name in nodes_to_change:
      input_index = 0
      for input_index, input_node in enumerate(node.input):
        if input_node == old_const_node:
          node.input[input_index] = new_const_node
          break
      assert input_index != len(
          node.input
      ), 'Missed input arg in saver GraphDef rewriting.'
      num_changed_nodes = num_changed_nodes + 1
    if num_changed_nodes == len(nodes_to_change):
      # Once we've changed all of the callsites, we stop.
      return graph_def
  return graph_def


def _build_client_graph_with_tensorflow_spec(
    client_work_comp: tff.Computation,
    dataspec,
    broadcasted_tensor_specs: Iterable[tf.TensorSpec],
    is_broadcast_empty: bool,
    *,
    experimental_checkpoint_write: checkpoint_type.CheckpointFormatType = checkpoint_type.CheckpointFormatType.TF1_SAVE_SLICES,
) -> tuple[tf.compat.v1.GraphDef, plan_pb2.ClientPhase]:
  """Builds the client graph and ClientPhase with TensorflowSpec populated.

  This function builds a client phase with tensorflow specs proto.

  Args:
    client_work_comp: A `tff.Computation` that represents the TensorFlow logic
      run on-device.
    dataspec: Either an instance of `data_spec.DataSpec` or a nested structure
      of these that matches the structure of the first element of the input to
      `client_work_comp`.
    broadcasted_tensor_specs: A list of `tf.TensorSpec` containing the name and
      dtype of the variables arriving via the broadcast checkpoint.
    is_broadcast_empty: A boolean indicating whether the MapReduce form
      initially called for an empty broadcast. In this case the
      broadcasted_tensor_specs will contain a single tf.int32, but it will be
      ignored.
    experimental_checkpoint_write: Determines the format of the final client
      update checkpoint. The value affects required operations and might have
      performance implications.

  Returns:
    A `tuple` of the client TensorFlow GraphDef and the client phase protocol
      message.

  Raises:
    SecureAggregationTensorShapeError: If SecAgg tensors do not have all
      dimensions of their shape fully defined.
    ValueError: If any of the arguments are found to be in an unexpected form.
  """
  if (
      not isinstance(client_work_comp.type_signature.parameter, tff.StructType)
      or len(client_work_comp.type_signature.parameter) < 1
  ):
    raise ValueError(
        'client_work_comp.type_signature.parameter should be a '
        '`tff.StructType` with length >= 1, but found: {p}.'.format(
            p=client_work_comp.type_signature.parameter
        )
    )

  if (
      not isinstance(client_work_comp.type_signature.result, tff.StructType)
      or len(client_work_comp.type_signature.result) != 4
  ):
    raise ValueError(
        'client_work_comp.type_signature.result should be a '
        '`tff.StructType` with length == 4, but found: {r}.'.format(
            r=client_work_comp.type_signature.result
        )
    )

  (
      simpleagg_update_type,
      secure_sum_bitwidth_update_type,
      secure_sum_update_type,
      secure_modular_sum_update_type,
  ) = client_work_comp.type_signature.result

  # A list of tensors that will be passed into TensorFlow, corresponding to
  # `plan_pb2.ClientPhase.tensorflow_spec.input_tensor_specs`. Note that the
  # dataset token is excluded from this list. In general, this list should
  # include the filepath placeholder tensors for the input checkpoint file and
  # output checkpoint file.
  input_tensors = []

  # A list of tensor specs that should be fetched from TensorFlow, corresponding
  # to `plan_pb2.ClientPhase.tensorflow_spec.output_tensor_specs`. In general,
  # this list should include the tensors that are not in the output checkpoint
  # file, such as secure aggregation tensors.
  output_tensor_specs = []

  # A list of node names in the client graph that should be executed but no
  # output returned, corresponding to
  # `plan_pb2.ClientPhase.tensorflow_spec.target_node_names`. In general, this
  # list should include the op that creates the output checkpoint file.
  target_nodes = []
  with tf.Graph().as_default() as client_graph:
    input_filepath_placeholder = None
    if not is_broadcast_empty:
      input_filepath_placeholder = tf.compat.v1.placeholder(
          name='input_filepath', shape=(), dtype=tf.string
      )
      weights_from_server = checkpoint_utils.restore_tensors_from_savepoint(
          broadcasted_tensor_specs, input_filepath_placeholder
      )
      input_tensors.append(input_filepath_placeholder)
    else:
      weights_from_server = []

    # Add the custom Dataset ops to the graph.
    token_placeholder, data_values, example_selector_placeholders = (
        graph_helpers.embed_data_logic(
            client_work_comp.type_signature.parameter[0], dataspec
        )
    )

    # Embed the graph coming from TFF into the client work graph.
    combined_update_tensors = graph_helpers.import_tensorflow(
        'work',
        client_work_comp,
        (data_values, weights_from_server),
        split_outputs=False,
        session_token_tensor=token_placeholder,
    )  # pytype: disable=wrong-arg-types

    num_simpleagg_tensors = len(tff.structure.flatten(simpleagg_update_type))
    simpleagg_tensors = combined_update_tensors[:num_simpleagg_tensors]
    secagg_tensors = combined_update_tensors[num_simpleagg_tensors:]

    # For tensors aggregated by secagg, we make sure the tensor names are
    # aligned in both client and sever graph by getting the names from the same
    # method.
    secagg_tensor_names = []
    secagg_tensor_types = []
    for uri, update_type in [
        (SECURE_SUM_BITWIDTH_URI, secure_sum_bitwidth_update_type),
        (SECURE_SUM_URI, secure_sum_update_type),
        (SECURE_MODULAR_SUM_URI, secure_modular_sum_update_type),
    ]:
      secagg_tensor_names += variable_helpers.get_shared_secagg_tensor_names(
          uri, update_type
      )
      secagg_tensor_types += tff.structure.flatten(update_type)

    secagg_tensors = [
        tf.identity(tensor, name=tensor_utils.bare_name(name))
        for tensor, name in zip(secagg_tensors, secagg_tensor_names)
    ]
    for t, type_spec in zip(secagg_tensors, secagg_tensor_types):
      secagg_tensor_spec = proto_helpers.make_tensor_spec_from_tensor(
          t, shape_hint=type_spec.shape
      )
      output_tensor_specs.append(secagg_tensor_spec.experimental_as_proto())

    # Verify that SecAgg Tensors have all dimension fully defined.
    for tensor_spec in output_tensor_specs:
      if not tf.TensorShape(tensor_spec.shape).is_fully_defined():
        raise SecureAggregationTensorShapeError(
            '`TensorflowSpec.output_tensor_specs` has unknown dimension.'
        )

    output_filepath_placeholder = None
    if simpleagg_tensors:
      output_filepath_placeholder = tf.compat.v1.placeholder(
          dtype=tf.string, shape=(), name='output_filepath'
      )
      simpleagg_variable_names = variable_helpers.variable_names_from_type(
          simpleagg_update_type, name=artifact_constants.UPDATE
      )
      if experimental_checkpoint_write in [
          checkpoint_type.CheckpointFormatType.APPEND_SLICES_MERGE_WRITE,
          checkpoint_type.CheckpointFormatType.APPEND_SLICES_MERGE_READ,
      ]:
        delete_op = delete_file.delete_file(output_filepath_placeholder)
        with tf.control_dependencies([delete_op]):
          append_ops = []
          for tensor_name, tensor in zip(
              simpleagg_variable_names, simpleagg_tensors
          ):
            append_ops.append(
                tensor_utils.save(
                    filename=output_filepath_placeholder,
                    tensor_names=[tensor_name],
                    tensors=[tensor],
                    save_op=append_slices.append_slices,
                )
            )
        if (
            experimental_checkpoint_write
            == checkpoint_type.CheckpointFormatType.APPEND_SLICES_MERGE_WRITE
        ):
          with tf.control_dependencies(append_ops):
            save_op = append_slices.merge_appended_slices(
                filename=output_filepath_placeholder
            )
        else:
          # APPEND_SLICES_MERGE_READ
          save_op = tf.group(*append_ops)

      elif (
          experimental_checkpoint_write
          == checkpoint_type.CheckpointFormatType.TF1_SAVE_SLICES
      ):
        save_op = tensor_utils.save(
            filename=output_filepath_placeholder,
            tensor_names=simpleagg_variable_names,
            tensors=simpleagg_tensors,
            name='save_client_update_tensors',
        )
      else:
        raise NotImplementedError(
            f'Unsupported CheckpointFormatType {experimental_checkpoint_write}.'
        )
      input_tensors.append(output_filepath_placeholder)
      target_nodes.append(save_op.name)

  tensorflow_spec = plan_pb2.TensorflowSpec()
  if token_placeholder is not None:
    tensorflow_spec.dataset_token_tensor_name = token_placeholder.name
  if input_tensors:
    tensorflow_spec.input_tensor_specs.extend(
        tf.TensorSpec.from_tensor(t, name=t.name).experimental_as_proto()
        for t in input_tensors
    )
  if output_tensor_specs:
    tensorflow_spec.output_tensor_specs.extend(output_tensor_specs)
  if target_nodes:
    tensorflow_spec.target_node_names.extend(target_nodes)
  if example_selector_placeholders:
    for placeholder in example_selector_placeholders:
      # Generating the default TensorProto will create a TensorProto with an
      # DT_INVALID DType. This identifies that there is a placeholder that is
      # needed. In order to have the Plan proto be completely runnable, the
      # value will need to be filled in with a real TensorProto that matches
      # the shape/type of the expected input.
      tensorflow_spec.constant_inputs[placeholder.name].dtype = 0

  io_router = plan_pb2.FederatedComputeIORouter()
  if input_filepath_placeholder is not None:
    io_router.input_filepath_tensor_name = input_filepath_placeholder.name
  if output_filepath_placeholder is not None:
    io_router.output_filepath_tensor_name = output_filepath_placeholder.name
  for secagg_tensor in secagg_tensors:
    io_router.aggregations[secagg_tensor.name].CopyFrom(
        plan_pb2.AggregationConfig(
            secure_aggregation=plan_pb2.SecureAggregationConfig()
        )
    )

  return client_graph.as_graph_def(), plan_pb2.ClientPhase(
      tensorflow_spec=tensorflow_spec, federated_compute=io_router
  )


def _build_client_phase_with_example_query_spec(
    client_work_comp: tff.Computation,
    example_query_spec: plan_pb2.ExampleQuerySpec,
) -> plan_pb2.ClientPhase:
  """Builds the ClientPhase with `ExampleQuerySpec` populated.

  Args:
    client_work_comp: A `tff.Computation` that represents the TensorFlow logic
      run on-device.
    example_query_spec: Field containing output vector information for client
      example query. The output vector names listed in the spec are expected to
      be consistent with the output names we would produce in the
      `MapReduceForm` client work computation, if we were to build a TF-based
      plan from that `MapReduceForm`.

  Returns:
    A client phase part of the federated protocol.
  """
  expected_vector_names = set(
      variable_helpers.variable_names_from_type(
          client_work_comp.type_signature.result[0], artifact_constants.UPDATE
      )
  )
  used_names = set()
  io_router = plan_pb2.FederatedExampleQueryIORouter()
  for example_query in example_query_spec.example_queries:
    vector_names = set(example_query.output_vector_specs.keys())
    if not all([name in expected_vector_names for name in vector_names]):
      raise ValueError(
          'Found unexpected vector names in supplied `example_query_spec`. '
          f'Expected names: {expected_vector_names}. '
          f'Found unexpected names: {vector_names-expected_vector_names}.'
      )

    if any([name in used_names for name in vector_names]):
      raise ValueError(
          'Duplicate vector names found in supplied `example_query_spec`. '
          f'Duplicates: {vector_names.intersection(used_names)}'
      )

    used_names.update(vector_names)

    for vector_name in vector_names:
      io_router.aggregations[vector_name].CopyFrom(
          plan_pb2.AggregationConfig(
              tf_v1_checkpoint_aggregation=plan_pb2.TFV1CheckpointAggregation()
          )
      )

  if used_names != expected_vector_names:
    raise ValueError(
        'Not all expected vector names were in supplied `example_query_spec`.'
        f' Expected names: {expected_vector_names}. Names not present in'
        f' `example_query_spec`: {expected_vector_names-vector_names}'
    )
  return plan_pb2.ClientPhase(
      example_query_spec=example_query_spec, federated_example_query=io_router
  )


def build_plan(
    mrf: tff.backends.mapreduce.MapReduceForm,
    daf: Optional[tff.backends.mapreduce.DistributeAggregateForm] = None,
    dataspec: Optional[data_spec.NestedDataSpec] = None,
    example_query_spec: Optional[plan_pb2.ExampleQuerySpec] = None,
    grappler_config: Optional[tf.compat.v1.ConfigProto] = None,
    additional_checkpoint_metadata_var_fn: Optional[
        Callable[[tff.StructType, tff.StructType, bool], list[tf.Variable]]
    ] = None,
    experimental_client_checkpoint_write: checkpoint_type.CheckpointFormatType = checkpoint_type.CheckpointFormatType.TF1_SAVE_SLICES,
    generate_server_phase_v2: bool = False,
    write_metrics_to_checkpoint: bool = True,
) -> plan_pb2.Plan:
  """Constructs an instance of `plan_pb2.Plan` given a `MapReduceForm` instance.

  Plans generated by this method are executable, but a number of features have
  yet to be implemented.

  These include:

  - Setting metrics' `stat_name` field based on externally-supplied metadata,
    such as that from the model stampers. Currently, these names are based on
    the names of TensorFlow variables, which in turn are based on the TFF
    type signatures.

  - Populating the client `example_selector` field. Currently not set.

  - Populating client-side `savepoint`. Currently not set.

  - Populating the plan's `tensorflow_config_proto`. Currently not set.

  - Setting a field in the plan that represets a token to drive the custom op
    that iplements the client-side dataset. There is no such field in the plan
    at the time of this writing.

  - Populating plan fields related to secure aggregation and side channels,
    such as the `read_aggregated_update` checkpoint op.

  Args:
    mrf: An instance of `tff.backends.mapreduce.MapReduceForm`.
    daf: An instance of `tff.backends.mapreduce.DistributeAggregateForm`.
    dataspec: If provided, either an instance of `data_spec.DataSpec` or a
      nested structure of these that matches the structure of the first element
      of the input to client-side processing computation `mrf.work`. If not
      provided and `example_query_spec` is also not provided, then placeholders
      are added to the client graph via `embed_data_logic()` and the example
      selectors will need to be passed to the client via the `constant_inputs`
      part of the `TensorflowSpec`. The constant_inputs field needs to be
      populated outside of `build_plan()`. Can only provide one of `dataspec` or
      `example_query_spec`.
    example_query_spec: An instance of `plan_pb2.ExampleQuerySpec`. If provided
      it is assumed a light weight client plan should be constructed. No client
      graph will be included in the produced plan object. Instead the generated
      plan will have an `ExampleQuerySpec` and `FederatedExampleQueryIORouter`.
      Can only supply one of `dataspec` or `example_query_spec`.
    grappler_config: The config specifying Grappler optimizations for TFF-
      generated graphs. Should be provided if daf is provided.
    additional_checkpoint_metadata_var_fn: An optional method that takes in a
      server state type, a server metrics type, and a boolean determining
      whether to revert to legacy metrics behavior to produce additional
      metadata variables.
    experimental_client_checkpoint_write: Determines the style of writing of the
      client checkpoint (client->server communication). The value affects the
      operation used and might have impact on overall task performance.
    generate_server_phase_v2: Iff `True`, will produce a ServerPhaseV2 message
      in the plan in addition to a ServerPhase message.
    write_metrics_to_checkpoint: If False, revert to legacy behavior where
      metrics values were handled by post-processing separate from the outputted
      checkpoint. Regardless, they will additionally continue to be written to
      recordio and accumulator checkpoints as defined by the Plan proto.

  Returns:
    An instance of `plan_pb2.Plan` corresponding to MapReduce form `mrf`.

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If any of the arguments are found to be in an unexpected form.
  """
  type_checks.check_type(mrf, tff.backends.mapreduce.MapReduceForm, name='mrf')
  client_plan_type = (
      ClientPlanType.TENSORFLOW
      if example_query_spec is None
      else ClientPlanType.EXAMPLE_QUERY
  )

  if example_query_spec is not None:
    if dataspec is not None:
      raise ValueError(
          '`example_query_spec` or `dataspec` cannot both be specified.'
      )

  with tff.framework.get_context_stack().install(
      tff.test.create_runtime_error_context()
  ):
    is_broadcast_empty = (
        isinstance(mrf.prepare.type_signature.result, tff.StructType)
        and not mrf.prepare.type_signature.result
    )
    if is_broadcast_empty:
      # This MapReduceForm does not send any server state to clients, however we
      # need something to satisfy current restrictions from the FCP server.
      # Use a placeholder scalar int.
      broadcast_tff_type = tff.TensorType(tf.int32)
    else:
      broadcast_tff_type = mrf.prepare.type_signature.result

    # Execute the bitwidths TFF computation using the default TFF executor.
    bitwidths, max_inputs, moduli = _compute_secagg_parameters(mrf)
    # Note: the variables below are flat lists, even though
    # `secure_sum_bitwidth_update_type`
    # could potentially represent a large group of nested tensors. In order
    # for each var to line up with the appropriate bitwidth, we must also
    # flatten the list of bitwidths.
    flattened_bitwidths = tff.structure.flatten(bitwidths)
    flattened_max_inputs = tff.structure.flatten(max_inputs)
    flattened_moduli = tff.structure.flatten(moduli)

    (
        server_graph_def,
        server_savepoint,
        server_phase,
        broadcasted_tensor_specs,
    ) = _build_server_graph(
        mrf,
        broadcast_tff_type,
        is_broadcast_empty,
        flattened_bitwidths,
        flattened_max_inputs,
        flattened_moduli,
        write_metrics_to_checkpoint,
        additional_checkpoint_metadata_var_fn,
        experimental_client_update_format=experimental_client_checkpoint_write,
    )

    if client_plan_type == ClientPlanType.TENSORFLOW:
      client_graph_def, client_phase = _build_client_graph_with_tensorflow_spec(
          mrf.work,
          dataspec,
          broadcasted_tensor_specs,
          is_broadcast_empty,
          experimental_checkpoint_write=experimental_client_checkpoint_write,
      )
    elif client_plan_type == ClientPlanType.EXAMPLE_QUERY:
      client_phase = _build_client_phase_with_example_query_spec(
          mrf.work, example_query_spec
      )
    else:
      raise ValueError(
          f'Unexpected value for `client_plan_type`: {client_plan_type}'
      )

    combined_phases = plan_pb2.Plan.Phase(
        server_phase=server_phase, client_phase=client_phase
    )

    if generate_server_phase_v2:
      assert daf
      assert grappler_config
      (server_graph_def_prepare, server_graph_def_result, server_phase_v2) = (
          _build_server_graphs_from_distribute_aggregate_form(
              daf, is_broadcast_empty, grappler_config
          )
      )
      combined_phases.server_phase_v2.CopyFrom(server_phase_v2)

    plan = plan_pb2.Plan(
        version=1, server_savepoint=server_savepoint, phase=[combined_phases]
    )

    plan.server_graph_bytes.Pack(server_graph_def)
    if client_plan_type == ClientPlanType.TENSORFLOW:
      plan.client_graph_bytes.Pack(client_graph_def)

    if generate_server_phase_v2:
      plan.server_graph_prepare_bytes.Pack(server_graph_def_prepare)
      plan.server_graph_result_bytes.Pack(server_graph_def_result)
  return plan


def build_cross_round_aggregation_execution(
    mrf: tff.backends.mapreduce.MapReduceForm,
) -> bytes:
  """Constructs an instance of `plan_pb2.CrossRoundAggregationExecution`.

  Args:
    mrf: An instance of `tff.backends.mapreduce.MapReduceForm`.

  Returns:
    A serialized instance of `plan_pb2.CrossRoundAggregationExecution` for given
    `mrf`.
  """
  type_checks.check_type(mrf, tff.backends.mapreduce.MapReduceForm, name='mrf')

  server_metrics_type = mrf.update.type_signature.result[1]
  (
      simpleagg_update_type,
      secure_sum_bitwidth_update_type,
      secure_sum_update_type,
      secure_modular_sum_update_type,
  ) = mrf.work.type_signature.result
  # We don't ever work directly on `simpleagg_update_type` because client
  # updates are transformed by `accumulate` and `merge` before ever being passed
  # into cross-round aggregation.
  del simpleagg_update_type
  simpleagg_merge_type = mrf.merge.type_signature.result
  flattened_moduli = tff.structure.flatten(mrf.secure_modular_sum_modulus())

  if not server_metrics_type:
    # No metrics to aggregrate; will initialize to no-op.
    server_metrics_type = tff.StructType([])
  elif isinstance(server_metrics_type, tff.TensorType):
    # Single tensor metric; must be wrapped inside of a NamedTuple for proper
    # variable initialiazation.
    server_metrics_type = tff.StructType([server_metrics_type])
  combined_aggregated_update_type = tff.StructType([
      simpleagg_merge_type,
      secure_sum_bitwidth_update_type,
      secure_sum_update_type,
      secure_modular_sum_update_type,
  ])

  with tf.Graph().as_default() as cross_round_aggregation_graph:
    server_state_vars = variable_helpers.create_vars_for_tff_type(
        mrf.update.type_signature.parameter[0],
        artifact_constants.SERVER_STATE_VAR_PREFIX,
    )

    combined_aggregated_update_vars, read_aggregated_update = (
        checkpoint_utils.create_state_vars_and_savepoint(
            combined_aggregated_update_type, 'aggregated_update'
        )
    )

    num_simpleagg_vars = len(tff.structure.flatten(simpleagg_merge_type))

    aggregated_update_vars = combined_aggregated_update_vars[
        :num_simpleagg_vars
    ]
    secagg_aggregated_update_vars = combined_aggregated_update_vars[
        num_simpleagg_vars:
    ]

    # Add a new output for metrics_loader `merge` and `report`.
    combined_final_accumulator_vars, read_write_final_accumulators = (
        checkpoint_utils.create_state_vars_and_savepoint(
            combined_aggregated_update_type, 'final_accumulators'
        )
    )

    final_accumulator_vars = combined_final_accumulator_vars[
        :num_simpleagg_vars
    ]
    secagg_final_accumulator_vars = combined_final_accumulator_vars[
        num_simpleagg_vars:
    ]

    var_init_op = tf.compat.v1.initializers.variables(
        server_state_vars
        + combined_aggregated_update_vars
        + combined_final_accumulator_vars
    )

    # Embeds the MapReduce form `merge` logic.
    merged_values = graph_helpers.import_tensorflow(
        'merge', mrf.merge, (final_accumulator_vars, aggregated_update_vars)
    )
    final_accumulator_assign_ops = tf.nest.map_structure(
        lambda variable, tensor: variable.assign(tensor),
        final_accumulator_vars,
        merged_values,
    )

    # SecAgg tensors' aggregation is not provided in the imported TensorFlow,
    # but is instead fixed based on the operator (e.g. `assign_add` for
    # variables passed into `secure_sum`).
    secagg_final_accumulator_ops = _merge_secagg_vars(
        secure_sum_bitwidth_update_type,
        secure_sum_update_type,
        secure_modular_sum_update_type,
        flattened_moduli,
        secagg_final_accumulator_vars,
        secagg_aggregated_update_vars,
    )
    final_accumulator_op = tf.group(
        *(final_accumulator_assign_ops + secagg_final_accumulator_ops)
    ).name

    # Embeds the `report` and `update` logic, and hooks up the assignments of
    # the results of the final update to the server state and metric vars, to
    # be triggered by `apply_aggregrated_updates_op`.
    simpleagg_reported_values = graph_helpers.import_tensorflow(
        'report', mrf.report, final_accumulator_vars
    )
    combined_final_vars = (
        simpleagg_reported_values + secagg_final_accumulator_vars
    )
    (_, server_metric_values) = graph_helpers.import_tensorflow(
        artifact_constants.UPDATE,
        mrf.update,
        (server_state_vars, combined_final_vars),
        split_outputs=True,
    )

    server_metrics_names = variable_helpers.variable_names_from_type(
        server_metrics_type, name=artifact_constants.SERVER_STATE_VAR_PREFIX
    )

    flattened_metrics_types = tff.structure.flatten(server_metrics_type)
    measurements = [
        proto_helpers.make_measurement(v, s, a)
        for v, s, a in zip(
            server_metric_values, server_metrics_names, flattened_metrics_types
        )
    ]

  cross_round_aggregation_execution = plan_pb2.CrossRoundAggregationExecution(
      init_op=var_init_op.name,
      read_aggregated_update=read_aggregated_update,
      merge_op=final_accumulator_op,
      read_write_final_accumulators=read_write_final_accumulators,
      measurements=measurements,
  )

  cross_round_aggregation_execution.cross_round_aggregation_graph_bytes.Pack(
      cross_round_aggregation_graph.as_graph_def()
  )

  return cross_round_aggregation_execution.SerializeToString()
