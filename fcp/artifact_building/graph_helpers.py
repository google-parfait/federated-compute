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
"""Utilities for manipulating TensorFlow graph logic."""

from typing import Optional, Union

import tensorflow as tf
import tensorflow_federated as tff

from fcp.artifact_building import data_spec
from fcp.artifact_building import tensor_utils
from fcp.artifact_building import type_checks
from fcp.tensorflow import external_dataset
from tensorflow_federated.proto.v0 import computation_pb2

TfValue = Union[tf.Variable, tf.Tensor]
DatasetTensor = tf.Tensor
Argument = Union[TfValue, list[TfValue], DatasetTensor]
Args = Optional[Union[Argument, tuple[Argument, ...]]]

Result = Argument
MaybeSplitOutputs = Union[Result, tuple[Result, ...]]


def embed_data_logic(
    client_data_type: tff.Type, dataspec: data_spec.NestedDataSpec
) -> tuple[tf.Tensor, list[MaybeSplitOutputs]]:
  """Embeds the data logic into the current TensorFlow graph.

  Adds dataset ops to the current graph, using the custom `ExternalDataset`
  which returns a placeholder token. The initialization op and data values are
  also returned.

  Args:
    client_data_type: The TFF type signature of the input client data.
    dataspec: Either an instance of `data_spec.DataSpec` or a nested structure
      of these that matches the structure of the first element of the input to
      the client work part of the computation.

  Returns:
    A `tuple` containing the following (in order):
      token_placeholder: A tuple of dataset token tensor
      data_values: A list of dataset output values

  Raises:
    ValueError: If the number of dataset output from one data source is not 1.
  """
  assert dataspec

  data_values = []
  # Embeds the token placeholder for the custom ExternalDataset op.
  token_placeholder = tf.compat.v1.placeholder(
      tf.string, shape=[], name='data_token')
  import_args = [token_placeholder]
  data_sources = make_data_sources(client_data_type, dataspec)

  # Embeds data source computation into the current graph.
  for index, data_comp in enumerate(data_sources):
    ds_values = import_tensorflow('data_{}'.format(index), data_comp,
                                  import_args)
    if len(ds_values) != 1:
      raise ValueError(
          'Expected one dataset output from a data source, found {}.'.format(
              str(len(ds_values))))
    data_values.extend(ds_values)

  return token_placeholder, data_values


def import_tensorflow(
    name: str,
    comp: tff.framework.ConcreteComputation,
    args: Args = None,
    split_outputs: bool = False,
    session_token_tensor: Optional[tf.Tensor] = None,
) -> MaybeSplitOutputs:
  """Imports a tensorflow computation into the current graph.

  Args:
    name: The string name to use as the graph import prefix.
    comp: An instance of `tff.framework.ConcreteComputation` with just the
      `tensorflow` section.
    args: Either a single argument, a tuple of arguments, or None. An argument
      must be either: - a Python `list` containing either tensors or variables,
      or - a single variant tensor representing a dataset input.
    split_outputs: Whether to unpack the result tuple into a Python tuple. If
      `True`, `import_tensorflow` will return a tuple with multiple result
      objects, corresponding to the return elements in the type signature of
      `comp`. Notice that the return type signature of `comp` must be a tuple in
      this case. If `False`, `import_tensorflow` will return the entire result
      in a flattened form as a single Python result object. Each Python result
      object, similar to the argumens in `args`, will be either a Python `list`
      of variant tensors or a singleton Python list containing only the dataset
      variant tensor.
    session_token_tensor: A tensor in the current graph containing the "session
      token" of the TensorFlow being imported. This is useful for passing a
      session-global identifier into the graph for use with ops like
      `ServeSlices` and `ExternalDataset` that take in a token which references
      session-global state.

  Returns:
    One of:
      - A single result (Python `list` of variable value or variant tensors) if
        `split_outputs` is `False`.
      - A Python `tuple` of such results, if `split_outputs` is `True`.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  type_checks.check_type(name, str, name='name')
  type_checks.check_type(comp, tff.framework.ConcreteComputation, name='comp')
  type_checks.check_type(split_outputs, bool, name='split_outputs')

  comp_proto = tff.framework.ConcreteComputation.get_proto(comp)
  type_checks.check_type(
      comp_proto, computation_pb2.Computation, name='comp_proto')

  which_comp = comp_proto.WhichOneof('computation')
  if which_comp != 'tensorflow':
    raise TypeError(
        'Expected a TensorFlow computation, found {}.'.format(which_comp))
  if args is None:
    input_map = None
  elif isinstance(args, tuple):
    which_binding = comp_proto.tensorflow.parameter.WhichOneof('binding')
    if which_binding != 'struct':
      raise TypeError(
          'Expected a struct binding with a struct of args, found {}.'.format(
              which_binding))
    input_map = {}
    for index, arg in enumerate(args):
      input_map.update(
          create_tensor_map(
              comp_proto.tensorflow.parameter.struct.element[index], arg))
  else:
    input_map = create_tensor_map(comp_proto.tensorflow.parameter, args)
  if input_map is not None:
    # Add remappings for all potential control dependencies in the graph as
    # well. Since `tf.graph_util.import_graph_def` input map works on the tensor
    # (not graph node) level, we must handle this case also.
    def control_dep_name(name: str) -> str:
      if name.startswith('^'):
        return name
      node_name = name.split(':', maxsplit=1)[0]
      return f'^{node_name}'

    input_map.update({
        control_dep_name(k): control_dep_name(v.name)
        for k, v in input_map.items()
        if not k.startswith('^')
    })
  input_map = {} if input_map is None else input_map
  if (session_token_tensor is not None and
      comp_proto.tensorflow.session_token_tensor_name):
    input_map[
        comp_proto.tensorflow.session_token_tensor_name] = session_token_tensor
  if split_outputs:
    return_elements = []
    subset_sizes = []
    which_binding = comp_proto.tensorflow.result.WhichOneof('binding')
    if which_binding != 'struct':
      raise TypeError(
          'If `split_outputs` is `True`, the result of the computation we are '
          'importing must be a `struct`; found {}.'.format(which_binding))
    for binding in comp_proto.tensorflow.result.struct.element:
      tensor_names = _list_tensor_names_in_binding(binding)
      return_elements.extend(tensor_names)
      subset_sizes.append(len(tensor_names))
  else:
    return_elements = _list_tensor_names_in_binding(
        comp_proto.tensorflow.result)
    subset_sizes = [len(return_elements)]

  graph_def = tensor_utils.import_graph_def_from_any(
      comp_proto.tensorflow.graph_def)

  # We will be importing multiple GraphDefs into the server or client graphs.
  # These individual graphs may have identifical `shared_name` attributes on
  # variable ops, which causes the runtime to reference the same resource, which
  # is highly undesired. We must uniquify the names before importing.
  def uniquify_shared_names(graph_def: tf.compat.v1.GraphDef,
                            suffix: bytes) -> tf.compat.v1.GraphDef:
    for x in graph_def.node:
      shared_name = x.attr.get('shared_name')
      if shared_name is not None:
        if not shared_name.s:
          # Encountered an empty string shared name, avoid creating a shared
          # name that starts with an underscore (not allowed by TF).
          shared_name.s = b'None'
        shared_name.s += b'_' + suffix
    return graph_def

  uniquified_graph_def = uniquify_shared_names(
      graph_def, suffix=name.encode('utf-8'))
  if comp_proto.tensorflow.initialize_op:
    uniquified_graph_def = add_control_deps_for_init_op(
        uniquified_graph_def, comp_proto.tensorflow.initialize_op)
  import_result = tf.graph_util.import_graph_def(
      uniquified_graph_def,
      input_map=input_map,
      return_elements=return_elements,
      name=name)

  if split_outputs:
    subsets = []
    offset = 0
    for subset_size in subset_sizes:
      next_offset = offset + subset_size
      subsets.append(import_result[offset:next_offset])
      offset = next_offset
    results = tuple(subsets)
  else:
    results = import_result[:subset_sizes[0]]
  return results


def _get_deps_for_graph_node(graph_def: tf.compat.v1.GraphDef,
                             node_name: str) -> set[str]:
  """Returns the set of node names that a node named `node_name` depends on.

  Note that this function does not work for nodes in the function library.

  Args:
    graph_def: The input graph, an instance of `tf.compat.v1.GraphDef`.
    node_name: The node name, a string.

  Returns:
    An instance of `set()` containing string names of the nodes `node_name`
    depends on in graph_def.

  Raises:
    TypeError: If either argument is of the wrong type.
  """
  type_checks.check_type(graph_def, tf.compat.v1.GraphDef, name='graph_def')
  type_checks.check_type(node_name, str, name='node_name')
  input_map = {}
  for node in graph_def.node:
    input_map[node.name] = set(tensor_utils.bare_name(x) for x in node.input)
  dependencies = set()
  initial_singleton = set([node_name])
  nodes_to_process = initial_singleton
  while nodes_to_process:
    dependencies.update(nodes_to_process)
    nodes_to_process = set.union(
        *[input_map[name]
          for name in nodes_to_process]).difference(dependencies)
  return dependencies.difference(initial_singleton)


def add_control_deps_for_init_op(graph_def: tf.compat.v1.GraphDef,
                                 init_op: str) -> tf.compat.v1.GraphDef:
  """Adds control deps on `init_op` to nodes in GraphDef.

  Note that control deps are not added to any of the ancestors of `init_op`
  (which would result in a control dep cycle) and control deps are not added to
  any nodes in the function library of a GraphDef.

  Args:
    graph_def: The input graph, an instance of `tf.compat.v1.GraphDef`.
    init_op: The init op name, a string.

  Returns:
    The updated graph, an instance of `tf.compat.v1.GraphDef`.

  Raises:
    TypeError: If either argument is of the wrong type.
  """
  type_checks.check_type(graph_def, tf.compat.v1.GraphDef, name='graph_def')
  type_checks.check_type(init_op, str, name='init_op')
  init_op_str = tensor_utils.bare_name(init_op)
  init_op_control_dep = '^{}'.format(init_op_str)
  deps = _get_deps_for_graph_node(graph_def,
                                  init_op_str).union(set([init_op_str]))
  new_graph_def = tf.compat.v1.GraphDef()
  new_graph_def.CopyFrom(graph_def)
  for new_node in new_graph_def.node:
    if new_node.name not in deps:
      node_inputs = new_node.input
      if init_op_control_dep not in node_inputs:
        new_node.input.append(init_op_control_dep)
  return new_graph_def


def create_tensor_map(
    binding: computation_pb2.TensorFlow.Binding,
    arg: list[Union[tf.Tensor, tf.Variable]]) -> dict[str, tf.Tensor]:
  """Creates a `dict` mapping tensor names in the binding to tensors in `arg`.

  Args:
    binding: An instance of `computation_pb2.TensorFlow.Binding`.
    arg: Either a singleton Python `list` with variant tensor in case of a
      sequence binding, or a Python `list` of tensors or resource variables
      otherwise for a tuple binding.

  Returns:
    An instance of Python `dict` with the map as specified above.

  Raises:
    TypeError: If the argument types are incorrect.
    ValueError: If the arguments are malformed (e.g., multiple variant tensors).
  """
  type_checks.check_type(
      binding, computation_pb2.TensorFlow.Binding, name='binding')
  type_checks.check_type(arg, list, name='arg')
  tensor_names_in_binding = _list_tensor_names_in_binding(binding)
  which_binding = binding.WhichOneof('binding')
  if which_binding == 'sequence':
    if (len(tensor_names_in_binding) != 1) or (len(arg) != 1):
      raise ValueError('Multiple variant tensors found.')
    variant_tensor_name = tensor_names_in_binding[0]
    arg = arg[0]
    if not tf.is_tensor(arg):
      raise TypeError('Expected a tensor, found {!r}.'.format(type(arg)))
    if arg.dtype != tf.variant:
      raise TypeError('Expected `tf.variant`, found {!r}.'.format(arg.dtype))
    return {variant_tensor_name: arg}
  else:
    return {
        k: (v.read_value() if hasattr(v, 'read_value') else v)
        for k, v in zip(tensor_names_in_binding, arg)
    }


def make_data_sources(type_spec, ds) -> list[tff.Computation]:
  """Creates a list of computations that feed data using the custom ExternalDataset op.

  Args:
    type_spec: The TFF type signature of the output, which must be either a
      sequence, or a named tuple of sequences.
    ds: Either a single `data_spec.DataSpec`, or a nested structure of these,
      made up of Python containers, that exactly matches the structure of the
      `type_spec`.

  Returns:
    A list of `tff.Computation`s, each of which accepts a single `string`-typed
    tensor as input (the token for the ExternalDataset op) and returns a
    sequence as output (with the result that matches the corresponding part of
    `type_spec`). The computations appear on the list in a depth-first order
    (matching exactly the convention used in the
    `_list_tensor_names_in_binding()` method below).

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  type_spec = tff.to_type(type_spec)
  type_checks.check_type(
      type_spec, (tff.SequenceType, tff.StructType), name='type_spec')
  if type_spec.is_sequence():
    type_checks.check_type(ds, data_spec.DataSpec)
    assert isinstance(ds, data_spec.DataSpec)
    sel_bytes = ds.example_selector_proto.SerializeToString()

    @tff.tf_computation(tf.string)
    def data_comp(token):
      """The data source computation.

      Args:
        token: The token placeholder tensor (`tf.string`).

      Returns:
        An instance of `tf.data.Dataset`.
      """
      if ds.preprocessing_fn is not None:
        processed_ds = ds.preprocessing_fn(
            external_dataset.ExternalDataset(token=token, selector=sel_bytes))
      else:
        processed_ds = external_dataset.ExternalDataset(
            token=token, selector=sel_bytes)

      if 'Dataset' not in type(processed_ds).__name__:
        raise TypeError(
            'The preprocessing function returned an unrecognized non-dataset '
            'type {!r}.'.format(type(processed_ds)))
      return processed_ds

    type_checks.check_type(data_comp.type_signature, tff.FunctionType)
    if not type_spec.is_assignable_from(data_comp.type_signature.result):
      type_mismatch_string = tff.types.type_mismatch_error_message(
          type_spec, data_comp.type_signature.result,
          tff.types.TypeRelation.ASSIGNABLE)
      raise TypeError(
          'The data source constructed with the supplied dataspec returns data '
          'which does not match type of request. Details of the mismatch:\n' +
          type_mismatch_string)
    return [data_comp]
  else:
    type_spec.check_struct()
    if isinstance(ds, data_spec.DataSpec):
      raise TypeError('Expected nested structure of `DataSpec`s conforming to '
                      f'the structure of the type {type_spec}. '
                      'Found single `DataSpec` instead.')
    ds = tff.structure.from_container(ds)
    assert isinstance(ds, tff.structure.Struct)
    type_spec_elements = tff.structure.to_elements(type_spec)
    data_spec_elements = tff.structure.to_elements(ds)
    type_spec_element_names = [str(k) for k, _ in type_spec_elements]
    data_spec_element_names = [str(k) for k, _ in data_spec_elements]
    if type_spec_element_names != data_spec_element_names:
      raise TypeError(
          'Type vs. data spec elements names mismatch: {} vs. {}.'.format(
              str(type_spec_element_names), str(data_spec_element_names)))
    elements = []
    for element_index, (_, element_type) in enumerate(type_spec_elements):
      elements.extend(make_data_sources(element_type, ds[element_index]))
    return elements


def _list_tensor_names_in_binding(
    binding: computation_pb2.TensorFlow.Binding) -> list[str]:
  """Returns a flat Python list of tensor names that appear in the `binding`.

  Args:
    binding: An instance of `computation_pb2.TensorFlow.Binding` in which any
      sequence bindings must contain variant tensors.

  Returns:
    A list of `str` instances with tensor names that appear in `binding` in the
    order in which they appear in the depth-first traversal of the potentially
    nested binding structure.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  type_checks.check_type(binding, computation_pb2.TensorFlow.Binding)
  which_binding = binding.WhichOneof('binding')
  if which_binding == 'tensor':
    return [str(binding.tensor.tensor_name)]
  elif which_binding == 'struct':
    result = []
    for element in binding.struct.element:
      result.extend(_list_tensor_names_in_binding(element))
    return result
  elif which_binding == 'sequence':
    which_sequence = binding.sequence.WhichOneof('binding')
    if which_sequence != 'variant_tensor_name':
      raise TypeError(
          'Expected a variant tensor in sequence binding, found {}.'.format(
              which_sequence))
    return [binding.sequence.variant_tensor_name]
  else:
    raise TypeError('Unexpected type of binding {}.'.format(which_binding))
