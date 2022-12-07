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
"""A class to specify on-device dataset inputs."""

from collections.abc import Callable
from typing import Any, Optional, Union

import tensorflow as tf
import tensorflow_federated as tff

from fcp.artifact_building import type_checks
from fcp.protos import plan_pb2


class DataSpec:
  """A specification of a single dataset input."""

  __slots__ = ('_example_selector_proto', '_preprocessing_fn',
               '_preprocessing_comp', '_fingerprint')

  def __init__(self,
               example_selector_proto: plan_pb2.ExampleSelector,
               preprocessing_fn: Optional[Callable[[tf.data.Dataset],
                                                   tf.data.Dataset]] = None):
    """Constructs a specification of a dataset input.

    Args:
      example_selector_proto: An instance of `plan_pb2.ExampleSelector` proto.
      preprocessing_fn: A callable that accepts as an argument the raw input
        `tf.data.Dataset` with `string`-serialized items, performs any desired
        preprocessing such as deserialization, filtering, batching, and
        formatting, and returns the transformed `tf.data.Dataset` as a result.
        If preprocessing_fn is set to None, it is expected that any client data
        preprocessing has already been incorporated into the `tff.Computation`
        that this `DataSpec` is associated with.

    Raises:
      TypeError: If the types of the arguments are invalid.
    """
    type_checks.check_type(
        example_selector_proto,
        plan_pb2.ExampleSelector,
        name='example_selector_proto')
    if preprocessing_fn is not None:
      type_checks.check_callable(preprocessing_fn, name='preprocessing_fn')
    self._example_selector_proto = example_selector_proto
    self._preprocessing_fn = preprocessing_fn
    # Set once self.preprocessing_comp is accessed, as we can't call
    # tff.computation in __init__.
    self._preprocessing_comp = None

  @property
  def example_selector_proto(self) -> plan_pb2.ExampleSelector:
    return self._example_selector_proto

  @property
  def preprocessing_fn(
      self) -> Optional[Callable[[tf.data.Dataset], tf.data.Dataset]]:
    return self._preprocessing_fn

  @property
  def preprocessing_comp(self) -> tff.Computation:
    """Returns the preprocessing computation for the input dataset."""
    if self._preprocessing_comp is None:
      if self.preprocessing_fn is None:
        raise ValueError('DataSpec\'s preprocessing_fn is None so a '
                         'preprocessing tff.Computation cannot be generated.')
      self._preprocessing_comp = tff.tf_computation(self.preprocessing_fn,
                                                    tff.SequenceType(tf.string))
    return self._preprocessing_comp

  @property
  def type_signature(self) -> tff.Type:
    """Returns the type signature of the result of the preprocessing_comp.

    Effectively the type or 'spec' of the parsed example from the example store
    pointed at by `example_selector_proto`.
    """
    return self.preprocessing_comp.type_signature.result


def is_data_spec_or_structure(x: Any) -> bool:
  """Returns True iff `x` is either a `DataSpec` or a nested structure of it."""
  if x is None:
    return False
  if isinstance(x, DataSpec):
    return True
  try:
    x = tff.structure.from_container(x)
    return all(
        is_data_spec_or_structure(y) for _, y in tff.structure.to_elements(x))
  except TypeError:
    return False


def check_data_spec_or_structure(x: Any, name: str):
  """Raises error iff `x` is not a `DataSpec` or a nested structure of it."""
  if not is_data_spec_or_structure(x):
    raise TypeError(f'Expected `{name}` to be a `DataSpec` or a nested '
                    f'structure of it, found {str(x)}.')


NestedDataSpec = Union[DataSpec, dict[str, 'NestedDataSpec']]
