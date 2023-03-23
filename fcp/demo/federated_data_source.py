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
"""TFF FederatedDataSource for the demo Federated Computation platform."""

import dataclasses
import functools
import re
from typing import Optional, Union

import tensorflow as tf
import tensorflow_federated as tff

from fcp.protos import plan_pb2
from fcp.protos.federatedcompute import eligibility_eval_tasks_pb2

POPULATION_NAME_REGEX = re.compile(r'\w+(/\w+)*')

_NestedExampleSelector = Union[plan_pb2.ExampleSelector,
                               dict[str, '_NestedExampleSelector']]
_TaskAssignmentMode = (
    eligibility_eval_tasks_pb2.PopulationEligibilitySpec.TaskInfo.TaskAssignmentMode
)


@dataclasses.dataclass
class DataSelectionConfig:
  population_name: str
  example_selector: _NestedExampleSelector
  task_assignment_mode: _TaskAssignmentMode
  num_clients: int


class FederatedDataSource(tff.program.FederatedDataSource):
  """A FederatedDataSource for use with the demo platform.

  A FederatedDataSource represents a population of client devices and the set of
  on-device data over which computations should be invoked.
  """

  _FEDERATED_TYPE = tff.FederatedType(tff.SequenceType(tf.string), tff.CLIENTS)

  def __init__(
      self,
      population_name: str,
      example_selector: _NestedExampleSelector,
      task_assignment_mode: _TaskAssignmentMode = (
          _TaskAssignmentMode.TASK_ASSIGNMENT_MODE_SINGLE
      ),
  ):
    """Constructs a new FederatedDataSource object.

    Args:
      population_name: The name of the population to execute computations on.
      example_selector: A `plan_pb2.ExampleSelector` or a structure of
        ExampleSelectors indicating CLIENTS-placed data to execute over.
      task_assignment_mode: The TaskAssignmentMode to use for this computation.
    """
    if not POPULATION_NAME_REGEX.fullmatch(population_name):
      raise ValueError(
          f'population_name must match "{POPULATION_NAME_REGEX.pattern}".')
    self._population_name = population_name
    self._example_selector = example_selector
    self._task_assignment_mode = task_assignment_mode

  @property
  def population_name(self) -> str:
    """The name of the population from which examples will be retrieved."""
    return self._population_name

  @property
  def example_selector(self) -> _NestedExampleSelector:
    """The NestedExampleSelector used to obtain the examples."""
    return self._example_selector

  @property
  def task_assignment_mode(self) -> _TaskAssignmentMode:
    """The TaskAssignmentMode to use for this computation."""
    return self._task_assignment_mode

  @functools.cached_property
  def federated_type(self) -> tff.FederatedType:

    def get_struct_type(value):
      if isinstance(value, dict):
        return tff.StructType([
            (k, get_struct_type(v)) for k, v in value.items()
        ])
      # ExternalDataset always returns a sequence of tf.strings, which should be
      # serialized `tf.train.Example` protos.
      return tff.SequenceType(tf.string)

    return tff.FederatedType(
        get_struct_type(self._example_selector), tff.CLIENTS)

  @functools.cached_property
  def capabilities(self) -> list[tff.program.Capability]:
    return [tff.program.Capability.SUPPORTS_REUSE]

  def iterator(self) -> tff.program.FederatedDataSourceIterator:
    return _FederatedDataSourceIterator(self)


class _FederatedDataSourceIterator(tff.program.FederatedDataSourceIterator):
  """A `FederatedDataSourceIterator` for use with the demo platform."""

  def __init__(self, data_source: FederatedDataSource):
    self._data_source = data_source

  @classmethod
  def from_bytes(cls, data: bytes) -> '_FederatedDataSourceIterator':
    """Deserializes the object from bytes."""
    raise NotImplementedError

  def to_bytes(self) -> bytes:
    """Serializes the object to bytes."""
    raise NotImplementedError

  @property
  def federated_type(self):
    return self._data_source.federated_type

  def select(self, num_clients: Optional[int] = None) -> DataSelectionConfig:
    if num_clients is None or num_clients <= 0:
      raise ValueError('num_clients must be positive.')
    return DataSelectionConfig(
        self._data_source.population_name,
        self._data_source.example_selector,
        self._data_source.task_assignment_mode,
        num_clients,
    )
