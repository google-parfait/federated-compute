# Copyright 2025, The TensorFlow Federated Authors.
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
"""Utilities for representing data sources with min-sep round participation."""

import concurrent.futures
import os
import random
from typing import Optional

from absl import logging
import federated_language
from federated_language.proto import array_pb2
from federated_language.proto import computation_pb2
from federated_language.proto import data_type_pb2

from google.protobuf import any_pb2
from fcp.confidentialcompute.python import constants
from fcp.confidentialcompute.python import program_input_provider
from fcp.protos.confidentialcompute import file_info_pb2
from tensorflow_federated.cc.core.impl.aggregation.core import tensor_pb2


_RESOLVE_URI_ERROR_MESSAGE_SUBSTRING = 'Failed to fetch Tensor'


class MinSepDataSourceIterator(
    federated_language.program.FederatedDataSourceIterator
):
  """A `FederatedDataSourceIterator` providing min-sep round participation.

  Clients, which are represented by client ids, are eligible for participation
  in rounds that are exactly `min_sep` rounds apart. The round indices for which
  a client is eligible are computed randomly at initialization time, and when
  `select` is called, the requested number of clients are randomly selected from
  those eligible for the current round.
  """

  def __init__(
      self,
      min_sep: int,
      input_provider: program_input_provider.ProgramInputProvider,
      computation_type: computation_pb2.Type,
      key_name: str = constants.OUTPUT_TENSOR_NAME,
      use_data_pointers: bool = False,
  ):
    """Returns an initialized `tff.program.MinSepDataSourceIterator`.

    Args:
      min_sep: The number of rounds that must elapse between successive
        participations for the same client. Must be a positive integer.
      input_provider: The program input provider that provides the client ids
        and client data directory.
      computation_type: The type of data represented by this data source.
      key_name: The name of the key to use when creating pointers to the
        underlying data. This should match the tensor name used when creating
        the federated checkpoint for the uploaded client data.
      use_data_pointers: Whether to return TFF value data pointers or return
        resolved TFF value literals.

    Raises:
      ValueError: If `client_ids` is empty or if `min_sep` is not a positive
        integer.
    """
    if min_sep <= 0:
      raise ValueError(
          'Expected `min_sep` to be a positive integer, found `min_sep` of '
          f'{min_sep}.'
      )

    if not input_provider.client_ids:
      raise ValueError('Expected `client_ids` to not be empty.')

    # The client ids are randomly assigned to `min_sep` rounds.
    #
    # We are currently optimizing for privacy without amplification, so we first
    # shuffle the client ids and then assign them to groups of equal size. If we
    # later decide to optimize for privacy with amplification (which will
    # require further obfuscation of which clients are participating in a round)
    # then we should assign each client independently to a random group such
    # that the groups may not have equal size.
    #
    # A client id will be eligible for participation in round `i` if it is
    # assigned to the `i % min_sep`th entry in `_client_id_round_assignments`.
    self._client_id_round_assignments = [[] for _ in range(min_sep)]
    random.shuffle(input_provider.client_ids)
    for i, client_id in enumerate(input_provider.client_ids):
      self._client_id_round_assignments[i % min_sep].append(client_id)

    self._min_sep = min_sep
    self._input_provider = input_provider
    self._computation_type = computation_type
    self._key_name = key_name
    self._use_data_pointers = use_data_pointers
    self._round_index = 0

  @classmethod
  def from_bytes(cls, buffer: bytes) -> 'MinSepDataSourceIterator':
    """Deserializes the object from bytes."""
    # TODO: b/420969188 - Add deserialization support if fault tolerance is
    # needed.
    raise NotImplementedError()

  def to_bytes(self) -> bytes:
    """Serializes the object to bytes."""
    # TODO: b/420969188 - Add serialization support if fault tolerance is
    # needed.
    raise NotImplementedError()

  @property
  def federated_type(self) -> federated_language.FederatedType:
    """The type of the data returned by calling `select`."""
    raise ValueError(
        'The `federated_type` property is not supported for this data source'
        " because we don't have a good type representation for a Data proto."
    )

  def select(self, k: Optional[int] = None) -> object:
    """Returns a new selection of client data protos for the present round.

    Args:
      k: A number of elements to select. Must be a positive integer. If greater
        than the number of eligible clients for this round (which will be either
        len(client_ids) // min_sep or this value+1), select will fail.

    Returns:
      A list of `federated_language` `Computation` protos that each contain a
      `Data` proto representing the selected client data via a
      `fcp.confidentialcompute.FileInfo` proto.

    Raises:
      ValueError: If `k` is not a positive integer or there are fewer than `k`
      eligible clients.
    """
    # Obtain the eligible client ids for the current round.
    eligible_ids = self._client_id_round_assignments[
        self._round_index % self._min_sep
    ]

    if k is None or k <= 0:
      raise ValueError(
          f'Expected `k` to be a positive integer, found `k` of {k}.'
      )

    if len(eligible_ids) < k:
      raise ValueError('Requested more than the number of eligible clients.')

    selected_ids = random.sample(eligible_ids, min(len(eligible_ids), k))
    self._round_index += 1

    selected_values = []
    if self._use_data_pointers:
      for client_id in selected_ids:
        any_proto = any_pb2.Any()
        any_proto.Pack(
            file_info_pb2.FileInfo(
                uri=os.path.join(
                    self._input_provider.client_data_directory, client_id
                ),
                key=self._key_name,
            )
        )
        selected_values.append(
            computation_pb2.Computation(
                type=self._computation_type,
                data=computation_pb2.Data(content=any_proto),
            )
        )
      return selected_values

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count() * 4
    ) as executor:
      future_to_uri = {}
      for client_id in selected_ids:
        uri = os.path.join(
            self._input_provider.client_data_directory, client_id
        )
        future = executor.submit(
            self._input_provider.resolve_uri_to_tensor,
            uri,
            self._key_name,
        )
        future_to_uri[future] = uri

      # Wait for all futures to complete. The order in which the results are
      # added does not matter.
      failure_count = 0
      for future in concurrent.futures.as_completed(future_to_uri):
        try:
          tensor = future.result()
          selected_values.append(
              array_pb2.Array(
                  dtype=data_type_pb2.DataType.Value(
                      tensor_pb2.DataType.Name(tensor.dtype)
                  ),
                  shape=array_pb2.ArrayShape(
                      dim=tensor.shape.dim_sizes, unknown_rank=False
                  ),
                  content=tensor.content,
              )
          )
        except RuntimeError as e:
          if _RESOLVE_URI_ERROR_MESSAGE_SUBSTRING in str(e):
            logging.warning(
                'Skipping URI: %s due to resolve error: %s',
                future_to_uri[future],
                e,
            )
            failure_count += 1
            continue
          else:
            raise
    if failure_count > 0:
      logging.warning(
          'Skipped %d inputs due to resolve errors in total of %d inputs.',
          failure_count,
          len(selected_ids),
      )
    return selected_values


class MinSepDataSource(federated_language.program.FederatedDataSource):
  """A `FederatedDataSource` providing min-sep round participation behavior."""

  def __init__(
      self,
      min_sep: int,
      input_provider: program_input_provider.ProgramInputProvider,
      computation_type: computation_pb2.Type = computation_pb2.Type(
          federated=computation_pb2.FederatedType(
              placement=computation_pb2.PlacementSpec(
                  value=computation_pb2.Placement(
                      uri=federated_language.CLIENTS.uri
                  )
              ),
              all_equal=False,
              member=computation_pb2.Type(
                  tensor=computation_pb2.TensorType(
                      dtype=data_type_pb2.DataType.DT_STRING
                  )
              ),
          )
      ),
      key_name: str = constants.OUTPUT_TENSOR_NAME,
      use_data_pointers: bool = False,
  ):
    """Returns an initialized `tff.program.MinSepDataSource`.

    Args:
      min_sep: The number of rounds that must elapse between successive
        participations for the same client. Must be a positive integer.
      input_provider: The program input provider that provides the client ids
        and client data directory.
      computation_type: The type of data represented by this data source.
      key_name: The name of the key to use when creating pointers to the
        underlying data. This should match the tensor name used when creating
        the federated checkpoint for the uploaded client data.
      use_data_pointers: Whether to return TFF value data pointers or return
        resolved TFF value literals.

    Raises:
      ValueError: If `client_ids` is empty or if `min_sep` is not a positive
        integer.
    """
    if min_sep <= 0:
      raise ValueError(
          'Expected `min_sep` to be a positive integer, found `min_sep` of '
          f'{min_sep}.'
      )

    if not input_provider.client_ids:
      raise ValueError('Expected `input_provider.client_ids` to not be empty.')

    self._min_sep = min_sep
    self._input_provider = input_provider
    self._computation_type = computation_type
    self._key_name = key_name
    self._use_data_pointers = use_data_pointers

  @property
  def federated_type(self) -> federated_language.FederatedType:
    raise ValueError(
        'The `federated_type` property is not supported for this data source'
        " because we don't have a good type representation for a Data proto."
    )

  def iterator(self) -> MinSepDataSourceIterator:
    return MinSepDataSourceIterator(
        self._min_sep,
        self._input_provider,
        self._computation_type,
        self._key_name,
        self._use_data_pointers,
    )
