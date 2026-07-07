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

from collections.abc import Sequence
import concurrent.futures
import hashlib
import math
import os
import time
from typing import Optional

from absl import logging
import federated_language
from federated_language.proto import array_pb2
from federated_language.proto import computation_pb2
from federated_language.proto import data_type_pb2
import jax
import numpy as np

from google.protobuf import any_pb2
from fcp.confidentialcompute.python import constants
from fcp.confidentialcompute.python import external_service_handle
from fcp.protos.confidentialcompute import file_info_pb2
from fcp.protos.confidentialcompute import min_sep_data_source_pb2
from tensorflow_federated.cc.core.impl.aggregation.core import tensor_pb2


def _compute_blob_ids_hash(blob_ids: Sequence[bytes]) -> bytes:
  """Computes a SHA-256 hash of a list of blob ids."""
  return hashlib.sha256(b','.join(blob_ids)).digest()


def assign_blob_ids_to_rounds(
    key: jax.Array,
    blob_ids: Sequence[bytes],
    min_sep: int,
) -> list[list[bytes]]:
  """Assigns blob ids to `min_sep` rounds randomly.

  We are currently optimizing for privacy without amplification, so we first
  shuffle the blob ids and then assign them to groups of equal size. If we
  later decide to optimize for privacy with amplification (which will
  require further obfuscation of which clients are participating in a round)
  then we should assign each client independently to a random group such
  that the groups may not have equal size.

  A blob id will be eligible for participation in round `i` if it is
  assigned to the `i % min_sep`th entry in the returned list.

  In order for this function to be deterministic, the same key and the same
  blob ids (in the same order) must be provided.

  Args:
    key: The key to use for shuffling.
    blob_ids: The list of blob ids to assign to rounds.
    min_sep: The minimum difference between the round indices of two consecutive
      participations for the same client. Must be a positive integer.

  Returns:
    A list of lists of blob ids, where the outer list has length `min_sep`
    and the inner lists contain the blob ids assigned to each round.
  """
  shuffled_blob_indices = jax.random.permutation(key, len(blob_ids))
  blob_id_round_assignments = [[] for _ in range(min_sep)]
  for i, blob_index in enumerate(shuffled_blob_indices):
    blob_id_round_assignments[i % min_sep].append(blob_ids[blob_index])
  return blob_id_round_assignments


class MinSepDataSourceIterator(
    federated_language.program.FederatedDataSourceIterator
):
  """A `FederatedDataSourceIterator` providing min-sep round participation.

  Clients, which are represented by blob ids, are eligible for participation
  in rounds that are exactly `min_sep` rounds apart. The round indices for which
  a client is eligible are computed randomly at initialization time, and when
  `select` is called, the requested number of clients are randomly selected from
  those eligible for the current round.
  """

  def __init__(
      self,
      min_sep: int,
      external_handle: external_service_handle.ExternalServiceHandle,
      computation_type: computation_pb2.Type,
      key_name: str = constants.OUTPUT_TENSOR_NAME,
      use_data_pointers: bool = False,
      key: Optional[jax.Array] = None,
  ):
    """Returns an initialized `tff.program.MinSepDataSourceIterator`.

    Args:
      min_sep: The minimum difference between the round indices of two
        consecutive participations for the same client. Must be a positive
        integer.
      external_handle: The external service handle that facilitates interactions
        between this program and untrusted space.
      computation_type: The type of data represented by this data source.
      key_name: The name of the key to use when creating pointers to the
        underlying data. This should match the tensor name used when creating
        the federated checkpoint for the uploaded client data.
      use_data_pointers: Whether to return TFF value data pointers or return
        resolved TFF value literals.
      key: The key to use for shuffling and selection. If None, a new key will
        be created based on the current time.

    Raises:
      ValueError: If `blob_ids` is empty or if `min_sep` is not a positive
        integer.
    """
    if min_sep <= 0:
      raise ValueError(
          'Expected `min_sep` to be a positive integer, found `min_sep` of '
          f'{min_sep}.'
      )

    if not external_handle.blob_ids:
      raise ValueError('Expected `blob_ids` to not be empty.')

    self._min_sep = min_sep
    self._external_handle = external_handle
    self._computation_type = computation_type
    self._key_name = key_name
    self._use_data_pointers = use_data_pointers
    self._round_index = 0
    # If no key is provided, create a new key based on the current time.
    if key is None:
      key = jax.random.key(int(time.time() * 1e6))
    self._shuffling_prng_key, self._selection_prng_key = jax.random.split(key)

    # Assign blob ids to rounds and compute hash of blob ids list for
    # validation upon recovery.
    self._blob_id_round_assignments = assign_blob_ids_to_rounds(
        self._shuffling_prng_key,
        external_handle.blob_ids,
        self._min_sep,
    )
    self._blob_ids_hash = _compute_blob_ids_hash(external_handle.blob_ids)

  @classmethod
  def restore(
      cls,
      buffer: bytes,
      external_handle: external_service_handle.ExternalServiceHandle,
  ) -> 'MinSepDataSourceIterator':
    """Deserializes the object from bytes."""
    state = min_sep_data_source_pb2.MinSepDataSourceState()
    state.ParseFromString(buffer)

    # Validate blob ids hash.
    current_hash = _compute_blob_ids_hash(external_handle.blob_ids)
    if current_hash != state.blob_ids_hash:
      raise ValueError(
          'Blob ids in input provider do not match recovered state.'
      )

    # Initialize with saved configuration.
    instance = cls(
        min_sep=state.min_sep,
        external_handle=external_handle,
        computation_type=computation_pb2.Type.FromString(
            state.computation_type_bytes
        ),
        key_name=state.key_name,
        use_data_pointers=state.use_data_pointers,
    )

    # Restore state.
    instance._round_index = state.round_index
    shuffling_arr = np.frombuffer(state.shuffling_prng_key, dtype=np.uint32)
    selection_arr = np.frombuffer(state.selection_prng_key, dtype=np.uint32)
    instance._shuffling_prng_key = jax.random.wrap_key_data(shuffling_arr)  # pyrefly: ignore[bad-argument-type]
    instance._selection_prng_key = jax.random.wrap_key_data(selection_arr)  # pyrefly: ignore[bad-argument-type]

    # Re-compute round assignments using the restored shuffling key.
    instance._blob_id_round_assignments = assign_blob_ids_to_rounds(
        instance._shuffling_prng_key,
        external_handle.blob_ids,
        instance._min_sep,
    )

    return instance

  def save(self) -> bytes:
    """Serializes the object to bytes."""
    state = min_sep_data_source_pb2.MinSepDataSourceState()
    state.min_sep = self._min_sep
    state.computation_type_bytes = self._computation_type.SerializeToString()
    state.key_name = self._key_name
    state.use_data_pointers = self._use_data_pointers
    state.round_index = self._round_index
    state.shuffling_prng_key = jax.random.key_data(
        self._shuffling_prng_key
    ).tobytes()
    state.selection_prng_key = jax.random.key_data(
        self._selection_prng_key
    ).tobytes()
    state.blob_ids_hash = self._blob_ids_hash
    return state.SerializeToString()

  @classmethod
  def from_bytes(cls, buffer: bytes) -> 'MinSepDataSourceIterator':
    """Not supported. Use `restore` instead."""
    raise NotImplementedError('Use restore(buffer, external_handle) instead.')

  def to_bytes(self) -> bytes:
    """Not supported. Use `save` instead."""
    raise NotImplementedError('Use save() instead.')

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
        len(blob_ids) // min_sep or this value+1), select will fail.

    Returns:
      A list of `federated_language` `Computation` protos that each contain a
      `Data` proto representing the selected client data via a
      `fcp.confidentialcompute.FileInfo` proto.

    Raises:
      ValueError: If `k` is not a positive integer or there are fewer than `k`
      eligible clients.
      RuntimeError: If too many blob ids fail to resolve to tensors.
    """
    # Obtain the eligible blob ids for the current round.
    eligible_ids = self._blob_id_round_assignments[
        self._round_index % self._min_sep
    ]

    if k is None or k <= 0:
      raise ValueError(
          f'Expected `k` to be a positive integer, found `k` of {k}.'
      )

    if len(eligible_ids) < k:
      raise ValueError(
          'Requested more than the number of eligible clients. '
          f'Requested {k}, eligible {len(eligible_ids)}.'
      )

    # Split the selection key to use for this round.
    self._selection_prng_key, subkey = jax.random.split(
        self._selection_prng_key
    )

    # Sample blob ids without replacement using JAX.
    selected_indices = jax.random.choice(
        subkey,
        len(eligible_ids),
        shape=(k,),
        replace=False,
    )
    selected_ids = [eligible_ids[int(i)] for i in selected_indices]
    self._round_index += 1

    selected_values = []
    if self._use_data_pointers:
      for blob_id in selected_ids:
        file_info = file_info_pb2.FileInfo(
            blob_id=blob_id,
            key=self._key_name,
        )
        any_proto = any_pb2.Any()
        any_proto.Pack(file_info)
        selected_values.append(
            computation_pb2.Computation(
                type=self._computation_type,
                data=computation_pb2.Data(content=any_proto),
            )
        )
      return selected_values

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count() * 4  # pyrefly: ignore[unsupported-operation]
    ) as executor:
      future_to_blob_id = {}
      for blob_id in selected_ids:
        future = executor.submit(
            self._external_handle.resolve_blob_id_to_tensor,
            blob_id,
            self._key_name,
        )
        future_to_blob_id[future] = blob_id

      # Wait for all futures to complete. The order in which the results are
      # added does not matter.
      failure_count = 0
      for future in concurrent.futures.as_completed(future_to_blob_id):
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
          logging.warning(
              'Skipping blob id: %s due to resolve error: %s',
              future_to_blob_id[future],
              e,
          )
          failure_count += 1
    if failure_count > math.ceil(0.1 * len(selected_ids)):
      raise RuntimeError(
          'More than 10% of blob ids failed to resolve. '
          f'There were {failure_count} resolve errors across '
          f'{len(selected_ids)} inputs.'
      )
    else:
      logging.info(
          'Resolved %d out of %d requested blob ids.',
          len(selected_ids) - failure_count,
          len(selected_ids),
      )
    return selected_values


class MinSepDataSource(federated_language.program.FederatedDataSource):
  """A `FederatedDataSource` providing min-sep round participation behavior."""

  def __init__(
      self,
      min_sep: int,
      external_handle: external_service_handle.ExternalServiceHandle,
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
      external_handle: The external service handle that facilitates interactions
        between this program and untrusted space.
      computation_type: The type of data represented by this data source.
      key_name: The name of the key to use when creating pointers to the
        underlying data. This should match the tensor name used when creating
        the federated checkpoint for the uploaded client data.
      use_data_pointers: Whether to return TFF value data pointers or return
        resolved TFF value literals.

    Raises:
      ValueError: If `blob_ids` is empty or if `min_sep` is not a positive
        integer.
    """
    if min_sep <= 0:
      raise ValueError(
          'Expected `min_sep` to be a positive integer, found `min_sep` of '
          f'{min_sep}.'
      )

    if not external_handle.blob_ids:
      raise ValueError('Expected `external_handle.blob_ids` to not be empty.')

    self._min_sep = min_sep
    self._external_handle = external_handle
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
        self._external_handle,
        self._computation_type,
        self._key_name,
        self._use_data_pointers,
    )
