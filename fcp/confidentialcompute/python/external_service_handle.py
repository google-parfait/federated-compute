# Copyright 2025 Google LLC
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
"""Interface for trusted federated program interaction with external services."""

import abc
from collections.abc import Callable, Mapping, Sequence

from tensorflow_federated.cc.core.impl.aggregation.core import tensor_pb2


class ExternalServiceHandle(abc.ABC):
  """An abstract class for helping a program interact with external services.

  Trusted federated programs will expect an instance of this class to be
  provided as an argument to the "trusted_program" function. This abstract class
  supports accessing client data and configuration data, releasing data, and
  setting up channels to other untrusted or trusted processes using the provided
  outgoing server address.
  """

  def __init__(
      self,
      outgoing_server_address: str,
      blob_ids: Sequence[bytes],
      config_id_to_filename: Mapping[str, str],
      *,
      resolve_blob_id_to_tensor_fn: Callable[
          [bytes, str], tensor_pb2.TensorProto
      ],
      release_unencrypted_fn: Callable[[bytes, str], None],
      save_recovery_info_fn: Callable[
          [bytes, str, Sequence[tuple[bytes, str]]], None
      ],
      restore_recovery_info_fn: Callable[[str], bytes | None],
  ):
    """Initializes an `ExternalServiceHandle`.

    Args:
      outgoing_server_address: The address at which external services can be
        reached.
      blob_ids: A list of blob ids representing the clients from this data
        source.
      config_id_to_filename: A dictionary mapping config ids to the paths of the
        files containing information for that id.
      resolve_blob_id_to_tensor_fn: A function that resolves pointers to data.
        Expects two args (the blob id and the key) and returns the resolved
        tensor.
      release_unencrypted_fn: A function that releases unencrypted values to the
        external service. Expects two args (the data and the key).
      save_recovery_info_fn: A function that saves recovery information. Expects
        three args (the recovery info, the recovery key, and a list of value and
        key pairs to release as unencrypted data).
      restore_recovery_info_fn: A function that restores recovery information.
        Expects one arg (the key) and returns the recovery info.
    """
    self._outgoing_server_address = outgoing_server_address
    self._blob_ids = blob_ids
    self._config_id_to_filename = (
        config_id_to_filename if config_id_to_filename is not None else {}
    )
    self._resolve_blob_id_to_tensor_fn = resolve_blob_id_to_tensor_fn
    self._release_unencrypted_fn = release_unencrypted_fn
    self._save_recovery_info_fn = save_recovery_info_fn
    self._restore_recovery_info_fn = restore_recovery_info_fn

  @property
  def outgoing_server_address(self) -> str:
    """Returns the outgoing server address."""
    return self._outgoing_server_address

  @property
  def blob_ids(self) -> Sequence[bytes]:
    """Returns the list of blob ids."""
    return self._blob_ids

  def get_filename_for_config_id(self, config_id: str) -> str:
    """Returns the filename for the given config id."""
    if config_id not in self._config_id_to_filename:
      raise ValueError(
          f'Config id {config_id} not found in config_id_to_filename. Available'
          f' config ids are {list(self._config_id_to_filename.keys())}'
      )
    return self._config_id_to_filename[config_id]

  def resolve_blob_id_to_tensor(
      self, blob_id: bytes, key: str
  ) -> tensor_pb2.TensorProto:
    """Resolves a pointer to a tensor.

    The tensor is assumed to be stored at a given key within a FcCheckpoint
    located at a given blob id.

    Args:
      blob_id: The blob id at which the FcCheckpoint is located.
      key: The key at which to find the tensor within the FcCheckpoint.

    Returns:
      The resolved tensor.
    """
    return self._resolve_blob_id_to_tensor_fn(blob_id, key)

  def release_unencrypted(self, value: bytes, key: str) -> None:
    """Releases an unencrypted value.

    Args:
      value: The value to release.
      key: The filename to release the value to.
    """
    self._release_unencrypted_fn(value, key)

  def save_recovery_info(
      self,
      recovery_info: bytes,
      recovery_key: str,
      value_key_pairs: Sequence[tuple[bytes, str]],
  ) -> None:
    """Saves recovery information.

    Args:
      recovery_info: The recovery information to save.
      recovery_key: The filename to save the recovery information to.
      value_key_pairs: A list of value and key pairs to release as unencrypted
        data while saving the recovery information.
    """
    self._save_recovery_info_fn(recovery_info, recovery_key, value_key_pairs)

  def restore_recovery_info(self, key: str) -> bytes | None:
    """Restores recovery information.

    Args:
      key: The filename to restore the recovery information from.

    Returns:
      The restored recovery information, or None if recovery information cannot
      be found for the given key.
    """
    return self._restore_recovery_info_fn(key)
