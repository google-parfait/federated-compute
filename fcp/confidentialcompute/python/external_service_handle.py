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
from collections.abc import Callable

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
      client_ids: list[str],
      client_data_directory: str,
      config_id_to_filename: dict[str, str],
      resolve_uri_to_tensor_fn: Callable[[str, str], tensor_pb2.TensorProto],
  ):
    """Initializes an `ExternalServiceHandle`.

    Args:
      outgoing_server_address: The address at which external services can be
        reached.
      client_ids: A list of strings representing the clients from this data
        source.
      client_data_directory: The directory containing the client data.
      config_id_to_filename: A dictionary mapping config ids to the paths of the
        files containing information for that id.
      resolve_uri_to_tensor_fn: A function that resolves pointers to data.
        Expects a uri and key and returns an AggCore tensor proto.
    """
    self._outgoing_server_address = outgoing_server_address
    self._client_ids = client_ids if client_ids is not None else []
    self._client_data_directory = client_data_directory
    self._config_id_to_filename = (
        config_id_to_filename if config_id_to_filename is not None else {}
    )
    self._resolve_uri_to_tensor_fn = resolve_uri_to_tensor_fn

  @property
  def outgoing_server_address(self) -> str:
    """Returns the outgoing server address."""
    return self._outgoing_server_address

  @property
  def client_ids(self) -> list[str]:
    """Returns the list of client ids."""
    return self._client_ids

  @property
  def client_data_directory(self) -> str:
    """Returns the directory containing the client data."""
    return self._client_data_directory

  def get_filename_for_config_id(self, config_id: str) -> str:
    """Returns the filename for the given config id."""
    if config_id not in self._config_id_to_filename:
      raise ValueError(
          f'Config id {config_id} not found in config_id_to_filename. Available'
          f' config ids are {list(self._config_id_to_filename.keys())}'
      )
    return self._config_id_to_filename[config_id]

  def resolve_uri_to_tensor(self, uri: str, key: str) -> tensor_pb2.TensorProto:
    """Resolves a pointer to data."""
    return self._resolve_uri_to_tensor_fn(uri, key)

  @abc.abstractmethod
  def release_unencrypted(
      self,
      value: bytes,
      key: bytes,
  ) -> None:
    """Releases an unencrypted value to the external service."""
    raise NotImplementedError

  @abc.abstractmethod
  def release_encrypted(
      self,
      value: bytes,
      key: bytes,
  ) -> None:
    """Releases an encrypted value to the external service."""
    raise NotImplementedError
