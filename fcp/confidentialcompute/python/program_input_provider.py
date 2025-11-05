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
"""Wrapper for providing inputs to a trusted federated program."""

from collections.abc import Callable

from tensorflow_federated.cc.core.impl.aggregation.core import tensor_pb2


class ProgramInputProvider:
  """A wrapper class for providing inputs to a trusted federated program.

  Trusted federated programs will expect an instance of this class to be
  provided as an argument to the "trusted_program" function. The program can
  then use the provided ProgramInputProvider instance to access the available
  client data and models.
  """

  def __init__(
      self,
      client_ids: list[str],
      client_data_directory: str,
      config_id_to_filename: dict[str, str],
      resolve_uri_to_tensor_fn: Callable[[str, str], tensor_pb2.TensorProto],
  ):
    """Returns an initialized `ProgramInputProvider`.

    Args:
      client_ids: A list of strings representing the clients from this data
        source. Must not be empty.
      client_data_directory: The directory containing the client data.
      config_id_to_filename: A dictionary mapping config ids to the paths of the
        files containing information for that id.
      resolve_uri_to_tensor_fn: A function that resolves pointers to data.
        Expects a uri and key and returns an AggCore tensor proto.
    """
    self._client_ids = client_ids
    self._client_data_directory = client_data_directory
    self._config_id_to_filename = config_id_to_filename
    self._resolve_uri_to_tensor_fn = resolve_uri_to_tensor_fn

  @property
  def client_ids(self):
    """Returns the list of client ids."""
    return self._client_ids

  @property
  def client_data_directory(self):
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
