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


class ExternalServiceHandle(abc.ABC):
  """An interface for helping a program interact with external services.

  Trusted federated programs will expect an instance of this interface to be
  provided as an argument to the "trusted_program" function. This interface
  supports releasing data as a first-class concept, however it could also be
  used to set up channels to other untrusted or trusted processes using the
  provided outgoing server address.
  """

  def __init__(self, outgoing_server_address):
    """Initializes an `ExternalServiceHandle`.

    Args:
      outgoing_server_address: The address at which external services can be
        reached.
    """
    self._outgoing_server_address = outgoing_server_address

  @property
  def outgoing_server_address(self):
    """Returns the outgoing server address."""
    return self._outgoing_server_address

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
