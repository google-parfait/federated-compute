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
"""Action handlers for file upload and download.

In a production system, download would likely be handled by an external service;
it's important that uploads are not handled separately to help ensure that
unaggregated client data is only held ephemerally.
"""

import contextlib
import http
import threading
from typing import Callable, Iterator, Optional
import uuid

from fcp.demo import http_actions
from fcp.protos.federatedcompute import common_pb2


class Service:
  """Implements a service for uploading and downloading data over HTTP."""

  def __init__(self, forwarding_info: Callable[[], common_pb2.ForwardingInfo]):
    self._forwarding_info = forwarding_info
    self._lock = threading.Lock()
    self._downloads: dict[str, http_actions.HttpResponse] = {}
    self._uploads: dict[str, Optional[bytes]] = {}

  @contextlib.contextmanager
  def register_download(
      self,
      data: bytes,
      content_type: str = 'application/octet-stream') -> Iterator[str]:
    """Registers data for download, returning a context manager with the path.

    The download path is unregistered when the context manager is closed.

    Args:
      data: The bytes to make available.
      content_type: The content type to include in the response.

    Yields:
      The URL from which to download the data.
    """
    name = str(uuid.uuid4())
    with self._lock:
      self._downloads[name] = http_actions.HttpResponse(
          body=data,
          headers={
              'Content-Length': len(data),
              'Content-Type': content_type,
          })
    try:
      yield f'{self._forwarding_info().target_uri_prefix}data/{name}'
    finally:
      with self._lock:
        del self._downloads[name]

  def register_upload(self) -> str:
    """Registers a path for single-use upload, returning the resource name."""
    name = str(uuid.uuid4())
    with self._lock:
      self._uploads[name] = None
    return name

  def finalize_upload(self, name: str) -> Optional[bytes]:
    """Returns the data from an upload, if any."""
    with self._lock:
      return self._uploads.pop(name)

  @http_actions.http_action(method='GET', pattern='/data/{name}')
  def download(self, body: bytes, name: str) -> http_actions.HttpResponse:
    """Handles a download request."""
    del body
    try:
      with self._lock:
        return self._downloads[name]
    except KeyError as e:
      raise http_actions.HttpError(http.HTTPStatus.NOT_FOUND) from e

  @http_actions.http_action(
      method='POST', pattern='/upload/v1/media/{name}?upload_protocol=raw')
  def upload(self, body: bytes, name: str) -> http_actions.HttpResponse:
    with self._lock:
      if name not in self._uploads or self._uploads[name] is not None:
        raise http_actions.HttpError(http.HTTPStatus.UNAUTHORIZED)
      self._uploads[name] = body
    return http_actions.HttpResponse(b'')
