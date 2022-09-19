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
"""Tests for media."""

import http
from unittest import mock
import uuid

from absl.testing import absltest

from fcp.demo import http_actions
from fcp.demo import media
from fcp.protos.federatedcompute import common_pb2


class MediaTest(absltest.TestCase):

  @mock.patch.object(uuid, 'uuid4', return_value=uuid.uuid4(), autospec=True)
  def test_download(self, mock_uuid):
    forwarding_info = common_pb2.ForwardingInfo(
        target_uri_prefix='https://media.example/')
    service = media.Service(lambda: forwarding_info)
    data = b'data'
    with service.register_download(data) as url:
      self.assertEqual(url,
                       f'https://media.example/data/{mock_uuid.return_value}')
      self.assertEqual(
          service.download(b'',
                           url.split('/')[-1]),
          http_actions.HttpResponse(
              body=data,
              headers={
                  'Content-Length': len(data),
                  'Content-Type': 'application/octet-stream',
              }))

  @mock.patch.object(uuid, 'uuid4', return_value=uuid.uuid4(), autospec=True)
  def test_download_with_content_type(self, mock_uuid):
    del mock_uuid
    service = media.Service(common_pb2.ForwardingInfo)
    data = b'data'
    with service.register_download(
        data, content_type='application/x-test') as url:
      self.assertEqual(
          service.download(b'',
                           url.split('/')[-1]),
          http_actions.HttpResponse(
              body=data,
              headers={
                  'Content-Length': len(data),
                  'Content-Type': 'application/x-test',
              }))

  @mock.patch.object(
      uuid, 'uuid4', side_effect=[uuid.uuid4(), uuid.uuid4()], autospec=True)
  def test_download_multiple(self, mock_uuid):
    service = media.Service(common_pb2.ForwardingInfo)
    uuids = [uuid.uuid4() for _ in range(2)]
    mock_uuid.side_effect = uuids
    data1 = b'data1'
    data2 = b'data2'
    with service.register_download(data1) as url1, service.register_download(
        data2) as url2:
      self.assertEqual(service.download(b'', url1.split('/')[-1]).body, data1)
      self.assertEqual(service.download(b'', url2.split('/')[-1]).body, data2)

  def test_download_unregistered(self):
    service = media.Service(common_pb2.ForwardingInfo)
    with self.assertRaises(http_actions.HttpError) as cm:
      service.download(b'', 'does-not-exist')
    self.assertEqual(cm.exception.code, http.HTTPStatus.NOT_FOUND)

  @mock.patch.object(uuid, 'uuid4', return_value=uuid.uuid4(), autospec=True)
  def test_download_no_longer_registered(self, mock_uuid):
    service = media.Service(common_pb2.ForwardingInfo)
    data = b'data'
    with service.register_download(data, content_type='application/x-test'):
      pass
    with self.assertRaises(http_actions.HttpError) as cm:
      service.download(b'', str(mock_uuid.return_value))
    self.assertEqual(cm.exception.code, http.HTTPStatus.NOT_FOUND)

  def test_upload(self):
    service = media.Service(common_pb2.ForwardingInfo)
    name = service.register_upload()
    data = b'data'
    self.assertEqual(
        service.upload(data, name), http_actions.HttpResponse(body=b''))
    self.assertEqual(service.finalize_upload(name), data)

  def test_upload_without_data(self):
    service = media.Service(common_pb2.ForwardingInfo)
    name = service.register_upload()
    self.assertIsNone(service.finalize_upload(name))

  def test_upload_multiple_times(self):
    service = media.Service(common_pb2.ForwardingInfo)
    name = service.register_upload()

    data = b'data1'
    self.assertEqual(
        service.upload(data, name), http_actions.HttpResponse(body=b''))

    with self.assertRaises(http_actions.HttpError) as cm:
      service.upload(b'data2', name)
    self.assertEqual(cm.exception.code, http.HTTPStatus.UNAUTHORIZED)

    self.assertEqual(service.finalize_upload(name), data)

  def test_upload_multiple(self):
    service = media.Service(common_pb2.ForwardingInfo)
    name1 = service.register_upload()
    name2 = service.register_upload()

    # Order shouldn't matter.
    service.upload(b'data2', name2)
    service.upload(b'data1', name1)

    self.assertEqual(service.finalize_upload(name1), b'data1')
    self.assertEqual(service.finalize_upload(name2), b'data2')

  def test_upload_unregistered(self):
    service = media.Service(common_pb2.ForwardingInfo)
    with self.assertRaises(http_actions.HttpError) as cm:
      service.upload(b'data', 'does-not-exist')
    self.assertEqual(cm.exception.code, http.HTTPStatus.UNAUTHORIZED)

    with self.assertRaises(KeyError):
      service.finalize_upload('does-not-exist')

  def test_upload_no_longer_registered(self):
    service = media.Service(common_pb2.ForwardingInfo)
    name = service.register_upload()
    self.assertIsNone(service.finalize_upload(name))

    with self.assertRaises(http_actions.HttpError) as cm:
      service.upload(b'data', name)
    self.assertEqual(cm.exception.code, http.HTTPStatus.UNAUTHORIZED)

    with self.assertRaises(KeyError):
      service.finalize_upload(name)


if __name__ == '__main__':
  absltest.main()
