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

from unittest import mock

from absl.testing import absltest

from fcp.confidentialcompute.python import external_service_handle
from tensorflow_federated.cc.core.impl.aggregation.core import tensor_pb2


_CLIENT_IDS = ['a', 'b', 'c']
_TEST_CLIENT_DATA_DIRECTORY = 'test_dir'
_TEST_OUTGOING_SERVER_ADDRESS = 'test_address'


class _TestExternalServiceHandle(external_service_handle.ExternalServiceHandle):
  """Concrete ExternalServiceHandle for testing."""

  def release_unencrypted(self, value: bytes, key: bytes) -> None:
    pass

  def release_encrypted(self, value: bytes, key: bytes) -> None:
    pass


class ExternalServiceHandleTest(absltest.TestCase):

  def test_init_succeeds(self):
    handle = _TestExternalServiceHandle(
        _TEST_OUTGOING_SERVER_ADDRESS,
        _CLIENT_IDS,
        _TEST_CLIENT_DATA_DIRECTORY,
        {},
        mock.Mock(),
    )
    self.assertEqual(
        handle.outgoing_server_address, _TEST_OUTGOING_SERVER_ADDRESS
    )
    self.assertEqual(handle.client_ids, _CLIENT_IDS)
    self.assertEqual(handle.client_data_directory, _TEST_CLIENT_DATA_DIRECTORY)

  def test_init_with_only_outgoing_server_address(self):
    handle = _TestExternalServiceHandle(
        _TEST_OUTGOING_SERVER_ADDRESS, [], '', {}, mock.Mock()
    )
    self.assertEqual(
        handle.outgoing_server_address, _TEST_OUTGOING_SERVER_ADDRESS
    )
    self.assertEqual(handle.client_ids, [])
    self.assertEqual(handle.client_data_directory, '')

  def test_get_filename_for_config_id_succeeds(self):
    config_id_to_filename = {'my_id': 'my_filename'}
    handle = _TestExternalServiceHandle(
        _TEST_OUTGOING_SERVER_ADDRESS,
        _CLIENT_IDS,
        _TEST_CLIENT_DATA_DIRECTORY,
        config_id_to_filename,
        mock.Mock(),
    )
    self.assertEqual(handle.get_filename_for_config_id('my_id'), 'my_filename')

  def test_get_filename_for_config_id_raises_error_if_id_not_found(self):
    config_id_to_filename = {'id_1': 'filename_1', 'id_2': 'filename_2'}
    handle = _TestExternalServiceHandle(
        _TEST_OUTGOING_SERVER_ADDRESS,
        _CLIENT_IDS,
        _TEST_CLIENT_DATA_DIRECTORY,
        config_id_to_filename,
        mock.Mock(),
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Config id my_id not found in config_id_to_filename. Available config'
        " ids are ['id_1', 'id_2']",
    ):
      handle.get_filename_for_config_id('my_id')

  def test_resolve_uri_to_tensor(self):
    mock_resolve_fn = mock.Mock()
    mock_resolve_fn.return_value = tensor_pb2.TensorProto(
        dtype=tensor_pb2.DataType.DT_STRING,
        content=b'test_tensor_content',
    )
    handle = _TestExternalServiceHandle(
        _TEST_OUTGOING_SERVER_ADDRESS,
        _CLIENT_IDS,
        _TEST_CLIENT_DATA_DIRECTORY,
        {},
        mock_resolve_fn,
    )
    tensor = handle.resolve_uri_to_tensor('uri', 'key')
    self.assertEqual(tensor.dtype, tensor_pb2.DataType.DT_STRING)
    self.assertEqual(tensor.content, b'test_tensor_content')
    mock_resolve_fn.assert_called_once_with('uri', 'key')


if __name__ == '__main__':
  absltest.main()
