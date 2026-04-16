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

from collections.abc import Mapping, Sequence
from unittest import mock

from absl.testing import absltest

from fcp.confidentialcompute.python import external_service_handle
from tensorflow_federated.cc.core.impl.aggregation.core import tensor_pb2


_CLIENT_IDS = ['a', 'b', 'c']
_TEST_CLIENT_DATA_DIRECTORY = 'test_dir'
_TEST_OUTGOING_SERVER_ADDRESS = 'test_address'


def _mock_resolve_uri_to_tensor(uri: str, key: str) -> tensor_pb2.TensorProto:
  del uri, key  # Unused
  return tensor_pb2.TensorProto()


def _mock_release_unencrypted(value: bytes, key: str) -> None:
  del value, key  # Unused


def _mock_save_recovery_info(
    recovery_info: bytes,
    recovery_key: str,
    value_key_pairs: Sequence[tuple[bytes, str]],
) -> None:
  del recovery_info, recovery_key, value_key_pairs  # Unused


def _mock_restore_recovery_info(key: str) -> bytes:
  del key  # Unused
  return b''


def _create_external_handle(
    outgoing_server_address: str = _TEST_OUTGOING_SERVER_ADDRESS,
    client_ids: Sequence[str] | None = None,
    client_data_directory: str = _TEST_CLIENT_DATA_DIRECTORY,
    config_id_to_filename: Mapping[str, str] | None = None,
    resolve_uri_to_tensor_fn=None,
    release_unencrypted_fn=None,
    save_recovery_info_fn=None,
    restore_recovery_info_fn=None,
) -> external_service_handle.ExternalServiceHandle:
  """Creates an ExternalServiceHandle with default mock functions."""
  return external_service_handle.ExternalServiceHandle(
      outgoing_server_address,
      client_ids if client_ids is not None else _CLIENT_IDS,
      client_data_directory,
      config_id_to_filename if config_id_to_filename is not None else {},
      resolve_uri_to_tensor_fn=resolve_uri_to_tensor_fn
      or mock.create_autospec(_mock_resolve_uri_to_tensor),
      release_unencrypted_fn=release_unencrypted_fn
      or mock.create_autospec(_mock_release_unencrypted),
      save_recovery_info_fn=save_recovery_info_fn
      or mock.create_autospec(_mock_save_recovery_info),
      restore_recovery_info_fn=restore_recovery_info_fn
      or mock.create_autospec(_mock_restore_recovery_info),
  )


class ExternalServiceHandleTest(absltest.TestCase):
  def test_init_succeeds(self):
    external_handle = _create_external_handle()
    self.assertEqual(
        external_handle.outgoing_server_address, _TEST_OUTGOING_SERVER_ADDRESS
    )
    self.assertEqual(external_handle.client_ids, _CLIENT_IDS)
    self.assertEqual(
        external_handle.client_data_directory, _TEST_CLIENT_DATA_DIRECTORY
    )

  def test_init_with_only_outgoing_server_address(self):
    external_handle = _create_external_handle(
        client_ids=[], client_data_directory=''
    )
    self.assertEqual(
        external_handle.outgoing_server_address, _TEST_OUTGOING_SERVER_ADDRESS
    )
    self.assertEqual(external_handle.client_ids, [])
    self.assertEqual(external_handle.client_data_directory, '')

  def test_get_filename_for_config_id(self):
    external_handle = _create_external_handle(
        config_id_to_filename={'id_1': 'filename_1', 'id_2': 'filename_2'}
    )
    self.assertEqual(
        external_handle.get_filename_for_config_id('id_2'), 'filename_2'
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Config id my_id not found in config_id_to_filename. Available config'
        " ids are ['id_1', 'id_2']",
    ):
      external_handle.get_filename_for_config_id('my_id')

  def test_resolve_uri_to_tensor(self):
    external_handle = _create_external_handle()
    external_handle._resolve_uri_to_tensor_fn.return_value = (
        tensor_pb2.TensorProto(
            dtype=tensor_pb2.DataType.DT_STRING,
            content=b'test_tensor_content',
        )
    )
    tensor = external_handle.resolve_uri_to_tensor('uri', 'key')
    self.assertEqual(tensor.dtype, tensor_pb2.DataType.DT_STRING)
    self.assertEqual(tensor.content, b'test_tensor_content')
    external_handle._resolve_uri_to_tensor_fn.assert_called_once_with(
        'uri', 'key'
    )

  def test_release_unencrypted(self):
    external_handle = _create_external_handle()
    external_handle.release_unencrypted(b'test_value', 'test_key')
    external_handle._release_unencrypted_fn.assert_called_once_with(
        b'test_value', 'test_key'
    )

  def test_save_recovery_info(self):
    external_handle = _create_external_handle()
    value_key_pairs = [(b'value1', 'key1'), (b'value2', 'key2')]
    external_handle.save_recovery_info(
        b'recovery_info', 'recovery_key', value_key_pairs
    )
    external_handle._save_recovery_info_fn.assert_called_once_with(
        b'recovery_info', 'recovery_key', value_key_pairs
    )

  def test_restore_recovery_info(self):
    external_handle = _create_external_handle()
    external_handle._restore_recovery_info_fn.return_value = b'restored_info'
    result = external_handle.restore_recovery_info('test_key')
    self.assertEqual(result, b'restored_info')
    external_handle._restore_recovery_info_fn.assert_called_once_with(
        'test_key'
    )


if __name__ == '__main__':
  absltest.main()
