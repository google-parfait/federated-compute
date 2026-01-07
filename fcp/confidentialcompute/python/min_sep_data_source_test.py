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

import os
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from federated_language.proto import array_pb2
from federated_language.proto import computation_pb2
from federated_language.proto import data_type_pb2

from fcp.confidentialcompute.python import min_sep_data_source
from fcp.confidentialcompute.python import program_input_provider
from fcp.protos.confidentialcompute import file_info_pb2
from tensorflow_federated.cc.core.impl.aggregation.core import tensor_pb2


_MIN_SEP = 10
_CLIENT_IDS = ['a', 'b', 'c']
_TEST_CLIENT_DATA_DIRECTORY = 'test_dir'
_KEY_NAME = 'key_name'
_COMPUTATION_TYPE = computation_pb2.Type(
    tensor=computation_pb2.TensorType(dtype=data_type_pb2.DataType.DT_INT32)
)


class MinSepDataSourceIteratorTest(parameterized.TestCase):

  def test_init_raises_value_error_with_client_ids_empty(self):
    client_ids = []

    input_provider = program_input_provider.ProgramInputProvider(
        client_ids,
        _TEST_CLIENT_DATA_DIRECTORY,
        {},
        mock.Mock(),
    )

    with self.assertRaisesRegex(
        ValueError, 'Expected `client_ids` to not be empty.'
    ):
      min_sep_data_source.MinSepDataSourceIterator(
          _MIN_SEP,
          input_provider,
          _COMPUTATION_TYPE,
          _KEY_NAME,
      )

  def test_init_raises_value_error_with_invalid_min_sep(self):
    min_sep = 0

    input_provider = program_input_provider.ProgramInputProvider(
        _CLIENT_IDS,
        _TEST_CLIENT_DATA_DIRECTORY,
        {},
        mock.Mock(),
    )

    with self.assertRaisesRegex(
        ValueError,
        'Expected `min_sep` to be a positive integer, found `min_sep` of 0.',
    ):
      min_sep_data_source.MinSepDataSourceIterator(
          min_sep,
          input_provider,
          _COMPUTATION_TYPE,
          _KEY_NAME,
      )

  @parameterized.named_parameters(
      (
          'none',
          None,
          'Expected `k` to be a positive integer, found `k` of None.',
      ),
      (
          'negative',
          -1,
          'Expected `k` to be a positive integer, found `k` of -1.',
      ),
      (
          'too_small',
          20,
          'Requested more than the number of eligible clients.',
      ),
  )
  def test_select_raises_value_error_with_k(self, k, expected_error_message):
    num_clients = 100
    client_ids = [str(i) for i in range(num_clients)]

    input_provider = program_input_provider.ProgramInputProvider(
        client_ids,
        _TEST_CLIENT_DATA_DIRECTORY,
        {},
        mock.Mock(),
    )

    iterator = min_sep_data_source.MinSepDataSourceIterator(
        _MIN_SEP,
        input_provider,
        _COMPUTATION_TYPE,
        _KEY_NAME,
    )

    with self.assertRaisesRegex(ValueError, expected_error_message):
      iterator.select(k)

  @parameterized.named_parameters(
      ('with_data_pointers', True),
      ('with_resolved_values', False),
  )
  def test_select_returns_expected_data(self, use_data_pointers):
    num_clients = 1000
    client_ids = [str(i) for i in range(num_clients)]

    def mock_resolve_uri_to_tensor_fn(
        uri: str, key: str
    ) -> tensor_pb2.TensorProto:
      """A mock function for resolving a URI to a tensor."""
      del key
      return tensor_pb2.TensorProto(
          dtype=tensor_pb2.DataType.DT_STRING,
          # The content of the tensor is the uri itself.
          content=uri.encode(),
          shape=tensor_pb2.TensorShapeProto(dim_sizes=[1]),
      )

    mock_resolve_fn = mock.Mock()
    mock_resolve_fn.side_effect = mock_resolve_uri_to_tensor_fn
    input_provider = program_input_provider.ProgramInputProvider(
        client_ids,
        _TEST_CLIENT_DATA_DIRECTORY,
        {},
        mock_resolve_fn,
    )

    # Create an iterator that will have 100 eligible clients per round.
    # (1000 clients that are eligible to participate every 10th round)
    iterator = min_sep_data_source.MinSepDataSourceIterator(
        _MIN_SEP,
        input_provider,
        _COMPUTATION_TYPE,
        _KEY_NAME,
        use_data_pointers=use_data_pointers,
    )

    # Call `select` 1000 times with `k=10`. Each client should be selected for
    # an average of 10 rounds (although each client will have been eligible for
    # 100 rounds).
    # (1000 rounds * 10 clients per round / 1000 clients = 10)
    client_id_to_round_indices = {}
    num_rounds = 1000
    k = 10
    for round_index in range(num_rounds):
      data_for_round = iterator.select(k)

      # Check that the number of client ids returned is equal to `k`.
      self.assertLen(data_for_round, k)

      # Track which rounds clients are chosen to participate in.
      for value in data_for_round:
        if use_data_pointers:
          self.assertIsInstance(value, computation_pb2.Computation)
          self.assertEqual(value.type, _COMPUTATION_TYPE)
          self.assertEqual(value.WhichOneof('computation'), 'data')
          unpacked_content = file_info_pb2.FileInfo()
          value.data.content.Unpack(unpacked_content)
          self.assertEqual(unpacked_content.key, _KEY_NAME)
          uri = unpacked_content.uri
        else:
          self.assertIsInstance(value, array_pb2.Array)
          self.assertEqual(value.dtype, data_type_pb2.DataType.DT_STRING)
          uri = value.content.decode()

        self.assertEqual(os.path.dirname(uri), _TEST_CLIENT_DATA_DIRECTORY)
        client_id = os.path.basename(uri)
        client_id_to_round_indices.setdefault(client_id, []).append(round_index)

    # Verify that the rounds that clients are chosen to participate in are
    # multiples of `_MIN_SEP` apart.
    for client_id, round_indices in client_id_to_round_indices.items():
      self.assertIn(client_id, client_ids)

      if len(round_indices) < 2:
        continue

      # Calculate the expected modulus from the first round index, and check
      # that the modulus is the same across all rounds for which this client was
      # chosen to participate.
      expected_modulus = round_indices[0] % _MIN_SEP
      for round_index in round_indices[1:]:
        self.assertEqual(round_index % _MIN_SEP, expected_modulus)

  def test_select_parallelizes_resolve_uri_to_tensor(self):
    num_clients = 200
    client_ids = [str(i) for i in range(num_clients)]
    k = 10
    sleep_seconds = 5

    def mock_resolve_uri_to_tensor_fn(uri: str, key: str):
      del key
      # Delay the resolution of each URI to simulate some work being done.
      time.sleep(sleep_seconds)
      return tensor_pb2.TensorProto(
          dtype=tensor_pb2.DataType.DT_STRING,
          content=uri.encode(),
          shape=tensor_pb2.TensorShapeProto(dim_sizes=[1]),
      )

    mock_resolve_fn = mock.Mock()
    mock_resolve_fn.side_effect = mock_resolve_uri_to_tensor_fn
    input_provider = program_input_provider.ProgramInputProvider(
        client_ids,
        _TEST_CLIENT_DATA_DIRECTORY,
        {},
        mock_resolve_fn,
    )

    iterator = min_sep_data_source.MinSepDataSourceIterator(
        _MIN_SEP,
        input_provider,
        _COMPUTATION_TYPE,
        _KEY_NAME,
        use_data_pointers=False,
    )

    # Call select and verify that resolve_uri_to_tensor was called k times and
    # that the time taken is within 1 second of the expected time if the
    # parallelization is working correctly.
    start_time = time.time()
    data_for_round = iterator.select(k)
    end_time = time.time()
    self.assertLess(end_time - start_time, sleep_seconds + 1)
    self.assertLen(data_for_round, k)
    self.assertEqual(mock_resolve_fn.call_count, k)

  def test_select_skips_bad_decrypt_errors(self):
    client_ids = ['0', '1']
    k = 2

    def mock_resolve_uri_to_tensor_fn(
        uri: str, key: str
    ) -> tensor_pb2.TensorProto:
      """A mock function for resolving a URI to a tensor."""
      del key
      client_id = os.path.basename(uri)
      if client_id == '0':
        raise RuntimeError(
            'Failed to fetch Tensor: Failed to unwrap symmetric key'
        )
      return tensor_pb2.TensorProto(
          dtype=tensor_pb2.DataType.DT_STRING,
          content=uri.encode(),
          shape=tensor_pb2.TensorShapeProto(dim_sizes=[1]),
      )

    mock_resolve_fn = mock.Mock()
    mock_resolve_fn.side_effect = mock_resolve_uri_to_tensor_fn
    input_provider = program_input_provider.ProgramInputProvider(
        client_ids,
        _TEST_CLIENT_DATA_DIRECTORY,
        {},
        mock_resolve_fn,
    )

    iterator = min_sep_data_source.MinSepDataSourceIterator(
        min_sep=1,
        input_provider=input_provider,
        computation_type=_COMPUTATION_TYPE,
        key_name=_KEY_NAME,
        use_data_pointers=False,
    )

    with self.assertLogs(level='WARNING') as logs:
      data_for_round = iterator.select(k)
      self.assertLen(data_for_round, 1)
      self.assertEqual(
          os.path.basename(data_for_round[0].content.decode()), '1'
      )
      self.assertLen(logs.output, 2)
      self.assertRegex(
          logs.output[0],
          'Skipping URI: test_dir/0 due to resolve error: Failed to fetch'
          ' Tensor: Failed to unwrap symmetric key',
      )
      self.assertRegex(
          logs.output[1],
          'Skipped 1 inputs due to resolve errors in total of 2 inputs.',
      )


class MinSepDataSourceTest(absltest.TestCase):

  def test_init_succeeds(self):
    input_provider = program_input_provider.ProgramInputProvider(
        _CLIENT_IDS, _TEST_CLIENT_DATA_DIRECTORY, {}, mock.Mock()
    )
    min_sep = min_sep_data_source.MinSepDataSource(
        _MIN_SEP,
        input_provider,
        _COMPUTATION_TYPE,
        _KEY_NAME,
    )
    self.assertIsInstance(
        min_sep.iterator(), min_sep_data_source.MinSepDataSourceIterator
    )

  def test_init_raises_value_error_with_client_ids_empty(self):
    client_ids = []
    input_provider = program_input_provider.ProgramInputProvider(
        client_ids, _TEST_CLIENT_DATA_DIRECTORY, {}, mock.Mock()
    )

    with self.assertRaisesRegex(
        ValueError, 'Expected `input_provider.client_ids` to not be empty.'
    ):
      min_sep_data_source.MinSepDataSource(
          _MIN_SEP,
          input_provider,
          _COMPUTATION_TYPE,
          _KEY_NAME,
      )

  def test_init_raises_value_error_with_invalid_min_sep(self):
    min_sep = 0
    input_provider = program_input_provider.ProgramInputProvider(
        _CLIENT_IDS, _TEST_CLIENT_DATA_DIRECTORY, {}, mock.Mock()
    )

    with self.assertRaisesRegex(
        ValueError,
        'Expected `min_sep` to be a positive integer, found `min_sep` of 0.',
    ):
      min_sep_data_source.MinSepDataSource(
          min_sep,
          input_provider,
          _COMPUTATION_TYPE,
          _KEY_NAME,
      )


if __name__ == '__main__':
  absltest.main()
