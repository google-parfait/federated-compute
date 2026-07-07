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

from collections.abc import Sequence
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from federated_language.proto import array_pb2
from federated_language.proto import computation_pb2
from federated_language.proto import data_type_pb2

from fcp.confidentialcompute.python import external_service_handle
from fcp.confidentialcompute.python import min_sep_data_source
from fcp.protos.confidentialcompute import file_info_pb2
from tensorflow_federated.cc.core.impl.aggregation.core import tensor_pb2


_MIN_SEP = 10
_BLOB_IDS = (b'a', b'b', b'c')
_KEY_NAME = 'key_name'
_COMPUTATION_TYPE = computation_pb2.Type(
    tensor=computation_pb2.TensorType(dtype=data_type_pb2.DataType.DT_INT32)
)


def _mock_resolve_blob_id_to_tensor(
    blob_id: bytes, key: str
) -> tensor_pb2.TensorProto:
  del blob_id, key  # Unused
  return tensor_pb2.TensorProto()


def _mock_release_unencrypted(value: bytes, key: str) -> None:
  del value, key  # Unused


def _mock_save_recovery_info(
    recovery_info: bytes,
    recovery_key: str,
    value_key_pairs: Sequence[tuple[bytes, str]],
) -> None:
  del recovery_info, recovery_key, value_key_pairs  # Unused


def _mock_restore_recovery_info(key: str) -> bytes | None:
  del key  # Unused
  return b''


def _create_external_handle(
    blob_ids=None,
    resolve_fn=None,
):
  """Creates an ExternalServiceHandle with default mock functions."""
  if blob_ids is None:
    blob_ids = _BLOB_IDS
  return external_service_handle.ExternalServiceHandle(
      outgoing_server_address='',
      blob_ids=blob_ids,
      config_id_to_filename={},
      resolve_blob_id_to_tensor_fn=resolve_fn
      or mock.create_autospec(_mock_resolve_blob_id_to_tensor),
      release_unencrypted_fn=mock.create_autospec(_mock_release_unencrypted),
      save_recovery_info_fn=mock.create_autospec(_mock_save_recovery_info),
      restore_recovery_info_fn=mock.create_autospec(
          _mock_restore_recovery_info
      ),
  )


class MinSepDataSourceIteratorTest(parameterized.TestCase):

  def test_init_raises_value_error_with_blob_ids_empty(self):
    blob_ids = []

    external_handle = _create_external_handle(blob_ids=blob_ids)

    with self.assertRaisesRegex(
        ValueError, 'Expected `blob_ids` to not be empty.'
    ):
      min_sep_data_source.MinSepDataSourceIterator(
          _MIN_SEP,
          external_handle,
          _COMPUTATION_TYPE,
          _KEY_NAME,
      )

  def test_init_raises_value_error_with_invalid_min_sep(self):
    min_sep = 0

    external_handle = _create_external_handle()

    with self.assertRaisesRegex(
        ValueError,
        'Expected `min_sep` to be a positive integer, found `min_sep` of 0.',
    ):
      min_sep_data_source.MinSepDataSourceIterator(
          min_sep,
          external_handle,
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
          (
              'Requested more than the number of eligible clients. '
              'Requested 20, eligible 10.'
          ),
      ),
  )
  def test_select_raises_value_error_with_k(self, k, expected_error_message):
    num_clients = 100
    blob_ids = [str(i).encode() for i in range(num_clients)]

    external_handle = _create_external_handle(blob_ids=blob_ids)

    iterator = min_sep_data_source.MinSepDataSourceIterator(
        _MIN_SEP,
        external_handle,
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
    blob_ids = [str(i).encode() for i in range(num_clients)]

    def mock_resolve_blob_id_to_tensor_fn(
        blob_id: bytes, key: str
    ) -> tensor_pb2.TensorProto:
      """A mock function for resolving a blob id to a tensor."""
      del key
      return tensor_pb2.TensorProto(
          dtype=tensor_pb2.DataType.DT_STRING,
          # The content of the tensor is the blob id itself.
          content=blob_id,
          shape=tensor_pb2.TensorShapeProto(dim_sizes=[1]),
      )

    mock_resolve_fn = mock.Mock()
    mock_resolve_fn.side_effect = mock_resolve_blob_id_to_tensor_fn
    external_handle = _create_external_handle(
        blob_ids=blob_ids,
        resolve_fn=mock_resolve_fn,
    )

    # Create an iterator that will have 100 eligible clients per round.
    # (1000 clients that are eligible to participate every 10th round)
    iterator = min_sep_data_source.MinSepDataSourceIterator(
        _MIN_SEP,
        external_handle,
        _COMPUTATION_TYPE,
        _KEY_NAME,
        use_data_pointers=use_data_pointers,
    )

    # Call `select` 1000 times with `k=10`. Each client should be selected for
    # an average of 10 rounds (although each client will have been eligible for
    # 100 rounds).
    # (1000 rounds * 10 clients per round / 1000 clients = 10)
    blob_id_to_round_indices = {}
    num_rounds = 1000
    k = 10
    for round_index in range(num_rounds):
      data_for_round = iterator.select(k)

      # Check that the number of client ids returned is equal to `k`.
      self.assertLen(data_for_round, k)

      # Track which rounds clients are chosen to participate in.
      for value in data_for_round:  # pyrefly: ignore[not-iterable]
        if use_data_pointers:
          self.assertIsInstance(value, computation_pb2.Computation)
          self.assertEqual(value.type, _COMPUTATION_TYPE)
          self.assertEqual(value.WhichOneof('computation'), 'data')
          unpacked_content = file_info_pb2.FileInfo()
          value.data.content.Unpack(unpacked_content)
          self.assertEqual(unpacked_content.key, _KEY_NAME)
          blob_id = unpacked_content.blob_id
        else:
          self.assertIsInstance(value, array_pb2.Array)
          self.assertEqual(value.dtype, data_type_pb2.DataType.DT_STRING)
          blob_id = value.content

        blob_id_to_round_indices.setdefault(blob_id, []).append(round_index)

    # Verify that the rounds that clients are chosen to participate in are
    # multiples of `_MIN_SEP` apart.
    for blob_id, round_indices in blob_id_to_round_indices.items():
      self.assertIn(blob_id, blob_ids)

      if len(round_indices) < 2:
        continue

      # Calculate the expected modulus from the first round index, and check
      # that the modulus is the same across all rounds for which this client was
      # chosen to participate.
      expected_modulus = round_indices[0] % _MIN_SEP
      for round_index in round_indices[1:]:
        self.assertEqual(round_index % _MIN_SEP, expected_modulus)

  def test_select_parallelizes_resolve_blob_id_to_tensor(self):
    num_clients = 200
    blob_ids = [str(i).encode() for i in range(num_clients)]
    k = 10
    sleep_seconds = 5

    def slow_resolve_blob_id_to_tensor_fn(blob_id: bytes, key: str):
      del key
      # Delay the resolution of each blob id to simulate some work being done.
      time.sleep(sleep_seconds)
      return tensor_pb2.TensorProto(
          dtype=tensor_pb2.DataType.DT_STRING,
          content=blob_id,
          shape=tensor_pb2.TensorShapeProto(dim_sizes=[1]),
      )

    mock_resolve_fn = mock.Mock()
    mock_resolve_fn.side_effect = slow_resolve_blob_id_to_tensor_fn
    external_handle = _create_external_handle(
        blob_ids=blob_ids,
        resolve_fn=mock_resolve_fn,
    )

    iterator = min_sep_data_source.MinSepDataSourceIterator(
        _MIN_SEP,
        external_handle,
        _COMPUTATION_TYPE,
        _KEY_NAME,
        use_data_pointers=False,
    )

    # Call select and verify that resolve_blob_id_to_tensor was called k times
    # and that the time taken is close to the expected time if the
    # parallelization is working correctly.
    start_time = time.time()
    data_for_round = iterator.select(k)
    end_time = time.time()
    # Without parallelization, the test would be expected to take at least
    # k * sleep_seconds = 50 seconds. We give a buffer of 10 seconds to account
    # for jax overhead.
    buffer = 10
    self.assertLess(end_time - start_time, sleep_seconds + buffer)
    self.assertLen(data_for_round, k)
    self.assertEqual(mock_resolve_fn.call_count, k)

  def test_select_skips_bad_decrypt_errors_below_threshold(self):
    blob_ids = [b'0', b'1', b'2']
    k = 3

    def failing_resolve_blob_id_to_tensor_fn(
        blob_id: bytes, key: str
    ) -> tensor_pb2.TensorProto:
      """A mock function for resolving a blob id to a tensor."""
      del key
      if blob_id == b'0':
        raise RuntimeError(
            'Failed to fetch Tensor: Failed to unwrap symmetric key'
        )
      return tensor_pb2.TensorProto(
          dtype=tensor_pb2.DataType.DT_STRING,
          content=blob_id,
          shape=tensor_pb2.TensorShapeProto(dim_sizes=[1]),
      )

    external_handle = _create_external_handle(
        blob_ids=blob_ids,
        resolve_fn=failing_resolve_blob_id_to_tensor_fn,
    )

    iterator = min_sep_data_source.MinSepDataSourceIterator(
        min_sep=1,
        external_handle=external_handle,
        computation_type=_COMPUTATION_TYPE,
        key_name=_KEY_NAME,
        use_data_pointers=False,
    )

    with self.assertLogs(level='INFO') as logs:
      data_for_round = iterator.select(k)
      self.assertLen(data_for_round, 2)
      self.assertLen(logs.output, 2)
      self.assertRegex(
          logs.output[0],
          "Skipping blob id: b'0' due to resolve error: Failed to fetch"
          ' Tensor: Failed to unwrap symmetric key',
      )
      self.assertRegex(
          logs.output[1],
          'Resolved 2 out of 3 requested blob ids.',
      )

  def test_select_bad_decrypt_errors_above_threshold(self):
    blob_ids = [b'0', b'1', b'2']
    k = 3

    def failing_resolve_blob_id_to_tensor_fn(
        blob_id: bytes, key: str
    ) -> tensor_pb2.TensorProto:
      """A mock function for resolving a blob id to a tensor."""
      del key
      if blob_id == b'0' or blob_id == b'1':
        raise RuntimeError(
            'Failed to fetch Tensor: Failed to unwrap symmetric key'
        )
      return tensor_pb2.TensorProto(
          dtype=tensor_pb2.DataType.DT_STRING,
          content=blob_id,
          shape=tensor_pb2.TensorShapeProto(dim_sizes=[1]),
      )

    external_handle = _create_external_handle(
        blob_ids=blob_ids,
        resolve_fn=failing_resolve_blob_id_to_tensor_fn,
    )

    iterator = min_sep_data_source.MinSepDataSourceIterator(
        min_sep=1,
        external_handle=external_handle,
        computation_type=_COMPUTATION_TYPE,
        key_name=_KEY_NAME,
        use_data_pointers=False,
    )

    with self.assertLogs(level='WARNING') as logs:
      with self.assertRaisesRegex(
          RuntimeError,
          'More than 10% of blob ids failed to resolve. There were 2 resolve'
          ' errors across 3 inputs.',
      ):
        iterator.select(k)
      self.assertLen(logs.output, 2)
      logs_string = '\n'.join(logs.output)
      self.assertRegex(
          logs_string,
          "Skipping blob id: b'0' due to resolve error: Failed to fetch"
          ' Tensor: Failed to unwrap symmetric key',
      )
      self.assertRegex(
          logs_string,
          "Skipping blob id: b'1' due to resolve error: Failed to fetch"
          ' Tensor: Failed to unwrap symmetric key',
      )

  def test_save_and_restore(self):
    num_clients = 55
    blob_ids = [str(i).encode() for i in range(num_clients)]
    external_handle = _create_external_handle(blob_ids=blob_ids)

    iterator = min_sep_data_source.MinSepDataSourceIterator(
        _MIN_SEP,
        external_handle,
        _COMPUTATION_TYPE,
        _KEY_NAME,
        use_data_pointers=True,
    )
    k = 5
    # Call `select` 3 times to advance the round index and update the PRNG keys.
    for _ in range(3):
      iterator.select(k)

    state = iterator.save()
    restored_iterator = min_sep_data_source.MinSepDataSourceIterator.restore(
        state, external_handle
    )

    # Check that the original and restored iterators produce the same results
    # for the next round.
    self.assertEqual(iterator.select(k), restored_iterator.select(k))

    # Check that if the iterators are advanced different numbers of times, they
    # will produce different results.
    iterator.select(k)
    self.assertNotEqual(iterator.select(k), restored_iterator.select(k))

  def test_save_and_restore_mismatched_blob_ids(self):
    blob_ids = [b'a', b'b', b'c']
    external_handle = _create_external_handle(blob_ids=blob_ids)
    iterator = min_sep_data_source.MinSepDataSourceIterator(
        _MIN_SEP,
        external_handle,
        _COMPUTATION_TYPE,
        _KEY_NAME,
    )
    state = iterator.save()

    mismatched_external_handle = _create_external_handle(
        blob_ids=[b'a', b'b', b'd']
    )
    with self.assertRaisesRegex(
        ValueError, 'Blob ids in input provider do not match recovered state.'
    ):
      min_sep_data_source.MinSepDataSourceIterator.restore(
          state, mismatched_external_handle
      )


class MinSepDataSourceTest(absltest.TestCase):

  def test_init_succeeds(self):
    external_handle = _create_external_handle()
    data_source = min_sep_data_source.MinSepDataSource(
        _MIN_SEP,
        external_handle,
        _COMPUTATION_TYPE,
        _KEY_NAME,
    )
    self.assertIsInstance(
        data_source.iterator(), min_sep_data_source.MinSepDataSourceIterator
    )

  def test_init_raises_value_error_with_blob_ids_empty(self):
    blob_ids = []
    external_handle = _create_external_handle(blob_ids=blob_ids)

    with self.assertRaisesRegex(
        ValueError, 'Expected `external_handle.blob_ids` to not be empty.'
    ):
      min_sep_data_source.MinSepDataSource(
          _MIN_SEP,
          external_handle,
          _COMPUTATION_TYPE,
          _KEY_NAME,
      )

  def test_init_raises_value_error_with_invalid_min_sep(self):
    min_sep = 0
    external_handle = _create_external_handle()

    with self.assertRaisesRegex(
        ValueError,
        'Expected `min_sep` to be a positive integer, found `min_sep` of 0.',
    ):
      min_sep_data_source.MinSepDataSource(
          min_sep,
          external_handle,
          _COMPUTATION_TYPE,
          _KEY_NAME,
      )


if __name__ == '__main__':
  absltest.main()
