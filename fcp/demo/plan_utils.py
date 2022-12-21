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
"""Utilities for working with Plan protos and TensorFlow.

See the field comments in plan.proto for more information about each operation
and when it should be run.
"""

import functools
import tempfile

import tensorflow as tf

from google.protobuf import message
from fcp.protos import plan_pb2


class Session:
  """A session for performing L2 Plan operations.

  This class only supports loading a single intermediate update.
  """

  def __init__(self, plan: plan_pb2.Plan, checkpoint: bytes):
    if len(plan.phase) != 1:
      raise ValueError('plan must contain exactly 1 phase.')
    if not plan.phase[0].HasField('server_phase'):
      raise ValueError('plan.phase[0] is missing server_phase.')

    graph_def = tf.compat.v1.GraphDef()
    try:
      plan.server_graph_bytes.Unpack(graph_def)
    except message.DecodeError as e:
      raise ValueError('Unable to parse server graph.') from e

    graph = tf.Graph()
    with graph.as_default():
      tf.import_graph_def(graph_def, name='')
    self._session = tf.compat.v1.Session(graph=graph)
    self._plan = plan
    self._restore_state(plan.server_savepoint, checkpoint)
    self._maybe_run(plan.phase[0].server_phase.phase_init_op)
    self._client_checkpoint = self._save_state(
        plan.phase[0].server_phase.write_client_init)

  def __enter__(self) -> 'Session':
    self._session.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, tb) -> None:
    self._session.__exit__(exc_type, exc_value, tb)

  def close(self) -> None:
    """Closes the session, releasing resources."""
    self._session.close()

  def _maybe_run(self, op: str) -> None:
    """Runs an operation if it's non-empty."""
    if op:
      self._session.run(op)

  def _restore_state(self, checkpoint_op: plan_pb2.CheckpointOp,
                     checkpoint: bytes) -> None:
    """Restores state from a TensorFlow checkpoint."""
    self._maybe_run(checkpoint_op.before_restore_op)
    if checkpoint_op.HasField('saver_def'):
      with tempfile.NamedTemporaryFile('wb') as tmpfile:
        tmpfile.write(checkpoint)
        tmpfile.flush()
        self._session.run(
            checkpoint_op.saver_def.restore_op_name,
            {checkpoint_op.saver_def.filename_tensor_name: tmpfile.name})
    self._maybe_run(checkpoint_op.after_restore_op)

  def _save_state(self, checkpoint_op: plan_pb2.CheckpointOp) -> bytes:
    """Saves state to a TensorFlow checkpoint."""
    self._maybe_run(checkpoint_op.before_save_op)
    result = b''
    if checkpoint_op.HasField('saver_def'):
      with tempfile.NamedTemporaryFile() as tmpfile:
        self._session.run(
            checkpoint_op.saver_def.save_tensor_name,
            {checkpoint_op.saver_def.filename_tensor_name: tmpfile.name})
        # TensorFlow overwrites (via move) the output file, so the data can't be
        # read from the filehandle. Deletion still works properly, though.
        with open(tmpfile.name, 'rb') as f:
          result = f.read()
    self._maybe_run(checkpoint_op.after_save_op)
    return result

  @functools.cached_property
  def client_plan(self) -> bytes:
    """The serialized ClientOnlyPlan corresponding to the Plan proto."""
    client_only_plan = plan_pb2.ClientOnlyPlan(
        phase=self._plan.phase[0].client_phase,
        graph=self._plan.client_graph_bytes.value,
        tflite_graph=self._plan.client_tflite_graph_bytes)
    if self._plan.HasField('tensorflow_config_proto'):
      client_only_plan.tensorflow_config_proto.CopyFrom(
          self._plan.tensorflow_config_proto)
    return client_only_plan.SerializeToString()

  @property
  def client_checkpoint(self) -> bytes:
    """The initial checkpoint for use by clients."""
    return self._client_checkpoint

  def finalize(self, update: bytes) -> bytes:
    """Loads an intermediate update and return the final result."""
    self._restore_state(
        self._plan.phase[0].server_phase.read_intermediate_update, update)
    self._maybe_run(self._plan.phase[0].server_phase
                    .intermediate_aggregate_into_accumulators_op)
    # write_accumulators and metrics are not needed by Federated Program
    # computations because all results are included in the server savepoint.
    self._maybe_run(
        self._plan.phase[0].server_phase.apply_aggregrated_updates_op)
    return self._save_state(self._plan.server_savepoint)
