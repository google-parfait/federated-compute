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
from typing import Any, Optional
import uuid

import tensorflow as tf

from google.protobuf import message
from fcp.protos import plan_pb2
from fcp.tensorflow import serve_slices as serve_slices_registry


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

    serve_slices_calls = []

    def record_serve_slices_call(*args):
      served_at_id = str(uuid.uuid4())
      serve_slices_calls.append((served_at_id, args))
      return served_at_id

    with serve_slices_registry.register_serve_slices_callback(
        record_serve_slices_call
    ) as token:
      self._client_checkpoint = self._save_state(
          plan.phase[0].server_phase.write_client_init, session_token=token
      )
    self._slices = {
        k: self._build_slices(*args) for k, args in serve_slices_calls
    }

  def __enter__(self) -> 'Session':
    self._session.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, tb) -> None:
    self._session.__exit__(exc_type, exc_value, tb)

  def close(self) -> None:
    """Closes the session, releasing resources."""
    self._session.close()

  def _maybe_run(
      self, op: str, feed_dict: Optional[dict[str, Any]] = None
  ) -> None:
    """Runs an operation if it's non-empty."""
    if op:
      self._session.run(op, feed_dict=feed_dict)

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

  def _save_state(
      self,
      checkpoint_op: plan_pb2.CheckpointOp,
      session_token: Optional[bytes] = None,
  ) -> bytes:
    """Saves state to a TensorFlow checkpoint."""
    before_and_after_inputs = {}
    if session_token and checkpoint_op.session_token_tensor_name:
      before_and_after_inputs[checkpoint_op.session_token_tensor_name] = (
          session_token
      )

    self._maybe_run(
        checkpoint_op.before_save_op, feed_dict=before_and_after_inputs
    )
    result = b''
    if checkpoint_op.HasField('saver_def'):
      with tempfile.NamedTemporaryFile() as tmpfile:
        save_tensor_inputs = before_and_after_inputs.copy()
        save_tensor_inputs[checkpoint_op.saver_def.filename_tensor_name] = (
            tmpfile.name
        )
        self._session.run(
            checkpoint_op.saver_def.save_tensor_name,
            feed_dict=save_tensor_inputs,
        )
        # TensorFlow overwrites (via move) the output file, so the data can't be
        # read from the filehandle. Deletion still works properly, though.
        with open(tmpfile.name, 'rb') as f:
          result = f.read()
    self._maybe_run(
        checkpoint_op.after_save_op, feed_dict=before_and_after_inputs
    )
    return result

  def _build_slices(
      self,
      callback_token: bytes,
      server_val: list[Any],
      max_key: int,
      select_fn_initialize_op: str,
      select_fn_server_val_input_tensor_names: list[str],
      select_fn_key_input_tensor_name: str,
      select_fn_filename_input_tensor_name: str,
      select_fn_target_tensor_name: str,
  ):
    """Builds the slices for a ServeSlices call."""
    del callback_token
    slices: list[bytes] = []
    for i in range(0, max_key + 1):
      self._maybe_run(select_fn_initialize_op)
      with tempfile.NamedTemporaryFile() as tmpfile:
        feed_dict = dict(
            zip(select_fn_server_val_input_tensor_names, server_val)
        )
        feed_dict[select_fn_key_input_tensor_name] = i
        feed_dict[select_fn_filename_input_tensor_name] = tmpfile.name
        self._session.run(select_fn_target_tensor_name, feed_dict=feed_dict)
        # TensorFlow overwrites (via move) the output file, so the data can't be
        # read from the filehandle. Deletion still works properly, though.
        with open(tmpfile.name, 'rb') as f:
          slices.append(f.read())
    return slices

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

  @property
  def slices(self) -> dict[str, list[bytes]]:
    """The Federated Select slices, keyed by served_at_id."""
    # Return a copy to prevent mutations.
    return self._slices.copy()
