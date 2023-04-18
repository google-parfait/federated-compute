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
"""Tests for plan_utils."""

import functools
import tempfile
from typing import Any, Optional

from absl.testing import absltest
import tensorflow as tf

from fcp.demo import plan_utils
from fcp.demo import test_utils
from fcp.protos import plan_pb2
from fcp.tensorflow import serve_slices

DEFAULT_INITIAL_CHECKPOINT = b'initial'
CHECKPOINT_TENSOR_NAME = 'checkpoint'
INTERMEDIATE_TENSOR_NAME = 'intermediate_value'
FINAL_TENSOR_NAME = 'final_value'
NUM_SLICES = 3


def create_plan(log_file: Optional[str] = None) -> plan_pb2.Plan:
  """Creates a test Plan that sums inputs."""

  def log_op(name: str) -> tf.Operation:
    """Helper function to log op invocations to a file."""
    if log_file:
      return tf.print(name, output_stream=f'file://{log_file}')
    return tf.raw_ops.NoOp()

  def create_checkpoint_op(
      name: str,
      filename_op: Any,
      save_op: Any = None,
      restore_op: Any = None,
      session_token_tensor_name: Optional[str] = None,
  ) -> plan_pb2.CheckpointOp:
    before_restore = log_op(f'{name}/before_restore')
    after_restore = log_op(f'{name}/after_restore')
    before_save = log_op(f'{name}/before_save')
    after_save = log_op(f'{name}/after_save')
    with tf.control_dependencies(
        [save_op if save_op is not None else tf.raw_ops.NoOp()]):
      save_op = log_op(f'{name}/save')
    with tf.control_dependencies(
        [restore_op if restore_op is not None else tf.raw_ops.NoOp()]):
      restore_op = log_op(f'{name}/restore')
    return plan_pb2.CheckpointOp(
        saver_def=tf.compat.v1.train.SaverDef(
            filename_tensor_name=filename_op.name,
            restore_op_name=restore_op.name,
            save_tensor_name=save_op.name,
            version=tf.compat.v1.train.SaverDef.V1,
        ),
        before_restore_op=before_restore.name,
        after_restore_op=after_restore.name,
        before_save_op=before_save.name,
        after_save_op=after_save.name,
        session_token_tensor_name=session_token_tensor_name,
    )

  with tf.compat.v1.Graph().as_default() as client_graph:
    tf.constant(0)

  with tf.compat.v1.Graph().as_default() as server_graph:
    # Initialization:
    last_client_update = tf.Variable(0, dtype=tf.int32)
    intermediate_acc = tf.Variable(0, dtype=tf.int32)
    last_intermediate_update = tf.Variable(0, dtype=tf.int32)
    final_acc = tf.Variable(0, dtype=tf.int32)
    with tf.control_dependencies([
        last_client_update.initializer, intermediate_acc.initializer,
        last_intermediate_update.initializer, final_acc.initializer
    ]):
      phase_init_op = log_op('phase_init')

    # Ops for Federated Select:
    select_fn_initialize_op = log_op('slices/initialize')
    select_fn_server_vals = [
        tf.constant(1234),
        tf.constant('asdf'),
        tf.constant([1, 2, 3]),
    ]
    select_fn_server_val_inputs = [
        tf.compat.v1.placeholder(v.dtype) for v in select_fn_server_vals
    ]
    select_fn_key_input = tf.compat.v1.placeholder(tf.int32, shape=())
    select_fn_filename_input = tf.compat.v1.placeholder(tf.string, shape=())
    assertions = [
        tf.debugging.assert_equal(placeholder, constant)
        for placeholder, constant in zip(
            select_fn_server_val_inputs, select_fn_server_vals
        )
    ]
    with tf.control_dependencies([log_op('slices/save_slice')] + assertions):
      select_fn_save_op = tf.io.write_file(
          select_fn_filename_input, tf.strings.as_string(select_fn_key_input)
      )
    # Some tests disable passing the callback token; set `served_at_id` to '-'
    # in that case.
    callback_token = tf.compat.v1.placeholder_with_default('', shape=())
    served_at_id = tf.cond(
        tf.equal(callback_token, ''),
        lambda: '-',
        functools.partial(
            serve_slices.serve_slices,
            callback_token=callback_token,
            server_val=select_fn_server_vals,
            max_key=NUM_SLICES - 1,
            select_fn_initialize_op=select_fn_initialize_op.name,
            select_fn_server_val_input_tensor_names=[
                v.name for v in select_fn_server_val_inputs
            ],
            select_fn_key_input_tensor_name=select_fn_key_input.name,
            select_fn_filename_input_tensor_name=select_fn_filename_input.name,
            select_fn_target_tensor_name=select_fn_save_op.name,
        ),
    )

    # Ops for L2 Aggregation:
    client_checkpoint_data = tf.Variable(
        DEFAULT_INITIAL_CHECKPOINT, dtype=tf.string)

    write_client_init_filename = tf.compat.v1.placeholder(tf.string, shape=())
    client_checkpoint_data_value = tf.cond(
        tf.compat.v1.is_variable_initialized(client_checkpoint_data),
        client_checkpoint_data.read_value,
        lambda: client_checkpoint_data.initial_value,
    )
    write_client_init_op = create_checkpoint_op(
        'write_client_init',
        write_client_init_filename,
        save_op=tf.io.write_file(
            write_client_init_filename,
            tf.strings.join(
                [client_checkpoint_data_value, served_at_id], separator=' '
            ),
        ),
        session_token_tensor_name=callback_token.name,
    )

    read_intermediate_update_filename = tf.compat.v1.placeholder(
        tf.string, shape=())
    read_intermediate_update_op = create_checkpoint_op(
        'read_intermediate_update',
        read_intermediate_update_filename,
        restore_op=last_intermediate_update.assign(
            tf.raw_ops.Restore(
                file_pattern=read_intermediate_update_filename,
                tensor_name=INTERMEDIATE_TENSOR_NAME,
                dt=tf.int32)))

    with tf.control_dependencies([log_op('apply_aggregated_updates')]):
      apply_aggregated_updates_op = final_acc.assign_add(
          last_intermediate_update)

    server_savepoint_filename = tf.compat.v1.placeholder(tf.string, shape=())
    server_savepoint_op = create_checkpoint_op(
        'server_savepoint',
        server_savepoint_filename,
        save_op=tf.raw_ops.Save(
            filename=server_savepoint_filename,
            tensor_names=[FINAL_TENSOR_NAME],
            data=[final_acc]),
        restore_op=client_checkpoint_data.assign(
            tf.raw_ops.Restore(
                file_pattern=server_savepoint_filename,
                tensor_name=CHECKPOINT_TENSOR_NAME,
                dt=tf.string)))

  config_proto = tf.compat.v1.ConfigProto(operation_timeout_in_ms=1234)

  plan = plan_pb2.Plan(
      phase=[
          plan_pb2.Plan.Phase(
              client_phase=plan_pb2.ClientPhase(name='ClientPhase'),
              server_phase=plan_pb2.ServerPhase(
                  phase_init_op=phase_init_op.name,
                  write_client_init=write_client_init_op,
                  read_intermediate_update=read_intermediate_update_op,
                  apply_aggregrated_updates_op=(
                      apply_aggregated_updates_op.name)))
      ],
      server_savepoint=server_savepoint_op,
      client_tflite_graph_bytes=b'tflite-graph',
      version=1)
  plan.client_graph_bytes.Pack(client_graph.as_graph_def())
  plan.server_graph_bytes.Pack(server_graph.as_graph_def())
  plan.tensorflow_config_proto.Pack(config_proto)
  return plan


def create_checkpoint(tensor_name=b'test'):
  """Creates a test initial checkpoint."""
  return test_utils.create_checkpoint({CHECKPOINT_TENSOR_NAME: tensor_name})


class PlanUtilsTest(absltest.TestCase):

  def test_session_enter_exit(self):
    self.assertIsNone(tf.compat.v1.get_default_session())
    with plan_utils.Session(create_plan(), create_checkpoint()):
      self.assertIsNotNone(tf.compat.v1.get_default_session())
    self.assertIsNone(tf.compat.v1.get_default_session())

  def test_session_without_phase(self):
    plan = create_plan()
    plan.ClearField('phase')
    with self.assertRaises(ValueError):
      plan_utils.Session(plan, create_checkpoint())

  def test_session_without_server_phase(self):
    plan = create_plan()
    plan.phase[0].ClearField('server_phase')
    with self.assertRaises(ValueError):
      plan_utils.Session(plan, create_checkpoint())

  def test_session_with_multiple_phases(self):
    plan = create_plan()
    plan.phase.append(plan.phase[0])
    with self.assertRaises(ValueError):
      plan_utils.Session(plan, create_checkpoint())

  def test_session_client_plan(self):
    plan = create_plan()
    with plan_utils.Session(plan, create_checkpoint()) as session:
      self.assertEqual(
          plan_pb2.ClientOnlyPlan.FromString(session.client_plan),
          plan_pb2.ClientOnlyPlan(
              phase=plan.phase[0].client_phase,
              graph=plan.client_graph_bytes.value,
              tflite_graph=plan.client_tflite_graph_bytes,
              tensorflow_config_proto=plan.tensorflow_config_proto))

  def test_session_client_plan_without_tensorflow_config(self):
    plan = create_plan()
    plan.ClearField('tensorflow_config_proto')
    with plan_utils.Session(plan, create_checkpoint()) as session:
      self.assertEqual(
          plan_pb2.ClientOnlyPlan.FromString(session.client_plan),
          plan_pb2.ClientOnlyPlan(
              phase=plan.phase[0].client_phase,
              graph=plan.client_graph_bytes.value,
              tflite_graph=plan.client_tflite_graph_bytes))

  def test_session_client_plan_without_tflite_graph(self):
    plan = create_plan()
    plan.ClearField('client_tflite_graph_bytes')
    with plan_utils.Session(plan, create_checkpoint()) as session:
      self.assertEqual(
          plan_pb2.ClientOnlyPlan.FromString(session.client_plan),
          plan_pb2.ClientOnlyPlan(
              phase=plan.phase[0].client_phase,
              graph=plan.client_graph_bytes.value,
              tensorflow_config_proto=plan.tensorflow_config_proto))

  def test_session_client_checkpoint(self):
    expected = b'test-client-checkpoint'
    with plan_utils.Session(
        create_plan(),
        test_utils.create_checkpoint({CHECKPOINT_TENSOR_NAME: expected
                                     })) as session:
      self.assertEqual(
          session.client_checkpoint,
          expected + b' ' + next(iter(session.slices)).encode(),
      )

  def test_session_client_checkpoint_without_server_savepoint(self):
    plan = create_plan()
    # If server_savepoint isn't set, the checkpoint shouldn't be loaded.
    plan.ClearField('server_savepoint')
    with plan_utils.Session(plan, create_checkpoint()) as session:
      self.assertStartsWith(
          session.client_checkpoint, DEFAULT_INITIAL_CHECKPOINT + b' '
      )

  def test_session_finalize(self):
    with tempfile.NamedTemporaryFile('r') as tmpfile:
      with plan_utils.Session(create_plan(tmpfile.name),
                              create_checkpoint()) as session:
        checkpoint = session.finalize(
            test_utils.create_checkpoint({INTERMEDIATE_TENSOR_NAME: 3}))
      self.assertSequenceEqual(
          tmpfile.read().splitlines(),
          [
              'server_savepoint/before_restore',
              'server_savepoint/restore',
              'server_savepoint/after_restore',
              'phase_init',
              'write_client_init/before_save',
              'write_client_init/save',
              'write_client_init/after_save',
          ]
          + ['slices/initialize', 'slices/save_slice'] * NUM_SLICES
          + [
              'read_intermediate_update/before_restore',
              'read_intermediate_update/restore',
              'read_intermediate_update/after_restore',
              'apply_aggregated_updates',
              'server_savepoint/before_save',
              'server_savepoint/save',
              'server_savepoint/after_save',
          ],
      )

    result = test_utils.read_tensor_from_checkpoint(checkpoint,
                                                    FINAL_TENSOR_NAME, tf.int32)
    # The value should be propagated from the intermediate aggregate.
    self.assertEqual(result, 3)

  def test_session_with_tensorflow_error(self):
    plan = create_plan()
    plan.phase[0].server_phase.phase_init_op = 'does-not-exist'
    with self.assertRaises(ValueError):
      plan_utils.Session(plan, create_checkpoint())

  def test_session_slices(self):
    with plan_utils.Session(create_plan(), create_checkpoint()) as session:
      # The served_at_id should match the value in the client checkpoint.
      served_at_id = session.client_checkpoint.split(b' ')[1].decode()
      self.assertSameElements(session.slices.keys(), [served_at_id])
      self.assertListEqual(
          session.slices[served_at_id],
          [str(i).encode() for i in range(NUM_SLICES)],
      )

  def test_session_without_slices(self):
    plan = create_plan()
    plan.phase[0].server_phase.write_client_init.ClearField(
        'session_token_tensor_name'
    )
    with plan_utils.Session(plan, create_checkpoint()) as session:
      self.assertEmpty(session.slices)


if __name__ == '__main__':
  absltest.main()
