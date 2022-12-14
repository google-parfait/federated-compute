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
"""Utilities related to plan protos.

This file holds methods that are applicable to generic Plan entities, when
there is no concept of a server and clients.
"""

from fcp.protos import plan_pb2


# TODO(team): Remove in favor of save_from_checkpoint_op.
def write_checkpoint(sess, checkpoint_op, checkpoint_filename):
  """Writes from a CheckpointOp, without executing before/after restore ops."""
  if not isinstance(checkpoint_op, plan_pb2.CheckpointOp):
    raise ValueError('A CheckpointOp is required.')
  if (checkpoint_op and checkpoint_op.saver_def and
      checkpoint_op.saver_def.save_tensor_name):
    sess.run(
        checkpoint_op.saver_def.save_tensor_name,
        {checkpoint_op.saver_def.filename_tensor_name: checkpoint_filename})


# TODO(team): Remove in favor of restore_from_checkpoint_op.
def read_checkpoint(sess, checkpoint_op, checkpoint_filename):
  """Reads from a CheckpointOp, without executing before/after restore ops."""
  if not isinstance(checkpoint_op, plan_pb2.CheckpointOp):
    raise ValueError('A CheckpointOp is required.')
  if (checkpoint_op and checkpoint_op.saver_def and
      checkpoint_op.saver_def.restore_op_name):
    sess.run(
        checkpoint_op.saver_def.restore_op_name,
        {checkpoint_op.saver_def.filename_tensor_name: checkpoint_filename})
