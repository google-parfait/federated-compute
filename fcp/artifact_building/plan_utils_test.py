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
"""Test class for plan_utils."""

import os

import tensorflow as tf

from fcp.artifact_building import checkpoint_utils
from fcp.artifact_building import plan_utils
from fcp.artifact_building import test_utils
from fcp.protos import plan_pb2


class PlanUtilsTest(tf.test.TestCase):

  def test_write_checkpoint(self):
    checkpoint_op = plan_pb2.CheckpointOp()
    graph = tf.Graph()
    with graph.as_default():
      v = tf.compat.v1.get_variable('v', initializer=tf.constant(1))
      saver = checkpoint_utils.create_deterministic_saver([v])
      test_utils.set_checkpoint_op(checkpoint_op, saver)
      init_op = v.assign(tf.constant(2))
      change_op = v.assign(tf.constant(3))

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      temp_file = self.create_tempfile().full_path
      plan_utils.write_checkpoint(sess, checkpoint_op, temp_file)
      # Change the variable in this session.
      sess.run(change_op)

    with tf.compat.v1.Session(graph=graph) as sess:
      saver.restore(sess, temp_file)
      # Should not see update to 3.
      self.assertEqual(2, sess.run(v))

  def test_write_checkpoint_not_checkpoint_op(self):
    with self.assertRaises(ValueError):
      plan_utils.write_checkpoint(None, 'not_checkpoint_op', None)

  def test_write_checkpoint_skips_when_no_saver_def(self):
    checkpoint_op = plan_pb2.CheckpointOp()
    with tf.compat.v1.Session() as sess:
      temp_file = self.create_tempfile().full_path
      # Close deletes the file, we just want a good name.
      os.remove(temp_file)
      plan_utils.write_checkpoint(sess, checkpoint_op, temp_file)
      self.assertFalse(os.path.isfile(temp_file))

  def test_read_checkpoint(self):
    checkpoint_op = plan_pb2.CheckpointOp()
    graph = tf.Graph()
    with graph.as_default():
      v = tf.compat.v1.get_variable('v', initializer=tf.constant(1))
      saver = checkpoint_utils.create_deterministic_saver([v])
      test_utils.set_checkpoint_op(checkpoint_op, saver)
      init_op = v.assign(tf.constant(2))
      change_op = v.assign(tf.constant(3))

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      temp_file = self.create_tempfile().full_path
      saver.save(sess, temp_file)
      sess.run(change_op)

      plan_utils.read_checkpoint(sess, checkpoint_op, temp_file)
      # Should not see update to 3.
      self.assertEqual(2, sess.run(v))


if __name__ == '__main__':
  tf.test.main()
