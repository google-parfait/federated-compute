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
"""Tests for the `delete_file` custom op."""

import os
import tensorflow as tf

from fcp.tensorflow import delete_file


class DeleteFileTest(tf.test.TestCase):

  def new_tempfile_path(self):
    """Returns a path that can be used to store a new tempfile."""
    return os.path.join(self.create_tempdir(), 'checkpoint.ckp')

  def test_delete_file_op(self):
    output_file = self.new_tempfile_path()
    expected_content = 'content'
    tf.io.write_file(output_file, expected_content)
    read_content = tf.io.read_file(output_file)
    self.assertEqual(expected_content, read_content)
    delete_file.delete_file(output_file)
    # Delete one more time to make sure no error when the file doesn't exist.
    delete_file.delete_file(output_file)
    check_if_exists = os.path.exists(output_file)
    self.assertEqual(False, check_if_exists)

  def test_exceptions(self):
    with self.subTest(name='non_string_dtype'):
      with self.assertRaises(TypeError):
        delete_file.delete_file(1.0)
    with self.subTest(name='non_scalar'):
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  '.*must be a string scalar.*'):
        checkpoint_path = self.new_tempfile_path()
        delete_file.delete_file([checkpoint_path, checkpoint_path])


if __name__ == '__main__':
  tf.test.main()
