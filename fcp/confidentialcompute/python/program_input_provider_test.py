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


from absl.testing import absltest
from fcp.confidentialcompute.python import program_input_provider

_CLIENT_IDS = ['a', 'b', 'c']
_TEST_CLIENT_DATA_DIRECTORY = 'test_dir'


class ProgramInputProviderTest(absltest.TestCase):

  def test_init_succeeds(self):
    input_provider = program_input_provider.ProgramInputProvider(
        _CLIENT_IDS,
        _TEST_CLIENT_DATA_DIRECTORY,
        {},
    )
    self.assertEqual(input_provider.client_ids, _CLIENT_IDS)
    self.assertEqual(
        input_provider.client_data_directory, _TEST_CLIENT_DATA_DIRECTORY
    )

  def test_get_model(self):
    input_provider = program_input_provider.ProgramInputProvider(
        _CLIENT_IDS,
        _TEST_CLIENT_DATA_DIRECTORY,
        {},
    )
    with self.assertRaisesRegex(
        NotImplementedError, "get_model isn't available yet"
    ):
      input_provider.get_model('model_id')


if __name__ == '__main__':
  absltest.main()
