# Copyright 2026 Google LLC
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
import jax
import jax.numpy as jnp
import numpy as np

from fcp.confidentialcompute.python import prng_keys


class PrngKeysTest(absltest.TestCase):

  def test_unwrap_and_rewrap_round_trip(self):
    pytree = {'key': jax.random.key(42), 'value': jnp.float32(1.0)}
    spec_tree = jax.eval_shape(lambda: pytree)

    unwrapped = prng_keys.unwrap_prng_keys(pytree)
    self.assertEqual(unwrapped['key'].dtype, jnp.uint32)
    self.assertEqual(unwrapped['value'], 1.0)

    rewrapped = prng_keys.rewrap_prng_keys(unwrapped, spec_tree)
    self.assertEqual(rewrapped['key'].dtype, pytree['key'].dtype)
    np.testing.assert_array_equal(
        jax.random.key_data(rewrapped['key']),
        jax.random.key_data(pytree['key']),
    )
    self.assertEqual(rewrapped['value'], 1.0)


if __name__ == '__main__':
  absltest.main()
