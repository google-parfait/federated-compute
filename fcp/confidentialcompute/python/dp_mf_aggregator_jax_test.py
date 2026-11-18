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
import collections

from absl.testing import absltest
import chex
import federated_language as flang
import federated_language_jax as flang_jax
import jax
import jax.numpy as jnp
from jax_privacy import noise_addition
from jax_privacy.matrix_factorization import buffered_toeplitz
import jaxtyping
import numpy as np
import optax
import tensorflow_federated as tff

from fcp.confidentialcompute.python import dp_mf_aggregator_jax


_TEST_STRUCT_TYPE = flang.StructWithPythonType(
    elements=(
        flang.TensorType(dtype=np.float32, shape=(3,)),
        flang.TensorType(dtype=np.float32),
    ),
    container_type=tuple,
)


def _create_test_grad_privatizer():

  def init(params) -> jaxtyping.PyTree:
    del params  # Unused.
    return jnp.int32(0)

  def privatize(
      sum_of_clipped_grads, noise_state
  ) -> tuple[jaxtyping.PyTree, jaxtyping.PyTree]:
    # Simple passthrough for tests, only increases the state counter. Test
    # coverage of the privatizer noise is done in the JaxPrivacy library.
    return sum_of_clipped_grads, noise_state + 1

  return optax.GradientTransformation(init, privatize)  # pyrefly: ignore[bad-argument-type]


class DPMFAggregatorFactoryComputationTest(absltest.TestCase):

  def test_create_does_not_raise(self):
    factory = dp_mf_aggregator_jax.DPMFAggregatorFactory(
        gradient_privatizer=_create_test_grad_privatizer(),
        clients_per_round=10,
    )

    try:
      factory.create(_TEST_STRUCT_TYPE)
    except Exception:  # pylint: disable=broad-exception-caught
      self.fail('Raised `Exception` unexpectedly.')

  def test_has_expected_type_signature(self):
    factory = dp_mf_aggregator_jax.DPMFAggregatorFactory(
        gradient_privatizer=_create_test_grad_privatizer(),
        clients_per_round=10,
    )
    process = factory.create(_TEST_STRUCT_TYPE)

    expected_state_type = flang.FederatedType(np.int32, flang.SERVER)
    with self.subTest('init'):
      expected_initialize_type = flang.FunctionType(
          parameter=None, result=expected_state_type
      )
      process.initialize.type_signature.check_equivalent_to(
          expected_initialize_type
      )
    with self.subTest('next'):
      param_value_type = flang.FederatedType(_TEST_STRUCT_TYPE, flang.CLIENTS)
      result_value_type = flang.FederatedType(_TEST_STRUCT_TYPE, flang.SERVER)
      expected_measurements_type = flang.FederatedType(
          {'num_clipped_updates': np.int32}, flang.SERVER
      )
      expected_parameter = collections.OrderedDict(
          state=expected_state_type,
          value=param_value_type,
      )
      expected_next_type = flang.FunctionType(
          parameter=expected_parameter,
          result=tff.templates.MeasuredProcessOutput(
              expected_state_type,
              result_value_type,
              expected_measurements_type,
          ),
      )
      process.next.type_signature.check_equivalent_to(expected_next_type)


class DPMFAggregatorFactoryExecutionTest(absltest.TestCase):

  def test_execution(self):
    flang.framework.set_default_context(
        flang_jax.create_sync_local_execution_context()
    )
    factory = dp_mf_aggregator_jax.DPMFAggregatorFactory(
        gradient_privatizer=_create_test_grad_privatizer(),
        clients_per_round=2,
        l2_clip_norm=1.0,
    )
    process = factory.create(_TEST_STRUCT_TYPE)
    client_updates = [
        (np.array([1.0, 1.0, 1.0], np.float32), np.float32(0.0)),
        (np.array([0.0, 0.0, 0.0], np.float32), np.float32(1.0)),
    ]
    state = process.initialize()
    output = process.next(state, client_updates)
    chex.assert_trees_all_close(
        output.result,
        (
            # Global norm of first client is sqrt(3) = 1.732, so we expect the
            # updates to be clipped by a factor of 1.0 / 1.732, and finally
            # averaged with 0.0 (divided by 2.0) for ~0.288.
            np.array([0.288675, 0.288675, 0.288675], np.float32),
            # Global norm of second client is 1.0, no clipping occurs. This
            # is averaged with 0.0 for 0.5.
            np.float32(0.5),
        ),
    )
    chex.assert_trees_all_equal(output.measurements, {'num_clipped_updates': 1})

  def test_matrix_factorization_privatizer_execution(self):
    flang.framework.set_default_context(
        flang_jax.create_sync_local_execution_context()
    )
    with jax.enable_x64(), jax.legacy_prng_key('error'):
      # A real privatizer, whose state contains typed PRNG keys. This uses fixed
      # BLT parameters since `buffered_toeplitz.optimize` is slow.
      blt = buffered_toeplitz.BufferedToeplitz.build(
          buf_decay=[0.9, 0.5], output_scale=[0.3, 0.2]
      )
      privatizer = noise_addition.matrix_factorization_privatizer(
          blt.inverse_as_streaming_matrix(),
          stddev=1.0,
          rng=0,
          dtype=jnp.float32,
      )
      factory = dp_mf_aggregator_jax.DPMFAggregatorFactory(
          gradient_privatizer=privatizer,
          clients_per_round=2,
          l2_clip_norm=1.0,
      )
      process = factory.create(_TEST_STRUCT_TYPE)
      client_updates = [
          (np.array([1.0, 1.0, 1.0], np.float32), np.float32(0.0)),
          (np.array([0.0, 0.0, 0.0], np.float32), np.float32(1.0)),
      ]
      output1 = process.next(process.initialize(), client_updates)
      output2 = process.next(output1.state, client_updates)
      for output in (output1, output2):
        self.assertEqual(output.result[0].shape, (3,))
        self.assertEqual(output.result[1].shape, ())
        chex.assert_trees_all_equal(
            output.measurements, {'num_clipped_updates': 1}
        )


if __name__ == '__main__':
  absltest.main()
