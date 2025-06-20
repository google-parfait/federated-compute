from typing import Any

from absl.testing import absltest
import federated_language as flang
import jax
from jax_privacy.stream_privatization import gradient_privatizer as gradient_privatizer_lib
import numpy as np
import tensorflow as tf

from fcp.confidentialcompute.python import dp_mf_aggregator

_TEST_STRUCT_TYPE = flang.StructWithPythonType(
    elements=(
        flang.TensorType(dtype=np.float32, shape=(3,)),
        flang.TensorType(dtype=np.float32),
    ),
    container_type=tuple,
)


def _create_test_grad_privatizer():

  def init(params):
    del params  # Unused.
    return np.int32(0)

  def privatize(
      *, sum_of_clipped_grads, noise_state, prng_key
  ) -> tuple[Any, Any]:
    del prng_key  # Unused.
    # Simple passthrough for tests, only increases the state counter. Test
    # coverage of the privatizer noise is done in the JaxPrivacy library.
    return sum_of_clipped_grads, noise_state + 1

  return gradient_privatizer_lib.GradientPrivatizer(init, privatize)


class DPMFAggregatorFactoryExecutionTest(tf.test.TestCase):

  def test_execution(self):
    initial_rng_key = 7
    dp_mf_factory = dp_mf_aggregator.DPMFAggregatorFactory(
        gradient_privatizer=_create_test_grad_privatizer(),
        clients_per_round=2,
        l2_clip_norm=1.0,
        initial_rng_key=initial_rng_key,
    )
    process = dp_mf_factory.create(_TEST_STRUCT_TYPE)
    client_updates = [
        (np.array([1.0, 1.0, 1.0], np.float32), np.float32(0.0)),
        (np.array([0.0, 0.0, 0.0], np.float32), np.float32(1.0)),
    ]
    state = process.initialize()
    self.assertEqual(
        jax.random.wrap_key_data(state[0]), jax.random.key(initial_rng_key)
    )
    output = process.next(state, client_updates)
    # Since we are using a fake pass-thru privatizer, we only expect clipped sum
    # to be returned.
    self.assertAllClose(
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
    self.assertNotEqual(
        jax.random.wrap_key_data(output.state[0]),
        jax.random.key(initial_rng_key),
    )
    self.assertAllClose(output.measurements, {'num_clipped_updates': 1})


if __name__ == '__main__':
  absltest.main()
