from typing import Any

from absl.testing import absltest
import federated_language as flang
import jax
import jax.numpy as jnp
from jax_privacy import noise_addition
from jax_privacy.matrix_factorization import buffered_toeplitz
import numpy as np
import optax
import tensorflow as tf

from fcp.confidentialcompute.python import dp_mf_aggregator

_TEST_STRUCT_TYPE = flang.StructWithPythonType(
    elements=(
        flang.TensorType(dtype=np.float32, shape=(3,)),
        flang.TensorType(dtype=np.float32),
    ),
    container_type=tuple,
)
_TEST_VECTOR_TYPE = flang.TensorType(dtype=np.float32, shape=(1,))
_NUM_ROUNDS = 10
_MIN_SEP = 1
_MIN_BUFFERS = 3
_MAX_BUFFERS = 5
_STDDEV = 1.0
_RNG_SEED = 0
_CLIENTS_PER_ROUND = 2
_L2_CLIP_NORM = 1.0
_INVALID_CLIENTS_PER_ROUND = 0
_INVALID_L2_CLIP_NORM = -1.0
_ETA_C = 0.2
_LEGACY_PRNG_KEY_ERROR = 'error'


def _create_matrix_factorization_privatizer() -> optax.GradientTransformation:
  blt = buffered_toeplitz.optimize(
      n=_NUM_ROUNDS,
      min_sep=_MIN_SEP,
      min_buffers=_MIN_BUFFERS,
      max_buffers=_MAX_BUFFERS,
  )
  return noise_addition.matrix_factorization_privatizer(
      blt.inverse_as_streaming_matrix(),
      stddev=_STDDEV,
      rng=_RNG_SEED,
      dtype=jnp.float32,
  )


def _create_test_grad_privatizer():

  def init(params):
    del params  # Unused.
    return np.int32(0)

  def privatize(sum_of_clipped_grads, noise_state) -> tuple[Any, Any]:
    # Simple passthrough for tests, only increases the state counter. Test
    # coverage of the privatizer noise is done in the JaxPrivacy library.
    return sum_of_clipped_grads, noise_state + 1

  return optax.GradientTransformation(init, privatize)  # pyrefly: ignore[bad-argument-type]


class DPMFAggregatorFactoryExecutionTest(tf.test.TestCase):

  def test_invalid_args(self):
    grad_privatizer = _create_test_grad_privatizer()
    with self.assertRaises(ValueError):
      dp_mf_aggregator.DPMFAggregatorFactory(
          gradient_privatizer=grad_privatizer,
          clients_per_round=_INVALID_CLIENTS_PER_ROUND,
      )
    with self.assertRaises(ValueError):
      dp_mf_aggregator.DPMFAggregatorFactory(
          gradient_privatizer=grad_privatizer,
          clients_per_round=_CLIENTS_PER_ROUND,
          l2_clip_norm=_INVALID_L2_CLIP_NORM,
      )

  def test_execution(self):
    dp_mf_factory = dp_mf_aggregator.DPMFAggregatorFactory(
        gradient_privatizer=_create_test_grad_privatizer(),
        clients_per_round=2,
        l2_clip_norm=1.0,
    )
    process = dp_mf_factory.create(_TEST_STRUCT_TYPE)
    client_updates = [
        (np.array([1.0, 1.0, 1.0], np.float32), np.float32(0.0)),
        (np.array([0.0, 0.0, 0.0], np.float32), np.float32(1.0)),
    ]
    state = process.initialize()
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

    self.assertAllClose(output.measurements, {'num_clipped_updates': 1})

  def test_matrix_factorization_privatizer_execution(self):
    with (
        jax.enable_x64(),
        jax.legacy_prng_key(_LEGACY_PRNG_KEY_ERROR),
    ):
      dp_mf_factory = dp_mf_aggregator.DPMFAggregatorFactory(
          gradient_privatizer=_create_matrix_factorization_privatizer(),
          clients_per_round=_CLIENTS_PER_ROUND,
          l2_clip_norm=_L2_CLIP_NORM,
      )
      process = dp_mf_factory.create(_TEST_STRUCT_TYPE)
      client_updates = [
          (np.array([1.0, 1.0, 1.0], np.float32), np.float32(0.0)),
          (np.array([0.0, 0.0, 0.0], np.float32), np.float32(1.0)),
      ]
      state = process.initialize()
      output1 = process.next(state, client_updates)
      self.assertEqual(output1.result[0].shape, (3,))
      self.assertEqual(output1.result[1].shape, ())
      self.assertAllClose(output1.measurements, {'num_clipped_updates': 1})
      output2 = process.next(output1.state, client_updates)
      self.assertEqual(output2.result[0].shape, (3,))
      self.assertEqual(output2.result[1].shape, ())
      self.assertAllClose(output2.measurements, {'num_clipped_updates': 1})


def _create_test_clipping_privatizer(noise: float = 0.0):

  def init(params):
    del params  # Unused.
    return np.int32(0)

  def privatize(unclipped_count, noise_state) -> tuple[Any, Any]:
    return unclipped_count + noise, noise_state + 1

  return optax.GradientTransformation(init, privatize)  # pyrefly: ignore[bad-argument-type]


class DPMFAdaptiveClippingAggregatorFactoryExecutionTest(tf.test.TestCase):

  def test_invalid_args(self):
    grad_privatizer = _create_test_grad_privatizer()
    clip_privatizer = _create_test_clipping_privatizer()
    with self.assertRaises(ValueError):
      dp_mf_aggregator.DPMFAdaptiveClippingAggregatorFactory(
          gradient_privatizer=grad_privatizer,
          clipping_privatizer=clip_privatizer,
          clients_per_round=0,
      )
    with self.assertRaises(ValueError):
      dp_mf_aggregator.DPMFAdaptiveClippingAggregatorFactory(
          gradient_privatizer=grad_privatizer,
          clipping_privatizer=clip_privatizer,
          clients_per_round=2,
          initial_l2_clip_norm=-1.0,
      )
    with self.assertRaises(ValueError):
      dp_mf_aggregator.DPMFAdaptiveClippingAggregatorFactory(
          gradient_privatizer=grad_privatizer,
          clipping_privatizer=clip_privatizer,
          clients_per_round=2,
          eta_c=-0.1,
      )

  def test_execution_single_round(self):
    factory = dp_mf_aggregator.DPMFAdaptiveClippingAggregatorFactory(
        gradient_privatizer=_create_test_grad_privatizer(),
        clipping_privatizer=_create_test_clipping_privatizer(),
        clients_per_round=2,
        initial_l2_clip_norm=1.0,
        eta_c=0.2,
    )
    process = factory.create(_TEST_STRUCT_TYPE)
    client_updates = [
        (np.array([1.0, 1.0, 1.0], np.float32), np.float32(0.0)),
        (np.array([0.0, 0.0, 0.0], np.float32), np.float32(1.0)),
    ]
    state = process.initialize()
    self.assertAllClose(state[2], 1.0)
    self.assertEqual(state[0], 0)
    self.assertEqual(state[1], 0)

    output = process.next(state, client_updates)
    # Global norm of first client is sqrt(3) ~ 1.732, clipped to 1.0.
    # Second client has norm 1.0, unclipped.
    self.assertAllClose(
        output.result,
        (
            np.array([0.288675, 0.288675, 0.288675], np.float32),
            np.float32(0.5),
        ),
    )
    self.assertAllClose(output.measurements, {'num_clipped_updates': 1})
    # avg_b_t = 1/2 = 0.5. Since noised_avg_b_t = 0.5, clip norm unchanged:
    # 1.0 * exp(-0.2 * (0.5 - 0.5)) = 1.0.
    self.assertAllClose(output.state[2], 1.0)
    self.assertEqual(output.state[0], 1)
    self.assertEqual(output.state[1], 1)

  def test_adaptive_clipping_multi_round(self):
    factory = dp_mf_aggregator.DPMFAdaptiveClippingAggregatorFactory(
        gradient_privatizer=_create_test_grad_privatizer(),
        clipping_privatizer=_create_test_clipping_privatizer(),
        clients_per_round=2,
        initial_l2_clip_norm=1.0,
        eta_c=0.2,
    )
    process = factory.create(_TEST_VECTOR_TYPE)
    state = process.initialize()

    # Round 1: Both clients clipped (norms: 2.0 and 2.0).
    round1_updates = [
        np.array([2.0], np.float32),
        np.array([2.0], np.float32),
    ]
    output1 = process.next(state, round1_updates)
    self.assertAllClose(output1.measurements, {'num_clipped_updates': 2})
    # Both clipped -> avg_b_t = 0.0 -> C_1 = 1.0 * exp(-0.2 * (0.0 - 0.5))
    # = exp(0.1) ~ 1.1051709 (clip norm goes UP).
    expected_c1 = np.float32(np.exp(0.1))
    self.assertAllClose(output1.state[2], expected_c1)

    # Round 2: Client 1 has norm 1.05. At C_0=1.0 it would have been clipped,
    # but at C_1 ~ 1.105 it is UNCLIPPED. Client 2 has norm 1.5 (clipped).
    round2_updates = [
        np.array([1.05], np.float32),
        np.array([1.5], np.float32),
    ]
    output2 = process.next(output1.state, round2_updates)
    self.assertAllClose(output2.measurements, {'num_clipped_updates': 1})
    # One unclipped -> avg_b_t = 0.5 -> C_2 = C_1 * exp(-0.2 * (0.5 - 0.5))
    # = C_1.
    self.assertAllClose(output2.state[2], expected_c1)

    # Round 3: Neither client clipped (norms 0.1 and 0.1).
    round3_updates = [
        np.array([0.1], np.float32),
        np.array([0.1], np.float32),
    ]
    output3 = process.next(output2.state, round3_updates)
    self.assertAllClose(output3.measurements, {'num_clipped_updates': 0})
    # Both unclipped -> avg_b_t = 1.0 -> C_3 = C_2 * exp(-0.2 * (1.0 - 0.5))
    # = C_1 * exp(-0.1) = 1.0 (clip norm goes DOWN).
    self.assertAllClose(output3.state[2], 1.0)

  def test_clipping_noise_effect(self):
    factory = dp_mf_aggregator.DPMFAdaptiveClippingAggregatorFactory(
        gradient_privatizer=_create_test_grad_privatizer(),
        clipping_privatizer=_create_test_clipping_privatizer(noise=0.2),
        clients_per_round=2,
        initial_l2_clip_norm=1.0,
        eta_c=0.2,
    )
    process = factory.create(_TEST_VECTOR_TYPE)
    state = process.initialize()
    # Client 1 has norm 2.0 (clipped). Client 2 has norm 0.5 (unclipped).
    client_updates = [
        np.array([2.0], np.float32),
        np.array([0.5], np.float32),
    ]
    output = process.next(state, client_updates)
    self.assertAllClose(output.measurements, {'num_clipped_updates': 1})
    # 1 unclipped client -> count = 1.0. With noise = 0.2 on the sum,
    # noised count = 1.2, so noised_b_t = 1.2 / 2 = 0.6.
    # C_1 = 1.0 * exp(-0.2 * (0.6 - 0.5)) = exp(-0.02) ~ 0.98019867.
    expected_c1 = np.float32(np.exp(-0.02))
    self.assertAllClose(output.state[2], expected_c1)

  def test_default_args(self):
    factory = dp_mf_aggregator.DPMFAdaptiveClippingAggregatorFactory(
        gradient_privatizer=_create_test_grad_privatizer(),
        clipping_privatizer=_create_test_clipping_privatizer(),
        clients_per_round=2,
    )
    self.assertEqual(factory._initial_l2_clip_norm, 0.1)
    self.assertEqual(factory._eta_c, 0.2)
    process = factory.create(_TEST_VECTOR_TYPE)
    state = process.initialize()
    self.assertAllClose(state[2], 0.1)

  def test_gradient_noise_scaled_by_clip_norm(self):

    def _grad_privatizer_with_noise(noise_val: float):
      def init(params):
        del params
        return np.int32(0)

      def privatize(grads, state) -> tuple[Any, Any]:
        noisy_grads = tf.nest.map_structure(lambda g: g + noise_val, grads)
        return noisy_grads, state + 1

      return optax.GradientTransformation(init, privatize)  # pyrefly: ignore[bad-argument-type]

    # Initial clip norm C_0 is 2.0. Base noise is 0.5 (calibrated for C=1.0).
    factory = dp_mf_aggregator.DPMFAdaptiveClippingAggregatorFactory(
        gradient_privatizer=_grad_privatizer_with_noise(0.5),
        clipping_privatizer=_create_test_clipping_privatizer(),
        clients_per_round=2,
        initial_l2_clip_norm=2.0,
        eta_c=0.2,
    )
    process = factory.create(_TEST_VECTOR_TYPE)
    state = process.initialize()

    # Round 1: Zero updates (unnoised sum is 0).
    # Since C_0 = 2.0, noise added to sum is 2.0 * 0.5 = 1.0.
    # Divided by clients_per_round (2), noise on mean is 1.0 / 2 = 0.5.
    zero_updates = [
        np.array([0.0], np.float32),
        np.array([0.0], np.float32),
    ]
    output1 = process.next(state, zero_updates)
    self.assertAllClose(output1.result, np.array([0.5], np.float32))
    # Both updates unclipped -> avg_b_t = 1.0.
    # C_1 = 2.0 * exp(-0.2 * (1.0 - 0.5)) = 2.0 * exp(-0.1).
    expected_c1 = np.float32(2.0 * np.exp(-0.1))
    self.assertAllClose(output1.state[2], expected_c1)

    # Round 2: Zero updates again, using output1.state with updated C_1.
    # Noise added to sum is C_1 * 0.5 = (2.0 * exp(-0.1)) * 0.5 = exp(-0.1).
    # Divided by clients_per_round (2), noise on mean is 0.5 * exp(-0.1).
    output2 = process.next(output1.state, zero_updates)
    expected_mean2 = np.float32(0.5 * np.exp(-0.1))
    self.assertAllClose(output2.result, np.array([expected_mean2], np.float32))

  def test_matrix_factorization_privatizer_execution(self):
    with (
        jax.enable_x64(),
        jax.legacy_prng_key(_LEGACY_PRNG_KEY_ERROR),
    ):
      factory = dp_mf_aggregator.DPMFAdaptiveClippingAggregatorFactory(
          gradient_privatizer=_create_matrix_factorization_privatizer(),
          clipping_privatizer=_create_matrix_factorization_privatizer(),
          clients_per_round=_CLIENTS_PER_ROUND,
          initial_l2_clip_norm=_L2_CLIP_NORM,
          eta_c=_ETA_C,
      )
      process = factory.create(_TEST_STRUCT_TYPE)
      client_updates = [
          (np.array([1.0, 1.0, 1.0], np.float32), np.float32(0.0)),
          (np.array([0.0, 0.0, 0.0], np.float32), np.float32(1.0)),
      ]
      state = process.initialize()
      self.assertAllClose(state[2], _L2_CLIP_NORM)
      output1 = process.next(state, client_updates)
      self.assertEqual(output1.result[0].shape, (3,))
      self.assertEqual(output1.result[1].shape, ())
      self.assertAllClose(output1.measurements, {'num_clipped_updates': 1})
      output2 = process.next(output1.state, client_updates)
      self.assertEqual(output2.result[0].shape, (3,))
      self.assertEqual(output2.result[1].shape, ())
      self.assertAllClose(output2.measurements, {'num_clipped_updates': 2})


if __name__ == '__main__':
  absltest.main()
