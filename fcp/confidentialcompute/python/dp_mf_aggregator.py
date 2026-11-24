"""An implementation of the DP MF aggregator using the JaxPrivacy library."""

import functools
from typing import cast

import federated_language as flang
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
from jax_privacy import clipping
import optax
import tensorflow as tf
import tensorflow_federated as tff

from fcp.confidentialcompute.python import prng_keys


def _shapes_dtypes_from_typespec(typespec: flang.Type):
  """Converts a spec to a tree of shapes and dtypes."""
  return tff.tensorflow.structure_from_tensor_type_tree(
      lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
      typespec,
  )


_jax2tf_convert_cpu_native = functools.partial(
    jax2tf.convert,
    native_serialization=True,
    native_serialization_platforms=['cpu'],
)


def _build_zero_accumulator_fn(value_type: flang.TensorType | flang.StructType):
  """Creates a computation returning zero accumulators and metrics."""

  @tff.tensorflow.computation
  def build_zero():
    def jax_zero():
      accumulator_zero = tff.tensorflow.structure_from_tensor_type_tree(
          lambda arr: jnp.zeros(shape=arr.shape, dtype=arr.dtype),
          value_type,
      )
      metrics_zero = {'num_clipped_updates': jnp.zeros([], dtype=jnp.int32)}
      return (accumulator_zero, metrics_zero)

    return _jax2tf_convert_cpu_native(jax_zero)()

  return build_zero


@tff.tensorflow.computation
def _merge_clipped_sums(a, b):
  """Merges two partial, clipped sums."""
  return tf.nest.map_structure(tf.add, a, b)


@tff.tensorflow.computation
def _identity(a):
  return a


def _jax_clip_and_accumulate(
    client_updates,
    clip_norm,
    accumulator_pytree,
    metrics_pytree,
):
  """Clips updates by global L2 norm and accumulates them with metrics."""
  clipped_updates, global_l2_norm = clipping.clip_pytree(
      client_updates,
      clip_norm=clip_norm,
      rescale_to_unit_norm=False,
  )
  was_clipped = jnp.int32(global_l2_norm > clip_norm)
  new_metrics = {'num_clipped_updates': was_clipped}
  new_accumulators = jax.tree.map(jnp.add, accumulator_pytree, clipped_updates)
  new_metrics = jax.tree.map(jnp.add, new_metrics, metrics_pytree)
  return new_accumulators, new_metrics


class DPMFAggregatorFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregation factory for Differentially Private mean across clients.

  This factory expects a privatizer from the `jax_privacy` to be created,
  supporting methods such as those from [Scaling up the Banded Matrix
  Factorization Mechanism for Differentially Private
  ML](https://arxiv.org/abs/2405.15913).
  """

  def __init__(
      self,
      *,
      gradient_privatizer: optax.GradientTransformation,
      clients_per_round: int,
      l2_clip_norm: float = 1.0,
  ):
    """Initializes `DPMFAggregatorFactory`.

    Args:
      gradient_privatizer: The gradient privatizer, calibrated to privatize the
        sum of clipped client updates with L2 clip norm `l2_clip_norm`. The
        aggregator divides the noised sum by `clients_per_round`.
      clients_per_round: The number of clients per round.
      l2_clip_norm: The L2 clip norm. The default is 1.0.
    """
    if clients_per_round <= 0.0:
      raise ValueError('clients_per_round must be a positive float.')
    if l2_clip_norm <= 0.0:
      raise ValueError('l2_clip_norm must be a positive float.')

    self._grad_privatizer = gradient_privatizer
    self._l2_clip_norm = l2_clip_norm
    self._clients_per_round = clients_per_round

  def create(
      self, value_type: flang.TensorType | flang.StructType
  ) -> tff.templates.AggregationProcess:
    """Creates a `tff.templates.AggregationProcess` for DP MF mean."""
    grad_state_spec = jax.eval_shape(
        self._grad_privatizer.init,
        _shapes_dtypes_from_typespec(value_type),
    )

    @flang.federated_computation()
    def init_fn():

      @tff.tensorflow.computation
      def init_privatizer():
        def jax_init_privatizer():
          noise_state = self._grad_privatizer.init(
              _shapes_dtypes_from_typespec(value_type)
          )
          return prng_keys.unwrap_prng_keys(noise_state)

        return _jax2tf_convert_cpu_native(jax_init_privatizer)()

      return flang.federated_eval(init_privatizer, flang.SERVER)

    build_zero = _build_zero_accumulator_fn(value_type)

    @flang.federated_computation(
        init_fn.type_signature.result,
        flang.FederatedType(value_type, flang.CLIENTS),
    )
    def next_fn(noise_state, value):

      @tff.tensorflow.computation(build_zero.type_signature.result, value_type)
      def clipped_sum(state, client_updates):
        def jax_clipped_sum(state, client_updates):
          accumulator_pytree, metrics_pytree = state
          return _jax_clip_and_accumulate(
              client_updates,
              self._l2_clip_norm,
              accumulator_pytree,
              metrics_pytree,
          )

        return _jax2tf_convert_cpu_native(jax_clipped_sum)(
            state, client_updates
        )

      unnoised_sum, metrics = flang.federated_aggregate(
          value,
          build_zero(),
          accumulate=clipped_sum,
          merge=_merge_clipped_sums,
          report=_identity,
      )

      @tff.tensorflow.computation
      def finalize_noise(noise_state, unnoised_aggregate):
        def jax_finalize_noise(noise_state, unnoised_aggregate):
          noise_state = prng_keys.rewrap_prng_keys(noise_state, grad_state_spec)
          noised_aggregate, new_noise_state = self._grad_privatizer.update(
              unnoised_aggregate, noise_state
          )
          noised_mean = jax.tree.map(
              lambda arr: arr / self._clients_per_round,
              noised_aggregate,
          )
          return prng_keys.unwrap_prng_keys(new_noise_state), noised_mean

        return _jax2tf_convert_cpu_native(jax_finalize_noise)(
            noise_state, unnoised_aggregate
        )

      new_state, noised_aggregate = flang.federated_map(
          finalize_noise, (noise_state, unnoised_sum)
      )
      return tff.templates.MeasuredProcessOutput(
          new_state, noised_aggregate, metrics
      )

    return tff.templates.AggregationProcess(init_fn, next_fn)


class DPMFAdaptiveClippingAggregatorFactory(
    tff.aggregators.UnweightedAggregationFactory
):
  """Aggregation factory for Differentially Private mean with adaptive clipping.

  This factory expects privatizers from `jax_privacy` to be created for both
  the gradients and the clipping indicator bits. Both privatizers should be
  configured to privatize the sum of client contributions (assuming
  single-client sensitivity of 1.0):
  - `gradient_privatizer` is calibrated assuming a clip norm of 1.0 (sensitivity
    1.0). In each round t, the aggregator normalizes the gradient sum by C_t
    (scaling client updates to unit norm) before privatizing. By default
    (`rescale_to_unit_norm=True`), the noised result is divided by N without
    multiplying back by C_t (so the learning rate controls the scale of
    updates); if `rescale_to_unit_norm=False`, it is scaled by C_t / N.
  - `clipping_privatizer` is calibrated to privatize the count of unclipped
    clients (sum of indicator bits b_i in {0, 1}, sensitivity 1.0). The
    aggregator passes the unclipped count to the privatizer and divides the
    noised count by N to obtain the noised fraction of unclipped clients.

  The clipping threshold C_t is adaptively updated from round to round:
    C_{t+1} = C_t * exp(-eta_c * (noised_b_t - 0.5))
  where b_t is the fraction of clients whose updates were unclipped in round t
  (i.e. b_i = 1 if unclipped, 0 otherwise), and noised_b_t is privatized using
  `clipping_privatizer`. As examples: If all clients are clipped, C_t increases;
  if all clients are unclipped, C_t decreases. If roughly half the clients are
  clipped, C_t remains roughly constant.
  """

  def __init__(
      self,
      *,
      gradient_privatizer: optax.GradientTransformation,
      clipping_privatizer: optax.GradientTransformation,
      clients_per_round: int,
      initial_l2_clip_norm: float = 0.1,
      eta_c: float = 0.2,
      rescale_to_unit_norm: bool = True,
  ):
    """Initializes `DPMFAdaptiveClippingAggregatorFactory`.

    Args:
      gradient_privatizer: The gradient privatizer. This should be calibrated to
        privatize the sum of client updates assuming an L2 clip norm of 1.0,
        e.g. for independent noise, the standard deviation just equals the noise
        multiplier. The aggregator normalizes the clipped sum by the clip norm
        before privatizing, then divides by `clients_per_round` (and multiplies
        back by the clip norm only when `rescale_to_unit_norm=False`).
      clipping_privatizer: The clipping privatizer. This should be calibrated to
        privatize the sum of unclipped indicator bits, e.g. for independent
        noise, the standard deviation equals the noise multiplier. The
        aggregator divides the noised count by `clients_per_round`.
      clients_per_round: The number of clients per round.
      initial_l2_clip_norm: The initial L2 clip norm. The default is 0.1, as in
        https://arxiv.org/pdf/1905.03871.
      eta_c: The adaptive clipping rate learning rate. The default is 0.2, as in
        https://arxiv.org/pdf/1905.03871.
      rescale_to_unit_norm: If True (default), keep clipped updates scaled to
        unit norm (1 / clip_norm) rather than multiplying back by `clip_norm`
        after adding noise, so the learning rate controls the update scale.
    """
    if clients_per_round <= 0.0:
      raise ValueError('clients_per_round must be a positive float.')
    if initial_l2_clip_norm <= 0.0:
      raise ValueError('initial_l2_clip_norm must be a positive float.')
    if eta_c < 0.0:
      raise ValueError('eta_c must be non-negative.')

    self._grad_privatizer = gradient_privatizer
    self._clipping_privatizer = clipping_privatizer
    self._clients_per_round = clients_per_round
    self._initial_l2_clip_norm = initial_l2_clip_norm
    self._eta_c = eta_c
    self._rescale_to_unit_norm = rescale_to_unit_norm

  def create(
      self, value_type: flang.TensorType | flang.StructType
  ) -> tff.templates.AggregationProcess:
    """Creates an `AggregationProcess` for DP MF adaptive clipping."""
    grad_state_spec = jax.eval_shape(
        self._grad_privatizer.init,
        _shapes_dtypes_from_typespec(value_type),
    )
    clipping_state_spec = jax.eval_shape(
        self._clipping_privatizer.init,
        jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )

    @flang.federated_computation()
    def init_fn():

      @tff.tensorflow.computation
      def init_privatizer():
        def jax_init_privatizer():
          grad_noise_state = self._grad_privatizer.init(
              _shapes_dtypes_from_typespec(value_type)
          )
          clipping_noise_state = self._clipping_privatizer.init(
              jnp.zeros([], dtype=jnp.float32)
          )
          clip_norm = jnp.float32(self._initial_l2_clip_norm)
          return (
              prng_keys.unwrap_prng_keys(grad_noise_state),
              prng_keys.unwrap_prng_keys(clipping_noise_state),
              clip_norm,
          )

        return _jax2tf_convert_cpu_native(jax_init_privatizer)()

      return flang.federated_eval(init_privatizer, flang.SERVER)

    build_zero = _build_zero_accumulator_fn(value_type)

    @flang.federated_computation(
        init_fn.type_signature.result,
        flang.FederatedType(value_type, flang.CLIENTS),
    )
    def next_fn(state, value):
      clip_norm = state[2]
      clients_clip_norm = flang.federated_broadcast(clip_norm)
      client_input = flang.federated_zip((value, clients_clip_norm))

      @tff.tensorflow.computation(
          build_zero.type_signature.result,
          client_input.type_signature.member,
      )
      def clipped_sum(state, client_input):
        def jax_clipped_sum(state, client_input):
          client_updates, clip_norm = client_input
          accumulator_pytree, metrics_pytree = state
          return _jax_clip_and_accumulate(
              client_updates,
              clip_norm,
              accumulator_pytree,
              metrics_pytree,
          )

        return _jax2tf_convert_cpu_native(jax_clipped_sum)(state, client_input)

      unnoised_sum, metrics = flang.federated_aggregate(
          client_input,
          build_zero(),
          accumulate=clipped_sum,
          merge=_merge_clipped_sums,
          report=_identity,
      )

      @tff.tensorflow.computation
      def finalize_noise(state, unnoised_aggregate, metrics):
        """Compute the noised mean and update the adaptive clip norm."""

        def jax_finalize_noise(state, unnoised_aggregate, metrics):
          grad_noise_state, clipping_noise_state, clip_norm = state
          grad_noise_state = prng_keys.rewrap_prng_keys(
              grad_noise_state, grad_state_spec
          )
          clipping_noise_state = prng_keys.rewrap_prng_keys(
              clipping_noise_state, clipping_state_spec
          )
          scaled_aggregate = jax.tree.map(
              lambda arr: arr / clip_norm, unnoised_aggregate
          )
          noised_scaled, new_grad_noise_state = self._grad_privatizer.update(
              scaled_aggregate, grad_noise_state
          )
          scaling = 1.0 if self._rescale_to_unit_norm else clip_norm
          noised_mean = jax.tree.map(
              lambda arr: arr * (scaling / self._clients_per_round),
              noised_scaled,
          )
          num_clipped = metrics['num_clipped_updates']
          unclipped_count = jnp.float32(self._clients_per_round) - jnp.float32(
              num_clipped
          )
          noised_unclipped, new_clipping_noise_state = (
              self._clipping_privatizer.update(
                  unclipped_count, clipping_noise_state
              )
          )
          noised_b_t = (
              cast(jax.Array, noised_unclipped) / self._clients_per_round
          )
          new_clip_norm = clip_norm * jnp.exp(-self._eta_c * (noised_b_t - 0.5))
          new_state = (
              prng_keys.unwrap_prng_keys(new_grad_noise_state),
              prng_keys.unwrap_prng_keys(new_clipping_noise_state),
              new_clip_norm,
          )
          return new_state, noised_mean

        return _jax2tf_convert_cpu_native(jax_finalize_noise)(
            state, unnoised_aggregate, metrics
        )

      new_state, noised_aggregate = flang.federated_map(
          finalize_noise, (state, unnoised_sum, metrics)
      )
      return tff.templates.MeasuredProcessOutput(
          new_state, noised_aggregate, metrics
      )

    return tff.templates.AggregationProcess(init_fn, next_fn)
