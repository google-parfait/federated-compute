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
"""An implementation of TFF aggregators in JAX."""

import federated_language
import federated_language_jax
import jax
import jax.numpy as jnp
import optax
import tensorflow_federated as tff

from fcp.confidentialcompute.python import prng_keys

# Nested structure of tensor specs.
TensorOrStructType = (
    federated_language.TensorType | federated_language.StructType
)


def _zeros_from_type_spec(type_spec: federated_language.Type) -> object:
  """Returns a structure of zeros from `type_spec`."""

  def _factory(type_spec):
    if isinstance(type_spec, federated_language.TensorType):
      return jnp.zeros(shape=type_spec.shape, dtype=type_spec.dtype)
    else:
      raise ValueError(
          'Expected `type_spec` to be a `federated_language.TensorType`,'
          f' found {type(type_spec)}.'
      )

  return federated_language.framework.structure_with_type(_factory, type_spec)


def _shapes_dtypes_from_type_spec(type_spec: federated_language.Type) -> object:
  """Returns a structure of shapes and dtypes from `type_spec`."""

  def _factory(type_spec):
    if isinstance(type_spec, federated_language.TensorType):
      return jax.ShapeDtypeStruct(shape=type_spec.shape, dtype=type_spec.dtype)
    else:
      raise ValueError(
          'Expected `type_spec` to be a `federated_language.TensorType`,'
          f' found {type(type_spec)}.'
      )

  return federated_language.framework.structure_with_type(_factory, type_spec)


class DPMFAggregatorFactory:
  """Aggregation factory for Differentially Private mean across clients.

  This uses the Matrix Factorization approach for DP.
  """

  def __init__(
      self,
      *,
      gradient_privatizer: optax.GradientTransformation,
      clients_per_round: int,
      l2_clip_norm: float = 1.0,
  ):
    """Initializes `DPMFAggregatorFactory`."""
    if clients_per_round <= 0.0:
      raise ValueError('clients_per_round must be a positive float.')
    if l2_clip_norm <= 0.0:
      raise ValueError('l2_clip_norm must be a positive float.')

    self._grad_privatizer = gradient_privatizer
    self._l2_clip_norm = l2_clip_norm
    self._clients_per_round = clients_per_round

  def create(
      self, value_type: TensorOrStructType
  ) -> tff.templates.AggregationProcess:
    """Creates a `tff.templates.AggregationProcess` for summing values."""
    grad_state_spec = jax.eval_shape(
        self._grad_privatizer.init, _shapes_dtypes_from_type_spec(value_type)
    )

    @federated_language.federated_computation()
    def init_fn():

      @federated_language_jax.computation
      def init_privatizer():
        params = _shapes_dtypes_from_type_spec(value_type)
        # Federated language types can't represent typed PRNG keys, so store
        # them as raw key data.
        return prng_keys.unwrap_prng_keys(self._grad_privatizer.init(params))  # pyrefly: ignore[bad-argument-type]

      return federated_language.federated_eval(
          init_privatizer, federated_language.SERVER
      )

    @federated_language.federated_computation(
        init_fn.type_signature.result,
        federated_language.FederatedType(
            value_type, federated_language.CLIENTS
        ),
    )
    def next_fn(state, value):

      @federated_language_jax.computation
      def build_zero():
        accumulator_zero = _zeros_from_type_spec(value_type)
        metrics_zero = {'num_clipped_updates': jnp.zeros([], dtype=jnp.int32)}
        return (accumulator_zero, metrics_zero)

      @federated_language_jax.computation(
          build_zero.type_signature.result, value_type
      )
      def clipped_sum(state, client_updates):

        def _clip_by_global_l2_norm(global_l2_norm, updates):
          clip_rescale_factor = self._l2_clip_norm / global_l2_norm
          clipped_updates = jax.tree.map(
              lambda update: update * clip_rescale_factor, updates
          )
          metrics = {'num_clipped_updates': 1}
          return clipped_updates, metrics

        global_l2_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(client_updates))
        )
        # Conditionally clip the updates if updates' norm is above the clipnorm.
        clipped_updates, new_metrics = jax.lax.cond(
            global_l2_norm > self._l2_clip_norm,
            _clip_by_global_l2_norm,
            lambda norm, updates: (updates, {'num_clipped_updates': 0}),
            global_l2_norm,
            client_updates,
        )
        accumulator_pytree, metrics_pytree = state
        new_accumulators = jax.tree_util.tree_map(
            jnp.add, accumulator_pytree, clipped_updates
        )
        new_metrics = jax.tree_util.tree_map(
            jnp.add, new_metrics, metrics_pytree
        )
        return new_accumulators, new_metrics

      @federated_language_jax.computation
      def add(a, b):
        return jax.tree.map(jnp.add, a, b)

      @federated_language_jax.computation
      def identity(a):
        return a

      unnoised_aggregate, metrics = federated_language.federated_aggregate(
          value,
          build_zero(),
          accumulate=clipped_sum,
          merge=add,
          report=identity,
      )

      @federated_language_jax.computation
      def finalize_noise(noise_state, unnoised_aggregate):
        # The PRNG keys are stored as raw uint32 key data, cast back to keys.
        noise_state = prng_keys.rewrap_prng_keys(noise_state, grad_state_spec)
        noised_aggregate, new_noise_state = self._grad_privatizer.update(
            unnoised_aggregate, noise_state
        )
        return prng_keys.unwrap_prng_keys(new_noise_state), jax.tree.map(
            lambda arr: arr / self._clients_per_round, noised_aggregate
        )

      new_state, noised_aggregate = federated_language.federated_map(
          finalize_noise, (state, unnoised_aggregate)
      )
      return tff.templates.MeasuredProcessOutput(
          new_state, noised_aggregate, metrics
      )

    return tff.templates.AggregationProcess(init_fn, next_fn)
