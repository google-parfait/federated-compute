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
"""Utilities for storing typed JAX PRNG keys in federated language values.

Federated language types can't represent typed JAX PRNG keys (e.g. those in the
state of JAX Privacy's matrix factorization privatizers), so computations store
them as raw uint32 key data and rewrap them into typed keys when they're used.
"""

from typing import Any

import jax


def _is_prng_key(leaf: Any) -> bool:
  """Returns True if the leaf has a JAX PRNG key dtype."""
  return hasattr(leaf, 'dtype') and jax.dtypes.issubdtype(
      leaf.dtype, jax.dtypes.prng_key
  )


def unwrap_prng_keys(pytree: Any) -> Any:
  """Unwraps typed JAX PRNG keys in a PyTree into raw uint32 key data."""

  def _unwrap_leaf(leaf: Any) -> Any:
    if _is_prng_key(leaf):
      return jax.random.key_data(leaf)
    return leaf

  return jax.tree.map(_unwrap_leaf, pytree)


def rewrap_prng_keys(pytree: Any, spec_tree: Any) -> Any:
  """Rewraps raw uint32 key data in a PyTree into typed JAX PRNG keys.

  Args:
    pytree: A PyTree in which PRNG keys are stored as raw uint32 key data.
    spec_tree: A PyTree with the same structure as `pytree` (e.g. from
      `jax.eval_shape`), whose leaves with PRNG key dtypes indicate which leaves
      of `pytree` to rewrap.

  Returns:
    `pytree`, with the indicated leaves rewrapped into typed PRNG keys.
  """

  def _rewrap_leaf(leaf: Any, spec_leaf: Any) -> Any:
    if _is_prng_key(spec_leaf):
      # Older JAX versions (e.g. those in TEE containers) don't support the
      # `dtype` argument of `wrap_key_data`, so pass the key dtype's PRNG
      # implementation instead, as JAX does internally.
      impl = spec_leaf.dtype._impl  # pylint: disable=protected-access
      return jax.random.wrap_key_data(leaf, impl=impl)
    return leaf

  return jax.tree.map(_rewrap_leaf, pytree, spec_tree)
