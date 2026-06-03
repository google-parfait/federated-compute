"""Accounting utilities specific to min_sep_data_source."""

import collections
import math

from jax_privacy.matrix_factorization import buffered_toeplitz
import numpy as np


def min_sep_data_source_zcdp(
    noise_multiplier: float,
    total_steps: int,
    min_separation: int,
) -> float:
  """Computes the zCDP for TreeNoRestart using min_sep_data_source to batch.

  This is Algorithm 4 of https://arxiv.org/pdf/2103.00039, with the data orders
  being those possible using min_sep_data_source.

  Args:
    noise_multiplier: The noise multiplier used in training.
    total_steps: The total number of steps in training.
    min_separation: The minimum separation between participations. That is, if a
      user participates in round t, they cannot participate in any round i such
      that 0 < |t - i| < min_separation.

  Returns:
    The zCDP parameter for DP-FTRL using TreeNoRestart and min_sep_data_source.
  """
  if noise_multiplier <= 0:
    raise ValueError('noise_multiplier must be positive')
  if total_steps <= 0:
    raise ValueError('total_steps must be positive')
  if min_separation <= 0:
    raise ValueError('min_separation must be positive')
  # Keep track of the squared sensitivity for each starting index
  squared_sensitivities = np.zeros(min_separation)
  # Start with the bottom layer of the tree, a bunch of singleton nodes
  nodes = [[i % min_separation] for i in range(total_steps)]
  while nodes:
    node_counters = [collections.Counter(n) for n in nodes]
    per_layer_squared_sensitivities = [
        sum([counter[i] ** 2 for counter in node_counters])
        for i in range(min_separation)
    ]
    squared_sensitivities += per_layer_squared_sensitivities
    # To form the next layer, merge every pair of nodes in the previous layer,
    # dropping remainder.
    nodes = [nodes[2 * i] + nodes[2 * i + 1] for i in range(len(nodes) // 2)]
  return max(squared_sensitivities) / (2 * noise_multiplier**2)


def min_sep_data_source_noise_multiplier(
    target_zcdp: float,
    total_steps: int,
    min_separation: int,
) -> float:
  """Computes noise_multiplier for TreeNoRestart to achieve target zCDP.

  Args:
    target_zcdp: The target zCDP parameter.
    total_steps: The total number of steps in training.
    min_separation: The minimum separation between participations.

  Returns:
    The noise multiplier satisfying the target zCDP.
  """
  if target_zcdp <= 0:
    raise ValueError('target_zcdp must be positive')
  zcdp_at_noise_1 = min_sep_data_source_zcdp(
      noise_multiplier=1.0,
      total_steps=total_steps,
      min_separation=min_separation,
  )
  # zCDP is proportional to 1 / noise_multiplier**2, i.e. noise_multiplier is
  # proportional to 1 / sqrt(zCDP).
  return math.sqrt(zcdp_at_noise_1 / target_zcdp)


def _minsep_sensitivity_squared(
    coefficients: np.ndarray,
    min_separation: int,
    max_participations: int | None = None,
) -> float:
  """Calculates the sensitivity squared for BLT mechanism.

  This is from Theorem 2 of https://arxiv.org/pdf/2405.13763.

  Args:
    coefficients: The coefficients of the BLT matrix.
    min_separation: The minimum separation between participations.
    max_participations: The maximum number of participations allowed.

  Returns:
    The sensitivity squared.

  Raises:
    ValueError: If coefficients is not a 1D array.
  """
  if coefficients.ndim != 1:
    raise ValueError(
        f'coefficients.shape={coefficients.shape!r} must be a 1D array'
    )
  n = coefficients.shape[0]

  if max_participations is None:
    k = math.ceil(n / min_separation)
  else:
    k = min(max_participations, math.ceil(n / min_separation))
  padding = (min_separation - n) % min_separation
  coefficients = np.pad(coefficients, (0, n - coefficients.size + padding))
  vector = coefficients.reshape(-1, min_separation).cumsum(axis=0).flatten()
  vector[min_separation * k :] = (
      vector[min_separation * k :] - vector[: -min_separation * k]
  )
  return float(vector[:n] @ vector[:n])


def zcdp_for_blt(
    matrix: buffered_toeplitz.BufferedToeplitz,
    total_steps: int,
    noise_multiplier: float,
    min_separation: int,
    max_participations: int | None = None,
) -> float:
  """Computes the zCDP for BLT.

  Args:
    matrix: The BLT matrix.
    total_steps: The total number of steps in training.
    noise_multiplier: The noise multiplier used in training.
    min_separation: The minimum separation between participations.
    max_participations: The maximum number of participations allowed. If None,
      the maximum number of participations will be determined by the minimum
      separation and total steps.

  Returns:
    The zCDP parameter for DP-FTRL using BLT.

  Raises:
    ValueError: If any of the total_steps, min_separation, max_participations,
      or noise_multiplier are not positive.
  """
  if noise_multiplier <= 0:
    raise ValueError('noise_multiplier must be positive.')
  if total_steps <= 0:
    raise ValueError('total_steps must be positive.')
  if min_separation <= 0:
    raise ValueError('min_separation must be positive.')
  if max_participations is not None and max_participations <= 0:
    raise ValueError('max_participations must be positive.')
  coefficients = np.array(matrix.toeplitz_coefs(total_steps))
  squared_sensitivity = _minsep_sensitivity_squared(
      coefficients, min_separation, max_participations
  )
  return squared_sensitivity / (2 * noise_multiplier**2)


def noise_multiplier_for_blt(
    matrix: buffered_toeplitz.BufferedToeplitz,
    total_steps: int,
    target_zcdp: float,
    min_separation: int,
    max_participations: int | None = None,
) -> float:
  """Computes the noise multiplier for BLT to satisfy zCDP.

  Args:
    matrix: The BLT matrix.
    total_steps: The total number of steps in training.
    target_zcdp: The target zCDP parameter.
    min_separation: The minimum separation between participations.
    max_participations: The maximum number of participations allowed. If None,
      the maximum number of participations will be determined by the minimum
      separation and total steps.

  Returns:
    The noise multiplier satisfying the target zCDP.
  """
  if target_zcdp <= 0:
    raise ValueError('target_zcdp must be positive.')
  zcdp_at_noise_1 = zcdp_for_blt(
      matrix=matrix,
      total_steps=total_steps,
      noise_multiplier=1.0,
      min_separation=min_separation,
      max_participations=max_participations,
  )
  # zCDP is proportional to 1 / noise_multiplier**2, i.e. noise_multiplier is
  # proportional to 1 / sqrt(zCDP).
  return math.sqrt(zcdp_at_noise_1 / target_zcdp)
