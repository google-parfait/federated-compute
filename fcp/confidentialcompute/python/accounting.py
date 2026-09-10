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


def _combine_noise_multipliers(
    gradient_noise_multiplier: float,
    clipping_noise_multiplier: float,
) -> float:
  """Computes the combined noise multiplier from gradient and clipping noise.

  When using adaptive clipping with a Gaussian mechanism, both the gradient
  noise (sigma_g) and the clipping noise (sigma_c) contribute to the overall
  privacy cost. The combined noise multiplier is:

    sigma = sqrt(1 / (1 / sigma_g^2 + 1 / sigma_c^2))

  This combined value should be used as the noise_multiplier argument to
  `zcdp_for_blt`.

  Args:
    gradient_noise_multiplier: The noise multiplier used for gradients
      (sigma_g).
    clipping_noise_multiplier: The noise multiplier used for adaptive clipping
      (sigma_c).

  Returns:
    The combined noise multiplier.

  Raises:
    ValueError: If either noise multiplier is not positive.
  """
  if gradient_noise_multiplier <= 0:
    raise ValueError('gradient_noise_multiplier must be positive.')
  if clipping_noise_multiplier <= 0:
    raise ValueError('clipping_noise_multiplier must be positive.')
  return math.sqrt(
      1.0
      / (
          1.0 / gradient_noise_multiplier**2
          + 1.0 / clipping_noise_multiplier**2
      )
  )


def zcdp_for_blt_with_adaptive_clipping(
    matrix: buffered_toeplitz.BufferedToeplitz,
    total_steps: int,
    gradient_noise_multiplier: float,
    clipping_noise_multiplier: float,
    min_separation: int,
    max_participations: int | None = None,
) -> float:
  """Computes the zCDP for BLT with adaptive clipping.

  When adaptive clipping is used, both the gradient mechanism (sigma_g) and the
  clipping mechanism (sigma_c) contribute to the privacy cost. This function
  computes the combined noise multiplier and delegates to `zcdp_for_blt`.

  Args:
    matrix: The BLT matrix.
    total_steps: The total number of steps in training.
    gradient_noise_multiplier: The noise multiplier used for gradients
      (sigma_g).
    clipping_noise_multiplier: The noise multiplier used for the adaptive
      clipping Gaussian mechanism (sigma_c). A recommended choice is
      `clients_per_round / 20`.
    min_separation: The minimum separation between participations.
    max_participations: The maximum number of participations allowed. If None,
      the maximum number of participations will be determined by the minimum
      separation and total steps.

  Returns:
    The zCDP parameter for DP-FTRL using BLT with adaptive clipping.
  """
  combined_nm = _combine_noise_multipliers(
      gradient_noise_multiplier, clipping_noise_multiplier
  )
  return zcdp_for_blt(
      matrix=matrix,
      total_steps=total_steps,
      noise_multiplier=combined_nm,
      min_separation=min_separation,
      max_participations=max_participations,
  )


def noise_multiplier_for_blt_with_adaptive_clipping(
    matrix: buffered_toeplitz.BufferedToeplitz,
    total_steps: int,
    target_zcdp: float,
    clipping_noise_multiplier: float,
    min_separation: int,
    max_participations: int | None = None,
) -> float:
  """Computes noise multipliers for BLT with adaptive clipping to satisfy zCDP.

  Given a target zCDP and a clipping noise multiplier (sigma_c), this function
  computes the combined noise multiplier needed to achieve the target zCDP,
  then derives the gradient noise multiplier (sigma_g).

  A recommended choice for sigma_c is `clients_per_round / 20`.

  Args:
    matrix: The BLT matrix.
    total_steps: The total number of steps in training.
    target_zcdp: The target zCDP parameter.
    clipping_noise_multiplier: The noise multiplier for the adaptive clipping
      Gaussian mechanism (sigma_c).
    min_separation: The minimum separation between participations.
    max_participations: The maximum number of participations allowed. If None,
      the maximum number of participations will be determined by the minimum
      separation and total steps.

  Returns:
    The gradient noise multiplier (sigma_g) that, when combined with
    clipping_noise_multiplier, achieves the target zCDP.

  Raises:
    ValueError: If the required combined noise multiplier is >= sigma_c, making
      it impossible to separate the gradient noise multiplier.
  """
  combined_nm = noise_multiplier_for_blt(
      matrix=matrix,
      total_steps=total_steps,
      target_zcdp=target_zcdp,
      min_separation=min_separation,
      max_participations=max_participations,
  )
  if clipping_noise_multiplier <= combined_nm:
    raise ValueError(
        f'Cannot achieve target_zcdp={target_zcdp} with'
        f' clipping_noise_multiplier={clipping_noise_multiplier}. The required'
        f' combined noise multiplier is {combined_nm}, which must be strictly'
        ' less than clipping_noise_multiplier. Either increase'
        ' clipping_noise_multiplier or relax target_zcdp.'
    )
  return math.sqrt(
      1.0 / (1.0 / combined_nm**2 - 1.0 / clipping_noise_multiplier**2)
  )
