"""Accounting utilities specific to min_sep_data_source."""

import collections

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
