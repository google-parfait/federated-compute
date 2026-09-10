import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax_privacy.matrix_factorization import buffered_toeplitz
import numpy as np

from fcp.confidentialcompute.python import accounting

# Required by jnp.float64.
jax.config.update('jax_enable_x64', True)

# (noise_multiplier, total_steps, min_separation, zcdp)
_TREE_NO_RESTART_TEST_CASES = (
    (1.0, 4, 2, 4.0),
    (2.0, 4, 2, 1.0),
    (1.0, 5, 2, 4.5),
    (1.0, 7, 2, 5.5),
    (1.0, 7, 3, 4.5),
)


class AccountingTest(parameterized.TestCase):

  @parameterized.parameters(*_TREE_NO_RESTART_TEST_CASES)
  def test_zcdp_simple(
      self, noise_multiplier, total_steps, min_separation, expected_zcdp
  ):
    """Test that the zCDP is correct for some cases solvable by-hand."""
    zcdp = accounting.min_sep_data_source_zcdp(
        noise_multiplier=noise_multiplier,
        total_steps=total_steps,
        min_separation=min_separation,
    )
    self.assertEqual(zcdp, expected_zcdp)

  @parameterized.parameters(*_TREE_NO_RESTART_TEST_CASES)
  def test_min_sep_data_source_noise_multiplier(
      self, expected_noise_multiplier, total_steps, min_separation, target_zcdp
  ):
    noise_multiplier = accounting.min_sep_data_source_noise_multiplier(
        target_zcdp=target_zcdp,
        total_steps=total_steps,
        min_separation=min_separation,
    )
    self.assertAlmostEqual(noise_multiplier, expected_noise_multiplier)

  def test_min_sep_data_source_noise_multiplier_invalid_inputs(self):
    with self.assertRaisesRegex(ValueError, 'target_zcdp must be positive'):
      accounting.min_sep_data_source_noise_multiplier(
          target_zcdp=0.0,
          total_steps=4,
          min_separation=2,
      )

  @parameterized.named_parameters(
      ('with_max_participations', 2),
      ('no_max_participations', None),
  )
  def test_zcdp_for_blt(self, max_participations):
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=[
            0.9999999999921251,
            0.9944453083640997,
            0.8985923474607591,
            0.4912001418098778,
        ],
        output_scale=[
            0.0070314825502323835,
            0.10613806907600574,
            0.1898159060327625,
            0.1966594748073734,
        ],
    )
    zcdp = accounting.zcdp_for_blt(
        blt,
        total_steps=100,
        min_separation=50,
        noise_multiplier=1.0,
        max_participations=max_participations,
    )
    self.assertAlmostEqual(zcdp, 3.17226586, places=6)

  @parameterized.named_parameters(
      ('with_max_participations', 2),
      ('no_max_participations', None),
  )
  def test_noise_multiplier_for_blt(self, max_participations):
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=[
            0.9999999999921251,
            0.9944453083640997,
            0.8985923474607591,
            0.4912001418098778,
        ],
        output_scale=[
            0.0070314825502323835,
            0.10613806907600574,
            0.1898159060327625,
            0.1966594748073734,
        ],
    )
    target_zcdp = 3.17226586
    noise_multiplier = accounting.noise_multiplier_for_blt(
        blt,
        total_steps=100,
        target_zcdp=target_zcdp,
        min_separation=50,
        max_participations=max_participations,
    )
    self.assertAlmostEqual(noise_multiplier, 1.0, places=6)

  def test_noise_multiplier_for_blt_invalid_inputs(self):
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=[1.0],
        output_scale=[1.0],
    )
    with self.assertRaisesRegex(ValueError, 'target_zcdp must be positive'):
      accounting.noise_multiplier_for_blt(
          blt,
          total_steps=100,
          target_zcdp=0.0,
          min_separation=50,
      )

  @parameterized.parameters(
      dict(
          total_steps=100,
          min_separation=50,
          max_participations=2,
          noise_multiplier=0.0,
      ),
      dict(
          total_steps=0,
          min_separation=50,
          max_participations=2,
          noise_multiplier=1.0,
      ),
      dict(
          total_steps=100,
          min_separation=0,
          max_participations=2,
          noise_multiplier=1.0,
      ),
      dict(
          total_steps=100,
          min_separation=50,
          max_participations=0,
          noise_multiplier=1.0,
      ),
  )
  def test_zcdp_for_blt_invalid_inputs(self, **kwargs):
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=[1.0],
        output_scale=[1.0],
    )
    with self.assertRaisesRegex(ValueError, 'must be positive'):
      accounting.zcdp_for_blt(blt, **kwargs)

  def test_minsep_sensitivity_squared_invalid_coefficients(self):
    with self.assertRaisesRegex(ValueError, 'must be a 1D array'):
      accounting._minsep_sensitivity_squared(
          coefficients=np.array([[1.0], [2.0]]),
          min_separation=1,
          max_participations=1,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='spec_example',
          gradient_noise_multiplier=1.0,
          clipping_noise_multiplier=10.0,
          expected=math.sqrt(1 / 1.01),
      ),
      dict(
          testcase_name='equal',
          gradient_noise_multiplier=2.0,
          clipping_noise_multiplier=2.0,
          expected=math.sqrt(2),
      ),
      dict(
          testcase_name='gradient_inf',
          gradient_noise_multiplier=math.inf,
          clipping_noise_multiplier=5.0,
          expected=5.0,
      ),
      dict(
          testcase_name='clipping_inf',
          gradient_noise_multiplier=3.0,
          clipping_noise_multiplier=math.inf,
          expected=3.0,
      ),
  )
  def test_combine_noise_multipliers(
      self, gradient_noise_multiplier, clipping_noise_multiplier, expected
  ):
    combined = accounting._combine_noise_multipliers(
        gradient_noise_multiplier=gradient_noise_multiplier,
        clipping_noise_multiplier=clipping_noise_multiplier,
    )
    self.assertAlmostEqual(combined, expected)

  @parameterized.parameters(
      dict(gradient_noise_multiplier=0.0, clipping_noise_multiplier=1.0),
      dict(gradient_noise_multiplier=1.0, clipping_noise_multiplier=0.0),
      dict(gradient_noise_multiplier=-1.0, clipping_noise_multiplier=1.0),
      dict(gradient_noise_multiplier=1.0, clipping_noise_multiplier=-1.0),
  )
  def test_combine_noise_multipliers_invalid_inputs(self, **kwargs):
    with self.assertRaisesRegex(ValueError, 'must be positive'):
      accounting._combine_noise_multipliers(**kwargs)

  @parameterized.named_parameters(
      ('with_max_participations', 2),
      ('no_max_participations', None),
  )
  def test_zcdp_for_blt_with_adaptive_clipping(self, max_participations):
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=[
            0.9999999999921251,
            0.9944453083640997,
            0.8985923474607591,
            0.4912001418098778,
        ],
        output_scale=[
            0.0070314825502323835,
            0.10613806907600574,
            0.1898159060327625,
            0.1966594748073734,
        ],
    )
    # Use sigma_g=1.0, sigma_c=10.0 -> combined ~0.9950
    zcdp = accounting.zcdp_for_blt_with_adaptive_clipping(
        blt,
        total_steps=100,
        gradient_noise_multiplier=1.0,
        clipping_noise_multiplier=10.0,
        min_separation=50,
        max_participations=max_participations,
    )
    # Compare against direct call with combined noise multiplier.
    combined_nm = accounting._combine_noise_multipliers(1.0, 10.0)
    expected_zcdp = accounting.zcdp_for_blt(
        blt,
        total_steps=100,
        noise_multiplier=combined_nm,
        min_separation=50,
        max_participations=max_participations,
    )
    self.assertAlmostEqual(zcdp, expected_zcdp)
    # zCDP with adaptive clipping should be higher (worse) than without,
    # since the combined noise multiplier is smaller.
    zcdp_without_clipping = accounting.zcdp_for_blt(
        blt,
        total_steps=100,
        noise_multiplier=1.0,
        min_separation=50,
        max_participations=max_participations,
    )
    self.assertGreater(zcdp, zcdp_without_clipping)

  @parameterized.named_parameters(
      ('with_max_participations', 2),
      ('no_max_participations', None),
  )
  def test_noise_multiplier_for_blt_with_adaptive_clipping(
      self, max_participations
  ):
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=[
            0.9999999999921251,
            0.9944453083640997,
            0.8985923474607591,
            0.4912001418098778,
        ],
        output_scale=[
            0.0070314825502323835,
            0.10613806907600574,
            0.1898159060327625,
            0.1966594748073734,
        ],
    )
    # First compute a zCDP using known sigma_g=1.0, sigma_c=10.0.
    target_zcdp = accounting.zcdp_for_blt_with_adaptive_clipping(
        blt,
        total_steps=100,
        gradient_noise_multiplier=1.0,
        clipping_noise_multiplier=10.0,
        min_separation=50,
        max_participations=max_participations,
    )
    # Then recover sigma_g.
    recovered_sigma_g = (
        accounting.noise_multiplier_for_blt_with_adaptive_clipping(
            blt,
            total_steps=100,
            target_zcdp=target_zcdp,
            clipping_noise_multiplier=10.0,
            min_separation=50,
            max_participations=max_participations,
        )
    )
    self.assertAlmostEqual(recovered_sigma_g, 1.0, places=6)

  def test_noise_multiplier_for_blt_with_adaptive_clipping_sigma_c_too_small(
      self,
  ):
    blt = buffered_toeplitz.BufferedToeplitz.build(
        buf_decay=[1.0],
        output_scale=[1.0],
    )
    # Use a very tight target_zcdp that requires a large combined nm,
    # which will exceed sigma_c.
    with self.assertRaisesRegex(ValueError, 'Cannot achieve target_zcdp'):
      accounting.noise_multiplier_for_blt_with_adaptive_clipping(
          blt,
          total_steps=100,
          target_zcdp=0.001,
          clipping_noise_multiplier=0.1,
          min_separation=50,
      )


if __name__ == '__main__':
  absltest.main()
