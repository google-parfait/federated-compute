from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax_privacy.matrix_factorization import buffered_toeplitz
import numpy as np

from fcp.confidentialcompute.python import accounting

# Required by jnp.float64.
jax.config.update('jax_enable_x64', True)


class AccountingTest(parameterized.TestCase):

  @parameterized.parameters(
      (1.0, 4, 2, 4.0),
      (2.0, 4, 2, 1.0),
      (1.0, 5, 2, 4.5),
      (1.0, 7, 2, 5.5),
      (1.0, 7, 3, 4.5),
  )
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


if __name__ == '__main__':
  absltest.main()
