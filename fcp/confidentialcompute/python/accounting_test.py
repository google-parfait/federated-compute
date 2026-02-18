from absl.testing import absltest
from absl.testing import parameterized

from fcp.confidentialcompute.python import accounting


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


if __name__ == "__main__":
  absltest.main()
