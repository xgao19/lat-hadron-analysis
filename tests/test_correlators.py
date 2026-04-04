import unittest

import numpy as np

from lqcd_analysis.correlators import effective_mass, jackknife_mean, jackknife_samples


class CorrelatorTests(unittest.TestCase):
    def test_effective_mass_constant_decay(self) -> None:
        correlator = np.exp(-0.5 * np.arange(6))
        masses = effective_mass(correlator)
        self.assertTrue(np.allclose(masses, 0.5))

    def test_jackknife_samples_shape(self) -> None:
        data = np.arange(12, dtype=float).reshape(4, 3)
        samples = jackknife_samples(data)
        self.assertEqual(samples.shape, (4, 3))

    def test_jackknife_mean_matches_naive_mean(self) -> None:
        data = np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ]
        )
        mean, error = jackknife_mean(data)
        self.assertTrue(np.allclose(mean, np.mean(data, axis=0)))
        self.assertTrue(np.all(error > 0))


if __name__ == "__main__":
    unittest.main()
