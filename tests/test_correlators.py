import numpy as np

from lqcd_analysis.correlators import effective_mass, jackknife_mean, jackknife_samples


def test_effective_mass_constant_decay():
    correlator = np.exp(-0.5 * np.arange(6))
    masses = effective_mass(correlator)
    assert np.allclose(masses, 0.5)


def test_jackknife_samples_shape():
    data = np.arange(12, dtype=float).reshape(4, 3)
    samples = jackknife_samples(data)
    assert samples.shape == (4, 3)


def test_jackknife_mean_matches_naive_mean():
    data = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]
    )
    mean, error = jackknife_mean(data)
    assert np.allclose(mean, np.mean(data, axis=0))
    assert np.all(error > 0)

