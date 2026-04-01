from __future__ import annotations

import numpy as np


def effective_mass(correlator: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Compute the log effective mass from a positive correlator."""
    values = np.asarray(correlator, dtype=float)
    if values.ndim != 1:
        raise ValueError("correlator must be one-dimensional")
    if len(values) < 2:
        raise ValueError("correlator must contain at least two time slices")
    if np.any(values <= 0):
        raise ValueError("effective mass requires strictly positive correlator values")
    return np.log(values[:-1] / values[1:]) / spacing


def jackknife_samples(data: np.ndarray) -> np.ndarray:
    """Build leave-one-out jackknife samples along the first axis."""
    values = np.asarray(data, dtype=float)
    if values.ndim == 0:
        raise ValueError("data must have at least one dimension")
    n_cfg = values.shape[0]
    if n_cfg < 2:
        raise ValueError("jackknife requires at least two samples")

    total = np.sum(values, axis=0)
    return (total - values) / (n_cfg - 1)


def jackknife_mean(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and jackknife error estimate."""
    samples = jackknife_samples(data)
    mean = np.mean(samples, axis=0)
    fluctuations = samples - mean
    error = np.sqrt((len(samples) - 1) * np.mean(fluctuations**2, axis=0))
    return mean, error

