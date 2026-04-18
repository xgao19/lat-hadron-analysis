from __future__ import annotations

import numpy as np

from . import bootstrap


def _apply_signed_fold(correlators: np.ndarray, nt: int, sign: float) -> np.ndarray:
    values = np.asarray(correlators, dtype=float)
    if values.ndim != 2 or values.shape[1] != nt:
        raise ValueError("correlators must have shape (n_cfg, Nt)")

    folded_extent = nt // 2 + 1
    folded = values[:, :folded_extent].copy()
    for t in range(1, folded_extent):
        partner = (nt - t) % nt
        if partner == t:
            continue
        folded[:, t] = 0.5 * (values[:, t] + sign * values[:, partner])
    return folded


def apply_fold_t(correlators: np.ndarray, nt: int, fold_t: str) -> np.ndarray:
    """Apply no fold, periodic fold, or antiperiodic fold.

    The fold_t mode is used to exploit periodicity/antiperiodicity in the
    temporal direction by combining data at t and Nt-t to reduce noise.
    """
    normalized = fold_t.strip().lower()
    if normalized == "none":
        return np.asarray(correlators, dtype=float)
    if normalized == "periodic":
        return _apply_signed_fold(correlators, nt, 1.0)
    if normalized == "antiperiodic":
        return _apply_signed_fold(correlators, nt, -1.0)
    raise ValueError(f"unsupported fold_t value: {fold_t}")


def bin_correlators(correlators: np.ndarray, binsize: int = 1) -> np.ndarray:
    """Bin correlator configurations along the first axis.

    This is a specialized wrapper around `bootstrap.bin_samples` that ensures
    the input is two-dimensional (n_cfg × Nt).
    """
    values = np.asarray(correlators, dtype=float)
    if values.ndim != 2:
        raise ValueError("correlators must be two-dimensional")
    return bootstrap.bin_samples(values, binsize=binsize)


def bootstrap_correlator_means(
    correlators: np.ndarray,
    n_samples: int | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Compute bootstrap means for correlator data.

    This is a specialized wrapper around `bootstrap.bootstrap_means` that ensures
    the input is two-dimensional (n_cfg × Nt).
    """
    values = np.asarray(correlators, dtype=float)
    if values.ndim != 2:
        raise ValueError("correlators must be two-dimensional")

    return bootstrap.bootstrap_means(
        values,
        n_boot=n_samples,
        draw_size=sample_size,
        seed=seed,
    )


def robust_mean_and_error(samples: np.ndarray) -> tuple[float, float]:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return np.nan, np.nan

    p16, p84 = np.percentile(values, [16.0, 84.0])
    mean = float(0.5 * (p16 + p84))
    err = float(0.5 * (p84 - p16))
    return mean, err
