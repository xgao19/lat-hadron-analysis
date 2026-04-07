from __future__ import annotations

import numpy as np


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value}")


def parse_fold_t(entries: dict[str, list[str]]) -> str:
    """Parse the fold_t option.

    The fold_t mode is used to exploit periodic or antiperiodic boundary
    conditions by averaging the data at t and Nt-t, which often improves the
    signal quality of two-point correlators.
    """
    if "fold_t" not in entries:
        return "none"

    value = entries["fold_t"][0].strip().lower()
    if value in {"true", "periodic"}:
        return "periodic"
    if value in {"false", "none"}:
        return "none"
    if value == "antiperiodic":
        return "antiperiodic"
    raise ValueError("fold_t must be one of: false, none, true, periodic, antiperiodic")


def apply_periodic_fold(correlators: np.ndarray, nt: int) -> np.ndarray:
    """Average C(t) with C(Nt-t) for periodic temporal boundary conditions."""
    values = np.asarray(correlators, dtype=float)
    if values.ndim != 2 or values.shape[1] != nt:
        raise ValueError("correlators must have shape (n_cfg, Nt)")

    folded_extent = nt // 2 + 1
    folded = values[:, :folded_extent].copy()
    for t in range(1, folded_extent):
        partner = (nt - t) % nt
        if partner == t:
            continue
        folded[:, t] = 0.5 * (values[:, t] + values[:, partner])
    return folded


def apply_antiperiodic_fold(correlators: np.ndarray, nt: int) -> np.ndarray:
    """Average C(t) with -C(Nt-t) for antiperiodic temporal boundary conditions."""
    values = np.asarray(correlators, dtype=float)
    if values.ndim != 2 or values.shape[1] != nt:
        raise ValueError("correlators must have shape (n_cfg, Nt)")

    folded_extent = nt // 2 + 1
    folded = values[:, :folded_extent].copy()
    for t in range(1, folded_extent):
        partner = (nt - t) % nt
        if partner == t:
            continue
        folded[:, t] = 0.5 * (values[:, t] - values[:, partner])
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
        return apply_periodic_fold(correlators, nt)
    if normalized == "antiperiodic":
        return apply_antiperiodic_fold(correlators, nt)
    raise ValueError(f"unsupported fold_t value: {fold_t}")


def bin_correlators(correlators: np.ndarray, binsize: int = 1) -> np.ndarray:
    values = np.asarray(correlators, dtype=float)
    if values.ndim != 2:
        raise ValueError("correlators must be two-dimensional")
    if binsize < 1:
        raise ValueError("binsize must be positive")
    if binsize == 1:
        return values.copy()

    n_cfg = values.shape[0]
    n_bins = n_cfg // binsize
    if n_bins < 2:
        raise ValueError("binning leaves fewer than two bins")
    trimmed = values[: n_bins * binsize]
    return trimmed.reshape(n_bins, binsize, values.shape[1]).mean(axis=1)


def bootstrap_correlator_means(
    correlators: np.ndarray,
    n_samples: int | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    values = np.asarray(correlators, dtype=float)
    if values.ndim != 2:
        raise ValueError("correlators must be two-dimensional")

    n_cfg = values.shape[0]
    if n_cfg < 2:
        raise ValueError("bootstrap requires at least two samples")

    n_boot = n_cfg if n_samples is None else n_samples
    draw_size = n_cfg if sample_size is None else sample_size
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_cfg, size=(n_boot, draw_size))
    return values[indices].mean(axis=1)


def robust_mean_and_error(samples: np.ndarray) -> tuple[float, float]:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return np.nan, np.nan

    p16, p84 = np.percentile(values, [16.0, 84.0])
    mean = float(0.5 * (p16 + p84))
    err = float(0.5 * (p84 - p16))
    return mean, err
