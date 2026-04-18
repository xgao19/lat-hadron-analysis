from __future__ import annotations

import numpy as np


def bootstrap_indices(
    n_samples: int,
    n_draws: int | None = None,
    *,
    seed: int | None = None,
    n_boot: int | None = None,
) -> np.ndarray:
    """Generate bootstrap resampling indices.

    Args:
        n_samples: Number of original samples (configurations).
        n_draws: Number of bootstrap draws per sample. If None, defaults to n_samples.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, n_draws) containing indices in [0, n_samples).

    Raises:
        ValueError: If n_samples < 2.
    """
    if n_samples < 2:
        raise ValueError("bootstrap requires at least two samples")
    n_draws = n_samples if n_draws is None else n_draws
    n_boot = n_samples if n_boot is None else n_boot
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_samples, size=(n_boot, n_draws))


def bootstrap_means(
    data: np.ndarray,
    indices: np.ndarray | None = None,
    *,
    n_boot: int | None = None,
    draw_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Compute bootstrap means for arbitrary numeric data.

    Args:
        data: Input data array of shape (n_orig, ...).
        indices: Precomputed bootstrap indices of shape (n_boot, draw_size).
            If provided, n_boot, draw_size, and seed are ignored.
        n_boot: Number of bootstrap samples to generate.
            If None and indices not provided, defaults to n_orig.
        draw_size: Number of draws per bootstrap sample.
            If None and indices not provided, defaults to n_orig.
        seed: Random seed for index generation (ignored if indices provided).

    Returns:
        Array of shape (n_boot, ...) containing bootstrap sample means.

    Raises:
        ValueError: If data has wrong shape or insufficient samples.
    """
    data_array = np.asarray(data)
    if data_array.ndim < 1:
        raise ValueError("data must have at least one dimension")
    n_orig = data_array.shape[0]

    if indices is None:
        n_boot = n_orig if n_boot is None else n_boot
        indices = bootstrap_indices(n_orig, draw_size, seed=seed, n_boot=n_boot)
    else:
        indices = np.asarray(indices, dtype=int)
        if indices.ndim != 2:
            raise ValueError("indices must be two-dimensional")
        if n_boot is not None and n_boot != indices.shape[0]:
            raise ValueError(
                f"n_boot={n_boot} does not match indices shape {indices.shape[0]}"
            )

    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 2:
        raise ValueError("indices must be two-dimensional")
    selected = data_array[indices]
    return np.mean(selected, axis=1)


def compute_bootstrap_covariance(bootstrap_means: np.ndarray) -> np.ndarray:
    """Compute covariance matrix from bootstrap samples.

    Args:
        bootstrap_means: Array of shape (n_boot, n_vars) containing
            bootstrap sample means.

    Returns:
        Covariance matrix of shape (n_vars, n_vars).

    Raises:
        ValueError: If bootstrap_means is not 2D.
    """
    values = np.asarray(bootstrap_means, dtype=float)
    if values.ndim != 2:
        raise ValueError("bootstrap_means must be a 2D array")
    covariance = np.cov(values, rowvar=False, ddof=1)
    return np.atleast_2d(np.asarray(covariance, dtype=float))


def shrink_covariance_to_diagonal(
    covariance: np.ndarray,
    shrinkage_lambda: float,
) -> np.ndarray:
    """Shrink covariance matrix toward its diagonal.

    The shrunk covariance is defined as:
        C_shrunk = (1 - λ) * C + λ * diag(C)

    Args:
        covariance: Input covariance matrix.
        shrinkage_lambda: Shrinkage parameter in [0, 1].

    Returns:
        Shrunk covariance matrix.
    """
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    diagonal = np.diag(np.diag(covariance))
    return (1.0 - shrinkage_lambda) * covariance + shrinkage_lambda * diagonal


def bin_samples(
    samples: np.ndarray,
    binsize: int = 1,
) -> np.ndarray:
    """Bin samples along the first axis.

    Args:
        samples: Input array of shape (n_samples, ...).
        binsize: Size of each bin (must be positive).

    Returns:
        Binned array of shape (n_bins, ...) where n_bins = n_samples // binsize.

    Raises:
        ValueError: If binsize < 1 or binning leaves fewer than two bins.
    """
    samples_array = np.asarray(samples)
    if samples_array.ndim < 1:
        raise ValueError("samples must have at least one dimension")
    if binsize < 1:
        raise ValueError("binsize must be positive")
    if binsize == 1:
        return samples_array.copy()

    n_samples = samples_array.shape[0]
    n_bins = n_samples // binsize
    if n_bins < 2:
        raise ValueError("binning leaves fewer than two bins")

    trimmed = samples_array[: n_bins * binsize]
    # Reshape to (n_bins, binsize, ...) and average over bin axis
    new_shape = (n_bins, binsize) + samples_array.shape[1:]
    return trimmed.reshape(new_shape).mean(axis=1)

