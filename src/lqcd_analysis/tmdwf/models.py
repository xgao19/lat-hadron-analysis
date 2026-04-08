from __future__ import annotations

import numpy as np


def evaluate_two_point_symmetric(
    times: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    nt: int,
) -> np.ndarray:
    t = np.asarray(times, dtype=float)[:, None]
    a = np.asarray(amplitudes, dtype=float)[None, :]
    e = np.asarray(energies, dtype=float)[None, :]
    return np.sum(a * (np.exp(-e * t) + np.exp(-e * (nt - t))), axis=1)


def evaluate_tmdwf_numerator_gamma_t_gamma5(
    times: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    matrix_elements: np.ndarray,
    nt: int,
) -> np.ndarray:
    t = np.asarray(times, dtype=float)[:, None]
    a = np.asarray(amplitudes, dtype=float)
    e = np.asarray(energies, dtype=float)
    m = np.asarray(matrix_elements, dtype=float)
    prefactor = 0.5 * np.sqrt(a * 2.0 * e) * m
    kernel = np.exp(-e[None, :] * t) - np.exp(-e[None, :] * (nt - t))
    return np.sum(prefactor[None, :] * kernel, axis=1)


def evaluate_tmdwf_ratio_gamma_t_gamma5(
    times: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    matrix_elements: np.ndarray,
    nt: int,
) -> np.ndarray:
    denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
    numerator = evaluate_tmdwf_numerator_gamma_t_gamma5(times, amplitudes, energies, matrix_elements, nt)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator != 0.0)
