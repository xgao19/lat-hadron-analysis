from __future__ import annotations

import numpy as np


def normalize_tmdwf_operator(gm: str) -> str:
    normalized = gm.strip().upper()
    if normalized not in {"T5", "Z5"}:
        raise ValueError(f"unsupported TMDWF operator label: {gm}")
    return normalized


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


def evaluate_tmdwf_numerator_t5(
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


def evaluate_tmdwf_numerator_z5(
    times: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    matrix_elements: np.ndarray,
    nt: int,
    *,
    pz: int,
    ns: int,
) -> np.ndarray:
    t = np.asarray(times, dtype=float)[:, None]
    a = np.asarray(amplitudes, dtype=float)
    e = np.asarray(energies, dtype=float)
    m = np.asarray(matrix_elements, dtype=float)
    momentum = 2.0 * np.pi * float(pz) / float(ns)
    prefactor = 0.5 * np.sqrt(a * 2.0 * e) * (momentum / e) * m
    kernel = np.exp(-e[None, :] * t) - np.exp(-e[None, :] * (nt - t))
    return np.sum(prefactor[None, :] * kernel, axis=1)


def evaluate_tmdwf_numerator(
    times: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    matrix_elements: np.ndarray,
    nt: int,
    *,
    gm: str,
    pz: int,
    ns: int,
) -> np.ndarray:
    operator = normalize_tmdwf_operator(gm)
    if operator == "T5":
        return evaluate_tmdwf_numerator_t5(times, amplitudes, energies, matrix_elements, nt)
    return evaluate_tmdwf_numerator_z5(times, amplitudes, energies, matrix_elements, nt, pz=pz, ns=ns)


def evaluate_tmdwf_ratio(
    times: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    matrix_elements: np.ndarray,
    nt: int,
    *,
    gm: str,
    pz: int,
    ns: int,
) -> np.ndarray:
    denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
    numerator = evaluate_tmdwf_numerator(times, amplitudes, energies, matrix_elements, nt, gm=gm, pz=pz, ns=ns)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator != 0.0)
