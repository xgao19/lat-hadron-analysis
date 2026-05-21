from __future__ import annotations

import numpy as np


def evaluate_emff_ratio_2state(
    tsep_values: np.ndarray,
    tau_values: np.ndarray,
    delta_e: float,
    r1: float,
    params: np.ndarray,
) -> np.ndarray:
    """Evaluate the 2-state EMFF ratio model.

    R(tsep, tau) = [M_00 + M_01 e^{-ΔE τ} + M_10 e^{-ΔE (tsep-τ)}
                     + M_11 e^{-ΔE tsep}] / [1 + R1 e^{-ΔE tsep}]

    Args:
        tsep_values: tsep for each data point (n_points,).
        tau_values: tau for each data point (n_points,).
        delta_e: ΔE = E_1 - E_0.
        r1: R_1 = (A_1/A_0)^2.
        params: [M_00] for 1-state, [M_00, M_01, M_10, M_11] for 2-state.

    Returns:
        Model predictions, shape (n_points,).
    """
    tsep = np.asarray(tsep_values, dtype=float)
    tau = np.asarray(tau_values, dtype=float)
    params = np.asarray(params, dtype=float)

    denominator = 1.0 + r1 * np.exp(-delta_e * tsep)

    nparams = len(params)
    if nparams == 1:
        m00 = params[0]
        numerator = np.full_like(tsep, m00, dtype=float)
    elif nparams == 4:
        m00, m01, m10, m11 = params
        numerator = (
            m00
            + m01 * np.exp(-delta_e * tau)
            + m10 * np.exp(-delta_e * (tsep - tau))
            + m11 * np.exp(-delta_e * tsep)
        )
    else:
        raise ValueError(f"params must have 1 or 4 elements, got {nparams}")

    return numerator / denominator


def evaluate_emff_summed_ratio(
    tsep_values: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """Evaluate the summation method model.

    S(tsep) = tsep * M_00 + B

    Args:
        tsep_values: tsep values (n_points,).
        params: [M_00] or [M_00, B].

    Returns:
        Model predictions, shape (n_points,).
    """
    tsep = np.asarray(tsep_values, dtype=float)
    params = np.asarray(params, dtype=float)

    if len(params) == 1:
        m00 = params[0]
        return tsep * m00
    elif len(params) == 2:
        m00, b = params
        return tsep * m00 + b
    else:
        raise ValueError(f"params must have 1 or 2 elements, got {len(params)}")


def evaluate_emff_plateau(
    n_points: int,
    params: np.ndarray,
) -> np.ndarray:
    """Evaluate the plateau method model.

    R(tsep, tau) = M_00 (constant)

    Args:
        n_points: number of data points.
        params: [M_00].

    Returns:
        Constant array of shape (n_points,).
    """
    params = np.asarray(params, dtype=float)
    if len(params) != 1:
        raise ValueError(f"params must have exactly 1 element, got {len(params)}")
    return np.full(n_points, params[0], dtype=float)


def compute_tau_range_for_tsep(
    tsep: int,
    tau_min: int,
    tau_offset: int,
) -> np.ndarray:
    """Compute actual tau values for a given tsep.

    Args:
        tsep: source-sink separation.
        tau_min: minimum tau (inclusive).
        tau_offset: offset from tsep for max tau. -1 means tsep-1.

    Returns:
        Array of tau values.
    """
    tau_max = tsep + tau_offset if tau_offset < 0 else tau_offset
    return np.arange(tau_min, tau_max + 1, dtype=int)
