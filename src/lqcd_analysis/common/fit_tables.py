from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitTableScanData:
    tmins: np.ndarray
    tmax: int
    amplitude_values: np.ndarray
    amplitude_errs: np.ndarray
    energy_values: np.ndarray
    energy_errs: np.ndarray
    plateau_tmin_range: tuple[int, int]
    fallback_params_mean: tuple[float, ...]
    fallback_params_err: tuple[float, ...]


def parse_fit_table_scan(table: np.ndarray, nstates: int) -> FitTableScanData:
    rows = np.atleast_2d(np.asarray(table, dtype=float))
    has_fallback_column = rows.shape[1] >= 10 + 4 * nstates
    plateau_flag_column = 9 if has_fallback_column else 8
    amp_mean_start = plateau_flag_column + 1
    amp_err_start = amp_mean_start + nstates
    energy_mean_start = amp_err_start + nstates
    energy_err_start = energy_mean_start + nstates
    plateau_mask = rows[:, plateau_flag_column] > 0.5
    plateau_rows = rows[plateau_mask]
    fallback_row = plateau_rows[len(plateau_rows) // 2] if len(plateau_rows) else rows[len(rows) // 2]
    amp_means = tuple(fallback_row[amp_mean_start : amp_mean_start + nstates])
    amp_errs = tuple(fallback_row[amp_err_start : amp_err_start + nstates])
    energy_means = tuple(fallback_row[energy_mean_start : energy_mean_start + nstates])
    energy_errs = tuple(fallback_row[energy_err_start : energy_err_start + nstates])
    fallback_params_mean = amp_means + energy_means
    fallback_params_err = amp_errs + energy_errs
    plateau_tmin_range = (
        int(plateau_rows[0, 0]) if len(plateau_rows) else int(fallback_row[0]),
        int(plateau_rows[-1, 0]) if len(plateau_rows) else int(fallback_row[0]),
    )
    return FitTableScanData(
        tmins=rows[:, 0],
        tmax=int(rows[0, 1]),
        amplitude_values=rows[:, amp_mean_start : amp_mean_start + nstates],
        amplitude_errs=rows[:, amp_err_start : amp_err_start + nstates],
        energy_values=rows[:, energy_mean_start : energy_mean_start + nstates],
        energy_errs=rows[:, energy_err_start : energy_err_start + nstates],
        plateau_tmin_range=plateau_tmin_range,
        fallback_params_mean=fallback_params_mean,
        fallback_params_err=fallback_params_err,
    )
