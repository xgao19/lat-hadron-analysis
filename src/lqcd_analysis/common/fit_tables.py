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
    selected_tmin_range: tuple[int, int]
    selected_params_mean: tuple[float, ...]
    selected_params_err: tuple[float, ...]


def parse_fit_table_scan(table: np.ndarray, nstates: int) -> FitTableScanData:
    rows = np.atleast_2d(np.asarray(table, dtype=float))
    has_fallback_column = rows.shape[1] >= 10 + 4 * nstates
    selection_flag_column = 9 if has_fallback_column else 8
    chi2_column = 7
    amp_mean_start = selection_flag_column + 1
    amp_err_start = amp_mean_start + nstates
    energy_mean_start = amp_err_start + nstates
    energy_err_start = energy_mean_start + nstates
    selected_mask = rows[:, selection_flag_column] > 0.5
    selected_rows = rows[selected_mask]
    if len(selected_rows) == 0:
        raise ValueError("fit table does not contain any selected tmin rows")
    selected_row = selected_rows[np.nanargmin(selected_rows[:, chi2_column])]
    amp_means = tuple(selected_row[amp_mean_start : amp_mean_start + nstates])
    amp_errs = tuple(selected_row[amp_err_start : amp_err_start + nstates])
    energy_means = tuple(selected_row[energy_mean_start : energy_mean_start + nstates])
    energy_errs = tuple(selected_row[energy_err_start : energy_err_start + nstates])
    selected_params_mean = amp_means + energy_means
    selected_params_err = amp_errs + energy_errs
    selected_tmin_range = (int(selected_rows[0, 0]), int(selected_rows[-1, 0]))
    return FitTableScanData(
        tmins=rows[:, 0],
        tmax=int(rows[0, 1]),
        amplitude_values=rows[:, amp_mean_start : amp_mean_start + nstates],
        amplitude_errs=rows[:, amp_err_start : amp_err_start + nstates],
        energy_values=rows[:, energy_mean_start : energy_mean_start + nstates],
        energy_errs=rows[:, energy_err_start : energy_err_start + nstates],
        selected_tmin_range=selected_tmin_range,
        selected_params_mean=selected_params_mean,
        selected_params_err=selected_params_err,
    )
