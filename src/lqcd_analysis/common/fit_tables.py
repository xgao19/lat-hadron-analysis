from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitTableLayout:
    has_fallback_column: bool
    plateau_flag_column: int
    amp_mean_start: int
    amp_err_start: int
    energy_mean_start: int
    energy_err_start: int


@dataclass(frozen=True)
class FitTableScanData:
    table: np.ndarray
    tmins: np.ndarray
    tmax: int
    amplitude_values: np.ndarray
    amplitude_errs: np.ndarray
    energy_values: np.ndarray
    energy_errs: np.ndarray
    plateau_mask: np.ndarray
    plateau_tmin_range: tuple[int, int]
    plateau_start_tmin: int
    fallback_row: np.ndarray
    fallback_params_mean: tuple[float, ...]
    fallback_params_err: tuple[float, ...]


def get_fit_table_layout(nstates: int, n_columns: int) -> FitTableLayout:
    has_fallback_column = n_columns >= 10 + 4 * nstates
    plateau_flag_column = 9 if has_fallback_column else 8
    amp_mean_start = plateau_flag_column + 1
    amp_err_start = amp_mean_start + nstates
    energy_mean_start = amp_err_start + nstates
    energy_err_start = energy_mean_start + nstates
    return FitTableLayout(
        has_fallback_column=has_fallback_column,
        plateau_flag_column=plateau_flag_column,
        amp_mean_start=amp_mean_start,
        amp_err_start=amp_err_start,
        energy_mean_start=energy_mean_start,
        energy_err_start=energy_err_start,
    )


def decode_fit_table_parameter_columns(
    raw: np.ndarray,
    nstates: int,
    layout: FitTableLayout,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    amp_means = tuple(raw[layout.amp_mean_start : layout.amp_mean_start + nstates])
    amp_errs = tuple(raw[layout.amp_err_start : layout.amp_err_start + nstates])
    energy_means = tuple(raw[layout.energy_mean_start : layout.energy_mean_start + nstates])
    energy_errs = tuple(raw[layout.energy_err_start : layout.energy_err_start + nstates])
    return amp_means + energy_means, amp_errs + energy_errs


def parse_fit_table_scan(table: np.ndarray, nstates: int) -> FitTableScanData:
    rows = np.atleast_2d(np.asarray(table, dtype=float))
    layout = get_fit_table_layout(nstates, rows.shape[1])
    plateau_mask = rows[:, layout.plateau_flag_column] > 0.5
    plateau_rows = rows[plateau_mask]
    fallback_row = plateau_rows[len(plateau_rows) // 2] if len(plateau_rows) else rows[len(rows) // 2]
    fallback_params_mean, fallback_params_err = decode_fit_table_parameter_columns(fallback_row, nstates, layout)
    plateau_tmin_range = (
        int(plateau_rows[0, 0]) if len(plateau_rows) else int(fallback_row[0]),
        int(plateau_rows[-1, 0]) if len(plateau_rows) else int(fallback_row[0]),
    )
    return FitTableScanData(
        table=rows,
        tmins=rows[:, 0],
        tmax=int(rows[0, 1]),
        amplitude_values=rows[:, layout.amp_mean_start : layout.amp_mean_start + nstates],
        amplitude_errs=rows[:, layout.amp_err_start : layout.amp_err_start + nstates],
        energy_values=rows[:, layout.energy_mean_start : layout.energy_mean_start + nstates],
        energy_errs=rows[:, layout.energy_err_start : layout.energy_err_start + nstates],
        plateau_mask=plateau_mask,
        plateau_tmin_range=plateau_tmin_range,
        plateau_start_tmin=plateau_tmin_range[0],
        fallback_row=fallback_row,
        fallback_params_mean=fallback_params_mean,
        fallback_params_err=fallback_params_err,
    )
