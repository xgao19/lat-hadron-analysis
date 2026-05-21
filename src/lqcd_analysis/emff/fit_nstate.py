from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from ..common.bootstrap import (
    bin_samples,
    bootstrap_indices as common_bootstrap_indices,
    bootstrap_means,
)
from ..common.constants import HBAR_C_GEV_FM, MIN_POSITIVE
from ..common.utils import robust_mean_and_error
from ..tmdwf.io import expand_template, resolve_two_point_fit_reference
from .io import (
    EMFFNStateInput,
    load_emff_c2pt_correlator,
    load_emff_correlator,
    parse_emff_fit_input,
    resolve_emff_h5_path,
)
from .models import (
    compute_tau_range_for_tsep,
    evaluate_emff_plateau,
    evaluate_emff_ratio_2state,
    evaluate_emff_summed_ratio,
)


@dataclass(frozen=True)
class EMFFFitterResult:
    params: np.ndarray
    chi2: float
    chi2_dof: float
    pvalue: float
    success: bool
    message: str


@dataclass(frozen=True)
class _PreparedFitData:
    data_samples: np.ndarray  # (n_boot, n_points)
    sigma: np.ndarray         # (n_points,)


@dataclass(frozen=True)
class EMFFOutputRecord:
    qx: int
    qy: int
    qz: int
    nstates: int
    fit_method: str
    fit_result: EMFFFitterResult
    sample_params: np.ndarray  # (n_boot, n_params)
    delta_e: float
    r1: float
    tsep_fit_values: tuple[int, ...]
    tau_range_used: tuple[int, int]


def _hadron_energy_from_dispersion(
    momentum: tuple[int, int, int],
    *,
    ns: int,
    lattice_spacing_fm: float,
    hadron_mass_gev: float,
) -> float:
    momentum_unit_gev = 2.0 * np.pi * HBAR_C_GEV_FM / (lattice_spacing_fm * ns)
    momentum_squared = sum(component**2 for component in momentum) * momentum_unit_gev**2
    return float(np.sqrt(hadron_mass_gev**2 + momentum_squared))


def _compute_emff_ratio(
    c3pt_tau: np.ndarray,
    c2pt_initial: np.ndarray,
    c2pt_final: np.ndarray,
    *,
    tsep: int,
    energy_initial: float,
    energy_final: float,
) -> np.ndarray:
    """Compute the pion EMFF ratio from Eq. (6) of arXiv:2102.06047."""
    tau_values = np.arange(tsep + 1)
    c2_i_tsep = c2pt_initial[:, tsep]
    c2_f_tsep = c2pt_final[:, tsep]

    sqrt_numerator = (
        c2pt_final[:, tsep - tau_values]
        * c2pt_initial[:, tau_values]
        * c2_i_tsep[:, None]
    )
    sqrt_denominator = (
        c2pt_initial[:, tsep - tau_values]
        * c2pt_final[:, tau_values]
        * c2_f_tsep[:, None]
    )
    sqrt_factor = np.divide(
        sqrt_numerator,
        sqrt_denominator,
        out=np.full_like(sqrt_numerator, np.nan + 0.0j),
        where=sqrt_denominator != 0.0,
    )
    c3_over_c2 = np.divide(
        c3pt_tau.T,
        c2_i_tsep[:, None],
        out=np.full_like(c3pt_tau.T, np.nan + 0.0j),
        where=c2_i_tsep[:, None] != 0.0,
    )
    energy_factor = 2.0 * np.sqrt(energy_final * energy_initial) / (
        energy_final + energy_initial
    )
    return energy_factor * c3_over_c2 * np.sqrt(sqrt_factor)


def _q_tag(qx: int, qy: int, qz: int) -> str:
    return f"q{qx:+d}_{qy:+d}_{qz:+d}".replace("+", "p").replace("-", "m")


def _transverse_orbit_members(
    qx: int,
    qy: int,
    qz: int,
    *,
    qx_values: set[int],
    qy_values: set[int],
) -> tuple[tuple[int, int, int], ...]:
    members: set[tuple[int, int, int]] = set()
    for ax, ay in {(abs(qx), abs(qy)), (abs(qy), abs(qx))}:
        for sx in ({-1, 1} if ax else {1}):
            for sy in ({-1, 1} if ay else {1}):
                member_qx, member_qy = sx * ax, sy * ay
                if member_qx in qx_values and member_qy in qy_values:
                    members.add((member_qx, member_qy, qz))
    return tuple(sorted(members))


def _build_q_groups(
    qxlist: tuple[int, ...],
    qylist: tuple[int, ...],
    qzlist: tuple[int, ...],
    *,
    average_transverse_orbits: bool,
    final_momentum: tuple[int, int, int],
) -> dict[tuple[int, int, int], tuple[tuple[int, int, int], ...]]:
    if not average_transverse_orbits:
        return {
            (qx, qy, qz): ((qx, qy, qz),)
            for qx in qxlist
            for qy in qylist
            for qz in qzlist
        }

    if final_momentum[0] != 0 or final_momentum[1] != 0:
        raise ValueError(
            "average_transverse_orbits currently requires final momentum "
            "to have no transverse component"
        )

    qx_values = set(qxlist)
    qy_values = set(qylist)
    transverse_pairs = sorted(
        {(max(abs(qx), abs(qy)), min(abs(qx), abs(qy))) for qx in qxlist for qy in qylist}
    )
    groups: dict[tuple[int, int, int], tuple[tuple[int, int, int], ...]] = {}
    seen_members: set[tuple[int, int, int]] = set()
    for qz in qzlist:
        for qx_abs, qy_abs in transverse_pairs:
            canonical = (qx_abs, qy_abs, qz)
            members = _transverse_orbit_members(
                qx_abs,
                qy_abs,
                qz,
                qx_values=qx_values,
                qy_values=qy_values,
            )
            if not members or any(member in seen_members for member in members):
                continue
            groups[canonical] = members
            seen_members.update(members)
    return groups


def summarize_parameter_samples(
    samples: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    means: list[float] = []
    errors: list[float] = []
    for column in range(samples.shape[1]):
        valid = samples[:, column][np.isfinite(samples[:, column])]
        mean, err = robust_mean_and_error(valid)
        means.append(mean)
        errors.append(err)
    return tuple(means), tuple(errors)


def sanitize_token(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"_", "-"} else "_" for char in value
    )


def _build_fit_data_2state(
    ratio_by_tsep: dict[int, np.ndarray],
    tau_range: tuple[int, int],
    tsep_fit_list: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build flattened (tsep, tau) fit data for 2-state or plateau fits.

    Args:
        ratio_by_tsep: dict mapping tsep -> bootstrap samples (n_boot, n_tau).
        tau_range: (tau_min, tau_offset) for per-tsep tau window.
        tsep_fit_list: tsep values to include.

    Returns:
        tsep_array: (n_points,) tsep value at each point.
        tau_array: (n_points,) tau value at each point.
        data_samples: (n_boot, n_points) ratio data.
    """
    tau_min, tau_offset = tau_range
    n_boot = next(iter(ratio_by_tsep.values())).shape[0]

    tsep_list: list[int] = []
    tau_list: list[int] = []
    sample_blocks: list[np.ndarray] = []

    for tsep in tsep_fit_list:
        boot_samples = ratio_by_tsep[tsep]  # (n_boot, n_tau_total)
        tau_vals = compute_tau_range_for_tsep(tsep, tau_min, tau_offset)
        # Map tau_vals to column indices (row 0 = tau=0)
        col_indices = tau_vals  # tau values match row indices directly
        valid = (col_indices >= 0) & (col_indices < boot_samples.shape[1])
        if not np.all(valid):
            raise ValueError(
                f"tau range {tau_range} for tsep={tsep} produces out-of-bounds indices"
            )
        for tau in tau_vals:
            tsep_list.append(tsep)
            tau_list.append(int(tau))
        sample_blocks.append(boot_samples[:, col_indices])

    return (
        np.array(tsep_list, dtype=int),
        np.array(tau_list, dtype=int),
        np.concatenate(sample_blocks, axis=1),
    )


def _build_fit_data_summation(
    ratio_by_tsep: dict[int, np.ndarray],
    tau_range: tuple[int, int],
    tsep_fit_list: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build summed ratio data for summation fit.

    Returns:
        tsep_array: (n_tsep,) tsep values.
        summed_samples: (n_boot, n_tsep) summed ratio for each tsep.
    """
    tau_min, tau_offset = tau_range

    tsep_array: list[int] = []
    summed_blocks: list[np.ndarray] = []

    for tsep in tsep_fit_list:
        boot_samples = ratio_by_tsep[tsep]  # (n_boot, n_tau_total)
        tau_vals = compute_tau_range_for_tsep(tsep, tau_min, tau_offset)
        col_indices = tau_vals
        valid = (col_indices >= 0) & (col_indices < boot_samples.shape[1])
        if not np.all(valid):
            raise ValueError(
                f"tau range {tau_range} for tsep={tsep} produces out-of-bounds indices"
            )
        summed = np.sum(boot_samples[:, col_indices], axis=1)  # (n_boot,)
        tsep_array.append(tsep)
        summed_blocks.append(summed)

    return (
        np.array(tsep_array, dtype=int),
        np.column_stack(summed_blocks),  # (n_boot, n_tsep)
    )


def _prepare_fit_data(
    data_samples: np.ndarray,
    *,
    component: str = "real",
) -> _PreparedFitData:
    """Compute sigma from bootstrap variance, extracting real or imag component."""
    if component == "real":
        values = np.asarray(np.real(data_samples), dtype=float)
    elif component == "imag":
        values = np.asarray(np.imag(data_samples), dtype=float)
    else:
        raise ValueError(f"component must be 'real' or 'imag', got {component}")
    sigma = np.nanstd(values, axis=0, ddof=1)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, MIN_POSITIVE)
    return _PreparedFitData(data_samples=values, sigma=sigma)


def fit_emff_2state(
    ratio_by_tsep: dict[int, np.ndarray],
    delta_e: float,
    r1: float,
    tau_range: tuple[int, int],
    tsep_fit_list: list[int],
    nstates: int,
) -> tuple[EMFFFitterResult, np.ndarray]:
    """Fit the 2-state EMFF ratio model.

    Args:
        ratio_by_tsep: dict tsep -> bootstrap samples (n_boot, n_tau).
        delta_e: ΔE = E_1 - E_0.
        r1: R_1 = (A_1/A_0)^2.
        tau_range: (tau_min, tau_offset).
        tsep_fit_list: tsep values to include in the fit.
        nstates: 1 or 2.

    Returns:
        (fit_result, sample_params) where sample_params has shape (n_boot, n_params).
    """
    tsep_array, tau_array, data_samples = _build_fit_data_2state(
        ratio_by_tsep, tau_range, tsep_fit_list
    )
    prepared = _prepare_fit_data(data_samples)

    n_params = 1 if nstates == 1 else 4
    theta0 = np.zeros(n_params, dtype=float)
    n_boot = data_samples.shape[0]

    sample_params = np.full((n_boot, n_params), np.nan, dtype=float)
    chi2_arr = np.full(n_boot, np.nan, dtype=float)
    chi2_dof_arr = np.full(n_boot, np.nan, dtype=float)
    pvalue_arr = np.full(n_boot, np.nan, dtype=float)

    def residuals(params: np.ndarray, data: np.ndarray) -> np.ndarray:
        model = evaluate_emff_ratio_2state(
            tsep_array, tau_array, delta_e, r1, params
        )
        return (model - data) / prepared.sigma

    for boot_idx, sample_data in enumerate(prepared.data_samples):
        result = least_squares(residuals, theta0, args=(sample_data,), max_nfev=5000)
        if result.success:
            p = np.asarray(result.x, dtype=float)
            sample_params[boot_idx] = p
            c2 = float(np.dot(result.fun, result.fun))
            dof = max(len(tsep_array) - n_params, 1)
            chi2_arr[boot_idx] = c2
            chi2_dof_arr[boot_idx] = c2 / dof
            pvalue_arr[boot_idx] = float(1.0 - chi2.cdf(c2, dof))

    success_mask = np.all(np.isfinite(sample_params), axis=1)
    if np.any(success_mask):
        params_mean, _ = summarize_parameter_samples(sample_params[success_mask])
        chi2_mean, _ = robust_mean_and_error(chi2_arr[success_mask])
        chi2_dof_mean, _ = robust_mean_and_error(chi2_dof_arr[success_mask])
        pvalue_mean, _ = robust_mean_and_error(pvalue_arr[success_mask])
        fit_result = EMFFFitterResult(
            params=np.asarray(params_mean, dtype=float),
            chi2=chi2_mean,
            chi2_dof=chi2_dof_mean,
            pvalue=pvalue_mean,
            success=True,
            message=f"fitted {int(np.count_nonzero(success_mask))} bootstrap samples",
        )
    else:
        fit_result = EMFFFitterResult(
            params=np.full(n_params, np.nan),
            chi2=float("nan"),
            chi2_dof=float("nan"),
            pvalue=float("nan"),
            success=False,
            message="all bootstrap sample fits failed",
        )
    return fit_result, sample_params


def fit_emff_summation(
    ratio_by_tsep: dict[int, np.ndarray],
    tau_range: tuple[int, int],
    tsep_fit_list: list[int],
    fit_intercept: bool = True,
) -> tuple[EMFFFitterResult, np.ndarray]:
    """Fit the summation method: S(tsep) = tsep * M_00 + B.

    Returns:
        (fit_result, sample_params) where sample_params has shape (n_boot, n_params).
    """
    tsep_array, summed_samples = _build_fit_data_summation(
        ratio_by_tsep, tau_range, tsep_fit_list
    )
    prepared = _prepare_fit_data(summed_samples)

    n_params = 2 if fit_intercept else 1
    theta0 = np.zeros(n_params, dtype=float)
    n_boot = summed_samples.shape[0]

    sample_params = np.full((n_boot, n_params), np.nan, dtype=float)
    chi2_arr = np.full(n_boot, np.nan, dtype=float)
    chi2_dof_arr = np.full(n_boot, np.nan, dtype=float)
    pvalue_arr = np.full(n_boot, np.nan, dtype=float)

    def residuals(params: np.ndarray, data: np.ndarray) -> np.ndarray:
        model = evaluate_emff_summed_ratio(tsep_array, params)
        return (model - data) / prepared.sigma

    for boot_idx, sample_data in enumerate(prepared.data_samples):
        result = least_squares(residuals, theta0, args=(sample_data,), max_nfev=5000)
        if result.success:
            p = np.asarray(result.x, dtype=float)
            sample_params[boot_idx] = p
            c2 = float(np.dot(result.fun, result.fun))
            dof = max(len(tsep_array) - n_params, 1)
            chi2_arr[boot_idx] = c2
            chi2_dof_arr[boot_idx] = c2 / dof
            pvalue_arr[boot_idx] = float(1.0 - chi2.cdf(c2, dof))

    success_mask = np.all(np.isfinite(sample_params), axis=1)
    if np.any(success_mask):
        params_mean, _ = summarize_parameter_samples(sample_params[success_mask])
        chi2_mean, _ = robust_mean_and_error(chi2_arr[success_mask])
        chi2_dof_mean, _ = robust_mean_and_error(chi2_dof_arr[success_mask])
        pvalue_mean, _ = robust_mean_and_error(pvalue_arr[success_mask])
        fit_result = EMFFFitterResult(
            params=np.asarray(params_mean, dtype=float),
            chi2=chi2_mean,
            chi2_dof=chi2_dof_mean,
            pvalue=pvalue_mean,
            success=True,
            message=f"fitted {int(np.count_nonzero(success_mask))} bootstrap samples",
        )
    else:
        fit_result = EMFFFitterResult(
            params=np.full(n_params, np.nan),
            chi2=float("nan"),
            chi2_dof=float("nan"),
            pvalue=float("nan"),
            success=False,
            message="all bootstrap sample fits failed",
        )
    return fit_result, sample_params


def fit_emff_plateau(
    ratio_by_tsep: dict[int, np.ndarray],
    tau_range: tuple[int, int],
    tsep_fit_list: list[int],
) -> tuple[EMFFFitterResult, np.ndarray]:
    """Fit the plateau method: R(tsep, tau) = M_00 (constant).

    Returns:
        (fit_result, sample_params) where sample_params has shape (n_boot, 1).
    """
    tsep_array, tau_array, data_samples = _build_fit_data_2state(
        ratio_by_tsep, tau_range, tsep_fit_list
    )
    prepared = _prepare_fit_data(data_samples)

    theta0 = np.zeros(1, dtype=float)
    n_boot = data_samples.shape[0]
    n_points = data_samples.shape[1]

    sample_params = np.full((n_boot, 1), np.nan, dtype=float)
    chi2_arr = np.full(n_boot, np.nan, dtype=float)
    chi2_dof_arr = np.full(n_boot, np.nan, dtype=float)
    pvalue_arr = np.full(n_boot, np.nan, dtype=float)

    def residuals(params: np.ndarray, data: np.ndarray) -> np.ndarray:
        model = evaluate_emff_plateau(n_points, params)
        return (model - data) / prepared.sigma

    for boot_idx, sample_data in enumerate(prepared.data_samples):
        result = least_squares(residuals, theta0, args=(sample_data,), max_nfev=5000)
        if result.success:
            p = np.asarray(result.x, dtype=float)
            sample_params[boot_idx] = p
            c2 = float(np.dot(result.fun, result.fun))
            dof = max(n_points - 1, 1)
            chi2_arr[boot_idx] = c2
            chi2_dof_arr[boot_idx] = c2 / dof
            pvalue_arr[boot_idx] = float(1.0 - chi2.cdf(c2, dof))

    success_mask = np.all(np.isfinite(sample_params), axis=1)
    if np.any(success_mask):
        params_mean, _ = summarize_parameter_samples(sample_params[success_mask])
        chi2_mean, _ = robust_mean_and_error(chi2_arr[success_mask])
        chi2_dof_mean, _ = robust_mean_and_error(chi2_dof_arr[success_mask])
        pvalue_mean, _ = robust_mean_and_error(pvalue_arr[success_mask])
        fit_result = EMFFFitterResult(
            params=np.asarray(params_mean, dtype=float),
            chi2=chi2_mean,
            chi2_dof=chi2_dof_mean,
            pvalue=pvalue_mean,
            success=True,
            message=f"fitted {int(np.count_nonzero(success_mask))} bootstrap samples",
        )
    else:
        fit_result = EMFFFitterResult(
            params=np.full(1, np.nan),
            chi2=float("nan"),
            chi2_dof=float("nan"),
            pvalue=float("nan"),
            success=False,
            message="all bootstrap sample fits failed",
        )
    return fit_result, sample_params


def _write_ratio_outputs(
    output_root: Path,
    stem: str,
    qx: int,
    qy: int,
    qz: int,
    ratio_by_tsep: dict[int, np.ndarray],
    tau_range: tuple[int, int],
    tsep_fit_list: list[int],
    averaged_q_members: tuple[tuple[int, int, int], ...] | None = None,
) -> Path:
    """Write ratio data at all (tsep, tau) to a table."""
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / f"{stem}_ratio.txt"
    tau_min, tau_offset = tau_range

    with path.open("w", encoding="utf-8") as handle:
        if averaged_q_members is None:
            handle.write(f"qx {qx} qy {qy} qz {qz}\n")
        else:
            handle.write(f"canonical_qx {qx} canonical_qy {qy} canonical_qz {qz}\n")
            members_text = " ".join(
                f"({member_qx},{member_qy},{member_qz})"
                for member_qx, member_qy, member_qz in averaged_q_members
            )
            handle.write(f"averaged_q_members {members_text}\n")
        handle.write(f"tau_range {tau_min} {tau_offset}\n")
        header = [
            "tsep", "tau", "in_fit_window",
            "ratio_real_mean", "ratio_real_err",
            "ratio_imag_mean", "ratio_imag_err",
        ]
        handle.write("\t".join(header) + "\n")
        for tsep in sorted(ratio_by_tsep.keys()):
            boot_samples = ratio_by_tsep[tsep]  # (n_boot, n_tau)
            tau_vals = compute_tau_range_for_tsep(tsep, tau_min, tau_offset)
            for col_idx, tau in enumerate(tau_vals):
                real_samples = np.real(boot_samples[:, col_idx])
                imag_samples = np.imag(boot_samples[:, col_idx])
                real_mean, real_err = robust_mean_and_error(real_samples)
                imag_mean, imag_err = robust_mean_and_error(imag_samples)
                in_window = int(tsep in tsep_fit_list)
                handle.write(
                    "\t".join([
                        str(int(tsep)),
                        str(int(tau)),
                        str(in_window),
                        f"{real_mean:.10e}",
                        f"{real_err:.10e}",
                        f"{imag_mean:.10e}",
                        f"{imag_err:.10e}",
                    ])
                    + "\n"
                )
    return path


def _write_fit_outputs(
    output_root: Path,
    stem: str,
    record: EMFFOutputRecord,
) -> list[Path]:
    """Write fit results (summary, table, samples)."""
    tables_dir = output_root / "tables"
    samples_dir = output_root / "samples"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    nstates = record.nstates
    n_params = record.fit_result.params.shape[0]
    outputs: list[Path] = []

    # Summary
    summary_path = output_root / f"{stem}_{nstates}state_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"qx {record.qx} qy {record.qy} qz {record.qz}\n")
        handle.write(f"nstates {nstates}\n")
        handle.write(f"fit_method {record.fit_method}\n")
        handle.write(f"delta_e {record.delta_e:.10e}\n")
        handle.write(f"r1 {record.r1:.10e}\n")
        handle.write(f"tau_range {record.tau_range_used[0]} {record.tau_range_used[1]}\n")
        handle.write(f"tsep_fit {record.tsep_fit_values[0]} {record.tsep_fit_values[-1]}\n")
        handle.write(f"success {int(record.fit_result.success)}\n")
        handle.write(f"chi2_dof {record.fit_result.chi2_dof:.10e}\n")
        handle.write(f"pvalue {record.fit_result.pvalue:.10e}\n")
        params_mean, params_err = summarize_parameter_samples(record.sample_params)
        for idx in range(n_params):
            handle.write(f"p{idx} {params_mean[idx]:.10e} {params_err[idx]:.10e}\n")
    outputs.append(summary_path)

    # Fit table
    table_path = tables_dir / f"{stem}_{nstates}state_fit.txt"
    with table_path.open("w", encoding="utf-8") as handle:
        header = ["qx", "qy", "qz", "nstates", "success", "chi2_dof", "pvalue"]
        header += [f"p{idx}_mean" for idx in range(n_params)]
        header += [f"p{idx}_err" for idx in range(n_params)]
        handle.write("\t".join(header) + "\n")
        params_mean, params_err = summarize_parameter_samples(record.sample_params)
        row = [
            str(record.qx), str(record.qy), str(record.qz),
            str(nstates),
            str(int(record.fit_result.success)),
            f"{record.fit_result.chi2_dof:.10e}",
            f"{record.fit_result.pvalue:.10e}",
            *[f"{v:.10e}" for v in params_mean],
            *[f"{v:.10e}" for v in params_err],
        ]
        handle.write("\t".join(row) + "\n")
    outputs.append(table_path)

    # Samples
    sample_path = samples_dir / f"{stem}_{nstates}state_samples.txt"
    with sample_path.open("w", encoding="utf-8") as handle:
        header = ["sample_id", "success"] + [f"p{idx}" for idx in range(n_params)]
        handle.write("\t".join(header) + "\n")
        for boot_idx, params in enumerate(record.sample_params):
            success = int(np.all(np.isfinite(params)))
            row = [str(boot_idx), str(success)] + [f"{v:.10e}" for v in params]
            handle.write("\t".join(row) + "\n")
    outputs.append(sample_path)

    return outputs


def run_emff_nstate_fit(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    """Run bootstrap-based EMFF N-state fits.

    Args:
        input_file: Path to the EMFF control file.
        results_dir: Override for the output directory.

    Returns:
        List of generated output file paths.
    """
    spec = parse_emff_fit_input(input_file, results_dir=results_dir)
    outputs: list[Path] = []

    # Keep the existing |Pf| label for 2pt fit references and {pz} path templates.
    pfx, pfy, pfz = spec.pflist
    pz_2pt = int(round(np.sqrt(pfx**2 + pfy**2 + pfz**2)))

    c2pt_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def load_c2pt(momentum: tuple[int, int, int]) -> np.ndarray:
        if momentum in c2pt_cache:
            return c2pt_cache[momentum]
        px, py, pz = momentum
        momentum_label = int(round(np.sqrt(px**2 + py**2 + pz**2)))
        c2pt_path = expand_template(
            spec.c2pt,
            src_gamma=spec.src_gamma,
            sink_gamma=spec.sink_gamma,
            pz=momentum_label,
            pfx=px,
            pfy=py,
            pfz=pz,
        )
        _, c2pt_raw = load_emff_c2pt_correlator(
            c2pt_path,
            sink_gamma=spec.sink_gamma,
            px=px,
            py=py,
            pz=pz,
        )
        if c2pt_raw.shape[1] != spec.nt:
            raise ValueError(
                f"2pt data has {c2pt_raw.shape[1]} time slices, expected Nt={spec.nt}"
            )
        c2pt_cache[momentum] = c2pt_raw
        return c2pt_raw

    final_momentum = (pfx, pfy, pfz)
    c2pt_final = load_c2pt(final_momentum)
    energy_final = _hadron_energy_from_dispersion(
        final_momentum,
        ns=spec.ns,
        lattice_spacing_fm=spec.lattice_spacing_fm,
        hadron_mass_gev=spec.hadron_mass_gev,
    )

    # Title for 2pt fit references
    title = expand_template(spec.title_pattern)
    dataset_root = spec.results_dir / title
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Resolve 2pt fit references for each nstates
    two_point_fit_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    two_point_window = spec.two_point_fit_window_by_pz.get(pz_2pt)
    if two_point_window is None:
        raise ValueError(
            f"missing two_point_fit_window_by_pz entry for pz={pz_2pt}"
        )
    two_point_tmin, two_point_tmax = two_point_window
    for nstates in spec.nstates:
        fit_ref = resolve_two_point_fit_reference(
            spec.two_point_fit_root,
            title=title,
            nstates=nstates,
            tmin=two_point_tmin,
            tmax=two_point_tmax,
        )
        two_point_fit_cache[nstates] = (
            np.asarray(fit_ref.amplitudes, dtype=float),
            np.asarray(fit_ref.energies, dtype=float),
        )

    q_groups = _build_q_groups(
        spec.qxlist,
        spec.qylist,
        spec.qzlist,
        average_transverse_orbits=spec.average_transverse_orbits,
        final_momentum=final_momentum,
    )

    # Determine tsep values for fitting
    tsep_fit_list = [
        tsep for tsep in spec.tslist
        if spec.tsep_range[0] <= tsep <= spec.tsep_range[1]
    ]
    if not tsep_fit_list:
        raise ValueError(
            f"no tsep values in tslist fall within tsep_range {spec.tsep_range}"
        )

    def load_initial_c2pt_for_q(qx: int, qy: int, qz: int) -> np.ndarray:
        initial_momentum = (pfx - qx, pfy - qy, pfz - qz)
        return load_c2pt(initial_momentum)

    def average_initial_c2pt(
        members: tuple[tuple[int, int, int], ...]
    ) -> np.ndarray:
        return np.mean(
            np.stack(
                [load_initial_c2pt_for_q(qx, qy, qz) for qx, qy, qz in members],
                axis=0,
            ),
            axis=0,
        )

    def average_c3pt_tau(
        members: tuple[tuple[int, int, int], ...],
        tsep: int,
    ) -> np.ndarray:
        c3pt_blocks: list[np.ndarray] = []
        h5_path = resolve_emff_h5_path(
            spec.c3pt_h5,
            src_gamma=spec.src_gamma,
            pfx=pfx,
            pfy=pfy,
            pfz=pfz,
            tsep=tsep,
        )
        for member_qx, member_qy, member_qz in members:
            c3pt_data = load_emff_correlator(
                h5_path,
                spec.c3pt_dataset_path,
                insert_gamma=spec.insert_gamma,
                qx=member_qx,
                qy=member_qy,
                qz=member_qz,
            )  # shape: (tsep+2, n_cfg)
            c3pt_blocks.append(c3pt_data[: tsep + 1, :])
        return np.mean(np.stack(c3pt_blocks, axis=0), axis=0)

    for (qx, qy, qz), q_members in q_groups.items():
        c2pt_initial = average_initial_c2pt(q_members)
        initial_momentum = (pfx - qx, pfy - qy, pfz - qz)
        energy_initial = _hadron_energy_from_dispersion(
            initial_momentum,
            ns=spec.ns,
            lattice_spacing_fm=spec.lattice_spacing_fm,
            hadron_mass_gev=spec.hadron_mass_gev,
        )

        # --- Load 3pt data for all tsep ---
        ratio_by_tsep: dict[int, np.ndarray] = {}
        for tsep in spec.tslist:
            c3pt_tau = average_c3pt_tau(q_members, tsep)
            ratio_by_tsep[tsep] = _compute_emff_ratio(
                c3pt_tau,
                c2pt_initial,
                c2pt_final,
                tsep=tsep,
                energy_initial=energy_initial,
                energy_final=energy_final,
            )

        # --- Bin ---
        if spec.binsize > 1:
            for tsep in spec.tslist:
                ratio_by_tsep[tsep] = bin_samples(
                    ratio_by_tsep[tsep], binsize=spec.binsize
                )

        # --- Bootstrap ---
        # All tsep should have the same n_cfg
        n_cfg = next(iter(ratio_by_tsep.values())).shape[0]
        if n_cfg < 2:
            raise ValueError(f"bootstrap requires at least 2 samples, got {n_cfg}")
        n_boot = n_cfg if spec.bootstrap_samples is None else spec.bootstrap_samples
        draw_size = n_cfg if spec.bootstrap_size is None else spec.bootstrap_size
        indices = common_bootstrap_indices(
            n_cfg, draw_size, seed=spec.seed, n_boot=n_boot
        )

        for tsep in spec.tslist:
            ratio_by_tsep[tsep] = bootstrap_means(
                ratio_by_tsep[tsep], indices=indices
            )  # (n_boot, tsep+1)

        # --- Write ratio table ---
        q_tag = _q_tag(qx, qy, qz)
        stem = f"{title}_{sanitize_token(spec.insert_gamma)}_{q_tag}"
        ratio_path = _write_ratio_outputs(
            dataset_root, stem, qx, qy, qz,
            ratio_by_tsep, spec.tau_range, tsep_fit_list,
            averaged_q_members=q_members if spec.average_transverse_orbits else None,
        )
        outputs.append(ratio_path)

        # --- Fit ---
        for nstates in spec.nstates:
            amplitudes, energies = two_point_fit_cache[nstates]
            delta_e = float(energies[1] - energies[0]) if nstates >= 2 else 0.0
            r1 = float((amplitudes[1] / amplitudes[0]) ** 2) if nstates >= 2 else 0.0

            if spec.fit_method == "2state":
                fit_result, sample_params = fit_emff_2state(
                    ratio_by_tsep, delta_e, r1,
                    spec.tau_range, tsep_fit_list, nstates,
                )
            elif spec.fit_method == "summation":
                fit_result, sample_params = fit_emff_summation(
                    ratio_by_tsep, spec.tau_range, tsep_fit_list,
                    fit_intercept=True,
                )
            elif spec.fit_method == "plateau":
                fit_result, sample_params = fit_emff_plateau(
                    ratio_by_tsep, spec.tau_range, tsep_fit_list,
                )

            record = EMFFOutputRecord(
                qx=qx,
                qy=qy,
                qz=qz,
                nstates=nstates,
                fit_method=spec.fit_method,
                fit_result=fit_result,
                sample_params=sample_params,
                delta_e=delta_e,
                r1=r1,
                tsep_fit_values=tuple(tsep_fit_list),
                tau_range_used=spec.tau_range,
            )

            fit_outputs = _write_fit_outputs(dataset_root, stem, record)
            outputs.extend(fit_outputs)

        # --- Print progress ---
        print(
            f"[emff-fit] q=({qx:+d},{qy:+d},{qz:+d}) "
            f"done ({len(spec.tslist)} tsep values)"
        )

    return outputs
