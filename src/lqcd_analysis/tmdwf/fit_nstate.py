from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from ..common.bootstrap import bin_samples, bootstrap_indices as common_bootstrap_indices, bootstrap_means
from ..common.constants import HBAR_C_GEV_FM, MIN_POSITIVE
from ..common.parsing import (
    load_fit_window_table,
    parse_bool,
    parse_fold_t,
    parse_int_list_or_range,
    parse_optional_int,
    parse_tsrange,
)
from ..common.utils import (
    apply_fold_t,
    robust_mean_and_error,
)
from ..two_point.io import load_correlator_csv
from .io import (
    _load_tmdwf_correlator_from_handle,
    expand_template,
    resolve_qtmdwf_h5_path,
    resolve_two_point_fit_reference,
)
from .models import evaluate_tmdwf_ratio, normalize_tmdwf_operator
from .plotting import RatioFitPlotSeries, plot_tmdwf_m0_from_fit_tables, plot_tmdwf_ratio_fit, write_tmdwf_plot_notebook


@dataclass(frozen=True)
class TMDWFNStateInput:
    title_pattern: str
    ns: int
    nt: int
    lattice_spacing_fm: float
    decay_constant_check: bool
    fit_target: str
    fit_component: str
    nstates: tuple[int, ...]
    pzlist: tuple[int, ...]
    gmlist: tuple[str, ...]
    etalist: tuple[str, ...]
    tdirlist: tuple[str, ...]
    bTlist: tuple[int, ...]
    bzlist: tuple[int, ...]
    binsize: int
    bootstrap_samples: int | None
    bootstrap_size: int | None
    seed: int
    fit_window: str
    qtmdwf_h5: str
    dataset_path_template: str
    c2pt: str
    fold_t: str
    tsrange: tuple[int, int]
    two_point_fit_root: str
    two_point_fit_window_by_pz: dict[int, tuple[int, int]]
    make_plots: bool
    results_dir: Path


@dataclass(frozen=True)
class TMDWFFitResult:
    params: np.ndarray
    chi2: float
    chi2_dof: float
    pvalue: float
    success: bool
    message: str


@dataclass(frozen=True)
class _PreparedFitData:
    times: np.ndarray
    data_samples: np.ndarray
    mean_data: np.ndarray
    sigma: np.ndarray


@dataclass(frozen=True)
class TMDWFOutputRecord:
    bz: int
    component: str
    nstates: int
    tmin: int
    tmax: int
    fit_result: TMDWFFitResult
    sample_params: np.ndarray
    amplitudes: np.ndarray
    energies: np.ndarray
    pz: int
    ns: int
    gm: str
    two_point_fit_tmin: int
    two_point_fit_tmax: int
    two_point_fit_table_resolved: str
    two_point_fit_tmax_source: str
    tsrange_start: int
    tsrange_end: int
    ratio_samples: np.ndarray


@dataclass(frozen=True)
class TMDWFRatioRecord:
    bz: int
    tmin: int
    tmax: int
    two_point_fit_table_resolved: str
    two_point_fit_tmax_source: str
    two_point_fit_tmax: int
    tsrange_start: int
    tsrange_end: int
    ratio_samples: np.ndarray








def parse_tmdwf_fit_input(path: str | Path, results_dir: str | Path | None = None) -> TMDWFNStateInput:
    file_path = Path(path)
    entries: dict[str, list[str]] = {}
    first_tokens: list[str] | None = None

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if first_tokens is None:
                first_tokens = tokens
            entries[tokens[0]] = tokens[1:]

    if first_tokens is None or len(first_tokens) < 4:
        raise ValueError("the first non-empty line must be: title Ns Nt a_fm")

    required = {
        "fit_target",
        "fit_component",
        "nstates",
        "pzlist",
        "gmlist",
        "etalist",
        "Tdirlist",
        "qtmdwf_h5",
        "dataset_path_template",
        "two_point_fit_root",
        "two_point_fit_window_by_pz",
        "c2pt",
        "fold_t",
    }
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")
    if "fit_window" not in entries:
        raise ValueError("missing required key: fit_window")

    fit_target = entries["fit_target"][0].lower()
    if fit_target != "ratio":
        raise ValueError("only fit_target ratio is supported")
    fit_component = entries["fit_component"][0].lower()
    if fit_component not in {"real", "imag", "both"}:
        raise ValueError("fit_component must be one of: real, imag, both")

    nstates = tuple(sorted({int(item) for item in entries["nstates"]}))
    if not nstates or any(state not in {1, 2} for state in nstates):
        raise ValueError("TMDWF nstates must contain only 1 and/or 2")
    for gm in entries["gmlist"]:
        normalize_tmdwf_operator(gm)
    input_path = file_path.resolve()
    two_point_fit_window_path = Path(entries["two_point_fit_window_by_pz"][0])
    two_point_fit_window_by_pz = load_fit_window_table(two_point_fit_window_path)
    missing_windows = sorted(set(int(pz) for pz in entries["pzlist"]) - {pz for gm, pz in two_point_fit_window_by_pz if gm is None})
    if missing_windows:
        raise ValueError(
            f"two_point_fit_window_by_pz is missing entries for pz values: {missing_windows}"
        )
    return TMDWFNStateInput(
        title_pattern=first_tokens[0],
        ns=int(first_tokens[1]),
        nt=int(first_tokens[2]),
        lattice_spacing_fm=float(first_tokens[3]),
        decay_constant_check=parse_bool(entries.get("decay_constant_check", ["false"])[0]),
        fit_target=fit_target,
        fit_component=fit_component,
        nstates=nstates,
        pzlist=tuple(int(item) for item in entries["pzlist"]),
        gmlist=tuple(entries["gmlist"]),
        etalist=tuple(entries["etalist"]),
        tdirlist=tuple(entries["Tdirlist"]),
        bTlist=parse_int_list_or_range(entries, "bTlist", "bTrange"),
        bzlist=parse_int_list_or_range(entries, "bzlist", "bzrange"),
        binsize=int(entries.get("binsize", ["1"])[0]),
        bootstrap_samples=parse_optional_int(entries.get("bootstrap_samples", ["auto"])[0]),
        bootstrap_size=parse_optional_int(entries.get("bootstrap_size", ["auto"])[0]),
        seed=int(entries.get("seed", ["2026"])[0]),
        fit_window=entries["fit_window"][0],
        qtmdwf_h5=entries["qtmdwf_h5"][0],
        dataset_path_template=entries["dataset_path_template"][0],
        c2pt=entries["c2pt"][0],
        fold_t=parse_fold_t(entries),
        tsrange=parse_tsrange(entries, int(first_tokens[2])),
        two_point_fit_root=entries["two_point_fit_root"][0],
        two_point_fit_window_by_pz={
            pz: window for (gm, pz), window in two_point_fit_window_by_pz.items() if gm is None
        },
        make_plots=parse_bool(entries.get("plot", ["false"])[0]),
        results_dir=(
            Path(entries["results_dir"][0])
            if "results_dir" in entries and results_dir is None
            else ((input_path.parent / "results_tmdwf_fit") if results_dir is None else Path(results_dir))
        ),
    )

def _prepare_fit_data(
    ratio_samples: np.ndarray,
    *,
    tmin: int,
    tmax: int,
    component: str,
) -> _PreparedFitData:
    component = component.lower()
    if component not in {"real", "imag"}:
        raise ValueError("component must be real or imag")
    times = np.arange(tmin, tmax + 1, dtype=int)
    sample_window = ratio_samples[:, tmin : tmax + 1]
    if component == "real":
        data_samples = np.real(sample_window)
    else:
        data_samples = np.imag(sample_window)
    mean_data = np.nanmean(data_samples, axis=0)
    sigma = np.nanstd(data_samples, axis=0, ddof=1)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, MIN_POSITIVE)
    return _PreparedFitData(
        times=times,
        data_samples=data_samples,
        mean_data=mean_data,
        sigma=sigma,
    )


def fit_tmdwf_component(
    ratio_samples: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    nt: int,
    pz: int,
    ns: int,
    gm: str,
    tmin: int,
    tmax: int,
    component: str,
) -> tuple[TMDWFFitResult, np.ndarray]:
    prepared = _prepare_fit_data(ratio_samples, tmin=tmin, tmax=tmax, component=component)

    def residuals(params: np.ndarray, data: np.ndarray) -> np.ndarray:
        model_values = evaluate_tmdwf_ratio(prepared.times, amplitudes, energies, params, nt, gm=gm, pz=pz, ns=ns)
        return (model_values - data) / prepared.sigma

    theta0 = np.zeros(len(amplitudes), dtype=float)
    sample_params = np.full((ratio_samples.shape[0], len(amplitudes)), np.nan, dtype=float)
    chi2_samples = np.full(ratio_samples.shape[0], np.nan, dtype=float)
    chi2_dof_samples = np.full(ratio_samples.shape[0], np.nan, dtype=float)
    pvalue_samples = np.full(ratio_samples.shape[0], np.nan, dtype=float)
    for sample_id, sample_data in enumerate(prepared.data_samples):
        sample_result = least_squares(residuals, theta0, args=(sample_data,), max_nfev=5000)
        if sample_result.success:
            params = np.asarray(sample_result.x, dtype=float)
            sample_params[sample_id] = params
            chi2_value = float(np.dot(sample_result.fun, sample_result.fun))
            dof = max(len(prepared.times) - len(theta0), 1)
            chi2_samples[sample_id] = chi2_value
            chi2_dof_samples[sample_id] = chi2_value / dof
            pvalue_samples[sample_id] = float(1.0 - chi2.cdf(chi2_value, dof))

    success_mask = np.all(np.isfinite(sample_params), axis=1)
    if np.any(success_mask):
        params_mean, _ = summarize_parameter_samples(sample_params[success_mask])
        chi2_mean, _ = robust_mean_and_error(chi2_samples[success_mask])
        chi2_dof_mean, _ = robust_mean_and_error(chi2_dof_samples[success_mask])
        pvalue_mean, _ = robust_mean_and_error(pvalue_samples[success_mask])
        success = True
        message = f"bootstrap-centered fit from {int(np.count_nonzero(success_mask))} successful samples"
    else:
        params_mean = tuple(np.nan for _ in range(len(amplitudes)))
        chi2_mean = float("nan")
        chi2_dof_mean = float("nan")
        pvalue_mean = float("nan")
        success = False
        message = "all bootstrap sample fits failed"
    return (
        TMDWFFitResult(
            params=np.asarray(params_mean, dtype=float),
            chi2=chi2_mean,
            chi2_dof=chi2_dof_mean,
            pvalue=pvalue_mean,
            success=success,
            message=message,
        ),
        sample_params,
    )


def summarize_parameter_samples(samples: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    means: list[float] = []
    errors: list[float] = []
    for column in range(samples.shape[1]):
        valid = samples[:, column][np.isfinite(samples[:, column])]
        mean, err = robust_mean_and_error(valid)
        means.append(mean)
        errors.append(err)
    return tuple(means), tuple(errors)


def _summarize_ground_state_matrix_element(
    sample_params: np.ndarray,
    *,
    lattice_spacing_fm: float,
) -> tuple[float, float, float, float]:
    success_mask = np.all(np.isfinite(sample_params), axis=1)
    if not np.any(success_mask):
        return float("nan"), float("nan"), float("nan"), float("nan")
    m0_samples = np.asarray(sample_params[success_mask, 0], dtype=float)
    m0_mean, m0_err = robust_mean_and_error(m0_samples)
    gev_scale = HBAR_C_GEV_FM / lattice_spacing_fm
    return m0_mean, m0_err, m0_mean * gev_scale, m0_err * gev_scale


def _iter_decay_constant_scan_windows(tmin: int, tmax: int, *, max_t: int) -> tuple[tuple[int, int], ...]:
    windows: list[tuple[int, int]] = []
    for scan_tmin in range(tmin - 2, tmin + 3):
        for scan_tmax in range(tmax - 2, tmax + 1):
            if 0 <= scan_tmin <= scan_tmax <= max_t:
                windows.append((scan_tmin, scan_tmax))
    return tuple(windows)


def _iter_decay_constant_two_point_tmins(tmin: int, tmax: int) -> tuple[int, ...]:
    return tuple(candidate for candidate in range(tmin - 1, tmin + 2) if candidate <= tmax)


def _summarize_bootstrap_series(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2:
        raise ValueError("bootstrap summary samples must be two-dimensional")
    p16 = np.percentile(values, 16.0, axis=0)
    p84 = np.percentile(values, 84.0, axis=0)
    center = 0.5 * (p16 + p84)
    return center, p16, p84


def _select_curve_component(values: np.ndarray, component: str) -> np.ndarray:
    curve_values = np.asarray(values)
    if component == "real":
        return np.asarray(np.real(curve_values), dtype=float)
    if component == "imag":
        return np.asarray(np.imag(curve_values), dtype=float)
    raise ValueError("component must be real or imag")






def sanitize_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def _component_list(fit_component: str) -> tuple[str, ...]:
    return ("real", "imag") if fit_component == "both" else (fit_component,)


def _write_ratio_outputs(
    output_root: Path,
    stem: str,
    records: tuple[TMDWFRatioRecord, ...],
) -> Path:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("ratio output requires at least one record")
    path = tables_dir / f"{stem}_ratio.txt"
    metadata = records[0]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"tsrange {metadata.tsrange_start} {metadata.tsrange_end}\n")
        handle.write(f"tfit {metadata.tmin} {metadata.tmax}\n")
        handle.write(f"two_point_fit_table_resolved {metadata.two_point_fit_table_resolved}\n")
        handle.write(f"two_point_fit_tmax_source {metadata.two_point_fit_tmax_source}\n")
        handle.write(f"two_point_fit_tmax {metadata.two_point_fit_tmax}\n")
        handle.write(
            "\t".join(
                [
                    "bz",
                    "t",
                    "in_fit_window",
                    "ratio_real_mean",
                    "ratio_real_err",
                    "ratio_imag_mean",
                    "ratio_imag_err",
                ]
            )
            + "\n"
        )
        for record in records:
            times = np.arange(record.tsrange_start, record.tsrange_end + 1, dtype=int)
            ratio_real_mean, ratio_real_p16, ratio_real_p84 = _summarize_bootstrap_series(np.real(record.ratio_samples))
            ratio_imag_mean, ratio_imag_p16, ratio_imag_p84 = _summarize_bootstrap_series(np.imag(record.ratio_samples))
            ratio_real_err = 0.5 * (ratio_real_p84 - ratio_real_p16)
            ratio_imag_err = 0.5 * (ratio_imag_p84 - ratio_imag_p16)
            for t, real_mean, real_err, imag_mean, imag_err in zip(
                times,
                ratio_real_mean,
                ratio_real_err,
                ratio_imag_mean,
                ratio_imag_err,
                strict=True,
            ):
                handle.write(
                    "\t".join(
                        [
                            str(record.bz),
                            str(int(t)),
                            str(int(record.tmin <= int(t) <= record.tmax)),
                            f"{real_mean:.10e}",
                            f"{real_err:.10e}",
                            f"{imag_mean:.10e}",
                            f"{imag_err:.10e}",
                        ]
                    )
                    + "\n"
                )
    return path


def _write_component_outputs(
    output_root: Path,
    stem: str,
    records: tuple[TMDWFOutputRecord, ...],
    nt: int,
    *,
    make_plots: bool,
) -> list[Path]:
    tables_dir = output_root / "tables"
    plots_dir = output_root / "plots"
    samples_dir = output_root / "samples"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return []
    component = records[0].component
    nstates = records[0].nstates
    summary_path = output_root / f"{stem}_{component}_{nstates}state_summary.txt"
    table_path = tables_dir / f"{stem}_{component}_{nstates}state_fit.txt"
    sample_path = samples_dir / f"{stem}_{component}_{nstates}state_samples.txt"
    curve_path = tables_dir / f"{stem}_{component}_{nstates}state_curve.txt"
    plot_path = plots_dir / f"{stem}_{component}_{nstates}state_ratio_fit.pdf"

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"component {component}\n")
        handle.write(f"nstates {nstates}\n")
        for record in records:
            params_mean, params_err = summarize_parameter_samples(record.sample_params)
            handle.write(f"begin_bz {record.bz}\n")
            handle.write(f"tfit {record.tmin} {record.tmax}\n")
            handle.write(f"two_point_fit_table_resolved {record.two_point_fit_table_resolved}\n")
            handle.write(f"two_point_fit_tmax_source {record.two_point_fit_tmax_source}\n")
            handle.write(f"two_point_fit_tmax {record.two_point_fit_tmax}\n")
            handle.write(f"success_bootstrap_center {int(record.fit_result.success)}\n")
            handle.write(f"chi2_dof {record.fit_result.chi2_dof:.10e}\n")
            handle.write(f"pvalue {record.fit_result.pvalue:.10e}\n")
            for idx in range(nstates):
                handle.write(f"m{idx} {params_mean[idx]:.10e} {params_err[idx]:.10e}\n")
            handle.write(f"end_bz {record.bz}\n")

    with table_path.open("w", encoding="utf-8") as handle:
        header = ["bz", "tmin", "tmax", "success_bootstrap_center", "chi2_dof", "pvalue", "two_point_fit_tmax"]
        header += [f"m{idx}_mean" for idx in range(nstates)]
        header += [f"m{idx}_err" for idx in range(nstates)]
        handle.write("\t".join(header) + "\n")
        for record in records:
            params_mean, params_err = summarize_parameter_samples(record.sample_params)
            row = [
                str(record.bz),
                str(record.tmin),
                str(record.tmax),
                str(int(record.fit_result.success)),
                f"{record.fit_result.chi2_dof:.10e}",
                f"{record.fit_result.pvalue:.10e}",
                str(record.two_point_fit_tmax),
                *[f"{value:.10e}" for value in params_mean],
                *[f"{value:.10e}" for value in params_err],
            ]
            handle.write("\t".join(row) + "\n")

    with sample_path.open("w", encoding="utf-8") as handle:
        header = ["bz", "sample_id", "success"] + [f"m{idx}" for idx in range(nstates)]
        handle.write("\t".join(header) + "\n")
        for record in records:
            for sample_id, params in enumerate(record.sample_params):
                success = int(np.all(np.isfinite(params)))
                row = [str(record.bz), str(sample_id), str(success), *[f"{value:.10e}" for value in params]]
                handle.write("\t".join(row) + "\n")

    with curve_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(["bz", "t", "in_fit_window", "fit_mean", "fit_p16", "fit_p84"]) + "\n")
        for record in records:
            times = np.arange(record.tsrange_start, record.tsrange_end + 1, dtype=int)
            valid_samples = record.sample_params[np.all(np.isfinite(record.sample_params), axis=1)]
            if len(valid_samples) > 0:
                component_curves = np.array(
                    [
                        _select_curve_component(
                            evaluate_tmdwf_ratio(
                                times,
                                record.amplitudes,
                                record.energies,
                                params,
                                nt,
                                gm=record.gm,
                                pz=record.pz,
                                ns=record.ns,
                            ),
                            component,
                        )
                        for params in valid_samples
                    ]
                )
                center, low, high = _summarize_bootstrap_series(component_curves)
            else:
                params_mean, _ = summarize_parameter_samples(record.sample_params)
                center = _select_curve_component(
                    evaluate_tmdwf_ratio(
                        times,
                        record.amplitudes,
                        record.energies,
                        np.asarray(params_mean),
                        nt,
                        gm=record.gm,
                        pz=record.pz,
                        ns=record.ns,
                    ),
                    component,
                )
                low = np.full_like(center, np.nan)
                high = np.full_like(center, np.nan)
            for t, mean_value, low_value, high_value in zip(times, center, low, high, strict=True):
                handle.write(
                    "\t".join(
                        [
                            str(record.bz),
                            str(int(t)),
                            str(int(record.tmin <= int(t) <= record.tmax)),
                            f"{mean_value:.10e}",
                            f"{low_value:.10e}",
                            f"{high_value:.10e}",
                        ]
                    )
                    + "\n"
                )
    outputs = [summary_path, table_path, sample_path, curve_path]
    if make_plots:
        plot_series: list[RatioFitPlotSeries] = []
        for record in records:
            times = np.arange(record.tsrange_start, record.tsrange_end + 1, dtype=int)
            ratio_real_mean, ratio_real_p16, ratio_real_p84 = _summarize_bootstrap_series(np.real(record.ratio_samples))
            ratio_imag_mean, ratio_imag_p16, ratio_imag_p84 = _summarize_bootstrap_series(np.imag(record.ratio_samples))
            ratio_mean = ratio_real_mean if component == "real" else ratio_imag_mean
            ratio_err = (
                0.5 * (ratio_real_p84 - ratio_real_p16)
                if component == "real"
                else 0.5 * (ratio_imag_p84 - ratio_imag_p16)
            )
            valid_samples = record.sample_params[np.all(np.isfinite(record.sample_params), axis=1)]
            if len(valid_samples) > 0:
                component_curves = np.array(
                    [
                        _select_curve_component(
                            evaluate_tmdwf_ratio(
                                times,
                                record.amplitudes,
                                record.energies,
                                params,
                                nt,
                                gm=record.gm,
                                pz=record.pz,
                                ns=record.ns,
                            ),
                            component,
                        )
                        for params in valid_samples
                    ]
                )
                fit_mean, fit_p16, fit_p84 = _summarize_bootstrap_series(component_curves)
            else:
                params_mean, _ = summarize_parameter_samples(record.sample_params)
                fit_mean = _select_curve_component(
                    evaluate_tmdwf_ratio(
                        times,
                        record.amplitudes,
                        record.energies,
                        np.asarray(params_mean),
                        nt,
                        gm=record.gm,
                        pz=record.pz,
                        ns=record.ns,
                    ),
                    component,
                )
                fit_p16 = np.full_like(fit_mean, np.nan)
                fit_p84 = np.full_like(fit_mean, np.nan)
            plot_series.append(
                RatioFitPlotSeries(
                    bz=record.bz,
                    times=times,
                    ratio_mean=np.asarray(ratio_mean, dtype=float),
                    ratio_err=np.asarray(ratio_err, dtype=float),
                    fit_mean=np.asarray(fit_mean, dtype=float),
                    fit_p16=np.asarray(fit_p16, dtype=float),
                    fit_p84=np.asarray(fit_p84, dtype=float),
                )
            )
        outputs.append(
            plot_tmdwf_ratio_fit(
                plot_path,
                tuple(plot_series),
                component=component,
                fit_window=(records[0].tmin, records[0].tmax),
                title=f"{stem} {component} {nstates}state",
            )
        )
    return outputs


def _write_decay_constant_check_summary(
    output_root: Path,
    stem: str,
    records: tuple[TMDWFOutputRecord, ...],
    *,
    lattice_spacing_fm: float,
    pz: int,
    gm: str,
    eta: str,
    base_tmin: int,
    base_tmax: int,
) -> Path:
    summary_path = output_root / f"{stem}_decay_constant_check_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("decay_constant_check true\n")
        handle.write(f"pz {pz}\n")
        handle.write(f"gm {gm}\n")
        handle.write(f"eta {eta}\n")
        handle.write(f"lattice_spacing_fm {lattice_spacing_fm:.4g}\n")
        handle.write(f"base_tfit {base_tmin} {base_tmax}\n")
        handle.write(
            "component\tnstates\tbT\tbz\ttmin\ttmax\ttwo_point_fit_tmin\t"
            "decay_constant_gev_mean\tdecay_constant_gev_err\tchi2_dof\tpvalue\t"
            "two_point_fit_tmax\tsuccess_bootstrap_center\n"
        )
        sortable_records: list[tuple[float, TMDWFOutputRecord, float, float]] = []
        for record in records:
            _, _, m0_gev_mean, m0_gev_err = _summarize_ground_state_matrix_element(
                record.sample_params,
                lattice_spacing_fm=lattice_spacing_fm,
            )
            sort_key = -m0_gev_mean if np.isfinite(m0_gev_mean) else float("inf")
            sortable_records.append((sort_key, record, m0_gev_mean, m0_gev_err))
        for _, record, m0_gev_mean, m0_gev_err in sorted(
            sortable_records,
            key=lambda item: (item[0], item[1].component, item[1].nstates, item[1].two_point_fit_tmin, item[1].tmin, item[1].tmax),
        ):
            handle.write(
                "\t".join(
                    [
                        record.component,
                        str(record.nstates),
                        "0",
                        "0",
                        str(record.tmin),
                        str(record.tmax),
                        str(record.two_point_fit_tmin),
                        f"{m0_gev_mean:.4g}",
                        f"{m0_gev_err:.4g}",
                        f"{record.fit_result.chi2_dof:.4g}",
                        f"{record.fit_result.pvalue:.4g}",
                        str(record.two_point_fit_tmax),
                        str(int(record.fit_result.success)),
                    ]
                )
                + "\n"
            )
    return summary_path


def run_tmdwf_nstate_fit(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_fit_input(input_file, results_dir=results_dir)
    outputs: list[Path] = []
    fit_windows = load_fit_window_table(spec.fit_window)

    for pz in spec.pzlist:
        title = expand_template(spec.title_pattern, pz=pz)
        c2pt_path = expand_template(spec.c2pt, pz=pz)
        _, c2pt_raw = load_correlator_csv(c2pt_path)
        c2pt_processed = apply_fold_t(c2pt_raw, spec.nt, spec.fold_t)
        t0, t1 = spec.tsrange
        c2pt_selected = c2pt_processed[:, t0 : t1 + 1]

        dataset_root = spec.results_dir / title
        dataset_root.mkdir(parents=True, exist_ok=True)

        fit_reference_cache: dict[int, tuple[Path, int, int, np.ndarray, np.ndarray]] = {}
        two_point_window = spec.two_point_fit_window_by_pz.get(pz)
        if two_point_window is None:
            raise ValueError(f"missing two_point_fit_window_by_pz entry for pz={pz}")
        two_point_tmin, two_point_tmax = two_point_window
        for nstates in spec.nstates:
            fit_reference = resolve_two_point_fit_reference(
                spec.two_point_fit_root,
                title=title,
                nstates=nstates,
                tmin=two_point_tmin,
                tmax=two_point_tmax,
            )
            fit_reference_cache[nstates] = (
                fit_reference.path,
                fit_reference.tmin,
                fit_reference.tmax,
                fit_reference.amplitudes,
                fit_reference.energies,
            )

        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("h5py is required to load TMDWF HDF5 data") from exc

        for gm in spec.gmlist:
            qtmdwf_path = resolve_qtmdwf_h5_path(spec.qtmdwf_h5, pz=pz, gm=gm)
            with h5py.File(qtmdwf_path, "r") as qtmdwf_handle:
                for eta in spec.etalist:
                    if spec.decay_constant_check:
                        decay_constant_records: list[TMDWFOutputRecord] = []
                        fit_window = fit_windows.get((gm, pz), fit_windows.get((None, pz)))
                        if fit_window is None:
                            raise ValueError(f"missing fit_window entry for gm={gm}, pz={pz}")
                        fit_tmin, fit_tmax = fit_window
                        bT = 0
                        bz = 0
                        numerator_selected = _load_tmdwf_correlator_from_handle(
                            qtmdwf_handle,
                            spec.dataset_path_template,
                            gm=gm,
                            eta=eta,
                            pz=pz,
                            tdirs=spec.tdirlist,
                            bT=bT,
                            bz=bz,
                            nt=spec.nt,
                            ns=spec.ns,
                            file_label=qtmdwf_path,
                        )[:, t0 : t1 + 1]
                        numerator_binned = bin_samples(numerator_selected, binsize=spec.binsize)
                        denominator_binned = bin_samples(c2pt_selected, binsize=spec.binsize)
                        if numerator_binned.shape != denominator_binned.shape:
                            raise ValueError("numerator and denominator must have matching post-binning shapes")
                        n_cfg = numerator_binned.shape[0]
                        if n_cfg < 2:
                            raise ValueError("bootstrap requires at least two samples")
                        n_boot = n_cfg if spec.bootstrap_samples is None else spec.bootstrap_samples
                        draw_size = n_cfg if spec.bootstrap_size is None else spec.bootstrap_size
                        indices = common_bootstrap_indices(n_cfg, draw_size, seed=spec.seed, n_boot=n_boot)
                        numerator_boot = bootstrap_means(numerator_binned, indices=indices)
                        denominator_boot = bootstrap_means(denominator_binned, indices=indices)
                        ratio_samples = np.divide(
                            numerator_boot,
                            denominator_boot,
                            out=np.full_like(numerator_boot, np.nan + 0.0j),
                            where=denominator_boot != 0.0,
                        )
                        scan_windows = _iter_decay_constant_scan_windows(
                            fit_tmin,
                            fit_tmax,
                            max_t=ratio_samples.shape[1] - 1,
                        )
                        if not scan_windows:
                            raise ValueError(f"no valid decay-constant scan windows for gm={gm}, pz={pz}")
                        two_point_tmin_values = _iter_decay_constant_two_point_tmins(two_point_tmin, two_point_tmax)
                        if not two_point_tmin_values:
                            raise ValueError(
                                f"no valid decay-constant two-point tmin values for gm={gm}, pz={pz}"
                            )
                        decay_fit_reference_cache: dict[tuple[int, int], tuple[Path, int, np.ndarray, np.ndarray]] = {}
                        for scan_two_point_tmin in two_point_tmin_values:
                            for nstates in spec.nstates:
                                cache_key = (nstates, scan_two_point_tmin)
                                if cache_key not in decay_fit_reference_cache:
                                    fit_reference = resolve_two_point_fit_reference(
                                        spec.two_point_fit_root,
                                        title=title,
                                        nstates=nstates,
                                        tmin=scan_two_point_tmin,
                                        tmax=two_point_tmax,
                                    )
                                    decay_fit_reference_cache[cache_key] = (
                                        fit_reference.path,
                                        fit_reference.tmax,
                                        fit_reference.amplitudes,
                                        fit_reference.energies,
                                    )
                                fit_table_path, resolved_two_point_tmax, amplitudes, energies = decay_fit_reference_cache[cache_key]
                                for scan_tmin, scan_tmax in scan_windows:
                                    fit_result, sample_params = fit_tmdwf_component(
                                        ratio_samples,
                                        amplitudes,
                                        energies,
                                        spec.nt,
                                        pz,
                                        spec.ns,
                                        gm,
                                        scan_tmin,
                                        scan_tmax,
                                        "real",
                                    )
                                    record = TMDWFOutputRecord(
                                        bz=0,
                                        component="real",
                                        nstates=nstates,
                                        tmin=scan_tmin,
                                        tmax=scan_tmax,
                                        fit_result=fit_result,
                                        sample_params=sample_params,
                                        amplitudes=amplitudes,
                                        energies=energies,
                                        pz=pz,
                                        ns=spec.ns,
                                        gm=gm,
                                        two_point_fit_tmin=scan_two_point_tmin,
                                        two_point_fit_tmax=resolved_two_point_tmax,
                                        two_point_fit_table_resolved=str(fit_table_path),
                                        two_point_fit_tmax_source="config",
                                        tsrange_start=t0,
                                        tsrange_end=t1,
                                        ratio_samples=ratio_samples,
                                    )
                                    decay_constant_records.append(record)
                                    _, _, decay_constant_gev_mean, decay_constant_gev_err = _summarize_ground_state_matrix_element(
                                        sample_params,
                                        lattice_spacing_fm=spec.lattice_spacing_fm,
                                    )
                                    print(
                                        f"[decay_constant_check] pz={pz} gm={gm} eta={eta} "
                                        f"two_point_tmin={scan_two_point_tmin} tfit={scan_tmin}:{scan_tmax} "
                                        f"nstates={nstates} decay_constant={decay_constant_gev_mean:.4g} "
                                        f"+/- {decay_constant_gev_err:.4g} GeV chi2_dof={fit_result.chi2_dof:.4g}"
                                    )
                        summary_path = _write_decay_constant_check_summary(
                            dataset_root,
                            f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}",
                            tuple(decay_constant_records),
                            lattice_spacing_fm=spec.lattice_spacing_fm,
                            pz=pz,
                            gm=gm,
                            eta=eta,
                            base_tmin=fit_tmin,
                            base_tmax=fit_tmax,
                        )
                        outputs.append(summary_path)
                        continue
                    eta_ratio_tables: dict[int, Path] = {}
                    eta_curve_tables: dict[int, dict[str, dict[int, Path]]] = {}
                    eta_fit_tables: dict[int, dict[str, dict[int, Path]]] = {}
                    eta_sample_tables: dict[int, dict[str, dict[int, Path]]] = {}
                    selected_bT_values = (0,) if spec.decay_constant_check else spec.bTlist
                    bz_values = (0,) if spec.decay_constant_check else spec.bzlist
                    for bT in selected_bT_values:
                        grouped_records: dict[tuple[str, int], list[TMDWFOutputRecord]] = {}
                        grouped_ratio_records: list[TMDWFRatioRecord] = []
                        primary_nstate = spec.nstates[0]
                        for bz in bz_values:
                            numerator_selected = _load_tmdwf_correlator_from_handle(
                                qtmdwf_handle,
                                spec.dataset_path_template,
                                gm=gm,
                                eta=eta,
                                pz=pz,
                                tdirs=spec.tdirlist,
                                bT=bT,
                                bz=bz,
                                nt=spec.nt,
                                ns=spec.ns,
                                file_label=qtmdwf_path,
                            )[:, t0 : t1 + 1]
                            numerator_binned = bin_samples(numerator_selected, binsize=spec.binsize)
                            denominator_binned = bin_samples(c2pt_selected, binsize=spec.binsize)
                            if numerator_binned.shape != denominator_binned.shape:
                                raise ValueError("numerator and denominator must have matching post-binning shapes")
                            n_cfg = numerator_binned.shape[0]
                            if n_cfg < 2:
                                raise ValueError("bootstrap requires at least two samples")
                            n_boot = n_cfg if spec.bootstrap_samples is None else spec.bootstrap_samples
                            draw_size = n_cfg if spec.bootstrap_size is None else spec.bootstrap_size
                            indices = common_bootstrap_indices(n_cfg, draw_size, seed=spec.seed, n_boot=n_boot)
                            numerator_boot = bootstrap_means(numerator_binned, indices=indices)
                            denominator_boot = bootstrap_means(denominator_binned, indices=indices)
                            ratio_samples = np.divide(
                                numerator_boot,
                                denominator_boot,
                                out=np.full_like(numerator_boot, np.nan + 0.0j),
                                where=denominator_boot != 0.0,
                            )

                            combo_stem = f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_bT{bT}"
                            fit_window = fit_windows.get((gm, pz), fit_windows.get((None, pz)))
                            if fit_window is None:
                                raise ValueError(f"missing fit_window entry for gm={gm}, pz={pz}")
                            fit_tmin, fit_tmax = fit_window
                            for nstates in spec.nstates:
                                fit_table_path, two_point_tmin, two_point_tmax, amplitudes, energies = fit_reference_cache[nstates]
                                if nstates == primary_nstate:
                                    grouped_ratio_records.append(
                                        TMDWFRatioRecord(
                                            bz=bz,
                                            tmin=fit_tmin,
                                            tmax=fit_tmax,
                                            two_point_fit_table_resolved=str(fit_table_path),
                                            two_point_fit_tmax_source="config",
                                            two_point_fit_tmax=two_point_tmax,
                                            tsrange_start=t0,
                                            tsrange_end=t1,
                                            ratio_samples=ratio_samples,
                                        )
                                    )
                                for component in _component_list(spec.fit_component):
                                    fit_result, sample_params = fit_tmdwf_component(
                                        ratio_samples,
                                        amplitudes,
                                        energies,
                                        spec.nt,
                                        pz,
                                        spec.ns,
                                        gm,
                                        fit_tmin,
                                        fit_tmax,
                                        component,
                                    )
                                    grouped_records.setdefault((component, nstates), []).append(
                                        TMDWFOutputRecord(
                                            bz=bz,
                                            component=component,
                                            nstates=nstates,
                                            tmin=fit_tmin,
                                            tmax=fit_tmax,
                                            fit_result=fit_result,
                                            sample_params=sample_params,
                                            amplitudes=amplitudes,
                                            energies=energies,
                                        pz=pz,
                                        ns=spec.ns,
                                        gm=gm,
                                        two_point_fit_tmin=two_point_tmin,
                                        two_point_fit_tmax=two_point_tmax,
                                        two_point_fit_table_resolved=str(fit_table_path),
                                        two_point_fit_tmax_source="config",
                                        tsrange_start=t0,
                                            tsrange_end=t1,
                                            ratio_samples=ratio_samples,
                                        )
                                    )
                        ratio_table_path = _write_ratio_outputs(dataset_root, combo_stem, tuple(grouped_ratio_records))
                        outputs.append(ratio_table_path)
                        eta_ratio_tables[bT] = ratio_table_path
                        eta_curve_tables[bT] = {}
                        eta_fit_tables[bT] = {}
                        eta_sample_tables[bT] = {}
                        for (component, nstates), records in grouped_records.items():
                            outputs.extend(
                                _write_component_outputs(
                                    dataset_root,
                                    combo_stem,
                                    tuple(records),
                                    spec.nt,
                                    make_plots=spec.make_plots,
                                )
                            )
                            eta_curve_tables[bT].setdefault(component, {})[nstates] = (
                                dataset_root / "tables" / f"{combo_stem}_{component}_{nstates}state_curve.txt"
                            )
                            eta_fit_tables[bT].setdefault(component, {})[nstates] = (
                                dataset_root / "tables" / f"{combo_stem}_{component}_{nstates}state_fit.txt"
                            )
                            eta_sample_tables[bT].setdefault(component, {})[nstates] = (
                                dataset_root / "samples" / f"{combo_stem}_{component}_{nstates}state_samples.txt"
                            )
                    if spec.make_plots:
                        plots_dir = dataset_root / "plots"
                        plots_dir.mkdir(parents=True, exist_ok=True)
                        for component in _component_list(spec.fit_component):
                            for nstates in spec.nstates:
                                fit_tables_by_bT = {
                                    bT: eta_fit_tables[bT][component][nstates]
                                    for bT in selected_bT_values
                                    if component in eta_fit_tables[bT] and nstates in eta_fit_tables[bT][component]
                                }
                                if fit_tables_by_bT:
                                    outputs.append(
                                        plot_tmdwf_m0_from_fit_tables(
                                            plots_dir / f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_{component}_{nstates}state_m0_vs_bz.pdf",
                                            fit_tables_by_bT,
                                            component=component,
                                            nstates=nstates,
                                            title=f"{title} {gm} {eta} {component} {nstates}state m0 vs bz",
                                        )
                                    )
                    notebook_root = spec.results_dir / "notebook_plots" / title
                    notebook_generated = notebook_root / f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}"
                    notebook_path = notebook_root / f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_tmdwf_plots.ipynb"
                    outputs.append(
                        write_tmdwf_plot_notebook(
                            notebook_path=notebook_path,
                            notebook_output_dir=notebook_generated,
                            ratio_tables=eta_ratio_tables,
                            curve_tables=eta_curve_tables,
                            fit_tables=eta_fit_tables,
                            sample_tables=eta_sample_tables,
                            title=title,
                            gm=gm,
                            eta=eta,
                            pz=pz,
                            ns=spec.ns,
                            lattice_spacing_fm=spec.lattice_spacing_fm,
                        )
                    )
    return outputs
