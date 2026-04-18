from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from ..common.utils import (
    apply_fold_t,
    parse_bool,
    parse_fold_t,
    robust_mean_and_error,
)
from ..two_point.io import load_correlator_csv
from .io import (
    _load_tmdwf_correlator_from_handle,
    expand_template,
    load_two_point_plateau_values,
    resolve_qtmdwf_h5_path,
    resolve_two_point_plateau_table,
)
from .models import evaluate_tmdwf_ratio, normalize_tmdwf_operator
from .plotting import RatioFitPlotSeries, plot_tmdwf_m0_from_fit_tables, plot_tmdwf_ratio_fit, write_tmdwf_plot_notebook


@dataclass(frozen=True)
class TMDWFNStateInput:
    title_pattern: str
    ns: int
    nt: int
    lattice_spacing_fm: float
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
    two_point_plateau_table: str
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
    fit_window_tmax_used: int
    two_point_fit_table_resolved: str
    two_point_tmax_source: str
    two_point_tmax_inferred: str
    tsrange_start: int
    tsrange_end: int
    ratio_samples: np.ndarray


@dataclass(frozen=True)
class TMDWFRatioRecord:
    bz: int
    tmin: int
    tmax: int
    two_point_fit_table_resolved: str
    two_point_tmax_source: str
    two_point_tmax_inferred: str
    tsrange_start: int
    tsrange_end: int
    ratio_samples: np.ndarray


def parse_optional_int(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    return int(value)


def _parse_int_list_or_range(entries: dict[str, list[str]], list_key: str, range_key: str) -> tuple[int, ...]:
    if list_key in entries:
        return tuple(int(item) for item in entries[list_key])
    if range_key in entries:
        start, stop = (int(item) for item in entries[range_key][:2])
        step = 1 if stop >= start else -1
        return tuple(range(start, stop + step, step))
    raise ValueError(f"missing required key: {list_key} or {range_key}")


def _parse_tsrange(entries: dict[str, list[str]], nt: int) -> tuple[int, int]:
    if "tsrange" in entries:
        return int(entries["tsrange"][0]), int(entries["tsrange"][1])
    return 0, max(0, nt // 2 - 1)


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
        "two_point_plateau_table",
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
    return TMDWFNStateInput(
        title_pattern=first_tokens[0],
        ns=int(first_tokens[1]),
        nt=int(first_tokens[2]),
        lattice_spacing_fm=float(first_tokens[3]),
        fit_target=fit_target,
        fit_component=fit_component,
        nstates=nstates,
        pzlist=tuple(int(item) for item in entries["pzlist"]),
        gmlist=tuple(entries["gmlist"]),
        etalist=tuple(entries["etalist"]),
        tdirlist=tuple(entries["Tdirlist"]),
        bTlist=_parse_int_list_or_range(entries, "bTlist", "bTrange"),
        bzlist=_parse_int_list_or_range(entries, "bzlist", "bzrange"),
        binsize=int(entries.get("binsize", ["1"])[0]),
        bootstrap_samples=parse_optional_int(entries.get("bootstrap_samples", ["auto"])[0]),
        bootstrap_size=parse_optional_int(entries.get("bootstrap_size", ["auto"])[0]),
        seed=int(entries.get("seed", ["2026"])[0]),
        fit_window=entries["fit_window"][0],
        qtmdwf_h5=entries["qtmdwf_h5"][0],
        dataset_path_template=entries["dataset_path_template"][0],
        c2pt=entries["c2pt"][0],
        fold_t=parse_fold_t(entries),
        tsrange=_parse_tsrange(entries, int(first_tokens[2])),
        two_point_plateau_table=entries["two_point_plateau_table"][0],
        make_plots=parse_bool(entries.get("plot", ["false"])[0]),
        results_dir=(
            Path(entries["results_dir"][0])
            if "results_dir" in entries and results_dir is None
            else ((input_path.parent / "results_tmdwf_fit") if results_dir is None else Path(results_dir))
        ),
    )


def bootstrap_indices(
    n_cfg: int,
    n_samples: int | None,
    sample_size: int | None,
    seed: int | None,
) -> np.ndarray:
    if n_cfg < 2:
        raise ValueError("bootstrap requires at least two samples")
    n_boot = n_cfg if n_samples is None else n_samples
    draw_size = n_cfg if sample_size is None else sample_size
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_cfg, size=(n_boot, draw_size))


def bootstrap_means_from_indices(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.complex128)[indices].mean(axis=1)


def _bin_numeric_samples(values: np.ndarray, binsize: int = 1) -> np.ndarray:
    samples = np.asarray(values)
    if samples.ndim != 2:
        raise ValueError("samples must be two-dimensional")
    if binsize < 1:
        raise ValueError("binsize must be positive")
    if binsize == 1:
        return samples.copy()

    n_cfg = samples.shape[0]
    n_bins = n_cfg // binsize
    if n_bins < 2:
        raise ValueError("binning leaves fewer than two bins")
    trimmed = samples[: n_bins * binsize]
    return trimmed.reshape(n_bins, binsize, samples.shape[1]).mean(axis=1)


def build_bootstrap_ratio_samples(
    numerator_correlators: np.ndarray,
    denominator_correlators: np.ndarray,
    *,
    binsize: int = 1,
    bootstrap_samples: int | None = None,
    bootstrap_size: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerator_binned = _bin_numeric_samples(np.asarray(numerator_correlators, dtype=np.complex128), binsize=binsize)
    denominator_binned = _bin_numeric_samples(np.asarray(denominator_correlators, dtype=float), binsize=binsize)
    if numerator_binned.shape != denominator_binned.shape:
        raise ValueError("numerator and denominator must have matching post-binning shapes")
    indices = bootstrap_indices(numerator_binned.shape[0], bootstrap_samples, bootstrap_size, seed)
    numerator_boot = bootstrap_means_from_indices(numerator_binned, indices)
    denominator_boot = bootstrap_means_from_indices(denominator_binned, indices)
    ratio_boot = np.divide(
        numerator_boot,
        denominator_boot,
        out=np.full_like(numerator_boot, np.nan + 0.0j),
        where=denominator_boot != 0.0,
    )
    return ratio_boot, numerator_boot, denominator_boot


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
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, 1e-12)
    return _PreparedFitData(
        times=times,
        data_samples=data_samples,
        mean_data=mean_data,
        sigma=sigma,
    )


def fit_tmdwf_mean_component(
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
) -> TMDWFFitResult:
    prepared = _prepare_fit_data(ratio_samples, tmin=tmin, tmax=tmax, component=component)

    def residuals(params: np.ndarray, data: np.ndarray) -> np.ndarray:
        model_values = evaluate_tmdwf_ratio(prepared.times, amplitudes, energies, params, nt, gm=gm, pz=pz, ns=ns)
        return (model_values - data) / prepared.sigma

    theta0 = np.zeros(len(amplitudes), dtype=float)
    result = least_squares(residuals, theta0, args=(prepared.mean_data,), max_nfev=5000)
    chi2_value = float(np.dot(result.fun, result.fun))
    dof = max(len(prepared.times) - len(theta0), 1)
    chi2_dof = chi2_value / dof
    pvalue = float(1.0 - chi2.cdf(chi2_value, dof))
    return TMDWFFitResult(
        params=np.asarray(result.x, dtype=float),
        chi2=chi2_value,
        chi2_dof=chi2_dof,
        pvalue=pvalue,
        success=bool(result.success),
        message=result.message,
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
    meanfit = fit_tmdwf_mean_component(
        ratio_samples,
        amplitudes,
        energies,
        nt,
        pz,
        ns,
        gm,
        tmin,
        tmax,
        component,
    )

    def residuals(params: np.ndarray, data: np.ndarray) -> np.ndarray:
        model_values = evaluate_tmdwf_ratio(prepared.times, amplitudes, energies, params, nt, gm=gm, pz=pz, ns=ns)
        return (model_values - data) / prepared.sigma

    sample_params = np.full((ratio_samples.shape[0], len(amplitudes)), np.nan, dtype=float)
    for sample_id, sample_data in enumerate(prepared.data_samples):
        sample_result = least_squares(residuals, meanfit.params, args=(sample_data,), max_nfev=5000)
        if sample_result.success:
            sample_params[sample_id] = sample_result.x
    return meanfit, sample_params


def summarize_parameter_samples(samples: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    means: list[float] = []
    errors: list[float] = []
    for column in range(samples.shape[1]):
        valid = samples[:, column][np.isfinite(samples[:, column])]
        mean, err = robust_mean_and_error(valid)
        means.append(mean)
        errors.append(err)
    return tuple(means), tuple(errors)


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


def _load_fit_window_table(
    path: str | Path,
) -> dict[tuple[str | None, int], tuple[int, int]]:
    file_path = Path(path)
    fit_windows: dict[tuple[str | None, int], tuple[int, int]] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if tokens[0].lower() in {"pz", "gm"}:
                continue
            if len(tokens) == 3:
                gm = None
                pz_text, tmin_text, tmax_text = tokens
            elif len(tokens) >= 4:
                gm = tokens[0]
                pz_text, tmin_text, tmax_text = tokens[1:4]
            else:
                raise ValueError(
                    f"invalid fit_window row at {file_path}:{line_number}; "
                    "expected: pz tmin tmax or gm pz tmin tmax"
                )
            pz = int(pz_text)
            tmin = int(tmin_text)
            tmax = int(tmax_text)
            if tmax < tmin:
                raise ValueError(
                    f"invalid fit window at {file_path}:{line_number}; tmax must be >= tmin"
                )
            fit_windows[(gm, pz)] = (tmin, tmax)
    return fit_windows


def _resolve_fit_window(
    fit_windows: dict[tuple[str | None, int], tuple[int, int]],
    *,
    gm: str,
    pz: int,
) -> tuple[int, int] | None:
    return fit_windows.get((gm, pz), fit_windows.get((None, pz)))


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
        handle.write(f"two_point_tmax_source {metadata.two_point_tmax_source}\n")
        handle.write(f"two_point_tmax_inferred {metadata.two_point_tmax_inferred}\n")
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
            handle.write(f"two_point_tmax_source {record.two_point_tmax_source}\n")
            handle.write(f"two_point_tmax_inferred {record.two_point_tmax_inferred}\n")
            handle.write(f"success_meanfit {int(record.fit_result.success)}\n")
            handle.write(f"chi2_dof {record.fit_result.chi2_dof:.10e}\n")
            handle.write(f"pvalue {record.fit_result.pvalue:.10e}\n")
            for idx in range(nstates):
                handle.write(f"m{idx} {params_mean[idx]:.10e} {params_err[idx]:.10e}\n")
            handle.write(f"end_bz {record.bz}\n")

    with table_path.open("w", encoding="utf-8") as handle:
        header = ["bz", "tmin", "tmax", "success_meanfit", "chi2_dof", "pvalue", "fit_window_tmax_used"]
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
                str(record.fit_window_tmax_used),
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


def run_tmdwf_nstate_fit(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_fit_input(input_file, results_dir=results_dir)
    outputs: list[Path] = []
    fit_windows = _load_fit_window_table(spec.fit_window)

    for pz in spec.pzlist:
        title = expand_template(spec.title_pattern, pz=pz)
        c2pt_path = expand_template(spec.c2pt, pz=pz)
        _, c2pt_raw = load_correlator_csv(c2pt_path)
        c2pt_processed = apply_fold_t(c2pt_raw, spec.nt, spec.fold_t)
        t0, t1 = spec.tsrange
        c2pt_selected = c2pt_processed[:, t0 : t1 + 1]

        dataset_root = spec.results_dir / title
        dataset_root.mkdir(parents=True, exist_ok=True)

        plateau_cache: dict[int, tuple[Path, int | None, int, np.ndarray, np.ndarray]] = {}
        for nstates in spec.nstates:
            plateau_path, inferred_tmax = resolve_two_point_plateau_table(
                spec.two_point_plateau_table,
                pz=pz,
            )
            if inferred_tmax is None:
                raise ValueError(
                    "could not infer tmax from plateau filename: "
                    f"{plateau_path}"
                )
            effective_tmax = inferred_tmax
            amplitudes, energies = load_two_point_plateau_values(plateau_path, nstates)
            plateau_cache[nstates] = (plateau_path, inferred_tmax, effective_tmax, amplitudes, energies)
            if "#" in str(spec.two_point_plateau_table):
                print(
                    f"[tmdwf-fit] Selected plateau table {plateau_path} with inferred tmax={inferred_tmax} for pz={pz}, nstates={nstates}."
                )

        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("h5py is required to load TMDWF HDF5 data") from exc

        for gm in spec.gmlist:
            qtmdwf_path = resolve_qtmdwf_h5_path(spec.qtmdwf_h5, pz=pz, gm=gm)
            with h5py.File(qtmdwf_path, "r") as qtmdwf_handle:
                for eta in spec.etalist:
                    eta_ratio_tables: dict[int, Path] = {}
                    eta_curve_tables: dict[int, dict[str, dict[int, Path]]] = {}
                    eta_fit_tables: dict[int, dict[str, dict[int, Path]]] = {}
                    eta_sample_tables: dict[int, dict[str, dict[int, Path]]] = {}
                    for bT in spec.bTlist:
                        grouped_records: dict[tuple[str, int], list[TMDWFOutputRecord]] = {}
                        grouped_ratio_records: list[TMDWFRatioRecord] = []
                        primary_nstate = spec.nstates[0]
                        for bz in spec.bzlist:
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
                            ratio_samples, _, _ = build_bootstrap_ratio_samples(
                                numerator_selected,
                                c2pt_selected,
                                binsize=spec.binsize,
                                bootstrap_samples=spec.bootstrap_samples,
                                bootstrap_size=spec.bootstrap_size,
                                seed=spec.seed,
                            )

                            combo_stem = f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_bT{bT}"
                            fit_window = _resolve_fit_window(fit_windows, gm=gm, pz=pz)
                            if fit_window is None:
                                raise ValueError(f"missing fit_window entry for gm={gm}, pz={pz}")
                            fit_tmin, fit_tmax = fit_window
                            for nstates in spec.nstates:
                                plateau_path, inferred_tmax, effective_tmax, amplitudes, energies = plateau_cache[nstates]
                                if nstates == primary_nstate:
                                    grouped_ratio_records.append(
                                        TMDWFRatioRecord(
                                            bz=bz,
                                            tmin=fit_tmin,
                                            tmax=fit_tmax,
                                            two_point_fit_table_resolved=str(plateau_path),
                                            two_point_tmax_source="inferred",
                                            two_point_tmax_inferred="none" if inferred_tmax is None else str(inferred_tmax),
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
                                            fit_window_tmax_used=effective_tmax,
                                            two_point_fit_table_resolved=str(plateau_path),
                                            two_point_tmax_source="inferred",
                                            two_point_tmax_inferred="none" if inferred_tmax is None else str(inferred_tmax),
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
                                    for bT in spec.bTlist
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
