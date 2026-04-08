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
from .io import expand_template, load_tmdwf_correlator, load_two_point_plateau_values
from .models import evaluate_tmdwf_ratio, normalize_tmdwf_operator


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
    tmin: int
    tmax: int
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
        "tsrange",
        "tmin",
        "tmax",
    }
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")

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
        tmin=int(entries["tmin"][0]),
        tmax=int(entries["tmax"][0]),
        qtmdwf_h5=entries["qtmdwf_h5"][0],
        dataset_path_template=entries["dataset_path_template"][0],
        c2pt=entries["c2pt"][0],
        fold_t=parse_fold_t(entries),
        tsrange=(int(entries["tsrange"][0]), int(entries["tsrange"][1])),
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

    def residuals(params: np.ndarray, data: np.ndarray) -> np.ndarray:
        model_values = evaluate_tmdwf_ratio(times, amplitudes, energies, params, nt, gm=gm, pz=pz, ns=ns)
        return (model_values - data) / sigma

    theta0 = np.zeros(len(amplitudes), dtype=float)
    result = least_squares(residuals, theta0, args=(mean_data,), max_nfev=5000)
    chi2_value = float(np.dot(result.fun, result.fun))
    dof = max(len(times) - len(theta0), 1)
    chi2_dof = chi2_value / dof
    pvalue = float(1.0 - chi2.cdf(chi2_value, dof))
    meanfit = TMDWFFitResult(
        params=np.asarray(result.x, dtype=float),
        chi2=chi2_value,
        chi2_dof=chi2_dof,
        pvalue=pvalue,
        success=bool(result.success),
        message=result.message,
    )

    sample_params = np.full((ratio_samples.shape[0], len(amplitudes)), np.nan, dtype=float)
    for sample_id, sample_data in enumerate(data_samples):
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


def sanitize_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def _component_list(fit_component: str) -> tuple[str, ...]:
    return ("real", "imag") if fit_component == "both" else (fit_component,)


def _write_component_outputs(
    output_root: Path,
    stem: str,
    component: str,
    nstates: int,
    tmin: int,
    tmax: int,
    fit_result: TMDWFFitResult,
    sample_params: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    nt: int,
    pz: int,
    ns: int,
    gm: str,
) -> list[Path]:
    tables_dir = output_root / "tables"
    samples_dir = output_root / "samples"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    params_mean, params_err = summarize_parameter_samples(sample_params)
    summary_path = output_root / f"{stem}_{component}_{nstates}state_summary.txt"
    table_path = tables_dir / f"{stem}_{component}_{nstates}state_fit.txt"
    sample_path = samples_dir / f"{stem}_{component}_{nstates}state_samples.txt"
    curve_path = tables_dir / f"{stem}_{component}_{nstates}state_curve.txt"

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"component {component}\n")
        handle.write(f"nstates {nstates}\n")
        handle.write(f"tfit {tmin} {tmax}\n")
        handle.write(f"success_meanfit {int(fit_result.success)}\n")
        handle.write(f"chi2_dof {fit_result.chi2_dof:.10e}\n")
        handle.write(f"pvalue {fit_result.pvalue:.10e}\n")
        for idx in range(nstates):
            handle.write(f"m{idx} {params_mean[idx]:.10e} {params_err[idx]:.10e}\n")

    np.savetxt(
        table_path,
        np.array([[tmin, tmax, int(fit_result.success), fit_result.chi2_dof, fit_result.pvalue, *params_mean, *params_err]]),
        header=" ".join(
            ["tmin", "tmax", "success_meanfit", "chi2_dof", "pvalue"]
            + [f"m{idx}_mean" for idx in range(nstates)]
            + [f"m{idx}_err" for idx in range(nstates)]
        ),
        fmt="%.10e",
    )

    sample_rows = []
    for sample_id, params in enumerate(sample_params):
        success = int(np.all(np.isfinite(params)))
        sample_rows.append([sample_id, success, *params])
    np.savetxt(
        sample_path,
        np.asarray(sample_rows, dtype=float),
        header=" ".join(["sample_id", "success"] + [f"m{idx}" for idx in range(nstates)]),
        fmt="%.10e",
    )

    times = np.arange(tmin, tmax + 1)
    center = evaluate_tmdwf_ratio(times, amplitudes, energies, np.asarray(params_mean), nt, gm=gm, pz=pz, ns=ns)
    valid_samples = sample_params[np.all(np.isfinite(sample_params), axis=1)]
    if len(valid_samples) > 0:
        curves = np.array(
            [evaluate_tmdwf_ratio(times, amplitudes, energies, params, nt, gm=gm, pz=pz, ns=ns) for params in valid_samples]
        )
        low = np.percentile(curves, 16.0, axis=0)
        high = np.percentile(curves, 84.0, axis=0)
    else:
        low = np.full_like(center, np.nan)
        high = np.full_like(center, np.nan)
    np.savetxt(curve_path, np.column_stack([times, center, low, high]), header="t fit_mean fit_p16 fit_p84", fmt="%.10e")
    return [summary_path, table_path, sample_path, curve_path]


def run_tmdwf_nstate_fit(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_fit_input(input_file, results_dir=results_dir)
    outputs: list[Path] = []

    for pz in spec.pzlist:
        title = expand_template(spec.title_pattern, pz=pz)
        c2pt_path = expand_template(spec.c2pt, pz=pz)
        _, c2pt_raw = load_correlator_csv(c2pt_path)
        c2pt_processed = apply_fold_t(c2pt_raw, spec.nt, spec.fold_t)
        t0, t1 = spec.tsrange
        c2pt_selected = c2pt_processed[:, t0 : t1 + 1]

        dataset_root = spec.results_dir / title
        dataset_root.mkdir(parents=True, exist_ok=True)

        for gm in spec.gmlist:
            for eta in spec.etalist:
                for bT in spec.bTlist:
                    for bz in spec.bzlist:
                        numerator_selected = load_tmdwf_correlator(
                            expand_template(spec.qtmdwf_h5, pz=pz),
                            spec.dataset_path_template,
                            gm=gm,
                            eta=eta,
                            pz=pz,
                            tdirs=spec.tdirlist,
                            bT=bT,
                            bz=bz,
                            nt=spec.nt,
                            ns=spec.ns,
                        )[:, t0 : t1 + 1]
                        ratio_samples, _, _ = build_bootstrap_ratio_samples(
                            numerator_selected,
                            c2pt_selected,
                            binsize=spec.binsize,
                            bootstrap_samples=spec.bootstrap_samples,
                            bootstrap_size=spec.bootstrap_size,
                            seed=spec.seed,
                        )

                        combo_stem = (
                            f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_bT{bT}_bz{bz}"
                        )
                        for nstates in spec.nstates:
                            plateau_path = expand_template(spec.two_point_plateau_table, pz=pz)
                            amplitudes, energies = load_two_point_plateau_values(plateau_path, nstates)
                            for component in _component_list(spec.fit_component):
                                fit_result, sample_params = fit_tmdwf_component(
                                    ratio_samples,
                                    amplitudes,
                                    energies,
                                    spec.nt,
                                    pz,
                                    spec.ns,
                                    gm,
                                    spec.tmin,
                                    spec.tmax,
                                    component,
                                )
                                outputs.extend(
                                    _write_component_outputs(
                                        dataset_root,
                                        combo_stem,
                                        component,
                                        nstates,
                                        spec.tmin,
                                        spec.tmax,
                                        fit_result,
                                        sample_params,
                                        amplitudes,
                                        energies,
                                        spec.nt,
                                        pz,
                                        spec.ns,
                                        gm,
                                    )
                                )
    return outputs
