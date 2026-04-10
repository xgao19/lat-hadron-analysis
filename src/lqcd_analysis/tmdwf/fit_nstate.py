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
    resolve_two_point_plateau_table,
)
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
    tmax: int | None
    shared_window_by_pz_gm: bool
    decay_constant: tuple[float, float] | None
    min_fit_dof: int
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
class SharedWindowScanRow:
    tmin: int
    tmax: int
    success: bool
    chi2_dof: float
    pvalue: float
    m0_mean: float
    fit_dof: int


@dataclass(frozen=True)
class SharedWindowCandidate:
    start_tmin: int
    end_tmin: int
    representative_tmin: int
    tmax: int
    length: int
    m0_mean: float
    chi2_dof: float
    overlaps_decay_constant: bool
    normalized_distance: float


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
    shared_window_flag: int
    reference_eta: str
    reference_bT: int
    reference_bz: int
    plateau_tmax_used: int
    two_point_plateau_table_resolved: str
    two_point_tmax_source: str
    two_point_tmax_inferred: str


def parse_optional_int(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    return int(value)


def _parse_decay_constant(entries: dict[str, list[str]]) -> tuple[float, float] | None:
    if "decay_constant" not in entries:
        return None
    tokens = entries["decay_constant"]
    if len(tokens) < 2:
        raise ValueError("decay_constant must provide: value error")
    return float(tokens[0]), float(tokens[1])


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
        "tmin",
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
    shared_window_by_pz_gm = parse_bool(entries.get("shared_window_by_pz_gm", ["false"])[0])
    decay_constant = _parse_decay_constant(entries)
    if shared_window_by_pz_gm and decay_constant is None:
        raise ValueError("decay_constant is required when shared_window_by_pz_gm is true")

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
        tmax=parse_optional_int(entries.get("tmax", ["auto"])[0]),
        shared_window_by_pz_gm=shared_window_by_pz_gm,
        decay_constant=decay_constant,
        min_fit_dof=int(entries.get("min_fit_dof", ["1"])[0]),
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


def scan_reference_tmin_rows(
    ratio_samples: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    *,
    nt: int,
    pz: int,
    ns: int,
    gm: str,
    tmin_start: int,
    tmax: int,
    nstates: int,
    min_fit_dof: int,
    component: str = "real",
) -> tuple[SharedWindowScanRow, ...]:
    rows: list[SharedWindowScanRow] = []
    for tmin in range(tmin_start, tmax + 1):
        n_points = tmax - tmin + 1
        fit_dof = n_points - nstates
        if fit_dof < min_fit_dof:
            continue
        fit_result = fit_tmdwf_mean_component(
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
        rows.append(
            SharedWindowScanRow(
                tmin=tmin,
                tmax=tmax,
                success=fit_result.success,
                chi2_dof=fit_result.chi2_dof,
                pvalue=fit_result.pvalue,
                m0_mean=float(fit_result.params[0]) if fit_result.params.size > 0 else np.nan,
                fit_dof=fit_dof,
            )
        )
    return tuple(rows)


def build_shared_window_candidates(
    rows: tuple[SharedWindowScanRow, ...],
    *,
    decay_constant: tuple[float, float],
) -> tuple[SharedWindowCandidate, ...]:
    valid_rows = [row for row in rows if row.success and np.isfinite(row.chi2_dof) and np.isfinite(row.m0_mean)]
    if not valid_rows:
        return ()
    valid_rows.sort(key=lambda row: row.tmin)
    target, target_err = decay_constant
    candidates: list[SharedWindowCandidate] = []
    start = 0
    while start < len(valid_rows):
        stop = start
        while stop + 1 < len(valid_rows) and valid_rows[stop + 1].tmin == valid_rows[stop].tmin + 1:
            stop += 1
        block = valid_rows[start : stop + 1]
        for block_start in range(len(block)):
            for block_stop in range(block_start, len(block)):
                window_rows = block[block_start : block_stop + 1]
                values = np.array([row.m0_mean for row in window_rows], dtype=float)
                mean = float(np.mean(values))
                if len(values) <= 1:
                    chi2_dof = 0.0
                else:
                    chi2 = float(np.sum(np.square(values - mean)))
                    chi2_dof = chi2 / (len(values) - 1)
                distance = abs(mean - target) if np.isfinite(mean) else np.nan
                overlaps = bool(np.isfinite(distance) and np.isfinite(target_err) and target_err >= 0.0 and distance <= target_err)
                normalized_distance = (
                    float(distance / target_err)
                    if np.isfinite(distance) and np.isfinite(target_err) and target_err > 0.0
                    else np.inf
                )
                start_tmin = window_rows[0].tmin
                end_tmin = window_rows[-1].tmin
                candidates.append(
                    SharedWindowCandidate(
                        start_tmin=start_tmin,
                        end_tmin=end_tmin,
                        representative_tmin=(start_tmin + end_tmin) // 2,
                        tmax=window_rows[0].tmax,
                        length=len(window_rows),
                        m0_mean=mean,
                        chi2_dof=chi2_dof,
                        overlaps_decay_constant=overlaps,
                        normalized_distance=normalized_distance,
                    )
                )
        start = stop + 1
    return tuple(candidates)


def select_best_shared_window(candidates: tuple[SharedWindowCandidate, ...]) -> SharedWindowCandidate:
    if not candidates:
        raise ValueError("no valid shared-window candidates are available")
    overlapping = [candidate for candidate in candidates if candidate.overlaps_decay_constant]
    if overlapping:
        return min(
            overlapping,
            key=lambda candidate: (
                candidate.chi2_dof,
                -candidate.length,
                -candidate.start_tmin,
                candidate.normalized_distance,
            ),
        )
    return min(
        candidates,
        key=lambda candidate: (
            candidate.normalized_distance,
            candidate.chi2_dof,
            -candidate.length,
            -candidate.start_tmin,
        ),
    )


def sanitize_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def _component_list(fit_component: str) -> tuple[str, ...]:
    return ("real", "imag") if fit_component == "both" else (fit_component,)


def _write_shared_window_summary(
    output_root: Path,
    stem: str,
    nstates: int,
    reference_dataset: str,
    rows: tuple[SharedWindowScanRow, ...],
    selected: SharedWindowCandidate,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{stem}_{nstates}state_shared_window.txt"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"reference_dataset {reference_dataset}\n")
        handle.write("selection_basis reference_mean_fits_only\n")
        handle.write(f"selected_tfit {selected.start_tmin} {selected.tmax}\n")
        handle.write(f"selected_window_tmin_range {selected.start_tmin} {selected.end_tmin}\n")
        handle.write(f"selected_window_length {selected.length}\n")
        handle.write(f"selected_window_m0_meanfit {selected.m0_mean:.10e}\n")
        handle.write(f"selected_window_chi2_dof {selected.chi2_dof:.10e}\n")
        handle.write(f"selected_window_overlaps_decay_constant {int(selected.overlaps_decay_constant)}\n")
        handle.write(f"selected_window_normalized_distance {selected.normalized_distance:.10e}\n")
        handle.write("candidate_rows\n")
        handle.write("tmin tmax success chi2_dof pvalue m0_meanfit fit_dof\n")
        for row in rows:
            handle.write(
                f"{row.tmin} {row.tmax} {int(row.success)} {row.chi2_dof:.10e} {row.pvalue:.10e} "
                f"{row.m0_mean:.10e} {row.fit_dof}\n"
            )
    return path


def _write_component_outputs(
    output_root: Path,
    stem: str,
    records: tuple[TMDWFOutputRecord, ...],
    nt: int,
) -> list[Path]:
    tables_dir = output_root / "tables"
    samples_dir = output_root / "samples"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return []
    component = records[0].component
    nstates = records[0].nstates
    summary_path = output_root / f"{stem}_{component}_{nstates}state_summary.txt"
    table_path = tables_dir / f"{stem}_{component}_{nstates}state_fit.txt"
    sample_path = samples_dir / f"{stem}_{component}_{nstates}state_samples.txt"
    curve_path = tables_dir / f"{stem}_{component}_{nstates}state_curve.txt"

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"component {component}\n")
        handle.write(f"nstates {nstates}\n")
        for record in records:
            params_mean, params_err = summarize_parameter_samples(record.sample_params)
            handle.write(f"begin_bz {record.bz}\n")
            handle.write(f"tfit {record.tmin} {record.tmax}\n")
            handle.write(f"two_point_plateau_table_resolved {record.two_point_plateau_table_resolved}\n")
            handle.write(f"two_point_tmax_source {record.two_point_tmax_source}\n")
            handle.write(f"two_point_tmax_inferred {record.two_point_tmax_inferred}\n")
            if record.shared_window_flag:
                handle.write(f"shared_tfit {record.tmin} {record.tmax}\n")
                handle.write(
                    f"shared_window_reference gm={record.gm} eta={record.reference_eta} pz={record.pz} "
                    f"bT={record.reference_bT} bz={record.reference_bz}\n"
                )
            handle.write(f"success_meanfit {int(record.fit_result.success)}\n")
            handle.write(f"chi2_dof {record.fit_result.chi2_dof:.10e}\n")
            handle.write(f"pvalue {record.fit_result.pvalue:.10e}\n")
            for idx in range(nstates):
                handle.write(f"m{idx} {params_mean[idx]:.10e} {params_err[idx]:.10e}\n")
            handle.write(f"end_bz {record.bz}\n")

    with table_path.open("w", encoding="utf-8") as handle:
        header = (
            ["bz", "tmin", "tmax", "success_meanfit", "chi2_dof", "pvalue", "shared_window_flag", "reference_eta", "reference_bT", "reference_bz", "plateau_tmax_used"]
            + [f"m{idx}_mean" for idx in range(nstates)]
            + [f"m{idx}_err" for idx in range(nstates)]
        )
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
                str(record.shared_window_flag),
                record.reference_eta,
                str(record.reference_bT),
                str(record.reference_bz),
                str(record.plateau_tmax_used),
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
        handle.write("\t".join(["bz", "t", "fit_mean", "fit_p16", "fit_p84"]) + "\n")
        for record in records:
            params_mean, _ = summarize_parameter_samples(record.sample_params)
            times = np.arange(record.tmin, record.tmax + 1)
            center = evaluate_tmdwf_ratio(
                times,
                record.amplitudes,
                record.energies,
                np.asarray(params_mean),
                nt,
                gm=record.gm,
                pz=record.pz,
                ns=record.ns,
            )
            valid_samples = record.sample_params[np.all(np.isfinite(record.sample_params), axis=1)]
            if len(valid_samples) > 0:
                curves = np.array(
                    [
                        evaluate_tmdwf_ratio(times, record.amplitudes, record.energies, params, nt, gm=record.gm, pz=record.pz, ns=record.ns)
                        for params in valid_samples
                    ]
                )
                low = np.percentile(curves, 16.0, axis=0)
                high = np.percentile(curves, 84.0, axis=0)
            else:
                low = np.full_like(center, np.nan)
                high = np.full_like(center, np.nan)
            for t, mean_value, low_value, high_value in zip(times, center, low, high, strict=True):
                handle.write(
                    "\t".join(
                        [
                            str(record.bz),
                            str(int(t)),
                            f"{mean_value:.10e}",
                            f"{low_value:.10e}",
                            f"{high_value:.10e}",
                        ]
                    )
                    + "\n"
                )
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
        qtmdwf_path = expand_template(spec.qtmdwf_h5, pz=pz)

        dataset_root = spec.results_dir / title
        dataset_root.mkdir(parents=True, exist_ok=True)

        plateau_cache: dict[int, tuple[Path, int | None, int, np.ndarray, np.ndarray]] = {}
        for nstates in spec.nstates:
            plateau_path, inferred_tmax = resolve_two_point_plateau_table(
                spec.two_point_plateau_table,
                pz=pz,
            )
            if spec.tmax is not None:
                effective_tmax = spec.tmax
            else:
                if inferred_tmax is None:
                    raise ValueError(
                        "could not infer tmax from plateau filename while tmax was omitted or set to auto: "
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

        with h5py.File(qtmdwf_path, "r") as qtmdwf_handle:
            for gm in spec.gmlist:
                shared_windows: dict[int, tuple[int, int, str]] = {}
                reference_combo = (spec.etalist[0], 0, 0) if spec.etalist else None
                for eta in spec.etalist:
                    for bT in spec.bTlist:
                        grouped_records: dict[tuple[str, int], list[TMDWFOutputRecord]] = {}
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
                            for nstates in spec.nstates:
                                plateau_path, inferred_tmax, effective_tmax, amplitudes, energies = plateau_cache[nstates]
                                fit_tmin = spec.tmin
                                fit_tmax = effective_tmax
                                shared_window_flag = 0
                                reference_eta_value = "none"
                                reference_bT_value = -1
                                reference_bz_value = -1
                                if spec.shared_window_by_pz_gm:
                                    if nstates not in shared_windows:
                                        if 0 not in spec.bTlist or 0 not in spec.bzlist or not spec.etalist:
                                            raise ValueError(
                                                "shared_window_by_pz_gm requires a reference dataset with "
                                                "bT=0, bz=0, and a non-empty etalist"
                                            )
                                        reference_eta = spec.etalist[0]
                                        reference_dataset = (
                                            f"gm={gm} eta={reference_eta} pz={pz} bT=0 bz=0"
                                        )
                                        if reference_combo == (eta, bT, bz):
                                            reference_ratio_samples = ratio_samples
                                        else:
                                            reference_numerator = _load_tmdwf_correlator_from_handle(
                                                qtmdwf_handle,
                                                spec.dataset_path_template,
                                                gm=gm,
                                                eta=reference_eta,
                                                pz=pz,
                                                tdirs=spec.tdirlist,
                                                bT=0,
                                                bz=0,
                                                nt=spec.nt,
                                                ns=spec.ns,
                                                file_label=qtmdwf_path,
                                            )[:, t0 : t1 + 1]
                                            reference_ratio_samples, _, _ = build_bootstrap_ratio_samples(
                                                reference_numerator,
                                                c2pt_selected,
                                                binsize=spec.binsize,
                                                bootstrap_samples=spec.bootstrap_samples,
                                                bootstrap_size=spec.bootstrap_size,
                                                seed=spec.seed,
                                            )
                                        rows = scan_reference_tmin_rows(
                                            reference_ratio_samples,
                                            amplitudes,
                                            energies,
                                            nt=spec.nt,
                                            pz=pz,
                                            ns=spec.ns,
                                            gm=gm,
                                            tmin_start=spec.tmin,
                                            tmax=effective_tmax,
                                            nstates=nstates,
                                            min_fit_dof=spec.min_fit_dof,
                                            component="real",
                                        )
                                        candidates = build_shared_window_candidates(
                                            rows,
                                            decay_constant=spec.decay_constant,
                                        )
                                        selected = select_best_shared_window(candidates)
                                        outputs.append(
                                            _write_shared_window_summary(
                                                dataset_root,
                                                f"{title}_{sanitize_token(gm)}",
                                                nstates,
                                                reference_dataset,
                                                rows,
                                                selected,
                                            )
                                        )
                                        shared_windows[nstates] = (
                                            selected.start_tmin,
                                            selected.tmax,
                                            reference_dataset,
                                        )
                                    fit_tmin, fit_tmax, _ = shared_windows[nstates]
                                    shared_window_flag = 1
                                    reference_eta_value = spec.etalist[0]
                                    reference_bT_value = 0
                                    reference_bz_value = 0
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
                                            shared_window_flag=shared_window_flag,
                                            reference_eta=reference_eta_value,
                                            reference_bT=reference_bT_value,
                                            reference_bz=reference_bz_value,
                                            plateau_tmax_used=effective_tmax,
                                            two_point_plateau_table_resolved=str(plateau_path),
                                            two_point_tmax_source="explicit" if spec.tmax is not None else "inferred",
                                            two_point_tmax_inferred="none" if inferred_tmax is None else str(inferred_tmax),
                                        )
                                    )
                        for (component, nstates), records in grouped_records.items():
                            outputs.extend(
                                _write_component_outputs(
                                    dataset_root,
                                    combo_stem,
                                    tuple(records),
                                    spec.nt,
                                )
                            )
    return outputs
