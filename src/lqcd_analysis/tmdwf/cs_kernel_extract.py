from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .cs_kernel_matching import (
    FM_GEV,
    build_cs_dgamma,
    evaluate_type2_matching_correction,
    normalize_cs_scheme,
    perturbative_order_from_label,
)
from .fourier import load_tmdwf_fourier_sample_table, resolve_tmdwf_fourier_output_paths
from .io import expand_template
from .plotting import CSKernelBreakdownSeries, plot_tmdwf_cs_kernel_adjacent_breakdown, plot_tmdwf_cs_kernel_band
from ..common.parsing import parse_int_list_or_range, parse_optional_int


@dataclass(frozen=True)
class TMDWFCSKernelInput:
    title_pattern: str
    input_root: Path
    ns: int
    lattice_spacing_fm: float
    gmlist: tuple[str, ...]
    etalist: tuple[str, ...]
    component: str
    nstates: int
    normalization_mode: str
    mu: float
    scheme: str
    extraction_type: str
    pair_mode: str
    reference_p1: int | None
    kernel_labels: tuple[str, ...]
    bTlist: tuple[int, ...]
    pzlist: tuple[int, ...]
    x_window: tuple[float, float]
    make_plots: bool
    results_dir: Path


@dataclass(frozen=True)
class CSKernelObservable:
    x: np.ndarray
    samples: np.ndarray


def parse_tmdwf_cs_kernel_input(
    path: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> TMDWFCSKernelInput:
    file_path = Path(path)
    entries: dict[str, list[str]] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            entries[tokens[0]] = tokens[1:]

    required = {
        "title_pattern",
        "input_root",
        "ns",
        "lattice_spacing_fm",
        "gmlist",
        "etalist",
        "component",
        "nstates",
        "normalization_mode",
        "mu",
    }
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")
    if "bTlist" not in entries and "bTrange" not in entries:
        raise ValueError(f"missing required key in {file_path}: bTlist or bTrange")
    if "pzlist" not in entries and "pzrange" not in entries:
        raise ValueError(f"missing required key in {file_path}: pzlist or pzrange")
    kernel_tokens = entries.get("kernel_labels", entries.get("kernel"))
    if not kernel_tokens:
        raise ValueError(f"missing required key in {file_path}: kernel_labels")

    input_root = Path(entries["input_root"][0])
    if not input_root.exists():
        raise FileNotFoundError(f"TMDWF CS-kernel input_root does not exist: {input_root}")
    component = entries["component"][0].lower()
    if component not in {"real", "imag"}:
        raise ValueError("component must be one of: real, imag")
    normalization_mode = entries["normalization_mode"][0].lower()
    if normalization_mode not in {"raw", "mode1", "mode2", "mode3"}:
        raise ValueError("normalization_mode must be one of: raw, mode1, mode2, mode3")

    x_window_tokens = entries.get("x_window", ["0.2", "0.8"])
    if len(x_window_tokens) != 2:
        raise ValueError("x_window must provide exactly two numbers: xmin xmax")
    xmin = float(x_window_tokens[0])
    xmax = float(x_window_tokens[1])
    if xmin > xmax:
        raise ValueError("x_window must satisfy xmin <= xmax")

    scheme = normalize_cs_scheme(entries.get("scheme", ["CG"])[0])
    extraction_type = entries.get("extraction_type", ["type2"])[0].lower()
    if extraction_type != "type2":
        raise ValueError("TMDWF CS-kernel extraction currently supports only extraction_type type2")
    pair_mode = entries.get("pair_mode", ["all"])[0].lower()
    if pair_mode not in {"all", "adjacent", "fixed_p1"}:
        raise ValueError("pair_mode must be one of: all, adjacent, fixed_p1")
    reference_p1 = parse_optional_int(entries.get("reference_p1", ["auto"])[0])

    output_root = (
        Path(results_dir)
        if results_dir is not None
        else Path(entries.get("results_dir", [file_path.parent / "results_tmdwf_cs_kernel"])[0])
    )
    return TMDWFCSKernelInput(
        title_pattern=entries["title_pattern"][0],
        input_root=input_root,
        ns=int(entries["ns"][0]),
        lattice_spacing_fm=float(entries["lattice_spacing_fm"][0]),
        gmlist=tuple(entries["gmlist"]),
        etalist=tuple(entries["etalist"]),
        component=component,
        nstates=int(entries["nstates"][0]),
        normalization_mode=normalization_mode,
        mu=float(entries["mu"][0]),
        scheme=scheme,
        extraction_type=extraction_type,
        pair_mode=pair_mode,
        reference_p1=reference_p1,
        kernel_labels=tuple(kernel_tokens),
        bTlist=parse_int_list_or_range(entries, "bTlist", "bTrange"),
        pzlist=parse_int_list_or_range(entries, "pzlist", "pzrange"),
        x_window=(xmin, xmax),
        make_plots=entries.get("plot", ["false"])[0].lower() not in {"false", "0", "no"},
        results_dir=output_root,
    )


def momentum_unit_gev(ns: int, lattice_spacing_fm: float) -> float:
    return float(2.0 * np.pi / (ns * lattice_spacing_fm * FM_GEV))


def _legacy_quantile_triplet(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return np.nan, np.nan, np.nan
    q16, q50, q84 = np.percentile(values, [16.0, 50.0, 84.0])
    return float(q16), float(q50), float(q84)


def load_cs_kernel_observable(sample_path: str | Path) -> CSKernelObservable:
    x_grid, sample_table = load_tmdwf_fourier_sample_table(sample_path)
    return CSKernelObservable(x=x_grid, samples=sample_table)


def load_cs_kernel_dataset(
    *,
    input_root: Path,
    title_pattern: str,
    gm: str,
    eta: str,
    component: str,
    nstates: int,
    normalization_mode: str,
    bTlist: tuple[int, ...],
    pzlist: tuple[int, ...],
) -> dict[tuple[int, int], CSKernelObservable]:
    dataset: dict[tuple[int, int], CSKernelObservable] = {}
    reference_x: np.ndarray | None = None
    sample_count: int | None = None
    for bT in bTlist:
        for pz in pzlist:
            title = expand_template(title_pattern, pz=pz)
            fourier_table, sample_path = resolve_tmdwf_fourier_output_paths(
                input_root,
                title=title,
                gm=gm,
                eta=eta,
                bT=bT,
                component=component,
                nstates=nstates,
                normalization_mode=normalization_mode,
            )
            del fourier_table
            observable = load_cs_kernel_observable(sample_path)
            if reference_x is None:
                reference_x = observable.x
            elif not np.allclose(observable.x, reference_x, atol=1e-12, rtol=0.0):
                raise ValueError(f"inconsistent x-grid detected in CS-kernel Fourier inputs: {sample_path}")
            if sample_count is None:
                sample_count = observable.samples.shape[0]
            elif observable.samples.shape[0] != sample_count:
                raise ValueError(
                    "inconsistent bootstrap sample count detected in CS-kernel Fourier inputs: "
                    f"expected {sample_count}, found {observable.samples.shape[0]} in {sample_path}"
                )
            dataset[(bT, pz)] = observable
    return dataset


def compute_pairwise_type2_estimators(
    reference_samples: np.ndarray,
    comparison_samples: list[np.ndarray],
    *,
    scheme: str,
    kernel_label: str,
    mu: float,
    x_value: float,
    p1_gev: float,
    p2_gevs: list[float],
    component: str = "real",
) -> tuple[list[np.ndarray], np.ndarray]:
    if len(comparison_samples) != len(p2_gevs):
        raise ValueError("comparison_samples and p2_gevs must have the same length")
    if reference_samples.ndim != 1:
        raise ValueError("reference_samples must be one-dimensional")
    estimators: list[np.ndarray] = []
    sigmas: list[float] = []
    for comparison, p2_gev in zip(comparison_samples, p2_gevs):
        if comparison.ndim != 1 or comparison.shape != reference_samples.shape:
            raise ValueError("each comparison sample array must match reference_samples")
        log_ratio = 1.0 / np.log(p1_gev / p2_gev) * np.log(np.abs(reference_samples / comparison))
        correction = evaluate_type2_matching_correction(
            scheme=scheme,
            kernel_label=kernel_label,
            mu=mu,
            p1=p1_gev,
            p2=p2_gev,
            x=x_value,
            component=component,
        )
        estimator = log_ratio + correction
        q16, _, q84 = _legacy_quantile_triplet(estimator)
        estimators.append(estimator)
        sigmas.append(float(max(0.5 * (q84 - q16), np.finfo(float).eps)))
    return estimators, np.asarray(sigmas, dtype=float)


def fit_gamma_direct_across_p2(
    reference_samples: np.ndarray,
    comparison_samples: list[np.ndarray],
    *,
    scheme: str,
    kernel_label: str,
    mu: float,
    x_value: float,
    p1_gev: float,
    p2_gevs: list[float],
    component: str = "real",
) -> tuple[np.ndarray, np.ndarray]:
    if len(comparison_samples) != len(p2_gevs):
        raise ValueError("comparison_samples and p2_gevs must have the same length")
    if reference_samples.ndim != 1:
        raise ValueError("reference_samples must be one-dimensional")
    comparison_sigmas: list[float] = []
    for comparison in comparison_samples:
        q16, _, q84 = _legacy_quantile_triplet(comparison)
        sigma = 0.5 * (q84 - q16)
        comparison_sigmas.append(float(max(sigma, np.finfo(float).eps)))
    q16, _, q84 = _legacy_quantile_triplet(reference_samples)
    reference_sigma = float(max(0.5 * (q84 - q16), np.finfo(float).eps))

    gamma_samples = np.empty(reference_samples.size, dtype=float)
    chi2_samples = np.empty(reference_samples.size, dtype=float)
    for sample_id, reference_value in enumerate(reference_samples):
        sample_comparisons = np.array([comparison[sample_id] for comparison in comparison_samples], dtype=float)
        if not np.isfinite(reference_value) or not np.all(np.isfinite(sample_comparisons)):
            raise ValueError(
                "non-finite TMDWF sample encountered while fitting the CS-kernel; "
                f"check the bootstrap samples near x={x_value:.6f}, P1={p1_gev:.6f}"
            )
        sample_sigmas = np.sqrt(reference_sigma**2 + np.asarray(comparison_sigmas, dtype=float) ** 2)
        sample_sigmas = np.where(sample_sigmas > 0.0, sample_sigmas, np.finfo(float).eps)
        pair_corrections = np.asarray(
            [
                evaluate_type2_matching_correction(
                    scheme=scheme,
                    kernel_label=kernel_label,
                    mu=mu,
                    p1=p1_gev,
                    p2=p2_gev,
                    x=x_value,
                    component=component,
                )
                for p2_gev in p2_gevs
            ],
            dtype=float,
        )

        def residuals(params: np.ndarray) -> np.ndarray:
            gamma = float(params[0])
            model = reference_value * np.exp(np.log(np.asarray(p2_gevs, dtype=float) / p1_gev) * (gamma - pair_corrections))
            return (sample_comparisons - model) / sample_sigmas

        result = least_squares(residuals, np.array([0.0]), method="lm")
        dof = sample_comparisons.size - 1
        chi2_samples[sample_id] = float(np.sum(result.fun**2) / dof) if dof > 0 else float("nan")
        gamma_samples[sample_id] = float(result.x[0])
    return gamma_samples, chi2_samples


def extract_cs_kernel_for_reference(
    dataset: dict[tuple[int, int], CSKernelObservable],
    *,
    bT: int,
    reference_pz: int,
    comparison_pz_list: list[int],
    kernel_label: str,
    mu: float,
    scheme: str,
    component: str = "real",
    ns: int,
    lattice_spacing_fm: float,
    x_window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    reference = dataset[(bT, reference_pz)]
    x_mask = (reference.x >= x_window[0]) & (reference.x <= x_window[1])
    x_values = reference.x[x_mask]
    d_p = momentum_unit_gev(ns, lattice_spacing_fm)
    p1_gev = reference_pz * d_p
    p2_gevs = [pz * d_p for pz in comparison_pz_list]

    gamma_samples_by_x: list[np.ndarray] = []
    chi2_samples_by_x: list[np.ndarray] = []
    for local_index, x_value in zip(np.where(x_mask)[0], x_values):
        reference_samples = reference.samples[:, local_index]
        comparison_samples = [dataset[(bT, pz)].samples[:, local_index] for pz in comparison_pz_list]
        gamma_samples, chi2_samples = fit_gamma_direct_across_p2(
            reference_samples,
            comparison_samples,
            scheme=scheme,
            kernel_label=kernel_label,
            mu=mu,
            x_value=float(x_value),
            p1_gev=p1_gev,
            p2_gevs=p2_gevs,
            component=component,
        )
        gamma_samples_by_x.append(gamma_samples)
        chi2_samples_by_x.append(chi2_samples)
    return (
        np.asarray(x_values, dtype=float),
        np.asarray(gamma_samples_by_x, dtype=float),
        np.asarray(chi2_samples_by_x, dtype=float),
        p2_gevs,
    )


def build_cs_kernel_pair_jobs(
    pzlist: tuple[int, ...],
    pair_mode: str,
    reference_p1: int | None = None,
) -> list[tuple[int, list[int], str]]:
    if len(pzlist) < 2:
        raise ValueError("TMDWF CS-kernel extraction requires at least two pz values")
    fixed_reference_p1 = pzlist[0] if reference_p1 is None else reference_p1
    if pair_mode in {"all", "fixed_p1"} and fixed_reference_p1 not in pzlist:
        raise ValueError("reference_p1 must be one of the requested pz values")
    if pair_mode == "all":
        comparison_pz_list = [pz for pz in pzlist if pz != fixed_reference_p1]
        if not comparison_pz_list:
            raise ValueError("reference_p1 must leave at least one comparison pz value")
        return [(fixed_reference_p1, comparison_pz_list, f"{fixed_reference_p1}-{comparison_pz_list[-1]}")]
    if pair_mode == "adjacent":
        return [(pz1, [pz2], f"{pz1}-{pz2}") for pz1, pz2 in zip(pzlist[:-1], pzlist[1:])]
    if pair_mode == "fixed_p1":
        comparison_pz_list = [pz for pz in pzlist if pz != fixed_reference_p1]
        if not comparison_pz_list:
            raise ValueError("reference_p1 must leave at least one comparison pz value")
        return [(fixed_reference_p1, [pz2], f"{fixed_reference_p1}-{pz2}") for pz2 in comparison_pz_list]
    raise ValueError(f"unsupported pair_mode '{pair_mode}'")


def format_cs_kernel_pair_group_label(pair_mode: str, reference_pz: int, pair_label: str) -> str:
    if pair_mode == "fixed_p1":
        return f"fixedp1_refpz{reference_pz}_{pair_label}"
    return pair_label


def summarize_cs_kernel_adjacent_breakdown(
    dataset: dict[tuple[int, int], CSKernelObservable],
    *,
    bT: int,
    reference_pz: int,
    comparison_pz: int,
    kernel_label: str,
    mu: float,
    scheme: str,
    component: str = "real",
    ns: int,
    lattice_spacing_fm: float,
    x_window: tuple[float, float],
) -> CSKernelBreakdownSeries:
    reference = dataset[(bT, reference_pz)]
    comparison = dataset[(bT, comparison_pz)]
    x_mask = (reference.x >= x_window[0]) & (reference.x <= x_window[1])
    x_values = np.asarray(reference.x[x_mask], dtype=float)
    d_p = momentum_unit_gev(ns, lattice_spacing_fm)
    p1_gev = reference_pz * d_p
    p2_gev = comparison_pz * d_p

    log_ratio_p16: list[float] = []
    log_ratio_p50: list[float] = []
    log_ratio_p84: list[float] = []
    total_p16: list[float] = []
    total_p50: list[float] = []
    total_p84: list[float] = []
    matching_values: list[float] = []
    for local_index, x_value in zip(np.where(x_mask)[0], x_values):
        reference_samples = reference.samples[:, local_index]
        comparison_samples = comparison.samples[:, local_index]
        log_ratio = 1.0 / np.log(p1_gev / p2_gev) * np.log(np.abs(reference_samples / comparison_samples))
        correction = evaluate_type2_matching_correction(
            scheme=scheme,
            kernel_label=kernel_label,
            mu=mu,
            p1=p1_gev,
            p2=p2_gev,
            x=float(x_value),
            component=component,
        )
        total = log_ratio + correction
        l16, l50, l84 = _legacy_quantile_triplet(log_ratio)
        t16, t50, t84 = _legacy_quantile_triplet(total)
        log_ratio_p16.append(l16)
        log_ratio_p50.append(l50)
        log_ratio_p84.append(l84)
        total_p16.append(t16)
        total_p50.append(t50)
        total_p84.append(t84)
        matching_values.append(float(correction))
    return CSKernelBreakdownSeries(
        label=f"{reference_pz}-{comparison_pz}",
        x_values=x_values,
        log_ratio_p16=np.asarray(log_ratio_p16, dtype=float),
        log_ratio_p50=np.asarray(log_ratio_p50, dtype=float),
        log_ratio_p84=np.asarray(log_ratio_p84, dtype=float),
        matching=np.asarray(matching_values, dtype=float),
        total_p16=np.asarray(total_p16, dtype=float),
        total_p50=np.asarray(total_p50, dtype=float),
        total_p84=np.asarray(total_p84, dtype=float),
    )


def _write_cs_kernel_outputs(
    output_root: Path,
    stem: str,
    *,
    metadata_lines: list[str],
    x_values: np.ndarray,
    gamma_samples: np.ndarray,
    chi2_samples: np.ndarray,
    comparison_pz_list: list[int],
) -> list[Path]:
    tables_dir = output_root / "tables"
    samples_dir = output_root / "samples"
    diagnostics_dir = output_root / "diagnostics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / f"{stem}_summary.txt"
    band_path = tables_dir / f"{stem}_band.txt"
    diagnostics_path = diagnostics_dir / f"{stem}_diagnostics.txt"
    sample_path = samples_dir / f"{stem}_samples.txt"

    with summary_path.open("w", encoding="utf-8") as handle:
        for line in metadata_lines:
            handle.write(line + "\n")
        handle.write(f"comparison_pz_list {' '.join(str(value) for value in comparison_pz_list)}\n")
        handle.write(f"x_count {x_values.size}\n")

    with band_path.open("w", encoding="utf-8") as handle:
        for line in metadata_lines:
            handle.write(line + "\n")
        handle.write(f"comparison_pz_list {' '.join(str(value) for value in comparison_pz_list)}\n")
        handle.write("x\tgamma_p16\tgamma_p50\tgamma_p84\n")
        for x_value, samples in zip(x_values, gamma_samples):
            q16, q50, q84 = _legacy_quantile_triplet(samples)
            handle.write(f"{x_value:.10e}\t{q16:.10e}\t{q50:.10e}\t{q84:.10e}\n")

    with diagnostics_path.open("w", encoding="utf-8") as handle:
        for line in metadata_lines:
            handle.write(line + "\n")
        handle.write(f"comparison_pz_list {' '.join(str(value) for value in comparison_pz_list)}\n")
        handle.write("x\tchi2_dof_p16\tchi2_dof_p50\tchi2_dof_p84\tn_pairs\n")
        for x_value, samples in zip(x_values, chi2_samples):
            finite = samples[np.isfinite(samples)]
            q16, q50, q84 = _legacy_quantile_triplet(finite)
            handle.write(f"{x_value:.10e}\t{q16:.10e}\t{q50:.10e}\t{q84:.10e}\t{len(comparison_pz_list)}\n")

    with sample_path.open("w", encoding="utf-8") as handle:
        for line in metadata_lines:
            handle.write(line + "\n")
        handle.write(f"comparison_pz_list {' '.join(str(value) for value in comparison_pz_list)}\n")
        handle.write("x\tsample_id\tsuccess\tgamma_zeta\tchi2_dof\n")
        for x_value, gamma_by_sample, chi2_by_sample in zip(x_values, gamma_samples, chi2_samples):
            for sample_id, (gamma_value, chi2_value) in enumerate(zip(gamma_by_sample, chi2_by_sample)):
                success = int(np.isfinite(gamma_value))
                handle.write(
                    f"{x_value:.10e}\t{sample_id}\t{success}\t{gamma_value:.10e}\t{chi2_value:.10e}\n"
                )

    return [summary_path, band_path, diagnostics_path, sample_path]


def run_tmdwf_cs_kernel_workflow(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_cs_kernel_input(input_file, results_dir=results_dir)
    outputs: list[Path] = []
    output_title = expand_template(spec.title_pattern, pz="multiPz")
    for gm in spec.gmlist:
        for eta in spec.etalist:
            dataset = load_cs_kernel_dataset(
                input_root=spec.input_root,
                title_pattern=spec.title_pattern,
                gm=gm,
                eta=eta,
                component=spec.component,
                nstates=spec.nstates,
                normalization_mode=spec.normalization_mode,
                bTlist=spec.bTlist,
                pzlist=spec.pzlist,
            )
            output_root = spec.results_dir / output_title
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "plots").mkdir(parents=True, exist_ok=True)
            pair_jobs = build_cs_kernel_pair_jobs(spec.pzlist, spec.pair_mode, spec.reference_p1)
            adjacent_breakdown_series: list[CSKernelBreakdownSeries] = []
            for kernel_label in spec.kernel_labels:
                perturbative_order_from_label(kernel_label)
                correction = build_cs_dgamma(spec.mu, kernel_label)
                adjacent_breakdown_series.clear()
                for bT in spec.bTlist:
                    for reference_pz, comparison_pz_list, pz_label in pair_jobs:
                        x_values, gamma_samples, chi2_samples, p2_gevs = extract_cs_kernel_for_reference(
                            dataset,
                            bT=bT,
                            reference_pz=reference_pz,
                            comparison_pz_list=comparison_pz_list,
                            kernel_label=kernel_label,
                            mu=spec.mu,
                            scheme=spec.scheme,
                            component=spec.component,
                            ns=spec.ns,
                            lattice_spacing_fm=spec.lattice_spacing_fm,
                            x_window=spec.x_window,
                        )
                        pair_group_label = format_cs_kernel_pair_group_label(spec.pair_mode, reference_pz, pz_label)
                        stem = (
                            f"{output_title}_{gm}_{eta}_{spec.normalization_mode}_{spec.component}_{spec.nstates}state_"
                            f"{spec.scheme}_{kernel_label}_bT{bT}_refpz{pair_group_label}_{spec.extraction_type}"
                        )
                        metadata_lines = [
                            f"title_pattern {spec.title_pattern}",
                            f"output_title {output_title}",
                            f"gm {gm}",
                            f"eta {eta}",
                            f"component {spec.component}",
                            f"nstates {spec.nstates}",
                            f"normalization_mode {spec.normalization_mode}",
                            f"scheme {spec.scheme}",
                            f"kernel_label {kernel_label}",
                            f"extraction_type {spec.extraction_type}",
                            f"pair_mode {spec.pair_mode}",
                            f"reference_p1 {spec.reference_p1 if spec.reference_p1 is not None else spec.pzlist[0]}",
                            f"mu {spec.mu:.10e}",
                            f"alphas_mu {correction.alphas(spec.mu):.10e}",
                            f"reference_bT {bT}",
                            f"reference_pz {reference_pz}",
                            f"reference_pz_label {pz_label}",
                            f"pair_group_label {pair_group_label}",
                            f"x_window {spec.x_window[0]:.10e} {spec.x_window[1]:.10e}",
                            f"dP_GeV {momentum_unit_gev(spec.ns, spec.lattice_spacing_fm):.10e}",
                            f"comparison_p2_GeV {' '.join(f'{value:.10e}' for value in p2_gevs)}",
                        ]
                        outputs.extend(
                            _write_cs_kernel_outputs(
                                output_root,
                                stem,
                                metadata_lines=metadata_lines,
                                x_values=x_values,
                                gamma_samples=gamma_samples,
                                chi2_samples=chi2_samples,
                                comparison_pz_list=comparison_pz_list,
                            )
                        )
                        if spec.make_plots and spec.pair_mode not in {"adjacent", "fixed_p1"}:
                            plot_path = output_root / "plots" / f"{stem}_band.pdf"
                            q16 = np.array([_legacy_quantile_triplet(samples)[0] for samples in gamma_samples], dtype=float)
                            q50 = np.array([_legacy_quantile_triplet(samples)[1] for samples in gamma_samples], dtype=float)
                            q84 = np.array([_legacy_quantile_triplet(samples)[2] for samples in gamma_samples], dtype=float)
                            plot_tmdwf_cs_kernel_band(
                                plot_path,
                                x_values=x_values,
                                band_p16=q16,
                                band_p50=q50,
                                band_p84=q84,
                                title=output_title,
                                scheme=spec.scheme,
                                kernel_label=kernel_label,
                                bT=bT,
                                reference_pz=reference_pz,
                            )
                            outputs.append(plot_path)
                        if spec.make_plots and spec.pair_mode in {"adjacent", "fixed_p1"} and len(comparison_pz_list) == 1:
                            adjacent_breakdown_series.append(
                                summarize_cs_kernel_adjacent_breakdown(
                                    dataset,
                                    bT=bT,
                                    reference_pz=reference_pz,
                                        comparison_pz=comparison_pz_list[0],
                                        kernel_label=kernel_label,
                                    mu=spec.mu,
                                    scheme=spec.scheme,
                                    component=spec.component,
                                    ns=spec.ns,
                                    lattice_spacing_fm=spec.lattice_spacing_fm,
                                    x_window=spec.x_window,
                                )
                            )
                    if spec.make_plots and spec.pair_mode in {"adjacent", "fixed_p1"} and adjacent_breakdown_series:
                        breakdown_tag = (
                            "adjacent"
                            if spec.pair_mode == "adjacent"
                            else f"fixedp1_refpz{spec.reference_p1 if spec.reference_p1 is not None else spec.pzlist[0]}"
                        )
                        breakdown_path = (
                            output_root
                            / "plots"
                            / f"{output_title}_{gm}_{eta}_{spec.normalization_mode}_{spec.component}_{spec.nstates}state_"
                              f"{spec.scheme}_{kernel_label}_bT{bT}_{spec.extraction_type}_{breakdown_tag}_breakdown.pdf"
                        )
                        plot_tmdwf_cs_kernel_adjacent_breakdown(
                            breakdown_path,
                            tuple(adjacent_breakdown_series),
                            title=f"{output_title} {gm} {eta} bT={bT} {kernel_label} pairwise breakdown",
                        )
                        outputs.append(breakdown_path)
                        adjacent_breakdown_series.clear()
    return outputs
