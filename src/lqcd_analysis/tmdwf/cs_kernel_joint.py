from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from .cs_kernel_extract import (
    CSKernelObservable,
    _legacy_quantile_triplet,
    load_cs_kernel_dataset,
    momentum_unit_gev,
)
from .cs_kernel_matching import (
    evaluate_type2_matching_correction,
    normalize_cs_scheme,
    perturbative_order_from_label,
)
from .plotting import (
    plot_tmdwf_joint_cs_kernel_x_band,
    plot_tmdwf_joint_cs_kernel_pz_diagnostics,
    write_tmdwf_cs_kernel_joint_diagnostics_notebook,
)


@dataclass(frozen=True)
class JointCSEnsembleInput:
    label: str
    input_root: Path
    title_pattern: str
    ns: int
    lattice_spacing_fm: float
    pzlist: tuple[int, ...]
    bTlist: tuple[int, ...]


@dataclass(frozen=True)
class TMDWFCSKernelJointInput:
    ensembles: tuple[JointCSEnsembleInput, ...]
    gm: str
    eta: str
    component: str
    nstates: int
    normalization_mode: str
    mu: float
    scheme: str
    kernel_label: str
    reference_p1_gev: float
    x_window: tuple[float, float]
    x_knots: np.ndarray | None
    bT_knots_fm: np.ndarray | None
    spline_kind: str
    make_plots: bool
    show_progress: bool
    progress_every: int | None
    results_dir: Path


@dataclass(frozen=True)
class JointCSObservation:
    group_id: int
    sample_id: int
    x: float
    bT_fm: float
    pz_gev: float
    value: float
    sigma: float
    ensemble_label: str


@dataclass(frozen=True)
class PerXFitResult:
    x_actual: float
    bT_knots_fm: np.ndarray
    coeff_samples: np.ndarray
    chi2_dof: np.ndarray
    n_observations: int
    n_groups: int


@dataclass(frozen=True)
class DiagnosticGroupData:
    ensemble_label: str
    bT_fm: float
    pz_gev: np.ndarray
    data_median: np.ndarray
    data_p16: np.ndarray
    data_p84: np.ndarray
    model_median: np.ndarray
    model_p16: np.ndarray
    model_p84: np.ndarray


# ---------------------------------------------------------------------------
# Input parsing (unchanged)
# ---------------------------------------------------------------------------

def _parse_int_items(value: str) -> tuple[int, ...]:
    output: list[int] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" in token:
            start, stop = (int(part) for part in token.split(":", 1))
            step = 1 if stop >= start else -1
            output.extend(range(start, stop + step, step))
        else:
            output.append(int(token))
    if not output:
        raise ValueError("integer list must not be empty")
    return tuple(output)


def _parse_float_items(tokens: list[str]) -> np.ndarray:
    return np.asarray([float(token) for token in tokens], dtype=float)


def _parse_ensemble(tokens: list[str], path: Path) -> JointCSEnsembleInput:
    if len(tokens) < 7:
        raise ValueError(
            "ensemble entries in "
            f"{path} must provide: label input_root title_pattern ns lattice_spacing_fm "
            "pz=... bT=..."
        )
    options: dict[str, str] = {}
    for token in tokens[5:]:
        if "=" not in token:
            raise ValueError(f"ensemble option must use key=value syntax in {path}: {token}")
        key, value = token.split("=", 1)
        options[key] = value
    if "pz" not in options or "bT" not in options:
        raise ValueError(f"ensemble entry in {path} must include pz=... and bT=...")
    input_root = Path(tokens[1])
    if not input_root.exists():
        raise FileNotFoundError(f"joint CS-kernel ensemble input_root does not exist: {input_root}")
    return JointCSEnsembleInput(
        label=tokens[0],
        input_root=input_root,
        title_pattern=tokens[2],
        ns=int(tokens[3]),
        lattice_spacing_fm=float(tokens[4]),
        pzlist=_parse_int_items(options["pz"]),
        bTlist=_parse_int_items(options["bT"]),
    )


def parse_tmdwf_cs_kernel_joint_input(
    path: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> TMDWFCSKernelJointInput:
    file_path = Path(path)
    entries: dict[str, list[str]] = {}
    ensembles: list[JointCSEnsembleInput] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if tokens[0] == "ensemble":
                ensembles.append(_parse_ensemble(tokens[1:], file_path))
            else:
                entries[tokens[0]] = tokens[1:]

    required = {
        "gm",
        "eta",
        "component",
        "nstates",
        "normalization_mode",
        "mu",
        "kernel_label",
        "reference_p1_gev",
    }
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")
    if not ensembles:
        raise ValueError(f"missing ensemble entries in {file_path}")

    component = entries["component"][0].lower()
    if component not in {"real", "imag"}:
        raise ValueError("component must be one of: real, imag")
    normalization_mode = entries["normalization_mode"][0].lower()
    if normalization_mode not in {"raw", "mode1", "mode2", "mode3"}:
        raise ValueError("normalization_mode must be one of: raw, mode1, mode2, mode3")
    x_window_tokens = entries.get("x_window", ["0.2", "0.8"])
    if len(x_window_tokens) != 2:
        raise ValueError("x_window must provide exactly two values")
    x_window = (float(x_window_tokens[0]), float(x_window_tokens[1]))
    if x_window[0] > x_window[1]:
        raise ValueError("x_window must satisfy xmin <= xmax")

    kernel_label = entries["kernel_label"][0]
    perturbative_order_from_label(kernel_label)
    spline_kind = entries.get("spline_kind", ["linear"])[0].lower()
    if spline_kind not in {"linear", "cubic"}:
        raise ValueError("spline_kind must be one of: linear, cubic")
    show_progress = entries.get("progress", ["true"])[0].lower() not in {"false", "0", "no"}
    progress_every = int(entries["progress_every"][0]) if "progress_every" in entries else None
    if progress_every is not None and progress_every < 1:
        raise ValueError("progress_every must be positive")
    default_results_dir = file_path.parent / "results_tmdwf_cs_kernel_joint"
    output_root = Path(results_dir) if results_dir is not None else Path(
        entries.get("results_dir", [default_results_dir])[0]
    )
    return TMDWFCSKernelJointInput(
        ensembles=tuple(ensembles),
        gm=entries["gm"][0],
        eta=entries["eta"][0],
        component=component,
        nstates=int(entries["nstates"][0]),
        normalization_mode=normalization_mode,
        mu=float(entries["mu"][0]),
        scheme=normalize_cs_scheme(entries.get("scheme", ["CG"])[0]),
        kernel_label=kernel_label,
        reference_p1_gev=float(entries["reference_p1_gev"][0]),
        x_window=x_window,
        x_knots=(
            _parse_float_items(entries["x_knots"]) if "x_knots" in entries else None
        ),
        bT_knots_fm=(
            _parse_float_items(entries["bT_knots_fm"])
            if "bT_knots_fm" in entries
            else None
        ),
        spline_kind=spline_kind,
        make_plots=entries.get("plot", ["true"])[0].lower() not in {"false", "0", "no"},
        show_progress=show_progress,
        progress_every=progress_every,
        results_dir=output_root,
    )


# ---------------------------------------------------------------------------
# Spline basis (unchanged, now used for bT only)
# ---------------------------------------------------------------------------

def _spline_basis(values: np.ndarray, knots: np.ndarray, *, kind: str) -> np.ndarray:
    if knots.ndim != 1 or knots.size == 0:
        raise ValueError("spline knots must contain at least one value")
    if knots.size == 1:
        return np.ones((values.size, 1), dtype=float)
    if np.any(np.diff(knots) <= 0.0):
        raise ValueError("spline knots must be strictly increasing")
    normalized = kind.lower()
    if normalized not in {"linear", "cubic"}:
        raise ValueError("spline_kind must be one of: linear, cubic")
    if normalized == "cubic" and knots.size < 3:
        raise ValueError("cubic spline_kind requires at least three knots")
    columns = []
    for idx in range(knots.size):
        unit = np.zeros(knots.size, dtype=float)
        unit[idx] = 1.0
        if normalized == "linear":
            column = np.interp(values, knots, unit, left=0.0, right=0.0)
        else:
            spline = CubicSpline(knots, unit, bc_type="natural", extrapolate=False)
            column = np.asarray(spline(values), dtype=float)
            column = np.where(np.isfinite(column), column, 0.0)
        columns.append(column)
    return np.asarray(columns, dtype=float).T


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

EnsembleDataset = tuple[JointCSEnsembleInput, dict[tuple[int, int], CSKernelObservable]]


def _preload_datasets(spec: TMDWFCSKernelJointInput) -> tuple[list[EnsembleDataset], np.ndarray, int]:
    """Load all ensemble datasets and validate consistency across ensembles."""
    datasets: list[EnsembleDataset] = []
    reference_x: np.ndarray | None = None
    sample_count: int | None = None
    for ensemble in spec.ensembles:
        ds = load_cs_kernel_dataset(
            input_root=ensemble.input_root,
            title_pattern=ensemble.title_pattern,
            gm=spec.gm,
            eta=spec.eta,
            component=spec.component,
            nstates=spec.nstates,
            normalization_mode=spec.normalization_mode,
            bTlist=ensemble.bTlist,
            pzlist=ensemble.pzlist,
        )
        this_sample_count = next(iter(ds.values())).samples.shape[0]
        if sample_count is None:
            sample_count = this_sample_count
        elif this_sample_count != sample_count:
            raise ValueError(
                "joint CS-kernel fit requires matching bootstrap sample counts across ensembles: "
                f"expected {sample_count}, found {this_sample_count} for {ensemble.label}"
            )
        this_x = next(iter(ds.values())).x
        if reference_x is None:
            reference_x = this_x
        elif not np.allclose(this_x, reference_x, atol=1e-12, rtol=0.0):
            raise ValueError(
                "inconsistent x-grid across ensembles in joint CS-kernel fit: "
                f"ensemble {ensemble.label} differs from reference"
            )
        datasets.append((ensemble, ds))
    if sample_count is None or reference_x is None or not datasets:
        raise ValueError("no joint CS-kernel datasets were loaded")
    return datasets, reference_x, sample_count


def _resolve_x_knots(
    x_knots: np.ndarray | None,
    reference_x: np.ndarray,
    x_window: tuple[float, float],
) -> np.ndarray:
    if x_knots is not None:
        return np.asarray(x_knots, dtype=float)
    x_mask = (reference_x >= x_window[0]) & (reference_x <= x_window[1])
    unique = np.unique(reference_x[x_mask])
    if unique.size <= 6:
        return unique
    return np.linspace(float(unique.min()), float(unique.max()), 6)


def _find_x_indices(
    x_knots: np.ndarray,
    reference_x: np.ndarray,
) -> list[tuple[int, float]]:
    """Map each x_knot to (index, actual_x) of the nearest point in reference_x."""
    result: list[tuple[int, float]] = []
    for xk in x_knots:
        idx = int(np.argmin(np.abs(reference_x - xk)))
        result.append((idx, float(reference_x[idx])))
    return result


def _build_observations_at_x(
    ensemble_datasets: list[EnsembleDataset],
    x_index: int,
    x_window: tuple[float, float],
    sample_count: int,
) -> list[JointCSObservation]:
    """Build observations at one x-grid index for all ensembles, bT, and pz values."""
    observations: list[JointCSObservation] = []
    group_id = 0
    for ensemble, dataset in ensemble_datasets:
        d_p = momentum_unit_gev(ensemble.ns, ensemble.lattice_spacing_fm)
        for bT in ensemble.bTlist:
            bT_fm = float(bT * ensemble.lattice_spacing_fm)
            reference = dataset[(bT, ensemble.pzlist[0])]
            x_value = float(reference.x[x_index])
            # Enforce x_window on actual x value
            if x_value < x_window[0] or x_value > x_window[1]:
                continue
            for pz in ensemble.pzlist:
                samples = dataset[(bT, pz)].samples[:, x_index]
                q16, _, q84 = _legacy_quantile_triplet(samples)
                sigma = float(max(0.5 * (q84 - q16), np.finfo(float).eps))
                for sample_id in range(sample_count):
                    observations.append(
                        JointCSObservation(
                            group_id=group_id,
                            sample_id=sample_id,
                            x=x_value,
                            bT_fm=bT_fm,
                            pz_gev=float(pz * d_p),
                            value=float(samples[sample_id]),
                            sigma=sigma,
                            ensemble_label=ensemble.label,
                        )
                    )
            group_id += 1
    return observations


# ---------------------------------------------------------------------------
# 1D bT-spline fit at a single x
# ---------------------------------------------------------------------------

def _default_bT_knots(
    bT_fm_values: np.ndarray,
    explicit: np.ndarray | None,
    max_count: int = 8,
) -> np.ndarray:
    if explicit is not None:
        return np.asarray(explicit, dtype=float)
    unique = np.unique(np.asarray(bT_fm_values, dtype=float))
    if unique.size <= max_count:
        return unique
    return np.linspace(float(unique.min()), float(unique.max()), max_count)


def fit_gamma_eff_at_x(
    observations: list[JointCSObservation],
    *,
    sample_count: int,
    x_value: float,
    bT_knots_fm: np.ndarray,
    spline_kind: str,
    reference_p1_gev: float,
    scheme: str,
    kernel_label: str,
    mu: float,
    component: str,
    show_progress: bool,
    progress_every: int | None,
) -> PerXFitResult:
    obs_bT = np.asarray([obs.bT_fm for obs in observations], dtype=float)
    design = _spline_basis(obs_bT, bT_knots_fm, kind=spline_kind)
    pz_values = np.asarray([obs.pz_gev for obs in observations], dtype=float)
    log_p = np.log(pz_values / reference_p1_gev)
    corrections = np.asarray(
        [
            evaluate_type2_matching_correction(
                scheme=scheme,
                kernel_label=kernel_label,
                mu=mu,
                p1=reference_p1_gev,
                p2=obs.pz_gev,
                x=x_value,
                component=component,
            )
            for obs in observations
        ],
        dtype=float,
    )
    group_ids = np.asarray([obs.group_id for obs in observations], dtype=int)
    sigma = np.asarray([obs.sigma for obs in observations], dtype=float)
    n_groups = int(group_ids.max()) + 1
    coeff_samples = np.empty((sample_count, design.shape[1]), dtype=float)
    chi2_dof = np.empty(sample_count, dtype=float)
    previous = np.zeros(design.shape[1], dtype=float)
    progress_every = progress_every or max(1, sample_count // 20)
    if show_progress:
        print(
            f"    bootstrap fit: {sample_count} samples, {len(observations)} observations, "
            f"{n_groups} nuisance groups, {design.shape[1]} bT-spline coefficients",
            flush=True,
        )
    t_start = time.monotonic()
    for sample_id in range(sample_count):
        mask = np.asarray(
            [obs.sample_id == sample_id for obs in observations],
            dtype=bool,
        )
        values = np.asarray([obs.value for obs in observations], dtype=float)[mask]
        local_design = design[mask]
        local_log_p = log_p[mask]
        local_corrections = corrections[mask]
        local_groups = group_ids[mask]
        local_sigma = sigma[mask]

        def residuals(coeffs: np.ndarray) -> np.ndarray:
            gamma = local_design @ coeffs
            evolution = np.exp(local_log_p * (gamma - local_corrections))
            amplitudes = np.zeros(n_groups, dtype=float)
            for g in np.unique(local_groups):
                g_mask = local_groups == g
                weights = 1.0 / local_sigma[g_mask] ** 2
                evo = evolution[g_mask]
                amplitudes[g] = float(
                    np.sum(weights * evo * values[g_mask]) / np.sum(weights * evo ** 2)
                )
            model = amplitudes[local_groups] * evolution
            return (values - model) / local_sigma

        result = least_squares(residuals, previous, method="trf")
        coeff_samples[sample_id] = result.x
        previous = result.x
        dof = values.size - n_groups - design.shape[1]
        chi2_dof[sample_id] = float(np.sum(result.fun ** 2) / dof) if dof > 0 else float("nan")
        done = sample_id + 1
        if show_progress and (done == 1 or done == sample_count or done % progress_every == 0):
            elapsed = time.monotonic() - t_start
            rate = elapsed / done
            remaining = rate * (sample_count - done)
            print(
                f"    bootstrap: {done}/{sample_count} complete, "
                f"elapsed {elapsed:.1f}s, eta {remaining:.1f}s, "
                f"chi2/dof {chi2_dof[sample_id]:.4g}",
                flush=True,
            )
    return PerXFitResult(
        x_actual=x_value,
        bT_knots_fm=bT_knots_fm,
        coeff_samples=coeff_samples,
        chi2_dof=chi2_dof,
        n_observations=len(observations),
        n_groups=n_groups,
    )


# ---------------------------------------------------------------------------
# Diagnostic data preparation
# ---------------------------------------------------------------------------

def _build_diagnostic_groups(
    observations: list[JointCSObservation],
    per_x_result: PerXFitResult,
    sample_count: int,
    reference_p1_gev: float,
    scheme: str,
    kernel_label: str,
    mu: float,
    component: str,
    spline_kind: str,
) -> list[DiagnosticGroupData]:
    unique_groups = sorted({obs.group_id for obs in observations})
    results: list[DiagnosticGroupData] = []

    # Precompute matching corrections per (pz_gev, x)
    unique_pz_set = sorted({obs.pz_gev for obs in observations})
    correction_cache: dict[float, float] = {}
    for pz in unique_pz_set:
        correction_cache[pz] = evaluate_type2_matching_correction(
            scheme=scheme,
            kernel_label=kernel_label,
            mu=mu,
            p1=reference_p1_gev,
            p2=pz,
            x=per_x_result.x_actual,
            component=component,
        )

    for gid in unique_groups:
        group_obs = [obs for obs in observations if obs.group_id == gid]
        pz_set = sorted({obs.pz_gev for obs in group_obs})

        pz_arr = np.asarray(pz_set, dtype=float)
        bT_fm = group_obs[0].bT_fm
        ensemble_label = group_obs[0].ensemble_label

        # -- data quantiles per pz --
        data_median = np.empty(len(pz_set), dtype=float)
        data_p16 = np.empty(len(pz_set), dtype=float)
        data_p84 = np.empty(len(pz_set), dtype=float)
        for idx, pz_val in enumerate(pz_set):
            vals = np.asarray(
                [obs.value for obs in group_obs if obs.pz_gev == pz_val],
                dtype=float,
            )
            data_median[idx], data_p16[idx], data_p84[idx] = np.percentile(
                vals, [50.0, 16.0, 84.0]
            )

        # -- model reconstruction per sample --
        model_samples = np.empty((sample_count, len(pz_set)), dtype=float)
        gamma_per_sample = _evaluate_bT_surface(
            per_x_result.coeff_samples,
            np.asarray([bT_fm]),
            per_x_result.bT_knots_fm,
            spline_kind,
        )[:, 0]  # shape (sample_count,)
        log_p = np.log(pz_arr / reference_p1_gev)
        corrections = np.asarray([correction_cache[pz] for pz in pz_set], dtype=float)
        sigma_by_pz = np.asarray(
            [
                next(obs.sigma for obs in group_obs if obs.pz_gev == pz_val)
                for pz_val in pz_set
            ],
            dtype=float,
        )

        for sample_id in range(sample_count):
            evolution = np.exp(log_p * (gamma_per_sample[sample_id] - corrections))
            weights = 1.0 / sigma_by_pz ** 2
            o_sample = np.asarray(
                [
                    next(
                        obs.value
                        for obs in group_obs
                        if obs.pz_gev == pz_val and obs.sample_id == sample_id
                    )
                    for pz_val in pz_set
                ],
                dtype=float,
            )
            amplitude = float(
                np.sum(weights * evolution * o_sample)
                / np.sum(weights * evolution ** 2)
            )
            model_samples[sample_id] = amplitude * evolution

        model_median = np.percentile(model_samples, 50.0, axis=0)
        model_p16 = np.percentile(model_samples, 16.0, axis=0)
        model_p84 = np.percentile(model_samples, 84.0, axis=0)

        results.append(
            DiagnosticGroupData(
                ensemble_label=ensemble_label,
                bT_fm=bT_fm,
                pz_gev=pz_arr,
                data_median=data_median,
                data_p16=data_p16,
                data_p84=data_p84,
                model_median=model_median,
                model_p16=model_p16,
                model_p84=model_p84,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _evaluate_bT_surface(
    coeffs: np.ndarray,
    bT_values_fm: np.ndarray,
    bT_knots_fm: np.ndarray,
    spline_kind: str,
) -> np.ndarray:
    """Evaluate gamma_eff(bT) from spline coefficients.

    coeffs may be 1D (single sample) or 2D (n_samples, n_knots).
    Returns shape (n_bT,) for 1D input or (n_samples, n_bT) for 2D input.
    """
    basis = _spline_basis(np.asarray(bT_values_fm, dtype=float), bT_knots_fm, kind=spline_kind)
    if coeffs.ndim == 1:
        return basis @ coeffs
    return coeffs @ basis.T


def _write_joint_outputs(
    spec: TMDWFCSKernelJointInput,
    x_fit_order: list[float],
    per_x_results: list[PerXFitResult],
) -> list[Path]:
    output_root = spec.results_dir / "joint_gamma_eff"
    tables_dir = output_root / "tables"
    samples_dir = output_root / "samples"
    diagnostics_dir = output_root / "diagnostics"
    plots_dir = output_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if spec.make_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"joint_{spec.gm}_{spec.eta}_{spec.normalization_mode}_{spec.component}_"
        f"{spec.nstates}state_{spec.scheme}_{spec.kernel_label}_gamma_eff"
    )
    bT_knots = per_x_results[0].bT_knots_fm

    # -- surface table (gamma_eff at knot points, quantiles across samples) --
    surface_path = tables_dir / f"{stem}_surface.txt"
    with surface_path.open("w", encoding="utf-8") as handle:
        handle.write("x\tbT_fm\tgamma_p16\tgamma_p50\tgamma_p84\n")
        for result in per_x_results:
            for j, bT_val in enumerate(result.bT_knots_fm):
                values = result.coeff_samples[:, j]
                q16, q50, q84 = _legacy_quantile_triplet(values)
                handle.write(
                    f"{result.x_actual:.10e}\t{bT_val:.10e}\t"
                    f"{q16:.10e}\t{q50:.10e}\t{q84:.10e}\n"
                )

    # -- samples (gamma_eff at knot points for every bootstrap sample) --
    samples_path = samples_dir / f"{stem}_samples.txt"
    with samples_path.open("w", encoding="utf-8") as handle:
        handle.write("x\tbT_fm\tsample_id\tgamma_eff\n")
        for result in per_x_results:
            for sample_id, coeffs in enumerate(result.coeff_samples):
                for j, bT_val in enumerate(result.bT_knots_fm):
                    handle.write(
                        f"{result.x_actual:.10e}\t{bT_val:.10e}\t"
                        f"{sample_id}\t{coeffs[j]:.10e}\n"
                    )

    # -- coefficients (spline coefficients for every bootstrap sample) --
    coeff_path = samples_dir / f"{stem}_coefficients.txt"
    with coeff_path.open("w", encoding="utf-8") as handle:
        header_cols = [f"c{j}" for j in range(bT_knots.size)]
        handle.write("x\tsample_id\t" + "\t".join(header_cols) + "\n")
        for result in per_x_results:
            for sample_id, coeffs in enumerate(result.coeff_samples):
                coeff_str = "\t".join(f"{c:.10e}" for c in coeffs)
                handle.write(f"{result.x_actual:.10e}\t{sample_id}\t{coeff_str}\n")

    # -- diagnostics (chi2/dof per sample per x) --
    diagnostics_path = diagnostics_dir / f"{stem}_diagnostics.txt"
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        handle.write("x\tsample_id\tchi2_dof\n")
        for result in per_x_results:
            for sample_id, chi2_val in enumerate(result.chi2_dof):
                handle.write(f"{result.x_actual:.10e}\t{sample_id}\t{chi2_val:.10e}\n")

    # -- summary --
    summary_path = output_root / f"{stem}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"gm {spec.gm}\n")
        handle.write(f"eta {spec.eta}\n")
        handle.write(f"component {spec.component}\n")
        handle.write(f"nstates {spec.nstates}\n")
        handle.write(f"normalization_mode {spec.normalization_mode}\n")
        handle.write(f"scheme {spec.scheme}\n")
        handle.write(f"kernel_label {spec.kernel_label}\n")
        handle.write(f"mu {spec.mu:.10e}\n")
        handle.write(f"reference_p1_GeV {spec.reference_p1_gev:.10e}\n")
        handle.write(f"x_window {spec.x_window[0]:.10e} {spec.x_window[1]:.10e}\n")
        handle.write(f"spline_kind {spec.spline_kind}\n")
        handle.write(f"bT_knots_fm {' '.join(f'{v:.10e}' for v in bT_knots)}\n")
        handle.write(f"x_fit_points {' '.join(f'{v:.10e}' for v in x_fit_order)}\n")
        handle.write(f"plot {str(spec.make_plots).lower()}\n")
        handle.write(f"progress {str(spec.show_progress).lower()}\n")
        if spec.progress_every is not None:
            handle.write(f"progress_every {spec.progress_every}\n")
        handle.write(f"n_ensembles {len(spec.ensembles)}\n")
        total_obs = sum(r.n_observations for r in per_x_results)
        total_groups = sum(r.n_groups for r in per_x_results)
        handle.write(f"n_observations_total {total_obs}\n")
        handle.write(f"n_nuisance_groups_total {total_groups}\n")
        for ensemble in spec.ensembles:
            handle.write(
                f"ensemble {ensemble.label} {ensemble.input_root} "
                f"{ensemble.title_pattern} {ensemble.ns} "
                f"{ensemble.lattice_spacing_fm:.10e} "
                f"pz={','.join(str(v) for v in ensemble.pzlist)} "
                f"bT={','.join(str(v) for v in ensemble.bTlist)}\n"
            )

    outputs = [summary_path, surface_path, samples_path, coeff_path, diagnostics_path]

    # -- plots: gamma_eff vs x band for each bT knot --
    if spec.make_plots:
        x_values = np.asarray([r.x_actual for r in per_x_results], dtype=float)
        for bT_val in bT_knots:
            surface_samples = np.asarray(
                [
                    _evaluate_bT_surface(
                        result.coeff_samples[si],
                        np.asarray([bT_val]),
                        result.bT_knots_fm,
                        spec.spline_kind,
                    )[0]
                    for result in per_x_results
                    for si in range(result.coeff_samples.shape[0])
                ],
                dtype=float,
            ).reshape(len(per_x_results), per_x_results[0].coeff_samples.shape[0])
            q16 = np.percentile(surface_samples, 16.0, axis=1)
            q50 = np.percentile(surface_samples, 50.0, axis=1)
            q84 = np.percentile(surface_samples, 84.0, axis=1)
            bT_token = f"{bT_val:.3f}".replace(".", "p")
            plot_path = plots_dir / f"{stem}_bT{bT_token}_x_band.pdf"
            outputs.append(
                plot_tmdwf_joint_cs_kernel_x_band(
                    plot_path,
                    x_values=x_values,
                    band_p16=q16,
                    band_p50=q50,
                    band_p84=q84,
                    bT_fm=float(bT_val),
                    title=f"{spec.gm} {spec.eta} {spec.normalization_mode}",
                    kernel_label=spec.kernel_label,
                    spline_kind=spec.spline_kind,
                )
            )
    return outputs


# ---------------------------------------------------------------------------
# Top-level workflow
# ---------------------------------------------------------------------------

def run_tmdwf_cs_kernel_joint_workflow(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_cs_kernel_joint_input(input_file, results_dir=results_dir)
    ensemble_datasets, reference_x, sample_count = _preload_datasets(spec)
    x_knots = _resolve_x_knots(spec.x_knots, reference_x, spec.x_window)
    x_indices = _find_x_indices(x_knots, reference_x)

    bT_fm_values: list[float] = []
    for ensemble, _ in ensemble_datasets:
        for bT in ensemble.bTlist:
            bT_fm_values.append(float(bT * ensemble.lattice_spacing_fm))
    bT_knots_fm = _default_bT_knots(np.array(bT_fm_values), spec.bT_knots_fm, max_count=8)

    if spec.show_progress:
        print(
            f"joint CS fit: {len(x_knots)} x-points, "
            f"{bT_knots_fm.size} bT-knots, "
            f"{len(spec.ensembles)} ensembles, "
            f"{sample_count} bootstrap samples",
            flush=True,
        )

    per_x_results: list[PerXFitResult] = []
    x_fit_order: list[float] = []
    diagnostic_plot_paths: list[Path] = []
    for xi, (x_idx, x_actual) in enumerate(x_indices):
        if spec.show_progress:
            print(
                f"x [{xi + 1}/{len(x_indices)}] x_actual={x_actual:.6f}",
                flush=True,
            )
        observations = _build_observations_at_x(
            ensemble_datasets,
            x_index=x_idx,
            x_window=spec.x_window,
            sample_count=sample_count,
        )
        if not observations:
            if spec.show_progress:
                print(f"    no observations in x_window, skipping", flush=True)
            continue
        result = fit_gamma_eff_at_x(
            observations,
            sample_count=sample_count,
            x_value=x_actual,
            bT_knots_fm=bT_knots_fm,
            spline_kind=spec.spline_kind,
            reference_p1_gev=spec.reference_p1_gev,
            scheme=spec.scheme,
            kernel_label=spec.kernel_label,
            mu=spec.mu,
            component=spec.component,
            show_progress=spec.show_progress,
            progress_every=spec.progress_every,
        )
        per_x_results.append(result)
        x_fit_order.append(x_actual)

        # -- diagnostic pz-fit plots per (ensemble, bT) group --
        if spec.make_plots:
            stem = (
                f"joint_{spec.gm}_{spec.eta}_{spec.normalization_mode}_{spec.component}_"
                f"{spec.nstates}state_{spec.scheme}_{spec.kernel_label}_gamma_eff"
            )
            diagnostics_dir = spec.results_dir / "joint_gamma_eff" / "plots" / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            diagnostic_groups = _build_diagnostic_groups(
                observations,
                result,
                sample_count=sample_count,
                reference_p1_gev=spec.reference_p1_gev,
                scheme=spec.scheme,
                kernel_label=spec.kernel_label,
                mu=spec.mu,
                component=spec.component,
                spline_kind=spec.spline_kind,
            )
            diagnostic_plot_paths.extend(
                plot_tmdwf_joint_cs_kernel_pz_diagnostics(
                    diagnostics_dir,
                    diagnostic_groups,
                    stem=stem,
                    x_actual=x_actual,
                )
            )

    if not per_x_results:
        raise ValueError("no x-points produced valid observations; check x_knots vs x_window")

    outputs = _write_joint_outputs(spec, x_fit_order, per_x_results)
    outputs.extend(diagnostic_plot_paths)

    # -- diagnostics notebook (reproducible plots from saved outputs) --
    if spec.make_plots:
        summary_path = outputs[0]
        stem = (
            f"joint_{spec.gm}_{spec.eta}_{spec.normalization_mode}_{spec.component}_"
            f"{spec.nstates}state_{spec.scheme}_{spec.kernel_label}_gamma_eff"
        )
        coeff_path = spec.results_dir / "joint_gamma_eff" / "samples" / f"{stem}_coefficients.txt"
        notebook_dir = spec.results_dir / "joint_gamma_eff" / "plots" / "diagnostics"
        notebook_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = notebook_dir / f"{stem}_diagnostics_notebook.ipynb"
        try:
            outputs.append(
                write_tmdwf_cs_kernel_joint_diagnostics_notebook(
                    notebook_path,
                    summary_path=summary_path,
                    coefficients_path=coeff_path,
                    results_dir=spec.results_dir,
                )
            )
        except Exception as exc:
            if spec.show_progress:
                print(f"    (diagnostics notebook skipped: {exc})", flush=True)

    return outputs


# ---------------------------------------------------------------------------
# Post-hoc diagnostics helpers
# ---------------------------------------------------------------------------

def parse_joint_summary(summary_path: str | Path) -> dict:
    """Parse a joint CS-kernel summary file back into a configuration dict."""
    summary_path = Path(summary_path)
    entries: dict[str, list[str]] = {}
    ensembles: list[dict] = []
    with summary_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if tokens[0] == "ensemble":
                pz_str = tokens[6].split("=", 1)[1]
                bT_str = tokens[7].split("=", 1)[1]
                ensembles.append({
                    "label": tokens[1],
                    "input_root": tokens[2],
                    "title_pattern": tokens[3],
                    "ns": int(tokens[4]),
                    "lattice_spacing_fm": float(tokens[5]),
                    "pzlist": _parse_int_items(pz_str),
                    "bTlist": _parse_int_items(bT_str),
                })
            else:
                entries[tokens[0]] = tokens[1:]

    x_window = tuple(float(v) for v in entries["x_window"][:2])
    return {
        "gm": entries["gm"][0],
        "eta": entries["eta"][0],
        "component": entries["component"][0],
        "nstates": int(entries["nstates"][0]),
        "normalization_mode": entries["normalization_mode"][0],
        "mu": float(entries["mu"][0]),
        "scheme": entries.get("scheme", ["CG"])[0],
        "kernel_label": entries["kernel_label"][0],
        "reference_p1_gev": float(entries["reference_p1_GeV"][0]),
        "x_window": x_window,
        "spline_kind": entries["spline_kind"][0],
        "bT_knots_fm": np.asarray([float(v) for v in entries["bT_knots_fm"]], dtype=float),
        "x_fit_points": np.asarray([float(v) for v in entries["x_fit_points"]], dtype=float),
        "ensembles": ensembles,
    }


def load_joint_coefficients_table(path: str | Path) -> dict[float, np.ndarray]:
    """Load coefficients file, returning {x_actual: coeffs_array (n_samples, n_coeffs)}."""
    path = Path(path)
    by_x: dict[float, list[np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        # header: x, sample_id, c0, c1, ...
        n_coeffs = len(header) - 2
        for line in handle:
            tokens = line.strip().split("\t")
            x_val = float(tokens[0])
            coeffs = np.asarray([float(t) for t in tokens[2:]], dtype=float)
            by_x.setdefault(x_val, []).append(coeffs)
    return {x: np.asarray(rows, dtype=float) for x, rows in by_x.items()}
