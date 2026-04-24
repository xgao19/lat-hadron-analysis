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
from .plotting import plot_tmdwf_joint_cs_kernel_x_band


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


@dataclass(frozen=True)
class JointCSFitResult:
    x_knots: np.ndarray
    bT_knots_fm: np.ndarray
    coeff_samples: np.ndarray
    chi2_dof: np.ndarray
    n_observations: int
    n_groups: int


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


def evaluate_joint_cs_surface(
    coeffs: np.ndarray,
    *,
    x_values: np.ndarray,
    bT_values_fm: np.ndarray,
    x_knots: np.ndarray,
    bT_knots_fm: np.ndarray,
    spline_kind: str = "linear",
) -> np.ndarray:
    x_basis = _spline_basis(np.asarray(x_values, dtype=float), x_knots, kind=spline_kind)
    bT_basis = _spline_basis(
        np.asarray(bT_values_fm, dtype=float),
        bT_knots_fm,
        kind=spline_kind,
    )
    coeff_matrix = np.asarray(coeffs, dtype=float).reshape(x_knots.size, bT_knots_fm.size)
    return np.einsum("xi,ij,bj->xb", x_basis, coeff_matrix, bT_basis)


def _build_observations_for_ensemble(
    ensemble: JointCSEnsembleInput,
    dataset: dict[tuple[int, int], CSKernelObservable],
    *,
    sample_count: int,
    x_window: tuple[float, float],
) -> list[JointCSObservation]:
    observations: list[JointCSObservation] = []
    d_p = momentum_unit_gev(ensemble.ns, ensemble.lattice_spacing_fm)
    group_index: dict[tuple[str, int, int], int] = {}
    for bT in ensemble.bTlist:
        reference = dataset[(bT, ensemble.pzlist[0])]
        x_mask = (reference.x >= x_window[0]) & (reference.x <= x_window[1])
        for local_index, x_value in zip(np.where(x_mask)[0], reference.x[x_mask], strict=True):
            group_key = (ensemble.label, bT, int(local_index))
            group_id = group_index.setdefault(group_key, len(group_index))
            for pz in ensemble.pzlist:
                samples = dataset[(bT, pz)].samples[:, local_index]
                q16, _, q84 = _legacy_quantile_triplet(samples)
                sigma = float(max(0.5 * (q84 - q16), np.finfo(float).eps))
                for sample_id in range(sample_count):
                    observations.append(
                        JointCSObservation(
                            group_id=group_id,
                            sample_id=sample_id,
                            x=float(x_value),
                            bT_fm=float(bT * ensemble.lattice_spacing_fm),
                            pz_gev=float(pz * d_p),
                            value=float(samples[sample_id]),
                            sigma=sigma,
                        )
                    )
    return observations


def load_joint_cs_observations(
    spec: TMDWFCSKernelJointInput,
) -> tuple[list[JointCSObservation], int]:
    observations: list[JointCSObservation] = []
    sample_count: int | None = None
    group_offset = 0
    for ensemble in spec.ensembles:
        dataset = load_cs_kernel_dataset(
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
        ensemble_sample_count = next(iter(dataset.values())).samples.shape[0]
        if sample_count is None:
            sample_count = ensemble_sample_count
        elif ensemble_sample_count != sample_count:
            raise ValueError(
                "joint CS-kernel fit requires matching bootstrap sample counts across ensembles: "
                f"expected {sample_count}, found {ensemble_sample_count} for {ensemble.label}"
            )
        ensemble_observations = _build_observations_for_ensemble(
            ensemble,
            dataset,
            sample_count=ensemble_sample_count,
            x_window=spec.x_window,
        )
        for observation in ensemble_observations:
            observations.append(
                JointCSObservation(
                    group_id=observation.group_id + group_offset,
                    sample_id=observation.sample_id,
                    x=observation.x,
                    bT_fm=observation.bT_fm,
                    pz_gev=observation.pz_gev,
                    value=observation.value,
                    sigma=observation.sigma,
                )
            )
        if ensemble_observations:
            group_offset = max(observation.group_id for observation in observations) + 1
    if sample_count is None or not observations:
        raise ValueError("no joint CS-kernel observations were loaded")
    return observations, sample_count


def _default_knots(
    values: np.ndarray,
    explicit: np.ndarray | None,
    *,
    max_count: int,
) -> np.ndarray:
    if explicit is not None:
        return np.asarray(explicit, dtype=float)
    unique = np.unique(np.asarray(values, dtype=float))
    if unique.size <= max_count:
        return unique
    return np.linspace(float(unique.min()), float(unique.max()), max_count)


def fit_joint_cs_gamma_eff(
    observations: list[JointCSObservation],
    *,
    sample_count: int,
    spec: TMDWFCSKernelJointInput,
) -> JointCSFitResult:
    obs_x = np.asarray([observation.x for observation in observations], dtype=float)
    obs_bT = np.asarray([observation.bT_fm for observation in observations], dtype=float)
    x_knots = _default_knots(obs_x, spec.x_knots, max_count=6)
    bT_knots = _default_knots(obs_bT, spec.bT_knots_fm, max_count=8)
    design = np.einsum(
        "oi,oj->oij",
        _spline_basis(obs_x, x_knots, kind=spec.spline_kind),
        _spline_basis(obs_bT, bT_knots, kind=spec.spline_kind),
    ).reshape(len(observations), x_knots.size * bT_knots.size)
    pz_values = np.asarray([observation.pz_gev for observation in observations], dtype=float)
    log_p = np.log(pz_values / spec.reference_p1_gev)
    corrections = np.asarray(
        [
            evaluate_type2_matching_correction(
                scheme=spec.scheme,
                kernel_label=spec.kernel_label,
                mu=spec.mu,
                p1=spec.reference_p1_gev,
                p2=observation.pz_gev,
                x=observation.x,
                component=spec.component,
            )
            for observation in observations
        ],
        dtype=float,
    )
    group_ids = np.asarray([observation.group_id for observation in observations], dtype=int)
    sigma = np.asarray([observation.sigma for observation in observations], dtype=float)
    n_groups = int(group_ids.max()) + 1
    coeff_samples = np.empty((sample_count, design.shape[1]), dtype=float)
    chi2_dof = np.empty(sample_count, dtype=float)
    previous = np.zeros(design.shape[1], dtype=float)
    progress_every = spec.progress_every or max(1, sample_count // 20)
    start_time = time.monotonic()
    if spec.show_progress:
        print(
            "joint CS bootstrap fit: "
            f"{sample_count} samples, {len(observations)} observations, "
            f"{n_groups} nuisance groups, {design.shape[1]} spline coefficients",
            flush=True,
        )
    for sample_id in range(sample_count):
        mask = np.asarray(
            [observation.sample_id == sample_id for observation in observations],
            dtype=bool,
        )
        values = np.asarray([observation.value for observation in observations], dtype=float)[mask]
        local_design = design[mask]
        local_log_p = log_p[mask]
        local_corrections = corrections[mask]
        local_groups = group_ids[mask]
        local_sigma = sigma[mask]

        def residuals(coeffs: np.ndarray) -> np.ndarray:
            gamma = local_design @ coeffs
            evolution = np.exp(local_log_p * (gamma - local_corrections))
            amplitudes = np.zeros(n_groups, dtype=float)
            for group_id in np.unique(local_groups):
                group_mask = local_groups == group_id
                weights = 1.0 / local_sigma[group_mask] ** 2
                evo = evolution[group_mask]
                amplitudes[group_id] = float(
                    np.sum(weights * evo * values[group_mask]) / np.sum(weights * evo**2)
                )
            model = amplitudes[local_groups] * evolution
            return (values - model) / local_sigma

        result = least_squares(residuals, previous, method="trf")
        coeff_samples[sample_id] = result.x
        previous = result.x
        dof = values.size - n_groups - design.shape[1]
        chi2_dof[sample_id] = float(np.sum(result.fun**2) / dof) if dof > 0 else float("nan")
        done = sample_id + 1
        if spec.show_progress and (done == 1 or done == sample_count or done % progress_every == 0):
            elapsed = time.monotonic() - start_time
            rate = elapsed / done
            remaining = rate * (sample_count - done)
            print(
                "joint CS bootstrap fit: "
                f"{done}/{sample_count} complete, "
                f"elapsed {elapsed:.1f}s, eta {remaining:.1f}s, "
                f"chi2/dof {chi2_dof[sample_id]:.4g}",
                flush=True,
            )
    return JointCSFitResult(
        x_knots=x_knots,
        bT_knots_fm=bT_knots,
        coeff_samples=coeff_samples,
        chi2_dof=chi2_dof,
        n_observations=len(observations),
        n_groups=n_groups,
    )


def _write_joint_outputs(spec: TMDWFCSKernelJointInput, fit: JointCSFitResult) -> list[Path]:
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
    summary_path = output_root / f"{stem}_summary.txt"
    surface_path = tables_dir / f"{stem}_surface.txt"
    samples_path = samples_dir / f"{stem}_samples.txt"
    diagnostics_path = diagnostics_dir / f"{stem}_diagnostics.txt"

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
        handle.write(f"x_knots {' '.join(f'{value:.10e}' for value in fit.x_knots)}\n")
        handle.write(
            f"bT_knots_fm {' '.join(f'{value:.10e}' for value in fit.bT_knots_fm)}\n"
        )
        handle.write(f"spline_kind {spec.spline_kind}\n")
        handle.write(f"plot {str(spec.make_plots).lower()}\n")
        handle.write(f"progress {str(spec.show_progress).lower()}\n")
        if spec.progress_every is not None:
            handle.write(f"progress_every {spec.progress_every}\n")
        handle.write(f"n_ensembles {len(spec.ensembles)}\n")
        handle.write(f"n_observations {fit.n_observations}\n")
        handle.write(f"n_nuisance_groups {fit.n_groups}\n")
        for ensemble in spec.ensembles:
            handle.write(
                f"ensemble {ensemble.label} {ensemble.input_root} "
                f"{ensemble.title_pattern} {ensemble.ns} "
                f"{ensemble.lattice_spacing_fm:.10e} "
                f"pz={','.join(str(value) for value in ensemble.pzlist)} "
                f"bT={','.join(str(value) for value in ensemble.bTlist)}\n"
            )

    with surface_path.open("w", encoding="utf-8") as handle:
        handle.write("x\tbT_fm\tgamma_p16\tgamma_p50\tgamma_p84\n")
        for x_value in fit.x_knots:
            for bT_value in fit.bT_knots_fm:
                values = np.asarray(
                    [
                        evaluate_joint_cs_surface(
                            coeffs,
                            x_values=np.asarray([x_value], dtype=float),
                            bT_values_fm=np.asarray([bT_value], dtype=float),
                            x_knots=fit.x_knots,
                            bT_knots_fm=fit.bT_knots_fm,
                            spline_kind=spec.spline_kind,
                        )[0, 0]
                        for coeffs in fit.coeff_samples
                    ],
                    dtype=float,
                )
                q16, q50, q84 = _legacy_quantile_triplet(values)
                handle.write(
                    f"{x_value:.10e}\t{bT_value:.10e}\t"
                    f"{q16:.10e}\t{q50:.10e}\t{q84:.10e}\n"
                )

    with samples_path.open("w", encoding="utf-8") as handle:
        handle.write("x\tbT_fm\tsample_id\tgamma_eff\n")
        for sample_id, coeffs in enumerate(fit.coeff_samples):
            surface = evaluate_joint_cs_surface(
                coeffs,
                x_values=fit.x_knots,
                bT_values_fm=fit.bT_knots_fm,
                x_knots=fit.x_knots,
                bT_knots_fm=fit.bT_knots_fm,
                spline_kind=spec.spline_kind,
            )
            for x_index, x_value in enumerate(fit.x_knots):
                for bT_index, bT_value in enumerate(fit.bT_knots_fm):
                    handle.write(
                        f"{x_value:.10e}\t{bT_value:.10e}\t{sample_id}\t"
                        f"{surface[x_index, bT_index]:.10e}\n"
                    )

    with diagnostics_path.open("w", encoding="utf-8") as handle:
        handle.write("sample_id\tchi2_dof\n")
        for sample_id, chi2_value in enumerate(fit.chi2_dof):
            handle.write(f"{sample_id}\t{chi2_value:.10e}\n")

    outputs = [summary_path, surface_path, samples_path, diagnostics_path]
    if spec.make_plots:
        for bT_value in fit.bT_knots_fm:
            values_by_x = []
            for coeffs in fit.coeff_samples:
                values_by_x.append(
                    evaluate_joint_cs_surface(
                        coeffs,
                        x_values=fit.x_knots,
                        bT_values_fm=np.asarray([bT_value], dtype=float),
                        x_knots=fit.x_knots,
                        bT_knots_fm=fit.bT_knots_fm,
                        spline_kind=spec.spline_kind,
                    )[:, 0]
                )
            surface_samples = np.asarray(values_by_x, dtype=float)
            q16 = np.percentile(surface_samples, 16.0, axis=0)
            q50 = np.percentile(surface_samples, 50.0, axis=0)
            q84 = np.percentile(surface_samples, 84.0, axis=0)
            bT_token = f"{bT_value:.3f}".replace(".", "p")
            plot_path = plots_dir / f"{stem}_bT{bT_token}_x_band.pdf"
            outputs.append(
                plot_tmdwf_joint_cs_kernel_x_band(
                    plot_path,
                    x_values=fit.x_knots,
                    band_p16=q16,
                    band_p50=q50,
                    band_p84=q84,
                    bT_fm=float(bT_value),
                    title=f"{spec.gm} {spec.eta} {spec.normalization_mode}",
                    kernel_label=spec.kernel_label,
                    spline_kind=spec.spline_kind,
                )
            )
    return outputs


def run_tmdwf_cs_kernel_joint_workflow(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_cs_kernel_joint_input(input_file, results_dir=results_dir)
    observations, sample_count = load_joint_cs_observations(spec)
    fit = fit_joint_cs_gamma_eff(observations, sample_count=sample_count, spec=spec)
    return _write_joint_outputs(spec, fit)
