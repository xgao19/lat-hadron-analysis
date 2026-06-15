from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from ..common.constants import HBAR_C_GEV_FM, HBAR_C_MEV_FM
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

CORRECTION_A0_FM = 0.1
CORRECTION_B0_FM = 1.0
CORRECTION_P0_GEV = 1.0


@dataclass(frozen=True)
class JointCSEnsembleInput:
    label: str
    input_root: Path
    title_pattern: str
    ns: int
    lattice_spacing_fm: float
    pzlist: tuple[int, ...]
    bTlist: tuple[int, ...]
    m_pi_mev: float = 140.0


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
    fit_a2_correction: bool = False
    fit_fv_correction: bool = False
    fit_pz2_correction: bool = False
    fit_apz2_correction: bool = False
    a2_correction_prior_width: float = 1.0
    fv_correction_prior_width: float = 1.0
    pz2_correction_prior_width: float = 1.0
    apz2_correction_prior_width: float = 1.0


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
    a_fm: float
    fv_prefactor: float
    fv_exp_m_pi_bT: float
    spatial_extent_fm: float


@dataclass(frozen=True)
class PerXFitResult:
    x_actual: float
    bT_knots_fm: np.ndarray
    coeff_samples: np.ndarray  # (n_samples, n_total_coeffs): [gamma, alpha?, beta?, kappa?, lambda?]
    chi2_dof: np.ndarray
    n_observations: int
    n_groups: int
    n_gamma_knots: int
    n_correction_params: int
    fit_a2_correction: bool = False
    fit_fv_correction: bool = False
    fit_pz2_correction: bool = False
    fit_apz2_correction: bool = False

    @property
    def gamma_samples(self) -> np.ndarray:
        return self.coeff_samples[:, :self.n_gamma_knots]

    def _block_size(self, block_index: int) -> int:
        return 1 if block_index == 3 else 2

    def _block_slice(self, block_index: int) -> slice:
        """Return slice for correction block 0=alpha, 1=beta, 2=kappa, 3=lambda (skipping disabled blocks)."""
        offset = self.n_gamma_knots
        blocks = [
            self.fit_a2_correction,
            self.fit_fv_correction,
            self.fit_pz2_correction,
            self.fit_apz2_correction,
        ]
        for i in range(block_index):
            if blocks[i]:
                offset += self._block_size(i)
        return slice(offset, offset + self._block_size(block_index))

    @property
    def alpha_samples(self) -> np.ndarray | None:
        if not self.fit_a2_correction:
            return None
        return self.coeff_samples[:, self._block_slice(0)]

    @property
    def beta_samples(self) -> np.ndarray | None:
        if not self.fit_fv_correction:
            return None
        return self.coeff_samples[:, self._block_slice(1)]

    @property
    def kappa_samples(self) -> np.ndarray | None:
        if not self.fit_pz2_correction:
            return None
        return self.coeff_samples[:, self._block_slice(2)]

    @property
    def lambda_samples(self) -> np.ndarray | None:
        if not self.fit_apz2_correction:
            return None
        return self.coeff_samples[:, self._block_slice(3)]


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


@dataclass(frozen=True)
class XReflectionPlan:
    fit_entries: list[tuple[int, float]]
    output_entries: list[tuple[int, float]]
    output_fit_keys: list[tuple[int, float]]
    uses_reflection: bool


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
        m_pi_mev=float(options["m_pi"]) if "m_pi" in options else 140.0,
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
    def _parse_positive_float(entries, key, default):
        value = float(entries.get(key, [default])[0])
        if value <= 0.0:
            raise ValueError(f"{key} must be positive")
        return value

    default_results_dir = file_path.parent / "results_tmdwf_cs_kernel_joint"
    output_root = Path(results_dir) if results_dir is not None else Path(
        entries.get("results_dir", [default_results_dir])[0]
    )
    def _parse_bool(entries, key, default=False):
        if key not in entries:
            return default
        return entries[key][0].lower() not in {"false", "0", "no"}

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
        fit_a2_correction=_parse_bool(entries, "fit_a2_correction"),
        fit_fv_correction=_parse_bool(entries, "fit_fv_correction"),
        fit_pz2_correction=_parse_bool(entries, "fit_pz2_correction"),
        fit_apz2_correction=_parse_bool(entries, "fit_apz2_correction"),
        a2_correction_prior_width=_parse_positive_float(entries, "a2_correction_prior_width", "1.0"),
        fv_correction_prior_width=_parse_positive_float(entries, "fv_correction_prior_width", "1.0"),
        pz2_correction_prior_width=_parse_positive_float(entries, "pz2_correction_prior_width", "1.0"),
        apz2_correction_prior_width=_parse_positive_float(entries, "apz2_correction_prior_width", "1.0"),
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


def _plan_x_reflection(
    x_indices: list[tuple[int, float]],
    *,
    atol: float = 1e-10,
) -> XReflectionPlan:
    """Fit one representative from each x <-> 1-x pair and mirror outputs."""
    if not x_indices:
        return XReflectionPlan(
            fit_entries=[],
            output_entries=[],
            output_fit_keys=[],
            uses_reflection=False,
        )

    fit_entries: list[tuple[int, float]] = []
    output_fit_keys: list[tuple[int, float]] = []
    fit_keys: set[tuple[int, float]] = set()
    uses_reflection = False

    for _, x_actual in x_indices:
        target_x = min(x_actual, 1.0 - x_actual)
        candidate = min(x_indices, key=lambda item: abs(item[1] - target_x))
        if abs(candidate[1] - target_x) > atol:
            candidate = min(x_indices, key=lambda item: abs(item[1] - x_actual))
        uses_reflection = uses_reflection or abs(candidate[1] - x_actual) > atol

        key = (candidate[0], round(candidate[1], 12))
        if key not in fit_keys:
            fit_entries.append(candidate)
            fit_keys.add(key)
        output_fit_keys.append(key)

    return XReflectionPlan(
        fit_entries=fit_entries,
        output_entries=x_indices,
        output_fit_keys=output_fit_keys,
        uses_reflection=uses_reflection,
    )


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
        a_fm = ensemble.lattice_spacing_fm
        spatial_extent_fm = ensemble.ns * a_fm
        m_pi_l = ensemble.m_pi_mev * ensemble.ns * a_fm / HBAR_C_MEV_FM
        fv_prefactor = float(np.exp(-m_pi_l) / np.sqrt(m_pi_l))
        for bT in ensemble.bTlist:
            bT_fm = float(bT * ensemble.lattice_spacing_fm)
            fv_exp_m_pi_bT = float(np.exp(ensemble.m_pi_mev * bT_fm / HBAR_C_MEV_FM))
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
                            a_fm=a_fm,
                            fv_prefactor=fv_prefactor,
                            fv_exp_m_pi_bT=fv_exp_m_pi_bT,
                            spatial_extent_fm=spatial_extent_fm,
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


def _requires_inverse_bT(
    *,
    fit_a2_correction: bool,
    fit_pz2_correction: bool,
) -> bool:
    return fit_a2_correction or fit_pz2_correction


def _check_nonzero_bT_for_inverse_corrections(
    bT_values_fm: np.ndarray,
    *,
    fit_a2_correction: bool,
    fit_pz2_correction: bool,
) -> None:
    if _requires_inverse_bT(
        fit_a2_correction=fit_a2_correction,
        fit_pz2_correction=fit_pz2_correction,
    ) and np.any(np.asarray(bT_values_fm, dtype=float) <= 0.0):
        raise ValueError(
            "analytic CS-kernel systematic corrections with 1/bT^2 require "
            "strictly positive bT values; remove bT=0 from the joint fit or "
            "disable a2 and pz2 corrections"
        )


def _check_bT_below_box_for_fv(observations: list[JointCSObservation]) -> None:
    bad = [
        obs
        for obs in observations
        if obs.bT_fm >= obs.spatial_extent_fm
    ]
    if bad:
        obs = bad[0]
        raise ValueError(
            "finite-volume correction requires bT below the spatial box size: "
            f"found bT={obs.bT_fm:.6g} fm and L={obs.spatial_extent_fm:.6g} fm "
            f"for ensemble {obs.ensemble_label}"
        )


def _evaluate_correction_shape(
    name: str,
    coeffs: np.ndarray,
    bT_values_fm: np.ndarray,
    *,
    fv_exp_m_pi_bT: np.ndarray | None = None,
) -> np.ndarray:
    coeff_array = np.asarray(coeffs, dtype=float)
    bT = np.asarray(bT_values_fm, dtype=float)
    c0 = coeff_array[..., 0]
    if name == "apz2":
        return np.expand_dims(c0, axis=-1) + np.zeros_like(bT, dtype=float)
    c1 = coeff_array[..., 1]
    if name == "fv":
        if fv_exp_m_pi_bT is None:
            raise ValueError("fv correction evaluation requires exp(M_pi bT)")
        fv_exp = np.asarray(fv_exp_m_pi_bT, dtype=float)
        return np.expand_dims(c0, axis=-1) + np.expand_dims(c1, axis=-1) * fv_exp
    if np.any(bT <= 0.0):
        raise ValueError(f"{name} correction uses 1/bT^2 and requires bT > 0")
    return np.expand_dims(c0, axis=-1) + np.expand_dims(c1, axis=-1) * (CORRECTION_B0_FM / bT) ** 2


def _evaluate_evolution_factor(
    log_p: np.ndarray,
    gamma: np.ndarray,
    delta_m: np.ndarray,
    correction_factor: np.ndarray,
) -> np.ndarray:
    return np.exp(log_p * (gamma * correction_factor - delta_m))


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
    fit_a2_correction: bool = False,
    fit_fv_correction: bool = False,
    fit_pz2_correction: bool = False,
    fit_apz2_correction: bool = False,
    a2_correction_prior_width: float = 1.0,
    fv_correction_prior_width: float = 1.0,
    pz2_correction_prior_width: float = 1.0,
    apz2_correction_prior_width: float = 1.0,
) -> PerXFitResult:
    prior_widths = {
        "a2": float(a2_correction_prior_width),
        "fv": float(fv_correction_prior_width),
        "pz2": float(pz2_correction_prior_width),
        "apz2": float(apz2_correction_prior_width),
    }
    for name, width in prior_widths.items():
        if width <= 0.0:
            raise ValueError(f"{name}_correction_prior_width must be positive")
    obs_bT = np.asarray([obs.bT_fm for obs in observations], dtype=float)
    _check_nonzero_bT_for_inverse_corrections(
        obs_bT,
        fit_a2_correction=fit_a2_correction,
        fit_pz2_correction=fit_pz2_correction,
    )
    if fit_fv_correction:
        _check_bT_below_box_for_fv(observations)
    design = _spline_basis(obs_bT, bT_knots_fm, kind=spline_kind)
    n_knots = design.shape[1]
    pz_values = np.asarray([obs.pz_gev for obs in observations], dtype=float)
    log_p = np.log(pz_values / reference_p1_gev)
    delta_m = np.asarray(
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

    # Precompute per-observation correction scales
    a2_vals = np.asarray([(obs.a_fm / CORRECTION_A0_FM) ** 2 for obs in observations], dtype=float)
    fv_prefactor_vals = np.asarray([obs.fv_prefactor for obs in observations], dtype=float)
    fv_exp_m_pi_bT_vals = np.asarray([obs.fv_exp_m_pi_bT for obs in observations], dtype=float)
    inv_pz2_vals = np.asarray(
        [
            (1.0 / obs.x ** 2 + 1.0 / (1.0 - obs.x) ** 2)
            * (CORRECTION_P0_GEV / obs.pz_gev) ** 2
            for obs in observations
        ],
        dtype=float,
    )
    apz2_vals = np.asarray(
        [(obs.a_fm * obs.pz_gev / HBAR_C_GEV_FM) ** 2 for obs in observations],
        dtype=float,
    )

    # Build active correction blocks
    correction_blocks: list[tuple[str, np.ndarray]] = []
    if fit_a2_correction:
        correction_blocks.append(("a2", a2_vals))
    if fit_fv_correction:
        correction_blocks.append(("fv", fv_prefactor_vals))
    if fit_pz2_correction:
        correction_blocks.append(("pz2", inv_pz2_vals))
    if fit_apz2_correction:
        correction_blocks.append(("apz2", apz2_vals))
    n_corr_blocks = len(correction_blocks)
    n_correction_params = (
        2 * int(fit_a2_correction)
        + 2 * int(fit_fv_correction)
        + 2 * int(fit_pz2_correction)
        + int(fit_apz2_correction)
    )
    n_total_coeffs = n_knots + n_correction_params

    coeff_samples = np.empty((sample_count, n_total_coeffs), dtype=float)
    chi2_dof = np.empty(sample_count, dtype=float)
    previous = np.zeros(n_total_coeffs, dtype=float)
    progress_every = progress_every or max(1, sample_count // 20)
    prior_slices: list[tuple[slice, float]] = []
    prior_offset = n_knots
    for name, _scale in correction_blocks:
        size = 1 if name == "apz2" else 2
        prior_slices.append((slice(prior_offset, prior_offset + size), prior_widths[name]))
        prior_offset += size

    if show_progress:
        corr_desc = ", ".join(b[0] for b in correction_blocks) if correction_blocks else "none"
        prior_desc = ", ".join(f"{name}:0+/-{prior_widths[name]:.3g}" for name, _ in correction_blocks)
        print(
            f"    bootstrap fit: {sample_count} samples, {len(observations)} observations, "
            f"{n_groups} nuisance groups, {n_knots} gamma bT-knots, "
            f"2 parameters per correction channel except apz2 has 1, "
            f"{n_total_coeffs} total coeffs "
            f"({n_knots} gamma + {n_correction_params} corrections [{corr_desc}]), "
            f"correction priors [{prior_desc or 'none'}]",
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
        local_bT = obs_bT[mask]
        local_log_p = log_p[mask]
        local_delta_m = delta_m[mask]
        local_groups = group_ids[mask]
        local_sigma = sigma[mask]
        local_a2 = a2_vals[mask]
        local_fv_prefactor = fv_prefactor_vals[mask]
        local_fv_exp_m_pi_bT = fv_exp_m_pi_bT_vals[mask]
        local_inv_pz2 = inv_pz2_vals[mask]
        local_apz2 = apz2_vals[mask]

        def residuals(coeffs: np.ndarray) -> np.ndarray:
            # gamma_MSbar(bT) from first n_knots coefficients
            gamma = local_design @ coeffs[:n_knots]
            # Multiplicative correction factor on gamma_MSbar inside the exponent.
            corr_factor = np.ones(len(gamma), dtype=float)
            block_offset = n_knots
            if fit_a2_correction:
                alpha = _evaluate_correction_shape("a2", coeffs[block_offset:block_offset + 2], local_bT)
                corr_factor += local_a2 * alpha
                block_offset += 2
            if fit_fv_correction:
                beta = _evaluate_correction_shape(
                    "fv",
                    coeffs[block_offset:block_offset + 2],
                    local_bT,
                    fv_exp_m_pi_bT=local_fv_exp_m_pi_bT,
                )
                corr_factor += local_fv_prefactor * beta
                block_offset += 2
            if fit_pz2_correction:
                kappa = _evaluate_correction_shape("pz2", coeffs[block_offset:block_offset + 2], local_bT)
                corr_factor += local_inv_pz2 * kappa
                block_offset += 2
            if fit_apz2_correction:
                lam = _evaluate_correction_shape("apz2", coeffs[block_offset:block_offset + 1], local_bT)
                corr_factor += local_apz2 * lam
                block_offset += 1
            evolution = _evaluate_evolution_factor(local_log_p, gamma, local_delta_m, corr_factor)
            amplitudes = np.zeros(n_groups, dtype=float)
            for g in np.unique(local_groups):
                g_mask = local_groups == g
                weights = 1.0 / local_sigma[g_mask] ** 2
                evo = evolution[g_mask]
                amplitudes[g] = float(
                    np.sum(weights * evo * values[g_mask]) / np.sum(weights * evo ** 2)
                )
            model = amplitudes[local_groups] * evolution
            data_residuals = (values - model) / local_sigma
            if not prior_slices:
                return data_residuals
            prior_residuals = [coeffs[sl] / width for sl, width in prior_slices]
            return np.concatenate([data_residuals, *prior_residuals])

        result = least_squares(residuals, previous, method="trf")
        coeff_samples[sample_id] = result.x
        previous = result.x
        dof = result.fun.size - n_groups - n_total_coeffs
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
        n_gamma_knots=n_knots,
        n_correction_params=n_correction_params,
        fit_a2_correction=fit_a2_correction,
        fit_fv_correction=fit_fv_correction,
        fit_pz2_correction=fit_pz2_correction,
        fit_apz2_correction=fit_apz2_correction,
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
        a_fm = group_obs[0].a_fm
        fv_prefactor = group_obs[0].fv_prefactor
        fv_exp_m_pi_bT = group_obs[0].fv_exp_m_pi_bT

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
            per_x_result.gamma_samples,
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
            # Build per-pz correction factor on gamma_MSbar inside the exponent.
            pz_corr = np.ones(len(pz_set), dtype=float)
            if per_x_result.fit_a2_correction and per_x_result.alpha_samples is not None:
                pz_corr += (a_fm / CORRECTION_A0_FM) ** 2 * _evaluate_correction_shape(
                    "a2",
                    per_x_result.alpha_samples[sample_id],
                    np.asarray([bT_fm]),
                )[0]
            if per_x_result.fit_fv_correction and per_x_result.beta_samples is not None:
                pz_corr += fv_prefactor * _evaluate_correction_shape(
                    "fv",
                    per_x_result.beta_samples[sample_id],
                    np.asarray([bT_fm]),
                    fv_exp_m_pi_bT=np.asarray([fv_exp_m_pi_bT]),
                )[0]
            if per_x_result.fit_pz2_correction and per_x_result.kappa_samples is not None:
                kappa_val = _evaluate_correction_shape(
                    "pz2",
                    per_x_result.kappa_samples[sample_id],
                    np.asarray([bT_fm]),
                )[0]
                x_weight = 1.0 / per_x_result.x_actual ** 2 + 1.0 / (1.0 - per_x_result.x_actual) ** 2
                pz_corr += x_weight * (CORRECTION_P0_GEV / pz_arr) ** 2 * kappa_val
            if per_x_result.fit_apz2_correction and per_x_result.lambda_samples is not None:
                lam_val = _evaluate_correction_shape(
                    "apz2",
                    per_x_result.lambda_samples[sample_id],
                    np.asarray([bT_fm]),
                )[0]
                pz_corr += (a_fm * pz_arr / HBAR_C_GEV_FM) ** 2 * lam_val
            evolution = _evaluate_evolution_factor(
                log_p,
                gamma_per_sample[sample_id] + np.zeros_like(pz_arr, dtype=float),
                corrections,
                pz_corr,
            )
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
    """Evaluate gamma_MSbar(bT) from spline coefficients.

    coeffs may be 1D (single sample) or 2D (n_samples, n_knots).
    Returns shape (n_bT,) for 1D input or (n_samples, n_bT) for 2D input.
    """
    basis = _spline_basis(np.asarray(bT_values_fm, dtype=float), bT_knots_fm, kind=spline_kind)
    if coeffs.ndim == 1:
        return basis @ coeffs
    return coeffs @ basis.T


def _write_joint_outputs(
    spec: TMDWFCSKernelJointInput,
    x_output_order: list[float],
    per_x_results: list[PerXFitResult],
    x_independent_fit_order: list[float],
    independent_results: list[PerXFitResult],
    x_reflection_symmetry: bool,
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
            gamma = result.gamma_samples
            for j, bT_val in enumerate(result.bT_knots_fm):
                values = gamma[:, j]
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
            gamma = result.gamma_samples
            for sample_id in range(gamma.shape[0]):
                coeffs = gamma[sample_id]
                for j, bT_val in enumerate(result.bT_knots_fm):
                    handle.write(
                        f"{result.x_actual:.10e}\t{bT_val:.10e}\t"
                        f"{sample_id}\t{coeffs[j]:.10e}\n"
                    )

    # -- coefficients (gamma spline coefficients for every bootstrap sample) --
    coeff_path = samples_dir / f"{stem}_coefficients.txt"
    with coeff_path.open("w", encoding="utf-8") as handle:
        header_cols = [f"c{j}" for j in range(bT_knots.size)]
        handle.write("x\tsample_id\t" + "\t".join(header_cols) + "\n")
        for result in per_x_results:
            gamma = result.gamma_samples
            for sample_id in range(gamma.shape[0]):
                coeff_str = "\t".join(f"{c:.10e}" for c in gamma[sample_id])
                handle.write(f"{result.x_actual:.10e}\t{sample_id}\t{coeff_str}\n")

    # -- correction coefficient files (one per enabled correction) --
    correction_stems = []
    if per_x_results[0].fit_a2_correction:
        correction_stems.append(("a2", "alpha"))
    if per_x_results[0].fit_fv_correction:
        correction_stems.append(("fv", "beta"))
    if per_x_results[0].fit_pz2_correction:
        correction_stems.append(("pz2", "kappa"))
    if per_x_results[0].fit_apz2_correction:
        correction_stems.append(("apz2", "lambda"))

    correction_output_paths: list[Path] = []
    for short_name, full_name in correction_stems:
        corr_coeff_path = samples_dir / f"{stem}_coefficients_{full_name}.txt"
        with corr_coeff_path.open("w", encoding="utf-8") as handle:
            header_cols = [f"{full_name}_0"] if short_name == "apz2" else [f"{full_name}_0", f"{full_name}_1"]
            handle.write("x\tsample_id\t" + "\t".join(header_cols) + "\n")
            for result in per_x_results:
                if short_name == "a2":
                    samples = result.alpha_samples
                elif short_name == "fv":
                    samples = result.beta_samples
                elif short_name == "pz2":
                    samples = result.kappa_samples
                else:
                    samples = result.lambda_samples
                if samples is None:
                    continue
                for sample_id in range(samples.shape[0]):
                    coeff_str = "\t".join(f"{c:.10e}" for c in samples[sample_id])
                    handle.write(f"{result.x_actual:.10e}\t{sample_id}\t{coeff_str}\n")
        correction_output_paths.append(corr_coeff_path)

        # Correction surface table
        corr_surface_path = tables_dir / f"{stem}_surface_{short_name}.txt"
        with corr_surface_path.open("w", encoding="utf-8") as handle:
            handle.write(f"x\tbT_fm\t{short_name}_p16\t{short_name}_p50\t{short_name}_p84\n")
            for result in per_x_results:
                if short_name == "a2":
                    samples = result.alpha_samples
                elif short_name == "fv":
                    samples = result.beta_samples
                elif short_name == "pz2":
                    samples = result.kappa_samples
                else:
                    samples = result.lambda_samples
                if samples is None:
                    continue
                bT_values = result.bT_knots_fm
                if short_name in {"a2", "pz2"}:
                    bT_values = bT_values[bT_values > 0.0]
                fv_exp_values = None
                if short_name == "fv":
                    m_pi_mev = spec.ensembles[0].m_pi_mev
                    fv_exp_values = np.exp(m_pi_mev * bT_values / HBAR_C_MEV_FM)
                for bT_val in bT_values:
                    idx = int(np.where(bT_values == bT_val)[0][0])
                    values = _evaluate_correction_shape(
                        short_name,
                        samples,
                        np.asarray([bT_val], dtype=float),
                        fv_exp_m_pi_bT=(
                            np.asarray([fv_exp_values[idx]], dtype=float)
                            if fv_exp_values is not None
                            else None
                        ),
                    )[:, 0]
                    q16, q50, q84 = _legacy_quantile_triplet(values)
                    handle.write(
                        f"{result.x_actual:.10e}\t{bT_val:.10e}\t"
                        f"{q16:.10e}\t{q50:.10e}\t{q84:.10e}\n"
                    )
        correction_output_paths.append(corr_surface_path)

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
        handle.write("correction_model analytic_two_parameter_gamma_multiplicative\n")
        handle.write(f"correction_a0_fm {CORRECTION_A0_FM:.10e}\n")
        handle.write(f"correction_b0_fm {CORRECTION_B0_FM:.10e}\n")
        handle.write(f"correction_p0_GeV {CORRECTION_P0_GEV:.10e}\n")
        handle.write(f"x_fit_points {' '.join(f'{v:.10e}' for v in x_output_order)}\n")
        handle.write(f"x_reflection_symmetry {str(x_reflection_symmetry).lower()}\n")
        handle.write(f"x_independent_fit_points {' '.join(f'{v:.10e}' for v in x_independent_fit_order)}\n")
        handle.write(f"fit_a2_correction {str(spec.fit_a2_correction).lower()}\n")
        handle.write(f"fit_fv_correction {str(spec.fit_fv_correction).lower()}\n")
        handle.write(f"fit_pz2_correction {str(spec.fit_pz2_correction).lower()}\n")
        handle.write(f"fit_apz2_correction {str(spec.fit_apz2_correction).lower()}\n")
        handle.write(f"a2_correction_prior_width {spec.a2_correction_prior_width:.10e}\n")
        handle.write(f"fv_correction_prior_width {spec.fv_correction_prior_width:.10e}\n")
        handle.write(f"pz2_correction_prior_width {spec.pz2_correction_prior_width:.10e}\n")
        handle.write(f"apz2_correction_prior_width {spec.apz2_correction_prior_width:.10e}\n")
        handle.write(f"n_gamma_knots {bT_knots.size}\n")
        if per_x_results:
            handle.write(f"n_correction_params {per_x_results[0].n_correction_params}\n")
        handle.write(f"n_output_x_points {len(per_x_results)}\n")
        handle.write(f"n_independent_x_fits {len(independent_results)}\n")
        handle.write(f"plot {str(spec.make_plots).lower()}\n")
        handle.write(f"progress {str(spec.show_progress).lower()}\n")
        if spec.progress_every is not None:
            handle.write(f"progress_every {spec.progress_every}\n")
        handle.write(f"n_ensembles {len(spec.ensembles)}\n")
        total_obs = sum(r.n_observations for r in independent_results)
        total_groups = sum(r.n_groups for r in independent_results)
        handle.write(f"n_observations_total {total_obs}\n")
        handle.write(f"n_nuisance_groups_total {total_groups}\n")
        for ensemble in spec.ensembles:
            handle.write(
                f"ensemble {ensemble.label} {ensemble.input_root} "
                f"{ensemble.title_pattern} {ensemble.ns} "
                f"{ensemble.lattice_spacing_fm:.10e} "
                f"pz={','.join(str(v) for v in ensemble.pzlist)} "
                f"bT={','.join(str(v) for v in ensemble.bTlist)} "
                f"m_pi={ensemble.m_pi_mev:.1f}\n"
            )

    outputs = [summary_path, surface_path, samples_path, coeff_path, *correction_output_paths, diagnostics_path]

    # -- plots: gamma_eff vs x band for each bT knot --
    if spec.make_plots:
        x_values = np.asarray([r.x_actual for r in per_x_results], dtype=float)
        for bT_val in bT_knots:
            surface_samples = np.asarray(
                [
                    _evaluate_bT_surface(
                        result.gamma_samples[si],
                        np.asarray([bT_val]),
                        result.bT_knots_fm,
                        spec.spline_kind,
                    )[0]
                    for result in per_x_results
                    for si in range(result.gamma_samples.shape[0])
                ],
                dtype=float,
            ).reshape(len(per_x_results), per_x_results[0].gamma_samples.shape[0])
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
    x_plan = _plan_x_reflection(x_indices)

    bT_fm_values: list[float] = []
    for ensemble, _ in ensemble_datasets:
        for bT in ensemble.bTlist:
            bT_fm_values.append(float(bT * ensemble.lattice_spacing_fm))
    bT_knots_fm = _default_bT_knots(np.array(bT_fm_values), spec.bT_knots_fm, max_count=8)

    if spec.show_progress:
        print(
            f"joint CS fit: {len(x_plan.fit_entries)} independent x-fits "
            f"for {len(x_plan.output_entries)} output x-points, "
            f"{bT_knots_fm.size} bT-knots, "
            f"{len(spec.ensembles)} ensembles, "
            f"{sample_count} bootstrap samples",
            flush=True,
        )

    independent_results: list[PerXFitResult] = []
    x_independent_fit_order: list[float] = []
    results_by_fit_key: dict[tuple[int, float], PerXFitResult] = {}
    diagnostic_plot_paths: list[Path] = []
    for xi, (x_idx, x_actual) in enumerate(x_plan.fit_entries):
        if spec.show_progress:
            print(
                f"x [{xi + 1}/{len(x_plan.fit_entries)}] x_actual={x_actual:.6f}",
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
            fit_a2_correction=spec.fit_a2_correction,
            fit_fv_correction=spec.fit_fv_correction,
            fit_pz2_correction=spec.fit_pz2_correction,
            fit_apz2_correction=spec.fit_apz2_correction,
            a2_correction_prior_width=spec.a2_correction_prior_width,
            fv_correction_prior_width=spec.fv_correction_prior_width,
            pz2_correction_prior_width=spec.pz2_correction_prior_width,
            apz2_correction_prior_width=spec.apz2_correction_prior_width,
        )
        independent_results.append(result)
        x_independent_fit_order.append(x_actual)
        results_by_fit_key[(x_idx, round(x_actual, 12))] = result

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

    if not independent_results:
        raise ValueError("no x-points produced valid observations; check x_knots vs x_window")

    per_x_results: list[PerXFitResult] = []
    x_output_order: list[float] = []
    for key, (_, x_actual) in zip(x_plan.output_fit_keys, x_plan.output_entries, strict=True):
        if key not in results_by_fit_key:
            continue
        per_x_results.append(replace(results_by_fit_key[key], x_actual=x_actual))
        x_output_order.append(x_actual)

    outputs = _write_joint_outputs(
        spec,
        x_output_order,
        per_x_results,
        x_independent_fit_order,
        independent_results,
        x_plan.uses_reflection,
    )
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
                m_pi_str = tokens[8].split("=", 1)[1] if len(tokens) > 8 else "140.0"
                ensembles.append({
                    "label": tokens[1],
                    "input_root": tokens[2],
                    "title_pattern": tokens[3],
                    "ns": int(tokens[4]),
                    "lattice_spacing_fm": float(tokens[5]),
                    "pzlist": _parse_int_items(pz_str),
                    "bTlist": _parse_int_items(bT_str),
                    "m_pi_mev": float(m_pi_str),
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
        "fit_a2_correction": entries.get("fit_a2_correction", ["false"])[0].lower() == "true",
        "fit_fv_correction": entries.get("fit_fv_correction", ["false"])[0].lower() == "true",
        "fit_pz2_correction": entries.get("fit_pz2_correction", ["false"])[0].lower() == "true",
        "fit_apz2_correction": entries.get("fit_apz2_correction", ["false"])[0].lower() == "true",
        "a2_correction_prior_width": float(entries.get("a2_correction_prior_width", ["1.0"])[0]),
        "fv_correction_prior_width": float(entries.get("fv_correction_prior_width", ["1.0"])[0]),
        "pz2_correction_prior_width": float(entries.get("pz2_correction_prior_width", ["1.0"])[0]),
        "apz2_correction_prior_width": float(entries.get("apz2_correction_prior_width", ["1.0"])[0]),
        "n_gamma_knots": int(entries["n_gamma_knots"][0]) if "n_gamma_knots" in entries else len(entries["bT_knots_fm"]),
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
