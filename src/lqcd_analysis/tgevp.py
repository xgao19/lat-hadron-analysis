from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import LinAlgError, eigh
from scipy.stats import gaussian_kde

from .io import load_correlator_csv
from .utils import (
    apply_fold_t,
    bin_correlators,
    bootstrap_correlator_means,
    parse_fold_t,
    robust_mean_and_error,
)

HBAR_C_GEV_FM = 0.1973269804


@dataclass(frozen=True)
class TGEVPInput:
    title_pattern: str
    ns: int
    nt: int
    lattice_spacing_fm: float
    correlator_path_pattern: str
    pzlist: tuple[int, ...]
    fold_t: str
    tsrange: tuple[int, int]
    binsize: int
    bootstrap_samples: int | None
    bootstrap_size: int | None
    seed: int
    results_dir: Path


@dataclass(frozen=True)
class AnalysisOptions:
    binsize: int = 1
    bootstrap_samples: int | None = None
    bootstrap_size: int | None = None
    seed: int | None = None
    results_dir: Path = Path("results")


@dataclass(frozen=True)
class SingleTSResult:
    ts: int
    e0_mean: float
    e0_err: float
    e1_mean: float
    e1_err: float
    z0sq_mean: float
    z0sq_err: float
    z1sq_mean: float
    z1sq_err: float
    corr_e0_e1: float
    n_accepted: int
    n_samples: int


def parse_optional_int(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    return int(value)


def parse_tgevp_input(path: str | Path) -> TGEVPInput:
    file_path = Path(path)
    entries: dict[str, list[str]] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            entries[parts[0]] = parts[1:]

    required = {"c2pt", "pzlist", "tsrange"}
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")

    title_parts = next(
        (
            line.split()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ),
        None,
    )
    if title_parts is None or len(title_parts) < 4:
        raise ValueError("the first non-empty line must be: title Ns Nt a_fm")

    title_pattern = title_parts[0]
    ns = int(title_parts[1])
    nt = int(title_parts[2])
    lattice_spacing_fm = float(title_parts[3])
    fold_t = parse_fold_t(entries)
    tsrange = (int(entries["tsrange"][0]), int(entries["tsrange"][1]))
    if tsrange[0] != 0:
        raise ValueError("this SS two-point TGEVP driver currently expects tsrange to start at 0")

    return TGEVPInput(
        title_pattern=title_pattern,
        ns=ns,
        nt=nt,
        lattice_spacing_fm=lattice_spacing_fm,
        correlator_path_pattern=entries["c2pt"][0],
        pzlist=tuple(int(item) for item in entries["pzlist"]),
        fold_t=fold_t,
        tsrange=tsrange,
        binsize=int(entries.get("binsize", ["1"])[0]),
        bootstrap_samples=parse_optional_int(entries.get("bootstrap_samples", ["auto"])[0]),
        bootstrap_size=parse_optional_int(entries.get("bootstrap_size", ["auto"])[0]),
        seed=int(entries.get("seed", ["2026"])[0]),
        results_dir=(
            Path(entries["results_dir"][0])
            if "results_dir" in entries
            else file_path.resolve().parent / "results"
        ),
    )
def build_tgevp_matrices(correlator: np.ndarray, ts: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(correlator, dtype=float)
    max_needed = 2 * ts + 1
    if values.ndim != 1:
        raise ValueError("correlator must be one-dimensional")
    if max_needed >= len(values):
        raise ValueError("correlator is too short for the requested ts")

    indices = np.arange(ts + 1)
    denom = np.sqrt(np.outer(values[2 * indices], values[2 * indices]))
    if np.any(~np.isfinite(denom)) or np.any(denom <= 0.0):
        raise ValueError("non-positive normalization encountered in TGEVP matrices")

    i_plus_j = np.add.outer(indices, indices)
    t_matrix = values[i_plus_j + 1] / denom
    v_matrix = values[i_plus_j] / denom
    return 0.5 * (t_matrix + t_matrix.T), 0.5 * (v_matrix + v_matrix.T)


def solve_tgevp(
    correlator: np.ndarray,
    ts: int,
    n_states: int = 2,
    regularization: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_matrix, v_matrix = build_tgevp_matrices(correlator, ts)
    dim = t_matrix.shape[0]
    scale = max(1.0, np.trace(v_matrix) / dim)

    last_error: Exception | None = None
    for factor in (0.0, 1.0, 10.0, 100.0):
        try:
            eigvals, eigvecs = eigh(t_matrix, v_matrix + factor * regularization * scale * np.eye(dim))
            break
        except LinAlgError as exc:
            last_error = exc
    else:
        eigvals, eigvecs = solve_tgevp_projected(
            t_matrix,
            v_matrix,
            n_states=n_states,
            relative_cutoff=regularization,
        )
        overlaps = compute_ss_overlaps(eigvecs, correlator, ts)
        return eigvals, eigvecs, overlaps

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[:n_states]
    eigvecs = eigvecs[:, :n_states]

    overlaps = compute_ss_overlaps(eigvecs, correlator, ts)
    return eigvals, eigvecs, overlaps


def solve_tgevp_projected(
    t_matrix: np.ndarray,
    v_matrix: np.ndarray,
    n_states: int,
    relative_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the TGEVP after projecting V onto its positive subspace."""
    v_evals, v_evecs = np.linalg.eigh(v_matrix)
    scale = max(np.max(np.abs(v_evals)), 1.0)
    keep = v_evals > relative_cutoff * scale
    if np.count_nonzero(keep) < n_states:
        raise ValueError("projected TGEVP rank is too small for the requested number of states")

    basis = v_evecs[:, keep] / np.sqrt(v_evals[keep])
    projected = basis.T @ t_matrix @ basis
    projected = 0.5 * (projected + projected.T)

    eigvals, coeffs = np.linalg.eigh(projected)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order][:n_states]
    coeffs = coeffs[:, order][:, :n_states]
    eigvecs = basis @ coeffs
    return eigvals, eigvecs


def compute_ss_overlaps(eigvecs: np.ndarray, correlator: np.ndarray, ts: int) -> np.ndarray:
    values = np.asarray(correlator, dtype=float)
    basis = np.arange(ts + 1)
    overlap_kernel = values[basis] / np.sqrt(values[2 * basis])
    amplitudes = overlap_kernel @ eigvecs
    return np.abs(amplitudes) ** 2


def tgevp_energies_from_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(eigenvalues, dtype=float)
    if np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("TGEVP eigenvalues must satisfy 0 < lambda <= 1")
    return -np.log(values)
def kde_peak_mask(samples: np.ndarray, grid_size: int = 512) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.size < 8:
        return np.ones(values.shape, dtype=bool)
    if np.allclose(values, values[0]):
        return np.ones(values.shape, dtype=bool)

    xmin = float(np.min(values))
    xmax = float(np.max(values))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        return np.ones(values.shape, dtype=bool)

    kde = gaussian_kde(values)
    padding = 0.1 * (xmax - xmin)
    grid = np.linspace(xmin - padding, xmax + padding, grid_size)
    density = kde(grid)
    peak_index = int(np.argmax(density))
    half_max = 0.5 * density[peak_index]

    left = peak_index
    while left > 0 and density[left] >= half_max:
        left -= 1
    right = peak_index
    while right < grid_size - 1 and density[right] >= half_max:
        right += 1

    center = grid[peak_index]
    left_edge = grid[max(left, 0)]
    right_edge = grid[min(right, grid_size - 1)]
    half_width = max(center - left_edge, right_edge - center)
    if half_width <= 0.0:
        return np.ones(values.shape, dtype=bool)
    return np.abs(values - center) <= 3.0 * half_width


def analyze_single_correlator(
    correlators: np.ndarray,
    ts_max: int,
    options: AnalysisOptions,
) -> tuple[list[SingleTSResult], list[np.ndarray]]:
    binned = bin_correlators(correlators, binsize=options.binsize)
    bootstrap_means = bootstrap_correlator_means(
        binned,
        n_samples=options.bootstrap_samples,
        sample_size=options.bootstrap_size,
        seed=options.seed,
    )

    summaries: list[SingleTSResult] = []
    sample_rows: list[np.ndarray] = []

    for ts in range(2, ts_max + 1):
        per_sample = np.full((len(bootstrap_means), 7), np.nan, dtype=float)
        for sample_id, sample_corr in enumerate(bootstrap_means):
            try:
                eigvals, _, overlaps = solve_tgevp(sample_corr, ts, n_states=2)
                energies = tgevp_energies_from_eigenvalues(eigvals)
            except ValueError:
                continue

            per_sample[sample_id, 0] = ts
            per_sample[sample_id, 1] = sample_id
            per_sample[sample_id, 2] = energies[0]
            per_sample[sample_id, 3] = energies[1]
            per_sample[sample_id, 4] = overlaps[0]
            per_sample[sample_id, 5] = overlaps[1]

        finite = np.isfinite(per_sample[:, 2]) & np.isfinite(per_sample[:, 3])
        accepted = finite.copy()
        if np.any(finite):
            accepted[finite] &= kde_peak_mask(np.exp(-per_sample[finite, 2]))
            accepted[finite] &= kde_peak_mask(np.exp(-per_sample[finite, 3]))

        per_sample[:, 6] = accepted.astype(float)
        sample_rows.append(per_sample)

        kept = per_sample[accepted]
        e0_mean, e0_err = robust_mean_and_error(kept[:, 2])
        e1_mean, e1_err = robust_mean_and_error(kept[:, 3])
        z0_mean, z0_err = robust_mean_and_error(kept[:, 4])
        z1_mean, z1_err = robust_mean_and_error(kept[:, 5])
        corr = float(np.corrcoef(kept[:, 2], kept[:, 3])[0, 1]) if len(kept) > 1 else np.nan
        summaries.append(
            SingleTSResult(
                ts=ts,
                e0_mean=e0_mean,
                e0_err=e0_err,
                e1_mean=e1_mean,
                e1_err=e1_err,
                z0sq_mean=z0_mean,
                z0sq_err=z0_err,
                z1sq_mean=z1_mean,
                z1sq_err=z1_err,
                corr_e0_e1=corr,
                n_accepted=int(np.sum(accepted)),
                n_samples=len(per_sample),
            )
        )

    return summaries, sample_rows


def run_ss_2pt_tgevp(
    input_file: str | Path,
    *,
    binsize: int = 1,
    bootstrap_samples: int | None = None,
    bootstrap_size: int | None = None,
    seed: int | None = None,
    results_dir: str | Path | None = None,
) -> list[Path]:
    input_path = Path(input_file).resolve()
    spec = parse_tgevp_input(input_path)
    options = AnalysisOptions(
        binsize=spec.binsize if binsize is None else binsize,
        bootstrap_samples=spec.bootstrap_samples if bootstrap_samples is None else bootstrap_samples,
        bootstrap_size=spec.bootstrap_size if bootstrap_size is None else bootstrap_size,
        seed=spec.seed if seed is None else seed,
        results_dir=spec.results_dir if results_dir is None else Path(results_dir),
    )
    output_files: list[Path] = []

    results_path = options.results_dir
    samples_path = results_path / "samples"
    results_path.mkdir(parents=True, exist_ok=True)
    samples_path.mkdir(parents=True, exist_ok=True)

    for pz in spec.pzlist:
        title = spec.title_pattern.replace("*", str(pz))
        csv_path = spec.correlator_path_pattern.replace("*", str(pz))
        _, correlators = load_correlator_csv(csv_path)
        if correlators.shape[1] != spec.nt:
            raise ValueError(f"{csv_path} has Nt={correlators.shape[1]}, expected {spec.nt}")

        processed = apply_fold_t(correlators, spec.nt, spec.fold_t)
        t0, t1 = spec.tsrange
        selected = processed[:, t0 : t1 + 1]
        ts_max = min(t1, (selected.shape[1] - 2) // 2)
        if ts_max < 2:
            raise ValueError(f"tsrange {spec.tsrange} is too short for TGEVP analysis")

        summaries, sample_rows = analyze_single_correlator(selected, ts_max, options)
        summary_rows = np.array(
            [
                [
                    row.ts,
                    row.e0_mean,
                    row.e0_err,
                    row.e1_mean,
                    row.e1_err,
                    row.z0sq_mean,
                    row.z0sq_err,
                    row.z1sq_mean,
                    row.z1sq_err,
                    row.n_accepted,
                    row.n_samples,
                ]
                for row in summaries
            ],
            dtype=float,
        )
        corr_rows = np.array(
            [[row.ts, row.corr_e0_e1, row.n_accepted, row.n_samples] for row in summaries],
            dtype=float,
        )
        sample_table = np.vstack(sample_rows)

        energy_scale = HBAR_C_GEV_FM / spec.lattice_spacing_fm
        summary_file = results_path / f"{title}_tgevp_summary.txt"
        correlation_file = results_path / f"{title}_tgevp_correlation.txt"
        samples_file = samples_path / f"{title}_tgevp_samples.txt"

        np.savetxt(
            summary_file,
            summary_rows,
            header=(
                "ts E0_lat_mean E0_lat_err E1_lat_mean E1_lat_err "
                "Z0sq_mean Z0sq_err Z1sq_mean Z1sq_err n_accepted n_samples\n"
                f"# convert_to_GeV_by_multiplying_energy_columns_with {energy_scale:.12f}"
            ),
            fmt="%.10e",
        )
        np.savetxt(
            correlation_file,
            corr_rows,
            header="ts corr_E0_E1 n_accepted n_samples",
            fmt="%.10e",
        )
        np.savetxt(
            samples_file,
            sample_table,
            header="ts sample_id E0_lat E1_lat Z0sq Z1sq accepted",
            fmt="%.10e",
        )

        output_files.extend([summary_file, correlation_file, samples_file])

    return output_files
