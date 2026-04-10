from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from ..common.utils import robust_mean_and_error
from ..two_point.plotting import prepare_matplotlib, save_plot_status

DEFAULT_X_VALUES = np.linspace(-0.5, 1.5, 201)
DEFAULT_ZSTEP_FM = 0.01
DEFAULT_INTERPOLATION_KIND = "cubic"


@dataclass(frozen=True)
class TMDWFFourierInput:
    title: str
    stem: str
    pz: int
    ns: int
    lattice_spacing_fm: float
    component: str
    nstates: int
    bT: int
    fit_table: Path
    sample_table: Path
    x_values: np.ndarray
    zstep_fm: float
    interpolation_kind: str
    make_plots: bool
    results_dir: Path


def parse_tmdwf_fourier_input(
    path: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> TMDWFFourierInput:
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
        "pz",
        "ns",
        "lattice_spacing_fm",
        "component",
        "nstates",
        "bT",
        "fit_table",
        "sample_table",
    }
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")

    component = entries["component"][0].lower()
    if component not in {"real", "imag"}:
        raise ValueError("component must be one of: real, imag")

    if "x_values" in entries:
        x_values = np.array([float(token) for token in entries["x_values"]], dtype=float)
    elif "x_range" in entries:
        x_range = entries["x_range"]
        if len(x_range) < 2:
            raise ValueError("x_range must provide: xmin xmax")
        x_count = int(entries.get("x_count", [str(DEFAULT_X_VALUES.size)])[0])
        if x_count < 2:
            raise ValueError("x_count must be at least 2")
        x_values = np.linspace(float(x_range[0]), float(x_range[1]), x_count)
    else:
        x_values = np.asarray(DEFAULT_X_VALUES, dtype=float)

    zstep_fm = float(entries.get("zstep_fm", [str(DEFAULT_ZSTEP_FM)])[0])
    if zstep_fm <= 0.0:
        raise ValueError("zstep_fm must be positive")

    fit_table = Path(entries["fit_table"][0])
    sample_table = Path(entries["sample_table"][0])
    if not fit_table.exists():
        raise FileNotFoundError(f"TMDWF Fourier fit table does not exist: {fit_table}")
    if not sample_table.exists():
        raise FileNotFoundError(f"TMDWF Fourier sample table does not exist: {sample_table}")

    stem = entries.get("stem", [f"{entries.get('title', ['tmdwf'])[0]}_bT{int(entries['bT'][0])}"])[0]
    title = entries.get("title", [stem])[0]
    output_root = Path(results_dir) if results_dir is not None else Path(entries.get("results_dir", [file_path.parent / "results_tmdwf_fourier"])[0])

    return TMDWFFourierInput(
        title=title,
        stem=stem,
        pz=int(entries["pz"][0]),
        ns=int(entries["ns"][0]),
        lattice_spacing_fm=float(entries["lattice_spacing_fm"][0]),
        component=component,
        nstates=int(entries["nstates"][0]),
        bT=int(entries["bT"][0]),
        fit_table=fit_table,
        sample_table=sample_table,
        x_values=x_values,
        zstep_fm=zstep_fm,
        interpolation_kind=entries.get("interpolation_kind", [DEFAULT_INTERPOLATION_KIND])[0],
        make_plots=entries.get("plot", ["true"])[0].lower() not in {"false", "0", "no"},
        results_dir=output_root,
    )


def _parse_grouped_table(path: str | Path) -> tuple[list[str], list[list[str]]]:
    header: list[str] | None = None
    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if "\t" not in line:
                continue
            tokens = line.split("\t")
            if header is None:
                header = tokens
            else:
                rows.append(tokens)
    if header is None:
        raise ValueError(f"grouped table is missing a tab-delimited header: {path}")
    return header, rows


def load_tmdwf_m0_fit_table(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header, rows = _parse_grouped_table(path)
    index = {name: idx for idx, name in enumerate(header)}
    required = {"bz", "m0_mean", "m0_err"}
    missing = required - index.keys()
    if missing:
        raise ValueError(f"grouped fit table is missing columns: {sorted(missing)}")
    sorted_rows = sorted(rows, key=lambda row: int(row[index["bz"]]))
    bz = np.array([int(row[index["bz"]]) for row in sorted_rows], dtype=int)
    mean = np.array([float(row[index["m0_mean"]]) for row in sorted_rows], dtype=float)
    err = np.array([float(row[index["m0_err"]]) for row in sorted_rows], dtype=float)
    return bz, mean, err


def load_tmdwf_m0_sample_table(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    header, rows = _parse_grouped_table(path)
    index = {name: idx for idx, name in enumerate(header)}
    required = {"bz", "sample_id", "success", "m0"}
    missing = required - index.keys()
    if missing:
        raise ValueError(f"grouped sample table is missing columns: {sorted(missing)}")

    bz_values = sorted({int(row[index["bz"]]) for row in rows})
    sample_maps: dict[int, dict[int, float]] = {}
    for row in rows:
        success = int(row[index["success"]])
        value = float(row[index["m0"]])
        if success != 1 or not np.isfinite(value):
            continue
        sample_id = int(row[index["sample_id"]])
        sample_maps.setdefault(sample_id, {})[int(row[index["bz"]])] = value

    valid_sample_ids = sorted(
        sample_id
        for sample_id, by_bz in sample_maps.items()
        if all(bz in by_bz and np.isfinite(by_bz[bz]) for bz in bz_values)
    )
    samples = np.array(
        [[sample_maps[sample_id][bz] for bz in bz_values] for sample_id in valid_sample_ids],
        dtype=float,
    )
    if samples.size == 0:
        samples = np.empty((0, len(bz_values)), dtype=float)
    return np.array(bz_values, dtype=int), samples


def _resolve_interpolation_kind(kind: str, n_points: int) -> str:
    normalized = kind.lower()
    if normalized == "cubic" and n_points < 4:
        return "linear" if n_points < 3 else "quadratic"
    if normalized == "quadratic" and n_points < 3:
        return "linear"
    return normalized


def compute_tmdwf_cosine_transform(
    bz_values: np.ndarray,
    m0_samples: np.ndarray,
    *,
    pz: int,
    ns: int,
    lattice_spacing_fm: float,
    x_values: np.ndarray | None = None,
    zstep_fm: float = DEFAULT_ZSTEP_FM,
    interpolation_kind: str = DEFAULT_INTERPOLATION_KIND,
) -> tuple[np.ndarray, np.ndarray]:
    bz_array = np.asarray(bz_values, dtype=int)
    sample_array = np.asarray(m0_samples, dtype=float)
    if bz_array.ndim != 1:
        raise ValueError("bz_values must be one-dimensional")
    if sample_array.ndim != 2 or sample_array.shape[1] != bz_array.size:
        raise ValueError("m0_samples must have shape (n_samples, n_bz)")
    if zstep_fm <= 0.0:
        raise ValueError("zstep_fm must be positive")
    x_grid = np.asarray(DEFAULT_X_VALUES if x_values is None else x_values, dtype=float)
    z_nodes = bz_array.astype(float) * float(lattice_spacing_fm)
    zmax = float(np.max(z_nodes))
    if zmax <= 0.0:
        z_grid = np.array([0.0], dtype=float)
    else:
        n_steps = int(np.ceil(zmax / zstep_fm))
        z_grid = np.linspace(0.0, zmax, n_steps + 1, dtype=float)
    if bz_array.size == 1:
        h_grid = np.repeat(sample_array[:, :1], z_grid.size, axis=1)
    else:
        kind = _resolve_interpolation_kind(interpolation_kind, bz_array.size)
        interpolator = interp1d(
            z_nodes,
            sample_array,
            axis=1,
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate",
        )
        h_grid = np.asarray(interpolator(z_grid), dtype=float)
    pz_phys = 2.0 * np.pi * float(pz) / (float(ns) * float(lattice_spacing_fm))
    phase = np.cos((x_grid[:, None] - 0.5) * pz_phys * z_grid[None, :])
    prefactor = pz_phys / np.pi
    transformed = prefactor * np.trapz(h_grid[:, None, :] * phase[None, :, :], z_grid, axis=2)
    return x_grid, np.asarray(transformed, dtype=float)


def summarize_tmdwf_fourier_samples(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2:
        raise ValueError("Fourier bootstrap samples must be two-dimensional")
    p16 = np.percentile(sample_array, 16.0, axis=0)
    p84 = np.percentile(sample_array, 84.0, axis=0)
    mean = 0.5 * (p16 + p84)
    err = 0.5 * (p84 - p16)
    return mean, err, p16, p84


def plot_tmdwf_fourier_transform(
    output_path: str | Path,
    x_values: np.ndarray,
    q_mean: np.ndarray,
    q_err: np.ndarray,
    q_p16: np.ndarray,
    q_p84: np.ndarray,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (6.5, 4.5),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Path:
    output_path = Path(output_path)
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(x_values, q_p16, q_p84, color="C0", alpha=0.2, linewidth=0)
    ax.errorbar(x_values, q_mean, yerr=q_err, fmt="-", linewidth=1.2, color="C0")
    ax.set_xlabel("x")
    ax.set_ylabel("q(x)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def run_tmdwf_fourier_from_fit_outputs(
    *,
    output_root: str | Path,
    stem: str,
    fit_table: str | Path,
    sample_table: str | Path,
    pz: int,
    ns: int,
    lattice_spacing_fm: float,
    bT: int,
    component: str,
    nstates: int,
    x_values: np.ndarray | None = None,
    zstep_fm: float = DEFAULT_ZSTEP_FM,
    interpolation_kind: str = DEFAULT_INTERPOLATION_KIND,
    make_plots: bool = False,
    plot_figsize: tuple[float, float] = (6.5, 4.5),
    plot_xlim: tuple[float, float] | None = None,
    plot_ylim: tuple[float, float] | None = None,
) -> list[Path]:
    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    samples_dir = output_root / "samples"
    plots_dir = output_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    bz_fit, mean_fit, err_fit = load_tmdwf_m0_fit_table(fit_table)
    bz_samples, m0_samples = load_tmdwf_m0_sample_table(sample_table)
    if not np.array_equal(bz_fit, bz_samples):
        raise ValueError("fit/sample bz grids do not match for TMDWF Fourier transform")

    x_grid = np.asarray(DEFAULT_X_VALUES if x_values is None else x_values, dtype=float)
    if m0_samples.shape[0] > 0:
        _, q_samples = compute_tmdwf_cosine_transform(
            bz_fit,
            m0_samples,
            pz=pz,
            ns=ns,
            lattice_spacing_fm=lattice_spacing_fm,
            x_values=x_grid,
            zstep_fm=zstep_fm,
            interpolation_kind=interpolation_kind,
        )
        q_mean, q_err, q_p16, q_p84 = summarize_tmdwf_fourier_samples(q_samples)
    else:
        _, q_mean_only = compute_tmdwf_cosine_transform(
            bz_fit,
            mean_fit[None, :],
            pz=pz,
            ns=ns,
            lattice_spacing_fm=lattice_spacing_fm,
            x_values=x_grid,
            zstep_fm=zstep_fm,
            interpolation_kind=interpolation_kind,
        )
        q_mean = q_mean_only[0]
        q_err = np.full_like(q_mean, np.nan)
        q_p16 = np.full_like(q_mean, np.nan)
        q_p84 = np.full_like(q_mean, np.nan)
        q_samples = np.empty((0, x_grid.size), dtype=float)

    table_path = tables_dir / f"{stem}_{component}_{nstates}state_fourier.txt"
    sample_path = samples_dir / f"{stem}_{component}_{nstates}state_fourier_samples.txt"
    plot_path = plots_dir / f"{stem}_{component}_{nstates}state_fourier.pdf"

    with table_path.open("w", encoding="utf-8") as handle:
        handle.write(f"pz {pz}\n")
        handle.write(f"bT {bT}\n")
        handle.write(f"component {component}\n")
        handle.write(f"nstates {nstates}\n")
        handle.write(f"lattice_spacing_fm {lattice_spacing_fm:.10e}\n")
        handle.write(f"zstep_fm {zstep_fm:.10e}\n")
        handle.write(f"interpolation_kind {_resolve_interpolation_kind(interpolation_kind, bz_fit.size)}\n")
        handle.write("x\tq_mean\tq_err\tq_p16\tq_p84\n")
        for x_value, mean_value, err_value, p16_value, p84_value in zip(x_grid, q_mean, q_err, q_p16, q_p84, strict=True):
            handle.write(
                "\t".join(
                    [
                        f"{x_value:.10e}",
                        f"{mean_value:.10e}",
                        f"{err_value:.10e}",
                        f"{p16_value:.10e}",
                        f"{p84_value:.10e}",
                    ]
                )
                + "\n"
            )

    with sample_path.open("w", encoding="utf-8") as handle:
        handle.write("sample_id\tx\tq_sample\n")
        for sample_id, sample_values in enumerate(q_samples):
            for x_value, q_value in zip(x_grid, sample_values, strict=True):
                handle.write(f"{sample_id}\t{x_value:.10e}\t{q_value:.10e}\n")

    outputs = [table_path, sample_path]
    if make_plots:
        outputs.append(
            plot_tmdwf_fourier_transform(
                plot_path,
                x_grid,
                q_mean,
                q_err,
                q_p16,
                q_p84,
                title=f"{stem} {component} {nstates}state bT{bT} Fourier",
                figsize=plot_figsize,
                xlim=plot_xlim,
                ylim=plot_ylim,
            )
        )
    return outputs


def run_tmdwf_fourier_workflow(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_fourier_input(input_file, results_dir=results_dir)
    return run_tmdwf_fourier_from_fit_outputs(
        output_root=spec.results_dir,
        stem=spec.stem,
        fit_table=spec.fit_table,
        sample_table=spec.sample_table,
        pz=spec.pz,
        ns=spec.ns,
        lattice_spacing_fm=spec.lattice_spacing_fm,
        bT=spec.bT,
        component=spec.component,
        nstates=spec.nstates,
        x_values=spec.x_values,
        zstep_fm=spec.zstep_fm,
        interpolation_kind=spec.interpolation_kind,
        make_plots=spec.make_plots,
    )
