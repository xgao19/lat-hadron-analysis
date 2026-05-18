from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from ..common.constants import MIN_POSITIVE
from ..common.parsing import load_fit_window_table, parse_bool, parse_int_list_or_range
from ..common.utils import robust_mean_and_error
from .fit_nstate import _load_two_point_sample_parameters, _two_point_sample_table_path, sanitize_token
from .io import expand_template, resolve_two_point_fit_reference
from .models import evaluate_tmdwf_ratio, normalize_tmdwf_operator
from ..two_point.plotting import prepare_matplotlib, save_plot_status


DEFAULT_XFIT_PLOT_PANELS = 9


@dataclass(frozen=True)
class TMDWFXNStateFitInput:
    title_pattern: str
    input_root: Path
    ns: int
    nt: int
    lattice_spacing_fm: float
    pzlist: tuple[int, ...]
    gmlist: tuple[str, ...]
    etalist: tuple[str, ...]
    bTlist: tuple[int, ...]
    component: str
    nstates: tuple[int, ...]
    fit_window: str
    two_point_fit_root: str
    two_point_fit_window_by_pz: dict[int, tuple[int, int]]
    two_point_fit_sample_coupled: bool
    results_dir: Path


@dataclass(frozen=True)
class XFitResult:
    params: np.ndarray
    chi2_dof: float
    pvalue: float
    success: bool


def parse_tmdwf_x_nstate_fit_input(
    path: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> TMDWFXNStateFitInput:
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
        "nt",
        "lattice_spacing_fm",
        "pzlist",
        "gmlist",
        "etalist",
        "component",
        "nstates",
        "fit_window",
        "two_point_fit_root",
        "two_point_fit_window_by_pz",
    }
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")
    if "bTlist" not in entries and "bTrange" not in entries:
        raise ValueError(f"missing required key in {file_path}: bTlist or bTrange")

    component = entries["component"][0].lower()
    if component not in {"real", "imag"}:
        raise ValueError("component must be one of: real, imag")
    nstates = tuple(sorted({int(item) for item in entries["nstates"]}))
    if not nstates or any(state not in {1, 2} for state in nstates):
        raise ValueError("TMDWF x-space nstates must contain only 1 and/or 2")
    for gm in entries["gmlist"]:
        normalize_tmdwf_operator(gm)

    two_point_windows = load_fit_window_table(entries["two_point_fit_window_by_pz"][0])
    pzlist = tuple(int(item) for item in entries["pzlist"])
    missing_windows = sorted(set(pzlist) - {pz for gm, pz in two_point_windows if gm is None})
    if missing_windows:
        raise ValueError(f"two_point_fit_window_by_pz is missing entries for pz values: {missing_windows}")

    return TMDWFXNStateFitInput(
        title_pattern=entries["title_pattern"][0],
        input_root=Path(entries["input_root"][0]),
        ns=int(entries["ns"][0]),
        nt=int(entries["nt"][0]),
        lattice_spacing_fm=float(entries["lattice_spacing_fm"][0]),
        pzlist=pzlist,
        gmlist=tuple(entries["gmlist"]),
        etalist=tuple(entries["etalist"]),
        bTlist=parse_int_list_or_range(entries, "bTlist", "bTrange"),
        component=component,
        nstates=nstates,
        fit_window=entries["fit_window"][0],
        two_point_fit_root=entries["two_point_fit_root"][0],
        two_point_fit_window_by_pz={
            pz: window for (gm, pz), window in two_point_windows.items() if gm is None
        },
        two_point_fit_sample_coupled=parse_bool(entries.get("two_point_fit_sample_coupled", ["false"])[0]),
        results_dir=(
            Path(results_dir)
            if results_dir is not None
            else Path(entries.get("results_dir", [file_path.parent / "results_tmdwf_x_nstate_fit"])[0])
        ),
    )


def resolve_ratio_fourier_t_sample_path(
    input_root: str | Path,
    *,
    title: str,
    gm: str,
    eta: str,
    bT: int,
    component: str,
) -> Path:
    stem = f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_bT{bT}_{component}_ratio_fourier_t_samples.txt"
    path = Path(input_root) / title / "samples" / stem
    if not path.exists():
        raise FileNotFoundError(f"TMDWF ratio-Fourier-t sample table does not exist: {path}")
    return path


def load_ratio_fourier_t_samples(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header: list[str] | None = None
    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split("\t")
            if header is None:
                header = tokens
            else:
                rows.append(tokens)
    if header is None:
        raise ValueError(f"sample table is empty: {path}")
    index = {name: idx for idx, name in enumerate(header)}
    required = {"sample_id", "t", "x", "q_sample"}
    missing = required - index.keys()
    if missing:
        raise ValueError(f"ratio-Fourier-t sample table is missing columns: {sorted(missing)}")

    times = sorted({int(row[index["t"]]) for row in rows})
    x_values = sorted({float(row[index["x"]]) for row in rows})
    sample_maps: dict[int, dict[tuple[int, float], float]] = {}
    for row in rows:
        sample_id = int(row[index["sample_id"]])
        key = (int(row[index["t"]]), float(row[index["x"]]))
        sample_maps.setdefault(sample_id, {})[key] = float(row[index["q_sample"]])
    valid_ids = sorted(
        sample_id
        for sample_id, values in sample_maps.items()
        if all((t, x) in values and np.isfinite(values[(t, x)]) for t in times for x in x_values)
    )
    samples = np.asarray(
        [
            [[sample_maps[sample_id][(t, x)] for x in x_values] for t in times]
            for sample_id in valid_ids
        ],
        dtype=float,
    )
    if samples.size == 0:
        samples = np.empty((0, len(times), len(x_values)), dtype=float)
    return np.asarray(times, dtype=int), np.asarray(x_values, dtype=float), samples


def fit_x_component(
    q_samples: np.ndarray,
    times: np.ndarray,
    x_index: int,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    *,
    nt: int,
    pz: int,
    ns: int,
    gm: str,
    tmin: int,
    tmax: int,
) -> tuple[XFitResult, np.ndarray]:
    fit_mask = (times >= tmin) & (times <= tmax)
    fit_times = times[fit_mask]
    if fit_times.size == 0:
        raise ValueError(f"no q(x,t) time slices fall inside fit window {tmin}:{tmax}")
    data_samples = q_samples[:, fit_mask, x_index]
    sigma = np.nanstd(data_samples, axis=0, ddof=1)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, MIN_POSITIVE)
    amplitudes_by_sample = amplitudes if amplitudes.ndim == 2 else None
    energies_by_sample = energies if energies.ndim == 2 else None
    nstates = amplitudes.shape[-1]

    def residuals(params: np.ndarray, data: np.ndarray, fit_amplitudes: np.ndarray, fit_energies: np.ndarray) -> np.ndarray:
        model_values = evaluate_tmdwf_ratio(
            fit_times,
            fit_amplitudes,
            fit_energies,
            params,
            nt,
            gm=gm,
            pz=pz,
            ns=ns,
        )
        return (model_values - data) / sigma

    theta0 = np.zeros(nstates, dtype=float)
    sample_params = np.full((q_samples.shape[0], nstates), np.nan, dtype=float)
    chi2_values = np.full(q_samples.shape[0], np.nan, dtype=float)
    chi2_dof_values = np.full(q_samples.shape[0], np.nan, dtype=float)
    pvalues = np.full(q_samples.shape[0], np.nan, dtype=float)
    for sample_id, sample_data in enumerate(data_samples):
        fit_amplitudes = amplitudes_by_sample[sample_id] if amplitudes_by_sample is not None else amplitudes
        fit_energies = energies_by_sample[sample_id] if energies_by_sample is not None else energies
        result = least_squares(residuals, theta0, args=(sample_data, fit_amplitudes, fit_energies), max_nfev=5000)
        if result.success:
            params = np.asarray(result.x, dtype=float)
            sample_params[sample_id] = params
            chi2_value = float(np.dot(result.fun, result.fun))
            dof = max(fit_times.size - theta0.size, 1)
            chi2_values[sample_id] = chi2_value
            chi2_dof_values[sample_id] = chi2_value / dof
            pvalues[sample_id] = float(1.0 - chi2.cdf(chi2_value, dof))

    success_mask = np.all(np.isfinite(sample_params), axis=1)
    if np.any(success_mask):
        params = []
        for column in range(nstates):
            mean, _ = robust_mean_and_error(sample_params[success_mask, column])
            params.append(mean)
        chi2_dof_mean, _ = robust_mean_and_error(chi2_dof_values[success_mask])
        pvalue_mean, _ = robust_mean_and_error(pvalues[success_mask])
        return XFitResult(np.asarray(params, dtype=float), chi2_dof_mean, pvalue_mean, True), sample_params
    return XFitResult(np.full(nstates, np.nan), np.nan, np.nan, False), sample_params


def _write_xfit_outputs(
    output_root: Path,
    stem: str,
    *,
    x_values: np.ndarray,
    sample_params_by_x: list[np.ndarray],
    results: list[XFitResult],
    component: str,
    nstates: int,
    tmin: int,
    tmax: int,
) -> list[Path]:
    tables_dir = output_root / "tables"
    samples_dir = output_root / "samples"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / f"{stem}_{component}_{nstates}state_xfit.txt"
    sample_path = samples_dir / f"{stem}_{component}_{nstates}state_xfit_samples.txt"

    with table_path.open("w", encoding="utf-8") as handle:
        header = ["x", "tmin", "tmax", "success_bootstrap_center", "chi2_dof", "pvalue"]
        header += [f"q{idx}_mean" for idx in range(nstates)]
        header += [f"q{idx}_err" for idx in range(nstates)]
        handle.write("\t".join(header) + "\n")
        for x_value, result, sample_params in zip(x_values, results, sample_params_by_x):
            success_mask = np.all(np.isfinite(sample_params), axis=1)
            errors = []
            for column in range(nstates):
                _, err = robust_mean_and_error(sample_params[success_mask, column]) if np.any(success_mask) else (np.nan, np.nan)
                errors.append(err)
            row = [
                f"{x_value:.10e}",
                str(tmin),
                str(tmax),
                str(int(result.success)),
                f"{result.chi2_dof:.10e}",
                f"{result.pvalue:.10e}",
                *[f"{value:.10e}" for value in result.params],
                *[f"{value:.10e}" for value in errors],
            ]
            handle.write("\t".join(row) + "\n")

    with sample_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(["sample_id", "x", "success", *[f"q{idx}" for idx in range(nstates)]]) + "\n")
        for x_value, sample_params in zip(x_values, sample_params_by_x):
            for sample_id, params in enumerate(sample_params):
                success = int(np.all(np.isfinite(params)))
                values = params if success else np.full(nstates, np.nan)
                handle.write(
                    "\t".join(
                        [str(sample_id), f"{x_value:.10e}", str(success), *[f"{value:.10e}" for value in values]]
                    )
                    + "\n"
                )
    return [table_path, sample_path]


def _select_x_indices(x_values: np.ndarray, max_x_panels: int = DEFAULT_XFIT_PLOT_PANELS) -> np.ndarray:
    if max_x_panels <= 0 or x_values.size <= max_x_panels:
        return np.arange(x_values.size, dtype=int)
    return np.unique(np.linspace(0, x_values.size - 1, max_x_panels, dtype=int))


def _summarize_sample_axis(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p16 = np.percentile(values, 16.0, axis=0)
    p84 = np.percentile(values, 84.0, axis=0)
    center = 0.5 * (p16 + p84)
    err = 0.5 * (p84 - p16)
    return center, err, p16, p84


def _fit_curves_for_x(
    times: np.ndarray,
    sample_params: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    *,
    nt: int,
    pz: int,
    ns: int,
    gm: str,
) -> np.ndarray:
    valid_mask = np.all(np.isfinite(sample_params), axis=1)
    valid_params = sample_params[valid_mask]
    if valid_params.size == 0:
        return np.empty((0, times.size), dtype=float)
    if amplitudes.ndim == 2 and energies.ndim == 2:
        sample_indices = np.flatnonzero(valid_mask)
        return np.asarray(
            [
                np.real(evaluate_tmdwf_ratio(
                    times,
                    amplitudes[sample_id],
                    energies[sample_id],
                    params,
                    nt,
                    gm=gm,
                    pz=pz,
                    ns=ns,
                ))
                for sample_id, params in zip(sample_indices, valid_params)
            ],
            dtype=float,
        )
    return np.asarray(
        [
            np.real(evaluate_tmdwf_ratio(times, amplitudes, energies, params, nt, gm=gm, pz=pz, ns=ns))
            for params in valid_params
        ],
        dtype=float,
    )


def plot_tmdwf_x_nstate_fit_diagnostics(
    output_path: str | Path,
    *,
    q_samples: np.ndarray,
    times: np.ndarray,
    x_values: np.ndarray,
    sample_params_by_x: list[np.ndarray],
    amplitudes: np.ndarray,
    energies: np.ndarray,
    nt: int,
    pz: int,
    ns: int,
    gm: str,
    bT: int,
    component: str,
    nstates: int,
    tmin: int,
    tmax: int,
    max_x_panels: int = DEFAULT_XFIT_PLOT_PANELS,
    x_indices: tuple[int, ...] | list[int] | np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")
    if q_samples.shape[0] == 0:
        return save_plot_status(output_path.with_suffix(".txt"), "no q(x,t) samples available for plotting")

    selected = np.asarray(x_indices, dtype=int) if x_indices is not None else _select_x_indices(x_values, max_x_panels)
    selected = selected[(0 <= selected) & (selected < x_values.size)]
    if selected.size == 0:
        return save_plot_status(output_path.with_suffix(".txt"), "no x indices selected for plotting")

    ncols = min(3, selected.size)
    nrows = int(np.ceil(selected.size / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False, sharex=True)
    # The TMDWF time model is defined for integer times. Use integer support
    # points for reconstruction to avoid pretending to interpolate the model.
    curve_times = times
    for ax, x_index in zip(axes.ravel(), selected, strict=False):
        data_center, data_err, _, _ = _summarize_sample_axis(q_samples[:, :, x_index])
        curves = _fit_curves_for_x(
            curve_times,
            sample_params_by_x[int(x_index)],
            amplitudes,
            energies,
            nt=nt,
            pz=pz,
            ns=ns,
            gm=gm,
        )
        ax.errorbar(times, data_center, yerr=data_err, fmt="o", ms=3, color="C0", label="q(x,t)")
        if curves.shape[0] > 0:
            fit_center, _, fit_p16, fit_p84 = _summarize_sample_axis(curves)
            ax.plot(curve_times, fit_center, color="C1", linewidth=1.2, label="fit")
            ax.fill_between(curve_times, fit_p16, fit_p84, color="C1", alpha=0.18, linewidth=0)
        ax.axvspan(tmin, tmax, color="0.9", alpha=0.5, zorder=-1)
        ax.set_title(f"x={x_values[int(x_index)]:.3g}")
        ax.tick_params(direction="in", top=True, right=True)
    for ax in axes.ravel()[selected.size:]:
        ax.axis("off")
    if xlim is not None:
        for ax in axes[-1, :]:
            ax.set_xlim(xlim)
    if ylim is not None:
        for ax in axes.ravel()[:selected.size]:
            ax.set_ylim(ylim)
    for ax in axes[-1, :]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("q(x,t)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"pz={pz} bT={bT} {component} {nstates}state")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def write_tmdwf_x_nstate_fit_plot_notebook(
    notebook_path: str | Path,
    *,
    title_pattern: str,
    input_root: str | Path,
    xfit_root: str | Path,
    two_point_fit_root: str | Path,
    two_point_fit_window_by_pz: dict[int, tuple[int, int]],
    fit_window: dict,
    ns: int,
    nt: int,
    pzlist: tuple[int, ...],
    gmlist: tuple[str, ...],
    etalist: tuple[str, ...],
    bTlist: tuple[int, ...],
    component: str,
    nstates: int,
) -> Path:
    notebook_path = Path(notebook_path)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# TMDWF x-space N-state Fit Diagnostics\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import sys\n",
                    "REPO_ROOT = Path('/Users/xiang/Desktop/codes/lat-hadron-analysis')\n",
                    "SRC_DIR = REPO_ROOT / 'src'\n",
                    "if str(SRC_DIR) not in sys.path:\n",
                    "    sys.path.insert(0, str(SRC_DIR))\n",
                    "import numpy as np\n",
                    "from lqcd_analysis.tmdwf.io import expand_template, resolve_two_point_fit_reference\n",
                    "from lqcd_analysis.tmdwf.x_nstate_fit import (\n",
                    "    load_ratio_fourier_t_samples,\n",
                    "    plot_tmdwf_x_nstate_fit_diagnostics,\n",
                    "    resolve_ratio_fourier_t_sample_path,\n",
                    ")\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plot_config = {\n",
                    f"    'title_pattern': {title_pattern!r},\n",
                    f"    'input_root': {str(input_root)!r},\n",
                    f"    'xfit_root': {str(xfit_root)!r},\n",
                    f"    'two_point_fit_root': {str(two_point_fit_root)!r},\n",
                    f"    'two_point_fit_window_by_pz': {two_point_fit_window_by_pz!r},\n",
                    f"    'fit_window': {fit_window!r},\n",
                    f"    'ns': {ns!r},\n",
                    f"    'nt': {nt!r},\n",
                    f"    'pzlist': {tuple(pzlist)!r},\n",
                    f"    'gmlist': {tuple(gmlist)!r},\n",
                    f"    'etalist': {tuple(etalist)!r},\n",
                    f"    'bTlist': {tuple(bTlist)!r},\n",
                    f"    'component': {component!r},\n",
                    f"    'nstates': {nstates!r},\n",
                    "    'max_x_panels': 9,\n",
                    "    'x_indices': None,\n",
                    "    'xlim': None,\n",
                    "    'ylim': None,\n",
                    "}\n",
                    "plot_config\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fit_windows = plot_config['fit_window']\n",
                    "outputs = []\n",
                    "for pz in plot_config['pzlist']:\n",
                    "    title = expand_template(plot_config['title_pattern'], pz=pz)\n",
                    "    two_point_tmin, two_point_tmax = plot_config['two_point_fit_window_by_pz'][pz]\n",
                    "    fit_ref = resolve_two_point_fit_reference(\n",
                    "        plot_config['two_point_fit_root'], title=title, nstates=plot_config['nstates'],\n",
                    "        tmin=two_point_tmin, tmax=two_point_tmax,\n",
                    "    )\n",
                    "    for gm in plot_config['gmlist']:\n",
                    "        for eta in plot_config['etalist']:\n",
                    "            tmin, tmax = fit_windows.get((gm, pz), fit_windows.get((None, pz)))\n",
                    "            for bT in plot_config['bTlist']:\n",
                    "                sample_path = resolve_ratio_fourier_t_sample_path(\n",
                    "                    plot_config['input_root'], title=title, gm=gm, eta=eta, bT=bT,\n",
                    "                    component=plot_config['component'],\n",
                    "                )\n",
                    "                times, x_values, q_samples = load_ratio_fourier_t_samples(sample_path)\n",
                    "                # Reuse x-fit sample params written by the fit workflow.\n",
                    "                import pandas as pd\n",
                    "                xfit_sample_path = Path(plot_config['xfit_root']) / title / 'samples' / f\"{title}_{gm}_{eta}_bT{bT}_{plot_config['component']}_{plot_config['nstates']}state_xfit_samples.txt\"\n",
                    "                df = pd.read_csv(xfit_sample_path, sep='\\t')\n",
                    "                sample_params_by_x = []\n",
                    "                for x in x_values:\n",
                    "                    sub = df[df['x'].round(12) == round(float(x), 12)].sort_values('sample_id')\n",
                    "                    q_columns = [f'q{idx}' for idx in range(plot_config['nstates'])]\n",
                    "                    sample_params_by_x.append(sub[q_columns].to_numpy(float))\n",
                    "                plot_path = Path(plot_config['xfit_root']) / title / 'plots' / f\"{title}_{gm}_{eta}_bT{bT}_{plot_config['component']}_{plot_config['nstates']}state_xfit_diagnostics_custom.pdf\"\n",
                    "                outputs.append(plot_tmdwf_x_nstate_fit_diagnostics(\n",
                    "                    plot_path, q_samples=q_samples, times=times, x_values=x_values,\n",
                    "                    sample_params_by_x=sample_params_by_x,\n",
                    "                    amplitudes=fit_ref.amplitudes, energies=fit_ref.energies,\n",
                    "                    nt=plot_config['nt'], pz=pz, ns=plot_config['ns'], gm=gm, bT=bT,\n",
                    "                    component=plot_config['component'], nstates=plot_config['nstates'],\n",
                    "                    tmin=tmin, tmax=tmax,\n",
                    "                    max_x_panels=plot_config['max_x_panels'],\n",
                    "                    x_indices=plot_config['x_indices'],\n",
                    "                    xlim=plot_config.get('xlim'),\n",
                    "                    ylim=plot_config.get('ylim'),\n",
                    "                ))\n",
                    "outputs\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(__import__("json").dumps(notebook, indent=2), encoding="utf-8")
    return notebook_path


def run_tmdwf_x_nstate_fit_workflow(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_x_nstate_fit_input(input_file, results_dir=results_dir)
    fit_windows = load_fit_window_table(spec.fit_window)
    outputs: list[Path] = []

    for pz in spec.pzlist:
        title = expand_template(spec.title_pattern, pz=pz)
        two_point_window = spec.two_point_fit_window_by_pz.get(pz)
        if two_point_window is None:
            raise ValueError(f"missing two_point_fit_window_by_pz entry for pz={pz}")
        two_point_tmin, two_point_tmax = two_point_window
        dataset_root = spec.results_dir / title
        for gm in spec.gmlist:
            for eta in spec.etalist:
                fit_window = fit_windows.get((gm, pz), fit_windows.get((None, pz)))
                if fit_window is None:
                    raise ValueError(f"missing fit_window entry for gm={gm}, pz={pz}")
                fit_tmin, fit_tmax = fit_window
                for bT in spec.bTlist:
                    sample_path = resolve_ratio_fourier_t_sample_path(
                        spec.input_root,
                        title=title,
                        gm=gm,
                        eta=eta,
                        bT=bT,
                        component=spec.component,
                    )
                    times, x_values, q_samples = load_ratio_fourier_t_samples(sample_path)
                    for nstates in spec.nstates:
                        fit_reference = resolve_two_point_fit_reference(
                            spec.two_point_fit_root,
                            title=title,
                            nstates=nstates,
                            tmin=two_point_tmin,
                            tmax=two_point_tmax,
                        )
                        amplitudes: np.ndarray = fit_reference.amplitudes
                        energies: np.ndarray = fit_reference.energies
                        if spec.two_point_fit_sample_coupled:
                            amplitudes, energies = _load_two_point_sample_parameters(
                                _two_point_sample_table_path(fit_reference.path),
                                tmin=two_point_tmin,
                                nstates=nstates,
                                sample_count=q_samples.shape[0],
                                fallback_amplitudes=fit_reference.amplitudes,
                                fallback_energies=fit_reference.energies,
                            )
                        results: list[XFitResult] = []
                        sample_params_by_x: list[np.ndarray] = []
                        for x_index in range(x_values.size):
                            result, sample_params = fit_x_component(
                                q_samples,
                                times,
                                x_index,
                                amplitudes,
                                energies,
                                nt=spec.nt,
                                pz=pz,
                                ns=spec.ns,
                                gm=gm,
                                tmin=fit_tmin,
                                tmax=fit_tmax,
                            )
                            results.append(result)
                            sample_params_by_x.append(sample_params)
                        stem = f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_bT{bT}"
                        outputs.extend(
                            _write_xfit_outputs(
                                dataset_root,
                                stem,
                                x_values=x_values,
                                sample_params_by_x=sample_params_by_x,
                                results=results,
                                component=spec.component,
                                nstates=nstates,
                                tmin=fit_tmin,
                                tmax=fit_tmax,
                            )
                        )
                        plot_path = dataset_root / "plots" / f"{stem}_{spec.component}_{nstates}state_xfit_diagnostics.pdf"
                        outputs.append(
                            plot_tmdwf_x_nstate_fit_diagnostics(
                                plot_path,
                                q_samples=q_samples,
                                times=times,
                                x_values=x_values,
                                sample_params_by_x=sample_params_by_x,
                                amplitudes=amplitudes,
                                energies=energies,
                                nt=spec.nt,
                                pz=pz,
                                ns=spec.ns,
                                gm=gm,
                                bT=bT,
                                component=spec.component,
                                nstates=nstates,
                                tmin=fit_tmin,
                                tmax=fit_tmax,
                            )
                        )
                        outputs.append(
                            write_tmdwf_x_nstate_fit_plot_notebook(
                                dataset_root / "plots" / f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_{spec.component}_{nstates}state_xfit_plots.ipynb",
                                title_pattern=spec.title_pattern,
                                input_root=spec.input_root,
                                xfit_root=spec.results_dir,
                                two_point_fit_root=spec.two_point_fit_root,
                                two_point_fit_window_by_pz=spec.two_point_fit_window_by_pz,
                                fit_window=fit_windows,
                                ns=spec.ns,
                                nt=spec.nt,
                                pzlist=spec.pzlist,
                                gmlist=spec.gmlist,
                                etalist=spec.etalist,
                                bTlist=spec.bTlist,
                                component=spec.component,
                                nstates=nstates,
                            )
                        )
    return outputs
