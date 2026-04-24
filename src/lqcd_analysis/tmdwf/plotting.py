from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..two_point.plotting import prepare_matplotlib, save_plot_status


@dataclass(frozen=True)
class RatioFitPlotSeries:
    bz: int
    times: np.ndarray
    ratio_mean: np.ndarray
    ratio_err: np.ndarray
    fit_mean: np.ndarray
    fit_p16: np.ndarray
    fit_p84: np.ndarray


@dataclass(frozen=True)
class TMDWFM0VsBZSeries:
    bT: int
    bz: np.ndarray
    m0_mean: np.ndarray
    m0_err: np.ndarray


@dataclass(frozen=True)
class CSKernelBreakdownSeries:
    label: str
    x_values: np.ndarray
    log_ratio_p16: np.ndarray
    log_ratio_p50: np.ndarray
    log_ratio_p84: np.ndarray
    matching: np.ndarray
    total_p16: np.ndarray
    total_p50: np.ndarray
    total_p84: np.ndarray


def plot_tmdwf_cs_kernel_average_bT(
    output_path: str | Path,
    rows: tuple[object, ...] | list[object],
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (6.8, 4.5),
) -> Path:
    output_path = Path(output_path)
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")

    if not rows:
        return save_plot_status(output_path.with_suffix(".txt"), "no averaged CS-kernel rows available for plotting")

    bT = np.array([float(row.bT_fm) if hasattr(row, "bT_fm") else float(row.bT) for row in rows], dtype=float)
    value = np.array([float(row.value) for row in rows], dtype=float)
    stat_err = np.array([float(row.stat_err) for row in rows], dtype=float)
    sys_err = np.array([float(row.sys_err) for row in rows], dtype=float)
    total_err = np.array([float(row.total_err) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        bT,
        value,
        yerr=total_err,
        fmt="o",
        color="C0",
        ecolor="0.35",
        elinewidth=1.3,
        capsize=3,
        label="total error",
    )
    ax.errorbar(
        bT,
        value,
        yerr=stat_err,
        fmt="none",
        ecolor="C0",
        elinewidth=2.0,
        capsize=5,
        label="statistical error",
    )
    ax.set_xlabel(r"$b_T$ [fm]")
    ax.set_ylabel(r"CS-kernel average")
    if title:
        ax.set_title(title)
    ax.legend(title="error bars")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_tmdwf_joint_cs_kernel_x_band(
    output_path: str | Path,
    *,
    x_values: np.ndarray,
    band_p16: np.ndarray,
    band_p50: np.ndarray,
    band_p84: np.ndarray,
    bT_fm: float,
    title: str,
    kernel_label: str,
    spline_kind: str,
    figsize: tuple[float, float] = (6.8, 4.5),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Path:
    output_path = Path(output_path)
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(x_values, band_p16, band_p84, color="C0", alpha=0.22, linewidth=0)
    ax.plot(
        x_values,
        band_p50,
        color="C0",
        linewidth=1.3,
        label=f"{kernel_label} {spline_kind}",
    )
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\gamma_{\mathrm{eff}}(x,b_T)$")
    ax.set_title(f"{title} bT={bT_fm:.3f} fm")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _parse_grouped_table(path: str | Path) -> tuple[dict[str, str], list[str], list[list[str]]]:
    metadata: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                tokens = line.split("\t")
                if header is None:
                    header = tokens
                else:
                    rows.append(tokens)
            else:
                key, _, value = line.partition(" ")
                metadata[key] = value
    if header is None:
        raise ValueError(f"grouped table is missing a tab-delimited header: {path}")
    return metadata, header, rows


def _build_series_from_grouped_tables(
    ratio_table: str | Path,
    curve_table: str | Path,
    *,
    component: str,
    bz_values: tuple[int, ...] | None = None,
) -> tuple[tuple[RatioFitPlotSeries, ...], tuple[int, int]]:
    ratio_metadata, ratio_header, ratio_rows = _parse_grouped_table(ratio_table)
    _, curve_header, curve_rows = _parse_grouped_table(curve_table)
    ratio_index = {name: idx for idx, name in enumerate(ratio_header)}
    curve_index = {name: idx for idx, name in enumerate(curve_header)}
    fit_window_tokens = ratio_metadata.get("tfit", "0 0").split()
    fit_window = (int(fit_window_tokens[0]), int(fit_window_tokens[1]))

    available_bz = sorted({int(row[ratio_index["bz"]]) for row in ratio_rows})
    chosen_bz = available_bz if bz_values is None else [bz for bz in bz_values if bz in available_bz]
    ratio_mean_key = "ratio_real_mean" if component == "real" else "ratio_imag_mean"
    ratio_err_key = "ratio_real_err" if component == "real" else "ratio_imag_err"
    series: list[RatioFitPlotSeries] = []
    for bz in chosen_bz:
        ratio_subset = [row for row in ratio_rows if int(row[ratio_index["bz"]]) == bz]
        curve_subset = [row for row in curve_rows if int(row[curve_index["bz"]]) == bz]
        ratio_times = np.array([int(row[ratio_index["t"]]) for row in ratio_subset], dtype=int)
        curve_times = np.array([int(row[curve_index["t"]]) for row in curve_subset], dtype=int)
        if ratio_subset and curve_subset and not np.array_equal(ratio_times, curve_times):
            raise ValueError(f"ratio/curve time grids do not match for bz={bz}")
        series.append(
            RatioFitPlotSeries(
                bz=bz,
                times=ratio_times,
                ratio_mean=np.array([float(row[ratio_index[ratio_mean_key]]) for row in ratio_subset], dtype=float),
                ratio_err=np.array([float(row[ratio_index[ratio_err_key]]) for row in ratio_subset], dtype=float),
                fit_mean=np.array([float(row[curve_index["fit_mean"]]) for row in curve_subset], dtype=float),
                fit_p16=np.array([float(row[curve_index["fit_p16"]]) for row in curve_subset], dtype=float),
                fit_p84=np.array([float(row[curve_index["fit_p84"]]) for row in curve_subset], dtype=float),
            )
        )
    return tuple(series), fit_window


def _build_m0_series_from_fit_tables(
    fit_tables_by_bT: dict[int, str | Path],
) -> tuple[TMDWFM0VsBZSeries, ...]:
    series: list[TMDWFM0VsBZSeries] = []
    for bT in sorted(fit_tables_by_bT):
        _, header, rows = _parse_grouped_table(fit_tables_by_bT[bT])
        index = {name: idx for idx, name in enumerate(header)}
        required = {"bz", "m0_mean", "m0_err"}
        missing = required - index.keys()
        if missing:
            raise ValueError(f"grouped fit table for bT={bT} is missing columns: {sorted(missing)}")
        sorted_rows = sorted(rows, key=lambda row: int(row[index["bz"]]))
        series.append(
            TMDWFM0VsBZSeries(
                bT=bT,
                bz=np.array([int(row[index["bz"]]) for row in sorted_rows], dtype=int),
                m0_mean=np.array([float(row[index["m0_mean"]]) for row in sorted_rows], dtype=float),
                m0_err=np.array([float(row[index["m0_err"]]) for row in sorted_rows], dtype=float),
            )
        )
    return tuple(series)


def _finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _finite_nonzero_abs(values: np.ndarray) -> np.ndarray:
    array = np.abs(np.asarray(values, dtype=float))
    return array[np.isfinite(array) & (array > 0.0)]


def _choose_ratio_axis_scale(
    series: tuple[RatioFitPlotSeries, ...],
    *,
    component: str,
) -> tuple[str, dict[str, float]]:
    all_values: list[np.ndarray] = []
    for item in series:
        for values in (
            item.ratio_mean,
            item.ratio_mean - item.ratio_err,
            item.ratio_mean + item.ratio_err,
            item.fit_mean,
            item.fit_p16,
            item.fit_p84,
        ):
            all_values.append(_finite_values(values))
    finite_values = np.concatenate(all_values) if all_values else np.array([], dtype=float)
    if component == "real" and finite_values.size > 0 and np.all(finite_values > 0.0):
        return "log", {}

    nonzero_abs = np.concatenate([_finite_nonzero_abs(values) for values in all_values]) if all_values else np.array([], dtype=float)
    if nonzero_abs.size == 0:
        linthresh = 1.0
    else:
        linthresh = float(np.percentile(nonzero_abs, 20.0))
        linthresh = max(linthresh, float(np.min(nonzero_abs)), 1e-6)
    return "symlog", {"linthresh": linthresh}


def plot_tmdwf_ratio_fit(
    output_path: Path,
    series: tuple[RatioFitPlotSeries, ...],
    *,
    component: str,
    fit_window: tuple[int, int],
    figsize: tuple[float, float] = (7.0, 4.5),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_fit_window: bool = True,
    title: str | None = None,
) -> Path:
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")

    fig, ax = plt.subplots(figsize=figsize)
    tmin, tmax = fit_window
    if show_fit_window:
        ax.axvspan(tmin, tmax, color="0.92", zorder=0, label=f"fit window [{tmin}, {tmax}]")
    for index, item in enumerate(series):
        color = f"C{index}"
        ax.errorbar(
            item.times,
            item.ratio_mean,
            yerr=item.ratio_err,
            fmt="o",
            ms=4,
            capsize=2,
            color=color,
            label=f"bz={item.bz}",
        )
        ax.fill_between(item.times, item.fit_p16, item.fit_p84, color=color, alpha=0.18, linewidth=0)
        ax.plot(item.times, item.fit_mean, color=color, linewidth=1.1)
    scale_name, scale_kwargs = _choose_ratio_axis_scale(series, component=component)
    ax.set_yscale(scale_name, **scale_kwargs)
    ax.set_xlabel("t")
    ax.set_ylabel(f"ratio ({component})")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_tmdwf_grouped_outputs(
    output_path: str | Path,
    ratio_table: str | Path,
    curve_table: str | Path,
    *,
    component: str,
    bz_values: tuple[int, ...] | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_fit_window: bool = True,
    title: str | None = None,
) -> Path:
    output_path = Path(output_path)
    series, fit_window = _build_series_from_grouped_tables(
        ratio_table,
        curve_table,
        component=component,
        bz_values=bz_values,
    )
    return plot_tmdwf_ratio_fit(
        output_path,
        series,
        component=component,
        fit_window=fit_window,
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
        show_fit_window=show_fit_window,
        title=title,
    )


def plot_tmdwf_m0_vs_bz(
    output_path: str | Path,
    series: tuple[TMDWFM0VsBZSeries, ...],
    *,
    component: str,
    nstates: int,
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
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1.0, zorder=0)
    for index, item in enumerate(series):
        ax.errorbar(
            item.bz,
            item.m0_mean,
            yerr=item.m0_err,
            fmt="o-",
            ms=4,
            capsize=2,
            color=f"C{index}",
            label=f"bT={item.bT}",
        )
    ax.set_xlabel("bz")
    ax.set_ylabel("m0")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"m0 vs bz ({component}, {nstates}state)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_tmdwf_m0_from_fit_tables(
    output_path: str | Path,
    fit_tables_by_bT: dict[int, str | Path],
    *,
    component: str,
    nstates: int,
    title: str | None = None,
    figsize: tuple[float, float] = (6.5, 4.5),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Path:
    series = _build_m0_series_from_fit_tables(fit_tables_by_bT)
    return plot_tmdwf_m0_vs_bz(
        output_path,
        series,
        component=component,
        nstates=nstates,
        title=title,
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
    )


def plot_tmdwf_cs_kernel_band(
    output_path: str | Path,
    *,
    x_values: np.ndarray,
    band_p16: np.ndarray,
    band_p50: np.ndarray,
    band_p84: np.ndarray,
    title: str,
    scheme: str,
    kernel_label: str,
    bT: int,
    reference_pz: int,
    figsize: tuple[float, float] = (6.8, 4.5),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Path:
    output_path = Path(output_path)
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(x_values, band_p16, band_p84, color="C0", alpha=0.2, linewidth=0)
    ax.plot(x_values, band_p50, color="C0", linewidth=1.2, label=f"{scheme} {kernel_label}")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\gamma_\zeta$")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(f"{title} bT={bT} ref pz={reference_pz}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_tmdwf_cs_kernel_adjacent_breakdown(
    output_path: str | Path,
    series: tuple[CSKernelBreakdownSeries, ...],
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8.5, 12.0),
) -> Path:
    output_path = Path(output_path)
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ax_data, ax_match, ax_total = axes
    for index, item in enumerate(series):
        color = f"C{index}"
        ax_data.plot(item.x_values, item.log_ratio_p50, "--", lw=2.0, color=color, label=item.label)
        ax_data.fill_between(item.x_values, item.log_ratio_p16, item.log_ratio_p84, color=color, alpha=0.18, linewidth=0)
        ax_match.plot(item.x_values, item.matching, "--", lw=2.0, color=color, label=item.label)
        ax_total.plot(item.x_values, item.total_p50, "--", lw=2.0, color=color, label=item.label)
        ax_total.fill_between(item.x_values, item.total_p16, item.total_p84, color=color, alpha=0.18, linewidth=0)

    ax_data.set_ylabel("data log-ratio")
    ax_match.set_ylabel("matching")
    ax_total.set_ylabel("total estimator")
    ax_total.set_xlabel("x")
    if title:
        ax_data.set_title(title)
    for ax in axes:
        ax.axvline(0.5, color="0.85", linewidth=1.0)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def write_tmdwf_plot_notebook(
    notebook_path: str | Path,
    notebook_output_dir: str | Path,
    ratio_tables: dict[int, str | Path],
    curve_tables: dict[int, dict[str, dict[int, str | Path]]],
    fit_tables: dict[int, dict[str, dict[int, str | Path]]],
    sample_tables: dict[int, dict[str, dict[int, str | Path]]],
    title: str,
    gm: str,
    eta: str,
    pz: int,
    ns: int,
    lattice_spacing_fm: float,
) -> Path:
    notebook_path = Path(notebook_path)
    notebook_output_dir = Path(notebook_output_dir)
    ratio_tables_repr = {str(key): str(Path(value)) for key, value in ratio_tables.items()}
    curve_tables_repr = {
        str(bT): {
            component: {str(key): str(Path(value)) for key, value in table_map.items()}
            for component, table_map in component_map.items()
        }
        for bT, component_map in curve_tables.items()
    }
    fit_tables_repr = {
        str(bT): {
            component: {str(key): str(Path(value)) for key, value in table_map.items()}
            for component, table_map in component_map.items()
        }
        for bT, component_map in fit_tables.items()
    }
    sample_tables_repr = {
        str(bT): {
            component: {str(key): str(Path(value)) for key, value in table_map.items()}
            for component, table_map in component_map.items()
        }
        for bT, component_map in sample_tables.items()
    }
    repo_src = Path(__file__).resolve().parents[1]
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# TMDWF Plot Notebook\n",
                    "This notebook redraws grouped post-fit TMDWF plots from existing output tables.\n",
                    "Use the ratio section to tune one chosen bT, and the m0-vs-bz section to compare multiple bT values without rerunning the fit.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import sys\n",
                    "import numpy as np\n",
                    f"sys.path.insert(0, {str(repo_src)!r})\n",
                    "from lqcd_analysis.tmdwf.plotting import plot_tmdwf_grouped_outputs, plot_tmdwf_m0_from_fit_tables\n",
                    "from lqcd_analysis.tmdwf.fourier import run_tmdwf_fourier_from_fit_outputs\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"title = {title!r}\n",
                    f"gm = {gm!r}\n",
                    f"eta = {eta!r}\n",
                    f"pz = {pz}\n",
                    f"ns = {ns}\n",
                    f"lattice_spacing_fm = {lattice_spacing_fm}\n",
                    f"ratio_tables = {json.dumps(ratio_tables_repr, indent=2)}\n",
                    f"curve_tables = {json.dumps(curve_tables_repr, indent=2)}\n",
                    f"fit_tables = {json.dumps(fit_tables_repr, indent=2)}\n",
                    f"sample_tables = {json.dumps(sample_tables_repr, indent=2)}\n",
                    f"output_dir = Path({str(notebook_output_dir)!r})\n",
                    "available_bT = sorted(int(key) for key in ratio_tables)\n",
                    "chosen_bT = available_bT[0]\n",
                    "component = 'real'\n",
                    "nstates = max(int(key) for key in curve_tables[str(chosen_bT)][component])\n",
                    "bz_values = None  # set to e.g. (0, 4, 8, 12, 16, 20) to draw a subset for the ratio plot\n",
                    "ratio_figsize = (7.0, 4.5)\n",
                    "ratio_xlim = None\n",
                    "ratio_ylim = None\n",
                    "show_fit_window = True\n",
                    "ratio_output_name = None  # e.g. 'custom_ratio_plot.pdf'\n",
                    "ratio_alternate_output_dir = None\n",
                    "selected_bT_values = tuple(value for value in available_bT if value % 2 == 0)  # even bT only for the m0-vs-bz plot\n",
                    "m0_figsize = (6.5, 4.5)\n",
                    "m0_xlim = None\n",
                    "m0_ylim = None\n",
                    "m0_output_name = None  # e.g. 'custom_m0_vs_bz.pdf'\n",
                    "m0_alternate_output_dir = None\n",
                    "x_values = np.linspace(-0.5, 1.5, 201)\n",
                    "zstep_fm = 0.01\n",
                    "interpolation_kind = 'cubic'\n",
                    "fourier_output_name = None  # e.g. 'custom_fourier.pdf'\n",
                    "fourier_alternate_output_dir = None\n",
                    "fourier_xlim = None\n",
                    "fourier_ylim = None\n",
                    "output_dir.mkdir(parents=True, exist_ok=True)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Quick Tuning\n",
                    "Pick `chosen_bT`, `component`, and `nstates` for the ratio plot, then adjust limits or output paths.\n",
                    "For m0-vs-bz, keep the same `component` and `nstates`, and optionally adjust `selected_bT_values` before rerunning that section.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Available bT:', available_bT)\n",
                    "print('Available components by bT:', {key: sorted(value) for key, value in curve_tables.items()})\n",
                    "print('Available nstates by bT/component:', {key: {comp: sorted(int(item) for item in value[comp]) for comp in value} for key, value in curve_tables.items()})\n",
                    "chosen_bT, component, nstates\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "ratio_output_dir = output_dir if ratio_alternate_output_dir is None else Path(ratio_alternate_output_dir)\n",
                    "ratio_output_dir.mkdir(parents=True, exist_ok=True)\n",
                    "default_ratio_name = f'{title}_{gm}_{eta}_bT{chosen_bT}_{component}_{nstates}state_ratio_fit.pdf'\n",
                    "ratio_output_path = ratio_output_dir / (default_ratio_name if ratio_output_name is None else ratio_output_name)\n",
                    "ratio_table = Path(ratio_tables[str(chosen_bT)])\n",
                    "curve_table = Path(curve_tables[str(chosen_bT)][component][str(nstates)])\n",
                    "plot_tmdwf_grouped_outputs(\n",
                    "    output_path=ratio_output_path,\n",
                    "    ratio_table=ratio_table,\n",
                    "    curve_table=curve_table,\n",
                    "    component=component,\n",
                    "    bz_values=bz_values,\n",
                    "    figsize=ratio_figsize,\n",
                    "    xlim=ratio_xlim,\n",
                    "    ylim=ratio_ylim,\n",
                    "    show_fit_window=show_fit_window,\n",
                    "    title=f'{title} {gm} {eta} bT{chosen_bT} {component} {nstates}state',\n",
                    ")\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## m0 vs bz\n",
                    "This section overlays one bT series per grouped fit table, using the existing `m0_mean` and `m0_err` columns from the fit tables.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "m0_output_dir = output_dir if m0_alternate_output_dir is None else Path(m0_alternate_output_dir)\n",
                    "m0_output_dir.mkdir(parents=True, exist_ok=True)\n",
                    "default_m0_name = f'{title}_{gm}_{eta}_{component}_{nstates}state_m0_vs_bz.pdf'\n",
                    "m0_output_path = m0_output_dir / (default_m0_name if m0_output_name is None else m0_output_name)\n",
                    "target_bT_values = available_bT if selected_bT_values is None else [int(value) for value in selected_bT_values if int(value) in available_bT]\n",
                    "fit_tables_by_bT = {bT: Path(fit_tables[str(bT)][component][str(nstates)]) for bT in target_bT_values}\n",
                    "plot_tmdwf_m0_from_fit_tables(\n",
                    "    output_path=m0_output_path,\n",
                    "    fit_tables_by_bT=fit_tables_by_bT,\n",
                    "    component=component,\n",
                    "    nstates=nstates,\n",
                    "    title=f'{title} {gm} {eta} {component} {nstates}state m0 vs bz',\n",
                    "    figsize=m0_figsize,\n",
                    "    xlim=m0_xlim,\n",
                    "    ylim=m0_ylim,\n",
                    ")\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Fourier Transform\n",
                    "This section recomputes the post-fit cosine transform from the existing grouped fit/sample tables only, so you can tune `x_values`, `zstep_fm`, and `interpolation_kind` without rerunning the TMDWF fit.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fourier_output_dir = output_dir if fourier_alternate_output_dir is None else Path(fourier_alternate_output_dir)\n",
                    "fourier_output_dir.mkdir(parents=True, exist_ok=True)\n",
                    "default_fourier_name = f'{title}_{gm}_{eta}_bT{chosen_bT}_{component}_{nstates}state_fourier.pdf'\n",
                    "fourier_plot_name = default_fourier_name if fourier_output_name is None else fourier_output_name\n",
                    "fit_table = Path(fit_tables[str(chosen_bT)][component][str(nstates)])\n",
                    "sample_table = Path(sample_tables[str(chosen_bT)][component][str(nstates)])\n",
                    "fourier_outputs = run_tmdwf_fourier_from_fit_outputs(\n",
                    "    output_root=fourier_output_dir,\n",
                    "    stem=f'{title}_{gm}_{eta}_bT{chosen_bT}',\n",
                    "    fit_table=fit_table,\n",
                    "    sample_table=sample_table,\n",
                    "    pz=pz,\n",
                    "    ns=ns,\n",
                    "    lattice_spacing_fm=lattice_spacing_fm,\n",
                    "    bT=chosen_bT,\n",
                    "    component=component,\n",
                    "    nstates=nstates,\n",
                    "    x_values=x_values,\n",
                    "    zstep_fm=zstep_fm,\n",
                    "    interpolation_kind=interpolation_kind,\n",
                    "    make_plots=True,\n",
                    "    plot_xlim=fourier_xlim,\n",
                    "    plot_ylim=fourier_ylim,\n",
                    ")\n",
                    "if fourier_output_name is not None:\n",
                    "    generated_plot = fourier_output_dir / 'plots' / f'{title}_{gm}_{eta}_bT{chosen_bT}_{component}_{nstates}state_fourier.pdf'\n",
                    "    target_plot = fourier_output_dir / fourier_plot_name\n",
                    "    generated_plot.replace(target_plot)\n",
                    "fourier_outputs\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return notebook_path
