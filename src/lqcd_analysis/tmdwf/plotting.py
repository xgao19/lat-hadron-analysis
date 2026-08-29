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


def _safe_tight_layout(fig, **kwargs) -> None:
    try:
        fig.tight_layout(**kwargs)
    except OverflowError:
        pass


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
    _safe_tight_layout(fig)
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
    _safe_tight_layout(fig)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_tmdwf_joint_cs_kernel_pz_diagnostics(
    output_dir: str | Path,
    groups: list[object],
    *,
    stem: str,
    x_actual: float,
    max_cols: int = 4,
    figsize_per_panel: tuple[float, float] = (3.8, 2.8),
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = prepare_matplotlib()
    if plt is None:
        return [save_plot_status(
            output_dir / f"{stem}_x{x_actual:.3f}_diagnostics.txt",
            "matplotlib not installed; plot was skipped",
        )]

    if not groups:
        return [save_plot_status(
            output_dir / f"{stem}_x{x_actual:.3f}_diagnostics.txt",
            "no groups with sufficient pz points for diagnostics",
        )]

    # Group by ensemble label
    ensemble_groups: dict[str, list[object]] = {}
    for g in groups:
        ensemble_groups.setdefault(g.ensemble_label, []).append(g)

    outputs: list[Path] = []
    x_token = f"{x_actual:.3f}".replace(".", "p")

    for ens_label, ens_groups in ensemble_groups.items():
        n_panels = len(ens_groups)
        n_cols = min(max_cols, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        figsize = (figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        # Collect y-range for shared scale
        all_y = np.concatenate(
            [
                np.concatenate(
                    [g.data_median, g.data_p16, g.data_p84,
                     g.model_median, g.model_p16, g.model_p84]
                )
                for g in ens_groups
            ]
        )
        y_min, y_max = np.percentile(all_y, [1.0, 99.0])
        y_pad = max(0.05 * (y_max - y_min), 1e-6)
        y_lim = (y_min - y_pad, y_max + y_pad)

        for idx, g in enumerate(ens_groups):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            ax.errorbar(
                g.pz_gev,
                g.data_median,
                yerr=[g.data_median - g.data_p16, g.data_p84 - g.data_median],
                fmt="o",
                ms=4,
                capsize=2,
                color="C0",
                label="data",
            )
            ax.fill_between(
                g.pz_gev,
                g.model_p16,
                g.model_p84,
                color="C1",
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(g.pz_gev, g.model_median, color="C1", linewidth=1.2, label="fit")
            ax.set_title(f"bT={g.bT_fm:.3f} fm", fontsize=8)
            ax.set_ylim(*y_lim)
            ax.tick_params(labelsize=7)

        # Hide unused panels
        for idx in range(n_panels, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        # Shared axis labels on figure
        fig.supxlabel("pz [GeV]", fontsize=9)
        fig.supylabel("O(x, bT, pz)", fontsize=9)
        fig.suptitle(
            f"{ens_label}  x={x_actual:.3f}  pz-diagnostics",
            fontsize=10,
        )
        _safe_tight_layout(fig, rect=[0.0, 0.0, 1.0, 0.97])

        plot_path = output_dir / f"{stem}_x{x_token}_{ens_label}_pz_diagnostics.pdf"
        fig.savefig(plot_path)
        plt.close(fig)
        outputs.append(plot_path)

    return outputs


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
    _safe_tight_layout(fig)
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
    _safe_tight_layout(fig)
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
    _safe_tight_layout(fig)
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
    _safe_tight_layout(fig)
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
    repo_src = Path(__file__).resolve().parents[2]
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


def write_tmdwf_cs_kernel_joint_diagnostics_notebook(
    notebook_path: str | Path,
    *,
    summary_path: str | Path,
    coefficients_path: str | Path,
    results_dir: str | Path,
) -> Path:
    notebook_path = Path(notebook_path)
    results_dir = Path(results_dir)
    repo_src = Path(__file__).resolve().parents[2]
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# TMDWF Joint CS-Kernel Diagnostics\n",
                    "\n",
                    "Redraw pz-diagnostics plots from saved joint-fit outputs.\n",
                    "Adjust `x_selection`, `bT_selection_fm`, and figure settings below, then run all cells.\n",
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
                    "from lqcd_analysis.tmdwf.cs_kernel_joint import (\n",
                    "    parse_joint_summary,\n",
                    "    load_joint_coefficients_table,\n",
                    "    _preload_datasets,\n",
                    "    _build_observations_at_x,\n",
                    "    _build_diagnostic_groups,\n",
                    "    _evaluate_bT_surface,\n",
                    "    _spline_basis,\n",
                    "    TMDWFCSKernelJointInput,\n",
                    "    JointCSEnsembleInput,\n",
                    ")\n",
                    "from lqcd_analysis.tmdwf.plotting import (\n",
                    "    plot_tmdwf_joint_cs_kernel_pz_diagnostics,\n",
                    "    prepare_matplotlib,\n",
                    ")\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"summary_path = Path({str(summary_path)!r})\n",
                    f"coefficients_path = Path({str(coefficients_path)!r})\n",
                    f"output_dir = Path({str(results_dir)!r}) / 'joint_gamma_eff' / 'plots' / 'diagnostics'\n",
                    "output_dir.mkdir(parents=True, exist_ok=True)\n",
                    "output_dir",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Parse saved results\n",
                    "cfg = parse_joint_summary(summary_path)\n",
                    "coeffs_by_x = load_joint_coefficients_table(coefficients_path)\n",
                    "n_knots = cfg.get('n_gamma_knots', len(cfg['bT_knots_fm']))\n",
                    "\n",
                    "# Load correction coefficients if enabled\n",
                    "stem = coefficients_path.name.replace('_coefficients.txt', '')\n",
                    "samples_dir = coefficients_path.parent\n",
                    "alpha_by_x = {}\n",
                    "beta_by_x = {}\n",
                    "pz1_by_x = {}\n",
                    "kappa_by_x = {}\n",
                    "if cfg.get('fit_a2_correction', False):\n",
                    "    alpha_path = samples_dir / f'{stem}_coefficients_alpha.txt'\n",
                    "    if alpha_path.exists():\n",
                    "        alpha_by_x = load_joint_coefficients_table(alpha_path)\n",
                    "if cfg.get('fit_fv_correction', False):\n",
                    "    beta_path = samples_dir / f'{stem}_coefficients_beta.txt'\n",
                    "    if beta_path.exists():\n",
                    "        beta_by_x = load_joint_coefficients_table(beta_path)\n",
                    "if cfg.get('fit_pz1_correction', False):\n",
                    "    pz1_path = samples_dir / f'{stem}_coefficients_pz1.txt'\n",
                    "    if pz1_path.exists():\n",
                    "        pz1_by_x = load_joint_coefficients_table(pz1_path)\n",
                    "if cfg.get('fit_pz2_correction', False):\n",
                    "    kappa_path = samples_dir / f'{stem}_coefficients_kappa.txt'\n",
                    "    if kappa_path.exists():\n",
                    "        kappa_by_x = load_joint_coefficients_table(kappa_path)\n",
                    "lambda_by_x = {}\n",
                    "if cfg.get('fit_apz2_correction', False):\n",
                    "    lambda_path = samples_dir / f'{stem}_coefficients_lambda.txt'\n",
                    "    if lambda_path.exists():\n",
                    "        lambda_by_x = load_joint_coefficients_table(lambda_path)\n",
                    "\n",
                    "# Rebuild ensemble spec for data loading\n",
                    "ensembles = tuple(\n",
                    "    JointCSEnsembleInput(\n",
                    "        label=e['label'],\n",
                    "        input_root=Path(e['input_root']),\n",
                    "        title_pattern=e['title_pattern'],\n",
                    "        ns=e['ns'],\n",
                    "        lattice_spacing_fm=e['lattice_spacing_fm'],\n",
                    "        pzlist=tuple(e['pzlist']),\n",
                    "        bTlist=tuple(e['bTlist']),\n",
                    "        m_pi_mev=e.get('m_pi_mev', 140.0),\n",
                    "    )\n",
                    "    for e in cfg['ensembles']\n",
                    ")\n",
                    "spec = TMDWFCSKernelJointInput(\n",
                    "    ensembles=ensembles,\n",
                    "    gm=cfg['gm'],\n",
                    "    eta=cfg['eta'],\n",
                    "    component=cfg['component'],\n",
                    "    nstates=cfg['nstates'],\n",
                    "    normalization_mode=cfg['normalization_mode'],\n",
                    "    mu=cfg['mu'],\n",
                    "    scheme=cfg['scheme'],\n",
                    "    kernel_label=cfg['kernel_label'],\n",
                    "    reference_p1_gev=cfg['reference_p1_gev'],\n",
                    "    x_window=cfg['x_window'],\n",
                    "    x_knots=cfg['x_fit_points'],\n",
                    "    bT_knots_fm=cfg['bT_knots_fm'],\n",
                    "    spline_kind=cfg['spline_kind'],\n",
                    "    make_plots=False,\n",
                    "    show_progress=True,\n",
                    "    progress_every=None,\n",
                    "    results_dir=output_dir.parent.parent,\n",
                    "    fit_a2_correction=cfg.get('fit_a2_correction', False),\n",
                    "    fit_fv_correction=cfg.get('fit_fv_correction', False),\n",
                    "    fit_pz1_correction=cfg.get('fit_pz1_correction', False),\n",
                    "    fit_pz2_correction=cfg.get('fit_pz2_correction', False),\n",
                    "    fit_apz2_correction=cfg.get('fit_apz2_correction', False),\n",
                    ")\n",
                    "cfg, spec",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Preload Fourier datasets (may take a moment)\n",
                    "datasets, x_grid, n_samples = _preload_datasets(spec)\n",
                    "f'Loaded {len(datasets)} ensembles, x_grid shape={x_grid.shape}, {n_samples} bootstrap samples'",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Plot Settings\n",
                    "\n",
                    "Edit this cell to control which x and bT values are plotted.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# -- x selection --\n",
                    "# 'all' or a list of x values (matched to nearest data-grid point)\n",
                    "x_selection = 'all'  # e.g. [0.2, 0.4, 0.6]\n",
                    "\n",
                    "# -- bT selection (physical fm) --\n",
                    "# None = all available; or a list of bT values in fm\n",
                    "# When set, only groups whose physical bT is within match_tolerance\n",
                    "# of a listed value are plotted.\n",
                    "bT_selection_fm = None  # e.g. [0.12, 0.24, 0.48]\n",
                    "match_tolerance_fm = 0.005  # tolerance for matching physical bT\n",
                    "\n",
                    "# -- plot every (ensemble, bT) group regardless of pz count --\n",
                    "\n",
                    "# -- figure layout --\n",
                    "max_cols = 4\n",
                    "figsize_per_panel = (3.8, 2.8)",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Resolve x values to plot\n",
                    "if x_selection == 'all':\n",
                    "    x_list = list(cfg['x_fit_points'])\n",
                    "else:\n",
                    "    x_list = [float(x_grid[np.argmin(np.abs(x_grid - xv))]) for xv in x_selection]\n",
                    "x_list",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Build diagnostic groups for each x\n",
                    "from lqcd_analysis.tmdwf.cs_kernel_joint import _find_x_indices\n",
                    "\n",
                    "all_groups: dict[float, list] = {}  # x_actual -> list of DiagnosticGroupData\n",
                    "for x_val in x_list:\n",
                    "    x_matches = _find_x_indices(np.asarray([x_val]), x_grid)\n",
                    "    if not x_matches:\n",
                    "        print(f'x={x_val:.3f}: not on data grid, skipping')\n",
                    "        continue\n",
                    "    x_idx, x_actual = x_matches[0]\n",
                    "    if x_actual not in coeffs_by_x:\n",
                    "        print(f'x={x_actual:.3f}: no coefficients found, skipping')\n",
                    "        continue\n",
                    "\n",
                    "    obs = _build_observations_at_x(datasets, x_idx, cfg['x_window'], n_samples)\n",
                    "    if not obs:\n",
                    "        print(f'x={x_actual:.3f}: no observations in x_window, skipping')\n",
                    "        continue\n",
                    "\n",
                    "    # Build PerXFitResult from saved coefficients\n",
                    "    coeffs = coeffs_by_x[x_actual]\n",
                    "    n_samples = coeffs.shape[0]\n",
                    "    n_obs = len({o.group_id for o in obs})\n",
                    "    fake_chi2 = np.zeros(n_samples, dtype=float)\n",
                    "    from lqcd_analysis.tmdwf.cs_kernel_joint import PerXFitResult\n",
                    "\n",
                    "    # Assemble full coefficient matrix: [gamma, alpha?, beta?, pz1?, kappa?, lambda?]\n",
                    "    blocks = [coeffs]\n",
                    "    fit_a2 = cfg.get('fit_a2_correction', False)\n",
                    "    fit_fv = cfg.get('fit_fv_correction', False)\n",
                    "    fit_pz1 = cfg.get('fit_pz1_correction', False)\n",
                    "    fit_pz2 = cfg.get('fit_pz2_correction', False)\n",
                    "    fit_apz2 = cfg.get('fit_apz2_correction', False)\n",
                    "    if fit_a2 and x_actual in alpha_by_x:\n",
                    "        blocks.append(alpha_by_x[x_actual])\n",
                    "    if fit_fv and x_actual in beta_by_x:\n",
                    "        blocks.append(beta_by_x[x_actual])\n",
                    "    if fit_pz1 and x_actual in pz1_by_x:\n",
                    "        blocks.append(pz1_by_x[x_actual])\n",
                    "    if fit_pz2 and x_actual in kappa_by_x:\n",
                    "        blocks.append(kappa_by_x[x_actual])\n",
                    "    if fit_apz2 and x_actual in lambda_by_x:\n",
                    "        blocks.append(lambda_by_x[x_actual])\n",
                    "    full_coeffs = np.column_stack(blocks) if len(blocks) > 1 else blocks[0]\n",
                    "\n",
                    "    result = PerXFitResult(\n",
                    "        x_actual=x_actual,\n",
                    "        bT_knots_fm=cfg['bT_knots_fm'],\n",
                    "        coeff_samples=full_coeffs,\n",
                    "        chi2_dof=fake_chi2,\n",
                    "        n_observations=len(obs),\n",
                    "        n_groups=n_obs,\n",
                    "        n_gamma_knots=n_knots,\n",
                    "        fit_a2_correction=fit_a2,\n",
                    "        fit_fv_correction=fit_fv,\n",
                    "        fit_pz1_correction=fit_pz1,\n",
                    "        fit_pz2_correction=fit_pz2,\n",
                    "        fit_apz2_correction=fit_apz2,\n",
                    "    )\n",
                    "\n",
                    "    groups = _build_diagnostic_groups(\n",
                    "        obs,\n",
                    "        result,\n",
                    "        sample_count=n_samples,\n",
                    "        reference_p1_gev=cfg['reference_p1_gev'],\n",
                    "        scheme=cfg['scheme'],\n",
                    "        kernel_label=cfg['kernel_label'],\n",
                    "        mu=cfg['mu'],\n",
                    "        component=cfg['component'],\n",
                    "        spline_kind=cfg['spline_kind'],\n",
                    "    )\n",
                    "    print(f'x={x_actual:.3f}: {len(groups)} groups')\n",
                    "    all_groups[x_actual] = groups\n",
                    "\n",
                    "f'Built diagnostic data for {len(all_groups)} x-points'",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Filter groups by bT selection\n",
                    "def _filter_by_bT(groups, bT_list, tol):\n",
                    "    if bT_list is None:\n",
                    "        return groups\n",
                    "    filtered = []\n",
                    "    for g in groups:\n",
                    "        for target in bT_list:\n",
                    "            if abs(g.bT_fm - target) <= tol:\n",
                    "                filtered.append(g)\n",
                    "                break\n",
                    "    return filtered\n",
                    "\n",
                    "for x_val, groups in all_groups.items():\n",
                    "    all_groups[x_val] = _filter_by_bT(groups, bT_selection_fm, match_tolerance_fm)\n",
                    "    print(f'x={x_val:.3f}: {len(all_groups[x_val])} groups after bT filter')",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Per-Ensemble Diagnostic Plots\n",
                    "\n",
                    "One multi-panel PDF per x per ensemble. Data points and reconstructed fit band\n",
                    "for each `(ensemble, bT)` group with sufficient pz values.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "stem = f\"joint_{cfg['gm']}_{cfg['eta']}_{cfg['normalization_mode']}_{cfg['component']}_\" \\\n",
                    "       f\"{cfg['nstates']}state_{cfg['scheme']}_{cfg['kernel_label']}_gamma_eff\"\n",
                    "\n",
                    "plot_paths = []\n",
                    "for x_val, groups in all_groups.items():\n",
                    "    paths = plot_tmdwf_joint_cs_kernel_pz_diagnostics(\n",
                    "        output_dir,\n",
                    "        groups,\n",
                    "        stem=stem,\n",
                    "        x_actual=x_val,\n",
                    "        max_cols=max_cols,\n",
                    "        figsize_per_panel=figsize_per_panel,\n",
                    "    )\n",
                    "    plot_paths.extend(paths)\n",
                    "plot_paths",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Cross-Ensemble bT Comparison\n",
                    "\n",
                    "For a user-specified list of physical bT values, overlay data and fit from all\n",
                    "ensembles at matching bT on the same axes.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# bT values (fm) to compare across ensembles\n",
                    "compare_bT_fm = [0.12, 0.24, 0.48, 0.72]  # edit as needed\n",
                    "compare_tolerance_fm = 0.008\n",
                    "compare_max_cols = 3\n",
                    "compare_panel_size = (4.0, 3.0)\n",
                    "\n",
                    "plt = prepare_matplotlib()\n",
                    "if plt is not None:\n",
                    "    for x_val, groups in all_groups.items():\n",
                    "        # Collect matching groups for each target bT\n",
                    "        bT_groups: dict[float, list] = {}\n",
                    "        for g in groups:\n",
                    "            for target in compare_bT_fm:\n",
                    "                if abs(g.bT_fm - target) <= compare_tolerance_fm:\n",
                    "                    bT_groups.setdefault(target, []).append(g)\n",
                    "                    break\n",
                    "\n",
                    "        if not bT_groups:\n",
                    "            continue\n",
                    "\n",
                    "        n_panels = len(bT_groups)\n",
                    "        n_cols = min(compare_max_cols, n_panels)\n",
                    "        n_rows = int(np.ceil(n_panels / n_cols))\n",
                    "        figsize = (compare_panel_size[0] * n_cols, compare_panel_size[1] * n_rows)\n",
                    "        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)\n",
                    "\n",
                    "        x_token = f\"{x_val:.3f}\".replace('.', 'p')\n",
                    "        for idx, (target_bT, bT_grp) in enumerate(sorted(bT_groups.items())):\n",
                    "            row, col = divmod(idx, n_cols)\n",
                    "            ax = axes[row, col]\n",
                    "            for ens_idx, g in enumerate(bT_grp):\n",
                    "                color = f'C{ens_idx}'\n",
                    "                label = f\"{g.ensemble_label} (bT={g.bT_fm:.3f})\"\n",
                    "                ax.errorbar(\n",
                    "                    g.pz_gev, g.data_median,\n",
                    "                    yerr=[g.data_median - g.data_p16, g.data_p84 - g.data_median],\n",
                    "                    fmt='o', ms=4, capsize=2, color=color, label=label,\n",
                    "                )\n",
                    "                ax.fill_between(\n",
                    "                    g.pz_gev, g.model_p16, g.model_p84,\n",
                    "                    color=color, alpha=0.18, linewidth=0,\n",
                    "                )\n",
                    "                ax.plot(g.pz_gev, g.model_median, color=color, linewidth=1.2)\n",
                    "            ax.set_title(f'bT={target_bT:.3f} fm', fontsize=9)\n",
                    "            ax.set_xlabel('pz [GeV]', fontsize=8)\n",
                    "            ax.set_ylabel('O(x, bT, pz)', fontsize=8)\n",
                    "            ax.legend(fontsize=7)\n",
                    "            ax.tick_params(labelsize=7)\n",
                    "\n",
                    "        for idx in range(n_panels, n_rows * n_cols):\n",
                    "            row, col = divmod(idx, n_cols)\n",
                    "            axes[row, col].set_visible(False)\n",
                    "\n",
                    "        fig.suptitle(f'Cross-ensemble comparison  x={x_val:.3f}', fontsize=11)\n",
                    "        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])\n",
                    "        comp_path = output_dir / f'{stem}_x{x_token}_cross_ensemble.pdf'\n",
                    "        fig.savefig(comp_path)\n",
                    "        plt.close(fig)\n",
                    "        print(f'Saved: {comp_path}')\n",
                    "else:\n",
                    "    print('matplotlib not installed')\n",
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
