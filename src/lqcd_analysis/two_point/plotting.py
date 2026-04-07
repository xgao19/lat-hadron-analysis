from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from ..common.fit_tables import parse_fit_table_scan

PLOT_SUFFIX = ".pdf"
HBAR_C_MEV_FM = 197.3269804


def save_plot_status(output_path: Path, message: str) -> Path:
    output_path.write_text(message + "\n", encoding="utf-8")
    return output_path


def prepare_matplotlib():
    if "MPLCONFIGDIR" not in os.environ:
        default_dir = Path.home() / ".matplotlib"
        if not os.access(default_dir, os.W_OK):
            os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def plot_effective_mass(
    output_path: Path,
    times: np.ndarray,
    meff_mean: np.ndarray,
    meff_err: np.ndarray,
    tmax: int,
    title: str | None = None,
) -> Path:
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(times, meff_mean, yerr=meff_err, fmt="o", ms=4)
    ax.axvline(tmax, color="tab:red", linestyle="--", label=f"tmax={tmax}")
    ax.set_xlabel("t")
    ax.set_ylabel("m_eff(t) [MeV]")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_parameter_scan(
    output_path: Path,
    tmins: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    prefix: str,
    ylabel: str,
    state_indices: np.ndarray | None = None,
    plateau_tmin_range: tuple[int, int] | None = None,
    plateau_values: np.ndarray | None = None,
    plateau_errors: np.ndarray | None = None,
    title: str | None = None,
) -> Path:
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")

    fig, ax = plt.subplots(figsize=(6, 4))
    nstates = values.shape[1]
    if state_indices is None:
        state_indices = np.arange(nstates, dtype=int)
    else:
        state_indices = np.asarray(state_indices, dtype=int)
    for state in range(nstates):
        color = f"C{state}"
        ax.errorbar(
            tmins,
            values[:, state],
            yerr=errors[:, state],
            fmt="o",
            ms=4,
            color=color,
            label=f"{prefix}{state_indices[state]}",
        )
        if (
            plateau_tmin_range is not None
            and plateau_values is not None
            and plateau_errors is not None
            and state < len(plateau_values)
            and state < len(plateau_errors)
        ):
            start_tmin, end_tmin = plateau_tmin_range
            center = plateau_values[state]
            half_width = plateau_errors[state]
            ax.fill_between(
                [start_tmin, end_tmin],
                [center - half_width, center - half_width],
                [center + half_width, center + half_width],
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            ax.plot([start_tmin, end_tmin], [center, center], color=color, linewidth=1.0, alpha=0.9)
    ax.set_xlabel("tmin")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def select_scan_state_indices(nstates: int) -> np.ndarray:
    if nstates <= 1:
        return np.array([0], dtype=int)
    return np.array([nstates - 1], dtype=int)


def plot_best_fit_reconstruction(
    output_path: Path,
    times: np.ndarray,
    data: np.ndarray,
    errors: np.ndarray,
    fit_curve: np.ndarray,
    fit_band_low: np.ndarray,
    fit_band_high: np.ndarray,
    title: str | None = None,
    log_scale: bool = True,
) -> Path:
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(6, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top.errorbar(times, data, yerr=errors, fmt="o", ms=4, label="bootstrap mean")
    ax_top.fill_between(times, fit_band_low, fit_band_high, alpha=0.25, label="fit band")
    ax_top.plot(times, fit_curve, linewidth=1.0, label="fit center")
    if log_scale:
        ax_top.set_yscale("log")
    ax_top.set_ylabel("C(t)")
    if title:
        ax_top.set_title(title)
    ax_top.legend()

    ratio = np.divide(data, fit_curve, out=np.full_like(data, np.nan), where=fit_curve != 0.0)
    ax_bottom.axhline(1.0, color="tab:red", linestyle="--")
    ax_bottom.plot(times, ratio, "o-")
    ax_bottom.set_xlabel("t")
    ax_bottom.set_ylabel("data/fit")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def lattice_energy_to_mev(values: np.ndarray, lattice_spacing_fm: float) -> np.ndarray:
    return np.asarray(values, dtype=float) * (HBAR_C_MEV_FM / lattice_spacing_fm)


def build_reconstruction_band(
    times: np.ndarray,
    amplitudes: np.ndarray,
    amplitude_errs: np.ndarray,
    energies: np.ndarray,
    energy_errs: np.ndarray,
    nt: int,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amplitudes = np.asarray(amplitudes, dtype=float)
    amplitude_errs = np.asarray(amplitude_errs, dtype=float)
    energies = np.asarray(energies, dtype=float)
    energy_errs = np.asarray(energy_errs, dtype=float)
    center = evaluate_model(times, amplitudes, energies, nt, model)

    lower = center.copy()
    upper = center.copy()
    nparams = len(amplitudes) + len(energies)
    for mask in range(1 << nparams):
        amp_trial = amplitudes.copy()
        energy_trial = energies.copy()
        for idx in range(len(amplitudes)):
            sign = 1.0 if (mask & (1 << idx)) else -1.0
            amp_trial[idx] = max(amplitudes[idx] + sign * amplitude_errs[idx], 1e-16)
        for idx in range(len(energies)):
            sign = 1.0 if (mask & (1 << (len(amplitudes) + idx))) else -1.0
            energy_trial[idx] = max(energies[idx] + sign * energy_errs[idx], 1e-12)
        energy_trial = np.maximum.accumulate(energy_trial)
        trial_curve = evaluate_model(times, amp_trial, energy_trial, nt, model)
        lower = np.minimum(lower, trial_curve)
        upper = np.maximum(upper, trial_curve)
    return center, lower, upper


def load_table(path: str | Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data


def plot_nstate_outputs(
    output_dir: str | Path,
    correlator_table: str | Path,
    meff_table: str | Path,
    fit_table: str | Path,
    plateau_table: str | Path | None,
    nstates: int,
    model: str,
    title: str,
    nt: int,
    lattice_spacing_fm: float,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    correlator = load_table(correlator_table)
    meff = load_table(meff_table)
    fit = load_table(fit_table)

    times = correlator[:, 0]
    corr_mean = correlator[:, 1]
    corr_err = correlator[:, 2]
    meff_times = meff[:, 0]
    meff_mean = lattice_energy_to_mev(meff[:, 1], lattice_spacing_fm)
    meff_err = lattice_energy_to_mev(meff[:, 2], lattice_spacing_fm)
    meff_mask = meff_times >= 2
    fit_scan = parse_fit_table_scan(fit, nstates)
    tmax = fit_scan.tmax
    tmins = fit_scan.tmins
    amplitude_values = fit_scan.amplitude_values
    amplitude_errs = fit_scan.amplitude_errs
    energy_values = lattice_energy_to_mev(fit_scan.energy_values, lattice_spacing_fm)
    energy_errs = lattice_energy_to_mev(fit_scan.energy_errs, lattice_spacing_fm)
    plateau_tmin_range = fit_scan.plateau_tmin_range
    plateau_start_tmin = fit_scan.plateau_start_tmin
    amplitudes = np.asarray(fit_scan.fallback_params_mean[:nstates], dtype=float)
    amplitude_band_errs = np.asarray(fit_scan.fallback_params_err[:nstates], dtype=float)
    energies = np.asarray(fit_scan.fallback_params_mean[nstates : 2 * nstates], dtype=float)
    energy_band_errs = np.asarray(fit_scan.fallback_params_err[nstates : 2 * nstates], dtype=float)
    if plateau_table is not None and Path(plateau_table).exists():
        plateau_summary = load_table(plateau_table)[0]
        amplitudes = plateau_summary[:nstates]
        energies = plateau_summary[nstates : 2 * nstates]
        amplitude_band_errs = plateau_summary[2 * nstates : 3 * nstates]
        energy_band_errs = plateau_summary[3 * nstates : 4 * nstates]
    scan_state_indices = select_scan_state_indices(nstates)
    plotted_amplitude_values = amplitude_values[:, scan_state_indices]
    plotted_amplitude_errs = amplitude_errs[:, scan_state_indices]
    plotted_energy_values = energy_values[:, scan_state_indices]
    plotted_energy_errs = energy_errs[:, scan_state_indices]
    plotted_amplitudes = amplitudes[scan_state_indices]
    plotted_amplitude_band_errs = amplitude_band_errs[scan_state_indices]
    plotted_energies = energies[scan_state_indices]
    plotted_energy_band_errs = energy_band_errs[scan_state_indices]

    fit_times = np.arange(plateau_start_tmin, tmax + 1)
    fit_curve, fit_band_low, fit_band_high = build_reconstruction_band(
        fit_times,
        amplitudes,
        amplitude_band_errs,
        energies,
        energy_band_errs,
        nt,
        model,
    )

    outputs = [
        plot_effective_mass(
            output_path / f"{title}_{model}_effective_mass_tmax{tmax}{PLOT_SUFFIX}",
            meff_times[meff_mask],
            meff_mean[meff_mask],
            meff_err[meff_mask],
            tmax,
            title=f"{title}: effective mass",
        ),
        plot_parameter_scan(
            output_path / f"{title}_{model}_{nstates}state_energies_tmax{tmax}{PLOT_SUFFIX}",
            tmins,
            plotted_energy_values,
            plotted_energy_errs,
            prefix="E",
            ylabel="Energy [MeV]",
            state_indices=scan_state_indices,
            plateau_tmin_range=plateau_tmin_range,
            plateau_values=lattice_energy_to_mev(plotted_energies, lattice_spacing_fm),
            plateau_errors=lattice_energy_to_mev(plotted_energy_band_errs, lattice_spacing_fm),
            title=f"{title}: {nstates}-state energies",
        ),
        plot_parameter_scan(
            output_path / f"{title}_{model}_{nstates}state_amplitudes_tmax{tmax}{PLOT_SUFFIX}",
            tmins,
            plotted_amplitude_values,
            plotted_amplitude_errs,
            prefix="A",
            ylabel="Amplitude",
            state_indices=scan_state_indices,
            plateau_tmin_range=plateau_tmin_range,
            plateau_values=plotted_amplitudes,
            plateau_errors=plotted_amplitude_band_errs,
            title=f"{title}: {nstates}-state amplitudes",
        ),
        plot_best_fit_reconstruction(
            output_path / f"{title}_{model}_{nstates}state_reconstruction_tmax{tmax}{PLOT_SUFFIX}",
            fit_times,
            corr_mean[plateau_start_tmin : tmax + 1],
            corr_err[plateau_start_tmin : tmax + 1],
            fit_curve,
            fit_band_low,
            fit_band_high,
            title=f"{title}: {nstates}-state reconstruction",
        ),
    ]
    return outputs


def evaluate_model(
    times: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    nt: int,
    model: str,
) -> np.ndarray:
    t = np.asarray(times, dtype=float)[:, None]
    a = np.asarray(amplitudes, dtype=float)[None, :]
    e = np.asarray(energies, dtype=float)[None, :]
    forward = np.exp(-t * e)
    if model == "normal":
        kernel = forward
    else:
        backward = np.exp(-(nt - t) * e)
        kernel = forward + backward if model == "symmetric" else forward - backward
    return np.sum(a * kernel, axis=1)


def write_nstate_plot_notebook(
    notebook_path: str | Path,
    notebook_output_dir: str | Path,
    correlator_table: str | Path,
    meff_table: str | Path,
    fit_tables: dict[int, str | Path],
    plateau_tables: dict[int, str | Path],
    model: str,
    title: str,
    nt: int,
    lattice_spacing_fm: float,
) -> Path:
    notebook_path = Path(notebook_path)
    notebook_output_dir = Path(notebook_output_dir)
    fit_tables_repr = {str(key): str(Path(value)) for key, value in fit_tables.items()}
    plateau_tables_repr = {str(key): str(Path(value)) for key, value in plateau_tables.items()}
    repo_src = Path(__file__).resolve().parents[1]
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 2pt Plot Notebook\n",
                    "This notebook calls the reusable plotting module.\n",
                    "You can edit paths, choose which state to draw, or tweak styles before rerunning cells.\n",
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
                    f"sys.path.insert(0, {str(repo_src)!r})\n",
                    "from lqcd_analysis.two_point.plotting import plot_nstate_outputs\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"title = {title!r}\n",
                    f"model = {model!r}\n",
                    f"output_dir = Path({str(notebook_output_dir)!r})\n",
                    f"correlator_table = Path({str(Path(correlator_table))!r})\n",
                    f"meff_table = Path({str(Path(meff_table))!r})\n",
                    f"nt = {nt}\n",
                    f"lattice_spacing_fm = {lattice_spacing_fm}\n",
                    f"fit_tables = {json.dumps(fit_tables_repr, indent=2)}\n",
                    f"plateau_tables = {json.dumps(plateau_tables_repr, indent=2)}\n",
                    "available_nstates = sorted(int(key) for key in fit_tables)\n",
                    "chosen_nstate = max(available_nstates)\n",
                    "output_dir.mkdir(parents=True, exist_ok=True)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Quick Tuning\n",
                    "Change `chosen_nstate` below to switch between 1-state, 2-state, and 3-state plots.\n",
                    "You can also edit `output_dir` to save alternate figure versions.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Available nstates:', available_nstates)\n",
                    "chosen_nstate = available_nstates[-1]\n",
                    "chosen_nstate\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plot_nstate_outputs(\n",
                    "    output_dir=output_dir,\n",
                    "    correlator_table=correlator_table,\n",
                    "    meff_table=meff_table,\n",
                    "    fit_table=Path(fit_tables[str(chosen_nstate)]),\n",
                    "    plateau_table=Path(plateau_tables[str(chosen_nstate)]),\n",
                    "    nstates=chosen_nstate,\n",
                    "    model=model,\n",
                    "    title=title,\n",
                    "    nt=nt,\n",
                    "    lattice_spacing_fm=lattice_spacing_fm,\n",
                    ")\n",
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
