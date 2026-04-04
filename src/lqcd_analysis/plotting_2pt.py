from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


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
    ax.set_ylabel("m_eff(t)")
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
    title: str | None = None,
) -> Path:
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")

    fig, ax = plt.subplots(figsize=(6, 4))
    nstates = values.shape[1]
    for state in range(nstates):
        ax.errorbar(
            tmins,
            values[:, state],
            yerr=errors[:, state],
            fmt="o-",
            ms=4,
            label=f"{prefix}{state}",
        )
    ax.set_xlabel("tmin")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_best_fit_reconstruction(
    output_path: Path,
    times: np.ndarray,
    data: np.ndarray,
    errors: np.ndarray,
    fit_curve: np.ndarray,
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
    ax_top.plot(times, fit_curve, label="fit")
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
    nstates: int,
    model: str,
    title: str,
    nt: int,
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
    meff_mean = meff[:, 1]
    meff_err = meff[:, 2]
    tmax = int(fit[0, 1])
    tmins = fit[:, 0]

    amp_start = 9
    energy_start = amp_start + nstates * 2
    amplitude_values = fit[:, amp_start : amp_start + nstates]
    amplitude_errs = fit[:, amp_start + nstates : amp_start + 2 * nstates]
    energy_values = fit[:, energy_start : energy_start + nstates]
    energy_errs = fit[:, energy_start + nstates : energy_start + 2 * nstates]
    plateau_rows = fit[fit[:, 8] > 0.5]
    representative = plateau_rows[len(plateau_rows) // 2] if len(plateau_rows) else fit[len(fit) // 2]
    representative_tmin = int(representative[0])
    amplitudes = representative[amp_start : amp_start + nstates]
    energies = representative[energy_start : energy_start + nstates]

    fit_times = np.arange(representative_tmin, tmax + 1)
    fit_curve = evaluate_model(fit_times, amplitudes, energies, nt, model)

    outputs = [
        plot_effective_mass(
            output_path / f"{title}_{model}_effective_mass_tmax{tmax}.png",
            meff_times,
            meff_mean,
            meff_err,
            tmax,
            title=f"{title}: effective mass",
        ),
        plot_parameter_scan(
            output_path / f"{title}_{model}_{nstates}state_energies_tmax{tmax}.png",
            tmins,
            energy_values,
            energy_errs,
            prefix="E",
            ylabel="Energy",
            title=f"{title}: {nstates}-state energies",
        ),
        plot_parameter_scan(
            output_path / f"{title}_{model}_{nstates}state_amplitudes_tmax{tmax}.png",
            tmins,
            amplitude_values,
            amplitude_errs,
            prefix="A",
            ylabel="Amplitude",
            title=f"{title}: {nstates}-state amplitudes",
        ),
        plot_best_fit_reconstruction(
            output_path / f"{title}_{model}_{nstates}state_reconstruction_tmax{tmax}.png",
            fit_times,
            corr_mean[representative_tmin : tmax + 1],
            corr_err[representative_tmin : tmax + 1],
            fit_curve,
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
    model: str,
    title: str,
    nt: int,
) -> Path:
    notebook_path = Path(notebook_path)
    notebook_output_dir = Path(notebook_output_dir)
    fit_tables_repr = {str(key): str(Path(value)) for key, value in fit_tables.items()}
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
                    "from lqcd_analysis.plotting_2pt import plot_nstate_outputs\n",
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
                    f"fit_tables = {json.dumps(fit_tables_repr, indent=2)}\n",
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
                    "    nstates=chosen_nstate,\n",
                    "    model=model,\n",
                    "    title=title,\n",
                    "    nt=nt,\n",
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
