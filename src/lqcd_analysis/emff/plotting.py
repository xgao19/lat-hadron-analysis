from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_emff_ratio(
    output_path: str | Path,
    ratio_by_tsep: dict[int, np.ndarray],
    tau_range: tuple[int, int],
    *,
    title: str = "",
    q_label: str = "",
) -> Path:
    """Plot ratio R(tsep, tau) with error bands for each tsep.

    Args:
        output_path: Path for the output PDF file.
        ratio_by_tsep: dict tsep -> bootstrap samples (n_boot, n_tau).
        tau_range: (tau_min, tau_offset) used for display.
        title: Plot title.
        q_label: Label for the momentum transfer (e.g., "q=(0,0,1)").
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 5))
    tau_min, tau_offset = tau_range

    for tsep in sorted(ratio_by_tsep.keys()):
        boot_samples = ratio_by_tsep[tsep]  # (n_boot, n_tau+1)
        tau_vals = np.arange(tsep + 1)
        real_samples = np.real(boot_samples)
        imag_samples = np.imag(boot_samples)
        real_mean = np.nanmean(real_samples, axis=0)
        real_std = np.nanstd(real_samples, axis=0, ddof=1)
        imag_mean = np.nanmean(imag_samples, axis=0)
        imag_std = np.nanstd(imag_samples, axis=0, ddof=1)

        ax_real.errorbar(
            tau_vals, real_mean, yerr=real_std,
            fmt="o-", ms=4, label=f"tsep={tsep}",
        )
        ax_imag.errorbar(
            tau_vals, imag_mean, yerr=imag_std,
            fmt="o-", ms=4, label=f"tsep={tsep}",
        )

    ax_real.set_xlabel("τ")
    ax_real.set_ylabel("Re R(tsep, τ)")
    ax_real.axvline(tau_min, color="gray", ls="--", alpha=0.5)
    if tau_offset < 0:
        # Can't draw a single vertical line for tsep-dependent tau_max
        pass
    ax_real.legend(fontsize=8)

    ax_imag.set_xlabel("τ")
    ax_imag.set_ylabel("Im R(tsep, τ)")
    ax_imag.legend(fontsize=8)

    full_title = f"{title} {q_label}".strip()
    if full_title:
        fig.suptitle(full_title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_emff_summation(
    output_path: str | Path,
    ratio_by_tsep: dict[int, np.ndarray],
    tau_range: tuple[int, int],
    tsep_fit_list: list[int],
    *,
    fit_result=None,
    title: str = "",
    q_label: str = "",
) -> Path:
    """Plot summed ratio S(tsep) vs tsep with optional linear fit.

    Args:
        output_path: Path for the output PDF file.
        ratio_by_tsep: dict tsep -> bootstrap samples (n_boot, n_tau).
        tau_range: (tau_min, tau_offset).
        tsep_fit_list: tsep values included in the fit.
        fit_result: Optional EMFFFitterResult from summation fit.
        title: Plot title.
        q_label: Label for the momentum transfer.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from .models import compute_tau_range_for_tsep

    tau_min, tau_offset = tau_range

    tsep_vals = sorted(ratio_by_tsep.keys())
    summed_mean = []
    summed_err = []
    for tsep in tsep_vals:
        boot_samples = ratio_by_tsep[tsep]
        tau_vals = compute_tau_range_for_tsep(tsep, tau_min, tau_offset)
        summed = np.sum(np.real(boot_samples[:, tau_vals]), axis=1)
        mean, err = (
            (np.percentile(summed, 50), 0.5 * (np.percentile(summed, 84) - np.percentile(summed, 16)))
            if len(summed) > 1
            else (np.mean(summed), 0.0)
        )
        # Use robust mean/error
        from ..common.utils import robust_mean_and_error
        mean, err = robust_mean_and_error(summed)
        summed_mean.append(mean)
        summed_err.append(err)

    tsep_arr = np.array(tsep_vals, dtype=float)
    summed_mean = np.array(summed_mean, dtype=float)
    summed_err = np.array(summed_err, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot all tsep
    all_mask = np.ones(len(tsep_arr), dtype=bool)
    fit_mask = np.array([t in tsep_fit_list for t in tsep_vals], dtype=bool)
    non_fit_mask = all_mask & ~fit_mask

    ax.errorbar(
        tsep_arr[fit_mask], summed_mean[fit_mask],
        yerr=summed_err[fit_mask], fmt="o", ms=6, label="in fit",
    )
    if np.any(non_fit_mask):
        ax.errorbar(
            tsep_arr[non_fit_mask], summed_mean[non_fit_mask],
            yerr=summed_err[non_fit_mask], fmt="o", ms=6,
            alpha=0.4, label="excluded",
        )

    # Draw fit line if available
    if fit_result is not None and fit_result.success:
        tsep_line = np.linspace(tsep_arr.min(), tsep_arr.max(), 100)
        from .models import evaluate_emff_summed_ratio
        fit_line = evaluate_emff_summed_ratio(tsep_line, fit_result.params)
        ax.plot(tsep_line, fit_line, "r-", label="fit")

    ax.set_xlabel("tsep")
    ax.set_ylabel("S(tsep) = Σ_τ R(tsep, τ)")
    ax.legend()

    full_title = f"{title} {q_label} Summation".strip()
    if full_title:
        ax.set_title(full_title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_emff_form_factor_q2(
    output_path: str | Path,
    q2_values: list[float],
    m00_values: list[float],
    m00_errors: list[float],
    *,
    title: str = "",
) -> Path:
    """Plot form factor F(Q^2) = M_00 vs Q^2.

    Args:
        output_path: Path for the output PDF file.
        q2_values: Q^2 values in lattice units or GeV^2.
        m00_values: F(Q^2) central values.
        m00_errors: F(Q^2) error bars.
        title: Plot title.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(q2_values, m00_values, yerr=m00_errors, fmt="o", ms=6, capsize=3)
    ax.set_xlabel("Q²")
    ax.set_ylabel("F(Q²)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
