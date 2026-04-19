from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from ..common.constants import (
    MIN_AMPLITUDE,
    MIN_COVARIANCE_DIAGONAL,
    MIN_POSITIVE,
    SHRINKAGE_LAMBDAS,
)
from .io import load_correlator_csv
from .plotting import plot_nstate_outputs, write_nstate_plot_notebook
from ..common.bootstrap import (
    compute_bootstrap_covariance,
    shrink_covariance_to_diagonal,
)
from ..common.parsing import load_fit_window_table, load_int_mapping_table, parse_bool, parse_fold_t, parse_optional_int
from ..common.utils import (
    apply_fold_t,
    bin_correlators,
    bootstrap_correlator_means,
    robust_mean_and_error,
)
from .effective_mass import effective_mass_with_bootstrap


@dataclass(frozen=True)
class NStateFitInput:
    title_pattern: str
    ns: int
    nt: int
    lattice_spacing_fm: float
    correlator_path_pattern: str
    pzlist: tuple[int, ...]
    fold_t: str
    tmax: int | str
    model: str
    fit_mode: str
    pz0_ground_energy: float | None
    fix_ground_energy_from_dispersion: bool
    nstates: tuple[int, ...]
    tmin_window: str
    binsize: int
    bootstrap_samples: int | None
    bootstrap_size: int | None
    seed: int
    lambda_prior: float
    make_plots: bool
    results_dir: Path


@dataclass(frozen=True)
class FitSummaryRow:
    nstates: int
    tmin: int
    tmax: int
    success_meanfit: int
    bootstrap_successes: int
    bootstrap_total: int
    bootstrap_success_fraction: float
    fallback_uncorrelated_successes: int
    chi2_dof: float
    pvalue: float
    selected_window_flag: int
    params_mean: tuple[float, ...]
    params_err: tuple[float, ...]


@dataclass(frozen=True)
class FitWindowSummary:
    start_tmin: int
    end_tmin: int
    representative_tmin: int
    energy_mean: float
    amplitude_mean: float


@dataclass(frozen=True)
class FitWindowParameterSummary:
    params_mean: tuple[float, ...]
    params_err: tuple[float, ...]


@dataclass(frozen=True)
class StateArtifacts:
    nstates: int
    fit_window_summary: FitWindowSummary
    fit_window_parameter_summary: FitWindowParameterSummary
    fit_window_start_fallback_uncorrelated_successes: int
    fit_window_start_shrinkage_lambda: float | None
    fit_table_path: Path
    fit_window_table_path: Path


@dataclass(frozen=True)
class FitResult:
    params: np.ndarray
    chi2: float
    chi2_dof: float
    pvalue: float
    success: bool
    message: str
    used_uncorrelated_fallback: bool = False


@dataclass(frozen=True)
class EnergyPrior:
    energy_index: int
    center: float
    sigma: float


@dataclass(frozen=True)
class ResidualModel:
    fit_mode: str
    sigma: np.ndarray | None = None
    cholesky_factor: np.ndarray | None = None
    fallback_sigma: np.ndarray | None = None
    shrinkage_lambda: float | None = None


def parse_nstate_fit_input(path: str | Path, results_dir: str | Path | None = None) -> NStateFitInput:
    file_path = Path(path)
    entries: dict[str, list[str]] = {}
    first_tokens: list[str] | None = None

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if first_tokens is None:
                first_tokens = tokens
            entries[tokens[0]] = tokens[1:]

    if first_tokens is None or len(first_tokens) < 4:
        raise ValueError("the first non-empty line must be: title Ns Nt a_fm")

    required = {"c2pt", "pzlist", "model", "nstates", "tmin_window", "tmax"}
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")

    model = entries["model"][0].lower()
    if model not in {"normal", "symmetric", "antisymmetric"}:
        raise ValueError("model must be one of: normal, symmetric, antisymmetric")

    fit_mode = entries.get("fit_mode", ["uncorrelated"])[0].lower()
    if fit_mode not in {"uncorrelated", "correlated"}:
        raise ValueError("fit_mode must be one of: uncorrelated, correlated")

    nstates = tuple(sorted({int(item) for item in entries["nstates"]}))
    if not nstates or any(state not in {1, 2, 3} for state in nstates):
        raise ValueError("nstates must contain only 1, 2, and/or 3")

    fix_ground_energy_from_dispersion = parse_bool(
        entries.get("fix_ground_energy_from_dispersion", ["false"])[0]
    )
    if fix_ground_energy_from_dispersion and "pz0_ground_energy" not in entries:
        raise ValueError("pz0_ground_energy is required when fix_ground_energy_from_dispersion is true")

    input_path = file_path.resolve()
    tmax_value = entries["tmax"][0]
    try:
        parsed_tmax: int | str = int(tmax_value)
    except ValueError:
        parsed_tmax = tmax_value

    return NStateFitInput(
        title_pattern=first_tokens[0],
        ns=int(first_tokens[1]),
        nt=int(first_tokens[2]),
        lattice_spacing_fm=float(first_tokens[3]),
        correlator_path_pattern=entries["c2pt"][0],
        pzlist=tuple(int(item) for item in entries["pzlist"]),
        fold_t=parse_fold_t(entries),
        tmax=parsed_tmax,
        model=model,
        fit_mode=fit_mode,
        pz0_ground_energy=(
            float(entries["pz0_ground_energy"][0]) if "pz0_ground_energy" in entries else None
        ),
        fix_ground_energy_from_dispersion=fix_ground_energy_from_dispersion,
        nstates=nstates,
        tmin_window=entries["tmin_window"][0],
        binsize=int(entries.get("binsize", ["1"])[0]),
        bootstrap_samples=parse_optional_int(entries.get("bootstrap_samples", ["auto"])[0]),
        bootstrap_size=parse_optional_int(entries.get("bootstrap_size", ["auto"])[0]),
        seed=int(entries.get("seed", ["2026"])[0]),
        lambda_prior=float(entries.get("lambda_prior", ["1.0"])[0]),
        make_plots=parse_bool(entries.get("plot", ["true"])[0]),
        results_dir=(
            Path(entries["results_dir"][0])
            if "results_dir" in entries and results_dir is None
            else ((input_path.parent / "results_nstate_fit") if results_dir is None else Path(results_dir))
        ),
    )










def build_residual_model(
    fit_mode: str,
    sigma: np.ndarray,
    covariance: np.ndarray | None = None,
    shrinkage_lambda: float = 0.0,
) -> ResidualModel:
    sigma = np.asarray(sigma, dtype=float)
    if fit_mode == "uncorrelated":
        return ResidualModel(fit_mode="uncorrelated", sigma=np.clip(sigma, MIN_POSITIVE, None))
    if covariance is None:
        raise ValueError("covariance is required for correlated fitting")
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    if shrinkage_lambda != 0.0:
        covariance = shrink_covariance_to_diagonal(covariance, shrinkage_lambda)
    cholesky_factor = np.linalg.cholesky(covariance)
    fallback_sigma = np.sqrt(np.clip(np.diag(covariance), MIN_COVARIANCE_DIAGONAL, None))
    return ResidualModel(
        fit_mode="correlated",
        cholesky_factor=cholesky_factor,
        fallback_sigma=np.clip(fallback_sigma, MIN_POSITIVE, None),
        shrinkage_lambda=shrinkage_lambda,
    )


def find_first_usable_correlated_residual_model(
    sigma: np.ndarray,
    covariance: np.ndarray,
) -> tuple[ResidualModel | None, float | None]:
    try:
        return build_residual_model("correlated", sigma, covariance, shrinkage_lambda=0.0), 0.0
    except np.linalg.LinAlgError:
        pass

    for shrinkage_lambda in SHRINKAGE_LAMBDAS:
        try:
            return (
                build_residual_model("correlated", sigma, covariance, shrinkage_lambda=shrinkage_lambda),
                shrinkage_lambda,
            )
        except np.linalg.LinAlgError:
            continue
    return None, None


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


def pack_fit_parameters(amplitudes: np.ndarray, energies: np.ndarray) -> np.ndarray:
    amps = np.clip(np.asarray(amplitudes, dtype=float), MIN_AMPLITUDE, None)
    en = np.asarray(energies, dtype=float)
    ordered = np.maximum.accumulate(np.clip(en, MIN_POSITIVE, None))

    theta = [*np.log(amps), np.log(ordered[0])]
    for idx in range(1, len(ordered)):
        theta.append(np.log(max(ordered[idx] - ordered[idx - 1], MIN_POSITIVE)))
    return np.array(theta, dtype=float)


def unpack_fit_parameters(theta: np.ndarray, nstates: int) -> tuple[np.ndarray, np.ndarray]:
    amps = np.exp(theta[:nstates])
    energy_parts = theta[nstates:]
    energies = np.empty(nstates, dtype=float)
    energies[0] = np.exp(energy_parts[0])
    for idx in range(1, nstates):
        energies[idx] = energies[idx - 1] + np.exp(energy_parts[idx])
    return amps, energies


def pack_fit_parameters_with_fixed_ground_energy(
    amplitudes: np.ndarray,
    energies: np.ndarray,
    *,
    fixed_ground_energy: float | None,
) -> np.ndarray:
    if fixed_ground_energy is None:
        return pack_fit_parameters(amplitudes, energies)

    amps = np.clip(np.asarray(amplitudes, dtype=float), MIN_AMPLITUDE, None)
    ordered = np.maximum.accumulate(np.clip(np.asarray(energies, dtype=float), MIN_POSITIVE, None))
    ground_energy = max(float(fixed_ground_energy), MIN_POSITIVE)
    theta = [*np.log(amps)]
    previous_energy = ground_energy
    for idx in range(1, len(ordered)):
        theta.append(np.log(max(ordered[idx] - previous_energy, MIN_POSITIVE)))
        previous_energy = ordered[idx]
    return np.array(theta, dtype=float)


def unpack_fit_parameters_with_fixed_ground_energy(
    theta: np.ndarray,
    nstates: int,
    *,
    fixed_ground_energy: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if fixed_ground_energy is None:
        return unpack_fit_parameters(theta, nstates)

    amps = np.exp(theta[:nstates])
    gap_parts = theta[nstates:]
    energies = np.empty(nstates, dtype=float)
    energies[0] = max(float(fixed_ground_energy), MIN_POSITIVE)
    previous_energy = energies[0]
    for idx in range(1, nstates):
        previous_energy = previous_energy + np.exp(gap_parts[idx - 1])
        energies[idx] = previous_energy
    return amps, energies


def fit_residuals(
    theta: np.ndarray,
    times: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    nt: int,
    model: str,
    nstates: int,
    priors: tuple[EnergyPrior, ...] = (),
    lambda_prior: float = 1.0,
    residual_model: ResidualModel | None = None,
    fixed_ground_energy: float | None = None,
) -> np.ndarray:
    amplitudes, energies = unpack_fit_parameters_with_fixed_ground_energy(
        theta,
        nstates,
        fixed_ground_energy=fixed_ground_energy,
    )
    model_values = evaluate_model(times, amplitudes, energies, nt, model)
    delta = model_values - data
    if residual_model is None or residual_model.fit_mode == "uncorrelated":
        data_residual = delta / np.asarray(
            sigma if residual_model is None or residual_model.sigma is None else residual_model.sigma,
            dtype=float,
        )
    elif residual_model.fit_mode == "correlated":
        if residual_model.cholesky_factor is None:
            raise ValueError("correlated residual model requires a Cholesky factor")
        data_residual = np.linalg.solve(residual_model.cholesky_factor, delta)
    else:
        raise ValueError(f"unsupported residual fit mode: {residual_model.fit_mode}")
    if lambda_prior <= 0.0 or not priors:
        return data_residual

    prior_residuals: list[float] = []
    scale = np.sqrt(lambda_prior)
    for prior in priors:
        if (
            prior.energy_index < 0
            or prior.energy_index >= len(energies)
            or not np.isfinite(prior.center)
            or not np.isfinite(prior.sigma)
            or prior.sigma <= 0.0
        ):
            continue
        prior_residuals.append(scale * (energies[prior.energy_index] - prior.center) / prior.sigma)

    if not prior_residuals:
        return data_residual
    return np.concatenate([data_residual, np.array(prior_residuals, dtype=float)])


def fit_nstate_sample(
    times: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    nt: int,
    model: str,
    initial_amplitudes: np.ndarray,
    initial_energies: np.ndarray,
    nstates: int,
    priors: tuple[EnergyPrior, ...] = (),
    lambda_prior: float = 1.0,
    residual_model: ResidualModel | None = None,
    covariance: np.ndarray | None = None,
    fixed_ground_energy: float | None = None,
) -> FitResult:
    primary_amplitudes = np.asarray(initial_amplitudes, dtype=float)
    primary_energies = np.asarray(initial_energies, dtype=float)

    def run_attempt(
        amp_guess: np.ndarray,
        energy_guess: np.ndarray,
        active_sigma: np.ndarray,
        active_model: ResidualModel | None,
    ) -> FitResult:
        theta0 = pack_fit_parameters_with_fixed_ground_energy(
            amp_guess,
            energy_guess,
            fixed_ground_energy=fixed_ground_energy,
        )
        result = least_squares(
            fit_residuals,
            theta0,
            method="trf",
            max_nfev=8000,
            args=(
                times,
                data,
                active_sigma,
                nt,
                model,
                nstates,
                priors,
                lambda_prior,
                active_model,
                fixed_ground_energy,
            ),
        )
        amplitudes, energies = unpack_fit_parameters_with_fixed_ground_energy(
            result.x,
            nstates,
            fixed_ground_energy=fixed_ground_energy,
        )
        residual = fit_residuals(
            result.x,
            times,
            data,
            active_sigma,
            nt,
            model,
            nstates,
            priors,
            lambda_prior,
            active_model,
            fixed_ground_energy,
        )
        chi2_value = float(np.dot(residual, residual))
        dof = max(len(residual) - len(result.x), 1)
        chi2_dof = chi2_value / dof
        pvalue = float(chi2.sf(chi2_value, dof))
        return FitResult(
            params=np.concatenate([amplitudes, energies]),
            chi2=chi2_value,
            chi2_dof=chi2_dof,
            pvalue=pvalue,
            success=bool(result.success),
            message=(
                f"shrinkage_lambda={active_model.shrinkage_lambda:.2f}; {result.message}"
                if active_model is not None
                and active_model.fit_mode == "correlated"
                and active_model.shrinkage_lambda is not None
                and active_model.shrinkage_lambda > 0.0
                else result.message
            ),
            used_uncorrelated_fallback=False,
        )

    def run_attempt_grid(active_sigma: np.ndarray, active_model: ResidualModel | None) -> FitResult:
        primary_result = run_attempt(primary_amplitudes, primary_energies, active_sigma, active_model)
        if primary_result.success and np.all(np.isfinite(primary_result.params)):
            return primary_result

        for amp_guess, energy_guess in build_fallback_fit_attempts(primary_amplitudes, primary_energies, nstates):
            current = run_attempt(amp_guess, energy_guess, active_sigma, active_model)
            if current.success and np.all(np.isfinite(current.params)):
                return current
        return FitResult(
            params=np.full(2 * nstates, np.nan),
            chi2=np.nan,
            chi2_dof=np.nan,
            pvalue=np.nan,
            success=False,
            message="all nonlinear fit attempts failed",
            used_uncorrelated_fallback=False,
        )

    result = run_attempt_grid(sigma, residual_model)
    if result.success:
        return result

    if residual_model is not None and residual_model.fit_mode == "correlated" and covariance is not None:
        current_lambda = 0.0 if residual_model.shrinkage_lambda is None else residual_model.shrinkage_lambda
        for shrinkage_lambda in SHRINKAGE_LAMBDAS:
            if shrinkage_lambda <= current_lambda + MIN_POSITIVE:
                continue
            try:
                shrunk_model = build_residual_model(
                    "correlated",
                    sigma,
                    covariance,
                    shrinkage_lambda=shrinkage_lambda,
                )
            except np.linalg.LinAlgError:
                continue
            fallback_result = run_attempt_grid(sigma, shrunk_model)
            if fallback_result.success:
                return FitResult(
                    params=fallback_result.params,
                    chi2=fallback_result.chi2,
                    chi2_dof=fallback_result.chi2_dof,
                    pvalue=fallback_result.pvalue,
                    success=True,
                    message=(
                        f"correlated fit failed; retried with shrinkage_lambda={shrinkage_lambda:.2f}; "
                        f"{fallback_result.message}"
                    ),
                    used_uncorrelated_fallback=abs(shrinkage_lambda - 1.0) < MIN_POSITIVE,
                )

    return result


def build_fallback_fit_attempts(
    amplitudes: np.ndarray,
    energies: np.ndarray,
    nstates: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    base_amp = np.clip(np.asarray(amplitudes, dtype=float), MIN_AMPLITUDE, None)
    base_energy = np.asarray(energies, dtype=float)
    amplitude_scales = [1.0, 0.5, 2.0]
    excited_scales = [1.0, 1.3, 1.6]
    attempts: list[tuple[np.ndarray, np.ndarray]] = []

    for amp_scale in amplitude_scales:
        for excited_scale in excited_scales:
            if amp_scale == 1.0 and excited_scale == 1.0:
                continue
            amp_guess = base_amp * amp_scale
            energy_guess = base_energy.copy()
            for idx in range(1, nstates):
                gap = max(base_energy[idx] - base_energy[idx - 1], 1e-6)
                energy_guess[idx] = energy_guess[idx - 1] + gap * excited_scale
            attempts.append((amp_guess, energy_guess))
    return attempts


def summarize_bootstrap_parameters(samples: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    means = []
    errors = []
    for column in range(samples.shape[1]):
        mean, error = robust_mean_and_error(samples[:, column])
        means.append(mean)
        errors.append(error)
    return tuple(means), tuple(errors)


def build_energy_priors_from_fit_window_summary(
    previous_summary: FitWindowParameterSummary | None,
    nstates: int,
) -> tuple[EnergyPrior, ...]:
    if previous_summary is None or nstates <= 1:
        return ()
    previous_nstates = len(previous_summary.params_mean) // 2
    max_index = min(previous_nstates, nstates - 1)
    priors: list[EnergyPrior] = []
    for energy_index in range(max_index):
        center = previous_summary.params_mean[previous_nstates + energy_index]
        sigma = previous_summary.params_err[previous_nstates + energy_index]
        if np.isfinite(center) and np.isfinite(sigma) and sigma > 0.0:
            priors.append(EnergyPrior(energy_index=energy_index, center=center, sigma=sigma))
    return tuple(priors)


def target_ground_energy_from_pz0(pz0_ground_energy: float, pz: int, ns: int) -> float:
    momentum = 2.0 * np.pi * float(pz) / float(ns)
    return float(np.sqrt(float(pz0_ground_energy) ** 2 + momentum**2))


def build_initial_guess_from_fit_window(
    fit_window: FitWindowSummary,
    nstates: int,
) -> tuple[np.ndarray, np.ndarray]:
    amplitudes = np.full(nstates, fit_window.amplitude_mean, dtype=float)
    energies = np.full(nstates, fit_window.energy_mean, dtype=float)
    if nstates >= 2:
        energies[1] = max(2.0 * fit_window.energy_mean, fit_window.energy_mean + 0.2)
    if nstates >= 3:
        energies[2] = max(1.5 * energies[1], 3.0 * fit_window.energy_mean)
        amplitudes[2] = amplitudes[1]
    return amplitudes, energies


def compute_fit_window_parameter_summary(
    rows: list[FitSummaryRow],
    sample_tables: dict[int, np.ndarray],
    fit_window: FitWindowSummary,
) -> FitWindowParameterSummary:
    fit_window_rows = [row for row in rows if fit_window.start_tmin <= row.tmin <= fit_window.end_tmin]
    if not fit_window_rows:
        raise ValueError("fit window does not overlap any fit rows")

    nparams = len(fit_window_rows[0].params_mean)
    row_errs = np.array([[max(value, MIN_POSITIVE) for value in row.params_err] for row in fit_window_rows], dtype=float)
    weights = 1.0 / row_errs**2

    sample_count = max(sample_table.shape[0] for sample_table in sample_tables.values())
    sample_averages = np.full((sample_count, nparams), np.nan, dtype=float)
    for sample_id in range(sample_count):
        for param_index in range(nparams):
            values = []
            param_weights = []
            for row_index, row in enumerate(fit_window_rows):
                sample_table = sample_tables.get(row.tmin)
                if sample_table is None or sample_id >= sample_table.shape[0]:
                    continue
                sample_row = sample_table[sample_id]
                if sample_row[2] < 0.5:
                    continue
                value = sample_row[5 + param_index]
                if not np.isfinite(value):
                    continue
                values.append(value)
                param_weights.append(weights[row_index, param_index])
            if values:
                value_array = np.asarray(values, dtype=float)
                weight_array = np.asarray(param_weights, dtype=float)
                sample_averages[sample_id, param_index] = np.sum(weight_array * value_array) / np.sum(weight_array)

    params_mean = []
    params_err = []
    for param_index in range(nparams):
        finite_values = sample_averages[np.isfinite(sample_averages[:, param_index]), param_index]
        mean, error = robust_mean_and_error(finite_values)
        params_mean.append(mean)
        params_err.append(error)
    return FitWindowParameterSummary(params_mean=tuple(params_mean), params_err=tuple(params_err))


def _prepare_fit_window_data(
    mean_correlator: np.ndarray,
    sigma: np.ndarray,
    covariance: np.ndarray | None,
    tmin: int,
    tmax: int,
    time_offset: int,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    label_tmin = int(tmin + time_offset)
    label_tmax = int(tmax + time_offset)
    times = np.arange(label_tmin, label_tmax + 1)
    data_mean = mean_correlator[tmin : tmax + 1]
    sigma_slice = np.clip(sigma[tmin : tmax + 1], MIN_POSITIVE, None)
    covariance_slice = None if covariance is None else covariance[tmin : tmax + 1, tmin : tmax + 1]
    return label_tmin, label_tmax, times, data_mean, sigma_slice, covariance_slice


def _build_window_residual_model(
    fit_mode: str,
    sigma_slice: np.ndarray,
    covariance_slice: np.ndarray | None,
    *,
    label_tmin: int,
    label_tmax: int,
) -> tuple[ResidualModel, str | None]:
    residual_model_note: str | None = None
    if fit_mode == "correlated" and covariance_slice is not None:
        residual_model, chosen_lambda = find_first_usable_correlated_residual_model(sigma_slice, covariance_slice)
        if residual_model is None:
            residual_model = build_residual_model("uncorrelated", sigma_slice)
            residual_model_note = (
                f"correlated window covariance setup failed for tmin={label_tmin}, tmax={label_tmax}; "
                "no usable shrinkage_lambda found"
            )
            print(f"[nstate-fit] {residual_model_note}")
        elif chosen_lambda is not None and chosen_lambda > 0.0:
            residual_model_note = (
                f"correlated window covariance setup used shrinkage_lambda={chosen_lambda:.2f} "
                f"for tmin={label_tmin}, tmax={label_tmax}"
            )
            print(f"[nstate-fit] {residual_model_note}")
        return residual_model, residual_model_note
    return build_residual_model(fit_mode, sigma_slice, covariance_slice), residual_model_note


def _run_mean_window_fit(
    times: np.ndarray,
    data_mean: np.ndarray,
    sigma_slice: np.ndarray,
    residual_model: ResidualModel,
    previous_meanfit: FitResult | None,
    initial_amplitudes: np.ndarray,
    initial_energies: np.ndarray,
    *,
    nt: int,
    model: str,
    nstates: int,
    priors: tuple[EnergyPrior, ...],
    lambda_prior: float,
    covariance_slice: np.ndarray | None,
    fixed_ground_energy: float | None,
) -> FitResult:
    meanfit_amplitudes = initial_amplitudes
    meanfit_energies = initial_energies
    if previous_meanfit is not None and previous_meanfit.success and np.all(np.isfinite(previous_meanfit.params)):
        meanfit_amplitudes = previous_meanfit.params[:nstates]
        meanfit_energies = previous_meanfit.params[nstates:]
    meanfit = fit_nstate_sample(
        times,
        data_mean,
        sigma_slice if residual_model.sigma is None else residual_model.sigma,
        nt,
        model,
        meanfit_amplitudes,
        meanfit_energies,
        nstates,
        priors=priors,
        lambda_prior=lambda_prior,
        residual_model=residual_model,
        covariance=covariance_slice,
        fixed_ground_energy=fixed_ground_energy,
    )
    return meanfit


def _run_sample_window_fits(
    bootstrap_means: np.ndarray,
    times: np.ndarray,
    tmin: int,
    tmax: int,
    sigma_slice: np.ndarray,
    residual_model: ResidualModel,
    previous_sample_fits: list[FitResult | None],
    meanfit: FitResult,
    initial_amplitudes: np.ndarray,
    initial_energies: np.ndarray,
    *,
    nt: int,
    model: str,
    nstates: int,
    priors: tuple[EnergyPrior, ...],
    lambda_prior: float,
    covariance_slice: np.ndarray | None,
    fixed_ground_energy: float | None,
) -> tuple[np.ndarray, list[FitResult | None], list[np.ndarray], int]:
    sample_rows = np.full((len(bootstrap_means), 5 + 2 * nstates), np.nan, dtype=float)
    success_params: list[np.ndarray] = []
    fallback_uncorrelated_successes = 0
    next_previous_sample_fits = previous_sample_fits
    for sample_id, sample in enumerate(bootstrap_means):
        sample_amplitudes = meanfit.params[:nstates] if meanfit.success else initial_amplitudes
        sample_energies = meanfit.params[nstates:] if meanfit.success else initial_energies
        previous_sample = next_previous_sample_fits[sample_id]
        if previous_sample is not None and previous_sample.success and np.all(np.isfinite(previous_sample.params)):
            sample_amplitudes = previous_sample.params[:nstates]
            sample_energies = previous_sample.params[nstates:]
        fit_result = fit_nstate_sample(
            times,
            sample[tmin : tmax + 1],
            sigma_slice if residual_model.sigma is None else residual_model.sigma,
            nt,
            model,
            sample_amplitudes,
            sample_energies,
            nstates,
            priors=priors,
            lambda_prior=lambda_prior,
            residual_model=residual_model,
            covariance=covariance_slice,
            fixed_ground_energy=fixed_ground_energy,
        )
        next_previous_sample_fits[sample_id] = (
            fit_result if fit_result.success and np.all(np.isfinite(fit_result.params)) else None
        )
        sample_rows[sample_id, 0] = tmin
        sample_rows[sample_id, 1] = sample_id
        sample_rows[sample_id, 2] = 1.0 if fit_result.success else 0.0
        sample_rows[sample_id, 3] = fit_result.chi2_dof
        sample_rows[sample_id, 4] = fit_result.pvalue
        sample_rows[sample_id, 5:] = fit_result.params
        if fit_result.success and np.all(np.isfinite(fit_result.params)):
            success_params.append(fit_result.params)
            if fit_result.used_uncorrelated_fallback:
                fallback_uncorrelated_successes += 1
    return sample_rows, next_previous_sample_fits, success_params, fallback_uncorrelated_successes


def run_sliding_fits(
    bootstrap_means: np.ndarray,
    sigma: np.ndarray,
    fit_mode: str,
    nt: int,
    model: str,
    nstates: int,
    tmin_values: range,
    tmax: int,
    initial_amplitudes: np.ndarray,
    initial_energies: np.ndarray,
    covariance: np.ndarray | None = None,
    priors: tuple[EnergyPrior, ...] = (),
    lambda_prior: float = 1.0,
    fixed_ground_energy: float | None = None,
    time_offset: int = 0,
) -> tuple[list[FitSummaryRow], dict[int, np.ndarray], dict[int, FitResult]]:
    rows: list[FitSummaryRow] = []
    sample_tables: dict[int, np.ndarray] = {}
    meanfit_results: dict[int, FitResult] = {}

    mean_correlator = np.mean(bootstrap_means, axis=0)
    previous_meanfit: FitResult | None = None
    previous_sample_fits: list[FitResult | None] = [None] * len(bootstrap_means)
    for tmin in tmin_values:
        label_tmin, label_tmax, times, data_mean, sigma_slice, covariance_slice = _prepare_fit_window_data(
            mean_correlator,
            sigma,
            covariance,
            tmin,
            tmax,
            time_offset,
        )
        residual_model, residual_model_note = _build_window_residual_model(
            fit_mode,
            sigma_slice,
            covariance_slice,
            label_tmin=label_tmin,
            label_tmax=label_tmax,
        )
        meanfit = _run_mean_window_fit(
            times,
            data_mean,
            sigma_slice,
            residual_model,
            previous_meanfit,
            initial_amplitudes,
            initial_energies,
            nt=nt,
            model=model,
            nstates=nstates,
            priors=priors,
            lambda_prior=lambda_prior,
            covariance_slice=covariance_slice,
            fixed_ground_energy=fixed_ground_energy,
        )
        if residual_model_note is not None and not meanfit.success:
            meanfit = FitResult(
                params=meanfit.params,
                chi2=meanfit.chi2,
                chi2_dof=meanfit.chi2_dof,
                pvalue=meanfit.pvalue,
                success=meanfit.success,
                message=f"{residual_model_note}; {meanfit.message}",
            )
        meanfit_results[label_tmin] = meanfit
        previous_meanfit = meanfit if meanfit.success and np.all(np.isfinite(meanfit.params)) else None

        sample_rows, previous_sample_fits, success_params, fallback_uncorrelated_successes = _run_sample_window_fits(
            bootstrap_means,
            times,
            tmin,
            tmax,
            sigma_slice,
            residual_model,
            previous_sample_fits,
            meanfit,
            initial_amplitudes,
            initial_energies,
            nt=nt,
            model=model,
            nstates=nstates,
            priors=priors,
            lambda_prior=lambda_prior,
            covariance_slice=covariance_slice,
            fixed_ground_energy=fixed_ground_energy,
        )

        sample_tables[tmin] = sample_rows
        if success_params:
            params_mean, params_err = summarize_bootstrap_parameters(np.vstack(success_params))
        else:
            params_mean = tuple(np.nan for _ in range(2 * nstates))
            params_err = tuple(np.nan for _ in range(2 * nstates))

        rows.append(
            FitSummaryRow(
                nstates=nstates,
                tmin=label_tmin,
                tmax=label_tmax,
                success_meanfit=1 if meanfit.success else 0,
                bootstrap_successes=len(success_params),
                bootstrap_total=len(bootstrap_means),
                bootstrap_success_fraction=len(success_params) / len(bootstrap_means),
                fallback_uncorrelated_successes=fallback_uncorrelated_successes,
                chi2_dof=meanfit.chi2_dof,
                pvalue=meanfit.pvalue,
                selected_window_flag=0,
                params_mean=params_mean,
                params_err=params_err,
            )
        )

    return rows, sample_tables, meanfit_results


def serialize_fit_rows(rows: list[FitSummaryRow]) -> np.ndarray:
    max_states = max(row.nstates for row in rows)
    columns = []
    for row in rows:
        base = [
            row.tmin,
            row.tmax,
            row.success_meanfit,
            row.bootstrap_successes,
            row.bootstrap_total,
            row.bootstrap_success_fraction,
            row.fallback_uncorrelated_successes,
            row.chi2_dof,
            row.pvalue,
            row.selected_window_flag,
        ]
        amps = list(row.params_mean[: row.nstates])
        amp_errs = list(row.params_err[: row.nstates])
        energies = list(row.params_mean[row.nstates :])
        energy_errs = list(row.params_err[row.nstates :])
        while len(amps) < max_states:
            amps.append(np.nan)
            amp_errs.append(np.nan)
            energies.append(np.nan)
            energy_errs.append(np.nan)
        columns.append(base + amps + amp_errs + energies + energy_errs)
    return np.array(columns, dtype=float)


def mark_fit_window(rows: list[FitSummaryRow], fit_window: FitWindowSummary) -> list[FitSummaryRow]:
    marked = []
    for row in rows:
        flagged = 1 if fit_window.start_tmin <= row.tmin <= fit_window.end_tmin else 0
        marked.append(
            FitSummaryRow(
                nstates=row.nstates,
                tmin=row.tmin,
                tmax=row.tmax,
                success_meanfit=row.success_meanfit,
                bootstrap_successes=row.bootstrap_successes,
                bootstrap_total=row.bootstrap_total,
                bootstrap_success_fraction=row.bootstrap_success_fraction,
                fallback_uncorrelated_successes=row.fallback_uncorrelated_successes,
                chi2_dof=row.chi2_dof,
                pvalue=row.pvalue,
                selected_window_flag=flagged,
                params_mean=row.params_mean,
                params_err=row.params_err,
            )
        )
    return marked


def header_for_fit_rows(max_states: int) -> str:
    columns = [
        "tmin",
        "tmax",
        "success_meanfit",
        "bootstrap_successes",
        "bootstrap_total",
        "bootstrap_success_fraction",
        "fallback_uncorrelated_successes",
        "chi2_dof",
        "pvalue",
        "selected_window_flag",
    ]
    columns += [f"A{idx}_mean" for idx in range(max_states)]
    columns += [f"A{idx}_err" for idx in range(max_states)]
    columns += [f"E{idx}_mean" for idx in range(max_states)]
    columns += [f"E{idx}_err" for idx in range(max_states)]
    return " ".join(columns)


def extract_shrinkage_lambda_from_message(message: str) -> float | None:
    match = re.search(r"shrinkage_lambda=([0-9]+(?:\.[0-9]+)?)", message)
    if match is None:
        return None
    return float(match.group(1))


def _state_output_paths(
    spec: NStateFitInput,
    title: str,
    nstates: int,
    tmax: int,
) -> tuple[Path, Path]:
    dataset_dir = spec.results_dir / title
    fit_path = dataset_dir / "tables" / f"{title}_{spec.model}_{nstates}state_tmax{tmax}_fits.txt"
    sample_path = dataset_dir / "samples" / f"{title}_{spec.model}_{nstates}state_tmax{tmax}_samples.txt"
    return fit_path, sample_path


def _fit_window_output_path(
    spec: NStateFitInput,
    title: str,
    nstates: int,
    tmax: int,
) -> Path:
    dataset_dir = spec.results_dir / title
    return dataset_dir / "tables" / f"{title}_{spec.model}_{nstates}state_tmax{tmax}_fit_window.txt"


def _write_state_outputs(
    spec: NStateFitInput,
    title: str,
    tmax: int,
    nstates: int,
    rows: list[FitSummaryRow],
    sample_tables: dict[int, np.ndarray],
    fit_window_summary: FitWindowSummary,
    fit_window_parameter_summary: FitWindowParameterSummary,
    meanfit_results: dict[int, FitResult],
) -> StateArtifacts:
    fit_window_start_row = next(row for row in rows if row.tmin == fit_window_summary.start_tmin)
    fit_window_start_shrinkage_lambda = extract_shrinkage_lambda_from_message(
        meanfit_results[fit_window_summary.start_tmin].message
    )
    table_path, sample_path = _state_output_paths(spec, title, nstates, tmax)
    fit_window_table_path = _fit_window_output_path(spec, title, nstates, tmax)

    np.savetxt(
        table_path,
        serialize_fit_rows(rows),
        header=header_for_fit_rows(nstates),
        fmt="%.10e",
    )

    fit_window_table = np.array(
        [[*fit_window_parameter_summary.params_mean, *fit_window_parameter_summary.params_err]],
        dtype=float,
    )
    np.savetxt(
        fit_window_table_path,
        fit_window_table,
        header=" ".join(
            [f"A{idx}_mean" for idx in range(nstates)]
            + [f"E{idx}_mean" for idx in range(nstates)]
            + [f"A{idx}_err" for idx in range(nstates)]
            + [f"E{idx}_err" for idx in range(nstates)]
        ),
        fmt="%.10e",
    )

    np.savetxt(
        sample_path,
        np.vstack([sample_tables[tmin] for tmin in sorted(sample_tables)]),
        header="tmin sample_id success chi2_dof pvalue " + " ".join(
            [f"A{idx}" for idx in range(nstates)] + [f"E{idx}" for idx in range(nstates)]
        ),
        fmt="%.10e",
    )

    return StateArtifacts(
        nstates=nstates,
        fit_window_summary=fit_window_summary,
        fit_window_parameter_summary=fit_window_parameter_summary,
        fit_window_start_fallback_uncorrelated_successes=fit_window_start_row.fallback_uncorrelated_successes,
        fit_window_start_shrinkage_lambda=fit_window_start_shrinkage_lambda,
        fit_table_path=table_path,
        fit_window_table_path=fit_window_table_path,
    )


@dataclass(frozen=True)
class _PreparedDataset:
    title: str
    tmin_window: tuple[int, int]
    time_offset: int
    selected: np.ndarray
    bootstrap_means: np.ndarray
    sigma: np.ndarray
    covariance: np.ndarray | None
    meff_mean: np.ndarray
    meff_err: np.ndarray
    fit_tmax_output: int
    dataset_dir: Path
    tables_dir: Path
    plots_dir: Path
    samples_dir: Path
    correlator_table: Path
    meff_table: Path
    scan_tmin_start: int
    scan_tmin_stop: int


def _load_and_preprocess_correlators(spec: NStateFitInput, pz: int) -> _PreparedDataset:
    title = spec.title_pattern.replace("*", str(pz))
    csv_path = spec.correlator_path_pattern.replace("*", str(pz))
    _, correlators = load_correlator_csv(csv_path)
    if correlators.shape[1] != spec.nt:
        raise ValueError(f"{csv_path} has Nt={correlators.shape[1]}, expected {spec.nt}")

    processed = apply_fold_t(correlators, spec.nt, spec.fold_t)
    if isinstance(spec.tmax, int):
        tmax_by_pz = {pz_value: spec.tmax for pz_value in spec.pzlist}
    else:
        tmax_by_pz = {
            pz_value: value
            for (gm, pz_value), value in load_int_mapping_table(spec.tmax).items()
            if gm is None
        }
    if pz not in tmax_by_pz:
        raise ValueError(f"missing tmax entry for pz={pz}")
    local_tmax = int(tmax_by_pz[pz])
    if local_tmax < 0:
        raise ValueError("tmax must be non-negative")
    if local_tmax >= processed.shape[1]:
        raise ValueError(
            f"tmax={local_tmax} exceeds the retained folded range of length {processed.shape[1]}"
        )
    processed = processed[:, : local_tmax + 1]
    tmin_windows = load_fit_window_table(spec.tmin_window)
    tmin_window = tmin_windows.get((None, pz))
    if tmin_window is None:
        raise ValueError(f"missing tmin_window entry for pz={pz}")
    fit_start, fit_end = tmin_window
    if fit_start < 0:
        raise ValueError(f"tmin_window for pz={pz} must start at or above 0")
    if fit_end > local_tmax:
        raise ValueError(f"tmin_window for pz={pz} ends at {fit_end}, which exceeds tmax={local_tmax}")
    selected = processed
    time_offset = 0
    binned = bin_correlators(selected, binsize=spec.binsize)
    bootstrap_means = bootstrap_correlator_means(
        binned,
        n_samples=spec.bootstrap_samples,
        sample_size=spec.bootstrap_size,
        seed=spec.seed,
    )
    sigma = np.nanstd(bootstrap_means, axis=0, ddof=1)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, MIN_POSITIVE)
    covariance = compute_bootstrap_covariance(bootstrap_means) if spec.fit_mode == "correlated" else None

    meff_mean, meff_err = effective_mass_with_bootstrap(bootstrap_means, spec.model, nt=spec.nt)
    fit_tmax_output = local_tmax

    dataset_dir = spec.results_dir / title
    tables_dir = dataset_dir / "tables"
    plots_dir = dataset_dir / "plots"
    samples_dir = dataset_dir / "samples"
    for directory in (tables_dir, plots_dir, samples_dir):
        directory.mkdir(parents=True, exist_ok=True)

    correlator_table = tables_dir / f"{title}_{spec.model}_correlator_mean.txt"
    meff_table = tables_dir / f"{title}_{spec.model}_effective_mass_tmax{fit_tmax_output}.txt"
    return _PreparedDataset(
        title=title,
        tmin_window=tmin_window,
        time_offset=0,
        selected=selected,
        bootstrap_means=bootstrap_means,
        sigma=sigma,
        covariance=covariance,
        meff_mean=meff_mean,
        meff_err=meff_err,
        fit_tmax_output=fit_tmax_output,
        dataset_dir=dataset_dir,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
        samples_dir=samples_dir,
        correlator_table=correlator_table,
        meff_table=meff_table,
        scan_tmin_start=fit_start,
        scan_tmin_stop=fit_end,
    )


def _initialize_fit_guesses(
    bootstrap_means: np.ndarray,
    fixed_ground_energy: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    one_state_energy_guess = 0.01 if fixed_ground_energy is None else float(fixed_ground_energy)
    sample_index = 2 if bootstrap_means.shape[1] > 2 else max(0, bootstrap_means.shape[1] - 1)
    one_state_amplitude_guess = max(
        bootstrap_means.mean(axis=0)[sample_index] * np.exp(one_state_energy_guess * sample_index),
        MIN_POSITIVE,
    )
    return (
        np.array([one_state_amplitude_guess]),
        np.array([one_state_energy_guess]),
    )


def _run_state_fits(
    spec: NStateFitInput,
    pz: int,
    dataset: _PreparedDataset,
    initial_guess: tuple[np.ndarray, np.ndarray],
) -> dict[int, StateArtifacts]:
    current_guess = initial_guess
    state_artifacts: dict[int, StateArtifacts] = {}

    def compute_state(nstates: int) -> None:
        nonlocal current_guess
        if nstates in state_artifacts:
            return

        if nstates == 1:
            tmin_start = dataset.scan_tmin_start
            tmin_stop = dataset.scan_tmin_stop + 1
            state_initial_guess = current_guess
            priors: tuple[EnergyPrior, ...] = ()
        else:
            previous_state = nstates - 1
            previous_artifact = state_artifacts.get(previous_state)
            if previous_artifact is None:
                compute_state(previous_state)
                previous_artifact = state_artifacts[previous_state]

            tmin_start = dataset.scan_tmin_start
            tmin_stop = dataset.scan_tmin_stop + 1
            state_initial_guess = build_initial_guess_from_fit_window(previous_artifact.fit_window_summary, nstates)
            priors = build_energy_priors_from_fit_window_summary(
                previous_artifact.fit_window_parameter_summary,
                nstates,
            )
            current_guess = state_initial_guess

        fixed_ground_energy = None
        if spec.fix_ground_energy_from_dispersion and spec.pz0_ground_energy is not None:
            fixed_ground_energy = target_ground_energy_from_pz0(spec.pz0_ground_energy, pz, spec.ns)
            if state_initial_guess[1].size > 0:
                fixed_energies = np.asarray(state_initial_guess[1], dtype=float).copy()
                fixed_energies[0] = fixed_ground_energy
                state_initial_guess = (np.asarray(state_initial_guess[0], dtype=float), fixed_energies)

        rows, sample_tables, meanfit_results = run_sliding_fits(
            dataset.bootstrap_means,
            dataset.sigma,
            spec.fit_mode,
            spec.nt,
            spec.model,
            nstates,
            range(tmin_start, tmin_stop),
            dataset.fit_tmax_output,
            state_initial_guess[0],
            state_initial_guess[1],
            covariance=dataset.covariance,
            priors=priors,
            lambda_prior=spec.lambda_prior,
            fixed_ground_energy=fixed_ground_energy,
            time_offset=0,
        )
        representative_row = next((row for row in reversed(rows) if row.success_meanfit), rows[0] if rows else None)
        if representative_row is None:
            raise ValueError(f"fit window for pz={pz} produced no fit row")
        fit_window_summary = FitWindowSummary(
            start_tmin=dataset.scan_tmin_start,
            end_tmin=dataset.scan_tmin_stop,
            representative_tmin=representative_row.tmin,
            energy_mean=representative_row.params_mean[nstates],
            amplitude_mean=representative_row.params_mean[0],
        )
        rows = mark_fit_window(rows, fit_window_summary)
        fit_window_parameter_summary = compute_fit_window_parameter_summary(
            rows, sample_tables, fit_window_summary
        )
        state_artifacts[nstates] = _write_state_outputs(
            spec=spec,
            title=dataset.title,
            tmax=dataset.fit_tmax_output,
            nstates=nstates,
            rows=rows,
            sample_tables=sample_tables,
            fit_window_summary=fit_window_summary,
            fit_window_parameter_summary=fit_window_parameter_summary,
            meanfit_results=meanfit_results,
        )

    for nstates in spec.nstates:
        compute_state(nstates)
    return state_artifacts


def _write_dataset_outputs(
    spec: NStateFitInput,
    pz: int,
    dataset: _PreparedDataset,
    state_artifacts: dict[int, StateArtifacts],
) -> list[Path]:
    outputs: list[Path] = []
    np.savetxt(
        dataset.correlator_table,
        np.column_stack(
            [
                np.arange(dataset.time_offset, dataset.time_offset + dataset.selected.shape[1]),
                np.mean(dataset.bootstrap_means, axis=0),
                dataset.sigma,
            ]
        ),
        header="t corr_mean corr_err",
        fmt="%.10e",
    )
    outputs.append(dataset.correlator_table)
    np.savetxt(
        dataset.meff_table,
        np.column_stack([np.arange(dataset.time_offset, dataset.time_offset + len(dataset.meff_mean)), dataset.meff_mean, dataset.meff_err]),
        header="t meff_mean meff_err",
        fmt="%.10e",
    )
    outputs.append(dataset.meff_table)

    summary_path = dataset.dataset_dir / f"{dataset.title}_{spec.model}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"title {dataset.title}\n")
        handle.write(f"model {spec.model}\n")
        handle.write(f"fit_mode {spec.fit_mode}\n")
        handle.write(f"tmax {dataset.fit_tmax_output}\n")
        handle.write(f"tmin_window {dataset.scan_tmin_start} {dataset.scan_tmin_stop}\n")
        if spec.pz0_ground_energy is not None:
            handle.write(f"pz0_ground_energy {spec.pz0_ground_energy:.10e}\n")
            handle.write(
                f"1state_target_energy_pz{pz} {target_ground_energy_from_pz0(spec.pz0_ground_energy, pz, spec.ns):.10e}\n"
            )
        handle.write(
            f"fix_ground_energy_from_dispersion {int(spec.fix_ground_energy_from_dispersion)}\n"
        )
        for nstates in sorted(state_artifacts):
            artifact = state_artifacts[nstates]
            fit_window_summary = artifact.fit_window_summary
            fit_window_parameter_summary = artifact.fit_window_parameter_summary
            handle.write(
                f"{nstates}state fit_window_tmin {fit_window_summary.start_tmin} {fit_window_summary.end_tmin}\n"
            )
            handle.write(
                f"{nstates}state fit_window_start_fallback_uncorrelated_successes "
                f"{artifact.fit_window_start_fallback_uncorrelated_successes}\n"
            )
            fit_window_start_lambda = artifact.fit_window_start_shrinkage_lambda
            if fit_window_start_lambda is not None:
                handle.write(f"{nstates}state fit_window_start_shrinkage_lambda {fit_window_start_lambda:.2f}\n")
            for idx in range(nstates):
                handle.write(
                    f"{nstates}state A{idx} {fit_window_parameter_summary.params_mean[idx]:.10e} "
                    f"{fit_window_parameter_summary.params_err[idx]:.10e} "
                    f"E{idx} {fit_window_parameter_summary.params_mean[nstates + idx]:.10e} "
                    f"{fit_window_parameter_summary.params_err[nstates + idx]:.10e}\n"
                )
    outputs.append(summary_path)

    if spec.make_plots:
        for nstates in sorted(state_artifacts):
            artifact = state_artifacts[nstates]
            outputs.extend(
                plot_nstate_outputs(
                    output_dir=dataset.plots_dir,
                    correlator_table=dataset.correlator_table,
                    meff_table=dataset.meff_table,
                    fit_table=artifact.fit_table_path,
                    nstates=nstates,
                    model=spec.model,
                    title=dataset.title,
                    nt=spec.nt,
                    lattice_spacing_fm=spec.lattice_spacing_fm,
                )
            )

    notebook_root = spec.results_dir / "notebook_plots" / dataset.title
    notebook_generated = notebook_root / "generated_plots"
    notebook_path = notebook_root / f"{dataset.title}_{spec.model}_nstate_plots.ipynb"
    outputs.append(
        write_nstate_plot_notebook(
            notebook_path=notebook_path,
            notebook_output_dir=notebook_generated,
            correlator_table=dataset.correlator_table,
            meff_table=dataset.meff_table,
            fit_tables={nstates: state_artifacts[nstates].fit_table_path for nstates in sorted(state_artifacts)},
            model=spec.model,
            title=dataset.title,
            nt=spec.nt,
            lattice_spacing_fm=spec.lattice_spacing_fm,
        )
    )
    return outputs


def run_single_dataset(spec: NStateFitInput, pz: int) -> list[Path]:
    dataset = _load_and_preprocess_correlators(spec, pz)
    fixed_ground_energy = None
    if spec.fix_ground_energy_from_dispersion and spec.pz0_ground_energy is not None:
        fixed_ground_energy = target_ground_energy_from_pz0(spec.pz0_ground_energy, pz, spec.ns)
    initial_guess = _initialize_fit_guesses(dataset.bootstrap_means, fixed_ground_energy)
    state_artifacts = _run_state_fits(spec, pz, dataset, initial_guess)
    return _write_dataset_outputs(spec, pz, dataset, state_artifacts)


def run_nstate_fit(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_nstate_fit_input(input_file, results_dir=results_dir)
    outputs: list[Path] = []
    for pz in spec.pzlist:
        outputs.extend(run_single_dataset(spec, pz))
    return outputs
