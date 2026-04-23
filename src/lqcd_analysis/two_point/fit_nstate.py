from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from ..common.constants import (
    MIN_AMPLITUDE,
    MIN_POSITIVE,
    SHRINKAGE_LAMBDAS,
)
from .io import load_correlator_csv
from .plotting import plot_nstate_outputs, write_nstate_plot_notebook
from ..common.bootstrap import (
    compute_bootstrap_covariance,
    shrink_covariance_to_diagonal,
)
from ..common.parsing import (
    load_control_file_entries,
    load_fit_window_table,
    load_int_mapping_table,
    parse_bool,
    parse_fold_t,
    parse_optional_int,
)
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
    low_state_prior_tmin: str | int | None
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
    params_mean: tuple[float, ...]
    params_err: tuple[float, ...]
    initial_amplitudes: tuple[float, ...] = ()
    initial_energies: tuple[float, ...] = ()
    shrinkage_lambda: float | None = None


@dataclass(frozen=True)
class FitWindowSummary:
    start_tmin: int
    end_tmin: int
    energy_mean: float
    amplitude_mean: float


@dataclass(frozen=True)
class StateArtifacts:
    fit_window_summary: FitWindowSummary
    fit_window_start_fallback_uncorrelated_successes: int
    fit_window_start_shrinkage_lambda: float | None
    fit_table_path: Path
    fit_rows: tuple[FitSummaryRow, ...] = ()


@dataclass(frozen=True)
class FitResult:
    params: np.ndarray
    chi2: float
    chi2_dof: float
    pvalue: float
    success: bool
    message: str
    used_uncorrelated_fallback: bool = False
    shrinkage_lambda: float | None = None


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
    shrinkage_lambda: float | None = None


def parse_nstate_fit_input(path: str | Path, results_dir: str | Path | None = None) -> NStateFitInput:
    file_path = Path(path)
    first_tokens, entries = load_control_file_entries(file_path)
    if len(first_tokens) < 4:
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
    default_results_dir = input_path.parent / (
        "results_nstate_fit" if max(nstates) <= 1 else f"results_nstate_fit_{max(nstates)}state"
    )

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
        low_state_prior_tmin=(
            None
            if "low_state_prior_tmin" not in entries
            else (
                int(entries["low_state_prior_tmin"][0])
                if entries["low_state_prior_tmin"][0].strip().lstrip("-").isdigit()
                else entries["low_state_prior_tmin"][0]
            )
        ),
        lambda_prior=float(entries.get("lambda_prior", ["1.0"])[0]),
        make_plots=parse_bool(entries.get("plot", ["true"])[0]),
        results_dir=(
            Path(entries["results_dir"][0])
            if "results_dir" in entries and results_dir is None
            else (default_results_dir if results_dir is None else Path(results_dir))
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
    return ResidualModel(
        fit_mode="correlated",
        cholesky_factor=cholesky_factor,
        shrinkage_lambda=shrinkage_lambda,
    )


def find_first_usable_correlated_residual_model(
    sigma: np.ndarray,
    covariance: np.ndarray,
) -> tuple[ResidualModel | None, float | None]:
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
            max_nfev=2000,
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
            shrinkage_lambda=(
                active_model.shrinkage_lambda
                if active_model is not None and active_model.fit_mode == "correlated"
                else None
            ),
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
            shrinkage_lambda=None,
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
                    shrinkage_lambda=shrinkage_lambda,
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


def select_window_reference_row(rows: list[FitSummaryRow]) -> FitSummaryRow:
    successful_rows = [row for row in rows if row.success_meanfit]
    if successful_rows:
        return min(successful_rows, key=lambda row: row.chi2_dof)
    if rows:
        return rows[0]
    raise ValueError("fit window for pz produced no fit row")


def load_fit_row_from_table(
    table_path: str | Path,
    *,
    tmin: int,
    nstates: int,
) -> FitSummaryRow:
    rows = np.atleast_2d(np.loadtxt(table_path, dtype=float))
    matching_rows = rows[rows[:, 0] == float(tmin)]
    if matching_rows.shape[0] != 1:
        raise ValueError(f"expected exactly one fit row at tmin={tmin} in {table_path}")

    row = matching_rows[0]
    if row.shape[0] < 9 + 4 * nstates:
        raise ValueError(
            f"fit table row at tmin={tmin} in {table_path} has too few columns for nstates={nstates}"
        )
    amp_mean_start = 9
    amp_err_start = amp_mean_start + nstates
    energy_mean_start = amp_err_start + nstates
    energy_err_start = energy_mean_start + nstates
    return FitSummaryRow(
        nstates=nstates,
        tmin=int(row[0]),
        tmax=int(row[1]),
        success_meanfit=int(row[2]),
        bootstrap_successes=int(row[3]),
        bootstrap_total=int(row[4]),
        bootstrap_success_fraction=float(row[5]),
        fallback_uncorrelated_successes=int(row[6]),
        chi2_dof=float(row[7]),
        pvalue=float(row[8]),
        params_mean=tuple(row[amp_mean_start : amp_mean_start + nstates]) + tuple(
            row[energy_mean_start : energy_mean_start + nstates]
        ),
        params_err=tuple(row[amp_err_start : amp_err_start + nstates]) + tuple(
            row[energy_err_start : energy_err_start + nstates]
        ),
        initial_amplitudes=(),
        initial_energies=(),
        shrinkage_lambda=None,
    )


def load_fit_rows_from_table(table_path: str | Path, nstates: int) -> list[FitSummaryRow]:
    rows = np.atleast_2d(np.loadtxt(table_path, dtype=float))
    if rows.size == 0:
        raise ValueError(f"fit table is empty: {table_path}")
    parsed_rows: list[FitSummaryRow] = []
    for row in rows:
        if row.shape[0] < 9 + 4 * nstates:
            raise ValueError(
                f"fit table row at tmin={int(row[0])} in {table_path} has too few columns for nstates={nstates}"
            )
        amp_mean_start = 9
        amp_err_start = amp_mean_start + nstates
        energy_mean_start = amp_err_start + nstates
        energy_err_start = energy_mean_start + nstates
        parsed_rows.append(
            FitSummaryRow(
                nstates=nstates,
                tmin=int(row[0]),
                tmax=int(row[1]),
                success_meanfit=int(row[2]),
                bootstrap_successes=int(row[3]),
                bootstrap_total=int(row[4]),
                bootstrap_success_fraction=float(row[5]),
                fallback_uncorrelated_successes=int(row[6]),
                chi2_dof=float(row[7]),
                pvalue=float(row[8]),
                params_mean=tuple(row[amp_mean_start : amp_mean_start + nstates]) + tuple(
                    row[energy_mean_start : energy_mean_start + nstates]
                ),
                params_err=tuple(row[amp_err_start : amp_err_start + nstates]) + tuple(
                    row[energy_err_start : energy_err_start + nstates]
                ),
                initial_amplitudes=(),
                initial_energies=(),
                shrinkage_lambda=None,
            )
        )
    return parsed_rows


def load_low_state_prior_tmin(
    prior_source: str | int | Path,
    *,
    pz: int,
) -> int:
    if isinstance(prior_source, int):
        return prior_source
    path = Path(prior_source)
    if path.is_file():
        mapping = load_int_mapping_table(path)
        key = (None, pz)
        if key in mapping:
            return int(mapping[key])
        raise ValueError(f"missing low_state_prior_tmin entry for pz={pz} in {path}")
    return int(prior_source)


def format_low_state_prior_tmin_for_summary(prior_source: str | int | Path) -> str:
    if isinstance(prior_source, int):
        return str(prior_source)
    path = Path(prior_source)
    if path.is_file():
        mapping = load_int_mapping_table(path)
        pairs = sorted((int(pz), int(value)) for (row_index, pz), value in mapping.items() if row_index is None)
        return "{" + ", ".join(f"{pz}: {value}" for pz, value in pairs) + "}"
    return str(prior_source)


def format_initial_guess_for_summary(row: FitSummaryRow) -> str:
    if not row.initial_amplitudes and not row.initial_energies:
        return ""
    amplitude_text = " ".join(f"A{idx}={value:.10e}" for idx, value in enumerate(row.initial_amplitudes))
    energy_text = " ".join(f"E{idx}={value:.10e}" for idx, value in enumerate(row.initial_energies))
    pieces = [piece for piece in (amplitude_text, energy_text) if piece]
    return " ".join(pieces)


def build_energy_priors_from_fit_row(
    fit_row: FitSummaryRow | None,
    nstates: int,
) -> tuple[EnergyPrior, ...]:
    if fit_row is None or nstates <= 1:
        return ()
    previous_nstates = fit_row.nstates
    max_index = min(previous_nstates, nstates - 1)
    priors: list[EnergyPrior] = []
    for energy_index in range(max_index):
        center = fit_row.params_mean[previous_nstates + energy_index]
        sigma = fit_row.params_err[previous_nstates + energy_index]
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


def _previous_state_results_dir(results_dir: Path, nstates: int) -> Path:
    current_state_label = f"{nstates}state"
    previous_state_label = f"{nstates - 1}state"
    if current_state_label not in results_dir.name:
        raise ValueError(
            f"cannot infer previous-state results_dir from {results_dir}; expected name containing {current_state_label}"
        )
    return results_dir.with_name(results_dir.name.replace(current_state_label, previous_state_label, 1))


def _load_previous_state_artifact(
    spec: NStateFitInput,
    title: str,
    *,
    current_nstates: int,
    tmax: int,
) -> StateArtifacts:
    previous_nstates = current_nstates - 1
    previous_results_dir = _previous_state_results_dir(spec.results_dir, current_nstates)
    fit_table_path = (
        previous_results_dir / title / "tables" / f"{title}_{spec.model}_{previous_nstates}state_tmax{tmax}_fits.txt"
    )
    rows = load_fit_rows_from_table(fit_table_path, previous_nstates)
    window_reference_row = select_window_reference_row(rows)
    fit_window_summary = FitWindowSummary(
        start_tmin=min(row.tmin for row in rows),
        end_tmin=max(row.tmin for row in rows),
        energy_mean=window_reference_row.params_mean[previous_nstates],
        amplitude_mean=window_reference_row.params_mean[0],
    )
    fit_window_start_row = next(row for row in rows if row.tmin == fit_window_summary.start_tmin)
    return StateArtifacts(
        fit_window_summary=fit_window_summary,
        fit_window_start_fallback_uncorrelated_successes=fit_window_start_row.fallback_uncorrelated_successes,
        fit_window_start_shrinkage_lambda=fit_window_start_row.shrinkage_lambda,
        fit_table_path=fit_table_path,
        fit_rows=tuple(rows),
    )


def _prepare_fit_window_data(
    mean_correlator: np.ndarray,
    sigma: np.ndarray,
    covariance: np.ndarray | None,
    tmin: int,
    tmax: int,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    label_tmin = int(tmin)
    label_tmax = int(tmax)
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
    meanfit_amplitudes: np.ndarray,
    meanfit_energies: np.ndarray,
    *,
    nt: int,
    model: str,
    nstates: int,
    priors: tuple[EnergyPrior, ...],
    lambda_prior: float,
    covariance_slice: np.ndarray | None,
    fixed_ground_energy: float | None,
) -> FitResult:
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


def _select_meanfit_warm_start_tmin(tmin_values: range) -> int:
    tmin_list = list(tmin_values)
    if not tmin_list:
        raise ValueError("tmin_values must not be empty")
    return tmin_list[min(len(tmin_list) - 1, len(tmin_list) // 3)]


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
) -> tuple[list[FitSummaryRow], dict[int, np.ndarray], dict[int, FitResult]]:
    rows: list[FitSummaryRow] = []
    sample_tables: dict[int, np.ndarray] = {}
    meanfit_warm_start_tmin = _select_meanfit_warm_start_tmin(tmin_values)

    mean_correlator = np.mean(bootstrap_means, axis=0)
    meanfit_reference: FitResult | None = None
    previous_sample_fits: list[FitResult | None] = [None] * len(bootstrap_means)
    for tmin in tmin_values:
        label_tmin, label_tmax, times, data_mean, sigma_slice, covariance_slice = _prepare_fit_window_data(
            mean_correlator,
            sigma,
            covariance,
            tmin,
            tmax,
        )
        residual_model, residual_model_note = _build_window_residual_model(
            fit_mode,
            sigma_slice,
            covariance_slice,
            label_tmin=label_tmin,
            label_tmax=label_tmax,
        )
        if tmin >= meanfit_warm_start_tmin and meanfit_reference is not None and meanfit_reference.success and np.all(
            np.isfinite(meanfit_reference.params)
        ):
            meanfit_initial_amplitudes = meanfit_reference.params[:nstates]
            meanfit_initial_energies = meanfit_reference.params[nstates:]
        else:
            meanfit_initial_amplitudes = initial_amplitudes
            meanfit_initial_energies = initial_energies
        meanfit = _run_mean_window_fit(
            times,
            data_mean,
            sigma_slice,
            residual_model,
            meanfit_initial_amplitudes,
            meanfit_initial_energies,
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
                used_uncorrelated_fallback=meanfit.used_uncorrelated_fallback,
                shrinkage_lambda=meanfit.shrinkage_lambda,
            )
        if tmin == meanfit_warm_start_tmin and meanfit.success and np.all(np.isfinite(meanfit.params)):
            meanfit_reference = meanfit

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
                params_mean=params_mean,
                params_err=params_err,
                initial_amplitudes=tuple(np.asarray(meanfit_initial_amplitudes, dtype=float)),
                initial_energies=tuple(np.asarray(meanfit_initial_energies, dtype=float)),
                shrinkage_lambda=meanfit.shrinkage_lambda,
            )
        )

    return rows, sample_tables, {}


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
    ]
    columns += [f"A{idx}_mean" for idx in range(max_states)]
    columns += [f"A{idx}_err" for idx in range(max_states)]
    columns += [f"E{idx}_mean" for idx in range(max_states)]
    columns += [f"E{idx}_err" for idx in range(max_states)]
    return " ".join(columns)


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


def _write_state_outputs(
    spec: NStateFitInput,
    title: str,
    tmax: int,
    nstates: int,
    rows: list[FitSummaryRow],
    sample_tables: dict[int, np.ndarray],
    fit_window_summary: FitWindowSummary,
) -> StateArtifacts:
    fit_window_start_row = next(row for row in rows if row.tmin == fit_window_summary.start_tmin)
    fit_window_start_shrinkage_lambda = fit_window_start_row.shrinkage_lambda
    table_path, sample_path = _state_output_paths(spec, title, nstates, tmax)

    np.savetxt(
        table_path,
        serialize_fit_rows(rows),
        header=header_for_fit_rows(nstates),
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
        fit_window_summary=fit_window_summary,
        fit_window_start_fallback_uncorrelated_successes=fit_window_start_row.fallback_uncorrelated_successes,
        fit_window_start_shrinkage_lambda=fit_window_start_shrinkage_lambda,
        fit_table_path=table_path,
        fit_rows=tuple(rows),
    )


@dataclass(frozen=True)
class _PreparedDataset:
    title: str
    dataset_dir: Path
    plots_dir: Path
    bootstrap_means: np.ndarray
    sigma: np.ndarray
    covariance: np.ndarray | None
    meff_mean: np.ndarray
    meff_err: np.ndarray
    fit_tmax_output: int
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
    binned = bin_correlators(processed, binsize=spec.binsize)
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
    plots_dir = dataset_dir / "plots"
    tables_dir = dataset_dir / "tables"
    samples_dir = dataset_dir / "samples"
    for directory in (tables_dir, plots_dir, samples_dir):
        directory.mkdir(parents=True, exist_ok=True)

    correlator_table = tables_dir / f"{title}_{spec.model}_correlator_mean.txt"
    meff_table = tables_dir / f"{title}_{spec.model}_effective_mass_tmax{fit_tmax_output}.txt"
    return _PreparedDataset(
        title=title,
        dataset_dir=dataset_dir,
        plots_dir=plots_dir,
        bootstrap_means=bootstrap_means,
        sigma=sigma,
        covariance=covariance,
        meff_mean=meff_mean,
        meff_err=meff_err,
        fit_tmax_output=fit_tmax_output,
        correlator_table=correlator_table,
        meff_table=meff_table,
        scan_tmin_start=fit_start,
        scan_tmin_stop=fit_end,
    )


def _initialize_fit_guesses(
    bootstrap_means: np.ndarray,
    initial_ground_energy: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    one_state_energy_guess = 0.1 if initial_ground_energy is None else float(initial_ground_energy)
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
    initial_ground_energy: float | None,
    fixed_ground_energy: float | None,
) -> dict[int, StateArtifacts]:
    state_artifacts: dict[int, StateArtifacts] = {}

    def compute_state(nstates: int) -> None:
        if nstates in state_artifacts:
            return

        if nstates == 1:
            tmin_start = dataset.scan_tmin_start
            tmin_stop = dataset.scan_tmin_stop + 1
            state_initial_guess = initial_guess
            priors: tuple[EnergyPrior, ...] = ()
        else:
            tmin_start = dataset.scan_tmin_start
            tmin_stop = dataset.scan_tmin_stop + 1
            previous_artifact = _load_previous_state_artifact(
                spec,
                dataset.title,
                current_nstates=nstates,
                tmax=dataset.fit_tmax_output,
            )
            state_initial_guess = build_initial_guess_from_fit_window(previous_artifact.fit_window_summary, nstates)
            prior_row = None
            if spec.low_state_prior_tmin is not None:
                prior_tmin = load_low_state_prior_tmin(
                    spec.low_state_prior_tmin,
                    pz=pz,
                )
                prior_row = load_fit_row_from_table(
                    previous_artifact.fit_table_path,
                    tmin=prior_tmin,
                    nstates=nstates - 1,
                )
            priors = build_energy_priors_from_fit_row(prior_row, nstates)

        if initial_ground_energy is not None and state_initial_guess[1].size > 0:
            initial_energies = np.asarray(state_initial_guess[1], dtype=float).copy()
            initial_energies[0] = initial_ground_energy
            state_initial_guess = (np.asarray(state_initial_guess[0], dtype=float), initial_energies)

        if fixed_ground_energy is not None and state_initial_guess[1].size > 0:
            fixed_energies = np.asarray(state_initial_guess[1], dtype=float).copy()
            fixed_energies[0] = fixed_ground_energy
            state_initial_guess = (np.asarray(state_initial_guess[0], dtype=float), fixed_energies)

        rows, sample_tables, _ = run_sliding_fits(
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
        )
        window_reference_row = select_window_reference_row(rows)
        fit_window_summary = FitWindowSummary(
            start_tmin=dataset.scan_tmin_start,
            end_tmin=dataset.scan_tmin_stop,
            energy_mean=window_reference_row.params_mean[nstates],
            amplitude_mean=window_reference_row.params_mean[0],
        )
        state_artifacts[nstates] = _write_state_outputs(
            spec=spec,
            title=dataset.title,
            tmax=dataset.fit_tmax_output,
            nstates=nstates,
            rows=rows,
            sample_tables=sample_tables,
            fit_window_summary=fit_window_summary,
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
                np.arange(dataset.bootstrap_means.shape[1]),
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
        np.column_stack([np.arange(len(dataset.meff_mean)), dataset.meff_mean, dataset.meff_err]),
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
            if spec.low_state_prior_tmin is not None and nstates > 1:
                prior_tmin_text = format_low_state_prior_tmin_for_summary(spec.low_state_prior_tmin)
                handle.write(
                    f"{nstates}state low_state_prior_tmin {prior_tmin_text}\n"
                )
            if artifact.fit_rows:
                handle.write(f"{nstates}state initial_guess_tmin_rows\n")
            for row in artifact.fit_rows:
                initial_guess_text = format_initial_guess_for_summary(row)
                if initial_guess_text:
                    handle.write(
                        f"  tmin {row.tmin}: {initial_guess_text}\n"
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
    initial_ground_energy = None
    if spec.pz0_ground_energy is not None:
        initial_ground_energy = target_ground_energy_from_pz0(spec.pz0_ground_energy, pz, spec.ns)
    fixed_ground_energy = None
    if spec.fix_ground_energy_from_dispersion and spec.pz0_ground_energy is not None:
        fixed_ground_energy = initial_ground_energy
    initial_guess = _initialize_fit_guesses(dataset.bootstrap_means, initial_ground_energy)
    state_artifacts = _run_state_fits(
        spec,
        pz,
        dataset,
        initial_guess,
        initial_ground_energy,
        fixed_ground_energy,
    )
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
