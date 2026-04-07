from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from .io import load_correlator_csv
from .plotting_2pt import plot_nstate_outputs, write_nstate_plot_notebook
from .utils import (
    apply_fold_t,
    bin_correlators,
    bootstrap_correlator_means,
    parse_bool,
    parse_fold_t,
    robust_mean_and_error,
)


@dataclass(frozen=True)
class NStateFitInput:
    title_pattern: str
    ns: int
    nt: int
    lattice_spacing_fm: float
    correlator_path_pattern: str
    pzlist: tuple[int, ...]
    fold_t: str
    tsrange: tuple[int, int]
    model: str
    fit_mode: str
    nstates: tuple[int, ...]
    tmax: int | None
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
    plateau_flag: int
    params_mean: tuple[float, ...]
    params_err: tuple[float, ...]


@dataclass(frozen=True)
class PlateauWindow:
    start_tmin: int
    end_tmin: int
    representative_tmin: int
    energy_mean: float
    amplitude_mean: float


@dataclass(frozen=True)
class PlateauCandidate:
    start_tmin: int
    end_tmin: int
    representative_tmin: int
    constant_chi2_dof: float
    slope: float
    slope_err: float
    slope_significance: float


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

    required = {"c2pt", "pzlist", "tsrange", "model", "nstates"}
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

    input_path = file_path.resolve()
    return NStateFitInput(
        title_pattern=first_tokens[0],
        ns=int(first_tokens[1]),
        nt=int(first_tokens[2]),
        lattice_spacing_fm=float(first_tokens[3]),
        correlator_path_pattern=entries["c2pt"][0],
        pzlist=tuple(int(item) for item in entries["pzlist"]),
        fold_t=parse_fold_t(entries),
        tsrange=(int(entries["tsrange"][0]), int(entries["tsrange"][1])),
        model=model,
        fit_mode=fit_mode,
        nstates=nstates,
        tmax=parse_optional_int(entries.get("tmax", ["auto"])[0]),
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


def parse_optional_int(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    return int(value)


def compute_bootstrap_covariance(bootstrap_means: np.ndarray) -> np.ndarray:
    values = np.asarray(bootstrap_means, dtype=float)
    if values.ndim != 2:
        raise ValueError("bootstrap_means must be a 2D array")
    covariance = np.cov(values, rowvar=False, ddof=1)
    return np.atleast_2d(np.asarray(covariance, dtype=float))


def build_residual_model(
    fit_mode: str,
    sigma: np.ndarray,
    covariance: np.ndarray | None = None,
) -> ResidualModel:
    sigma = np.asarray(sigma, dtype=float)
    if fit_mode == "uncorrelated":
        return ResidualModel(fit_mode="uncorrelated", sigma=np.clip(sigma, 1e-12, None))
    if covariance is None:
        raise ValueError("covariance is required for correlated fitting")
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    cholesky_factor = np.linalg.cholesky(covariance)
    fallback_sigma = np.sqrt(np.clip(np.diag(covariance), 1e-24, None))
    return ResidualModel(
        fit_mode="correlated",
        cholesky_factor=cholesky_factor,
        fallback_sigma=np.clip(fallback_sigma, 1e-12, None),
    )


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


def solve_cosh_effective_mass(
    t: int,
    ratio: float,
    nt: int,
    lower: float = 2e-5,
    upper: float = 20.0,
    tol: float = 1e-8,
    max_iter: int = 256,
) -> float:
    """Solve the periodic cosh effective-mass equation by bisection.

    We solve

        ratio * cosh(m * (t + 1 - Nt/2)) - cosh(m * (t - Nt/2)) = 0

    for m, where ratio = C(t) / C(t+1). This is more explicit than relying only
    on a local arccosh identity and remains consistent with the periodic/cosh
    fit model used elsewhere in the workflow.

    If the ratio is invalid, the bracket is unsafe, or no root is found
    robustly, the function returns np.nan.
    """
    if nt <= 0 or t < 0 or not np.isfinite(ratio) or ratio <= 0.0:
        return np.nan
    if lower <= 0.0 or upper <= lower:
        return np.nan

    center = 0.5 * nt

    def f(mass: float) -> float:
        return ratio * np.cosh(mass * (t + 1 - center)) - np.cosh(mass * (t - center))

    f_lower = f(lower)
    f_upper = f(upper)
    if not np.isfinite(f_lower) or not np.isfinite(f_upper):
        return np.nan
    if f_lower == 0.0:
        return float(lower)
    if f_upper == 0.0:
        return float(upper)
    if f_lower * f_upper > 0.0:
        return np.nan

    lo = float(lower)
    hi = float(upper)
    flo = float(f_lower)
    fhi = float(f_upper)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if not np.isfinite(fmid):
            return np.nan
        if abs(fmid) < tol or 0.5 * (hi - lo) < tol:
            return float(mid)
        if flo * fmid <= 0.0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return float(0.5 * (lo + hi)) if np.isfinite(flo) and np.isfinite(fhi) else np.nan


def solve_antisymmetric_effective_mass(
    t: int,
    ratio: float,
    nt: int,
    lower: float = 2e-5,
    upper: float = 20.0,
    tol: float = 1e-8,
    max_iter: int = 256,
) -> float:
    """Solve the antisymmetric/sinh effective-mass equation by bisection.

    For an antisymmetric correlator with time dependence

        C(t) ~ exp(-m t) - exp(-m (Nt - t))

    the ratio R(t) = C(t) / C(t+1) satisfies

        R(t) * sinh(m * (t + 1 - Nt/2)) - sinh(m * (t - Nt/2)) = 0.

    This equation is distinct from the symmetric/cosh case and should not be
    treated as a reuse of the symmetric local estimator. Failed solves and
    invalid ratios return np.nan.
    """
    if nt <= 0 or t < 0 or not np.isfinite(ratio) or ratio <= 0.0:
        return np.nan
    if lower <= 0.0 or upper <= lower:
        return np.nan

    center = 0.5 * nt

    def f(mass: float) -> float:
        return ratio * np.sinh(mass * (t + 1 - center)) - np.sinh(mass * (t - center))

    f_lower = f(lower)
    f_upper = f(upper)
    if not np.isfinite(f_lower) or not np.isfinite(f_upper):
        return np.nan
    if f_lower == 0.0:
        return float(lower)
    if f_upper == 0.0:
        return float(upper)
    if f_lower * f_upper > 0.0:
        return np.nan

    lo = float(lower)
    hi = float(upper)
    flo = float(f_lower)
    fhi = float(f_upper)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if not np.isfinite(fmid):
            return np.nan
        if abs(fmid) < tol or 0.5 * (hi - lo) < tol:
            return float(mid)
        if flo * fmid <= 0.0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return float(0.5 * (lo + hi)) if np.isfinite(flo) and np.isfinite(fhi) else np.nan


def compute_effective_mass_cosh_root(
    values: np.ndarray,
    nt: int,
    lower: float = 2e-5,
    upper: float = 20.0,
    tol: float = 1e-8,
) -> np.ndarray:
    """Compute m_eff(t) from C(t)/C(t+1) for a periodic/cosh correlator.

    The returned array has the same length as the input correlator. Entry `t`
    stores the solution derived from the ratio `C(t) / C(t+1)`, so valid points
    live on `t = 0, ..., len(values)-2`. The final entry is always `np.nan`.
    Failed solves and invalid ratios also produce `np.nan`.
    """
    correlator = np.asarray(values, dtype=float)
    output = np.full(len(correlator), np.nan, dtype=float)
    if len(correlator) < 2:
        return output

    valid = (
        np.isfinite(correlator[:-1])
        & np.isfinite(correlator[1:])
        & (correlator[:-1] != 0.0)
        & (correlator[1:] != 0.0)
    )
    ratios = np.full(len(correlator) - 1, np.nan, dtype=float)
    ratios[valid] = correlator[:-1][valid] / correlator[1:][valid]
    for t, ratio in enumerate(ratios):
        if np.isfinite(ratio):
            output[t] = solve_cosh_effective_mass(t, float(ratio), nt, lower=lower, upper=upper, tol=tol)
    return output


def compute_effective_mass_antisymmetric_root(
    values: np.ndarray,
    nt: int,
    lower: float = 2e-5,
    upper: float = 20.0,
    tol: float = 1e-8,
) -> np.ndarray:
    """Compute m_eff(t) from C(t)/C(t+1) for an antisymmetric/sinh correlator.

    The returned array has the same length as the input correlator. Entry `t`
    stores the solution derived from `C(t) / C(t+1)`, so valid points live on
    `t = 0, ..., len(values)-2`. The final entry is always `np.nan`.
    """
    correlator = np.asarray(values, dtype=float)
    output = np.full(len(correlator), np.nan, dtype=float)
    if len(correlator) < 2:
        return output

    valid = (
        np.isfinite(correlator[:-1])
        & np.isfinite(correlator[1:])
        & (correlator[:-1] != 0.0)
        & (correlator[1:] != 0.0)
    )
    ratios = np.full(len(correlator) - 1, np.nan, dtype=float)
    ratios[valid] = correlator[:-1][valid] / correlator[1:][valid]
    for t, ratio in enumerate(ratios):
        if np.isfinite(ratio):
            output[t] = solve_antisymmetric_effective_mass(
                t,
                float(ratio),
                nt,
                lower=lower,
                upper=upper,
                tol=tol,
            )
    return output


def effective_mass_single(correlator: np.ndarray, model: str, nt: int | None = None) -> np.ndarray:
    values = np.asarray(correlator, dtype=float)
    if model == "normal":
        output = np.full(len(values) - 1, np.nan, dtype=float)
        valid = (values[:-1] > 0.0) & (values[1:] > 0.0)
        output[valid] = np.log(values[:-1][valid] / values[1:][valid])
        return output

    if model == "symmetric":
        if nt is None:
            raise ValueError("nt is required for symmetric effective-mass estimation")
        return compute_effective_mass_cosh_root(values, nt)

    if model == "antisymmetric":
        if nt is None:
            raise ValueError("nt is required for antisymmetric effective-mass estimation")
        return compute_effective_mass_antisymmetric_root(values, nt)
    raise ValueError(f"unsupported model for effective mass: {model}")


def effective_mass_with_bootstrap(
    bootstrap_means: np.ndarray,
    model: str,
    nt: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.array([effective_mass_single(sample, model, nt=nt) for sample in bootstrap_means])
    mean = np.nanmean(samples, axis=0)
    err = np.nanstd(samples, axis=0, ddof=1)
    return mean, err


def choose_default_tmax(
    meff_mean: np.ndarray,
    meff_err: np.ndarray,
    nt: int,
    max_available_t: int,
) -> int:
    cap = min(max_available_t, nt // 2 - 4)
    if cap < 4:
        return max_available_t

    candidate = None
    for t in range(2, min(len(meff_mean), cap + 1)):
        if not np.isfinite(meff_mean[t]) or meff_mean[t] == 0.0:
            candidate = t
            break
        relative_error = abs(meff_err[t] / meff_mean[t])
        if relative_error >= 0.5:
            candidate = t
            break
    if candidate is None:
        return cap
    return max(4, min(candidate, cap))


def pack_fit_parameters(amplitudes: np.ndarray, energies: np.ndarray) -> np.ndarray:
    amps = np.clip(np.asarray(amplitudes, dtype=float), 1e-16, None)
    en = np.asarray(energies, dtype=float)
    ordered = np.maximum.accumulate(np.clip(en, 1e-12, None))

    theta = [*np.log(amps), np.log(ordered[0])]
    for idx in range(1, len(ordered)):
        theta.append(np.log(max(ordered[idx] - ordered[idx - 1], 1e-12)))
    return np.array(theta, dtype=float)


def unpack_fit_parameters(theta: np.ndarray, nstates: int) -> tuple[np.ndarray, np.ndarray]:
    amps = np.exp(theta[:nstates])
    energy_parts = theta[nstates:]
    energies = np.empty(nstates, dtype=float)
    energies[0] = np.exp(energy_parts[0])
    for idx in range(1, nstates):
        energies[idx] = energies[idx - 1] + np.exp(energy_parts[idx])
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
) -> np.ndarray:
    amplitudes, energies = unpack_fit_parameters(theta, nstates)
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
) -> FitResult:
    primary_amplitudes = np.asarray(initial_amplitudes, dtype=float)
    primary_energies = np.asarray(initial_energies, dtype=float)

    def run_attempt(amp_guess: np.ndarray, energy_guess: np.ndarray) -> FitResult:
        theta0 = pack_fit_parameters(amp_guess, energy_guess)
        result = least_squares(
            fit_residuals,
            theta0,
            method="trf",
            max_nfev=8000,
            args=(times, data, sigma, nt, model, nstates, priors, lambda_prior, residual_model),
        )
        amplitudes, energies = unpack_fit_parameters(result.x, nstates)
        residual = fit_residuals(
            result.x,
            times,
            data,
            sigma,
            nt,
            model,
            nstates,
            priors,
            lambda_prior,
            residual_model,
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
            message=result.message,
            used_uncorrelated_fallback=False,
        )

    primary_result = run_attempt(primary_amplitudes, primary_energies)
    if primary_result.success and np.all(np.isfinite(primary_result.params)):
        return primary_result

    for amp_guess, energy_guess in build_fallback_fit_attempts(primary_amplitudes, primary_energies, nstates):
        current = run_attempt(amp_guess, energy_guess)
        if current.success and np.all(np.isfinite(current.params)):
            return current

    if residual_model is not None and residual_model.fit_mode == "correlated":
        fallback_sigma = residual_model.fallback_sigma
        if fallback_sigma is not None and np.all(np.isfinite(fallback_sigma)) and np.all(fallback_sigma > 0.0):
            diagonal_model = ResidualModel(fit_mode="uncorrelated", sigma=fallback_sigma)
            fallback_result = fit_nstate_sample(
                times,
                data,
                fallback_sigma,
                nt,
                model,
                initial_amplitudes,
                initial_energies,
                nstates,
                priors=priors,
                lambda_prior=lambda_prior,
                residual_model=diagonal_model,
            )
            if fallback_result.success:
                return FitResult(
                    params=fallback_result.params,
                chi2=fallback_result.chi2,
                chi2_dof=fallback_result.chi2_dof,
                pvalue=fallback_result.pvalue,
                success=True,
                message="correlated fit failed; fell back to diagonal covariance fit",
                used_uncorrelated_fallback=True,
            )

    return FitResult(
        params=np.full(2 * nstates, np.nan),
        chi2=np.nan,
        chi2_dof=np.nan,
        pvalue=np.nan,
        success=False,
        message="all nonlinear fit attempts failed",
        used_uncorrelated_fallback=False,
    )


def build_fallback_fit_attempts(
    amplitudes: np.ndarray,
    energies: np.ndarray,
    nstates: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    base_amp = np.clip(np.asarray(amplitudes, dtype=float), 1e-16, None)
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


def build_energy_priors_from_previous_row(
    previous_row: FitSummaryRow | None,
    nstates: int,
) -> tuple[EnergyPrior, ...]:
    if previous_row is None or nstates <= 1:
        return ()
    previous_nstates = previous_row.nstates
    max_index = min(previous_nstates, nstates - 1)
    priors: list[EnergyPrior] = []
    for energy_index in range(max_index):
        center = previous_row.params_mean[previous_nstates + energy_index]
        sigma = previous_row.params_err[previous_nstates + energy_index]
        if np.isfinite(center) and np.isfinite(sigma) and sigma > 0.0:
            priors.append(EnergyPrior(energy_index=energy_index, center=center, sigma=sigma))
    return tuple(priors)


def fit_constant_window(
    tmins: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
) -> tuple[float, float]:
    del tmins
    del errors
    values = np.asarray(values, dtype=float)
    constant = float(np.mean(values))
    residual = values - constant
    chi2_value = float(np.dot(residual, residual))
    dof = max(len(values) - 1, 1)
    return constant, chi2_value / dof


def fit_linear_window(
    tmins: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
) -> tuple[float, float, float, float]:
    del errors
    tmins = np.asarray(tmins, dtype=float)
    values = np.asarray(values, dtype=float)
    design = np.column_stack([np.ones_like(tmins, dtype=float), tmins])
    normal = design.T @ design
    if np.linalg.det(normal) <= 0.0:
        return np.nan, np.nan, np.nan, np.nan
    covariance = np.linalg.inv(normal)
    beta = covariance @ (design.T @ values)
    intercept = float(beta[0])
    slope = float(beta[1])
    model = intercept + slope * tmins
    residual = values - model
    chi2_value = float(np.dot(residual, residual))
    dof = max(len(values) - 2, 1)
    variance = chi2_value / dof
    slope_err = float(np.sqrt(max(variance * covariance[1, 1], 0.0)))
    return intercept, slope, slope_err, variance


def window_has_no_significant_trend(slope: float, slope_err: float, slope_nsigma: float) -> bool:
    if not np.isfinite(slope) or not np.isfinite(slope_err) or slope_err <= 0.0:
        return False
    return abs(slope) / slope_err < slope_nsigma


def select_best_plateau_window(candidates: list[PlateauCandidate]) -> PlateauCandidate:
    if not candidates:
        raise ValueError("no plateau candidates satisfied the constant-fit and trend criteria")
    return max(
        candidates,
        key=lambda item: (
            item.end_tmin - item.start_tmin + 1,
            item.end_tmin,
            -item.slope_significance,
        ),
    )


def suggest_plateau(
    rows: list[FitSummaryRow],
    energy_index: int = 0,
    min_window_len: int = 2,
    slope_nsigma: float = 1.3,
    max_constant_chi2_dof: float = 5.0,
    max_row_chi2_dof: float = 5.0,
    allow_relaxed_fallback: bool = False,
) -> PlateauWindow:
    valid_rows = [
        row
        for row in rows
        if row.success_meanfit and np.isfinite(row.chi2_dof) and row.chi2_dof <= max_row_chi2_dof
    ]
    if not valid_rows:
        raise ValueError("no successful fit rows are available for plateau selection")

    candidates: list[PlateauCandidate] = []
    fallback_candidates: list[PlateauCandidate] = []
    for start in range(len(valid_rows)):
        for end in range(start + min_window_len - 1, len(valid_rows)):
            window_rows = valid_rows[start : end + 1]
            tmins = np.array([row.tmin for row in window_rows], dtype=float)
            if not np.all(np.diff(tmins) == 1):
                break
            energy_column = len(window_rows[0].params_mean) // 2 + energy_index
            energy_values = np.array([row.params_mean[energy_column] for row in window_rows], dtype=float)
            energy_errors = np.array(
                [max(row.params_err[energy_column], 1e-12) for row in window_rows],
                dtype=float,
            )
            if not np.all(np.isfinite(energy_values)) or not np.all(np.isfinite(energy_errors)):
                continue

            _, constant_chi2_dof = fit_constant_window(tmins, energy_values, energy_errors)
            _, slope, slope_err, _ = fit_linear_window(tmins, energy_values, energy_errors)
            slope_significance = abs(slope) / slope_err if np.isfinite(slope_err) and slope_err > 0.0 else np.inf
            candidate = PlateauCandidate(
                start_tmin=int(window_rows[0].tmin),
                end_tmin=int(window_rows[-1].tmin),
                representative_tmin=int(window_rows[len(window_rows) // 2].tmin),
                constant_chi2_dof=float(constant_chi2_dof),
                slope=float(slope),
                slope_err=float(slope_err),
                slope_significance=float(slope_significance),
            )
            fallback_candidates.append(candidate)
            passes = (
                np.isfinite(constant_chi2_dof)
                and constant_chi2_dof <= max_constant_chi2_dof
                and window_has_no_significant_trend(slope, slope_err, slope_nsigma)
            )
            if passes:
                candidates.append(candidate)

    if candidates:
        best_candidate = select_best_plateau_window(candidates)
    elif allow_relaxed_fallback and fallback_candidates:
        best_candidate = select_best_plateau_window(fallback_candidates)
    elif allow_relaxed_fallback and valid_rows:
        representative = max(valid_rows, key=lambda row: row.tmin)
        best_candidate = PlateauCandidate(
            start_tmin=representative.tmin,
            end_tmin=representative.tmin,
            representative_tmin=representative.tmin,
            constant_chi2_dof=0.0,
            slope=0.0,
            slope_err=np.inf,
            slope_significance=0.0,
        )
    else:
        raise ValueError("no plateau candidates satisfied the constant-fit and trend criteria")
    plateau_rows = [
        row for row in valid_rows if best_candidate.start_tmin <= row.tmin <= best_candidate.end_tmin
    ]
    energy_column = len(plateau_rows[0].params_mean) // 2 + energy_index
    amplitude_column = energy_index
    energy_values = np.array([row.params_mean[energy_column] for row in plateau_rows])
    energy_errors = np.array([max(row.params_err[energy_column], 1e-12) for row in plateau_rows])
    amplitude_values = np.array([row.params_mean[amplitude_column] for row in plateau_rows])
    amplitude_errors = np.array([max(row.params_err[amplitude_column], 1e-12) for row in plateau_rows])
    energy_weights = 1.0 / energy_errors**2
    amplitude_weights = 1.0 / amplitude_errors**2
    representative = plateau_rows[len(plateau_rows) // 2]
    return PlateauWindow(
        start_tmin=plateau_rows[0].tmin,
        end_tmin=plateau_rows[-1].tmin,
        representative_tmin=best_candidate.representative_tmin,
        energy_mean=float(np.sum(energy_weights * energy_values) / np.sum(energy_weights)),
        amplitude_mean=float(np.sum(amplitude_weights * amplitude_values) / np.sum(amplitude_weights)),
    )


def build_initial_guess_from_plateau(
    plateau: PlateauWindow,
    nstates: int,
) -> tuple[np.ndarray, np.ndarray]:
    amplitudes = np.full(nstates, plateau.amplitude_mean, dtype=float)
    energies = np.full(nstates, plateau.energy_mean, dtype=float)
    if nstates >= 2:
        energies[1] = max(2.0 * plateau.energy_mean, plateau.energy_mean + 0.2)
    if nstates >= 3:
        energies[2] = max(1.5 * energies[1], 3.0 * plateau.energy_mean)
        amplitudes[2] = amplitudes[1]
    return amplitudes, energies


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
) -> tuple[list[FitSummaryRow], dict[int, np.ndarray], dict[int, FitResult]]:
    rows: list[FitSummaryRow] = []
    sample_tables: dict[int, np.ndarray] = {}
    meanfit_results: dict[int, FitResult] = {}

    mean_correlator = np.mean(bootstrap_means, axis=0)
    previous_meanfit: FitResult | None = None
    previous_sample_fits: list[FitResult | None] = [None] * len(bootstrap_means)
    for tmin in tmin_values:
        times = np.arange(tmin, tmax + 1)
        data_mean = mean_correlator[tmin : tmax + 1]
        sigma_slice = np.clip(sigma[tmin : tmax + 1], 1e-12, None)
        residual_model_note: str | None = None
        try:
            covariance_slice = None if covariance is None else covariance[tmin : tmax + 1, tmin : tmax + 1]
            residual_model = build_residual_model(fit_mode, sigma_slice, covariance_slice)
        except np.linalg.LinAlgError as error:
            if fit_mode == "correlated" and covariance is not None:
                covariance_slice = covariance[tmin : tmax + 1, tmin : tmax + 1]
                diagonal_sigma = np.sqrt(np.clip(np.diag(covariance_slice), 1e-24, None))
                residual_model = ResidualModel(fit_mode="uncorrelated", sigma=np.clip(diagonal_sigma, 1e-12, None))
                residual_model_note = (
                    f"correlated window covariance factorization failed for tmin={tmin}, "
                    f"tmax={tmax}; fell back to diagonal covariance fit: {error}"
                )
                print(f"[nstate-fit] {residual_model_note}")
            else:
                residual_model = build_residual_model("uncorrelated", sigma_slice)
                residual_model_note = (
                    f"{fit_mode} residual model setup failed for tmin={tmin}, tmax={tmax}: {error}"
                )
                print(f"[nstate-fit] {residual_model_note}")

        meanfit_amplitudes = initial_amplitudes
        meanfit_energies = initial_energies
        if (
            previous_meanfit is not None
            and previous_meanfit.success
            and np.all(np.isfinite(previous_meanfit.params))
        ):
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
        meanfit_results[tmin] = meanfit
        previous_meanfit = meanfit if meanfit.success and np.all(np.isfinite(meanfit.params)) else None

        sample_rows = np.full((len(bootstrap_means), 5 + 2 * nstates), np.nan, dtype=float)
        success_params: list[np.ndarray] = []
        fallback_uncorrelated_successes = 0
        for sample_id, sample in enumerate(bootstrap_means):
            sample_amplitudes = meanfit.params[:nstates] if meanfit.success else initial_amplitudes
            sample_energies = meanfit.params[nstates:] if meanfit.success else initial_energies
            previous_sample = previous_sample_fits[sample_id]
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
            )
            previous_sample_fits[sample_id] = (
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

        sample_tables[tmin] = sample_rows
        if success_params:
            params_mean, params_err = summarize_bootstrap_parameters(np.vstack(success_params))
        else:
            params_mean = tuple(np.nan for _ in range(2 * nstates))
            params_err = tuple(np.nan for _ in range(2 * nstates))

        rows.append(
            FitSummaryRow(
                nstates=nstates,
                tmin=tmin,
                tmax=tmax,
                success_meanfit=1 if meanfit.success else 0,
                bootstrap_successes=len(success_params),
                bootstrap_total=len(bootstrap_means),
                bootstrap_success_fraction=len(success_params) / len(bootstrap_means),
                fallback_uncorrelated_successes=fallback_uncorrelated_successes,
                chi2_dof=meanfit.chi2_dof,
                pvalue=meanfit.pvalue,
                plateau_flag=0,
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
            row.plateau_flag,
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


def mark_plateau(rows: list[FitSummaryRow], plateau: PlateauWindow) -> list[FitSummaryRow]:
    marked = []
    for row in rows:
        flagged = 1 if plateau.start_tmin <= row.tmin <= plateau.end_tmin else 0
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
                plateau_flag=flagged,
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
        "plateau_flag",
    ]
    columns += [f"A{idx}_mean" for idx in range(max_states)]
    columns += [f"A{idx}_err" for idx in range(max_states)]
    columns += [f"E{idx}_mean" for idx in range(max_states)]
    columns += [f"E{idx}_err" for idx in range(max_states)]
    return " ".join(columns)


def recommended_plateau_note() -> str:
    return (
        "Recommended practical plateau strategy:\n"
        "1. Start from the 1-state E0(tmin) table at fixed tmax.\n"
        "2. Keep only rows whose mean fit succeeded and whose row-level chi2/dof is reasonable.\n"
        "3. For each consecutive candidate window, use the mean E0(tmin) values only.\n"
        "4. Fit that mean series to both a constant and a line: E(tmin) = a + b tmin.\n"
        "5. Accept the window if the constant-fit chi2/dof is reasonable and the slope is not significant,\n"
        "   using the default criterion |b|/sigma_b < 1.3.\n"
        "6. Among acceptable windows, prefer the longest one; if lengths tie, prefer the later-time window;\n"
        "   if still tied, prefer the one with the smallest slope significance.\n"
        "7. Bootstrap samples are still used for final uncertainties, but not for the plateau decision itself.\n"
        "8. Use the weighted average of the selected window to seed the next-state fit.\n"
    )


def write_plateau_note(path: Path) -> None:
    path.write_text(recommended_plateau_note(), encoding="utf-8")


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


def _weighted_average(values: np.ndarray, errors: np.ndarray) -> float:
    safe_errors = np.clip(np.asarray(errors, dtype=float), 1e-12, None)
    weights = 1.0 / safe_errors**2
    return float(np.sum(weights * values) / np.sum(weights))


def _plateau_from_fit_table(table: np.ndarray, nstates: int) -> PlateauWindow | None:
    if table.size == 0:
        return None

    rows = np.atleast_2d(np.asarray(table, dtype=float))
    plateau_flag_column = 9 if rows.shape[1] >= 10 + 4 * nstates else 8
    base_offset = plateau_flag_column + 1
    plateau_rows = rows[np.isclose(rows[:, plateau_flag_column], 1.0)]
    if len(plateau_rows) == 0:
        return None

    tmins = plateau_rows[:, 0].astype(int)
    amplitude_values = plateau_rows[:, base_offset]
    amplitude_errors = plateau_rows[:, base_offset + nstates]
    energy_values = plateau_rows[:, base_offset + 2 * nstates]
    energy_errors = plateau_rows[:, base_offset + 3 * nstates]

    representative_tmin = int(tmins[len(tmins) // 2])
    return PlateauWindow(
        start_tmin=int(tmins[0]),
        end_tmin=int(tmins[-1]),
        representative_tmin=representative_tmin,
        energy_mean=_weighted_average(energy_values, energy_errors),
        amplitude_mean=_weighted_average(amplitude_values, amplitude_errors),
    )


def _representative_row_from_fit_table(table: np.ndarray, nstates: int) -> FitSummaryRow | None:
    if table.size == 0:
        return None

    rows = np.atleast_2d(np.asarray(table, dtype=float))
    has_fallback_column = rows.shape[1] >= 10 + 4 * nstates
    plateau_flag_column = 9 if has_fallback_column else 8
    base_offset = plateau_flag_column + 1
    plateau_rows = rows[np.isclose(rows[:, plateau_flag_column], 1.0)]
    if len(plateau_rows) == 0:
        return None
    representative = plateau_rows[len(plateau_rows) // 2]

    params_mean = tuple(representative[base_offset : base_offset + 2 * nstates])
    params_err = tuple(representative[base_offset + 2 * nstates : base_offset + 4 * nstates])
    return FitSummaryRow(
        nstates=nstates,
        tmin=int(representative[0]),
        tmax=int(representative[1]),
        success_meanfit=int(representative[2]),
        bootstrap_successes=int(representative[3]),
        bootstrap_total=int(representative[4]),
        bootstrap_success_fraction=float(representative[5]),
        fallback_uncorrelated_successes=int(representative[6]) if has_fallback_column else 0,
        chi2_dof=float(representative[7] if has_fallback_column else representative[6]),
        pvalue=float(representative[8] if has_fallback_column else representative[7]),
        plateau_flag=int(representative[plateau_flag_column]),
        params_mean=params_mean,
        params_err=params_err,
    )


def _try_load_previous_plateau(
    spec: NStateFitInput,
    title: str,
    nstates: int,
    tmax: int,
) -> PlateauWindow | None:
    if nstates <= 1:
        return None

    previous_state = nstates - 1
    fit_path, sample_path = _state_output_paths(spec, title, previous_state, tmax)
    if not fit_path.exists() or not sample_path.exists():
        return None
    if fit_path.stat().st_size == 0 or sample_path.stat().st_size == 0:
        return None

    try:
        table = np.loadtxt(fit_path, ndmin=2)
    except Exception:
        return None

    return _plateau_from_fit_table(table, previous_state)


def _try_load_previous_representative_row(
    spec: NStateFitInput,
    title: str,
    nstates: int,
    tmax: int,
) -> FitSummaryRow | None:
    if nstates <= 1:
        return None

    previous_state = nstates - 1
    fit_path, sample_path = _state_output_paths(spec, title, previous_state, tmax)
    if not fit_path.exists() or not sample_path.exists():
        return None
    if fit_path.stat().st_size == 0 or sample_path.stat().st_size == 0:
        return None

    try:
        table = np.loadtxt(fit_path, ndmin=2)
    except Exception:
        return None
    return _representative_row_from_fit_table(table, previous_state)


def run_single_dataset(spec: NStateFitInput, pz: int) -> list[Path]:
    title = spec.title_pattern.replace("*", str(pz))
    csv_path = spec.correlator_path_pattern.replace("*", str(pz))
    _, correlators = load_correlator_csv(csv_path)
    if correlators.shape[1] != spec.nt:
        raise ValueError(f"{csv_path} has Nt={correlators.shape[1]}, expected {spec.nt}")

    processed = apply_fold_t(correlators, spec.nt, spec.fold_t)
    t0, t1 = spec.tsrange
    selected = processed[:, t0 : t1 + 1]
    binned = bin_correlators(selected, binsize=spec.binsize)
    bootstrap_means = bootstrap_correlator_means(
        binned,
        n_samples=spec.bootstrap_samples,
        sample_size=spec.bootstrap_size,
        seed=spec.seed,
    )
    sigma = np.nanstd(bootstrap_means, axis=0, ddof=1)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, 1e-12)
    covariance = compute_bootstrap_covariance(bootstrap_means) if spec.fit_mode == "correlated" else None

    meff_mean, meff_err = effective_mass_with_bootstrap(bootstrap_means, spec.model, nt=spec.nt)
    tmax = (
        spec.tmax
        if spec.tmax is not None
        else choose_default_tmax(meff_mean, meff_err, spec.nt, selected.shape[1] - 2)
    )

    dataset_dir = spec.results_dir / title
    tables_dir = dataset_dir / "tables"
    plots_dir = dataset_dir / "plots"
    samples_dir = dataset_dir / "samples"
    for directory in (tables_dir, plots_dir, samples_dir):
        directory.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    correlator_table = tables_dir / f"{title}_{spec.model}_correlator_mean.txt"
    np.savetxt(
        correlator_table,
        np.column_stack([np.arange(selected.shape[1]), np.mean(bootstrap_means, axis=0), sigma]),
        header="t corr_mean corr_err",
        fmt="%.10e",
    )
    outputs.append(correlator_table)
    meff_table = tables_dir / f"{title}_{spec.model}_effective_mass_tmax{tmax}.txt"
    np.savetxt(
        meff_table,
        np.column_stack([np.arange(len(meff_mean)), meff_mean, meff_err]),
        header="t meff_mean meff_err",
        fmt="%.10e",
    )
    outputs.append(meff_table)

    plateau_note = dataset_dir / "plateau_recommendation.txt"
    write_plateau_note(plateau_note)
    outputs.append(plateau_note)

    one_state_energy_guess = np.nanmedian(meff_mean[np.isfinite(meff_mean) & (np.arange(len(meff_mean)) >= 2)])
    if not np.isfinite(one_state_energy_guess):
        one_state_energy_guess = 0.5
    one_state_amplitude_guess = max(bootstrap_means.mean(axis=0)[2] * np.exp(one_state_energy_guess * 2), 1e-12)
    current_guess = (np.array([one_state_amplitude_guess]), np.array([one_state_energy_guess]))

    representative_rows: dict[int, FitSummaryRow] = {}
    plateau_cache: dict[int, PlateauWindow] = {}
    fit_tables: dict[int, Path] = {}
    written_states: set[int] = set()
    state_sources: dict[int, str] = {}

    def compute_state(nstates: int) -> None:
        nonlocal current_guess
        if nstates in written_states:
            return

        if nstates == 1:
            tmin_start = 2
            initial_guess = current_guess
            priors: tuple[EnergyPrior, ...] = ()
        else:
            previous_plateau = plateau_cache.get(nstates - 1)
            previous_representative = representative_rows.get(nstates - 1)
            if previous_plateau is None:
                cached_plateau = _try_load_previous_plateau(spec, title, nstates, tmax)
                cached_representative = _try_load_previous_representative_row(spec, title, nstates, tmax)
                if cached_plateau is not None:
                    plateau_cache[nstates - 1] = cached_plateau
                    state_sources.setdefault(nstates - 1, "cache_reused")
                    previous_plateau = cached_plateau
                    if cached_representative is not None:
                        representative_rows[nstates - 1] = cached_representative
                        previous_representative = cached_representative
                    print(
                        f"[nstate-fit] Reusing cached {nstates - 1}-state plateau for "
                        f"{title} at tmax={tmax}."
                    )
                else:
                    print(
                        f"[nstate-fit] No cached {nstates - 1}-state plateau found for "
                        f"{title} at tmax={tmax}; computing it now."
                    )
                    compute_state(nstates - 1)
                    previous_plateau = plateau_cache[nstates - 1]
                    previous_representative = representative_rows[nstates - 1]

            tmin_start = min(previous_plateau.start_tmin + 2, tmax - 2)
            initial_guess = build_initial_guess_from_plateau(previous_plateau, nstates)
            priors = build_energy_priors_from_previous_row(previous_representative, nstates)
            current_guess = initial_guess

        tmin_values = range(tmin_start, tmax - 1)
        rows, sample_tables, _ = run_sliding_fits(
            bootstrap_means,
            sigma,
            spec.fit_mode,
            spec.nt,
            spec.model,
            nstates,
            tmin_values,
            tmax,
            initial_guess[0],
            initial_guess[1],
            covariance=covariance,
            priors=priors,
            lambda_prior=spec.lambda_prior,
        )
        plateau = suggest_plateau(rows, energy_index=0, allow_relaxed_fallback=True)
        plateau_cache[nstates] = plateau
        rows = mark_plateau(rows, plateau)
        representative = next(row for row in rows if row.tmin == plateau.representative_tmin)
        representative_rows[nstates] = representative

        table_path, sample_path = _state_output_paths(spec, title, nstates, tmax)
        table = serialize_fit_rows(rows)
        np.savetxt(table_path, table, header=header_for_fit_rows(nstates), fmt="%.10e")
        outputs.append(table_path)
        fit_tables[nstates] = table_path

        sample_blocks = []
        for tmin in sorted(sample_tables):
            sample_blocks.append(sample_tables[tmin])
        np.savetxt(
            sample_path,
            np.vstack(sample_blocks),
            header="tmin sample_id success chi2_dof pvalue " + " ".join(
                [f"A{idx}" for idx in range(nstates)] + [f"E{idx}" for idx in range(nstates)]
            ),
            fmt="%.10e",
        )
        outputs.append(sample_path)
        written_states.add(nstates)
        state_sources[nstates] = "computed_fresh"

    for nstates in spec.nstates:
        compute_state(nstates)

    summary_path = dataset_dir / f"{title}_{spec.model}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"title {title}\n")
        handle.write(f"model {spec.model}\n")
        handle.write(f"fit_mode {spec.fit_mode}\n")
        handle.write(f"tmax {tmax}\n")
        for nstates in sorted(representative_rows):
            plateau = plateau_cache[nstates]
            representative = representative_rows[nstates]
            state_source = state_sources.get(nstates, "computed_fresh")
            handle.write(f"{nstates}state source {state_source}\n")
            handle.write(
                f"{nstates}state plateau_tmin {plateau.start_tmin} {plateau.end_tmin} "
                f"representative_tmin {plateau.representative_tmin}\n"
            )
            handle.write(
                f"{nstates}state representative_fallback_uncorrelated_successes "
                f"{representative.fallback_uncorrelated_successes}\n"
            )
            for idx in range(nstates):
                handle.write(
                    f"{nstates}state A{idx} {representative.params_mean[idx]:.10e} "
                    f"{representative.params_err[idx]:.10e} "
                    f"E{idx} {representative.params_mean[nstates + idx]:.10e} "
                    f"{representative.params_err[nstates + idx]:.10e}\n"
                )
    outputs.append(summary_path)

    if spec.make_plots:
        for nstates, fit_table in fit_tables.items():
            outputs.extend(
                plot_nstate_outputs(
                    output_dir=plots_dir,
                    correlator_table=correlator_table,
                    meff_table=meff_table,
                    fit_table=fit_table,
                    nstates=nstates,
                    model=spec.model,
                    title=title,
                    nt=spec.nt,
                    lattice_spacing_fm=spec.lattice_spacing_fm,
                )
            )

    notebook_root = spec.results_dir / "notebook_plots" / title
    notebook_generated = notebook_root / "generated_plots"
    notebook_path = notebook_root / f"{title}_{spec.model}_nstate_plots.ipynb"
    outputs.append(
        write_nstate_plot_notebook(
            notebook_path=notebook_path,
            notebook_output_dir=notebook_generated,
            correlator_table=correlator_table,
            meff_table=meff_table,
            fit_tables=fit_tables,
            model=spec.model,
            title=title,
            nt=spec.nt,
            lattice_spacing_fm=spec.lattice_spacing_fm,
        )
    )
    return outputs


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
