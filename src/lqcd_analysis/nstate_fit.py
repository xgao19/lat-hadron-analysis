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
    nstates: tuple[int, ...]
    tmax: int | None
    binsize: int
    bootstrap_samples: int | None
    bootstrap_size: int | None
    seed: int
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
class FitResult:
    params: np.ndarray
    chi2: float
    chi2_dof: float
    pvalue: float
    success: bool
    message: str


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
        nstates=nstates,
        tmax=parse_optional_int(entries.get("tmax", ["auto"])[0]),
        binsize=int(entries.get("binsize", ["1"])[0]),
        bootstrap_samples=parse_optional_int(entries.get("bootstrap_samples", ["auto"])[0]),
        bootstrap_size=parse_optional_int(entries.get("bootstrap_size", ["auto"])[0]),
        seed=int(entries.get("seed", ["2026"])[0]),
        make_plots=parse_bool(entries.get("plot", ["true"])[0]),
        results_dir=(input_path.parent / "results_nstate_fit")
        if results_dir is None
        else Path(results_dir),
    )


def parse_optional_int(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    return int(value)


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


def effective_mass_single(correlator: np.ndarray, model: str) -> np.ndarray:
    values = np.asarray(correlator, dtype=float)
    if model == "normal":
        output = np.full(len(values) - 1, np.nan, dtype=float)
        valid = (values[:-1] > 0.0) & (values[1:] > 0.0)
        output[valid] = np.log(values[:-1][valid] / values[1:][valid])
        return output

    output = np.full(len(values), np.nan, dtype=float)
    if len(values) < 3:
        return output
    numerator = values[:-2] + values[2:]
    denominator = 2.0 * values[1:-1]
    ratio = np.full(len(values) - 2, np.nan, dtype=float)
    valid = (denominator != 0.0) & np.isfinite(numerator) & np.isfinite(denominator)
    ratio[valid] = numerator[valid] / denominator[valid]
    valid = valid & (ratio >= 1.0)
    output[1:-1][valid] = np.arccosh(ratio[valid])
    return output


def effective_mass_with_bootstrap(bootstrap_means: np.ndarray, model: str) -> tuple[np.ndarray, np.ndarray]:
    samples = np.array([effective_mass_single(sample, model) for sample in bootstrap_means])
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
) -> np.ndarray:
    amplitudes, energies = unpack_fit_parameters(theta, nstates)
    model_values = evaluate_model(times, amplitudes, energies, nt, model)
    return (model_values - data) / sigma


def fit_nstate_sample(
    times: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    nt: int,
    model: str,
    initial_amplitudes: np.ndarray,
    initial_energies: np.ndarray,
    nstates: int,
) -> FitResult:
    attempts = build_fit_attempts(initial_amplitudes, initial_energies, nstates)
    best_result: FitResult | None = None
    lowest_cost = np.inf

    for amp_guess, energy_guess in attempts:
        theta0 = pack_fit_parameters(amp_guess, energy_guess)
        result = least_squares(
            fit_residuals,
            theta0,
            method="trf",
            max_nfev=8000,
            args=(times, data, sigma, nt, model, nstates),
        )
        amplitudes, energies = unpack_fit_parameters(result.x, nstates)
        residual = fit_residuals(result.x, times, data, sigma, nt, model, nstates)
        chi2_value = float(np.dot(residual, residual))
        dof = max(len(times) - len(result.x), 1)
        chi2_dof = chi2_value / dof
        pvalue = float(chi2.sf(chi2_value, dof))

        current = FitResult(
            params=np.concatenate([amplitudes, energies]),
            chi2=chi2_value,
            chi2_dof=chi2_dof,
            pvalue=pvalue,
            success=bool(result.success),
            message=result.message,
        )
        if result.success and chi2_value < lowest_cost:
            best_result = current
            lowest_cost = chi2_value

    if best_result is not None:
        return best_result

    return FitResult(
        params=np.full(2 * nstates, np.nan),
        chi2=np.nan,
        chi2_dof=np.nan,
        pvalue=np.nan,
        success=False,
        message="all nonlinear fit attempts failed",
    )


def build_fit_attempts(
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


def suggest_plateau(rows: list[FitSummaryRow], energy_index: int = 0) -> PlateauWindow:
    best_window: tuple[int, int] | None = None
    current_start = None
    valid_rows = [row for row in rows if row.bootstrap_success_fraction >= 0.5 and row.success_meanfit]
    if not valid_rows:
        raise ValueError("no successful fit rows are available for plateau selection")

    for idx, row in enumerate(valid_rows):
        if idx == 0:
            current_start = idx
            continue
        previous = valid_rows[idx - 1]
        energy_prev = previous.params_mean[len(previous.params_mean) // 2 + energy_index]
        error_prev = previous.params_err[len(previous.params_err) // 2 + energy_index]
        energy_curr = row.params_mean[len(row.params_mean) // 2 + energy_index]
        error_curr = row.params_err[len(row.params_err) // 2 + energy_index]
        consistent = abs(energy_curr - energy_prev) <= 2.0 * np.sqrt(error_prev**2 + error_curr**2)
        chi2_ok = np.isfinite(row.chi2_dof) and row.chi2_dof <= 5.0
        contiguous = row.tmin == previous.tmin + 1
        if not (consistent and chi2_ok and contiguous):
            window = (current_start, idx - 1)
            if best_window is None or window[1] - window[0] > best_window[1] - best_window[0]:
                best_window = window
            current_start = idx

    if current_start is not None:
        window = (current_start, len(valid_rows) - 1)
        if best_window is None or window[1] - window[0] > best_window[1] - best_window[0]:
            best_window = window

    assert best_window is not None
    plateau_rows = valid_rows[best_window[0] : best_window[1] + 1]
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
        representative_tmin=representative.tmin,
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
    nt: int,
    model: str,
    nstates: int,
    tmin_values: range,
    tmax: int,
    initial_amplitudes: np.ndarray,
    initial_energies: np.ndarray,
) -> tuple[list[FitSummaryRow], dict[int, np.ndarray], dict[int, FitResult]]:
    rows: list[FitSummaryRow] = []
    sample_tables: dict[int, np.ndarray] = {}
    meanfit_results: dict[int, FitResult] = {}

    mean_correlator = np.mean(bootstrap_means, axis=0)
    for tmin in tmin_values:
        times = np.arange(tmin, tmax + 1)
        data_mean = mean_correlator[tmin : tmax + 1]
        sigma_slice = np.clip(sigma[tmin : tmax + 1], 1e-12, None)
        meanfit = fit_nstate_sample(
            times,
            data_mean,
            sigma_slice,
            nt,
            model,
            initial_amplitudes,
            initial_energies,
            nstates,
        )
        meanfit_results[tmin] = meanfit

        sample_rows = np.full((len(bootstrap_means), 5 + 2 * nstates), np.nan, dtype=float)
        success_params: list[np.ndarray] = []
        for sample_id, sample in enumerate(bootstrap_means):
            fit_result = fit_nstate_sample(
                times,
                sample[tmin : tmax + 1],
                sigma_slice,
                nt,
                model,
                meanfit.params[:nstates] if meanfit.success else initial_amplitudes,
                meanfit.params[nstates:] if meanfit.success else initial_energies,
                nstates,
            )
            sample_rows[sample_id, 0] = tmin
            sample_rows[sample_id, 1] = sample_id
            sample_rows[sample_id, 2] = 1.0 if fit_result.success else 0.0
            sample_rows[sample_id, 3] = fit_result.chi2_dof
            sample_rows[sample_id, 4] = fit_result.pvalue
            sample_rows[sample_id, 5:] = fit_result.params
            if fit_result.success and np.all(np.isfinite(fit_result.params)):
                success_params.append(fit_result.params)

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
        "2. Keep only rows with acceptable chi2/dof and bootstrap success fraction.\n"
        "3. Look for the longest contiguous tmin window where adjacent E0 values agree within 2 sigma.\n"
        "4. Prefer the earliest such stable window before late-time noise dominates.\n"
        "5. Use the weighted average of that window to seed the next-state fit.\n"
    )


def write_plateau_note(path: Path) -> None:
    path.write_text(recommended_plateau_note(), encoding="utf-8")


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

    meff_mean, meff_err = effective_mass_with_bootstrap(bootstrap_means, spec.model)
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
    for nstates in spec.nstates:
        if nstates == 1:
            tmin_start = 2
        else:
            previous_plateau = plateau_cache[nstates - 1]
            tmin_start = min(previous_plateau.start_tmin + 2, tmax - 2)
            current_guess = build_initial_guess_from_plateau(previous_plateau, nstates)

        tmin_values = range(tmin_start, tmax - 1)
        rows, sample_tables, _ = run_sliding_fits(
            bootstrap_means,
            sigma,
            spec.nt,
            spec.model,
            nstates,
            tmin_values,
            tmax,
            current_guess[0],
            current_guess[1],
        )
        plateau = suggest_plateau(rows, energy_index=0)
        plateau_cache[nstates] = plateau
        rows = mark_plateau(rows, plateau)
        representative = next(row for row in rows if row.tmin == plateau.representative_tmin)
        representative_rows[nstates] = representative

        table_path = tables_dir / f"{title}_{spec.model}_{nstates}state_tmax{tmax}_fits.txt"
        table = serialize_fit_rows(rows)
        np.savetxt(table_path, table, header=header_for_fit_rows(nstates), fmt="%.10e")
        outputs.append(table_path)
        fit_tables[nstates] = table_path

        sample_path = samples_dir / f"{title}_{spec.model}_{nstates}state_tmax{tmax}_samples.txt"
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

    summary_path = dataset_dir / f"{title}_{spec.model}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"title {title}\n")
        handle.write(f"model {spec.model}\n")
        handle.write(f"tmax {tmax}\n")
        for nstates in spec.nstates:
            plateau = plateau_cache[nstates]
            representative = representative_rows[nstates]
            handle.write(
                f"{nstates}state plateau_tmin {plateau.start_tmin} {plateau.end_tmin} "
                f"representative_tmin {plateau.representative_tmin}\n"
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
                )
            )

    notebook_root = spec.results_dir.parent / "notebook_plots" / title
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
