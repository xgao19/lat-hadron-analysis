from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
from scipy.optimize import brentq

from ..common.parsing import load_control_file_entries, parse_fold_t, parse_optional_int, parse_tsrange
from ..common.utils import apply_fold_t, bin_correlators, bootstrap_correlator_means
from .io import load_correlator_csv


@dataclass(frozen=True)
class EffectiveMassInput:
    title_pattern: str
    ns: int
    nt: int
    lattice_spacing_fm: float
    correlator_path_pattern: str
    pzlist: tuple[int, ...]
    fold_t: str
    tsrange: tuple[int, int]
    model: str
    binsize: int
    bootstrap_samples: int | None
    bootstrap_size: int | None
    seed: int
    results_dir: Path


def solve_cosh_effective_mass(
    t: int,
    ratio: float,
    nt: int,
    lower: float = 2e-5,
    upper: float = 20.0,
    tol: float = 1e-8,
    max_iter: int = 256,
) -> float:
    if nt <= 0 or t < 0 or not np.isfinite(ratio) or ratio <= 0.0:
        return np.nan
    if lower <= 0.0 or upper <= lower:
        return np.nan

    center = 0.5 * nt

    def f(mass: float) -> float:
        return ratio * np.cosh(mass * (t + 1 - center)) - np.cosh(mass * (t - center))

    try:
        return float(brentq(f, lower, upper, xtol=tol, maxiter=max_iter))
    except ValueError:
        return np.nan


def solve_antisymmetric_effective_mass(
    t: int,
    ratio: float,
    nt: int,
    lower: float = 2e-5,
    upper: float = 20.0,
    tol: float = 1e-8,
    max_iter: int = 256,
) -> float:
    if nt <= 0 or t < 0 or not np.isfinite(ratio) or ratio <= 0.0:
        return np.nan
    if lower <= 0.0 or upper <= lower:
        return np.nan

    center = 0.5 * nt

    def f(mass: float) -> float:
        return ratio * np.sinh(mass * (t + 1 - center)) - np.sinh(mass * (t - center))

    try:
        return float(brentq(f, lower, upper, xtol=tol, maxiter=max_iter))
    except ValueError:
        return np.nan


def _compute_effective_mass_root(
    values: np.ndarray,
    nt: int,
    solver: Callable[..., float],
    *,
    lower: float,
    upper: float,
    tol: float,
) -> np.ndarray:
    correlator = np.asarray(values, dtype=float)
    output = np.full(len(correlator), np.nan, dtype=float)
    if len(correlator) < 2:
        return output

    valid = np.isfinite(correlator[:-1]) & np.isfinite(correlator[1:]) & (correlator[:-1] != 0.0) & (correlator[1:] != 0.0)
    ratios = np.full(len(correlator) - 1, np.nan, dtype=float)
    np.divide(correlator[:-1], correlator[1:], out=ratios, where=valid)
    for t, ratio in enumerate(ratios):
        if np.isfinite(ratio):
            output[t] = solver(t, float(ratio), nt, lower=lower, upper=upper, tol=tol)
    return output


def compute_effective_mass_cosh_root(
    values: np.ndarray,
    nt: int,
    lower: float = 2e-5,
    upper: float = 20.0,
    tol: float = 1e-8,
) -> np.ndarray:
    return _compute_effective_mass_root(
        values,
        nt,
        solve_cosh_effective_mass,
        lower=lower,
        upper=upper,
        tol=tol,
    )


def compute_effective_mass_antisymmetric_root(
    values: np.ndarray,
    nt: int,
    lower: float = 2e-5,
    upper: float = 20.0,
    tol: float = 1e-8,
) -> np.ndarray:
    return _compute_effective_mass_root(
        values,
        nt,
        solve_antisymmetric_effective_mass,
        lower=lower,
        upper=upper,
        tol=tol,
    )


def effective_mass_single(correlator: np.ndarray, model: str, nt: int | None = None) -> np.ndarray:
    values = np.asarray(correlator, dtype=float)
    if model == "normal":
        output = np.full(len(values) - 1, np.nan, dtype=float)
        valid = (values[:-1] > 0.0) & (values[1:] > 0.0)
        np.divide(values[:-1], values[1:], out=output, where=valid)
        np.log(output, out=output, where=valid)
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
    samples = np.stack([effective_mass_single(sample, model, nt=nt) for sample in bootstrap_means], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(samples, axis=0)
        err = np.nanstd(samples, axis=0, ddof=1)
    return mean, err


def parse_effective_mass_input(path: str | Path, results_dir: str | Path | None = None) -> EffectiveMassInput:
    file_path = Path(path)
    first_tokens, entries = load_control_file_entries(file_path)
    if len(first_tokens) < 4:
        raise ValueError("the first non-empty line must be: title Ns Nt a_fm")
    required = {"c2pt", "pzlist", "model"}
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")
    input_path = file_path.resolve()
    return EffectiveMassInput(
        title_pattern=first_tokens[0],
        ns=int(first_tokens[1]),
        nt=int(first_tokens[2]),
        lattice_spacing_fm=float(first_tokens[3]),
        correlator_path_pattern=entries["c2pt"][0],
        pzlist=tuple(int(item) for item in entries["pzlist"]),
        fold_t=parse_fold_t(entries),
        tsrange=parse_tsrange(entries, int(first_tokens[2])),
        model=entries["model"][0].lower(),
        binsize=int(entries.get("binsize", ["1"])[0]),
        bootstrap_samples=parse_optional_int(entries.get("bootstrap_samples", ["auto"])[0]),
        bootstrap_size=parse_optional_int(entries.get("bootstrap_size", ["auto"])[0]),
        seed=int(entries.get("seed", ["2026"])[0]),
        results_dir=(
            Path(entries["results_dir"][0])
            if "results_dir" in entries and results_dir is None
            else ((input_path.parent / "results_effective_mass") if results_dir is None else Path(results_dir))
        ),
    )


def run_effective_mass_workflow(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_effective_mass_input(input_file, results_dir=results_dir)
    outputs: list[Path] = []
    for pz in spec.pzlist:
        title = spec.title_pattern.replace("*", str(pz))
        csv_path = spec.correlator_path_pattern.replace("*", str(pz))
        _, correlators = load_correlator_csv(csv_path)
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
        meff_mean, meff_err = effective_mass_with_bootstrap(bootstrap_means, spec.model, nt=spec.nt)
        dataset_dir = spec.results_dir / title
        tables_dir = dataset_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_path = tables_dir / f"{title}_{spec.model}_effective_mass.txt"
        np.savetxt(
            table_path,
            np.column_stack([np.arange(t0, t0 + len(meff_mean)), meff_mean, meff_err]),
            header="t meff_mean meff_err",
            fmt="%.10e",
        )
        outputs.append(table_path)
    return outputs
