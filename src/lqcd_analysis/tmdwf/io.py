from __future__ import annotations

import glob
from pathlib import Path
import re

import numpy as np

from .models import normalize_tmdwf_operator

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    h5py = None


def expand_template(pattern: str, **kwargs: object) -> str:
    text = pattern
    if "{" in text and "}" in text:
        return text.format(**kwargs)
    if "*" in text and "pz" in kwargs:
        return text.replace("*", str(kwargs["pz"]))
    return text


def load_two_point_plateau_values(path: str | Path, nstates: int) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, ndmin=2)
    row = np.atleast_2d(np.asarray(table, dtype=float))[0]
    total_states = row.size // 4
    if total_states < nstates:
        raise ValueError(f"plateau table {path} has only {total_states} states, requested {nstates}")
    amplitudes = np.asarray(row[:nstates], dtype=float)
    energies = np.asarray(row[total_states : total_states + nstates], dtype=float)
    if not np.all(np.isfinite(amplitudes)) or not np.all(np.isfinite(energies)):
        raise ValueError(f"non-finite two-point plateau values found in {path}")
    return amplitudes, energies


def _infer_tmax_from_plateau_filename(path: str | Path) -> int | None:
    match = re.search(r"_tmax(\d+)_plateau\.txt$", Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def resolve_two_point_plateau_table(path_pattern: str | Path, *, pz: int) -> tuple[Path, int | None]:
    expanded = Path(expand_template(str(path_pattern), pz=pz))
    if "#" not in str(expanded):
        if not expanded.exists():
            raise FileNotFoundError(expanded)
        return expanded, _infer_tmax_from_plateau_filename(expanded)

    glob_pattern = str(expanded).replace("#", "*")
    matches = [Path(match) for match in sorted(glob.glob(glob_pattern))]
    if not matches:
        raise FileNotFoundError(path_pattern)

    ranked_matches: list[tuple[int, Path]] = []
    for candidate in matches:
        inferred_tmax = _infer_tmax_from_plateau_filename(candidate)
        if inferred_tmax is not None:
            ranked_matches.append((inferred_tmax, candidate))
    if not ranked_matches:
        raise ValueError(f"could not infer tmax from any plateau filename matching pattern: {path_pattern}")
    ranked_matches.sort(key=lambda item: item[0])
    selected_tmax, selected_path = ranked_matches[-1]
    return selected_path, selected_tmax


def _normalize_time_axis(data: np.ndarray, nt: int) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("TMDWF dataset must be a 2D array with configuration and time dimensions")
    if values.shape[1] == nt:
        return values
    if values.shape[0] == nt:
        return values.T
    raise ValueError(f"TMDWF dataset shape {values.shape} is incompatible with Nt={nt}")


def _sign_pattern(nt: int) -> np.ndarray:
    pattern = np.ones(nt, dtype=float)
    pattern[nt // 2 :] = -1.0
    return pattern


def apply_tmdwf_preprocessing(values: np.ndarray, nt: int, gm: str) -> np.ndarray:
    processed = np.asarray(values, dtype=np.complex128).copy()
    operator = normalize_tmdwf_operator(gm)
    if operator == "T5":
        processed *= _sign_pattern(nt)[None, :]
    else:
        processed *= -1j
    return processed


def fold_antisymmetric_complex(values: np.ndarray, nt: int) -> np.ndarray:
    correlators = np.asarray(values, dtype=np.complex128)
    if correlators.ndim != 2 or correlators.shape[1] != nt:
        raise ValueError("complex correlators must have shape (n_cfg, Nt)")
    folded_extent = nt // 2 + 1
    folded = correlators[:, :folded_extent].copy()
    for t in range(1, folded_extent):
        partner = (nt - t) % nt
        if partner == t:
            continue
        folded[:, t] = 0.5 * (correlators[:, t] - correlators[:, partner])
    return folded


def load_tmdwf_correlator(
    h5_path: str | Path,
    dataset_path_template: str,
    *,
    gm: str,
    eta: str,
    pz: int,
    tdirs: tuple[str, ...],
    bT: int,
    bz: int,
    nt: int,
    ns: int,
) -> np.ndarray:
    if h5py is None:
        raise ModuleNotFoundError("h5py is required to load TMDWF HDF5 data")

    file_path = Path(h5_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    with h5py.File(file_path, "r") as handle:
        return _load_tmdwf_correlator_from_handle(
            handle,
            dataset_path_template,
            gm=gm,
            eta=eta,
            pz=pz,
            tdirs=tdirs,
            bT=bT,
            bz=bz,
            nt=nt,
            ns=ns,
            file_label=str(file_path),
        )


def _load_tmdwf_correlator_from_handle(
    handle: "h5py.File",
    dataset_path_template: str,
    *,
    gm: str,
    eta: str,
    pz: int,
    tdirs: tuple[str, ...],
    bT: int,
    bz: int,
    nt: int,
    ns: int,
    file_label: str = "<open-h5>",
) -> np.ndarray:
    phase = 2.0 * np.pi * float(pz) / float(ns)
    bz_values = (0,) if bz == 0 else (abs(bz), -abs(bz))
    tdir_blocks: list[np.ndarray] = []
    for tdir in tdirs:
        bz_blocks: list[np.ndarray] = []
        for signed_bz in bz_values:
            dataset_path = expand_template(
                dataset_path_template,
                gm=gm,
                eta=eta,
                pz=pz,
                Tdir=tdir,
                bT=bT,
                bz=signed_bz,
            )
            if dataset_path not in handle:
                raise KeyError(f"dataset path not found in {file_label}: {dataset_path}")
            raw = _normalize_time_axis(np.asarray(handle[dataset_path]), nt)
            rotation = np.exp(-1j * phase * float(signed_bz) / 2.0)
            bz_blocks.append(np.asarray(raw, dtype=np.complex128) * rotation)
        averaged_bz = bz_blocks[0] if len(bz_blocks) == 1 else 0.5 * (bz_blocks[0] + bz_blocks[1])
        tdir_blocks.append(averaged_bz)

    averaged = np.mean(np.stack(tdir_blocks, axis=0), axis=0)
    return fold_antisymmetric_complex(apply_tmdwf_preprocessing(averaged, nt, gm), nt)
