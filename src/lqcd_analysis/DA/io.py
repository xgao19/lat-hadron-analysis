from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from .models import normalize_da_operator

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


def resolve_qda_h5_path(path_pattern: str | Path, *, pz: int, gm: str) -> Path:
    resolved = Path(expand_template(str(path_pattern), pz=pz, gm=gm))
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


@dataclass(frozen=True)
class TwoPointFitReference:
    path: Path
    tmin: int
    tmax: int
    amplitudes: np.ndarray
    energies: np.ndarray


def _extract_fit_row_parameters(row: np.ndarray, nstates: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    values = np.asarray(row, dtype=float)
    if values.ndim != 1:
        raise ValueError("fit table row must be one-dimensional")
    has_fallback_column = values.size >= 10 + 4 * nstates
    selection_flag_column = 9 if has_fallback_column else 8
    amp_mean_start = selection_flag_column + 1
    amp_err_start = amp_mean_start + nstates
    energy_mean_start = amp_err_start + nstates
    energy_err_start = energy_mean_start + nstates
    required_size = energy_err_start + nstates
    if values.size < required_size:
        raise ValueError(
            f"fit table row has {values.size} columns, expected at least {required_size} for {nstates} states"
        )
    amplitudes = np.asarray(values[amp_mean_start : amp_mean_start + nstates], dtype=float)
    energies = np.asarray(values[energy_mean_start : energy_mean_start + nstates], dtype=float)
    if not np.all(np.isfinite(amplitudes)) or not np.all(np.isfinite(energies)):
        raise ValueError("non-finite two-point fit parameters found in fit table row")
    return int(values[0]), int(values[1]), amplitudes, energies


def resolve_two_point_fit_reference(
    fit_root: str | Path,
    *,
    title: str,
    nstates: int,
    tmin: int,
    tmax: int,
) -> TwoPointFitReference:
    tables_dir = Path(fit_root) / title / "tables"
    if not tables_dir.exists():
        raise FileNotFoundError(f"two-point nstate fit tables directory does not exist: {tables_dir}")

    pattern = f"{title}_*_{nstates}state_tmax{int(tmax)}_fits.txt"
    candidates = sorted(tables_dir.glob(pattern))
    if not candidates:
        candidates = sorted(tables_dir.glob(f"*_{nstates}state_tmax{int(tmax)}_fits.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"two-point nstate fit table does not exist for title={title}, nstates={nstates}, tmax={tmax}: {tables_dir / pattern}"
        )

    for candidate in candidates:
        table = np.loadtxt(candidate, ndmin=2)
        rows = np.atleast_2d(np.asarray(table, dtype=float))
        matching_rows = rows[rows[:, 0].astype(int) == int(tmin)]
        if matching_rows.size == 0:
            continue
        selected_row = matching_rows[0]
        row_tmin, row_tmax, amplitudes, energies = _extract_fit_row_parameters(selected_row, nstates)
        if row_tmin != int(tmin):
            raise ValueError(
                f"two-point fit table row tmin mismatch in {candidate}: expected {tmin}, found {row_tmin}"
            )
        if row_tmax != int(tmax):
            raise ValueError(
                f"two-point fit table row tmax mismatch in {candidate}: expected {tmax}, found {row_tmax}"
            )
        return TwoPointFitReference(
            path=candidate,
            tmin=int(tmin),
            tmax=int(tmax),
            amplitudes=amplitudes,
            energies=energies,
        )

    raise ValueError(
        f"could not find tmin={tmin}, tmax={tmax} in any two-point fit table matching title={title}, nstates={nstates}"
    )


def _normalize_time_axis(data: np.ndarray, nt: int) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("DA dataset must be a 2D array with configuration and time dimensions")
    if values.shape[1] == nt:
        return values
    if values.shape[0] == nt:
        return values.T
    raise ValueError(f"DA dataset shape {values.shape} is incompatible with Nt={nt}")


def _sign_pattern(nt: int) -> np.ndarray:
    pattern = np.ones(nt, dtype=float)
    pattern[nt // 2 :] = -1.0
    return pattern


def apply_da_preprocessing(values: np.ndarray, nt: int, gm: str) -> np.ndarray:
    processed = np.asarray(values, dtype=np.complex128).copy()
    operator = normalize_da_operator(gm)
    if operator == "T5":
        processed *= _sign_pattern(nt)[None, :]
    else:
        processed *= -1j
    return processed


def fold_symmetric_complex(values: np.ndarray, nt: int) -> np.ndarray:
    correlators = np.asarray(values, dtype=np.complex128)
    if correlators.ndim != 2 or correlators.shape[1] != nt:
        raise ValueError("complex correlators must have shape (n_cfg, Nt)")
    folded_extent = nt // 2 + 1
    folded = correlators[:, :folded_extent].copy()
    for t in range(1, folded_extent):
        partner = (nt - t) % nt
        if partner == t:
            continue
        folded[:, t] = 0.5 * (correlators[:, t] + correlators[:, partner])
    return folded


def load_da_correlator(
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
        raise ModuleNotFoundError("h5py is required to load DA HDF5 data")

    file_path = Path(h5_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    with h5py.File(file_path, "r") as handle:
        return _load_da_correlator_from_handle(
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


def _load_da_correlator_from_handle(
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
    return fold_symmetric_complex(apply_da_preprocessing(averaged, nt, gm), nt)
