from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    h5py = None


def load_array(path: str | Path, dataset: str | None = None) -> np.ndarray:
    """Load an array from .npy or HDF5 data."""
    file_path = Path(path)
    if file_path.suffix == ".npy":
        return np.load(file_path)
    if file_path.suffix in {".h5", ".hdf5"}:
        if h5py is None:
            raise ModuleNotFoundError("h5py is required to load HDF5 files")
        if dataset is None:
            raise ValueError("dataset must be provided for HDF5 input")
        with h5py.File(file_path, "r") as handle:
            return np.asarray(handle[dataset])
    raise ValueError(f"unsupported file format: {file_path.suffix}")


def load_correlator_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a correlator CSV with columns: t, cfg_0, cfg_1, ..."""
    file_path = Path(path)
    raw = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"invalid correlator CSV format: {file_path}")

    times = raw[:, 0].astype(int)
    values = raw[:, 1:].T
    if np.isnan(values).any():
        raise ValueError(f"NaN values found in correlator CSV: {file_path}")
    return times, np.asarray(values, dtype=float)
