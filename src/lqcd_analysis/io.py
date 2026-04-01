from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_array(path: str | Path, dataset: str | None = None) -> np.ndarray:
    """Load an array from .npy or HDF5 data."""
    file_path = Path(path)
    if file_path.suffix == ".npy":
        return np.load(file_path)
    if file_path.suffix in {".h5", ".hdf5"}:
        if dataset is None:
            raise ValueError("dataset must be provided for HDF5 input")
        with h5py.File(file_path, "r") as handle:
            return np.asarray(handle[dataset])
    raise ValueError(f"unsupported file format: {file_path.suffix}")

