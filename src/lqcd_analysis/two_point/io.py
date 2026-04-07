from __future__ import annotations

from pathlib import Path

import numpy as np


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
