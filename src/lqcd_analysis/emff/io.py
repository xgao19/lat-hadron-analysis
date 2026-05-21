from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover
    h5py = None

from ..common.parsing import (
    load_control_file_entries,
    load_fit_window_table,
    parse_bool,
    parse_optional_int,
)


def expand_template(pattern: str, **kwargs: object) -> str:
    """Expand a template string with keyword arguments via str.format."""
    return pattern.format(**kwargs)


def resolve_emff_h5_path(
    file_pattern: str,
    *,
    src_gamma: str,
    pfx: int,
    pfy: int,
    pfz: int,
    tsep: int,
) -> Path:
    """Find a single HDF5 file matching the expanded pattern."""
    resolved = expand_template(
        file_pattern,
        src_gamma=src_gamma,
        pfx=pfx,
        pfy=pfy,
        pfz=pfz,
        tsep=tsep,
    )
    matches = sorted(glob.glob(resolved))
    if not matches:
        raise FileNotFoundError(f"no HDF5 file matches pattern: {resolved}")
    if len(matches) > 1:
        # Use the first match; warn about ambiguity
        pass
    return Path(matches[0])


def load_emff_correlator(
    h5_path: str | Path,
    dataset_path_template: str,
    *,
    insert_gamma: str,
    qx: int,
    qy: int,
    qz: int,
) -> np.ndarray:
    """Load 3-point correlator data from an EMFF HDF5 file.

    Returns complex128 array of shape (tsep+2, n_cfg).
    Rows 0..tsep: C_3pt(tsep, tau) for each insertion time.
    Row tsep+1: C_2pt(tsep) reference.
    """
    if h5py is None:
        raise ModuleNotFoundError("h5py is required to load EMFF HDF5 data")

    file_path = Path(h5_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    dataset_path = expand_template(
        dataset_path_template,
        insert_gamma=insert_gamma,
        qx=qx,
        qy=qy,
        qz=qz,
    )

    with h5py.File(file_path, "r") as handle:
        if dataset_path not in handle:
            raise KeyError(f"dataset not found in {file_path}: {dataset_path}")
        data = np.asarray(handle[dataset_path], dtype=np.complex128)
    return data


def load_emff_c2pt_correlator(
    h5_path: str | Path,
    *,
    sink_gamma: str,
    px: int,
    py: int,
    pz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a 2-point correlator from an EMFF HDF5 file.

    Returns ``(times, correlators)`` where correlators has shape ``(n_cfg, Nt)``.
    """
    if h5py is None:
        raise ModuleNotFoundError("h5py is required to load EMFF HDF5 data")

    file_path = Path(h5_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    dataset_path = f"SS/{sink_gamma}/PX{px}PY{py}PZ{pz}"
    with h5py.File(file_path, "r") as handle:
        if dataset_path not in handle:
            raise KeyError(f"dataset not found in {file_path}: {dataset_path}")
        data = np.asarray(handle[dataset_path], dtype=np.complex128)

    if data.ndim != 2:
        raise ValueError(
            f"{file_path}:{dataset_path} must be a 2D array, got shape {data.shape}"
        )

    times = np.arange(data.shape[0], dtype=int)
    correlators = data.T
    return times, correlators


def _parse_int_list_or_range_single(
    entries: dict[str, list[str]],
    list_key: str,
    range_key: str,
    key_label: str,
) -> tuple[int, ...]:
    """Parse a list or range key, raising a unified error message."""
    if list_key in entries:
        return tuple(int(item) for item in entries[list_key])
    if range_key in entries:
        start, stop = (int(item) for item in entries[range_key][:2])
        step = 1 if stop >= start else -1
        return tuple(range(start, stop + step, step))
    raise ValueError(f"missing required key: {list_key} or {range_key} ({key_label})")


@dataclass(frozen=True)
class EMFFNStateInput:
    title_pattern: str
    ns: int
    nt: int
    lattice_spacing_fm: float
    hadron_mass_gev: float
    src_gamma: str
    sink_gamma: str
    insert_gamma: str
    nstates: tuple[int, ...]
    c2pt: str
    c3pt_h5: str
    c3pt_dataset_path: str
    pflist: tuple[int, int, int]
    qxlist: tuple[int, ...]
    qylist: tuple[int, ...]
    qzlist: tuple[int, ...]
    tslist: tuple[int, ...]
    average_transverse_orbits: bool
    fit_method: str
    tau_range: tuple[int, int]
    tsep_range: tuple[int, int]
    binsize: int
    bootstrap_samples: int | None
    bootstrap_size: int | None
    seed: int
    two_point_fit_root: str
    two_point_fit_window_by_pz: dict[int, tuple[int, int]]
    make_plots: bool
    results_dir: Path


def parse_emff_fit_input(
    path: str | Path,
    results_dir: str | Path | None = None,
) -> EMFFNStateInput:
    """Parse an EMFF n-state fit control file."""
    file_path = Path(path)
    first_tokens, entries = load_control_file_entries(file_path)

    if len(first_tokens) < 4:
        raise ValueError("the first non-empty line must be: title_pattern Ns Nt a_fm")

    required = {
        "c2pt",
        "c3pt_h5",
        "c3pt_dataset_path",
        "hadron_mass_gev",
        "src_gamma",
        "sink_gamma",
        "insert_gamma",
        "nstates",
        "pflist",
        "tslist",
        "fit_method",
        "tau_range",
        "tsep_range",
        "two_point_fit_root",
        "two_point_fit_window_by_pz",
    }
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")

    # Validate fit_method
    fit_method = entries["fit_method"][0].lower()
    if fit_method not in {"2state", "summation", "plateau"}:
        raise ValueError("fit_method must be one of: 2state, summation, plateau")

    # Validate nstates
    nstates = tuple(sorted({int(item) for item in entries["nstates"]}))
    if not nstates or any(state not in {1, 2} for state in nstates):
        raise ValueError("EMFF nstates must contain only 1 and/or 2")

    # Parse momentum lists
    pflist = tuple(int(item) for item in entries["pflist"])
    if len(pflist) != 3:
        raise ValueError("pflist must contain exactly 3 values: pfx pfy pfz")

    qxlist = _parse_int_list_or_range_single(entries, "qxlist", "qxrange", "qx")
    qylist = _parse_int_list_or_range_single(entries, "qylist", "qyrange", "qy")
    qzlist = _parse_int_list_or_range_single(entries, "qzlist", "qzrange", "qz")

    tslist = _parse_int_list_or_range_single(entries, "tslist", "tsrange_emff", "tsep")
    if not tslist:
        raise ValueError("tslist must contain at least one time separation")

    # Parse fit ranges
    tau_range = (int(entries["tau_range"][0]), int(entries["tau_range"][1]))
    tsep_range = (int(entries["tsep_range"][0]), int(entries["tsep_range"][1]))

    # Two-point fit reference
    two_point_fit_window_path = Path(entries["two_point_fit_window_by_pz"][0])
    two_point_fit_window_by_pz = load_fit_window_table(two_point_fit_window_path)
    # Filter to entries without gm (gm=None)
    two_point_fit_window_by_pz = {
        pz: window
        for (gm, pz), window in two_point_fit_window_by_pz.items()
        if gm is None
    }

    return EMFFNStateInput(
        title_pattern=first_tokens[0],
        ns=int(first_tokens[1]),
        nt=int(first_tokens[2]),
        lattice_spacing_fm=float(first_tokens[3]),
        hadron_mass_gev=float(entries["hadron_mass_gev"][0]),
        src_gamma=entries["src_gamma"][0],
        sink_gamma=entries["sink_gamma"][0],
        insert_gamma=entries["insert_gamma"][0],
        nstates=nstates,
        c2pt=entries["c2pt"][0],
        c3pt_h5=entries["c3pt_h5"][0],
        c3pt_dataset_path=entries["c3pt_dataset_path"][0],
        pflist=pflist,
        qxlist=qxlist,
        qylist=qylist,
        qzlist=qzlist,
        tslist=tslist,
        average_transverse_orbits=parse_bool(
            entries.get("average_transverse_orbits", ["true"])[0]
        ),
        fit_method=fit_method,
        tau_range=tau_range,
        tsep_range=tsep_range,
        binsize=int(entries.get("binsize", ["1"])[0]),
        bootstrap_samples=parse_optional_int(
            entries.get("bootstrap_samples", ["auto"])[0]
        ),
        bootstrap_size=parse_optional_int(
            entries.get("bootstrap_size", ["auto"])[0]
        ),
        seed=int(entries.get("seed", ["2026"])[0]),
        two_point_fit_root=entries["two_point_fit_root"][0],
        two_point_fit_window_by_pz=two_point_fit_window_by_pz,
        make_plots=parse_bool(entries.get("plot", ["false"])[0]),
        results_dir=(
            Path(entries["results_dir"][0])
            if "results_dir" in entries and results_dir is None
            else (
                (file_path.resolve().parent / "results_emff")
                if results_dir is None
                else Path(results_dir)
            )
        ),
    )
