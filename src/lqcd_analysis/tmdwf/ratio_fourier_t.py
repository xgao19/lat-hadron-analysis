from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common.bootstrap import bin_samples, bootstrap_indices, bootstrap_means
from ..common.parsing import parse_fold_t, parse_int_list_or_range, parse_optional_int, parse_tsrange
from ..common.utils import apply_fold_t
from ..two_point.io import load_correlator_csv
from .fit_nstate import sanitize_token
from .fourier import DEFAULT_INTERPOLATION_KIND, DEFAULT_X_VALUES, DEFAULT_ZSTEP_FM, compute_tmdwf_cosine_transform
from .io import _load_tmdwf_correlator_from_handle, expand_template, resolve_qtmdwf_h5_path
from .models import normalize_tmdwf_operator


@dataclass(frozen=True)
class TMDWFRatioFourierTInput:
    title_pattern: str
    ns: int
    nt: int
    lattice_spacing_fm: float
    pzlist: tuple[int, ...]
    gmlist: tuple[str, ...]
    etalist: tuple[str, ...]
    tdirlist: tuple[str, ...]
    bTlist: tuple[int, ...]
    bzlist: tuple[int, ...]
    component: str
    qtmdwf_h5: str
    dataset_path_template: str
    c2pt: str
    fold_t: str
    tsrange: tuple[int, int]
    binsize: int
    bootstrap_samples: int | None
    bootstrap_size: int | None
    seed: int
    x_values: np.ndarray
    zstep_fm: float
    interpolation_kind: str
    results_dir: Path


def _parse_x_values(entries: dict[str, list[str]]) -> np.ndarray:
    if "x_values" in entries:
        return np.asarray([float(token) for token in entries["x_values"]], dtype=float)
    if "x_range" in entries:
        tokens = entries["x_range"]
        if len(tokens) < 2:
            raise ValueError("x_range must provide: xmin xmax")
        count = int(entries.get("x_count", [str(DEFAULT_X_VALUES.size)])[0])
        if count < 2:
            raise ValueError("x_count must be at least 2")
        return np.linspace(float(tokens[0]), float(tokens[1]), count)
    return np.asarray(DEFAULT_X_VALUES, dtype=float)


def parse_tmdwf_ratio_fourier_t_input(
    path: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> TMDWFRatioFourierTInput:
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
    required = {
        "pzlist",
        "gmlist",
        "etalist",
        "Tdirlist",
        "component",
        "qtmdwf_h5",
        "dataset_path_template",
        "c2pt",
        "fold_t",
    }
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")
    if "bTlist" not in entries and "bTrange" not in entries:
        raise ValueError(f"missing required key in {file_path}: bTlist or bTrange")
    if "bzlist" not in entries and "bzrange" not in entries:
        raise ValueError(f"missing required key in {file_path}: bzlist or bzrange")

    component = entries["component"][0].lower()
    if component not in {"real", "imag"}:
        raise ValueError("component must be one of: real, imag")
    for gm in entries["gmlist"]:
        normalize_tmdwf_operator(gm)
    zstep_fm = float(entries.get("zstep_fm", [str(DEFAULT_ZSTEP_FM)])[0])
    if zstep_fm <= 0.0:
        raise ValueError("zstep_fm must be positive")

    return TMDWFRatioFourierTInput(
        title_pattern=first_tokens[0],
        ns=int(first_tokens[1]),
        nt=int(first_tokens[2]),
        lattice_spacing_fm=float(first_tokens[3]),
        pzlist=tuple(int(item) for item in entries["pzlist"]),
        gmlist=tuple(entries["gmlist"]),
        etalist=tuple(entries["etalist"]),
        tdirlist=tuple(entries["Tdirlist"]),
        bTlist=parse_int_list_or_range(entries, "bTlist", "bTrange"),
        bzlist=parse_int_list_or_range(entries, "bzlist", "bzrange"),
        component=component,
        qtmdwf_h5=entries["qtmdwf_h5"][0],
        dataset_path_template=entries["dataset_path_template"][0],
        c2pt=entries["c2pt"][0],
        fold_t=parse_fold_t(entries),
        tsrange=parse_tsrange(entries, int(first_tokens[2])),
        binsize=int(entries.get("binsize", ["1"])[0]),
        bootstrap_samples=parse_optional_int(entries.get("bootstrap_samples", ["auto"])[0]),
        bootstrap_size=parse_optional_int(entries.get("bootstrap_size", ["auto"])[0]),
        seed=int(entries.get("seed", ["2026"])[0]),
        x_values=_parse_x_values(entries),
        zstep_fm=zstep_fm,
        interpolation_kind=entries.get("interpolation_kind", [DEFAULT_INTERPOLATION_KIND])[0],
        results_dir=(
            Path(results_dir)
            if results_dir is not None
            else Path(entries.get("results_dir", [file_path.parent / "results_tmdwf_ratio_fourier_t"])[0])
        ),
    )


def _summarize_q_samples(q_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p16 = np.percentile(q_samples, 16.0, axis=0)
    p84 = np.percentile(q_samples, 84.0, axis=0)
    mean = 0.5 * (p16 + p84)
    err = 0.5 * (p84 - p16)
    return mean, err, p16, p84


def _write_ratio_fourier_t_outputs(
    output_root: Path,
    stem: str,
    *,
    q_samples: np.ndarray,
    times: np.ndarray,
    x_values: np.ndarray,
    pz: int,
    bT: int,
    component: str,
    lattice_spacing_fm: float,
    zstep_fm: float,
    interpolation_kind: str,
) -> list[Path]:
    tables_dir = output_root / "tables"
    samples_dir = output_root / "samples"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / f"{stem}_{component}_ratio_fourier_t.txt"
    sample_path = samples_dir / f"{stem}_{component}_ratio_fourier_t_samples.txt"

    q_mean, q_err, q_p16, q_p84 = _summarize_q_samples(q_samples)
    with table_path.open("w", encoding="utf-8") as handle:
        handle.write(f"pz {pz}\n")
        handle.write(f"bT {bT}\n")
        handle.write(f"component {component}\n")
        handle.write(f"lattice_spacing_fm {lattice_spacing_fm:.10e}\n")
        handle.write(f"zstep_fm {zstep_fm:.10e}\n")
        handle.write(f"interpolation_kind {interpolation_kind}\n")
        handle.write("t\tx\tq_mean\tq_err\tq_p16\tq_p84\n")
        for t_index, t_value in enumerate(times):
            for x_index, x_value in enumerate(x_values):
                handle.write(
                    "\t".join(
                        [
                            str(int(t_value)),
                            f"{x_value:.10e}",
                            f"{q_mean[t_index, x_index]:.10e}",
                            f"{q_err[t_index, x_index]:.10e}",
                            f"{q_p16[t_index, x_index]:.10e}",
                            f"{q_p84[t_index, x_index]:.10e}",
                        ]
                    )
                    + "\n"
                )

    with sample_path.open("w", encoding="utf-8") as handle:
        handle.write("sample_id\tt\tx\tq_sample\n")
        for sample_id, sample_values in enumerate(q_samples):
            for t_index, t_value in enumerate(times):
                for x_index, x_value in enumerate(x_values):
                    handle.write(
                        f"{sample_id}\t{int(t_value)}\t{x_value:.10e}\t{sample_values[t_index, x_index]:.10e}\n"
                    )
    return [table_path, sample_path]


def run_tmdwf_ratio_fourier_t_workflow(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_ratio_fourier_t_input(input_file, results_dir=results_dir)
    outputs: list[Path] = []

    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required to load TMDWF HDF5 data") from exc

    for pz in spec.pzlist:
        title = expand_template(spec.title_pattern, pz=pz)
        c2pt_path = expand_template(spec.c2pt, pz=pz)
        _, c2pt_raw = load_correlator_csv(c2pt_path)
        c2pt_processed = apply_fold_t(c2pt_raw, spec.nt, spec.fold_t)
        t0, t1 = spec.tsrange
        times = np.arange(t0, t1 + 1, dtype=int)
        denominator_binned = bin_samples(c2pt_processed[:, t0 : t1 + 1], binsize=spec.binsize)
        n_cfg = denominator_binned.shape[0]
        if n_cfg < 2:
            raise ValueError("bootstrap requires at least two samples")
        n_boot = n_cfg if spec.bootstrap_samples is None else spec.bootstrap_samples
        draw_size = n_cfg if spec.bootstrap_size is None else spec.bootstrap_size
        indices = bootstrap_indices(n_cfg, draw_size, seed=spec.seed, n_boot=n_boot)
        denominator_boot = bootstrap_means(denominator_binned, indices=indices)

        dataset_root = spec.results_dir / title
        dataset_root.mkdir(parents=True, exist_ok=True)
        for gm in spec.gmlist:
            qtmdwf_path = resolve_qtmdwf_h5_path(spec.qtmdwf_h5, pz=pz, gm=gm)
            with h5py.File(qtmdwf_path, "r") as qtmdwf_handle:
                for eta in spec.etalist:
                    for bT in spec.bTlist:
                        ratio_by_bz: list[np.ndarray] = []
                        for bz in spec.bzlist:
                            numerator = _load_tmdwf_correlator_from_handle(
                                qtmdwf_handle,
                                spec.dataset_path_template,
                                gm=gm,
                                eta=eta,
                                pz=pz,
                                tdirs=spec.tdirlist,
                                bT=bT,
                                bz=bz,
                                nt=spec.nt,
                                ns=spec.ns,
                                file_label=str(qtmdwf_path),
                            )[:, t0 : t1 + 1]
                            numerator_binned = bin_samples(numerator, binsize=spec.binsize)
                            if numerator_binned.shape != denominator_binned.shape:
                                raise ValueError("numerator and denominator must have matching post-binning shapes")
                            numerator_boot = bootstrap_means(numerator_binned, indices=indices)
                            ratio = np.divide(
                                numerator_boot,
                                denominator_boot,
                                out=np.full_like(numerator_boot, np.nan + 0.0j),
                                where=denominator_boot != 0.0,
                            )
                            ratio_by_bz.append(ratio)

                        ratio_samples = np.stack(ratio_by_bz, axis=1)
                        component_samples = np.real(ratio_samples) if spec.component == "real" else np.imag(ratio_samples)
                        q_by_t = []
                        for t_index in range(component_samples.shape[2]):
                            _, q_samples_t = compute_tmdwf_cosine_transform(
                                np.asarray(spec.bzlist, dtype=int),
                                component_samples[:, :, t_index],
                                pz=pz,
                                ns=spec.ns,
                                lattice_spacing_fm=spec.lattice_spacing_fm,
                                x_values=spec.x_values,
                                zstep_fm=spec.zstep_fm,
                                interpolation_kind=spec.interpolation_kind,
                            )
                            q_by_t.append(q_samples_t)
                        q_samples = np.stack(q_by_t, axis=1)
                        stem = f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_bT{bT}"
                        outputs.extend(
                            _write_ratio_fourier_t_outputs(
                                dataset_root,
                                stem,
                                q_samples=q_samples,
                                times=times,
                                x_values=spec.x_values,
                                pz=pz,
                                bT=bT,
                                component=spec.component,
                                lattice_spacing_fm=spec.lattice_spacing_fm,
                                zstep_fm=spec.zstep_fm,
                                interpolation_kind=spec.interpolation_kind,
                            )
                        )
    return outputs
