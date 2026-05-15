from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common.parsing import parse_int_list_or_range
from .fit_nstate import sanitize_token
from .fourier import DEFAULT_INTERPOLATION_KIND, DEFAULT_ZSTEP_FM, compute_da_cosine_transform
from .io import expand_template
from .normalize import _compute_normalized_sample, _load_dataset_rows_and_samples, _summarize_percentile
from ..two_point.plotting import prepare_matplotlib, save_plot_status


@dataclass(frozen=True)
class DAXFitNormalizeInput:
    title_pattern: str
    input_root: Path
    bare_matrix_root: Path
    ns: int
    lattice_spacing_fm: float
    pzlist: tuple[int, ...]
    gmlist: tuple[str, ...]
    etalist: tuple[str, ...]
    bTlist: tuple[int, ...]
    bzlist: tuple[int, ...]
    component: str
    nstates: int
    normalization_mode: str
    zstep_fm: float
    interpolation_kind: str
    results_dir: Path


def parse_da_xfit_normalize_input(
    path: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> DAXFitNormalizeInput:
    file_path = Path(path)
    entries: dict[str, list[str]] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            entries[tokens[0]] = tokens[1:]

    required = {
        "title_pattern",
        "input_root",
        "bare_matrix_root",
        "ns",
        "lattice_spacing_fm",
        "pzlist",
        "gmlist",
        "etalist",
        "component",
        "nstates",
        "normalization_mode",
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
    mode = entries["normalization_mode"][0].lower()
    if mode not in {"mode1", "mode2", "mode3"}:
        raise ValueError("normalization_mode must be one of: mode1, mode2, mode3")
    zstep_fm = float(entries.get("zstep_fm", [str(DEFAULT_ZSTEP_FM)])[0])
    if zstep_fm <= 0.0:
        raise ValueError("zstep_fm must be positive")

    return DAXFitNormalizeInput(
        title_pattern=entries["title_pattern"][0],
        input_root=Path(entries["input_root"][0]),
        bare_matrix_root=Path(entries["bare_matrix_root"][0]),
        ns=int(entries["ns"][0]),
        lattice_spacing_fm=float(entries["lattice_spacing_fm"][0]),
        pzlist=tuple(int(item) for item in entries["pzlist"]),
        gmlist=tuple(entries["gmlist"]),
        etalist=tuple(entries["etalist"]),
        bTlist=parse_int_list_or_range(entries, "bTlist", "bTrange"),
        bzlist=parse_int_list_or_range(entries, "bzlist", "bzrange"),
        component=component,
        nstates=int(entries["nstates"][0]),
        normalization_mode=mode,
        zstep_fm=zstep_fm,
        interpolation_kind=entries.get("interpolation_kind", [DEFAULT_INTERPOLATION_KIND])[0],
        results_dir=(
            Path(results_dir)
            if results_dir is not None
            else Path(entries.get("results_dir", [file_path.parent / "results_da_xfit_normalized"])[0])
        ),
    )


def _load_xfit_samples(
    input_root: Path,
    title: str,
    gm: str,
    eta: str,
    bT: int,
    component: str,
    nstates: int,
) -> tuple[np.ndarray, dict[int, dict[float, tuple[int, float]]]]:
    stem = f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_bT{bT}_{component}_{nstates}state_xfit_samples.txt"
    path = input_root / title / "samples" / stem
    if not path.exists():
        raise FileNotFoundError(f"DA x-fit sample table does not exist: {path}")
    header: list[str] | None = None
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split("\t")
            if header is None:
                header = tokens
            else:
                rows.append(tokens)
    if header is None:
        raise ValueError(f"x-fit sample table is empty: {path}")
    index = {name: idx for idx, name in enumerate(header)}
    required = {"sample_id", "x", "success", "q0"}
    missing = required - index.keys()
    if missing:
        raise ValueError(f"x-fit sample table is missing columns: {sorted(missing)}")
    x_values = np.asarray(sorted({float(row[index["x"]]) for row in rows}), dtype=float)
    sample_map: dict[int, dict[float, tuple[int, float]]] = {}
    for row in rows:
        sample_map.setdefault(int(row[index["sample_id"]]), {})[float(row[index["x"]])] = (
            int(row[index["success"]]),
            float(row[index["q0"]]),
        )
    return x_values, sample_map


def _old_fourier_samples_from_bare_matrix_outputs(
    spec: DAXFitNormalizeInput,
    *,
    title: str,
    title_pz0: str,
    gm: str,
    eta: str,
    pz: int,
    bT: int,
    x_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    _, _, target_samples, _, _, _ = _load_dataset_rows_and_samples(
        spec.bare_matrix_root,
        title,
        gm,
        eta,
        bT,
        spec.component,
        spec.nstates,
    )
    mode1_samples = mode2_samples = mode3_samples = None
    if spec.normalization_mode in {"mode1", "mode3"}:
        _, _, mode1_samples, _, _, _ = _load_dataset_rows_and_samples(
            spec.bare_matrix_root,
            title,
            gm,
            eta,
            0,
            spec.component,
            spec.nstates,
        )
    if spec.normalization_mode in {"mode2", "mode3"}:
        _, _, mode2_samples, _, _, _ = _load_dataset_rows_and_samples(
            spec.bare_matrix_root,
            title_pz0,
            gm,
            eta,
            bT,
            spec.component,
            spec.nstates,
        )
    if spec.normalization_mode == "mode3":
        _, _, mode3_samples, _, _, _ = _load_dataset_rows_and_samples(
            spec.bare_matrix_root,
            title_pz0,
            gm,
            eta,
            0,
            spec.component,
            spec.nstates,
        )

    sample_ids = sorted(target_samples)
    raw_rows: list[list[float]] = []
    normalized_rows: list[list[float]] = []
    valid_sample_ids: list[int] = []
    for sample_id in sample_ids:
        raw_by_bz: list[float] = []
        norm_by_bz: list[float] = []
        valid = True
        for bz in spec.bzlist:
            target_entry = target_samples.get(sample_id, {}).get(bz)
            if target_entry is None or target_entry[0] != 1 or not np.isfinite(target_entry[1]):
                valid = False
                break
            ref1 = ref2 = ref3 = None
            if mode1_samples is not None:
                ref_entry = mode1_samples.get(sample_id, {}).get(0)
                if ref_entry is None or ref_entry[0] != 1 or not np.isfinite(ref_entry[1]) or ref_entry[1] == 0.0:
                    valid = False
                    break
                ref1 = ref_entry[1]
            if mode2_samples is not None:
                ref_entry = mode2_samples.get(sample_id, {}).get(0)
                if ref_entry is None or ref_entry[0] != 1 or not np.isfinite(ref_entry[1]) or ref_entry[1] == 0.0:
                    valid = False
                    break
                ref2 = ref_entry[1]
            if mode3_samples is not None:
                ref_entry = mode3_samples.get(sample_id, {}).get(0)
                if ref_entry is None or ref_entry[0] != 1 or not np.isfinite(ref_entry[1]) or ref_entry[1] == 0.0:
                    valid = False
                    break
                ref3 = ref_entry[1]
            normalized = _compute_normalized_sample(spec.normalization_mode, target_entry[1], ref1, ref2, ref3)
            raw_by_bz.append(float(target_entry[1].real if spec.component == "real" else target_entry[1].imag))
            norm_by_bz.append(float(normalized.real if spec.component == "real" else normalized.imag))
        if valid:
            raw_rows.append(raw_by_bz)
            normalized_rows.append(norm_by_bz)
            valid_sample_ids.append(sample_id)

    if not raw_rows:
        raise ValueError(f"no valid bare matrix-element samples for title={title}, gm={gm}, eta={eta}, bT={bT}")
    bz_values = np.asarray(spec.bzlist, dtype=int)
    _, raw_fourier = compute_da_cosine_transform(
        bz_values,
        np.asarray(raw_rows, dtype=float),
        pz=pz,
        ns=spec.ns,
        lattice_spacing_fm=spec.lattice_spacing_fm,
        x_values=x_values,
        zstep_fm=spec.zstep_fm,
        interpolation_kind=spec.interpolation_kind,
    )
    _, norm_fourier = compute_da_cosine_transform(
        bz_values,
        np.asarray(normalized_rows, dtype=float),
        pz=pz,
        ns=spec.ns,
        lattice_spacing_fm=spec.lattice_spacing_fm,
        x_values=x_values,
        zstep_fm=spec.zstep_fm,
        interpolation_kind=spec.interpolation_kind,
    )
    return raw_fourier, norm_fourier, valid_sample_ids


def _write_normalized_xfit_outputs(
    output_root: Path,
    stem: str,
    *,
    x_values: np.ndarray,
    sample_rows: list[tuple[float, int, int, float]],
    component: str,
    nstates: int,
    normalization_mode: str,
) -> list[Path]:
    tables_dir = output_root / "tables"
    samples_dir = output_root / "samples"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / f"{stem}_{normalization_mode}_{component}_{nstates}state_xfit.txt"
    sample_path = samples_dir / f"{stem}_{normalization_mode}_{component}_{nstates}state_xfit_samples.txt"

    by_x: dict[float, list[float]] = {}
    for x_value, _, success, value in sample_rows:
        if success:
            by_x.setdefault(x_value, []).append(value)
    with table_path.open("w", encoding="utf-8") as handle:
        handle.write(f"normalization_mode {normalization_mode}\n")
        handle.write("x\tq0_mean\tq0_err\n")
        for x_value in x_values:
            center, err = _summarize_percentile(np.asarray(by_x.get(float(x_value), []), dtype=float))
            handle.write(f"{x_value:.10e}\t{center:.10e}\t{err:.10e}\n")

    with sample_path.open("w", encoding="utf-8") as handle:
        handle.write(f"normalization_mode {normalization_mode}\n")
        handle.write("sample_id\tx\tsuccess\tq0\n")
        for x_value, sample_id, success, value in sample_rows:
            handle.write(f"{sample_id}\t{x_value:.10e}\t{success}\t{value:.10e}\n")
    return [table_path, sample_path]


def plot_da_xfit_normalized_x_dependence(
    output_path: str | Path,
    *,
    x_values: np.ndarray,
    q0_mean: np.ndarray,
    q0_err: np.ndarray,
    title: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = prepare_matplotlib()
    if plt is None:
        return save_plot_status(output_path.with_suffix(".txt"), "matplotlib not installed; plot was skipped")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(x_values, q0_mean, yerr=q0_err, fmt="o-", ms=3, linewidth=1.1, color="C0")
    ax.set_xlabel("x")
    ax.set_ylabel("normalized q0(x)")
    ax.tick_params(direction="in", top=True, right=True)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def run_da_xfit_normalization(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_da_xfit_normalize_input(input_file, results_dir=results_dir)
    outputs: list[Path] = []
    for pz in spec.pzlist:
        title = expand_template(spec.title_pattern, pz=pz)
        title_pz0 = expand_template(spec.title_pattern, pz=0)
        dataset_root = spec.results_dir / title
        for gm in spec.gmlist:
            for eta in spec.etalist:
                for bT in spec.bTlist:
                    x_values, xfit_samples = _load_xfit_samples(
                        spec.input_root,
                        title,
                        gm,
                        eta,
                        bT,
                        spec.component,
                        spec.nstates,
                    )
                    old_raw, old_norm, old_sample_ids = _old_fourier_samples_from_bare_matrix_outputs(
                        spec,
                        title=title,
                        title_pz0=title_pz0,
                        gm=gm,
                        eta=eta,
                        pz=pz,
                        bT=bT,
                        x_values=x_values,
                    )
                    factor_by_sample = {
                        sample_id: np.divide(
                            old_norm[row_index],
                            old_raw[row_index],
                            out=np.full_like(old_norm[row_index], np.nan),
                            where=old_raw[row_index] != 0.0,
                        )
                        for row_index, sample_id in enumerate(old_sample_ids)
                    }
                    sample_rows: list[tuple[float, int, int, float]] = []
                    for sample_id, by_x in sorted(xfit_samples.items()):
                        factors = factor_by_sample.get(sample_id)
                        for x_index, x_value in enumerate(x_values):
                            entry = by_x.get(float(x_value))
                            if entry is None or entry[0] != 1 or factors is None or not np.isfinite(factors[x_index]):
                                sample_rows.append((float(x_value), sample_id, 0, np.nan))
                                continue
                            value = entry[1] * factors[x_index]
                            success = int(np.isfinite(value))
                            sample_rows.append((float(x_value), sample_id, success, float(value)))
                    stem = f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_bT{bT}"
                    outputs.extend(
                        _write_normalized_xfit_outputs(
                            dataset_root,
                            stem,
                            x_values=x_values,
                            sample_rows=sample_rows,
                            component=spec.component,
                            nstates=spec.nstates,
                            normalization_mode=spec.normalization_mode,
                        )
                    )
                    by_x: dict[float, list[float]] = {}
                    for x_value, _, success, value in sample_rows:
                        if success:
                            by_x.setdefault(x_value, []).append(value)
                    q0_mean = []
                    q0_err = []
                    for x_value in x_values:
                        center, err = _summarize_percentile(np.asarray(by_x.get(float(x_value), []), dtype=float))
                        q0_mean.append(center)
                        q0_err.append(err)
                    outputs.append(
                        plot_da_xfit_normalized_x_dependence(
                            dataset_root
                            / "plots"
                            / f"{stem}_{spec.normalization_mode}_{spec.component}_{spec.nstates}state_xfit.pdf",
                            x_values=x_values,
                            q0_mean=np.asarray(q0_mean, dtype=float),
                            q0_err=np.asarray(q0_err, dtype=float),
                            title=f"{title} {gm} {eta} bT{bT} {spec.normalization_mode} {spec.component} {spec.nstates}state",
                        )
                    )
    return outputs
