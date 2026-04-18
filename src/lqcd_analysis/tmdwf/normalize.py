from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fit_nstate import _parse_int_list_or_range
from .io import expand_template
from .plotting import plot_tmdwf_m0_from_fit_tables


@dataclass(frozen=True)
class TMDWFNormalizeInput:
    title_pattern: str
    input_root: Path
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
    make_plots: bool
    results_dir: Path


def parse_tmdwf_normalize_input(
    path: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> TMDWFNormalizeInput:
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
    normalization_mode = entries["normalization_mode"][0].lower()
    if normalization_mode not in {"mode1", "mode2", "mode3"}:
        raise ValueError("normalization_mode must be one of: mode1, mode2, mode3")

    output_root = (
        Path(results_dir)
        if results_dir is not None
        else Path(entries.get("results_dir", [file_path.parent / "results_tmdwf_normalized"])[0])
    )
    return TMDWFNormalizeInput(
        title_pattern=entries["title_pattern"][0],
        input_root=Path(entries["input_root"][0]),
        ns=int(entries["ns"][0]),
        lattice_spacing_fm=float(entries["lattice_spacing_fm"][0]),
        pzlist=tuple(int(item) for item in entries["pzlist"]),
        gmlist=tuple(entries["gmlist"]),
        etalist=tuple(entries["etalist"]),
        bTlist=_parse_int_list_or_range(entries, "bTlist", "bTrange"),
        bzlist=_parse_int_list_or_range(entries, "bzlist", "bzrange"),
        component=component,
        nstates=int(entries["nstates"][0]),
        normalization_mode=normalization_mode,
        make_plots=entries.get("plot", ["false"])[0].lower() not in {"false", "0", "no"},
        results_dir=output_root,
    )


def _parse_grouped_table(path: str | Path) -> tuple[dict[str, str], list[str], list[list[str]]]:
    metadata: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                tokens = line.split("\t")
                if header is None:
                    header = tokens
                else:
                    rows.append(tokens)
            else:
                key, _, value = line.partition(" ")
                metadata[key] = value
    if header is None:
        raise ValueError(f"grouped table is missing a tab-delimited header: {path}")
    return metadata, header, rows


def _load_fit_rows_by_bz(path: str | Path) -> tuple[list[str], dict[int, dict[str, str]], dict[str, str]]:
    metadata, header, rows = _parse_grouped_table(path)
    index = {name: idx for idx, name in enumerate(header)}
    required = {"bz", "m0_mean", "m0_err"}
    missing = required - index.keys()
    if missing:
        raise ValueError(f"grouped fit table is missing columns: {sorted(missing)}")
    row_map: dict[int, dict[str, str]] = {}
    for row in rows:
        row_dict = {name: row[idx] for name, idx in index.items()}
        row_map[int(row_dict["bz"])] = row_dict
    return header, row_map, metadata


def _load_sample_rows(path: str | Path) -> tuple[list[str], dict[int, dict[int, tuple[int, float]]]]:
    _, header, rows = _parse_grouped_table(path)
    index = {name: idx for idx, name in enumerate(header)}
    required = {"bz", "sample_id", "success", "m0"}
    missing = required - index.keys()
    if missing:
        raise ValueError(f"grouped sample table is missing columns: {sorted(missing)}")
    sample_map: dict[int, dict[int, tuple[int, float]]] = {}
    for row in rows:
        bz = int(row[index["bz"]])
        sample_id = int(row[index["sample_id"]])
        success = int(row[index["success"]])
        value = float(row[index["m0"]])
        sample_map.setdefault(sample_id, {})[bz] = (success, value)
    return header, sample_map


def _normalized_fit_header(target_header: list[str]) -> list[str]:
    preferred = [
        "bz",
        "tmin",
        "tmax",
        "success_meanfit",
        "chi2_dof",
        "pvalue",
        "shared_window_flag",
        "reference_eta",
        "reference_bT",
        "reference_bz",
        "fit_window_tmax_used",
        "m0_mean",
        "m0_err",
    ]
    return [name for name in preferred if name in target_header]


def _resolve_fit_sample_paths(
    input_root: Path,
    title: str,
    gm: str,
    eta: str,
    bT: int,
    component: str,
    nstates: int,
) -> tuple[Path, Path]:
    stem = f"{title}_{gm}_{eta}_bT{bT}_{component}_{nstates}state"
    fit_path = input_root / title / "tables" / f"{stem}_fit.txt"
    sample_path = input_root / title / "samples" / f"{stem}_samples.txt"
    if not fit_path.exists():
        raise FileNotFoundError(f"TMDWF normalization fit table does not exist: {fit_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"TMDWF normalization sample table does not exist: {sample_path}")
    return fit_path, sample_path


def _try_resolve_fit_sample_paths(
    input_root: Path,
    title: str,
    gm: str,
    eta: str,
    bT: int,
    component: str,
    nstates: int,
) -> tuple[Path, Path] | None:
    try:
        return _resolve_fit_sample_paths(input_root, title, gm, eta, bT, component, nstates)
    except FileNotFoundError:
        return None


def _combine_complex_samples(
    real_samples: dict[int, dict[int, tuple[int, float]]],
    imag_samples: dict[int, dict[int, tuple[int, float]]],
) -> dict[int, dict[int, tuple[int, complex]]]:
    combined: dict[int, dict[int, tuple[int, complex]]] = {}
    sample_ids = sorted(set(real_samples) | set(imag_samples))
    for sample_id in sample_ids:
        by_bz_real = real_samples.get(sample_id, {})
        by_bz_imag = imag_samples.get(sample_id, {})
        bz_values = sorted(set(by_bz_real) | set(by_bz_imag))
        for bz in bz_values:
            real_entry = by_bz_real.get(bz)
            imag_entry = by_bz_imag.get(bz)
            if real_entry is None or imag_entry is None:
                continue
            real_success, real_value = real_entry
            imag_success, imag_value = imag_entry
            success = int(
                real_success == 1
                and imag_success == 1
                and np.isfinite(real_value)
                and np.isfinite(imag_value)
            )
            combined.setdefault(sample_id, {})[bz] = (
                success,
                complex(real_value, imag_value) if success == 1 else complex(np.nan, np.nan),
            )
    return combined


def _load_dataset_rows_and_samples(
    input_root: Path,
    title: str,
    gm: str,
    eta: str,
    bT: int,
    component: str,
    nstates: int,
) -> tuple[list[str], dict[int, dict[str, str]], dict[int, dict[int, tuple[int, complex]]], Path, Path, bool]:
    fit_path, sample_path = _resolve_fit_sample_paths(input_root, title, gm, eta, bT, component, nstates)
    header, row_map, _ = _load_fit_rows_by_bz(fit_path)
    _, sample_map = _load_sample_rows(sample_path)

    complex_sample_map: dict[int, dict[int, tuple[int, complex]]] = {
        sample_id: {
            bz: (success, complex(value, 0.0))
            for bz, (success, value) in by_bz.items()
        }
        for sample_id, by_bz in sample_map.items()
    }
    used_complex_samples = False

    counterpart = "imag" if component == "real" else "real"
    counterpart_paths = _try_resolve_fit_sample_paths(input_root, title, gm, eta, bT, counterpart, nstates)
    if counterpart_paths is not None:
        _, counterpart_sample_map = _load_sample_rows(counterpart_paths[1])
        if component == "real":
            complex_sample_map = _combine_complex_samples(sample_map, counterpart_sample_map)
        else:
            complex_sample_map = _combine_complex_samples(counterpart_sample_map, sample_map)
        used_complex_samples = True

    return header, row_map, complex_sample_map, fit_path, sample_path, used_complex_samples


def _require_reference_bz0(
    row_map: dict[int, dict[str, str]],
    sample_map: dict[int, dict[int, tuple[int, complex]]],
    *,
    label: str,
) -> None:
    if 0 not in row_map:
        raise ValueError(f"TMDWF normalization missing bz=0 reference row for {label}")
    available = any(0 in by_bz for by_bz in sample_map.values())
    if not available:
        raise ValueError(f"TMDWF normalization missing bz=0 reference samples for {label}")


def _summarize_percentile(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    p16 = float(np.percentile(values, 16.0))
    p84 = float(np.percentile(values, 84.0))
    return 0.5 * (p16 + p84), 0.5 * (p84 - p16)


def _compute_normalized_sample(
    mode: str,
    target_value: complex,
    ref_same_pz_bt0: complex | None,
    ref_pz0_same_bt: complex | None,
    ref_pz0_bt0: complex | None,
) -> complex:
    if mode == "mode1":
        assert ref_same_pz_bt0 is not None
        return target_value / ref_same_pz_bt0
    if mode == "mode2":
        assert ref_pz0_same_bt is not None
        return target_value / ref_pz0_same_bt
    assert ref_same_pz_bt0 is not None and ref_pz0_same_bt is not None and ref_pz0_bt0 is not None
    return (target_value / ref_same_pz_bt0) / (ref_pz0_same_bt / ref_pz0_bt0)


def _write_normalized_outputs(
    output_root: Path,
    stem: str,
    *,
    normalization_mode: str,
    fit_header: list[str],
    fit_rows: list[dict[str, str]],
    sample_rows: list[tuple[int, int, int, float]],
    metadata_lines: list[str],
) -> list[Path]:
    tables_dir = output_root / "tables"
    samples_dir = output_root / "samples"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"{stem}_summary.txt"
    fit_path = tables_dir / f"{stem}_fit.txt"
    sample_path = samples_dir / f"{stem}_samples.txt"

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"normalization_mode {normalization_mode}\n")
        for line in metadata_lines:
            handle.write(line + "\n")
        for row in fit_rows:
            handle.write(f"begin_bz {row['bz']}\n")
            handle.write(f"m0 {float(row['m0_mean']):.10e} {float(row['m0_err']):.10e}\n")
            handle.write(f"end_bz {row['bz']}\n")

    with fit_path.open("w", encoding="utf-8") as handle:
        handle.write(f"normalization_mode {normalization_mode}\n")
        for line in metadata_lines:
            handle.write(line + "\n")
        handle.write("\t".join(fit_header) + "\n")
        for row in fit_rows:
            handle.write("\t".join(row[name] for name in fit_header) + "\n")

    with sample_path.open("w", encoding="utf-8") as handle:
        handle.write(f"normalization_mode {normalization_mode}\n")
        for line in metadata_lines:
            handle.write(line + "\n")
        handle.write("bz\tsample_id\tsuccess\tm0\n")
        for bz, sample_id, success, value in sample_rows:
            handle.write(f"{bz}\t{sample_id}\t{success}\t{value:.10e}\n")

    return [summary_path, fit_path, sample_path]


def run_tmdwf_normalization(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_normalize_input(input_file, results_dir=results_dir)
    outputs: list[Path] = []

    for pz in spec.pzlist:
        title = expand_template(spec.title_pattern, pz=pz)
        title_pz0 = expand_template(spec.title_pattern, pz=0)
        for gm in spec.gmlist:
            for eta in spec.etalist:
                fit_tables_by_bT: dict[int, Path] = {}
                for bT in spec.bTlist:
                    target_header, target_rows, target_samples, target_fit_path, target_sample_path, target_used_complex = _load_dataset_rows_and_samples(
                        spec.input_root,
                        title,
                        gm,
                        eta,
                        bT,
                        spec.component,
                        spec.nstates,
                    )

                    mode1_fit_rows = mode1_samples = None
                    mode2_fit_rows = mode2_samples = None
                    mode3_fit_rows = mode3_samples = None
                    reference_uses_complex = target_used_complex

                    if spec.normalization_mode in {"mode1", "mode3"}:
                        (
                            _,
                            mode1_fit_rows,
                            mode1_samples,
                            _mode1_fit_path,
                            _mode1_sample_path,
                            mode1_used_complex,
                        ) = _load_dataset_rows_and_samples(
                            spec.input_root,
                            title,
                            gm,
                            eta,
                            0,
                            spec.component,
                            spec.nstates,
                        )
                        reference_uses_complex = reference_uses_complex or mode1_used_complex
                        _require_reference_bz0(mode1_fit_rows, mode1_samples, label=f"pz={pz} gm={gm} eta={eta} bT=0")
                    if spec.normalization_mode in {"mode2", "mode3"}:
                        (
                            _,
                            mode2_fit_rows,
                            mode2_samples,
                            _mode2_fit_path,
                            _mode2_sample_path,
                            mode2_used_complex,
                        ) = _load_dataset_rows_and_samples(
                            spec.input_root,
                            title_pz0,
                            gm,
                            eta,
                            bT,
                            spec.component,
                            spec.nstates,
                        )
                        reference_uses_complex = reference_uses_complex or mode2_used_complex
                        _require_reference_bz0(mode2_fit_rows, mode2_samples, label=f"pz=0 gm={gm} eta={eta} bT={bT}")
                    if spec.normalization_mode == "mode3":
                        assert mode2_fit_rows is not None and mode2_samples is not None
                        (
                            _,
                            mode3_fit_rows,
                            mode3_samples,
                            _mode3_fit_path,
                            _mode3_sample_path,
                            mode3_used_complex,
                        ) = _load_dataset_rows_and_samples(
                            spec.input_root,
                            title_pz0,
                            gm,
                            eta,
                            0,
                            spec.component,
                            spec.nstates,
                        )
                        reference_uses_complex = reference_uses_complex or mode3_used_complex
                        _require_reference_bz0(mode3_fit_rows, mode3_samples, label=f"pz=0 gm={gm} eta={eta} bT=0")

                    fit_rows_out: list[dict[str, str]] = []
                    sample_rows_out: list[tuple[int, int, int, float]] = []
                    all_sample_ids = sorted(target_samples)
                    for bz in spec.bzlist:
                        if bz not in target_rows:
                            raise ValueError(
                                f"TMDWF normalization missing target bz={bz} in {target_fit_path}"
                            )
                        normalized_values: list[float] = []
                        for sample_id in all_sample_ids:
                            target_entry = target_samples.get(sample_id, {}).get(bz)
                            if target_entry is None or target_entry[0] != 1 or not np.isfinite(target_entry[1]):
                                sample_rows_out.append((bz, sample_id, 0, np.nan))
                                continue

                            ref1 = ref2 = ref3 = None
                            if mode1_samples is not None:
                                ref_entry = mode1_samples.get(sample_id, {}).get(0)
                                if ref_entry is None or ref_entry[0] != 1 or not np.isfinite(ref_entry[1]) or ref_entry[1] == 0.0:
                                    sample_rows_out.append((bz, sample_id, 0, np.nan))
                                    continue
                                ref1 = ref_entry[1]
                            if mode2_samples is not None:
                                ref_entry = mode2_samples.get(sample_id, {}).get(0)
                                if ref_entry is None or ref_entry[0] != 1 or not np.isfinite(ref_entry[1]) or ref_entry[1] == 0.0:
                                    sample_rows_out.append((bz, sample_id, 0, np.nan))
                                    continue
                                ref2 = ref_entry[1]
                            if mode3_samples is not None:
                                ref_entry = mode3_samples.get(sample_id, {}).get(0)
                                if ref_entry is None or ref_entry[0] != 1 or not np.isfinite(ref_entry[1]) or ref_entry[1] == 0.0:
                                    sample_rows_out.append((bz, sample_id, 0, np.nan))
                                    continue
                                ref3 = ref_entry[1]

                            normalized_complex = _compute_normalized_sample(
                                spec.normalization_mode,
                                target_entry[1],
                                ref1,
                                ref2,
                                ref3,
                            )
                            normalized = float(normalized_complex.real if spec.component == "real" else normalized_complex.imag)
                            success = int(np.isfinite(normalized))
                            sample_rows_out.append((bz, sample_id, success, normalized))
                            if success:
                                normalized_values.append(normalized)

                        center, err = _summarize_percentile(np.asarray(normalized_values, dtype=float))
                        row_out = {name: target_rows[bz].get(name, "") for name in _normalized_fit_header(target_header)}
                        row_out["m0_mean"] = f"{center:.10e}"
                        row_out["m0_err"] = f"{err:.10e}"
                        fit_rows_out.append(row_out)

                    dataset_root = spec.results_dir / title
                    stem = f"{title}_{gm}_{eta}_bT{bT}_{spec.normalization_mode}_{spec.component}_{spec.nstates}state"
                    metadata_lines = [
                        f"title {title}",
                        f"gm {gm}",
                        f"eta {eta}",
                        f"pz {pz}",
                        f"bT {bT}",
                        f"component {spec.component}",
                        f"nstates {spec.nstates}",
                        f"normalization_sample_domain {'complex' if reference_uses_complex else 'scalar'}",
                        f"input_fit_table {target_fit_path}",
                        f"input_sample_table {target_sample_path}",
                    ]
                    if spec.normalization_mode in {"mode1", "mode3"}:
                        metadata_lines.append(f"reference_same_pz_bT0_bz0 {title} {gm} {eta} pz={pz} bT=0 bz=0")
                    if spec.normalization_mode in {"mode2", "mode3"}:
                        metadata_lines.append(f"reference_pz0_same_bT_bz0 {title_pz0} {gm} {eta} pz=0 bT={bT} bz=0")
                    if spec.normalization_mode == "mode3":
                        metadata_lines.append(f"reference_pz0_bT0_bz0 {title_pz0} {gm} {eta} pz=0 bT=0 bz=0")
                    outputs.extend(
                        _write_normalized_outputs(
                            dataset_root,
                            stem,
                            normalization_mode=spec.normalization_mode,
                            fit_header=_normalized_fit_header(target_header),
                            fit_rows=fit_rows_out,
                            sample_rows=sample_rows_out,
                            metadata_lines=metadata_lines,
                        )
                    )
                    fit_tables_by_bT[bT] = dataset_root / "tables" / f"{stem}_fit.txt"

                if spec.make_plots and fit_tables_by_bT:
                    plots_dir = (spec.results_dir / title / "plots")
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    outputs.append(
                        plot_tmdwf_m0_from_fit_tables(
                            plots_dir / f"{title}_{gm}_{eta}_{spec.normalization_mode}_{spec.component}_{spec.nstates}state_m0_vs_bz.pdf",
                            fit_tables_by_bT,
                            component=spec.component,
                            nstates=spec.nstates,
                            title=f"{title} {gm} {eta} {spec.normalization_mode} {spec.component} {spec.nstates}state m0 vs bz",
                        )
                    )
    return outputs
