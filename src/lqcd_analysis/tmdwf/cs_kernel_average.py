from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common.parsing import parse_int_list_or_range
from .plotting import plot_tmdwf_cs_kernel_average_bT


@dataclass(frozen=True)
class TMDWFCSKernelAverageInput:
    title_pattern: str
    input_root: Path
    lattice_spacing_fm: float
    gm: str
    eta: str
    component: str
    nstates: int
    normalization_mode: str
    scheme: str
    extraction_type: str
    kernel_label: str
    bTlist: tuple[int, ...]
    x_range: tuple[float, float] | None
    reference_pz_labels: tuple[str, ...]
    results_dir: Path


@dataclass(frozen=True)
class TMDWFCSKernelBandRecord:
    path: Path
    metadata: dict[str, str]
    x: np.ndarray
    gamma_p16: np.ndarray
    gamma_p50: np.ndarray
    gamma_p84: np.ndarray


@dataclass(frozen=True)
class TMDWFCSKernelSampleRecord:
    path: Path
    metadata: dict[str, str]
    x: np.ndarray
    sample_ids: np.ndarray
    samples: np.ndarray


@dataclass(frozen=True)
class TMDWFCSKernelAverageSelection:
    bT: int
    reference_pz: int
    reference_pz_label: str
    reference_pz_gev: float
    x_min: float
    x_max: float
    n_selected_points: int
    band_path: Path
    sample_path: Path


@dataclass(frozen=True)
class TMDWFCSKernelAverageRow:
    bT: int
    bT_fm: float
    value: float
    stat_err: float
    sys_err: float
    total_err: float
    n_selected_sources: int
    n_selected_points: int
    n_samples: int


@dataclass(frozen=True)
class TMDWFCSKernelAverageSampleRow:
    bT: int
    sample_id: int
    success: int
    mean: float
    std: float
    n_selected_points: int
    n_selected_sources: int


def _read_key_value_metadata(path: Path) -> tuple[dict[str, list[str]], list[str]]:
    metadata: dict[str, list[str]] = {}
    data_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            first = tokens[0].lower()
            if first == "x":
                continue
            try:
                float(tokens[0])
            except ValueError:
                metadata[tokens[0]] = tokens[1:]
                continue
            data_lines.append(stripped)
    return metadata, data_lines


def _require_single(metadata: dict[str, list[str] | str], key: str, path: Path) -> str:
    if key not in metadata:
        raise ValueError(f"missing required metadata key {key} in {path}")
    value = metadata[key]
    if isinstance(value, str):
        if not value:
            raise ValueError(f"missing required metadata key {key} in {path}")
        return value
    if not value:
        raise ValueError(f"missing required metadata key {key} in {path}")
    return value[0]


def _parse_header_float(value: list[str] | str, key: str, path: Path) -> float:
    if isinstance(value, str):
        if not value:
            raise ValueError(f"missing required metadata key {key} in {path}")
        return float(value)
    if not value:
        raise ValueError(f"missing required metadata key {key} in {path}")
    return float(value[0])


def _parse_x_range(entries: dict[str, list[str]], path: Path) -> tuple[float, float] | None:
    if "x_range" not in entries:
        return None
    values = entries["x_range"]
    if len(values) != 2:
        raise ValueError(f"x_range in {path} must contain exactly two values: x_min x_max")
    x_min = float(values[0])
    x_max = float(values[1])
    if x_max < x_min:
        raise ValueError(f"x_range in {path} must satisfy x_max >= x_min")
    return x_min, x_max


def _legacy_quantile_triplet(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return np.nan, np.nan, np.nan
    q16, q50, q84 = np.percentile(values, [16.0, 50.0, 84.0])
    return float(q16), float(q50), float(q84)


def parse_tmdwf_cs_kernel_average_input(
    path: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> TMDWFCSKernelAverageInput:
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
        "lattice_spacing_fm",
        "gm",
        "eta",
        "component",
        "nstates",
        "normalization_mode",
        "kernel_label",
        "scheme",
        "extraction_type",
    }
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")
    if "bTlist" not in entries and "bTrange" not in entries:
        raise ValueError(f"missing required key in {file_path}: bTlist or bTrange")

    input_root = Path(entries["input_root"][0])
    if not input_root.exists():
        raise FileNotFoundError(f"TMDWF CS-kernel average input_root does not exist: {input_root}")
    lattice_spacing_fm = float(entries["lattice_spacing_fm"][0])
    if lattice_spacing_fm <= 0.0:
        raise ValueError("lattice_spacing_fm must be positive")

    component = entries["component"][0].lower()
    if component not in {"real", "imag"}:
        raise ValueError("component must be one of: real, imag")
    normalization_mode = entries["normalization_mode"][0].lower()
    if normalization_mode not in {"raw", "mode1", "mode2", "mode3"}:
        raise ValueError("normalization_mode must be one of: raw, mode1, mode2, mode3")

    scheme = entries["scheme"][0].strip().upper()
    extraction_type = entries["extraction_type"][0].strip().lower()
    kernel_label = entries["kernel_label"][0].strip().upper()
    x_range = _parse_x_range(entries, file_path)

    reference_pz_labels = tuple(entries.get("reference_pz_labels", []))
    output_root = (
        Path(results_dir)
        if results_dir is not None
        else Path(entries.get("results_dir", [file_path.parent / "results_tmdwf_cs_kernel_average"])[0])
    )
    return TMDWFCSKernelAverageInput(
        title_pattern=entries["title_pattern"][0],
        input_root=input_root,
        lattice_spacing_fm=lattice_spacing_fm,
        gm=entries["gm"][0],
        eta=entries["eta"][0],
        component=component,
        nstates=int(entries["nstates"][0]),
        normalization_mode=normalization_mode,
        scheme=scheme,
        extraction_type=extraction_type,
        kernel_label=kernel_label,
        bTlist=parse_int_list_or_range(entries, "bTlist", "bTrange"),
        x_range=x_range,
        reference_pz_labels=reference_pz_labels,
        results_dir=output_root,
    )


def _parse_band_file(path: Path) -> TMDWFCSKernelBandRecord:
    metadata, data_lines = _read_key_value_metadata(path)
    if not data_lines:
        raise ValueError(f"no data rows found in {path}")
    data = np.loadtxt(data_lines)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        raise ValueError(f"expected 4 data columns in {path}, found {data.shape[1]}")
    return TMDWFCSKernelBandRecord(
        path=path,
        metadata={key: " ".join(values) for key, values in metadata.items()},
        x=data[:, 0].astype(float),
        gamma_p16=data[:, 1].astype(float),
        gamma_p50=data[:, 2].astype(float),
        gamma_p84=data[:, 3].astype(float),
    )


def _parse_sample_file(path: Path) -> TMDWFCSKernelSampleRecord:
    metadata, data_lines = _read_key_value_metadata(path)
    if not data_lines:
        raise ValueError(f"no data rows found in {path}")
    data = np.loadtxt(data_lines)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise ValueError(f"expected at least 5 data columns in {path}, found {data.shape[1]}")

    x_values = np.unique(data[:, 0].astype(float))
    sample_ids = np.unique(data[:, 1].astype(int))
    x_index = {float(x): idx for idx, x in enumerate(x_values)}
    sample_index = {int(sample_id): idx for idx, sample_id in enumerate(sample_ids)}
    samples = np.full((sample_ids.size, x_values.size), np.nan, dtype=float)
    for row in data:
        x = float(row[0])
        sample_id = int(row[1])
        success = int(row[2])
        if success == 0:
            continue
        value = float(row[3])
        samples[sample_index[sample_id], x_index[x]] = value
    return TMDWFCSKernelSampleRecord(
        path=path,
        metadata={key: " ".join(values) for key, values in metadata.items()},
        x=x_values,
        sample_ids=sample_ids,
        samples=samples,
    )


def _matches_source(record: TMDWFCSKernelBandRecord, spec: TMDWFCSKernelAverageInput) -> bool:
    metadata = record.metadata
    checks = {
        "title_pattern": spec.title_pattern,
        "gm": spec.gm,
        "eta": spec.eta,
        "component": spec.component,
        "normalization_mode": spec.normalization_mode,
        "scheme": spec.scheme,
        "extraction_type": spec.extraction_type,
        "kernel_label": spec.kernel_label,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            return False
    return True


def discover_tmdwf_cs_kernel_sources(spec: TMDWFCSKernelAverageInput) -> list[tuple[TMDWFCSKernelBandRecord, TMDWFCSKernelSampleRecord]]:
    band_paths = sorted(spec.input_root.glob("**/*_band.txt"))
    sources: list[tuple[TMDWFCSKernelBandRecord, TMDWFCSKernelSampleRecord]] = []
    for band_path in band_paths:
        band = _parse_band_file(band_path)
        if not _matches_source(band, spec):
            continue
        sample_path = band_path.parent.parent / "samples" / band_path.name.replace("_band.txt", "_samples.txt")
        if not sample_path.exists():
            raise FileNotFoundError(f"missing CS-kernel sample table for {band_path}: {sample_path}")
        sample = _parse_sample_file(sample_path)
        sources.append((band, sample))
    if not sources:
        raise ValueError(
            "no TMDWF CS-kernel outputs matched the requested filters: "
            f"title_pattern={spec.title_pattern}, gm={spec.gm}, eta={spec.eta}, component={spec.component}, "
            f"normalization_mode={spec.normalization_mode}, scheme={spec.scheme}, "
            f"extraction_type={spec.extraction_type}, kernel_label={spec.kernel_label}"
        )
    return sources


def _selected_x_mask(
    x: np.ndarray,
    *,
    bT: int,
    pz_gev: float,
    x_range: tuple[float, float] | None,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if x_range is not None:
        x_min, x_max = x_range
        return (values >= x_min) & (values <= x_max)
    return (
        (2.0 * values * pz_gev > 1.0)
        & (2.0 * (1.0 - values) * pz_gev > 1.0)
        & (bT * pz_gev * values > 0.5)
        & (bT * pz_gev * (1.0 - values) > 0.5)
    )


def _sample_summary(samples: np.ndarray) -> tuple[float, float]:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    q16, q84 = np.percentile(values, [16.0, 84.0])
    return float(np.mean(values)), float(0.5 * (q84 - q16))


def _summarize_selected_bootstrap_rows(concatenated: np.ndarray, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if concatenated.shape[0] != sample_ids.size:
        raise ValueError("sample_ids and concatenated bootstrap rows must have the same number of samples")
    sample_means = np.full(sample_ids.size, np.nan, dtype=float)
    sample_stds = np.full(sample_ids.size, np.nan, dtype=float)
    success = np.zeros(sample_ids.size, dtype=int)
    for idx, row in enumerate(concatenated):
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        sample_means[idx] = float(np.mean(finite))
        sample_stds[idx] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        success[idx] = 1
    return success, sample_means, sample_stds


def summarize_tmdwf_cs_kernel_average(
    spec: TMDWFCSKernelAverageInput,
    sources: list[tuple[TMDWFCSKernelBandRecord, TMDWFCSKernelSampleRecord]],
) -> tuple[
    list[TMDWFCSKernelAverageRow],
    list[TMDWFCSKernelAverageSampleRow],
    list[TMDWFCSKernelAverageSelection],
]:
    rows: list[TMDWFCSKernelAverageRow] = []
    sample_rows: list[TMDWFCSKernelAverageSampleRow] = []
    selections: list[TMDWFCSKernelAverageSelection] = []

    output_title = _require_single(sources[0][0].metadata, "output_title", sources[0][0].path)
    dP_gev = _parse_header_float(sources[0][0].metadata.get("dP_GeV", []), "dP_GeV", sources[0][0].path)
    sample_ids_reference = sources[0][1].sample_ids

    for bT in spec.bTlist:
        selected_arrays: list[np.ndarray] = []
        source_count = 0
        selected_points = 0
        for band, sample in sources:
            current_output_title = _require_single(band.metadata, "output_title", band.path)
            if current_output_title != output_title:
                raise ValueError(
                    "inconsistent output_title detected among CS-kernel sources: "
                    f"expected {output_title}, found {current_output_title} in {band.path}"
                )
            current_dP_gev = _parse_header_float(band.metadata.get("dP_GeV", []), "dP_GeV", band.path)
            if not np.isclose(current_dP_gev, dP_gev, atol=1e-12, rtol=0.0):
                raise ValueError(
                    "inconsistent dP_GeV detected among CS-kernel sources: "
                    f"expected {dP_gev:.10e}, found {current_dP_gev:.10e} in {band.path}"
                )
            if int(float(_require_single(band.metadata, "reference_bT", band.path))) != bT:
                continue
            ref_label = _require_single(band.metadata, "reference_pz_label", band.path)
            if spec.reference_pz_labels and ref_label not in spec.reference_pz_labels:
                continue
            pz_label = int(float(_require_single(band.metadata, "reference_pz", band.path)))
            pz_gev = float(pz_label * dP_gev)
            mask = _selected_x_mask(sample.x, bT=bT, pz_gev=pz_gev, x_range=spec.x_range)
            if not np.any(mask):
                continue
            if not np.array_equal(sample.sample_ids, sample_ids_reference):
                raise ValueError(f"inconsistent sample_id grid detected in {sample.path}")
            selected = sample.samples[:, mask]
            selected_arrays.append(selected)
            source_count += 1
            selected_points += int(mask.sum())
            selections.append(
                TMDWFCSKernelAverageSelection(
                    bT=bT,
                    reference_pz=pz_label,
                    reference_pz_label=ref_label,
                    reference_pz_gev=pz_gev,
                    x_min=float(sample.x[mask].min()),
                    x_max=float(sample.x[mask].max()),
                    n_selected_points=int(mask.sum()),
                    band_path=band.path,
                    sample_path=sample.path,
                )
            )

        if not selected_arrays:
            raise ValueError(
                f"no CS-kernel x points satisfied the selection criteria for bT={bT}; "
                + (
                    "check the explicit x_range"
                    if spec.x_range is not None
                    else "check the momentum and bT thresholds"
                )
            )

        concatenated = np.concatenate(selected_arrays, axis=1)
        success, sample_means, sample_stds = _summarize_selected_bootstrap_rows(concatenated, sample_ids_reference)
        finite_means = sample_means[np.isfinite(sample_means)]
        finite_stds = sample_stds[np.isfinite(sample_stds)]
        if finite_means.size == 0:
            raise ValueError(f"no finite bootstrap averages available for bT={bT}")
        value, stat_err = _sample_summary(finite_means)
        sys_err = float(np.mean(finite_stds)) if finite_stds.size else np.nan
        total_err = float(np.sqrt(stat_err**2 + sys_err**2))
        rows.append(
            TMDWFCSKernelAverageRow(
                bT=bT,
                bT_fm=float(bT * spec.lattice_spacing_fm),
                value=value,
                stat_err=stat_err,
                sys_err=sys_err,
                total_err=total_err,
                n_selected_sources=source_count,
                n_selected_points=selected_points,
                n_samples=int(finite_means.size),
            )
        )
        for sample_id, sample_success, mean, std in zip(sample_ids_reference, success, sample_means, sample_stds, strict=True):
            sample_rows.append(
                TMDWFCSKernelAverageSampleRow(
                    bT=bT,
                    sample_id=int(sample_id),
                    success=int(sample_success),
                    mean=float(mean) if np.isfinite(mean) else np.nan,
                    std=float(std) if np.isfinite(std) else np.nan,
                    n_selected_points=selected_points,
                    n_selected_sources=source_count,
                )
            )
    return rows, sample_rows, selections


def _write_average_outputs(
    *,
    spec: TMDWFCSKernelAverageInput,
    output_title: str,
    rows: list[TMDWFCSKernelAverageRow],
    sample_rows: list[TMDWFCSKernelAverageSampleRow],
    selections: list[TMDWFCSKernelAverageSelection],
) -> list[Path]:
    output_root = spec.results_dir / output_title
    tables_dir = output_root / "tables"
    samples_dir = output_root / "samples"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    refpz_token = "all" if not spec.reference_pz_labels else "_".join(spec.reference_pz_labels)
    x_token = "xavg"
    if spec.x_range is not None:
        x_min, x_max = spec.x_range
        x_token = f"xrange{x_min:.3f}_{x_max:.3f}".replace(".", "p")
    stem = (
        f"{output_title}_{spec.gm}_{spec.eta}_{spec.normalization_mode}_{spec.component}"
        f"_{spec.nstates}state_{spec.scheme}_{spec.kernel_label}_{spec.extraction_type}"
        f"_refpz{refpz_token}_{x_token}"
    )
    summary_path = output_root / f"{stem}_summary.txt"
    values_path = tables_dir / f"{stem}_values.txt"
    selection_path = tables_dir / f"{stem}_selection.txt"
    samples_path = samples_dir / f"{stem}_samples.txt"

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"title_pattern {spec.title_pattern}\n")
        handle.write(f"output_title {output_title}\n")
        handle.write(f"input_root {spec.input_root}\n")
        handle.write(f"gm {spec.gm}\n")
        handle.write(f"eta {spec.eta}\n")
        handle.write(f"component {spec.component}\n")
        handle.write(f"nstates {spec.nstates}\n")
        handle.write(f"normalization_mode {spec.normalization_mode}\n")
        handle.write(f"scheme {spec.scheme}\n")
        handle.write(f"extraction_type {spec.extraction_type}\n")
        handle.write(f"kernel_label {spec.kernel_label}\n")
        if spec.x_range is not None:
            handle.write(f"x_range {spec.x_range[0]:.10e} {spec.x_range[1]:.10e}\n")
        handle.write(f"reference_pz_labels {' '.join(spec.reference_pz_labels) if spec.reference_pz_labels else 'all'}\n")
        handle.write(f"bTlist {' '.join(str(value) for value in spec.bTlist)}\n")
        handle.write(f"results_dir {spec.results_dir}\n")
        if spec.x_range is None:
            handle.write(
                "source_selection_formula 2*x*pz>1GeV, 2*(1-x)*pz>1GeV, bT*pz*x>0.5, bT*pz*(1-x)>0.5\n"
            )
        else:
            handle.write(
                f"source_selection_formula explicit_x_range {spec.x_range[0]:.10e} {spec.x_range[1]:.10e}\n"
            )

    with values_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "bT\tbT_fm\tvalue\tstat_err\tsys_err\ttotal_err\tn_selected_sources\tn_selected_points\tn_samples\n"
        )
        for row in rows:
            handle.write(
                f"{row.bT}\t{row.bT_fm:.10e}\t{row.value:.10e}\t{row.stat_err:.10e}\t{row.sys_err:.10e}\t{row.total_err:.10e}\t"
                f"{row.n_selected_sources}\t{row.n_selected_points}\t{row.n_samples}\n"
            )

    with selection_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "bT\treference_pz\treference_pz_label\treference_pz_GeV\tx_min\tx_max\tn_selected_points\tband_path\tsample_path\n"
        )
        for row in selections:
            handle.write(
                f"{row.bT}\t{row.reference_pz}\t{row.reference_pz_label}\t{row.reference_pz_gev:.10e}\t"
                f"{row.x_min:.10e}\t{row.x_max:.10e}\t{row.n_selected_points}\t{row.band_path}\t{row.sample_path}\n"
            )

    with samples_path.open("w", encoding="utf-8") as handle:
        handle.write("bT\tsample_id\tsuccess\tmean\tstd\tn_selected_points\tn_selected_sources\n")
        for row in sample_rows:
            mean_text = f"{row.mean:.10e}" if np.isfinite(row.mean) else "nan"
            std_text = f"{row.std:.10e}" if np.isfinite(row.std) else "nan"
            handle.write(
                f"{row.bT}\t{row.sample_id}\t{row.success}\t{mean_text}\t{std_text}\t"
                f"{row.n_selected_points}\t{row.n_selected_sources}\n"
            )

    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"{stem}_bT_average.pdf"
    plot_output = plot_tmdwf_cs_kernel_average_bT(
        plot_path,
        rows,
        title=f"{output_title} {spec.gm} {spec.eta} {spec.normalization_mode} {spec.component} {spec.kernel_label}",
    )

    return [summary_path, values_path, selection_path, samples_path, plot_output]


def run_tmdwf_cs_kernel_average_workflow(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_cs_kernel_average_input(input_file, results_dir=results_dir)
    sources = discover_tmdwf_cs_kernel_sources(spec)
    rows, sample_rows, selections = summarize_tmdwf_cs_kernel_average(spec, sources)
    output_title = _require_single(sources[0][0].metadata, "output_title", sources[0][0].path)
    return _write_average_outputs(
        spec=spec,
        output_title=output_title,
        rows=rows,
        sample_rows=sample_rows,
        selections=selections,
    )
